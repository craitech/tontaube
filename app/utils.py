"""Audio encoding/decoding utilities."""
import base64
import gc
import hashlib
import io
import json
import math
import os
import re
import tempfile
from collections import OrderedDict

import torch
import soundfile as sf_backend
import torchaudio
from pydub import AudioSegment
import dualcodec

from app.config import (
    DUALCODEC_SAMPLE_RATE,
    END_OF_SPEECH,
    TEXT_MARKER,
    AUDIO_MARKER,
    BOUNDARY_TOKENS,
    BOUNDARY_TOKENS_FILENAME,
    VOICE_REF_LEVEL_WINDOW,
)
from app.logging_utils import get_logger
from app.model.text_vocab import char_token_id, clean_text, is_model_token

logger = get_logger("audio")

_SPECIAL_TOKEN_RE = re.compile(r'(<\|[^|]+\|>|<<[^>]+>>|\|\|)')


class VoicePromptError(ValueError):
    """The supplied voice reference cannot be used."""


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif _LIBC is not None:
        _LIBC.malloc_trim(0)  # hand freed heap (encode/decode activations) back to the OS


def _load_libc():
    import ctypes
    import sys

    if not sys.platform.startswith("linux"):
        return None
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim  # glibc only
        return libc
    except (OSError, AttributeError):
        return None


_LIBC = _load_libc()


def release_mapped_weights(module: torch.nn.Module) -> int:
    """Drop the resident pages of a module's file-mapped CPU weights (Linux).

    Weights loaded from safetensors on the CPU are private file mappings. While
    no page of a mapping has been written (no anonymous pages), dropping its
    pages is lossless: the next access re-reads them from the page cache.
    Returns the number of bytes released.
    """
    ptrs = [p.data_ptr() for p in module.parameters() if p.device.type == "cpu" and p.numel()]
    if not ptrs or _LIBC is None or not os.path.exists("/proc/self/smaps"):
        return 0
    import ctypes

    mappings, cur = [], None
    with open("/proc/self/smaps") as fh:
        for line in fh:
            head = line.split()
            if "-" in head[0] and len(head) >= 5 and not head[0].endswith(":"):
                lo, hi = (int(x, 16) for x in head[0].split("-"))
                cur = [lo, hi, len(head) >= 6 and head[5].startswith("/"), None]
                mappings.append(cur)
            elif head[0] == "Anonymous:" and cur is not None:
                cur[3] = int(head[1])
    madvise = _LIBC.madvise
    madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    released = 0
    for lo, hi, is_file, anon_kb in mappings:
        if is_file and anon_kb == 0 and any(lo <= ptr < hi for ptr in ptrs):
            if madvise(lo, hi - lo, 4) == 0:  # MADV_DONTNEED
                released += hi - lo
    return released


def get_dc_inference(
    dualcodec_path: str,
    semantic_model_path: str,
    model_type: str = "12hz_v1",
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = dualcodec.get_model(model_type, pretrained_model_path=dualcodec_path)
    return dualcodec.Inference(
        dualcodec_model=model,
        dualcodec_path=dualcodec_path,
        w2v_path=semantic_model_path,
        device=device,
    )


def load_and_prep_audio(audio_path: str, target_sr: int = DUALCODEC_SAMPLE_RATE) -> torch.Tensor:
    data, sr = sf_backend.read(audio_path, dtype="float32")
    wav = torch.from_numpy(data)
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    else:
        wav = wav.T  # soundfile returns (frames, channels), we need (channels, frames)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")


def gated_level_db(audio: torch.Tensor, sample_rate: int = DUALCODEC_SAMPLE_RATE) -> float:
    """Speech level in dB: BS.1770-style gating of 400 ms blocks (no frequency weighting), so pauses do not count."""
    x = audio.reshape(-1).float()
    block, hop = int(0.4 * sample_rate), int(0.1 * sample_rate)
    if x.numel() < block:
        return 10 * math.log10(float(x.pow(2).mean()) + 1e-12)
    energy = x.unfold(0, block, hop).pow(2).mean(dim=1)
    energy = energy[energy > 1e-7]
    if energy.numel() == 0:
        return -99.0
    gated = energy[energy > energy.mean() * 0.1]
    return 10 * math.log10(float(gated.mean()) + 1e-12)


def fit_reference_level(
    audio: torch.Tensor,
    window: tuple[float, float] | None = VOICE_REF_LEVEL_WINDOW,
    peak_db: float = -1.0,
    sample_rate: int = DUALCODEC_SAMPLE_RATE,
) -> torch.Tensor:
    """Scale a reference whose level lies outside ``window`` (low, high dB) to the nearer edge, peak <= ``peak_db``."""
    if window is None:
        return audio
    level = gated_level_db(audio, sample_rate)
    low, high = window
    if low <= level <= high:
        return audio
    gain_db = (low if level < low else high) - level
    peak = float(audio.abs().max())
    if peak > 0:
        gain_db = min(gain_db, peak_db - 20 * math.log10(peak))
    return audio * 10 ** (gain_db / 20)


_SAMPLES_PER_TOKEN = 1920  # DualCodec: 24000 Hz / 12.5 Hz


def dc_encode(dc_inference, audio: torch.Tensor, n_quantizers: int):
    """DualCodec encode; on the CPU without its hard-coded fp16 autocast.

    `dualcodec.Inference.encode` always runs W2V-BERT under fp16 autocast.
    CPUs without native fp16 arithmetic execute that on a slow emulation path
    (minutes per reference), so the CPU path repeats the same steps in fp32.
    """
    if torch.device(dc_inference.device).type != "cpu":
        return dc_inference.encode(audio, n_quantizers=n_quantizers)
    audio = audio.cpu()
    audio_16k = torchaudio.functional.resample(audio, 24000, 16000)
    inputs = dc_inference.semantic_cfg.feature_extractor(
        audio_16k[0].cpu(), sampling_rate=16000, return_tensors="pt"
    )
    with torch.inference_mode():
        feat = dc_inference._extract_semantic_code(
            inputs["input_features"], inputs["attention_mask"]
        ).transpose(1, 2)
        factor = dc_inference.model.semantic_downsample_factor
        feat = torch.nn.functional.avg_pool1d(feat, factor, factor)
        return dc_inference.model.encode(audio, num_quantizers=n_quantizers, semantic_repr=feat)


def _encode_chunked(
    dc_inference, audio: torch.Tensor, n_codebooks: int,
    stride_seconds: float = 20, pad_seconds: float = 8,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Encode audio in strided chunks to avoid OOM on long prompts."""
    stride_samples = int(stride_seconds * DUALCODEC_SAMPLE_RATE)
    pad_samples = int(pad_seconds * DUALCODEC_SAMPLE_RATE)
    n_samples = audio.shape[-1]

    sem_parts, aco_parts = [], []
    main_start = 0
    while main_start < n_samples:
        main_end = min(main_start + stride_samples, n_samples)
        chunk_start = max(0, main_start - pad_samples)
        chunk_end = min(n_samples, main_end + pad_samples)
        chunk = audio[:, :, chunk_start:chunk_end]

        sem, aco = dc_encode(dc_inference, chunk, n_codebooks)

        left_pad_tok = (main_start - chunk_start) // _SAMPLES_PER_TOKEN
        main_tok = (main_end - main_start) // _SAMPLES_PER_TOKEN
        sem_parts.append(sem[:, :, left_pad_tok:left_pad_tok + main_tok])
        if aco is not None:
            aco_parts.append(aco[:, :, left_pad_tok:left_pad_tok + main_tok])
        main_start = main_end

    semantic_codes = torch.cat(sem_parts, dim=-1)
    acoustic_codes = torch.cat(aco_parts, dim=-1) if aco_parts else None
    return semantic_codes, acoustic_codes


def encode_voice_prompt(
    dc_inference,
    prompt_path: str,
    prompt_seconds: float | None,
    prompt_max_tokens: int | None,
    n_codebooks: int,
) -> tuple[list[int] | None, list[list[int]] | None]:
    if not prompt_path or not os.path.exists(prompt_path):
        return None, None

    audio = load_and_prep_audio(prompt_path)
    if prompt_seconds:
        audio = audio[..., :int(prompt_seconds * DUALCODEC_SAMPLE_RATE)]
    if audio.shape[-1] < _SAMPLES_PER_TOKEN:
        raise VoicePromptError("the voice reference contains no audio")
    audio = fit_reference_level(audio)

    max_samples_direct = 30 * DUALCODEC_SAMPLE_RATE  # 30s fits in one encode call
    if audio.shape[-1] > max_samples_direct:
        semantic_codes, acoustic_codes = _encode_chunked(dc_inference, audio, n_codebooks)
    else:
        semantic_codes, acoustic_codes = dc_encode(dc_inference, audio, n_codebooks)

    semantic_list = semantic_codes[0, 0, :].cpu().tolist()
    acoustic_lists = (
        [
            acoustic_codes[0, i, :].cpu().tolist()
            for i in range(acoustic_codes.shape[1])
        ]
        if acoustic_codes is not None
        else None
    )

    if prompt_max_tokens is not None:
        semantic_list = semantic_list[:prompt_max_tokens]
        if acoustic_lists:
            acoustic_lists = [ac[:prompt_max_tokens] for ac in acoustic_lists]

    logger.debug("Encoded %d voice-prompt tokens from %s", len(semantic_list), prompt_path)
    semantic_model = dc_inference.semantic_cfg.get("semantic_model")
    if semantic_model is not None and torch.device(dc_inference.device).type == "cpu":
        # W2V-BERT is only needed for new voices: leave its weights in the page cache.
        release_mapped_weights(semantic_model)
    return semantic_list, acoustic_lists


def codes_to_audio_tensor(
    dc_inference,
    semantic_codes: list[int],
    acoustic_codes: list[list[int]] | None,
) -> torch.Tensor:
    semantic_tensor = torch.tensor(semantic_codes, dtype=torch.long).unsqueeze(0).unsqueeze(0)
    
    if acoustic_codes and all(len(ac) > 0 for ac in acoustic_codes):
        acoustic_tensor = torch.tensor(acoustic_codes, dtype=torch.long).unsqueeze(0)
    else:
        acoustic_tensor = None
    
    device = next(dc_inference.model.parameters()).device
    semantic_tensor = semantic_tensor.to(device)
    if acoustic_tensor is not None:
        acoustic_tensor = acoustic_tensor.to(device)
    
    return dc_inference.decode(semantic_tensor, acoustic_tensor)


def load_tokenizer(model_path: str):
    """Load the tokenizer packaged with a release-ready codebook model."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        fix_mistral_regex=False,
    )
    required_tokens = (END_OF_SPEECH, TEXT_MARKER, AUDIO_MARKER, "<|dc_0_0|>")
    missing = [
        token
        for token in required_tokens
        if tokenizer.convert_tokens_to_ids(token) in (None, tokenizer.unk_token_id)
    ]
    if tokenizer.pad_token_id is None or missing:
        raise ValueError(
            f"model tokenizer is incomplete: missing pad token or required tokens {missing}"
        )
    return tokenizer


def _safetensors_shape(model_path: str, key: str) -> list[int] | None:
    """Shape of ``key`` from the bundle's safetensors header(s), without loading weights; None if absent."""
    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path) as f:
            filename = json.load(f).get("weight_map", {}).get(key)
        if filename is None:
            return None
    else:
        filename = "model.safetensors"
    path = os.path.join(model_path, filename)
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as f:
        header = json.loads(f.read(int.from_bytes(f.read(8), "little")))
    entry = header.get(key)
    return None if entry is None else list(entry["shape"])


def check_boundary_tokens(
    model_path: str,
    tokenizer,
    codebook: int,
    slider_tokens=(),
) -> tuple[int, int] | None:
    """Boundary tokens: a bundle's tokenizer carries <|utt_end|> /
    <|utt_cont|> iff the bundle declares them in boundary_tokens.json: CB0 only, ids right after the slider
    tokens and last in the tokenizer, rows in config.json's vocab and the token table. Returns their ids for a
    declaring bundle, None for a legacy one (fails closed on anything in between). tokenize_text silently drops
    an unknown <|...|>, so the token window flag is refused for a bundle without the rows (by the caller)."""
    vocab = tokenizer.get_vocab()
    known = [token for token in BOUNDARY_TOKENS if token in vocab]
    decl_path = os.path.join(model_path, BOUNDARY_TOKENS_FILENAME)
    if not os.path.isfile(decl_path):
        if known:
            raise ValueError(
                f"{model_path}: tokenizer carries {known} but the bundle has no {BOUNDARY_TOKENS_FILENAME}"
            )
        return None
    with open(decl_path) as f:
        decl = json.load(f)
    if codebook != 0 or decl.get("codebooks") != ["cb0"]:
        raise ValueError(
            f"{decl_path}: boundary tokens are CB0-only (codebooks {decl.get('codebooks')}, loaded as cb{codebook})"
        )
    if decl.get("tokens") != list(BOUNDARY_TOKENS):
        raise ValueError(f"{decl_path}: tokens {decl.get('tokens')} != {list(BOUNDARY_TOKENS)}")
    ids = [vocab.get(token) for token in BOUNDARY_TOKENS]
    if len(known) != len(BOUNDARY_TOKENS) or ids != decl.get("ids") or ids[1] != ids[0] + 1:
        raise ValueError(f"{model_path}: tokenizer ids of {list(BOUNDARY_TOKENS)} = {ids}, declared {decl.get('ids')}")
    if len(tokenizer) != ids[1] + 1:
        raise ValueError(f"{model_path}: tokenizer has {len(tokenizer)} tokens; the boundary tokens must be the last two")
    slider_ids = [vocab[token] for token in slider_tokens]
    if slider_ids and max(slider_ids) + 1 != ids[0]:
        raise ValueError(f"{model_path}: boundary ids {ids} do not follow the last slider id {max(slider_ids)}")
    with open(os.path.join(model_path, "config.json")) as f:
        vocab_size = json.load(f).get("vocab_size")
    if vocab_size != len(tokenizer):
        raise ValueError(f"{model_path}: config.json vocab_size {vocab_size} != tokenizer {len(tokenizer)}")
    table = _safetensors_shape(model_path, "model.embed_tokens.weight")
    if table is None or table[0] != len(tokenizer):
        raise ValueError(f"{model_path}: token table shape {table} has no rows for the boundary ids {ids}")
    return ids[0], ids[1]


def check_cb0_window_ids(
    text: str,
    ids: list[int],
    boundary_ids: tuple[int, int] | None,
    token_window: bool,
) -> None:
    """Fail closed when a CB0 window's boundary tokens did not reach the model as their ids (tokenize_text drops an
    unknown <|...|>): a token window carries exactly one, a legacy window on a declaring bundle none. A legacy window
    on a legacy bundle is not checked (unchanged)."""
    if boundary_ids is None and not token_window:
        return
    expected = 1 if token_window else 0
    n_text = sum(text.count(token) for token in BOUNDARY_TOKENS)
    n_ids = 0 if boundary_ids is None else sum(1 for i in ids if i in boundary_ids)
    if n_text != expected or n_ids != expected:
        raise ValueError(
            f"CB0 window must carry {expected} boundary token(s): {n_text} in the text, {n_ids} in the ids "
            f"(boundary ids {boundary_ids})"
        )


_char_token_cache = {}


def tokenize_text(text: str, tokenizer) -> list[int]:
    """Training input scheme: one token per character, BPE only inside the system prompt.

    Only rows the models were trained on are produced (app/model/text_vocab.py):
    other <|...|> tokens, <<tags>> and "||" are dropped and characters are cleaned.
    """
    result = []
    in_im_block = False
    for part in _SPECIAL_TOKEN_RE.split(text):
        if part.startswith('<|') and part.endswith('|>'):
            if not is_model_token(part):
                continue
            tid = tokenizer.convert_tokens_to_ids(part)
            if tid is not None and tid != tokenizer.unk_token_id:
                result.append(tid)
            if part == '<|im_start|>':
                in_im_block = True
            elif part == '<|im_end|>':
                in_im_block = False
        elif part.startswith('<<') and part.endswith('>>') or part == '||':
            continue
        elif in_im_block:
            result.extend(tokenizer.encode(part, add_special_tokens=False))
        else:
            for char in clean_text(part):
                if char not in _char_token_cache:
                    _char_token_cache[char] = char_token_id(tokenizer, char)
                result.append(_char_token_cache[char])
    return result


def decode_voice_audio_b64(voice_audio_b64: str, prompt_seconds: float) -> str:
    """Decode base64-encoded audio to a temporary WAV file."""
    try:
        audio_bytes = base64.b64decode(voice_audio_b64, validate=True)
        audio_segment = AudioSegment.from_file(
            io.BytesIO(audio_bytes),
            duration=prompt_seconds,
        )
        audio_segment = audio_segment[:int(prompt_seconds * 1000)]

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            audio_segment.export(tmp.name, format='wav')
            return tmp.name
    except Exception as e:
        raise VoicePromptError(
            "voice_audio_b64 is not valid base64-encoded audio"
        ) from e


def truncate_voice_prompt(
    semantic: list[int] | None,
    acoustic: list[list[int]] | None,
    prompt_max_tokens: int,
) -> tuple[list[int] | None, list[list[int]] | None]:
    """Return a request-capped copy without modifying the cached encoding."""
    semantic_view = semantic[:prompt_max_tokens] if semantic is not None else None
    acoustic_view = (
        [codes[:prompt_max_tokens] for codes in acoustic]
        if acoustic is not None
        else None
    )
    return semantic_view, acoustic_view


def resolve_voice_prompt(
    voice_audio_b64: str | None,
    voice_path: str | None,
    dc_inference,
    prompt_seconds: float,
    prompt_max_tokens: int,
    n_codebooks: int,
    cache: OrderedDict,
    cache_max: int,
    default_prompt: tuple[list | None, list | None],
) -> tuple[list | None, list | None]:
    """Resolve and cache the full encoded prompt, then apply the request cap."""
    if voice_audio_b64:
        cache_key = hashlib.sha256(voice_audio_b64.encode()).hexdigest()
    elif voice_path:
        cache_key = "path:" + voice_path
    else:
        return truncate_voice_prompt(*default_prompt, prompt_max_tokens)
    cache_key += f"|{n_codebooks}"  # /encode_voice can ask for more rows than generation uses

    cached = cache.get(cache_key)
    if cached is not None:
        cache.move_to_end(cache_key)
        logger.debug("Voice-prompt cache hit (%s…)", cache_key[:16])
        return truncate_voice_prompt(*cached, prompt_max_tokens)

    voice_temp_path = None
    try:
        if voice_audio_b64:
            voice_temp_path = decode_voice_audio_b64(voice_audio_b64, prompt_seconds)
        voice_to_use = voice_temp_path or voice_path
        if not voice_to_use or not os.path.exists(voice_to_use):
            raise VoicePromptError("the requested voice reference could not be found")

        semantic, acoustic = encode_voice_prompt(
            dc_inference, voice_to_use, prompt_seconds, None, n_codebooks,
        )
        cache[cache_key] = (semantic, acoustic)
        if len(cache) > cache_max:
            cache.popitem(last=False)
            logger.debug("Cached voice prompt (%s…, %d entries)", cache_key[:16], len(cache))
        return truncate_voice_prompt(semantic, acoustic, prompt_max_tokens)
    finally:
        if voice_temp_path and os.path.exists(voice_temp_path):
            os.remove(voice_temp_path)
