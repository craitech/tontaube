"""Audio encoding/decoding utilities."""
import base64
import gc
import hashlib
import io
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
)
from app.logging_utils import get_logger

logger = get_logger("audio")

_SPECIAL_TOKEN_RE = re.compile(r'(<\|[^|]+\|>|<<[^>]+>>|\|\|)')


class VoicePromptError(ValueError):
    """The supplied voice reference cannot be used."""


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


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


_SAMPLES_PER_TOKEN = 1920  # DualCodec: 24000 Hz / 12.5 Hz


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

        sem, aco = dc_inference.encode(chunk, n_quantizers=n_codebooks)

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

    max_samples_direct = 30 * DUALCODEC_SAMPLE_RATE  # 30s fits in one encode call
    if audio.shape[-1] > max_samples_direct:
        semantic_codes, acoustic_codes = _encode_chunked(dc_inference, audio, n_codebooks)
    else:
        semantic_codes, acoustic_codes = dc_inference.encode(audio, n_quantizers=n_codebooks)

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


_char_token_cache = {}


def tokenize_text(text: str, tokenizer) -> list[int]:
    result = []
    in_im_block = False
    for part in _SPECIAL_TOKEN_RE.split(text):
        if part.startswith('<|') and part.endswith('|>'):
            tid = tokenizer.convert_tokens_to_ids(part)
            if tid is not None and tid != tokenizer.unk_token_id:
                result.append(tid)
            if part == '<|im_start|>':
                in_im_block = True
            elif part == '<|im_end|>':
                in_im_block = False
        elif part.startswith('<<') and part.endswith('>>') or part == '||' or in_im_block:
            result.extend(tokenizer.encode(part, add_special_tokens=False))
        else:
            for char in part:
                if char not in _char_token_cache:
                    _char_token_cache[char] = tokenizer(char)['input_ids'][0]
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
