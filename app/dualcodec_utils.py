"""DualCodec chunked decoding utilities."""
import math

import torch

from app.config import DUALCODEC_SAMPLE_RATE
from app.utils import codes_to_audio_tensor
from app.vibevoice_utils import vv_encode, vv_decode_streaming
from app.logging_utils import get_logger
from app.vv_adapter import VVAdapterRequest

logger = get_logger("decode")

def split_at_markers(codes: list, marker_indices: list) -> list[list]:
    """Split flat code list into segments at marker positions."""
    boundaries = [0] + sorted(marker_indices) + [len(codes)]
    return [codes[boundaries[i]:boundaries[i + 1]] for i in range(len(boundaries) - 1)]


VV_HZ = 7.5  # VibeVoice tokenizer frame rate
VV_HOP_SAMPLES = round(DUALCODEC_SAMPLE_RATE / VV_HZ)  # 3200 samples per VibeVoice frame


def code_samples(n_tokens: int, hz: float) -> int:
    """Samples DualCodec renders for n_tokens codes (1920 per code at 12.5 Hz, 24 kHz)."""
    return round(n_tokens * DUALCODEC_SAMPLE_RATE / hz)


def end_fade(audio: torch.Tensor, fade_ms: float, sample_rate: int = DUALCODEC_SAMPLE_RATE) -> torch.Tensor:
    """Raised-cosine fade-out over the last fade_ms of audio (a new tensor; the last sample becomes 0)."""
    n = min(audio.shape[-1], round(fade_ms * sample_rate / 1000))
    if n <= 0:
        return audio
    k = torch.arange(1, n + 1, dtype=torch.float64)
    ramp = (0.5 * (1.0 + torch.cos(math.pi * k / n))).to(device=audio.device, dtype=audio.dtype)
    out = audio.clone()
    out[..., -n:] = out[..., -n:] * ramp
    return out


def start_fade(
    audio: torch.Tensor, fade_ms: float, sample_rate: int = DUALCODEC_SAMPLE_RATE, offset: int = 0,
) -> torch.Tensor:
    """PREFIX_START_FADE: raised-cosine fade-in over the first fade_ms of a response.

    The ramp is end_fade's, reversed in time: gain 0.5 * (1 - cos(pi * j / n)) at response sample j < n
    (n = fade_ms at sample_rate), so the response's first sample becomes 0 and sample n on is unchanged.
    offset = samples of the same response already emitted before `audio` (streaming): the ramp continues
    across chunks, so a first chunk shorter than the fade is faded in full and the next chunk carries on.
    Returns a new tensor, or `audio` itself when none of its samples lies inside the fade.
    """
    n = round(fade_ms * sample_rate / 1000)
    m = min(audio.shape[-1], n - offset)
    if n <= 0 or m <= 0:
        return audio
    j = torch.arange(offset, offset + m, dtype=torch.float64)
    ramp = (0.5 * (1.0 - torch.cos(math.pi * j / n))).to(device=audio.device, dtype=audio.dtype)
    out = audio.clone()
    out[..., :m] = out[..., :m] * ramp
    return out


def finish_request_end(
    audio: torch.Tensor, n_samples: int, fade_ms: float, sample_rate: int = DUALCODEC_SAMPLE_RATE,
) -> torch.Tensor:
    """DECODE_END_FIX: end a request's audio at the codes' length, faded."""
    return end_fade(audio[..., :max(0, n_samples)], fade_ms, sample_rate)


def headroom_gain(headroom_db: float) -> float:
    """Linear gain for OUTPUT_HEADROOM_DB: 10^(-dB/20) (3.6 dB -> 0.6607)."""
    return 10.0 ** (-headroom_db / 20.0)


def apply_output_headroom(audio: torch.Tensor, headroom_db: float, inplace: bool = False) -> torch.Tensor:
    """OUTPUT_HEADROOM_DB: fixed output gain of -headroom_db dB.

    headroom_db == 0 returns `audio` itself, untouched (the old output, bit-exact). inplace=True scales the
    tensor in place; use it only on a tensor the caller owns (the fresh decode output of /predict).
    """
    if not headroom_db:
        return audio
    gain = headroom_gain(headroom_db)
    return audio.mul_(gain) if inplace else audio * gain


def decode_chunked_with_context(
    dc_inference,
    vv_tokenizer,
    semantic_codes: list[int],
    acoustic_codes: list[list[int]],
    hz: float,
    context_seconds: float,
    return_pre_vv: bool = False,
    end_fix: bool = False,
    end_fade_ms: float = 15.0,
    vv_adapter=None,
    window_seconds: float = 30.0,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Decode DC tokens to audio with windowed VV postprocessing.

    1. Flat DC decode → wav (pre-VV)
    2. Windowed VV encode with overlap padding → collect VV frames
    3. Streaming VV decode on full frame sequence → seamless audio

    If return_pre_vv=True, returns (post_vv_audio, pre_vv_audio).

    end_fix (ServerConfig.decode_end_fix): the last window keeps its partial
    VibeVoice frame (ceil instead of floor, so the audio no longer stops 0-107 ms before the codes end),
    the output ends exactly at the codes' length, and its last end_fade_ms are faded out (raised cosine).
    The pre-VV audio returned with return_pre_vv=True is never changed. end_fix=False: output unchanged.

    vv_adapter (VV_ADAPTER_CKPT; app/vv_adapter.py): a residual adapter with a silence gate that
    corrects the collected VibeVoice frames from the codes' DualCodec latent before the streaming decode. The output
    keeps the length it has without the adapter. vv_adapter=None (default) or VibeVoice off: output unchanged.
    """
    # Windows must hold whole 0.4 s units (5 DualCodec tokens = 3 VibeVoice
    # frames) or the VibeVoice frame bookkeeping drifts.
    if abs(window_seconds / 0.4 - round(window_seconds / 0.4)) > 1e-6:
        raise ValueError("window_seconds must be a multiple of 0.4")
    total_tokens = len(semantic_codes)
    total_duration = total_tokens / hz
    pad_seconds = context_seconds  # padding on each side of a window
    window_tokens = int(window_seconds * hz)
    pad_tokens = int(pad_seconds * hz)

    logger.debug(
        "Windowed decode: %d tokens (%.1fs), window=%.1fs, padding=%.1fs",
        total_tokens,
        total_duration,
        window_seconds,
        pad_seconds,
    )

    # Windowed DC decode → VV encode → collect VV frames
    # Each iteration: slice tokens [start-pad : end+pad], DC decode → wav,
    # VV encode → extract content VV frames for [start : end].
    all_vv_frames = []  # list of [1, T_chunk, D] tensors (cpu)
    # Raw DualCodec audio is also the final output when VV is disabled. Collect
    # it independently of the short-audio quality-comparison flag.
    pre_vv_parts = [] if return_pre_vv or vv_tokenizer is None else None
    offset_tokens = 0
    adapter = None
    if vv_adapter is not None and vv_tokenizer is not None:
        adapter = VVAdapterRequest.create(vv_adapter, total_tokens, hz, len(acoustic_codes))

    while offset_tokens < total_tokens:
        win_start = max(0, offset_tokens - pad_tokens)
        win_end_content = min(offset_tokens + window_tokens, total_tokens)
        win_end = min(win_end_content + pad_tokens, total_tokens)

        # DC decode this window (tokens → wav on GPU)
        win_sem = semantic_codes[win_start:win_end]
        win_ac = [ac[win_start:win_end] for ac in acoustic_codes]
        with torch.no_grad():
            window_audio = codes_to_audio_tensor(dc_inference, win_sem, win_ac)

        left_pad_tokens = offset_tokens - win_start
        right_pad_tokens = win_end - win_end_content
        left_pad_samples = int(left_pad_tokens * DUALCODEC_SAMPLE_RATE / hz)
        right_pad_samples = int(right_pad_tokens * DUALCODEC_SAMPLE_RATE / hz)

        # Collect pre-VV audio (content only, no padding)
        if pre_vv_parts is not None:
            if right_pad_samples > 0:
                pre_vv_parts.append(window_audio[..., left_pad_samples:-right_pad_samples].cpu())
            else:
                pre_vv_parts.append(window_audio[..., left_pad_samples:].cpu())

        if vv_tokenizer is not None:
            if adapter is not None:
                # collect the adapter's DC latent and gate for this window's content; the last window's audio is
                # zero-padded so the encode covers the adapter's frame grid (3 * ceil(K / 5) frames)
                window_audio, adapter_vv = adapter.add_window(
                    dc_inference, win_sem, win_ac, window_audio, left_pad_tokens, left_pad_samples,
                    win_end_content - offset_tokens, last=win_end_content == total_tokens,
                )
            # VV encode the full padded window
            with torch.no_grad():
                frames = vv_encode(vv_tokenizer, window_audio)  # [1, T_vv, D]

            left_pad_vv = int(left_pad_tokens / hz * VV_HZ)
            content_vv = int((win_end_content - offset_tokens) / hz * VV_HZ)
            if end_fix and win_end_content == total_tokens:
                # the encoder pads the last, partial frame: keep it (ceil) instead of flooring it away
                content_vv = -(-code_samples(win_end_content - offset_tokens, hz) // VV_HOP_SAMPLES)
            if adapter is not None:
                adapter.n_frames_off += content_vv  # this window's frames with the adapter off (the output length)
                content_vv = adapter_vv
            content_frames = frames[:, left_pad_vv:left_pad_vv + content_vv, :]
            all_vv_frames.append(content_frames.cpu())

            logger.debug(
                "Decode window [%d:%d]: %d tokens -> %d VibeVoice frames "
                "(%d content frames)",
                win_start,
                win_end,
                len(win_sem),
                frames.shape[1],
                content_frames.shape[1],
            )
            del frames

        del window_audio
        torch.cuda.empty_cache()
        offset_tokens = win_end_content

    pre_vv_audio = torch.cat(pre_vv_parts, dim=-1) if pre_vv_parts else None

    if vv_tokenizer is None:
        out = pre_vv_audio
        if end_fix and out is not None:
            out = finish_request_end(out, code_samples(total_tokens, hz), end_fade_ms)
        return (out, pre_vv_audio) if return_pre_vv else out

    # Streaming VV decode → seamless audio
    vv_frames = torch.cat(all_vv_frames, dim=1)  # [1, T_total_vv, D]
    del all_vv_frames
    if adapter is not None:
        vv_frames = adapter.apply(vv_frames)  # z = z_prod + gate * dz, [1, 3 * ceil(K / 5), D] (cpu, float32)
    logger.debug("Decoding %d total VibeVoice frames", vv_frames.shape[1])

    vv_decode_chunk = int(round(window_seconds * VV_HZ))
    cache = None
    audio_parts = []

    with torch.no_grad():
        for i in range(0, vv_frames.shape[1], vv_decode_chunk):
            chunk_frames = vv_frames[:, i:i + vv_decode_chunk, :]
            audio_chunk, cache = vv_decode_streaming(vv_tokenizer, chunk_frames, cache=cache)
            audio_parts.append(audio_chunk)
            logger.debug(
                "VibeVoice frames [%d:%d] -> %d samples",
                i,
                i + chunk_frames.shape[1],
                audio_chunk.shape[-1],
            )

    post_vv_audio = torch.cat(audio_parts, dim=-1)
    if end_fix:
        post_vv_audio = finish_request_end(post_vv_audio, code_samples(total_tokens, hz), end_fade_ms)
    elif adapter is not None:
        post_vv_audio = post_vv_audio[..., :adapter.n_frames_off * VV_HOP_SAMPLES]  # the length with the adapter off
    logger.debug(
        "Decoded %d samples (%.2fs)",
        post_vv_audio.shape[-1],
        post_vv_audio.shape[-1] / DUALCODEC_SAMPLE_RATE,
    )

    if return_pre_vv:
        return post_vv_audio, pre_vv_audio
    return post_vv_audio


def decode_with_prefix_context(
    dc_inference,
    vv_tokenizer,
    prefix_semantic: list[int],
    prefix_acoustic: list[list[int]],
    semantic_codes: list[int],
    acoustic_codes: list[list[int]],
    hz: float,
    **decode_kwargs,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """PREFIX_JOINT_DECODE: decode a prefix continuation with the prefix as the decoder's acoustic left context.

    The prefix + continuation codes go through decode_chunked_with_context in ONE pass (decode_kwargs as there),
    then the prefix's code_samples(len(prefix_semantic), hz) samples are cut off, so only the continuation's
    audio is returned (both tensors when return_pre_vv=True). The continuation's first frames are no longer a
    cold decoder start; nothing is level-matched or crossfaded. The end fix, if on, applies to the
    continuation's end, as without a prefix.
    """
    if len(prefix_acoustic) != len(acoustic_codes) or any(len(p) != len(prefix_semantic) for p in prefix_acoustic):
        raise ValueError(
            f"prefix codes do not line up: {len(prefix_semantic)} semantic, "
            f"{[len(p) for p in prefix_acoustic]} acoustic for {len(acoustic_codes)} codebooks"
        )
    out = decode_chunked_with_context(
        dc_inference, vv_tokenizer,
        list(prefix_semantic) + list(semantic_codes),
        [list(p) + list(a) for p, a in zip(prefix_acoustic, acoustic_codes)],
        hz=hz,
        **decode_kwargs,
    )
    cut = code_samples(len(prefix_semantic), hz)
    if isinstance(out, tuple):
        return tuple(x[..., cut:] if x is not None else None for x in out)
    return out[..., cut:]
