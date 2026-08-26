"""DualCodec chunked decoding utilities."""
import torch

from app.config import DUALCODEC_SAMPLE_RATE
from app.utils import codes_to_audio_tensor
from app.vibevoice_utils import vv_encode, vv_decode_streaming
from app.logging_utils import get_logger

logger = get_logger("decode")

def split_at_markers(codes: list, marker_indices: list) -> list[list]:
    """Split flat code list into segments at marker positions."""
    boundaries = [0] + sorted(marker_indices) + [len(codes)]
    return [codes[boundaries[i]:boundaries[i + 1]] for i in range(len(boundaries) - 1)]


VV_HZ = 7.5  # VibeVoice tokenizer frame rate


def decode_chunked_with_context(
    dc_inference,
    vv_tokenizer,
    semantic_codes: list[int],
    acoustic_codes: list[list[int]],
    hz: float,
    context_seconds: float,
    return_pre_vv: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Decode DC tokens to audio with windowed VV postprocessing.

    1. Flat DC decode → wav (pre-VV)
    2. Windowed VV encode with overlap padding → collect VV frames
    3. Streaming VV decode on full frame sequence → seamless audio

    If return_pre_vv=True, returns (post_vv_audio, pre_vv_audio).
    """
    total_tokens = len(semantic_codes)
    total_duration = total_tokens / hz
    pad_seconds = context_seconds  # padding on each side of a window
    window_seconds = 30.0  # main window size in seconds
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
            # VV encode the full padded window
            with torch.no_grad():
                frames = vv_encode(vv_tokenizer, window_audio)  # [1, T_vv, D]

            left_pad_vv = int(left_pad_tokens / hz * VV_HZ)
            content_vv = int((win_end_content - offset_tokens) / hz * VV_HZ)
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
        return (pre_vv_audio, pre_vv_audio) if return_pre_vv else pre_vv_audio

    # Streaming VV decode → seamless audio
    vv_frames = torch.cat(all_vv_frames, dim=1)  # [1, T_total_vv, D]
    del all_vv_frames
    logger.debug("Decoding %d total VibeVoice frames", vv_frames.shape[1])

    vv_decode_chunk = int(30.0 * VV_HZ)
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
    logger.debug(
        "Decoded %d samples (%.2fs)",
        post_vv_audio.shape[-1],
        post_vv_audio.shape[-1] / DUALCODEC_SAMPLE_RATE,
    )

    if return_pre_vv:
        return post_vv_audio, pre_vv_audio
    return post_vv_audio
