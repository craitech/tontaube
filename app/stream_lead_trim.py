"""Leading-silence trim for the first streamed chunk (STREAM_LEAD_TRIM, default on).

/predict's edge trim (main._trim_boundary_silence) keeps `padding_ms` before the first 20 ms frame at or
above -50 dBFS. The stream had no trim, so its first chunk played the forced first-silence frame and the
model's lead silence. With STREAM_LEAD_TRIM=1, generate_streaming cuts the first non-empty chunk with the
same frames, threshold and padding.

- The trim runs before the output headroom gain, so the threshold is at the decoder's own level. /predict
  trims the scaled audio with a threshold lowered by the same dB: both find the same first sound.
- The cut is rounded down to the Opus frame grid (480 samples = 20 ms at 24 kHz). The WebSocket encoder
  zero-pads each chunk's partial last frame, so an off-grid chunk 0 would insert up to 20 ms of zeros
  before chunk 1. Rounding down keeps at least `padding_ms` before the first sound.
- A first chunk without sound, or whose lead is already within the padding, is passed through. Later
  chunks are never touched, and generation is unchanged.
"""
import torch

from app.config import DUALCODEC_SAMPLE_RATE

# opus_stream.FRAME_DURATION (20 ms) at the 24 kHz stream rate
OPUS_GRID_SAMPLES = 480


def stream_lead_cut(
    audio: torch.Tensor,
    *,
    padding_ms: int,
    silence_threshold_dbfs: float = -50.0,
    grid_samples: int = OPUS_GRID_SAMPLES,
) -> int:
    """Samples to drop from the start of the first streamed chunk; 0 leaves it untouched.

    The frame RMS, threshold and padding are main._trim_boundary_silence's leading edge at headroom 0.
    """
    if padding_ms < 0 or audio.shape[-1] == 0:
        return 0

    samples = audio.detach().float().reshape(-1, audio.shape[-1])
    mono_power = samples.square().mean(dim=0)
    frame_samples = max(1, round(DUALCODEC_SAMPLE_RATE * 0.02))
    hop_samples = max(1, round(DUALCODEC_SAMPLE_RATE * 0.01))
    if mono_power.numel() < frame_samples:
        return 0

    frame_rms = mono_power.unfold(0, frame_samples, hop_samples).mean(dim=-1).sqrt()
    threshold = 10.0 ** (silence_threshold_dbfs / 20.0)
    active_frames = (frame_rms >= threshold).nonzero(as_tuple=False).flatten()
    if active_frames.numel() == 0:
        return 0

    padding_samples = round(DUALCODEC_SAMPLE_RATE * padding_ms / 1000)
    start = max(0, int(active_frames[0].item()) * hop_samples - padding_samples)
    return start // grid_samples * grid_samples


def trim_stream_lead(audio: torch.Tensor, *, padding_ms: int) -> torch.Tensor:
    """The first streamed chunk without its lead silence beyond `padding_ms` (the chunk itself if nothing is cut)."""
    cut = stream_lead_cut(audio, padding_ms=padding_ms)
    return audio[..., cut:] if cut else audio
