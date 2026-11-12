"""Early-chunk check for streaming (STREAM_EARLY_CHECK, default on).

The stream plays its first chunk (the first ~40 CB0 codes, generated in early mode) before segment 0 exists, and then
forces those codes as the prefix of every segment-0 retry. In early mode the CB0 check is only a silence-token ratio, so
a silent start that the ratio misses is played, locked in, and served even when every retry fails. With
STREAM_EARLY_CHECK=1 the early chunk is checked before it is played (inference.py): its codes with 12794 counted as
silence, and its decoded audio with /predict's silent-audio check and an active-speech floor. A failed early chunk is
re-sampled from scratch; a failed early prefix is never forced into segment 0.
"""
import torch

from app.config import DUALCODEC_SAMPLE_RATE

# The model's dominant lead-silence code; not in SILENCE_TOKEN_IDS, so the early silence ratio did not count it.
EARLY_LEAD_SILENCE_TOKEN = 12794


def active_fraction(audio: torch.Tensor, threshold_dbfs: float = -50.0) -> float:
    """Share of 20 ms frames (10 ms hop) whose RMS is at or above threshold_dbfs (/predict's edge-trim frames); 0.0 if shorter."""
    samples = audio.detach().float().reshape(-1, audio.shape[-1])
    mono_power = samples.square().mean(dim=0)
    frame_samples = max(1, round(DUALCODEC_SAMPLE_RATE * 0.02))
    hop_samples = max(1, round(DUALCODEC_SAMPLE_RATE * 0.01))
    if mono_power.numel() < frame_samples:
        return 0.0
    frame_rms = mono_power.unfold(0, frame_samples, hop_samples).mean(dim=-1).sqrt()
    return float((frame_rms >= 10.0 ** (threshold_dbfs / 20.0)).float().mean().item())
