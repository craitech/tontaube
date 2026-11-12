"""Client-side helpers for joining a continuation to your own audio (requests with ``prefix_text`` / ``prefix_tokens``).

Serving returns the continuation only. The client keeps its own audio up to the cut, fades the last
SPLICE_FADE_MS of it out, and appends the continuation as served (plain concatenation, no crossfade).
Serving fades the continuation's first 10 ms in (PREFIX_START_FADE, on by default), so both sides of the
join meet at 0 and a cut where the kept audio still sounds does not click. The fade-in alone is not enough:
the kept audio would still stop dead at the cut.

Only numpy is needed. Audio arrays are float or integer PCM with time on axis 0 (soundfile's layout: (T,) or
(T, channels)); the result is float (float32 for float32 / integer input, float64 for float64 input).
"""
import numpy as np

SPLICE_FADE_MS = 10.0     # client fade-out; the same length as serving's default PREFIX_START_FADE_MS
CODE_HZ = 12.5            # code frames per second: one frame = 80 ms = 1920 samples at 24 kHz


def cut_sample(cut_frame: int, sample_rate: int = 24000) -> int:
    """First sample of code frame ``cut_frame`` (1920 * cut_frame at 24 kHz, 3840 * cut_frame at 48 kHz).

    Keep ``audio[:cut_sample(...)]`` and send ``prefix_tokens`` = every codebook row up to ``cut_frame``. The
    index is in the timeline of the codes' decode (start of the audio = frame 0): if the kept audio was
    edge-trimmed (``trim_silence_padding_ms``), subtract the samples trimmed from its start.
    """
    return round(cut_frame * sample_rate / CODE_HZ)


def fade_out_tail(audio, fade_ms: float = SPLICE_FADE_MS, sample_rate: int = 24000) -> np.ndarray:
    """Raised-cosine fade-out over the last ``fade_ms`` of ``audio`` (a new array; the last sample becomes 0).

    The same ramp as serving's request-end fade: gain 0.5 * (1 + cos(pi * k / n)), k = 1..n, on the last
    n = fade_ms * sample_rate / 1000 samples (all of them when the audio is shorter).
    """
    x = np.asarray(audio)
    out = x.astype(np.result_type(x.dtype, np.float32), copy=True)
    n = min(out.shape[0], round(fade_ms * sample_rate / 1000))
    if n <= 0:
        return out
    k = np.arange(1, n + 1, dtype=np.float64)
    ramp = 0.5 * (1.0 + np.cos(np.pi * k / n))
    out[-n:] = out[-n:] * ramp.reshape((n,) + (1,) * (out.ndim - 1)).astype(out.dtype)
    return out


def splice_continuation(kept, continuation, sample_rate: int = 24000, fade_ms: float = SPLICE_FADE_MS) -> np.ndarray:
    """Your kept audio (already cut at the cut sample) with its last ``fade_ms`` faded out, then the continuation
    exactly as served (same sample rate; serving has faded its start in). Plain concatenation, no crossfade."""
    head = fade_out_tail(kept, fade_ms, sample_rate)
    tail = np.asarray(continuation)
    return np.concatenate([head, tail.astype(np.result_type(head.dtype, tail.dtype), copy=False)], axis=0)
