"""Shorten long quiet runs without muting their retained beginning or end."""

from __future__ import annotations

import math
import numpy as np


def clip_long_pauses(
    audio: np.ndarray,
    sample_rate: int,
    *,
    cap_seconds: float = 2.5,
) -> tuple[np.ndarray, int, int]:
    """Return mono audio, removed sample count, and number of middle cuts.

    Detection uses 20ms RMS frames below -55 dBFS. Only runs strictly longer
    than the cap are changed. The retained pause includes a 20ms complementary
    cosine crossfade, so there is no additional duration loss from overlap.
    """
    if not math.isfinite(cap_seconds) or cap_seconds < 0.04:
        raise ValueError("pause cap must be finite and at least 0.04 seconds")
    source = np.asarray(audio, dtype=np.float32)
    if source.ndim != 1 or sample_rate < 1000 or not np.isfinite(source).all():
        raise ValueError("expected finite mono audio and a valid sample rate")
    if not len(source):
        return source.copy(), 0, 0

    frame = round(sample_rate * 0.02)
    starts = np.arange(0, len(source), frame)
    energy = np.add.reduceat(source.astype(np.float64) ** 2, starts)
    rms = np.sqrt(energy / np.minimum(frame, len(source) - starts))
    quiet = rms < 10 ** (-55.0 / 20)
    changes = np.diff(np.r_[False, quiet, False].astype(np.int8))
    cap = round(sample_rate * cap_seconds)
    pieces = []
    cursor = removed = cuts = 0
    for first, last in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
        start = int(first * frame)
        end = min(int(last * frame), len(source))
        excess = end - start - cap
        if excess <= 0:
            continue
        width = min(round(sample_rate * 0.02), excess)
        if width < 2:
            continue
        left = start + (cap - width) // 2
        right = left + excess
        lhs = source[left:left + width]
        rhs = source[right:right + width]
        # RMS can hide a narrow transient. Skip an unsafe proposed join.
        if max(np.max(np.abs(lhs)), np.max(np.abs(rhs))) > 0.01:
            continue
        ramp = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, width))).astype(np.float32)
        blend = (1 - ramp) * lhs + ramp * rhs
        pieces.extend([source[cursor:left], blend])
        cursor = right + width
        removed += excess
        cuts += 1
    if not cuts:
        return source, 0, 0
    pieces.append(source[cursor:])
    return np.concatenate(pieces), removed, cuts
