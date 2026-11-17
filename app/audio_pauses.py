"""Shorten long quiet runs without muting their retained beginning or end."""

from __future__ import annotations

import math
import numpy as np

# Detection thresholds at the decoder's own output level (OUTPUT_HEADROOM_DB = 0). /predict scales the decode by
# -OUTPUT_HEADROOM_DB dB before the cap, so both thresholds move down by the same amount: the cap finds the same
# quiet runs and joins at any headroom, and the headroom stays a pure level change.
QUIET_DBFS = -55.0  # 20 ms RMS frames below this are quiet
JOIN_PEAK = 0.01  # sample peak (-40 dBFS) above which a proposed join is unsafe


def pause_thresholds(headroom_db: float = 0.0) -> tuple[float, float]:
    """Return (quiet RMS, join peak) as linear amplitudes, both lowered by headroom_db dB.

    headroom_db == 0 gives exactly the fixed thresholds 10 ** (-55 / 20) and 0.01.
    """
    if not math.isfinite(headroom_db) or headroom_db < 0:
        raise ValueError("headroom must be a finite dB value >= 0")
    quiet_rms = 10 ** ((QUIET_DBFS - headroom_db) / 20)
    join_peak = JOIN_PEAK * 10.0 ** (-headroom_db / 20.0)
    return quiet_rms, join_peak


def find_pause_cuts(
    source: np.ndarray,
    sample_rate: int,
    *,
    cap_seconds: float = 2.5,
    headroom_db: float = 0.0,
) -> list[tuple[int, int, int]]:
    """Plan clip_long_pauses' cuts on finite mono float32 audio, as (left, right, width) per cut.

    A cut keeps source[:left], crossfades source[left:left + width] into
    source[right:right + width], continues at right + width and removes
    right - left samples.
    """
    quiet_rms, join_peak = pause_thresholds(headroom_db)
    if not len(source):
        return []
    frame = round(sample_rate * 0.02)
    starts = np.arange(0, len(source), frame)
    energy = np.add.reduceat(source.astype(np.float64) ** 2, starts)
    rms = np.sqrt(energy / np.minimum(frame, len(source) - starts))
    quiet = rms < quiet_rms
    changes = np.diff(np.r_[False, quiet, False].astype(np.int8))
    cap = round(sample_rate * cap_seconds)
    cuts = []
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
        if max(np.max(np.abs(lhs)), np.max(np.abs(rhs))) > join_peak:
            continue
        cuts.append((left, right, width))
    return cuts


def clip_long_pauses(
    audio: np.ndarray,
    sample_rate: int,
    *,
    cap_seconds: float = 2.5,
    headroom_db: float = 0.0,
    cuts_out: list | None = None,
) -> tuple[np.ndarray, int, int]:
    """Return mono audio, removed sample count, and number of middle cuts.

    cuts_out: a list that receives the (left, right, width) cuts applied (find_pause_cuts' tuples,
    in input samples), for /predict's timestamp remapping (app/timestamps.py); None = not reported.

    Detection uses 20ms RMS frames below -55 dBFS (at the decoder's level;
    -55 - headroom_db dBFS for audio scaled by OUTPUT_HEADROOM_DB). Only runs
    strictly longer than the cap are changed. The retained pause includes a
    20ms complementary cosine crossfade, so there is no additional duration
    loss from overlap.
    """
    if not math.isfinite(cap_seconds) or cap_seconds < 0.04:
        raise ValueError("pause cap must be finite and at least 0.04 seconds")
    if not math.isfinite(headroom_db) or headroom_db < 0:
        raise ValueError("headroom must be a finite dB value >= 0")
    source = np.asarray(audio, dtype=np.float32)
    if source.ndim != 1 or sample_rate < 1000 or not np.isfinite(source).all():
        raise ValueError("expected finite mono audio and a valid sample rate")
    if not len(source):
        return source.copy(), 0, 0

    cuts = find_pause_cuts(source, sample_rate, cap_seconds=cap_seconds, headroom_db=headroom_db)
    if cuts_out is not None:
        cuts_out.extend(cuts)
    if not cuts:
        return source, 0, 0
    pieces = []
    cursor = removed = 0
    for left, right, width in cuts:
        lhs = source[left:left + width]
        rhs = source[right:right + width]
        ramp = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, width))).astype(np.float32)
        blend = (1 - ramp) * lhs + ramp * rhs
        pieces.extend([source[cursor:left], blend])
        cursor = right + width
        removed += right - left
    pieces.append(source[cursor:])
    return np.concatenate(pieces), removed, len(cuts)
