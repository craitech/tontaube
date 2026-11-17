"""Streaming start buffer: a startup calibration and the per-request choice of ``streaming_initial_seconds``.

Why. A stream first sends an early chunk of ``streaming_initial_seconds`` of audio, then one chunk per text segment
(``TTSEngine._generate_streaming_inner``). Segment 0 is generated in full (up to 350 characters, about 20 seconds of
speech) before its chunk is sent, so playback pauses unless the buffered audio lasts until the next chunk arrives. A
fast GPU generates many times faster than real time and needs little buffer; a weak GPU or a CPU needs more, and
how much depends on the length of the request's first segments.

Calibration. At startup the server streams ``CALIBRATION_TEXT`` (a short and a longer segment, so the per-segment
overhead and the generation rate can be told apart) through the normal streaming path and records when the early
chunk and each segment are ready (``generate_streaming(timings=...)``). ``StreamCalibration.from_timings`` fits
``segment time = segment_overhead_s + new audio seconds / rate``.

Choice. For a request without ``streaming_initial_seconds``, ``StreamCalibration.plan`` simulates the stream with
the calibrated numbers (generation times multiplied by ``margin``) for its own segments, whose speech duration is
estimated from their character counts, and returns the smallest buffer on the 0.4 s grid, at least ``floor``, for
which no chunk arrives after the audio before it has finished playing. If even the largest buffer pauses (a machine
slower than real time, or long first segments on a slow one), it returns the largest and reports the pause.

The calibration measures one stream on an idle server; concurrent streams share the machine and run slower.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from app.config import (
    DUALCODEC_HZ,
    MAX_STREAMING_INITIAL_SECONDS,
    STREAMING_DC_PAD_TOKENS,
    STREAMING_SECONDS_QUANTUM,
)

# A short and a longer segment (one client segment each), plain English without numbers or abbreviations.
CALIBRATION_TEXT = (
    "Good morning, and welcome back to the show.",
    (
        "Today we take a slow walk through an old harbor town, where the fishing boats leave before sunrise, "
        "the bakery opens its doors long before anyone is awake, and the lighthouse keeper still climbs the "
        "stairs every single evening."
    ),
)
CALIBRATION_INITIAL_SECONDS = 1.2    # long enough to hold speech (a 0.4 s chunk is mostly the forced lead silence)
STREAMING_CHUNK_SECONDS = 2.0       # EngineConfig.streaming_chunk_seconds: segment chunks are whole multiples of it
_PAD_SECONDS = STREAMING_DC_PAD_TOKENS / DUALCODEC_HZ


def quantize_up(seconds: float) -> float:
    """``seconds`` rounded up to the 0.4 s streaming grid."""
    return round(math.ceil(round(seconds / STREAMING_SECONDS_QUANTUM, 6)) * STREAMING_SECONDS_QUANTUM, 1)


def quantize_down(seconds: float) -> float:
    """``seconds`` rounded down to the 0.4 s streaming grid."""
    return round(math.floor(round(seconds / STREAMING_SECONDS_QUANTUM, 6)) * STREAMING_SECONDS_QUANTUM, 1)


@dataclass(frozen=True)
class StreamCalibration:
    rate: float                 # audio seconds generated per second of wall time (segments)
    segment_overhead_s: float   # fixed time per segment (prefill, acoustic passes, decode, verbalizer)
    first_overhead_s: float     # time before the early chunk beyond its audio / rate (voice prompt, prefill)
    chars_per_second: float     # speech rate of the calibration text, for estimating segment durations
    took_s: float               # wall time of the calibration

    @classmethod
    def from_timings(cls, timings, segment_chars, took_s: float) -> "StreamCalibration":
        """Fit from ``generate_streaming(timings=...)`` of the calibration stream (an early chunk and two
        segments). Without an early chunk (STREAM_EARLY_CHECK rejected every attempt) segment 0 starts at time 0,
        so its time also holds the start-up work (the safe side). Raises ValueError without two segments."""
        early = [(t, audio) for event, t, audio in timings if event == "early"]
        segments = [(t, audio) for event, t, audio in timings if event == "segment"]
        if len(early) > 1 or len(segments) < 2:
            raise ValueError(f"calibration stream has {len(early)} early chunks and {len(segments)} segments")
        (t_early, a_early), (t0, cum0), (t1, cum1) = (early or [(0.0, 0.0)])[0], segments[0], segments[1]
        g0, new0 = t0 - t_early, cum0 - a_early
        g1, new1 = t1 - t0, cum1 - cum0
        if min(g0, g1, new0, new1) <= 0:
            raise ValueError("calibration segments produced no audio or took no time")
        rate, overhead = 0.0, -1.0
        if new1 - new0 > 0.5 and g1 > g0:
            rate = (new1 - new0) / (g1 - g0)
            overhead = g0 - new0 / rate
        if overhead < 0:   # noisy fit: no fixed overhead, the average rate (the slower, safe reading)
            rate, overhead = (new0 + new1) / (g0 + g1), 0.0
        total_audio = cum1
        return cls(
            rate=rate,
            segment_overhead_s=overhead,
            first_overhead_s=max(0.0, t_early - a_early / rate),
            chars_per_second=sum(segment_chars) / total_audio,
            took_s=took_s,
        )

    def _pauses(self, initial: float, durations: list[float], margin: float) -> bool:
        """Whether a stream of segments with these speech durations pauses with this early chunk."""
        early = min(initial, max(0.0, quantize_down(durations[0] - _PAD_SECONDS)))
        t = margin * (self.first_overhead_s + early / self.rate)
        play_end = t + early            # playback starts with the first chunk
        emitted, cumulative = early, 0.0
        for k, duration in enumerate(durations):
            new = duration - early if k == 0 else duration
            t += margin * (self.segment_overhead_s + max(0.0, new) / self.rate)
            cumulative += duration
            if k == len(durations) - 1:
                target = cumulative     # the flush sends the rest
            else:
                target = math.floor((cumulative - _PAD_SECONDS) / STREAMING_CHUNK_SECONDS) * STREAMING_CHUNK_SECONDS
            if target <= emitted:
                continue
            if emitted > 0 and t > play_end:
                return True
            play_end = max(play_end, t) + (target - emitted)
            emitted = target
        return False

    def plan(
        self, segment_chars: list[int], *, floor: float, ceiling: float = MAX_STREAMING_INITIAL_SECONDS,
        margin: float = 1.25,
    ) -> tuple[float, bool]:
        """(streaming_initial_seconds, pauses expected) for a stream of segments with these character counts."""
        floor = min(quantize_up(max(floor, STREAMING_SECONDS_QUANTUM)), ceiling)
        durations = [max(chars, 1) / self.chars_per_second for chars in segment_chars] or [0.0]
        steps = round((ceiling - floor) / STREAMING_SECONDS_QUANTUM)
        for step in range(steps + 1):
            initial = round(floor + step * STREAMING_SECONDS_QUANTUM, 1)
            if not self._pauses(initial, durations, margin):
                return initial, False
        return ceiling, True

    def summary(self, floor: float, margin: float) -> str:
        longest, pauses = self.plan([350], floor=STREAMING_SECONDS_QUANTUM, margin=margin)
        need = f"{longest:.1f} s" + (" (pauses expected even then)" if pauses else "")
        return (
            f"{self.rate:.2f}x real time, {self.segment_overhead_s:.2f} s per segment, first chunk overhead "
            f"{self.first_overhead_s:.2f} s; a 350-character first segment needs a {need} start buffer "
            f"(automatic minimum {floor:.1f} s); calibration took {self.took_s:.1f} s"
        )
