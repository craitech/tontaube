"""Append-only audio from an engine's stream, for hosts whose clients cannot take replacements (a server that sends
MP3 or Opus, a player that only appends). One rule for every engine:

- each piece (start, audio) is written to a buffer; audio from `start` on is replaced (the redraw);
- audio is released when it is final: frame 0 (never redrawn) at once, later audio HOLD frames (4) behind the newest
  sample, which no redraw reaches;
- the first release waits until the engine leads far enough that each later frame is final `jitter` seconds before
  a client that plays in real time from the first release needs it, with room for one redraw near the start (the
  frames it draws again, about 4 frames of engine time; a fast engine: no wait);
- safety net: if the client's buffer (released audio minus real-time play since the first release) would fall below
  `jitter` (a stalled engine, a long redraw), the next frames are released even if not final;
- the engine's boundary is the released audio: a redraw never replaces what was released.

    out = AppendOnly(start_final=lambda: HOP * eng.sample_rate // SR - eng.stats.get("cut", 0), sample_rate=eng.sample_rate)
    for piece in eng.stream(text, boundary=out.boundary):
        send(out.put(*piece))
    send(out.flush())
"""
import math
import time

import numpy as np

from .stream import HOP, SR

HOLD = 4                                     # frames
REDRAW = 4                                   # frames a redraw draws again (b0 .. f + 2), in engine time


class AppendOnly:
    def __init__(self, hold=HOLD, jitter=0.05, start_final=None, frame_s=None, clock=time.monotonic, redraw=REDRAW,
                 sample_rate=SR):
        """sample_rate: the engine's output rate (a frame is HOP * sample_rate / SR samples; every count below is at
        that rate); hold: frames kept back behind the newest sample (None: nothing before flush(), e.g. a whole file);
        start_final: samples final from the start (frame 0 after the start cut; None: one frame), a number or a function
        read at each piece; frame_s: the engine's seconds per frame if known, else measured from the pieces; redraw:
        frames of engine time the first release leaves room for (one redraw near the start)."""
        self.sr, self.hop = sample_rate, HOP * sample_rate // SR
        self.hold = None if hold is None else hold * self.hop
        self.jitter, self.frame_s, self.clock, self.redraw = jitter, frame_s, clock, redraw
        start_final = self.hop if start_final is None else start_final
        self.start_final = start_final if callable(start_final) else (lambda: start_final)
        self.buf, self.n = np.zeros(self.sr * 10, np.float32), 0
        self.released, self.t_first, self.t0, self.n0 = 0, None, None, 0
        self.early = 0                       # samples released before they were final (safety net)
        self.reached = 0                     # pieces that started inside released audio (kept as released)

    def boundary(self) -> int:
        """Samples released: what a redraw may no longer replace."""
        return self.released

    def put(self, start: int, audio: np.ndarray) -> np.ndarray:
        """Writes a piece; returns the audio it releases (often empty)."""
        if start < self.released:
            self.reached += 1
            audio, start = audio[self.released - start:], self.released
        end = start + len(audio)
        if end > len(self.buf):
            self.buf = np.concatenate([self.buf[:self.n], np.zeros(max(len(self.buf), end), np.float32)])
        self.buf[start:end] = audio
        self.n = end
        now = self.clock()
        if self.t0 is None:
            self.t0, self.n0 = now, self.n
        if self.hold is None:
            return self.buf[:0].copy()
        final = max(self.released, self.n - self.hold, min(self.n, int(self.start_final())))
        if self.t_first is None:
            return self._release(final) if self._lead_enough(final, now) else self.buf[:0].copy()
        upto = final
        while upto < self.n and (upto - (now - self.t_first) * self.sr) / self.sr < self.jitter:
            upto = min(self.n, upto + self.hop)
        self.early += upto - final
        return self._release(upto)

    def flush(self) -> np.ndarray:
        """The rest, at the end of the stream."""
        return self._release(self.n)

    def _lead_enough(self, final: int, now: float) -> bool:
        """Released up to `final` now, is the next sample final `jitter` before the client needs it, even if a redraw
        comes first? (For an engine faster than real time the later samples then are too.)"""
        if final <= 0:
            return False
        if self.n > self.n0 and now > self.t0:
            p = (now - self.t0) * self.hop / (self.n - self.n0)   # measured seconds per frame
        elif self.frame_s is not None:
            p = self.frame_s
        else:
            return False                                       # the next piece tells the engine's speed
        frames = math.ceil((final + 1 + self.hold - self.n) / self.hop)   # until sample `final` is final
        return (frames + self.redraw) * p + self.jitter <= final / self.sr

    def _release(self, upto: int) -> np.ndarray:
        out = self.buf[self.released:upto].copy()
        if len(out) and self.t_first is None:
            self.t_first = self.clock()
        self.released = max(self.released, upto)
        return out
