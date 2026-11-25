"""Redraw: a small detector listens to the audio as it is made; where it hears the pitch jump to another note, the
engine goes back a frame and draws the frames again with fresh noise, before the player gets there.

The rule: frame 0 is never judged or redrawn. A spot is the earliest final step whose spot measure reaches the cutoff
(0.85); its frame f. Attempts go back to b0 = f - 1 (else f; never before the chunk's first frame, never into audio
that is played), draw frames b0 .. f + 2 and are scored over the steps from 48 before the first candidate frame on.
A clean attempt (below the cutoff) is kept; otherwise up to 3 redraws, then the take with the lowest score, the
original included. An attempt starts only if its frames can be ready before the player reaches them (time per frame
measured as the engine runs, plus a 50 ms margin).

Per request: the utterance's audio on the GPU for the detector, the final scores m of the audio as it stands, the spot
in hand and the counters. The engine calls `after_decode` for every decoded frame and `point` between frames; `point`
judges what is final and, for a spot, restores the generator to just before the swap point (key/value cache cut back by
its counters, the state that predicts the frame, autoencoder and VibeVoice decoder state from the per-frame snapshots,
noise index and position) and draws frames again with fresh noise from b0 on, up to f + 2: an attempt stops early once a
newly final step of the scored range reaches the cutoff, and at once when playback reaches the swap point. Every take
of a spot is swapped in from the first attempt's b0; later attempts build on the best take so far. A take other than
the generator's current one is made again from its noise (the engine is deterministic: the same noise gives the same
frames bit for bit).

Frame numbers here are the utterance's (frame 0 = its first audio frame); t is the chunk's own frame index as in the
engine (t = k0 + frame - G, G = the chunk's first frame in the utterance).
"""
import time

import numpy as np
import torch

from .detector import CONTEXT, HOP, LAG

FRAME = 3200
SR = 24000


class Redraw:
    def __init__(self, eng, frames_max, played=None, boundary=None):
        self.eng = eng
        self.cut, self.max_takes, self.margin = eng.cutoff, eng.max_takes, eng.margin_ms / 1000
        W = eng.judge.W
        n = frames_max * FRAME + W * HOP + FRAME
        if eng._rd_audio is None or eng._rd_audio.numel() < n:
            eng._rd_audio = torch.zeros(n, device=eng.dev)
        self.audio = eng._rd_audio
        self.m = np.zeros(-(-frames_max * FRAME // HOP) + 1, np.float32)
        self.fin = -1                  # last final step of the audio as it stands whose score is asked for
        self.known = -1                # ... and whose score is on the host
        self.handled = -1              # steps up to here are not looked at again
        self.pending = []              # detector results not yet on the host: (lo, hi, values on the GPU, events)
        self.take = None               # while a take is drawn: where its scores go
        self.spot = None               # (step, frame) of the spot in hand
        self.played_fn, self.boundary_fn, self.t_first = played, boundary, None
        self.offset = 0                # player sample p = utterance sample p + offset (the cut at the request's start)
        self.stats = {"spots": 0, "attempts": 0, "clean_after": [0] * (self.max_takes - 1), "kept_best_unclean": 0,
                      "kept_original": 0, "given_up_played": 0, "given_up_previous_chunk": 0, "cut_by_budget": 0,
                      "stopped_early": 0, "stopped_by_playback": 0, "swaps_too_late": 0,
                      "extra_frames": 0, "replayed_frames": 0, "replay_mismatch": 0, "detector_runs": 0, "detector_ms": 0.0, "redraw_ms": 0.0,
                      "spot_log": []}
        self._ev = []

    # ------------------------------------------------------------------------------------------- playback position
    def played(self, ahead=0.0):
        """Player samples consumed `ahead` seconds from now (playback at real time)."""
        if self.played_fn is not None:
            return self.played_fn() + ahead * SR
        if self.t_first is None:
            return 0.0
        return (time.perf_counter() - self.t_first + ahead) * SR

    def unplayed(self, frame, seconds):
        """Is the first sample of `frame` still unplayed after `seconds` of work plus the safety margin? With a
        boundary (the samples that can no longer be replaced, e.g. sent on by a server that holds audio back): is it
        at or after the boundary? Nothing is added there for the work or as a margin. (played() is
        in player samples, frames in utterance samples.)"""
        if self.boundary_fn is not None:
            return self.boundary_fn() + self.offset <= FRAME * frame
        return self.played(seconds + self.margin) + self.offset <= FRAME * frame

    # ------------------------------------------------------------------------------------------------ the detector
    def after_decode(self, g):
        """Frame g's audio (in eng.audio) into the detector's buffer; scores of the steps it makes final."""
        e = self.eng
        if self.pending:                                 # the detector's last window has been read before this is written
            torch.cuda.current_stream().wait_event(self.pending[-1][4])
        self.audio[g * FRAME:(g + 1) * FRAME].copy_(e.audio)
        if self.take is not None and self.take.get("replay"):
            return
        lo = (self.take["fin"] if self.take is not None else self.fin) + 1
        hi = (g + 1) * FRAME // HOP - LAG
        if hi >= lo:
            self._judge(lo, hi)

    def _judge(self, lo, hi, total=None):
        """Scores of steps lo..hi on the detector's own stream (it overlaps the next backbone step); the host reads them
        at the next point where it waits for the GPU anyway. An engine with judge_later (the batch engine) scores the
        windows of all its requests at once and adds them to self.pending itself."""
        later = getattr(self.eng, "judge_later", None) if total is None else None
        if later is not None:
            later(self, lo, hi, self.take)
        else:
            J = self.eng.judge
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()                                   # on the main stream: the audio is in the buffer
            J.stream.wait_event(a)
            span = J.W - CONTEXT - LAG if total is None else 1 << 30
            with torch.cuda.stream(J.stream):
                vals = [J.scores(self.audio, s, min(hi, s + span - 1), total).clone() for s in range(lo, hi + 1, span)]
                b.record()
            self.pending.append((lo, hi, vals[0] if len(vals) == 1 else torch.cat(vals), a, b, self.take))
        if self.take is not None:
            self.take["fin"] = hi
        else:
            self.fin = hi
        self.stats["detector_runs"] += 1

    def collect(self, block=True):
        """Move finished detector results to the host (their take's scores or the main track)."""
        while self.pending and (block or self.pending[0][4].query()):
            lo, hi, v, a, b, take = self.pending.pop(0)
            b.synchronize()                              # written on the detector's stream
            v = v if isinstance(v, np.ndarray) else v.cpu().numpy()
            self.stats["detector_ms"] += a.elapsed_time(b)
            if take is None:
                self.m[lo:hi + 1] = v
                self.known = hi
            else:
                take["m"][lo:hi + 1] = v

    def finish_scores(self, total):
        """The utterance has ended at `total` samples: every remaining step is final."""
        T = -(-total // HOP)
        lo = (self.take["fin"] if self.take is not None else self.fin) + 1
        if T - 1 >= lo:
            self._judge(lo, T - 1, total)

    # ---------------------------------------------------------------------------------------------- the decisions
    def point(self, ch, t, stopped):
        """Between frames: judge what is final, decide a spot whose original take is complete. ch: the engine's chunk
        state; t: the chunk's next frame; stopped: the chunk has ended by its stop value. Returns (pieces to hand to the
        player, t, stopped) after any redraw. A generator (yield from) like the engine's run_frames: the frames of a
        redraw are drawn through it."""
        e = self.eng
        pieces = []
        while True:
            ended = stopped or t >= ch["max_f"]
            synced = ended or t >= e.min_frames
            if ended:
                torch.cuda.current_stream().synchronize()
            self.collect(block=synced)
            if ended and ch["last"] and not self.pending:
                total = FRAME * (ch["G"] + t - ch["k0"])
                if self.fin < -(-total // HOP) - 1:
                    self.finish_scores(total)
                    self.collect()
            if self.spot is None:
                self.spot = self._find(ch)
            if self.spot is None:
                return pieces, t, stopped
            f = self.spot[1]
            decoded = ch["G"] + t - ch["k0"]            # frames of the utterance decoded so far
            if not (ended or decoded >= f + 3):
                return pieces, t, stopped                # the original take is not complete yet (frames up to f + 2)
            out, t, stopped = yield from self._decide(ch, t, stopped)
            pieces += out

    def _find(self, ch):
        """The earliest final hot step not yet handled, as (step, frame); spots that cannot be redrawn are given up."""
        lo = self.handled + 1
        if self.known < lo:
            return None
        hot = lo + np.flatnonzero(self.m[lo:self.known + 1] >= self.cut)
        for s in hot.tolist():
            if s <= self.handled:
                continue
            f = s * HOP // FRAME
            if f == 0:                                   # frame 0 is never judged
                self.handled = s
                continue
            self.stats["spots"] += 1
            if f < ch["G"]:                              # in an earlier chunk: b0 may not go before this chunk's first frame
                self.stats["given_up_previous_chunk"] += 1
                self.stats["spot_log"].append({"step": s, "frame": f, "score": float(self.m[s]), "result": "given_up_previous_chunk"})
                end = s                                  # its hot stretch (gaps up to 30 steps) is handled with it
                while end < self.known and (self.m[end + 1:min(end + 31, self.known + 1)] >= self.cut).any():
                    end += 1 + int(np.flatnonzero(self.m[end + 1:min(end + 31, self.known + 1)] >= self.cut)[-1])
                self.handled = end
                continue
            return s, f
        self.handled = self.known
        return None

    def _decide(self, ch, t_now, stopped_now):
        """Attempts for the spot in hand; returns (pieces, t, stopped) of the take kept."""
        e, st = self.eng, self.stats
        w0 = time.perf_counter()
        s, f = self.spot
        self.spot = None
        self.collect()
        k0, G = ch["k0"], ch["G"]
        loc = lambda g: k0 + g - G                       # utterance frame -> chunk frame
        cands = [b for b in (f - 1, f) if b >= max(1, G)]
        # scored range: from 48 steps before the first candidate's frame (new audio from b0 on changes the
        # scores that far back, so a jump drawn into the backtrack frame is judged), or t - 30 if that is earlier
        L = max(-(-FRAME // HOP), min(s - 30, cands[0] * FRAME // HOP - LAG))
        orig = {"kind": "original", "t": t_now, "stopped": stopped_now, "fin": self.known,
                "score": float(self.m[L:self.known + 1].max()) if self.known >= L else 0.0}
        takes, best, last = [orig], orig, None           # last: the take whose state the generator holds (None: the original's)
        log = {"step": s, "frame": f, "score": float(self.m[s]), "L": L, "original": orig["score"], "takes": []}
        snap = {b: e.snapshot(loc(b), b) for b in cands}  # the original's states just before frame b
        noise0 = e.noise.clone()
        audio0 = self.audio[FRAME * cands[0]:FRAME * (G + t_now - k0)].clone()
        swap, stop, ran = None, None, False
        for k in range(self.max_takes - 1):
            b = next((b for b in cands if self.unplayed(b, (f + 3 - b) * e.frame_s)), None)
            if b is None:
                stop = "budget"
                break
            swap = b if swap is None else swap           # every attempt is swapped in from the first one's b0
            # the attempt builds on the best take so far: from the swap point its frames up to b, then fresh noise
            e.noise.copy_(noise0)
            if best is not orig and b > swap:
                e.noise[loc(swap) - k0:loc(b) - k0].copy_(best["noise"][:b - swap])
            e.restore(snap[swap], loc(swap), ch)
            e.fresh_noise(loc(b), loc(f + 2) - loc(b) + 1)
            take = {"kind": "attempt", "b": b, "fin": L - 1, "m": np.zeros_like(self.m),
                    "noise": e.noise[loc(swap) - k0:min(loc(f + 3) - k0, e.MAXF)].clone()}
            e.head_again()

            def check():                                 # after every frame of the attempt: played? already failed?
                if not self.unplayed(swap, 0.0):
                    return "played"
                self.collect()
                if take["fin"] >= L and take["m"][L:take["fin"] + 1].max() >= self.cut:
                    return "early"
                return None
            self.take, ran = take, True
            ta = time.perf_counter()
            take["t"], take["stopped"], take["audio"], why = yield from e.run_frames(ch, loc(swap), loc(f + 3), check)
            n = take["t"] - loc(swap)
            st["attempts"] += 1
            st["extra_frames"] += n
            if n:                                        # frames drawn back to back: the budget's time per frame
                e.frame_s += 0.2 * ((time.perf_counter() - ta) / n - e.frame_s)
            last = take
            if why == "played":                          # the swap point is being played: the spot ends here
                self.take = None
                st["stopped_by_playback"] += 1
                stop = "played"
                log["takes"].append({"b": b, "frames": n, "stop": "played"})
                break
            if why is None and (take["stopped"] or take["t"] >= ch["max_f"]) and ch["last"]:
                self.finish_scores(FRAME * (G + take["t"] - k0))
            self.collect()
            self.take = None
            st["stopped_early"] += why == "early"
            take["score"] = float(take["m"][L:take["fin"] + 1].max()) if take["fin"] >= L else 0.0
            takes.append(take)
            log["takes"].append({"b": b, "frames": n, "score": take["score"], **({"stop": why} if why else {})})
            if take["score"] < best["score"]:
                best = take
            if take["score"] < self.cut:
                st["clean_after"][k] += 1
                break
        if not ran:                                      # no attempt could start in time: nothing changes
            st["given_up_played"] += 1
            log["result"] = "given_up_played"
            st["spot_log"].append(log)
            self.handled = max(self.handled, self.known, s)
            st["redraw_ms"] += 1e3 * (time.perf_counter() - w0)
            return [], t_now, stopped_now
        if len(takes) > 1 and takes[-1]["score"] >= self.cut:
            st["kept_best_unclean"] += 1
            st["cut_by_budget"] += stop == "budget"
        # keep the lowest-scoring take; an attempt only while the swap point is more than the margin ahead
        keep = best
        if keep is not orig and not self.unplayed(swap, 0.0 if keep is last else (keep["t"] - loc(swap)) * e.frame_s):
            keep = orig
            st["swaps_too_late"] += 1
        pieces, fix = [], None
        for _ in range(2):
            if keep is not last:                         # make the kept take again from its noise
                fix = yield from self._replay(ch, keep, orig, swap, cands, snap, noise0, audio0, loc)
                last = keep
            if keep is orig or self.unplayed(swap, 0.0):
                break
            keep = orig                                  # the replay took too long: the original plays on
            st["swaps_too_late"] += 1
        if keep is not orig:
            pieces.append((FRAME * swap, np.concatenate(keep["audio"]) if keep["audio"] else np.zeros(0, np.float32)))
            if keep["m"] is not None:
                self.m[L:keep["fin"] + 1] = keep["m"][L:keep["fin"] + 1]
                self.fin = self.known = keep["fin"]
            else:                                        # replayed audio not scored: judged again from before the swap point
                self.fin = self.known = min(self.known, FRAME * swap // HOP - LAG - 1)
        else:
            st["kept_original"] += 1
            if keep is last and fix is not None:         # the original's replay differed: hand the player the replay
                pieces.append((FRAME * fix[0], np.concatenate(fix[1]) if fix[1] else np.zeros(0, np.float32)))
                self.fin = self.known = min(self.known, FRAME * fix[0] // HOP - LAG - 1)
        self.handled = max(self.handled, keep["fin"], s)
        log["result"] = "clean" if keep["score"] < self.cut else "best_unclean"
        log["kept"] = next((i for i, tk in enumerate(takes) if tk is keep), -1)    # 0 = the original
        log["stop"] = stop
        st["spot_log"].append(log)
        st["redraw_ms"] += 1e3 * (time.perf_counter() - w0)
        return pieces, keep["t"], keep["stopped"]

    def _replay(self, ch, keep, orig, swap, cands, snap, noise0, audio0, loc):
        """The generator back to the end of take `keep`, drawn again from its noise (the original: from its own
        noise rows). The same noise gives the same frames bit for bit; should a GPU ever differ, the player gets what
        the replay gave (returned as a piece), so it always holds what the generator holds."""
        e, st, k0, G = self.eng, self.stats, ch["k0"], ch["G"]
        e.noise.copy_(noise0)
        start = swap if keep is not orig else cands[0]
        if keep is not orig:
            e.noise[loc(swap) - k0:loc(swap) - k0 + len(keep["noise"])].copy_(keep["noise"])
        e.restore(snap[start], loc(start), ch)
        e.head_again()
        self.take = {"replay": True}
        t_end, stopped, audio, _ = yield from e.run_frames(ch, loc(start), keep["t"])
        self.take = None
        st["replayed_frames"] += t_end - loc(start)
        if keep is orig:
            same = t_end == keep["t"] and torch.equal(self.audio[FRAME * start:FRAME * (G + t_end - k0)], audio0)
        else:
            same = t_end == keep["t"] and all(np.array_equal(x, y) for x, y in zip(audio, keep["audio"]))
        keep["t"], keep["stopped"] = t_end, stopped
        if same:
            return None
        st["replay_mismatch"] += 1
        if keep is not orig:
            keep.update(audio=audio, m=None)
            return None
        return start, audio
