"""TontaubeV1.1 Flash on the CPU (voice: Marcus): the engine of the browser and the phone (C++, flash/engine), as a
native library (downloaded prebuilt or built on first use, see build.py), with the GPU engine's API.

    from flash.cpu.engine import Engine
    eng = Engine()                                  # downloads the weights (web/*.bin) and the library once
    for start, audio in eng.stream("Some text."):   # float32 audio (48 kHz) for the player's samples from `start` on
        ...
    wave = eng.speak("Some text.")

The engine's threads are pinned to physical cores of their own (another Engine in the same process takes other
cores; the model files are mapped once and shared). A request runs on a thread of its own inside stream(): pieces
come as they are made, whether or not the caller has taken the previous ones. One request at a time per engine.
"""
import ctypes
import json
import queue
import threading
import time
import warnings
from pathlib import Path

import numpy as np

from ..common.build import engine_units, library
from ..common.frontend import FrontEnd
from ..common.stream import Speaker, release

FILES = ("lm.bin", "misc.bin", "vv.bin", "detector.bin", "bwe.bin")   # detector.bin: for the redraw (without it: none); bwe.bin: 48 kHz
BWE_DELAY = 2494        # output samples the 48 kHz upsampler holds back at most (its look-ahead)
_P = ctypes.POINTER
AUDIO = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int64, _P(ctypes.c_float), ctypes.c_int64)
REPLACE = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_int64, _P(ctypes.c_float), ctypes.c_int64, ctypes.c_int64)
PLAYED = ctypes.CFUNCTYPE(ctypes.c_int64, ctypes.c_void_p)
CANCELLED = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)


class Sink(ctypes.Structure):
    _fields_ = [("user", ctypes.c_void_p), ("audio", AUDIO), ("replace", REPLACE), ("played", PLAYED), ("cancelled", CANCELLED)]


def engine_library() -> Path:
    """The engine's library: from the cache, prebuilt or built on first use."""
    return library(*engine_units())


def weight_dir(weights=None, bwe=True) -> Path:
    """The folder with the CPU weights (the release's web/*.bin; bwe.bin only with bwe, for 48 kHz): weights = a local
    folder laid out as the release, or None: downloaded from Hugging Face once."""
    cfg, get = release(weights)
    need = FILES if bwe else FILES[:4]
    names = [n for n in cfg.get("web", [f"web/{f}" for f in FILES]) if Path(n).name in need]
    paths = {}
    for n in names:
        try:
            paths[Path(n).name] = get(n)
        except Exception:                               # no detector: the engine runs without the redraw
            if Path(n).name != "detector.bin":
                raise
    missing = [str(paths.get(f, f)) for f in need if f != "detector.bin" and (f not in paths or not paths[f].exists())]
    if missing:
        raise FileNotFoundError("weight files not found: " + ", ".join(missing) + " (weights= is the release folder: config.json, gpu/, web/"
                                + ("; bwe.bin is for 48 kHz, sample_rate=24000 runs without it" if bwe else "") + ")")
    return paths["lm.bin"].parent


class Engine(Speaker):
    def __init__(self, variant="int4", threads=None, cpus=None, solver="midpoint", passes=8, temp=0.7, redraw=True, cutoff=0.85,
                 max_takes=4, margin_ms=50.0, lead_cut=True, weights=None, sample_rate=48000, options="", verbose=False):
        """variant: int4 (the one CPU variant: backbone int4, decoder int8). sample_rate: 48000 (the model's 24 kHz
        raised to 48 kHz by an upsampler, web/bwe.bin) or 24000; self.sample_rate is the output rate, and every position
        (pieces, played, boundary, stats["cut"]) counts samples at it. threads: None = automatic (free physical
        cores of one NUMA node, at most 12, within a container's CPU quota); a number above what the process can run
        at once is cut to that (self.info: threads, asked, usable). cpus: None = automatic pinning, "none" =
        no pinning, or a list of CPU ids (one thread each). weights: a local folder laid out as the release; None =
        download the release from Hugging Face once. The rest as in the GPU engine: redraw (flagged pitch jumps are
        drawn again; per request too), cutoff, max_takes, margin_ms, lead_cut (start at the first sound). options: the
        engine's speed options (as the browser's tts_config, e.g. "pipe=3,3 share=1"; the same for every engine of a
        process)."""
        if variant != "int4":
            raise ValueError(f"the CPU engine has one variant, int4 (not {variant!r})")
        if solver not in ("midpoint", "euler"):
            raise ValueError(f"solver: midpoint or euler (not {solver!r})")
        if sample_rate not in (48000, 24000):
            raise ValueError(f"sample_rate: 48000 or 24000 (not {sample_rate!r})")
        t0 = time.perf_counter()
        self.variant, self.files = variant, weight_dir(weights, bwe=sample_rate == 48000)
        self.redraw, self.cutoff, self.max_takes, self.margin_ms, self.lead_cut = redraw, cutoff, max_takes, margin_ms, lead_cut
        self.timing = {}
        self.lib = lib = ctypes.CDLL(str(engine_library()))
        self.timing["build_s"] = time.perf_counter() - t0
        lib.flash_engine_create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
        lib.flash_engine_create.restype = ctypes.c_void_p
        for f in (lib.flash_engine_info, lib.flash_engine_stats):
            f.argtypes, f.restype = [ctypes.c_void_p], ctypes.c_char_p
        lib.flash_engine_speak_chunks.argtypes = [ctypes.c_void_p, _P(ctypes.c_int32), _P(ctypes.c_int32), ctypes.c_int, ctypes.c_uint64,
                                                  ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, _P(Sink)]
        lib.flash_engine_destroy.argtypes = [ctypes.c_void_p]
        lib.flash_engine_set_sampler.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_double]
        lib.flash_engine_cut.argtypes, lib.flash_engine_cut.restype = [ctypes.c_void_p], ctypes.c_int64
        self.front = FrontEnd(lib._name)
        cpu_spec = "auto" if cpus is None else cpus if isinstance(cpus, str) else ",".join(str(int(c)) for c in cpus)
        opts = (f"temp={temp} passes={passes} solver={solver} cutoff={cutoff} max_takes={max_takes} margin_ms={margin_ms} "
                f"bwe={int(sample_rate == 48000)} {options}")
        err = ctypes.create_string_buffer(1024)
        self.h = lib.flash_engine_create(str(self.files).encode(), int(threads or 0), cpu_spec.encode(), opts.encode(), err, len(err))
        if not self.h:
            raise RuntimeError("the CPU engine could not start: " + err.value.decode())
        self.info = json.loads(lib.flash_engine_info(self.h))
        self.sample_rate = self.info["rate"]
        self.delay = BWE_DELAY if self.sample_rate == 48000 else 0
        if self.sample_rate != sample_rate:
            lib.flash_engine_destroy(self.h)
            self.h = None
            raise RuntimeError(f"the CPU engine could not load the upsampler for 48 kHz: {self.info['message']}")
        if self.info["asked"] > self.info["threads"]:
            warnings.warn(f"the CPU engine runs with {self.info['threads']} threads, not {self.info['asked']}: that is what the "
                          "process can run at once (physical cores it may use, CPU quota, other engines)", RuntimeWarning, stacklevel=2)
        self.judge = True if self.info["redraw"] else None    # the redraw's detector is there
        self.stats, self._busy = {}, threading.Lock()
        self.sampler, self.temp = (solver, passes if solver == "euler" else 2 * max(1, passes // 2)), float(temp)
        self.timing["total_s"] = time.perf_counter() - t0
        if verbose:
            print(f"CPU engine {variant} ({self.sample_rate // 1000} kHz): ready in {self.timing['total_s']:.2f} s, {self.info['threads']} threads"
                  + (f" on cores {self.info['cpus']}" if self.info["cpus"] else " (not pinned)"), flush=True)

    def close(self):
        """Stops the engine's threads and frees its memory (not while a request runs)."""
        if getattr(self, "h", None):
            if not self._busy.acquire(blocking=False):
                raise RuntimeError("a request is running: close its stream first")
            try:
                self.lib.flash_engine_destroy(self.h)
                self.h = None
            finally:
                self._busy.release()

    def __del__(self):
        self.close()

    def set_sampler(self, solver="midpoint", passes=6, temp=0.8):
        """solver euler | midpoint, head passes per frame (midpoint: 2 per step), temperature of the start noise; for
        the next requests (not while one runs)."""
        if solver not in ("euler", "midpoint"):
            raise ValueError(f"solver: midpoint or euler (not {solver!r})")
        if self.lib.flash_engine_set_sampler(self.h, int(solver == "euler"), int(passes), float(temp)):
            raise RuntimeError("a request is running")
        self.sampler, self.temp = (solver, passes if solver == "euler" else 2 * max(1, passes // 2)), float(temp)

    def stream(self, text, seed=0, verbalize=True, chunks=None, first_max=0, redraw=None, played=None, cut=None, boundary=None):
        """Yields (start, audio): float32 audio (self.sample_rate) for the utterance's samples from `start` on, as soon
        as it exists: normally one frame (3,200 samples at 24 kHz, about 6,400 at 48 kHz) right after the previous
        piece. After a redraw `start` lies before
        the end of what came so far: the audio from `start` on is replaced (only audio not yet played: at least
        margin_ms ahead of the player). chunks: code-point arrays instead of text (no front end). first_max > 0: a
        first chunk of about that many characters. redraw: None = the engine's setting. played: a function giving the
        samples the player has consumed so far (called from the request's thread; None: a player that starts with the
        first piece and plays at real time). cut: None = the engine's setting; on, the request starts at its first
        sound and every position (pieces, played) is in player samples, player sample p = utterance sample
        p + self.stats["cut"] (set when the first piece comes). boundary (instead of played): a function giving the samples that can no longer be
        replaced, for a caller that holds audio back before it passes it on; a redraw may then replace from any point
        at or after it (nothing is added for the time the redraw takes, the hold-back is the margin). With a boundary
        the engine goes on only when the caller has taken the piece, so the boundary it reads is up to date.
        self.stats holds the request's numbers afterwards (redraw counters in self.stats["redraw"], the native
        engine's own in self.stats["native"])."""
        if played is not None and boundary is not None:
            raise ValueError("played or boundary, not both")
        t0 = time.perf_counter()
        if chunks is None:
            chunks = self.front.chunks(text, verbalize, first_max)
        st = self.stats = {"chunks": len(chunks), "chars": int(sum(len(c) for c in chunks)), "frames": [], "t_front": time.perf_counter() - t0,
                           "t_first": None, "t_first_frame": None, "prefix": [], "t_chunk_first": [], "gap_max": 0.0, "cut": 0}
        if not chunks:
            return
        if not self._busy.acquire(blocking=False):
            raise RuntimeError("the engine is busy: one request at a time per engine")
        if not self.h:
            self._busy.release()
            raise RuntimeError("the engine is closed")
        use = self.judge is not None and (self.redraw if redraw is None else redraw)
        use_cut = self.lead_cut if cut is None else cut
        cps = np.ascontiguousarray(np.concatenate(chunks), np.int32)
        lens = np.array([len(c) for c in chunks], np.int32)
        q, stop, last = queue.SimpleQueue(), threading.Event(), [None]
        end, taken = object(), threading.Semaphore(0)
        where = played if played is not None else boundary
        up = self.sample_rate // 24000                  # the native start cut counts 24 kHz samples

        def audio(_, start, x, n):
            now = time.perf_counter()
            if last[0] is not None:
                st["gap_max"] = max(st["gap_max"], now - last[0])
            if last[0] is None:                         # the start cut is known with the first audio (as on the GPU)
                st["cut"] = max(0, self.lib.flash_engine_cut(self.h)) * up
            last[0] = now
            q.put((int(start), np.ctypeslib.as_array(x, (n,)).copy()))
            while boundary is not None and not stop.is_set() and not taken.acquire(timeout=0.05):
                pass                                    # until the caller has taken the piece (or left)
        sink = Sink(None, AUDIO(audio), REPLACE(), PLAYED(lambda _: int(where())) if where is not None else PLAYED(),
                    CANCELLED(lambda _: int(stop.is_set())))
        mode = 1 if where is None else 2 if boundary is None else 3      # real-time clock, playing position, boundary
        result = {}

        def run():
            try:
                result["r"] = self.lib.flash_engine_speak_chunks(self.h, cps.ctypes.data_as(_P(ctypes.c_int32)), lens.ctypes.data_as(_P(ctypes.c_int32)),
                                                                 len(chunks), seed, int(use), int(use_cut), mode, 0.0, ctypes.byref(sink))
            except BaseException as e:                  # noqa: BLE001  (handed to the caller's thread)
                result["e"] = e
            q.put(end)
        th = threading.Thread(target=run, name="flash-cpu-request", daemon=True)
        try:
            th.start()
            while (piece := q.get()) is not end:
                if st["t_first"] is None:
                    st["t_first"] = time.perf_counter() - t0
                yield piece
                taken.release()
        finally:
            stop.set()
            th.join()
            self._busy.release()
            st["t_total"] = time.perf_counter() - t0
        if "e" in result:
            raise result["e"]
        if result.get("r") == -2:
            raise ValueError("a chunk is too long for the key/value cache")
        if result.get("r", 0) < 0:
            raise RuntimeError(f"the CPU engine refused the request ({result['r']})")
        self._take_stats(st, use)

    def _take_stats(self, st, use):
        """The native engine's numbers in the GPU engine's terms."""
        js = json.loads(self.lib.flash_engine_stats(self.h))
        s = js["status"]
        st.update(frames=js["frames"], prefix=js["prefix"], cut=max(0, js["cut"]) * (self.sample_rate // 24000), unknown=s["unknown"], native=s)
        if use:
            spots = js["spots"]
            st["redraw"] = {"spots": s["spots"], "attempts": s["attempts"], "clean_after": s["clean"], "kept_best_unclean": s["kept_best"],
                            "given_up_played": sum(1 for p in spots if p.get("stop") == "budget" and len(p["takes"]) == 1),
                            "given_up_previous_chunk": s["previous_chunk"],
                            "cut_by_budget": sum(1 for p in spots if p.get("stop") == "budget" and len(p["takes"]) > 1),
                            "stopped_early": s["early_stop"], "stopped_by_playback": s["played_stop"], "swaps_too_late": s["refused"] + s["player_refused"],
                            "extra_frames": s["extra_frames"], "detector_ms": s["det_ms"] * s["frames"], "spot_log": spots}
