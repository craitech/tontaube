"""TontaubeV1.1 Flash in the server (TTS_MODEL=flash): one English voice (Marcus), on the GPU or the CPU.

On the GPU one batch engine (flash/gpu/batch.py) generates many requests at once, all their frames in each step;
FlashEngine runs a worker thread per request it can take (FLASH_ENGINES sets that number, by default as many as the
GPU's memory holds, at most 64). On the CPU the engine speaks one request at a time: FlashEngine keeps one or more of
them (one on the free cores by default; FLASH_ENGINES sets the number), each on its own worker thread, and hands each
request to the next free one; further requests wait.

The server's side of the interface: generate() and generate_streaming() (which takes the request at once, so a
refusal comes before any response is sent); EngineBusy when an engine cannot take a request now (status 503),
UnsupportedOption for a setting the engines cannot apply (status 422).

An engine's stream may replace recent audio (the redraw: start < end of what came so far). The server's clients only
append, so a stream goes through flash.common.append.AppendOnly (final audio; just in time only as a safety net)
and the engine's boundary is the audio sent. /predict waits for the whole utterance and lets every redraw happen.
"""
from __future__ import annotations

import asyncio
import contextlib
import importlib
import itertools
import os
import secrets
import threading
import time

import numpy as np

from app.config import LITE_ON_CPU
from app.logging_utils import get_logger
from app.priority_gate import get_gate

logger = get_logger("flash")

VARIANT = "int4"
DEFAULT_TEMPERATURE = 0.7
GPU_BATCH_MAX, GPU_RESERVE = 64, 2 << 30      # requests generated at once on one GPU at most; its memory left free
GPU_BASE, GPU_PER_REQUEST = 3 << 30, 128 << 20   # the batch engine's memory: weights, graphs; and per request


def device() -> str:
    """"cpu" or "gpu" (LITE_DEVICE, else the GPU when one is available)."""
    return "cpu" if LITE_ON_CPU else "gpu"


def weights_path() -> str | None:
    """FLASH_MODEL_PATH: a local copy of the weights laid out as on Hugging Face; None downloads them."""
    return os.getenv("FLASH_MODEL_PATH") or None


def engine_module(dev: str):
    """flash/<dev>/engine.py (the flash package next to app/)."""
    return importlib.import_module(f"flash.{dev}.engine")


def engine_count() -> int | None:
    """FLASH_ENGINES: requests served at once (CPU: one engine each); None = the default (see build_engines)."""
    raw = os.getenv("FLASH_ENGINES")
    return int(raw) if raw else None


def sample_rate() -> int:
    """FLASH_SAMPLE_RATE: 48000 (default; the model's 24 kHz raised by the upsampler) or 24000, which is faster."""
    rate = int(os.getenv("FLASH_SAMPLE_RATE") or 48000)
    if rate not in (48000, 24000):
        raise ValueError(f"unsupported FLASH_SAMPLE_RATE {rate}; expected 48000 or 24000")
    return rate


def build_engines(dev: str, weights: str | None = None, n: int | None = None, rate: int = 48000) -> list:
    """The engines. CPU: one on the free physical cores by default; n > 1 share them equally (cores and container quota
    read from a one-thread probe). GPU: one batch engine for n requests at once (default: as many as the free memory
    holds, at most GPU_BATCH_MAX, GPU_RESERVE left free). rate: the output sample rate."""
    mod = engine_module(dev)
    if dev == "cpu":
        threads = {}
        if n and n > 1:
            probe = mod.Engine(VARIANT, threads=1, weights=weights, sample_rate=rate)
            cores = min(probe.info["physical_cores"], int(probe.info["quota"]) or probe.info["physical_cores"])
            probe.close()
            threads = {"threads": max(1, cores // n)}
        return [mod.Engine(VARIANT, weights=weights, sample_rate=rate, **threads) for _ in range(n or 1)]
    import torch

    if n is None:
        free = torch.cuda.mem_get_info()[0]
        n = int(max(1, min(GPU_BATCH_MAX, (free - GPU_RESERVE - GPU_BASE) // GPU_PER_REQUEST)))
    return [batch_module().BatchEngine(VARIANT, weights=weights, max_batch=n, sample_rate=rate)]


def batch_module():
    return importlib.import_module("flash.gpu.batch")


class EngineBusy(RuntimeError):
    """The engine cannot take the request now (all its streams busy); the server answers 503."""


class UnsupportedOption(ValueError):
    """A request setting the engines cannot apply; the server answers 422."""


class _Turns:
    """A lock that lets waiting threads in in arrival order (GPU engines take turns frame by frame)."""

    def __init__(self):
        self._cond = threading.Condition()
        self._tickets = itertools.count()
        self._serving = 0

    def __enter__(self):
        with self._cond:
            ticket = next(self._tickets)
            self._cond.wait_for(lambda: self._serving == ticket)

    def __exit__(self, *exc):
        with self._cond:
            self._serving += 1
            self._cond.notify_all()


class _Job:
    def __init__(self, text, seed, temperature, verbalize, priority, live):
        self.loop = asyncio.get_running_loop()
        self.queue: asyncio.Queue = asyncio.Queue()
        self.text, self.verbalize, self.live = text, verbalize, live
        self.seed = secrets.randbelow(2**31) if seed is None else int(seed)
        self.temperature = DEFAULT_TEMPERATURE if temperature is None else float(temperature)
        self.low = priority != "high"
        self.cancelled = False

    def put(self, item) -> None:
        try:
            self.loop.call_soon_threadsafe(self.queue.put_nowait, item)
        except RuntimeError:                    # event loop closed
            self.cancelled = True


class _Pieces:
    """A stream request's pieces as an async iterator; leaving it early (or dropping it) cancels the request."""

    def __init__(self, job: _Job):
        self.job = job

    def __aiter__(self):
        return self

    async def __anext__(self) -> np.ndarray:
        try:
            piece = await self.job.queue.get()
        except BaseException:
            self.job.cancelled = True
            raise
        if piece is None or isinstance(piece, BaseException):
            self.job.cancelled = True
            if piece is None:
                raise StopAsyncIteration
            raise piece
        return piece

    async def aclose(self) -> None:
        self.job.cancelled = True

    def __del__(self):
        self.job.cancelled = True


class FlashEngine:
    """Engines behind the interface main.py uses: generate() -> float32 audio at self.sample_rate (the engines' rate:
    48 kHz by default), generate_streaming() -> float32 pieces to append."""

    can_stream = True
    controls = None             # no style sliders (GET /controls lists none)

    def __init__(self, engines: list, serialize: bool = False):
        self.engines = engines
        self.sample_rate = engines[0].sample_rate     # the engines are built alike
        self.fixed_temperature = not all(hasattr(eng, "set_sampler") for eng in engines)   # set when they are built
        self._turn = _Turns() if serialize and len(engines) > 1 else contextlib.nullcontext()
        self._jobs: list[_Job] = []
        self._running: set[_Job] = set()
        self._cond = threading.Condition()
        self._stopping = False
        # a worker thread per request an engine takes at once (a batch engine: max_batch)
        self._threads = [threading.Thread(target=self._work, args=(eng,), name=f"flash-{i}-{k}", daemon=True)
                         for i, eng in enumerate(engines) for k in range(getattr(eng, "max_batch", 1))]
        for thread in self._threads:
            thread.start()

    # ------------------------------------------------------------------------------------------------ public API
    async def generate(self, text, seed=None, temperature=None, verbalize=True, priority="low") -> np.ndarray:
        """The whole utterance (every redraw applied)."""
        job = self._submit(text, seed, temperature, verbalize, priority, live=False)
        try:
            audio = await job.queue.get()
            if isinstance(audio, BaseException):
                raise audio
            return audio
        finally:
            job.cancelled = True

    def generate_streaming(self, text, seed=None, temperature=None, verbalize=True, priority="low") -> _Pieces:
        """Takes the request now; the pieces of the utterance in order, each to be appended to the ones before."""
        return _Pieces(self._submit(text, seed, temperature, verbalize, priority, live=True))

    def shutdown(self) -> None:
        with self._cond:
            self._stopping = True
            for job in [*self._jobs, *self._running]:
                job.cancelled = True
            self._cond.notify_all()
        for thread in self._threads:
            thread.join(timeout=10)
        for eng in self.engines:
            try:
                if hasattr(eng, "close"):   # the CPU engine's threads and memory
                    eng.close()
            except RuntimeError as exc:     # a request did not stop in time
                logger.warning("Flash engine not closed: %s", exc)
        self.engines.clear()

    # ----------------------------------------------------------------------------------------------- the workers
    def _submit(self, text, seed, temperature, verbalize, priority, live) -> _Job:
        if self.fixed_temperature and temperature not in (None, DEFAULT_TEMPERATURE):
            raise UnsupportedOption(f"this server's engines use a fixed temperature ({DEFAULT_TEMPERATURE}); "
                                    "omit temperature")
        job = _Job("\n".join(text) if isinstance(text, list) else text, seed, temperature, verbalize, priority, live)
        with self._cond:
            if self._stopping:
                raise EngineBusy("the server is shutting down")
            self._jobs.append(job)
            self._cond.notify()
        return job

    def _work(self, eng) -> None:
        while True:
            with self._cond:
                self._cond.wait_for(lambda: self._jobs or self._stopping)
                if self._stopping:
                    return
                job = self._jobs.pop(0)
                self._running.add(job)
            try:
                if not job.cancelled:
                    self._run(eng, job)
            except Exception as exc:              # the request fails, the engine stays
                logger.exception("Flash request failed")
                job.put(exc)
            finally:
                with self._cond:
                    self._running.discard(job)

    def _run(self, eng, job: _Job) -> None:
        from flash.common.append import HOP, SR, AppendOnly

        t0, first, sr = time.perf_counter(), None, eng.sample_rate
        pieces = None
        stats = lambda: getattr(pieces, "stats", None) or eng.stats   # noqa: E731  (a batch engine's request has its own)
        # frame 0 (never redrawn) is final at once: its player samples after the start cut (engines that report the
        # cut only at the end count it as 0), less what an upsampler still holds back of it
        hop, delay = HOP * sr // SR, getattr(eng, "delay", 0)
        out = (AppendOnly(start_final=lambda: hop - (stats().get("cut") or 0) - delay, frame_s=getattr(eng, "frame_s", None),
                          sample_rate=sr) if job.live else AppendOnly(hold=None, sample_rate=sr))
        where = {"boundary": out.boundary} if job.live else {"played": lambda: -1e18}   # /predict: no playback, every redraw
        gate = get_gate()
        batched = hasattr(eng, "submit")              # temperature and priority per request
        with self._turn:
            if not self.fixed_temperature and not batched:
                eng.set_sampler(*eng.sampler, job.temperature)
            extra = {"temp": job.temperature, "priority": int(job.low)} if batched else {}
            pieces = eng.stream(job.text, seed=job.seed, verbalize=job.verbalize, **where, **extra)
        try:
            while not job.cancelled:
                while job.low and gate.holds_low() and not job.cancelled:   # priority: low requests wait
                    time.sleep(0.005)
                with self._turn:
                    piece = next(pieces, None)
                if piece is None:
                    break
                audio = out.put(*piece)
                if len(audio):
                    job.put(audio)
                    first = first or time.perf_counter() - t0
            else:
                return
        finally:
            with self._turn:
                pieces.close()
        rest = out.flush()
        if job.live:
            if len(rest):
                job.put(rest)
            job.put(None)
        else:
            job.put(rest)
        seconds, st = time.perf_counter() - t0, stats()
        rd = st.get("redraw") or {}
        refused = sum(rd.get(k, 0) for k in ("given_up_played", "cut_by_budget", "stopped_by_playback", "swaps_too_late"))
        logger.info("Flash: %.2f s of audio in %.2f s (%.1fx real time)%s%s", out.n / sr, seconds,
                    out.n / sr / max(seconds, 1e-9),
                    f", first audio after {1e3 * (first or seconds):.0f} ms (engine: {1e3 * (st.get('t_first') or 0):.0f} ms)"
                    if job.live else "",
                    f"; redraw: {rd['spots']} spot(s), {rd['attempts']} attempt(s), {refused} refused by the boundary"
                    if rd else "")
        if out.early:
            logger.info("Flash: %.2f s released before final (the client would have run out of audio)", out.early / sr)
        if out.reached:
            logger.warning("Flash: %d piece(s) started inside audio already sent; kept as sent", out.reached)


_engine: FlashEngine | None = None


async def get_engine() -> FlashEngine:
    global _engine
    if _engine is None:
        dev, started = device(), time.perf_counter()
        engines = await asyncio.to_thread(build_engines, dev, weights_path(), engine_count(), sample_rate())
        _engine = FlashEngine(engines, serialize=dev == "gpu")
        logger.info("TontaubeV1.1 Flash (%s, %s): %d request(s) at once, ready in %.1f s", VARIANT, dev,
                    sum(getattr(eng, "max_batch", 1) for eng in engines), time.perf_counter() - started)
    return _engine


def unload_engine() -> None:
    global _engine
    if _engine is not None:
        _engine.shutdown()
        _engine = None


def preflight() -> str:
    """Fail-fast checks before the engines load: the native library (the CPU engine, on the GPU the text front end;
    from the cache, prebuilt, or built here with a C++ compiler) and the weights (downloaded here when not local).
    Returns a one-line summary."""
    from flash.common.build import engine_units, library, text_units

    dev = device()
    library(*(text_units() if dev == "gpu" else engine_units()))
    mod = engine_module(dev)
    if dev == "gpu":
        files = [str(p) for p in mod.weight_files(mod.VARIANTS[VARIANT], weights_path()).values() if p]
        folder = os.path.commonpath(files)
    else:
        folder = mod.weight_dir(weights_path(), bwe=sample_rate() == 48000)
    return f"TontaubeV1.1 Flash {VARIANT} on the {dev.upper()}, weights in {folder}"
