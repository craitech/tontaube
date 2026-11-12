"""LiteEngine-compatible generate() for one CPU codebook model, and building the four engines."""
from __future__ import annotations

import asyncio
import heapq
import itertools
import os
import threading
from pathlib import Path

import torch

from app.lite.cpu.model import CpuCodebookModel
from app.lite.requests import Completion, Output, Request
from app.lite.sampling import SamplingParams, SeqSampling, sample, update_counts
from app.logging_utils import get_logger
from app.native import kernels

logger = get_logger("lite")

_EXEC_LOCK = threading.Lock()  # one model step at a time across all CPU engines


class CpuEngine:
    """LiteEngine-compatible generate() for one CpuCodebookModel (batch of one)."""

    def __init__(self, model: CpuCodebookModel, name: str):
        self.model = model
        self.name = name
        self._g2l = {g: i for i, g in enumerate(model.local_ids.tolist())}
        self._l2g = model.local_ids.tolist()
        self._pending: list = []
        self._arrival = itertools.count()
        self._cond = threading.Condition()
        self._running = True
        self._fatal: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name=f"lite-cpu-{name}", daemon=True)
        self._thread.start()

    async def generate(self, prompt, sampling_params: SamplingParams, request_id: str, priority: int = 0):
        ids = list(prompt["prompt_token_ids"] if isinstance(prompt, dict) else prompt)
        if not ids or len(ids) >= self.model.max_len:
            raise ValueError(f"prompt of {len(ids)} tokens does not fit the {self.name} context "
                             f"limit of {self.model.max_len} tokens")
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        req = Request(request_id, ids, sampling_params, int(priority), loop, fut)
        requested = sampling_params.max_tokens if sampling_params.max_tokens is not None else self.model.max_len
        req.max_tokens = max(1, min(int(requested), self.model.max_len - len(ids)))

        def _on_done(f):
            if f.cancelled():
                req.aborted = True

        fut.add_done_callback(_on_done)
        with self._cond:
            if self._fatal is not None:
                raise RuntimeError(f"{self.name} engine stopped") from self._fatal
            heapq.heappush(self._pending, (req.priority, next(self._arrival), req))
            self._cond.notify()
        token_ids, reason = await fut
        yield Output(request_id, [Completion(token_ids, reason)])

    def shutdown(self):
        with self._cond:
            self._running = False
            self._cond.notify()
        self._thread.join(timeout=10)

    def _resolve(self, req, token_ids=None, exc=None, reason="stop"):
        def _set():
            if not req.future.done():
                if exc is not None:
                    req.future.set_exception(exc)
                else:
                    req.future.set_result((token_ids, reason))
        try:
            req.loop.call_soon_threadsafe(_set)
        except RuntimeError:
            pass

    def _run(self):
        try:
            while True:
                with self._cond:
                    while self._running and not self._pending:
                        self._cond.wait()
                    if not self._running:
                        break
                    _, _, req = heapq.heappop(self._pending)
                if req.aborted:
                    continue
                try:
                    self._serve(req)
                except Exception as exc:
                    logger.exception("%s request failed", self.name)
                    self._resolve(req, exc=exc)
            exc: BaseException = RuntimeError(f"{self.name} engine was shut down")
        except BaseException as err:
            logger.exception("%s engine worker stopped", self.name)
            exc = err
        with self._cond:
            self._fatal = exc
            failed = [r for _, _, r in self._pending]
            self._pending = []
        for req in failed:
            self._resolve(req, exc=exc)

    @torch.inference_mode()
    def _serve(self, req: Request):
        m = self.model
        seq = SeqSampling(req.params, m.n_local, self._g2l, m.eos_id, torch.device("cpu"))
        ids = req.prompt
        n = len(ids)
        pos = m.logical_positions(ids)
        with _EXEC_LOCK:
            reuse, offset = m.reusable_prefix(ids, pos)
            if reuse:
                logger.debug("%s: reusing %d of %d prompt tokens from the KV cache", self.name, reuse, n)
            logits = m.forward(ids[reuse:], [p + offset for p in pos[reuse:]], reuse)
        out: list[int] = []
        while True:
            tok = sample(logits[None].clone(), [seq], [len(out)])
            update_counts([seq], tok)
            gid = self._l2g[int(tok[0])]
            out.append(gid)
            if (gid in seq.stop_global and len(out) >= req.params.min_tokens) or len(out) >= req.max_tokens:
                reason = "stop" if gid in seq.stop_global else "length"
                self._resolve(req, out, reason=reason)
                return
            if req.aborted:
                return
            t = n + len(out) - 1  # sequence index of the fed token
            with _EXEC_LOCK:
                logits = m.forward([gid], [t + offset], t)


def build_cpu_engines(config, model_paths: list[str], plans, bundle,
                      text_rows: list[list[int]] | None = None) -> list[CpuEngine]:
    from app.lite.plans import parse_plan
    from app.lite.weights import LazyWeights, load_bundle

    torch.set_num_threads(int(os.getenv("LITE_CPU_THREADS", str(torch.get_num_threads()))))
    kernels()
    engines = []
    for cb, path in enumerate(model_paths):
        prequant = None
        if plans is None:
            prequant, plan = load_bundle(bundle, cb, path, torch.device("cpu"))
            plan = parse_plan(plan)
        else:
            plan = plans[cb]
        model = CpuCodebookModel(path, cb, plan, max_len=config.max_model_len_per_cb[cb],
                                 weights=LazyWeights(Path(path) / "model.safetensors", mmap=True),
                                 prequant=prequant, text_rows=text_rows[cb] if text_rows else None)
        logger.info("CB%d lite CPU model: %s; weights %.0f MiB, KV cache %.0f MiB (%d threads)",
                    cb, plan.describe(), model.weight_bytes() / 2**20,
                    model.kv.numel() * model.kv.element_size() / 2**20, torch.get_num_threads())
        engines.append(CpuEngine(model, f"cb{cb}"))
    return engines
