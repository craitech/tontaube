"""Asynchronous continuous-batching engine around one LiteCodebookModel.

`LiteEngine.generate()` mirrors the part of vLLM's AsyncLLMEngine.generate()
that app.inference uses: it takes a TokensPrompt, SamplingParams, a request
id and a priority (lower runs first) and yields a final output object with
`.outputs[0].token_ids` (stop tokens included, like vLLM).

Each engine owns a worker thread with its own CUDA stream. Requests are
admitted in (priority, arrival) order when a batch slot and enough KV blocks
for prompt + max_tokens are free; prefill runs per request, decode steps run
batched through the model's captured CUDA graphs.
"""
from __future__ import annotations

import asyncio
import heapq
import itertools
import math
import threading
import time
from pathlib import Path

import torch

from app.lite.gpu.model import BLOCK_SIZE, LiteCodebookModel
from app.lite.requests import Completion, Output, Request
from app.lite.weights import LazyWeights, load_bundle
from app.lite.plans import parse_plan
from app.lite.sampling import SamplingParams, SeqSampling, sample, update_counts
from app.logging_utils import get_logger

logger = get_logger("lite")


class LiteEngine:
    def __init__(self, model: LiteCodebookModel, name: str):
        self.model = model
        self.name = name
        self.max_batch = model.max_batch
        self._g2l = {g: i for i, g in enumerate(model.local_ids.tolist())}
        self._l2g = model.local_ids.tolist()
        self._l2g_dev = model.local_ids.to(model.device)
        self._batch_key: tuple = ()
        self._free_blocks = list(range(model.n_blocks))
        self._pending: list[tuple[int, int, Request]] = []
        self._active: list[Request] = []
        self._arrival = itertools.count()
        self._cond = threading.Condition()
        self._running = True
        self._fatal: BaseException | None = None
        self._stream = torch.cuda.Stream(device=model.device)
        self._table_host = torch.full((self.max_batch, model.blocks_per_seq),
                                      model.scratch_block, dtype=torch.int32)
        self._thread = threading.Thread(target=self._run, name=f"lite-{name}", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------ public API
    async def generate(self, prompt, sampling_params: SamplingParams, request_id: str,
                       priority: int = 0):
        ids = list(prompt["prompt_token_ids"] if isinstance(prompt, dict) else prompt)
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        req = Request(request_id, ids, sampling_params, int(priority), loop, fut)
        self._validate(req)

        def _on_done(f: asyncio.Future):
            if f.cancelled():
                req.aborted = True
                with self._cond:
                    self._cond.notify()

        fut.add_done_callback(_on_done)
        with self._cond:
            if self._fatal is not None:
                raise RuntimeError(f"{self.name} engine stopped") from self._fatal
            req.arrival = next(self._arrival)
            heapq.heappush(self._pending, (req.priority, req.arrival, req))
            self._cond.notify()
        token_ids, reason = await fut
        yield Output(request_id, [Completion(token_ids, reason)])

    def shutdown(self):
        with self._cond:
            self._running = False
            self._cond.notify()
        self._thread.join(timeout=10)

    # --------------------------------------------------------------- helpers
    def _validate(self, req: Request):
        max_len = self.model.max_len
        if req.n_prompt == 0:
            raise ValueError("empty prompt")
        if req.n_prompt >= max_len:
            raise ValueError(
                f"prompt of {req.n_prompt} tokens exceeds the {self.name} context "
                f"limit of {max_len} tokens"
            )
        requested = req.params.max_tokens if req.params.max_tokens is not None else max_len
        req.max_tokens = max(1, min(int(requested), max_len - req.n_prompt))
        needed = math.ceil((req.n_prompt + req.max_tokens) / BLOCK_SIZE)
        if needed > self.model.n_blocks:
            raise ValueError(f"request needs {needed} KV blocks, pool has {self.model.n_blocks}")

    def _resolve(self, req: Request, token_ids=None, exc: BaseException | None = None,
                 reason: str = "stop"):
        def _set():
            if req.future.done():
                return
            if exc is not None:
                req.future.set_exception(exc)
            else:
                req.future.set_result((token_ids, reason))

        try:
            req.loop.call_soon_threadsafe(_set)
        except RuntimeError:  # event loop closed
            pass

    def _release(self, req: Request):
        self._free_blocks.extend(req.blocks)
        req.blocks = []

    def _admit(self) -> list[Request]:
        """Pop admissible requests (called with the condition held)."""
        admitted = []
        while self._pending and len(self._active) + len(admitted) < self.max_batch:
            _, _, req = self._pending[0]
            if req.aborted:
                heapq.heappop(self._pending)
                continue
            needed = math.ceil((req.n_prompt + req.max_tokens) / BLOCK_SIZE)
            if needed > len(self._free_blocks):
                break
            heapq.heappop(self._pending)
            req.blocks = [self._free_blocks.pop() for _ in range(needed)]
            admitted.append(req)
        return admitted

    def _finish_if_done(self, req: Request, gid: int) -> bool:
        if gid in req.sampling.stop_global and len(req.out) >= req.params.min_tokens:
            self._release(req)
            self._resolve(req, list(req.out), reason="stop")
            return True
        if len(req.out) >= req.max_tokens:
            self._release(req)
            self._resolve(req, list(req.out), reason="length")
            return True
        return False

    # ------------------------------------------------------------ main loop
    def _run(self):
        try:
            self._loop()
            exc: BaseException = RuntimeError(f"{self.name} engine was shut down")
        except BaseException as err:  # never leave callers waiting forever
            logger.exception("%s engine worker stopped", self.name)
            exc = err
        with self._cond:
            self._fatal = exc
            failed = [r for _, _, r in self._pending] + self._active
            self._pending, self._active = [], []
        for req in failed:
            self._resolve(req, exc=exc)

    def _loop(self):
        torch.cuda.set_device(self.model.device)
        with torch.cuda.stream(self._stream), torch.inference_mode():
            while True:
                with self._cond:
                    while self._running and not self._pending and not self._active:
                        self._cond.wait()
                    if not self._running:
                        break
                    admitted = self._admit()
                for req in admitted:
                    self._prefill(req)
                self._active = [r for r in self._active if not self._drop_if_aborted(r)]
                if self._active:
                    try:
                        self._decode_step()
                    except Exception as exc:  # pragma: no cover - surfaced to callers
                        logger.exception("%s decode step failed", self.name)
                        for req in self._active:
                            self._release(req)
                            self._resolve(req, exc=exc)
                        self._active = []
                elif not admitted:
                    # Nothing runnable (e.g. waiting for KV blocks): block until notified.
                    with self._cond:
                        if self._running and not self._active:
                            self._cond.wait(timeout=0.05)

    def _drop_if_aborted(self, req: Request) -> bool:
        if req.aborted:
            self._release(req)
            return True
        return False

    def _prefill(self, req: Request):
        try:
            req.sampling = SeqSampling(
                req.params, self.model.n_local, self._g2l,
                self.model.eos_id, self.model.device,
            )
            logits = self.model.prefill(req.prompt, req.blocks)
            tok = sample(logits[None].clone(), [req.sampling], [0])
            update_counts([req.sampling], tok)
            gid = self._l2g[int(tok[0])]
        except Exception as exc:
            logger.exception("%s prefill failed", self.name)
            self._release(req)
            self._resolve(req, exc=exc)
            return
        req.out.append(gid)
        if not self._finish_if_done(req, gid):
            self._active.append(req)

    def _compose(self, active: list[Request]) -> None:
        """Load the device-side state for a new batch composition."""
        ids, pos, rows, lens = [], [], [], []
        for i, req in enumerate(active):
            t = req.n_prompt + len(req.out) - 1  # sequence index of the fed token
            ids.append(req.out[-1])
            pos.append(t)
            rows.append(req.blocks[t // BLOCK_SIZE] * BLOCK_SIZE + t % BLOCK_SIZE)
            lens.append(t + 1)
            nb = len(req.blocks)
            self._table_host[i, :nb] = torch.tensor(req.blocks, dtype=torch.int32)
            self._table_host[i, nb:] = self.model.scratch_block
        self.model.set_batch(ids, pos, self._table_host[: len(active)], lens, rows)
        self._batch_key = tuple(r.arrival for r in active)

    @staticmethod
    def _greedy_only(req: Request) -> bool:
        """Greedy without biases, penalties or reachable stop tokens."""
        s = req.sampling
        return s.greedy and s.bias is None and s.counts is None and s.stop_local is None

    def _decode_step(self):
        active = self._active
        if tuple(r.arrival for r in active) != self._batch_key:
            self._compose(active)
        still = []
        if all(self._greedy_only(r) for r in active):
            # No sampling state and nothing can stop early: run a chunk of
            # steps on the device with a single synchronization.
            k = min(self.model.GREEDY_CHUNK, min(r.max_tokens - len(r.out) for r in active))
            for req, toks in zip(active, self.model.greedy_steps(k).tolist()):
                req.out.extend(toks)
                if not self._finish_if_done(req, toks[-1]):
                    still.append(req)
        else:
            logits = self.model.step_logits()
            seqs = [r.sampling for r in active]
            toks = sample(logits.clone(), seqs, [len(r.out) for r in active])
            update_counts(seqs, toks)
            self.model.advance(self._l2g_dev[toks])
            for req, lid in zip(active, toks.tolist()):
                gid = self._l2g[lid]
                req.out.append(gid)
                if not self._finish_if_done(req, gid):
                    still.append(req)
        self._active = still


def graph_batch_sizes(max_batch: int) -> list[int]:
    sizes = [s for s in range(1, min(max_batch, 8) + 1)]
    sizes += [s for s in (12, 16, 24, 32, 48, 64) if s < max_batch]
    if max_batch not in sizes:
        sizes.append(max_batch)
    return sizes


def build_gpu_engines(config, model_paths: list[str], plans, bundle: Path,
                      text_rows: list[list[int]] | None = None) -> list["LiteEngine"]:
    """The four codebook engines on the GPU (plans=None: load the pre-quantized bundle)."""
    engines = []
    for cb, path in enumerate(model_paths):
        t0 = time.time()
        prequant = None
        if plans is None:
            prequant, plan = load_bundle(bundle, cb, path, torch.device("cuda"))
            plan = parse_plan(plan)
        else:
            plan = plans[cb]
        model = LiteCodebookModel(
            path,
            target_codebook=cb,
            plan=plan,
            prequant=prequant,
            max_len=config.max_model_len_per_cb[cb],
            max_batch=config.max_num_seqs_per_cb[cb],
            kv_tokens=config.lite_kv_tokens_per_cb[cb],
            weights=LazyWeights(Path(path) / "model.safetensors"),
            text_rows=text_rows[cb] if text_rows else None,
        )
        model.capture_graphs(graph_batch_sizes(model.max_batch))
        sizes = model.weight_bytes()
        logger.info(
            "CB%d lite model: %s; GPU weights %.0f MiB (linear %.0f, head %.0f, "
            "audio embeddings %.0f), KV pool %.0f MiB for %d tokens, "
            "text embeddings %.0f MiB in host memory, %.1fs",
            cb, plan.describe(),
            (sizes["linear"] + sizes["head"] + sizes["embed_gpu"] + sizes["norms"]) / 2**20,
            sizes["linear"] / 2**20, sizes["head"] / 2**20, sizes["embed_gpu"] / 2**20,
            sizes["kv_pool"] / 2**20, model.n_blocks * 64,
            sizes["embed_cpu"] / 2**20, time.time() - t0,
        )
        engines.append(LiteEngine(model, f"cb{cb}"))
    return engines
