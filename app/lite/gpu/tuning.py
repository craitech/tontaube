"""Launch-configuration tuning for the quantized GEMV kernel.

The best tile width, K split and warp count depend on the weight shape, the
bit width and the GPU. On the first use of a shape (outside CUDA-graph
capture) candidate configurations are timed inside a CUDA graph, as they run
during decoding, and the fastest is kept. Results are cached on disk per GPU
model and Triton version, so later starts skip the measurement.
"""
from __future__ import annotations

import itertools
import json
import os
import threading
from pathlib import Path

import torch

from app.logging_utils import get_logger

logger = get_logger("lite")
_LOCK = threading.Lock()
_CACHE: dict[str, list[int]] | None = None
_CACHE_PATH: Path | None = None


def _cache_path() -> Path:
    import triton

    root = Path(os.getenv("TONTAUBE_CACHE_DIR", Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "tontaube"))
    name = torch.cuda.get_device_name().replace(" ", "_").replace("/", "_")
    return root.expanduser() / "gemv-tuning" / f"{name}-triton{triton.__version__}.json"


def _load() -> dict[str, list[int]]:
    global _CACHE, _CACHE_PATH
    if _CACHE is None:
        _CACHE_PATH = _cache_path()
        try:
            _CACHE = json.loads(_CACHE_PATH.read_text())
        except (OSError, ValueError):
            _CACHE = {}
    return _CACHE


def _save() -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _CACHE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(_CACHE, indent=0, sort_keys=True))
        tmp.replace(_CACHE_PATH)
    except OSError:
        pass  # tuning results are an optimization only


def heuristic(n: int, k: int, group: int) -> tuple[int, int, int]:
    """(BLOCK_N, SPLIT_K, num_warps) used when tuning is not possible."""
    n_chunks = k // min(128, group)
    split_k = 1
    while (n // 32) * split_k < 256 and n_chunks // (split_k * 2) >= 2:
        split_k *= 2
    return 32, split_k, 4


def _candidates(n: int, k: int, group: int):
    n_chunks = k // min(128, group)
    for block_n, split_k, warps in itertools.product((16, 32, 64, 128), (1, 2, 4, 8), (2, 4, 8)):
        programs = -(-n // block_n) * split_k
        if split_k <= n_chunks and 32 <= programs <= 4096:
            yield block_n, split_k, warps


def _time(run, reps: int = 20) -> float:
    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, capture_error_mode="thread_local"):
        for _ in range(reps):
            run()
    graph.replay()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(3):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end)


def gemv_config(bits: int, group: int, n: int, k: int, launch) -> tuple[int, int, int]:
    """Best (BLOCK_N, SPLIT_K, num_warps) for this shape.

    `launch(block_n, split_k, warps, x, out, part)` runs one GEMV; it is
    used for timing when the shape has not been tuned yet.
    """
    key = f"{bits},{group},{n},{k}"
    cache = _load()
    if key in cache:
        return tuple(cache[key])
    if torch.cuda.is_current_stream_capturing():
        return heuristic(n, k, group)
    # All graph captures in the process run in inference mode: the first one
    # creates PyTorch's graph-safe RNG state as inference tensors, which a
    # capture outside inference mode cannot update.
    with _LOCK, torch.inference_mode():
        if key in cache:
            return tuple(cache[key])
        dev = torch.cuda.current_device()
        x = torch.randn((1, k), device=dev, dtype=torch.bfloat16)
        out = torch.empty((1, n), device=dev, dtype=torch.bfloat16)
        part = torch.empty((8, 1, n), device=dev, dtype=torch.float32)
        best, best_t = heuristic(n, k, group), float("inf")
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for cfg in _candidates(n, k, group):
                try:
                    launch(*cfg, x, out, part)  # compile / resource check outside capture
                    torch.cuda.synchronize()
                except Exception:  # e.g. shared-memory limits on small GPUs
                    continue
                try:
                    t = _time(lambda: launch(*cfg, x, out, part))
                except Exception as exc:
                    # A failed capture can leave CUDA state inconsistent: stop
                    # measuring and keep the best result so far.
                    logger.warning("GEMV tuning stopped for %s: %s", key, exc)
                    break
                if t < best_t:
                    best, best_t = cfg, t
        torch.cuda.current_stream().wait_stream(stream)
        cache[key] = list(best)
        _save()
        return best
