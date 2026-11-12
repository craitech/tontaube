"""The lite engines: TontaubeV1 codebook models in this process, quantized, on a GPU or a CPU.

Public API (everything outside app/lite uses only this):
  build_codebook_engines()  the four codebook engines for a ServerConfig
  build_causal_lm_engine()  a GPU engine for a plain causal LM (the verbalizer)
  lite_device()             "cuda" or "cpu" (LITE_DEVICE, else the GPU when available)
  SamplingParams, TokensPrompt   vLLM-compatible request types

Layout: config / quant / plans / weights / sampling / requests are shared; gpu/
holds the Triton kernels, paged model and continuous-batching engine; cpu/ the
AVX2 model and engine (kernels in app/native); quantize/ builds GPTQ bundles.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

from app.lite.plans import parse_codebook_plans
from app.lite.sampling import SamplingParams, TokensPrompt
from app.logging_utils import get_logger

__all__ = ["SamplingParams", "TokensPrompt", "build_causal_lm_engine", "build_codebook_engines", "lite_device"]

logger = get_logger("lite")


def lite_device() -> str:
    """LITE_DEVICE (cuda/cpu); defaults to the GPU when one is available."""
    device = os.getenv("LITE_DEVICE", "cuda" if torch.cuda.is_available() else "cpu").strip().lower()
    if device not in ("cuda", "cpu"):
        raise ValueError(f"unsupported LITE_DEVICE {device!r}; expected 'cuda' or 'cpu'")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("LITE_DEVICE=cuda but no CUDA GPU is available")
    return device


def resolve_lite_quant(quant: str, model_paths: list[str]) -> str:
    """LITE_QUANT "auto": the int4/ bundle next to the codebook models if there is one, else bf16."""
    if quant != "auto":
        return quant
    bundle = Path(model_paths[0]).parent / "int4"
    return str(bundle) if (bundle / "bundle.json").is_file() else "bf16"


def build_codebook_engines(config, model_paths: list[str], text_rows: list[list[int]] | None = None) -> list:
    """Codebook engines; text_rows[cb] restricts each model's text embedding table."""
    quant = resolve_lite_quant(config.lite_quant, model_paths)
    bundle = Path(quant).expanduser()
    if bundle.is_dir():
        plans = None
        logger.info("Loading pre-quantized lite bundle from %s", bundle)
    else:
        plans = parse_codebook_plans(quant, len(model_paths))
    if lite_device() == "cpu":
        from app.lite.cpu.engine import build_cpu_engines

        return build_cpu_engines(config, model_paths, plans, bundle, text_rows)
    from app.lite.gpu.engine import build_gpu_engines

    return build_gpu_engines(config, model_paths, plans, bundle, text_rows)


def build_causal_lm_engine(model_path: str | Path, quant: str, max_len: int, max_batch: int, name: str = "causal-lm"):
    """A GPU engine for a plain Qwen3 causal LM with tied head (the verbalizer), CUDA graphs captured."""
    from app.lite.gpu.engine import LiteEngine, graph_batch_sizes
    from app.lite.gpu.model import LiteCausalLM
    from app.lite.plans import parse_plan
    from app.lite.weights import LazyWeights

    model = LiteCausalLM(model_path, parse_plan(quant), max_len=max_len, max_batch=max_batch,
                         kv_tokens=max_len * max_batch, weights=LazyWeights(Path(model_path) / "model.safetensors"))
    model.capture_graphs(graph_batch_sizes(max_batch))
    return LiteEngine(model, name)
