"""vLLM engines (TTS_BACKEND=vllm): one AsyncLLMEngine per codebook model, and the verbalizer.

The codebook models run through the Qwen3DualCodecForCausalLM adapter
(app/model/vllm_model.py, registered by app/model/vllm_plugin.py).
"""
from __future__ import annotations

import asyncio
import os

import torch

from app.logging_utils import get_logger

logger = get_logger("inference")


async def build_codebook_engines(config, model_paths: list[str]) -> list:
    gpu_mem_fracs = config.gpu_memory_per_cb
    max_num_seqs = config.max_num_seqs_per_cb
    max_num_batched_tokens = config.max_num_batched_tokens_per_cb
    logger.info(
        "Initializing %d vLLM engines with the %s capacity profile",
        len(model_paths),
        config.capacity_profile_name,
    )

    engine_tasks = []
    for cb, model_dir_name in enumerate(config.codebook_model_dirs):
        max_len = config.max_model_len_per_cb[cb]
        gpu_mem = gpu_mem_fracs[cb]
        logger.info(
            "CB%d: context=%d, active_sequences=%d, batched_tokens=%d, GPU fraction=%.3f",
            cb,
            max_len,
            max_num_seqs[cb],
            max_num_batched_tokens[cb],
            gpu_mem,
        )
        engine_tasks.append(_codebook_engine(
            config,
            model_paths[cb],
            target_codebook=cb,
            max_model_len=max_len,
            gpu_memory_utilization=gpu_mem,
            scheduling_policy="priority",
            max_num_seqs=max_num_seqs[cb],
            max_num_batched_tokens=max_num_batched_tokens[cb],
            kv_cache_memory_bytes=(
                int(config.kv_cache_gib_per_cb[cb] * 1024 ** 3)
                if config.kv_cache_gib_per_cb else None
            ),
        ))

    return list(await asyncio.gather(*engine_tasks))


async def _codebook_engine(
    config,
    model_path: str,
    target_codebook: int,
    max_model_len: int,
    gpu_memory_utilization: float = 0.2,
    max_num_seqs: int | None = 8,
    max_num_batched_tokens: int | None = None,
    scheduling_policy: str = "fcfs",
    kv_cache_memory_bytes: int | None = None,
):
    from vllm.config import CompilationConfig
    from vllm.config.compilation import CUDAGraphMode
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine

    logger.debug("Loading CB%d with vLLM from %s", target_codebook, model_path)

    # CPU-only path: no GPU memory fraction, no CUDA graphs / inductor.
    # vllm-cpu rejects compilation_config and gpu_memory_utilization.
    if not torch.cuda.is_available():
        engine_args = AsyncEngineArgs(
            model=model_path,
            dtype="bfloat16",
            hf_overrides={
                "architectures": ["Qwen3DualCodecForCausalLM"],
                "target_codebook": target_codebook,
            },
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs or 4,
            max_num_batched_tokens=max_num_batched_tokens,
            enforce_eager=True,
            enable_prefix_caching=False,
            disable_log_stats=True,
            scheduling_policy=scheduling_policy,
        )
        return AsyncLLMEngine.from_engine_args(engine_args)

    # Dense capture at small batch sizes (always used) + sparser capture up
    # to max_num_seqs so high-concurrency decode steps don't fall back to
    # piecewise execution. Each captured size costs ~7 MiB VRAM.
    effective_max_seqs = max_num_seqs or 8
    capture_sizes = [s for s in (1, 2, 3, 4, 5, 6, 7, 8) if s <= effective_max_seqs]
    for s in (12, 16, 20, 24, 28, 32, 40, 48, 56, 64):
        if s <= effective_max_seqs and s not in capture_sizes:
            capture_sizes.append(s)
    if effective_max_seqs not in capture_sizes:
        capture_sizes.append(effective_max_seqs)

    compilation_config = CompilationConfig(
        cudagraph_mode=CUDAGraphMode.FULL,
        cudagraph_capture_sizes=capture_sizes,
    )

    attention_backend = os.environ.get("VLLM_ATTENTION_BACKEND")

    engine_args = AsyncEngineArgs(
        model=model_path,
        dtype="bfloat16",
        hf_overrides={
            "architectures": ["Qwen3DualCodecForCausalLM"],
            "target_codebook": target_codebook,
        },
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enforce_eager=False,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        disable_log_stats=True,
        max_num_batched_tokens=max_num_batched_tokens,
        compilation_config=compilation_config,
        scheduling_policy=scheduling_policy,
        **({'max_num_seqs': max_num_seqs} if max_num_seqs is not None else {}),
        **({'attention_backend': attention_backend} if attention_backend else {}),
        **({'kv_cache_memory_bytes': kv_cache_memory_bytes} if kv_cache_memory_bytes else {}),
    )

    return AsyncLLMEngine.from_engine_args(engine_args)


def build_verbalizer_engine(model_path: str, max_len: int, max_num_seqs: int, gpu_util: float):
    """AsyncLLMEngine for the verbalizer LM, with n-gram speculative decoding."""
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine

    engine_args = AsyncEngineArgs(
        model=model_path,
        speculative_config={
            "method": "ngram",
            "num_speculative_tokens": 10,
            "prompt_lookup_max": 8,
            "prompt_lookup_min": 3,
        },
        max_model_len=max_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_len * max_num_seqs,
        gpu_memory_utilization=gpu_util,
        enforce_eager=False,
        dtype="bfloat16",
    )
    return AsyncLLMEngine.from_engine_args(engine_args)
