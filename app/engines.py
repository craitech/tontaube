"""Token engines for the configured backend: the only module that knows both backends.

The pipeline (app/inference.py, app/verbalizer.py) builds its engines here and
uses them only through generate(TokensPrompt(...), SamplingParams(...),
request_id, priority), whose final output carries .outputs[0].token_ids: the
interface of vLLM's AsyncLLMEngine, which the lite engines implement.

  lite  app/lite: all models in this process, quantized, on a GPU or the CPU
  vllm  app/model/vllm_engines.py: one AsyncLLMEngine per model
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from app.config import TTS_BACKEND
from app.logging_utils import get_logger

logger = get_logger("inference")

if TTS_BACKEND == "vllm":
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    try:
        from vllm import SamplingParams, TokensPrompt
    except ImportError as exc:
        raise ImportError("TTS_BACKEND=vllm needs vLLM, an optional extra: install it with `uv sync --extra vllm` "
                          "(or run with `uv run --extra vllm ...`)") from exc

    from app.model.vllm_plugin import register as _register_vllm_model

    _register_vllm_model()
else:
    from app.lite import SamplingParams, TokensPrompt

__all__ = ["SamplingParams", "TokensPrompt", "build_codebook_engines", "build_verbalizer_block_lm",
           "build_verbalizer_engine", "shutdown_engine"]


async def build_codebook_engines(config, model_paths: list[str], tokenizers: list) -> list:
    """One engine per codebook model, in codebook order."""
    if config.backend == "lite":
        from app.lite import build_codebook_engines as build_lite
        from app.model.text_vocab import text_token_rows

        logger.info("Initializing %d lite engines with the %s capacity profile",
                    len(model_paths), config.capacity_profile_name)
        return build_lite(config, model_paths, text_rows=[text_token_rows(t) for t in tokenizers])
    from app.model.vllm_engines import build_codebook_engines as build_vllm

    return await build_vllm(config, model_paths)


def build_verbalizer_engine(backend: str, model_path: str, max_len: int, max_num_seqs: int,
                            gpu_util: float, lite_quant: str):
    """(engine, greedy SamplingParams, TokensPrompt) for the verbalizer LM."""
    if backend == "lite":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("the verbalizer needs a CUDA GPU; start without ENABLE_VERBALIZATION=1 on a machine "
                               "without one")
        from app.lite import SamplingParams as LiteSamplingParams
        from app.lite import TokensPrompt as LiteTokensPrompt
        from app.lite import build_causal_lm_engine

        engine = build_causal_lm_engine(model_path, lite_quant, max_len=max_len, max_batch=max_num_seqs,
                                        name="verbalizer")
        # vLLM stops on every EOS id of the generation config.
        stop_ids = {engine.model.eos_id}
        gen = Path(model_path) / "generation_config.json"
        if gen.exists():
            eos = json.loads(gen.read_text()).get("eos_token_id")
            stop_ids.update(eos if isinstance(eos, list) else [eos])
        params = LiteSamplingParams(temperature=0, max_tokens=1024,
                                    stop_token_ids=sorted(i for i in stop_ids if i is not None))
        return engine, params, LiteTokensPrompt
    from vllm import SamplingParams as VllmSamplingParams
    from vllm.inputs import TokensPrompt as VllmTokensPrompt

    from app.model.vllm_engines import build_verbalizer_engine as build_vllm

    engine = build_vllm(model_path, max_len, max_num_seqs, gpu_util)
    return engine, VllmSamplingParams(temperature=0, max_tokens=1024), VllmTokensPrompt


def build_verbalizer_block_lm(model_path: str, lite_quant: str, max_len: int, lanes: int, prefix_ids: list[int]):
    """The lite backend's block verifier for the verbalizer LM (draft-and-verify decoding)."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("the verbalizer needs a CUDA GPU; start without ENABLE_VERBALIZATION=1 on a machine "
                           "without one")
    from app.lite.gpu.block import BlockLM

    return BlockLM(model_path, lite_quant, max_len, lanes, prefix_ids)


def shutdown_engine(engine) -> None:
    shutdown = getattr(engine, "shutdown", None)
    if shutdown is not None:
        shutdown()
    else:
        engine.shutdown_background_loop()
