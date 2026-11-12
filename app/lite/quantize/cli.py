"""`tontaube quantize`: build a pre-quantized bundle for the lite backend.

1. Record calibration token streams: the bf16 lite engine synthesizes the
   packaged calibration texts (all seven languages) and every codebook
   call's prompt and output tokens are kept.
2. Quantize each codebook model layer by layer with GPTQ (app.lite.quantize.calibrate)
   and write <out>/cb0.safetensors ... cb3.safetensors plus bundle.json.

Needs a GPU that can hold the bf16 models (about 12 GiB); the resulting
bundle then runs on much smaller GPUs with TTS_BACKEND=lite LITE_QUANT=<out>.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

PRESETS = {
    # Near-lossless: every layer int8 except the last MLP projection, which
    # feeds the audio head and is the most quantization-sensitive matrix.
    "int8": "int8g128,layers.{last}.down=bf16,embed=int8",
    # Smallest: int4 transformer layers; head and last MLP projection int8.
    "int4": "int4g64,layers.{last}.down=int8g128,head=int8g128,embed=int8",
}
CALIBRATION_TEXTS = Path(__file__).with_name("calibration_texts.json")


def preset_plans(preset: str, model_paths: list[str]) -> list[str]:
    template = PRESETS[preset]
    plans = []
    for path in model_paths:
        n_layers = json.loads((Path(path) / "config.json").read_text())["num_hidden_layers"]
        plans.append(template.format(last=n_layers - 1))
    return plans


async def record_calls(seeds: list[int], log=print) -> tuple[list[dict], list[str]]:
    from app.config import BUNDLED_VOICE_PATH, ServerConfig
    from app.inference import TTSEngine
    from app.model_sources import resolve_codebook_model_paths
    from app.text import sanitize_text_for_tts, split_long_text

    config = ServerConfig()
    config.backend = "lite"
    config.lite_quant = "bf16"
    config.enable_vibevoice = False
    engine = TTSEngine(config)
    calls: list[dict] = []
    original = engine._generate_with_engine

    async def logged(eng, prompt_ids, sampling_params, priority=0):
        out = await original(eng, prompt_ids, sampling_params, priority=priority)
        calls.append({"item": current[0], "cb": engine.engines.index(eng),
                      "prompt_ids": list(prompt_ids), "output_ids": list(out)})
        return out

    engine._generate_with_engine = logged
    current = [""]
    await engine.load()
    items = json.loads(CALIBRATION_TEXTS.read_text())["items"]
    t0 = time.time()
    for i, item in enumerate(items):
        for seed in seeds:
            current[0] = f"{item['id']}_s{seed}"
            await engine.generate(
                text=split_long_text(sanitize_text_for_tts(item["text"])),
                seed=seed, voice_path=str(BUNDLED_VOICE_PATH / item["voice"]),
                language=item["language"], tag=item["tag"], vibevoice_postprocess=False,
            )
        log(f"calibration text {i + 1}/{len(items)} ({time.time() - t0:.0f}s)")
    model_paths = list(resolve_codebook_model_paths(config.codebook_model_dirs))
    engine.unload()
    return calls, model_paths


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="tontaube quantize", description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="bundle directory to create")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="int8")
    ap.add_argument("--plan", help="custom plan(s) instead of a preset (see app/lite/plans.py)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--n-seqs", type=int, default=64, help="calibration sequences per codebook")
    ap.add_argument("--calls", help="reuse recorded calls (pickle) instead of recording")
    args = ap.parse_args(argv)

    os.environ["TTS_BACKEND"] = "lite"
    from app.model_sources import FULL_MODEL_REPO_ID

    os.environ.setdefault("TTS_MODEL_REPO_ID", FULL_MODEL_REPO_ID)  # quantizing needs the full-precision weights
    import pickle

    import torch

    from app.lite.quantize.calibrate import calibrate_codebook, select_sequences
    from app.lite.weights import save_linears, weights_fingerprint
    from app.lite.gpu.model import LiteCodebookModel
    from app.lite.plans import parse_plan
    from app.lite.weights import LazyWeights
    from app.model_sources import resolve_codebook_model_paths

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    if args.calls:
        calls = pickle.load(open(args.calls, "rb"))
        model_paths = list(resolve_codebook_model_paths())
    else:
        calls, model_paths = asyncio.run(record_calls(args.seeds))
        with open(out / "calibration_calls.pkl", "wb") as fh:
            pickle.dump(calls, fh)
    plans = args.plan.split(";") if args.plan else preset_plans(args.preset, model_paths)
    if len(plans) == 1:
        plans = plans * len(model_paths)
    for cb, path in enumerate(model_paths):
        t0 = time.time()
        model = LiteCodebookModel(path, cb, parse_plan("bf16"), max_len=3300, max_batch=1, kv_tokens=64,
                                  weights=LazyWeights(Path(path) / "model.safetensors"))
        seqs = select_sequences(calls, cb, args.n_seqs)
        print(f"cb{cb}: {plans[cb]} ({len(seqs)} calibration sequences)", flush=True)
        linears = calibrate_codebook(model, parse_plan(plans[cb]), seqs, "gptq", log=lambda m: None)
        save_linears(out / f"cb{cb}.safetensors", linears,
                     {"plan": plans[cb], "method": "gptq", "calibration": f"{len(seqs)} sequences",
                      "source": weights_fingerprint(Path(path) / "model.safetensors")})
        print(f"cb{cb}: done in {time.time() - t0:.0f}s", flush=True)
        del model, linears
        torch.cuda.empty_cache()
    (out / "bundle.json").write_text(json.dumps({
        "format": "tontaube-lite-bundle-v1",
        "preset": None if args.plan else args.preset,
        "plans": plans,
        "method": "gptq",
        "source_models": model_paths,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))
    print(f"Bundle written to {out}. Serve it with TTS_BACKEND=lite LITE_QUANT={out}")


if __name__ == "__main__":
    main()
