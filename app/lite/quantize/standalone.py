"""Build a standalone lite release from full codebook models and a bundle quantized from them.

    python -m app.lite.quantize.standalone --models DIR --bundle DIR --out DIR

DIR/cb0 ... cb3 are the full models. The output holds cb0 ... cb3 with every file except model.safetensors copied,
a reduced model.safetensors with only what the bundle leaves out (norms, the text-embedding rows the input can
reach, the audio-embedding rows), and the bundle in int4/, fingerprinted against the reduced files. The lite
backend serves it as a model root (MODEL_PATH, or the downloaded repository); LITE_QUANT=auto picks up int4/.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch

from app.config import CODEBOOK_MODEL_DIRS
from app.lite.config import ModelConfig
from app.lite.weights import TRIMMED_AUDIO_WEIGHT, TRIMMED_TEXT_ROWS, TRIMMED_TEXT_WEIGHT, weights_fingerprint
from app.model.text_vocab import text_token_rows
from app.utils import load_tokenizer

BUNDLE_DIR = "int4"


def _in_bundle(key: str) -> bool:
    return key.endswith("_proj.weight") or key.startswith("audio_head.")


def export_standalone(models: Path, bundle: Path, out: Path) -> None:
    from safetensors import safe_open
    from safetensors.torch import save_file

    (out / BUNDLE_DIR).mkdir(parents=True, exist_ok=True)
    for cb, name in enumerate(CODEBOOK_MODEL_DIRS):
        src, dst = models / name, out / name
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.is_file() and f.name != "model.safetensors":
                shutil.copy2(f, dst / f.name)
        start = ModelConfig.from_dir(src, cb).gpu_vocab_start
        rows = [r for r in text_token_rows(load_tokenizer(str(src))) if r < start]
        tensors = {}
        with safe_open(str(src / "model.safetensors"), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key == "model.embed_tokens.weight":
                    emb = f.get_tensor(key)
                    tensors[TRIMMED_TEXT_ROWS] = torch.tensor(rows, dtype=torch.long)
                    tensors[TRIMMED_TEXT_WEIGHT] = emb[rows].contiguous()
                    tensors[TRIMMED_AUDIO_WEIGHT] = emb[start:].contiguous()
                elif not _in_bundle(key):
                    tensors[key] = f.get_tensor(key)
        save_file(tensors, str(dst / "model.safetensors"))
        with safe_open(str(bundle / f"cb{cb}.safetensors"), framework="pt", device="cpu") as f:
            meta = dict(f.metadata() or {})
            linears = {key: f.get_tensor(key) for key in f.keys()}
        meta["source"] = weights_fingerprint(dst / "model.safetensors")
        save_file(linears, str(out / BUNDLE_DIR / f"cb{cb}.safetensors"), metadata=meta)
    info = json.loads((bundle / "bundle.json").read_text())
    info["source_models"] = list(CODEBOOK_MODEL_DIRS)
    info.pop("created", None)
    (out / BUNDLE_DIR / "bundle.json").write_text(json.dumps(info, indent=1) + "\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    export_standalone(args.models, args.bundle, args.out)
    print(f"Standalone release written to {args.out}")


if __name__ == "__main__":
    main()
