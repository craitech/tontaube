"""Prepare the VibeVoice acoustic tokenizer used by Tontaube."""

from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path


PREFIX = "model.acoustic_tokenizer."


def extract_acoustic_tokenizer(source_dir: Path, output_dir: Path) -> None:
    """Extract the acoustic tokenizer from a downloaded VibeVoice snapshot."""
    config_path = source_dir / "config.json"
    index_path = source_dir / "model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise RuntimeError(
            "VibeVoice snapshot must contain config.json and "
            "model.safetensors.index.json"
        )

    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    tokenizer_config = dict(model_config["acoustic_tokenizer_config"])
    tokenizer_config["architectures"] = ["VibeVoiceAcousticTokenizerModel"]
    dtype = model_config.get("torch_dtype", model_config.get("dtype"))
    if dtype is not None:
        tokenizer_config["torch_dtype"] = dtype
    transformers_version = model_config.get("transformers_version")
    if transformers_version is not None:
        tokenizer_config["transformers_version"] = transformers_version

    weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    selected = {
        name.removeprefix(PREFIX): shard
        for name, shard in weight_map.items()
        if name.startswith(PREFIX)
    }
    if not selected:
        raise RuntimeError("No VibeVoice acoustic-tokenizer tensors found")

    missing_shards = sorted(
        shard for shard in set(selected.values()) if not (source_dir / shard).is_file()
    )
    if missing_shards:
        raise RuntimeError("Missing VibeVoice model shards: " + ", ".join(missing_shards))

    from safetensors import safe_open
    from safetensors.torch import save_file

    tensors = {}
    with ExitStack() as stack:
        readers = {
            shard: stack.enter_context(
                safe_open(str(source_dir / shard), framework="pt", device="cpu")
            )
            for shard in sorted(set(selected.values()))
        }
        for output_name, shard in selected.items():
            tensors[output_name] = readers[shard].get_tensor(PREFIX + output_name)

        output_dir.mkdir(parents=True, exist_ok=True)
        save_file(
            tensors,
            output_dir / "model.safetensors",
            metadata={"format": "pt"},
        )

    (output_dir / "config.json").write_text(
        json.dumps(tokenizer_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
