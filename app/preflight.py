"""Fail-fast validation for the model bundle."""

from __future__ import annotations

from importlib.metadata import entry_points
from pathlib import Path

from app.config import DUALCODEC_SEMANTIC_NORMALIZATION_FILE, TTS_MODEL, ServerConfig
from app.model_sources import (
    RuntimeModelPaths,
    read_codebook_metadata,
    resolve_codebook_model_paths,
    resolve_runtime_model_paths,
    resolve_verbalizer_model_path,
)


def require_vllm_plugin() -> None:
    plugins = entry_points(group="vllm.general_plugins")
    if not any(plugin.name == "tontaube" for plugin in plugins):
        raise SystemExit(
            "Tontaube's vLLM plugin is not installed. Run `uv sync --frozen "
            "--python 3.12`, or follow the README's locked pip installation."
        )


def required_model_files(
    codebook_paths: tuple[str, ...],
    runtime_paths: RuntimeModelPaths,
    verbalizer_path: str | None = None,
) -> list[Path]:
    required: list[Path] = []

    for model_path in codebook_paths:
        model_dir = Path(model_path)
        required.extend(
            [
                model_dir / "config.json",
                model_dir / "model.safetensors",
                model_dir / "tokenizer.json",
            ]
        )

    required.extend(
        [
            Path(runtime_paths.dualcodec) / "dualcodec_12hz_16384_4096.safetensors",
            Path(runtime_paths.dualcodec) / DUALCODEC_SEMANTIC_NORMALIZATION_FILE,
            Path(runtime_paths.w2vbert) / "config.json",
            Path(runtime_paths.w2vbert) / "model.safetensors",
            Path(runtime_paths.w2vbert) / "preprocessor_config.json",
        ]
    )
    if runtime_paths.vibevoice:
        required.extend(
            [
                Path(runtime_paths.vibevoice) / "config.json",
                Path(runtime_paths.vibevoice) / "model.safetensors",
            ]
        )
    if verbalizer_path:
        verbalizer_dir = Path(verbalizer_path)
        required.extend(
            [
                verbalizer_dir / "config.json",
                verbalizer_dir / "model.safetensors",
                verbalizer_dir / "tokenizer.json",
            ]
        )

    return required


def validate_codebook_stack(codebook_paths: tuple[str, ...]) -> None:
    """Validate the target order and shared codebook count before loading models."""
    metadata = tuple(read_codebook_metadata(path) for path in codebook_paths)
    targets = tuple(target for target, _ in metadata)
    expected = tuple(range(len(codebook_paths)))
    declared_counts = {count for _, count in metadata}
    if targets != expected or declared_counts != {len(codebook_paths)}:
        raise ValueError(
            "codebook configs must be ordered by target_codebook and all declare "
            f"n_codebooks={len(codebook_paths)}; found {metadata}"
        )


def validate_lite_quant(config: ServerConfig) -> None:
    """A lite plan must parse; a bundle directory must hold all codebook files."""
    bundle = Path(config.lite_quant).expanduser()
    if bundle.is_dir():
        missing = [
            str(bundle / f"cb{cb}.safetensors")
            for cb in range(len(config.codebook_model_dirs))
            if not (bundle / f"cb{cb}.safetensors").is_file()
        ]
        if missing:
            raise ValueError(f"lite bundle {bundle} is missing {', '.join(missing)}")
        return
    from app.lite.plans import parse_codebook_plans, parse_spec

    if config.lite_quant != "auto":  # "auto" resolves after the models are downloaded (app/lite)
        parse_codebook_plans(config.lite_quant, len(config.codebook_model_dirs))
    parse_spec(config.lite_vibevoice_quant)
    if config.enable_verbalization:
        parse_codebook_plans(config.lite_verbalizer_quant, 1)


def flash_main() -> None:
    from app.flash import preflight

    try:
        summary = preflight()
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Model preflight failed: {exc}") from exc
    print(f"Model preflight passed: {summary}", flush=True)


def main() -> None:
    if TTS_MODEL == "flash":
        flash_main()
        return
    config = ServerConfig()
    if config.backend == "lite":
        try:
            validate_lite_quant(config)
        except ValueError as exc:
            raise SystemExit(f"Model preflight failed: {exc}") from exc
    else:
        require_vllm_plugin()
    try:
        codebook_paths = resolve_codebook_model_paths(config.codebook_model_dirs)
        runtime_paths = resolve_runtime_model_paths(
            include_vibevoice=config.enable_vibevoice
        )
        verbalizer_path = (
            resolve_verbalizer_model_path(config.verbalization_model_path)
            if config.enable_verbalization
            else None
        )
    except RuntimeError as exc:
        raise SystemExit(f"Model preflight failed: {exc}") from exc

    missing = [
        path
        for path in required_model_files(
            codebook_paths,
            runtime_paths,
            verbalizer_path,
        )
        if not path.is_file()
    ]
    if missing:
        lines = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(f"Model preflight failed; missing required files:\n{lines}")
    try:
        validate_codebook_stack(codebook_paths)
    except ValueError as exc:
        raise SystemExit(f"Model preflight failed: {exc}") from exc
    print(
        "Model preflight passed: "
        + ", ".join(codebook_paths)
        + "; runtime models: "
        + ", ".join(
            path
            for path in (
                runtime_paths.dualcodec,
                runtime_paths.w2vbert,
                runtime_paths.vibevoice,
            )
            if path
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
