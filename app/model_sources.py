"""Resolve Tontaube model files from Hugging Face or explicit local paths."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

from app.config import CODEBOOK_MODEL_DIRS
from app.logging_utils import get_logger
from app.vibevoice_assets import extract_acoustic_tokenizer


logger = get_logger("model_sources")


DEFAULT_MODEL_REPO_ID = "TontaubeAI/TontaubeV1"
DEFAULT_MODEL_REVISION = "v1.0.0"
DEFAULT_VERBALIZER_REPO_ID = "TontaubeAI/TontaubeV1-Verbalizer"
DEFAULT_VERBALIZER_REVISION = "v1.0.0"
DEFAULT_DUALCODEC_REPO_ID = "amphion/dualcodec"
DEFAULT_DUALCODEC_REVISION = "a4243540cfb149e38c82dc80dfa5c83d5e0af2a9"
DEFAULT_W2VBERT_REPO_ID = "facebook/w2v-bert-2.0"
DEFAULT_W2VBERT_REVISION = "da985ba0987f70aaeb84a80f2851cfac8c697a7b"
DEFAULT_VIBEVOICE_REPO_ID = "microsoft/VibeVoice-1.5B"
DEFAULT_VIBEVOICE_REVISION = "c00898d257e6b46004e3e2866a47534085fb685a"

MODEL_REPOSITORY_FILES = (
    "LICENSE",
    "LICENSE-APACHE-2.0",
    "COVERED_FILES.md",
    "THIRD_PARTY_NOTICES.md",
    "README.md",
)

DUALCODEC_FILES = (
    "dualcodec_12hz_16384_4096.safetensors",
    "w2vbert2_mean_var_stats_emilia.pt",
)
W2VBERT_FILES = (
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
)
VIBEVOICE_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
)


@dataclass(frozen=True)
class RuntimeModelPaths:
    dualcodec: str
    w2vbert: str
    vibevoice: str | None


def read_codebook_metadata(model_dir: str) -> tuple[int, int]:
    """Read the target row and codebook count from the standard HF config."""
    config_path = Path(model_dir) / "config.json"
    with config_path.open() as handle:
        config = json.load(handle)
    try:
        return int(config["target_codebook"]), int(config["n_codebooks"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{config_path} must define integer target_codebook and n_codebooks"
        ) from exc


def _expanded(path: str) -> str:
    return str(Path(path).expanduser().resolve())


@lru_cache(maxsize=8)
def _download_snapshot(
    repo_id: str,
    revision: str,
    allow_patterns: tuple[str, ...],
) -> str:
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=list(allow_patterns),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not resolve {repo_id}@{revision} from Hugging Face. "
            "Check network access, authentication, and the requested revision, "
            "or configure local model paths."
        ) from exc


def resolve_codebook_model_paths(
    model_dirs: tuple[str, ...] = CODEBOOK_MODEL_DIRS,
) -> tuple[str, ...]:
    """Resolve each codebook, downloading only unoverridden HF subfolders."""
    local_root = os.getenv("MODEL_PATH")
    resolved: list[str | None] = []
    remote_dirs: list[str] = []

    for model_dir in model_dirs:
        override = os.getenv(f"{model_dir.upper()}_MODEL_PATH")
        if override:
            resolved.append(_expanded(override))
        elif local_root:
            resolved.append(_expanded(str(Path(local_root) / model_dir)))
        else:
            resolved.append(None)
            remote_dirs.append(model_dir)

    if remote_dirs:
        repo_id = os.getenv("TTS_MODEL_REPO_ID", DEFAULT_MODEL_REPO_ID)
        revision = os.getenv("TTS_MODEL_REVISION", DEFAULT_MODEL_REVISION)
        snapshot = Path(
            _download_snapshot(
                repo_id,
                revision,
                MODEL_REPOSITORY_FILES
                + tuple(f"{model_dir}/**" for model_dir in remote_dirs),
            )
        )
        resolved = [
            str(snapshot / model_dir) if path is None else path
            for model_dir, path in zip(model_dirs, resolved)
        ]

    return tuple(path for path in resolved if path is not None)


def resolve_verbalizer_model_path(explicit_path: str | None = None) -> str:
    """Resolve the optional verbalizer from a local path or its own HF repo."""
    local_path = explicit_path or os.getenv("VERBALIZATION_MODEL_PATH")
    if local_path:
        return _expanded(local_path)

    repo_id = os.getenv("VERBALIZATION_MODEL_REPO_ID", DEFAULT_VERBALIZER_REPO_ID)
    revision = os.getenv(
        "VERBALIZATION_MODEL_REVISION",
        DEFAULT_VERBALIZER_REVISION,
    )
    return _download_snapshot(repo_id, revision, ("*",))


def _local_runtime_path(variable: str) -> str | None:
    override = os.getenv(variable)
    if override:
        return _expanded(override)
    return None


def _tontaube_cache_dir() -> Path:
    configured = os.getenv("TONTAUBE_CACHE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    cache_home = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"))
    return cache_home.expanduser().resolve() / "tontaube"


def _prepared_vibevoice_path(snapshot_path: str) -> str:
    source = Path(snapshot_path)
    snapshot_id = re.sub(r"[^A-Za-z0-9._-]", "-", source.name)
    target = _tontaube_cache_dir() / "vibevoice-acoustic-tokenizer" / snapshot_id
    required = (target / "config.json", target / "model.safetensors")
    if all(path.is_file() for path in required):
        return str(target)
    if target.exists():
        raise RuntimeError(
            f"Incomplete VibeVoice cache at {target}; remove that directory and retry"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{snapshot_id}-", dir=target.parent))
    try:
        logger.info("Preparing VibeVoice acoustic model in %s", target)
        extract_acoustic_tokenizer(source, temporary)
        try:
            temporary.rename(target)
        except OSError:
            if not all(path.is_file() for path in required):
                raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    logger.info("VibeVoice acoustic model prepared")
    return str(target)


def resolve_runtime_model_paths(include_vibevoice: bool = True) -> RuntimeModelPaths:
    """Resolve always-on codec assets and the optional VibeVoice model."""
    dualcodec = _local_runtime_path("DUALCODEC_MODEL_PATH")
    if dualcodec is None:
        dualcodec = _download_snapshot(
            os.getenv("DUALCODEC_MODEL_REPO_ID", DEFAULT_DUALCODEC_REPO_ID),
            os.getenv("DUALCODEC_MODEL_REVISION", DEFAULT_DUALCODEC_REVISION),
            DUALCODEC_FILES,
        )

    w2vbert = _local_runtime_path("W2VBERT_MODEL_PATH")
    if w2vbert is None:
        w2vbert = _download_snapshot(
            os.getenv("W2VBERT_MODEL_REPO_ID", DEFAULT_W2VBERT_REPO_ID),
            os.getenv("W2VBERT_MODEL_REVISION", DEFAULT_W2VBERT_REVISION),
            W2VBERT_FILES,
        )

    vibevoice = None
    if include_vibevoice:
        vibevoice = _local_runtime_path("VIBEVOICE_MODEL_PATH")
        if vibevoice is None:
            snapshot = _download_snapshot(
                os.getenv("VIBEVOICE_MODEL_REPO_ID", DEFAULT_VIBEVOICE_REPO_ID),
                os.getenv("VIBEVOICE_MODEL_REVISION", DEFAULT_VIBEVOICE_REVISION),
                VIBEVOICE_FILES,
            )
            vibevoice = _prepared_vibevoice_path(snapshot)

    return RuntimeModelPaths(
        dualcodec=_expanded(dualcodec),
        w2vbert=_expanded(w2vbert),
        vibevoice=_expanded(vibevoice) if vibevoice else None,
    )
