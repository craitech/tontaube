import json
from pathlib import Path

import pytest

from app.model_sources import RuntimeModelPaths
from app.preflight import required_model_files, validate_codebook_stack


def test_vibevoice_assets_are_optional():
    codebooks = tuple(f"/models/cb{i}" for i in range(4))
    enabled_paths = RuntimeModelPaths(
        dualcodec="/cache/dualcodec",
        w2vbert="/cache/w2vbert",
        vibevoice="/cache/vibevoice",
    )
    disabled_paths = RuntimeModelPaths(
        dualcodec="/cache/dualcodec",
        w2vbert="/cache/w2vbert",
        vibevoice=None,
    )

    enabled = required_model_files(codebooks, enabled_paths)
    disabled = required_model_files(codebooks, disabled_paths)

    assert Path("/cache/vibevoice/config.json") in enabled
    assert Path("/cache/vibevoice/model.safetensors") in enabled
    assert not any("vibevoice" in str(path) for path in disabled)
    assert Path("/cache/dualcodec/dualcodec_12hz_16384_4096.safetensors") in disabled


def test_codebook_stack_metadata_must_match_directory_order(tmp_path):
    paths = []
    for index in range(4):
        model_dir = tmp_path / f"cb{index}"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(
            json.dumps({"target_codebook": index, "n_codebooks": 4}),
            encoding="utf-8",
        )
        paths.append(str(model_dir))

    validate_codebook_stack(tuple(paths))
    (tmp_path / "cb2" / "config.json").write_text(
        json.dumps({"target_codebook": 3, "n_codebooks": 4}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ordered by target_codebook"):
        validate_codebook_stack(tuple(paths))
