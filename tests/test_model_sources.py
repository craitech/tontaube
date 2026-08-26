from pathlib import Path
import json

from app import model_sources


def test_codebook_metadata_comes_from_hf_config(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"target_codebook": 2, "n_codebooks": 6})
    )
    assert model_sources.read_codebook_metadata(str(tmp_path)) == (2, 6)


def _clear_model_env(monkeypatch):
    for name in (
        "MODEL_PATH",
        "CB0_MODEL_PATH",
        "CB1_MODEL_PATH",
        "CB2_MODEL_PATH",
        "CB3_MODEL_PATH",
        "TTS_MODEL_REPO_ID",
        "TTS_MODEL_REVISION",
        "VERBALIZATION_MODEL_PATH",
        "VERBALIZATION_MODEL_REPO_ID",
        "VERBALIZATION_MODEL_REVISION",
        "DUALCODEC_MODEL_PATH",
        "DUALCODEC_MODEL_REPO_ID",
        "DUALCODEC_MODEL_REVISION",
        "W2VBERT_MODEL_PATH",
        "W2VBERT_MODEL_REPO_ID",
        "W2VBERT_MODEL_REVISION",
        "VIBEVOICE_MODEL_PATH",
        "VIBEVOICE_MODEL_REPO_ID",
        "VIBEVOICE_MODEL_REVISION",
        "TONTAUBE_CACHE_DIR",
        "XDG_CACHE_HOME",
    ):
        monkeypatch.delenv(name, raising=False)


def test_codebooks_default_to_hf_subfolders(monkeypatch, tmp_path):
    _clear_model_env(monkeypatch)
    calls = []

    def fake_download(repo_id, revision, allow_patterns):
        calls.append((repo_id, revision, allow_patterns))
        return str(tmp_path / "snapshot")

    monkeypatch.setattr(model_sources, "_download_snapshot", fake_download)
    paths = model_sources.resolve_codebook_model_paths()

    assert paths == tuple(str(tmp_path / "snapshot" / f"cb{i}") for i in range(4))
    assert calls == [
        (
            "TontaubeAI/TontaubeV1",
            "v1.0.0",
            model_sources.MODEL_REPOSITORY_FILES
            + ("cb0/**", "cb1/**", "cb2/**", "cb3/**"),
        )
    ]


def test_codebooks_accept_individual_and_root_overrides(monkeypatch, tmp_path):
    _clear_model_env(monkeypatch)
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "models"))
    monkeypatch.setenv("CB2_MODEL_PATH", str(tmp_path / "custom-cb2"))
    monkeypatch.setattr(
        model_sources,
        "_download_snapshot",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected download")),
    )

    paths = model_sources.resolve_codebook_model_paths()

    assert paths[0] == str((tmp_path / "models" / "cb0").resolve())
    assert paths[2] == str((tmp_path / "custom-cb2").resolve())


def test_verbalizer_defaults_to_separate_hf_repo(monkeypatch, tmp_path):
    _clear_model_env(monkeypatch)
    calls = []

    def fake_download(repo_id, revision, allow_patterns):
        calls.append((repo_id, revision, allow_patterns))
        return str(tmp_path / "verbalizer-snapshot")

    monkeypatch.setattr(model_sources, "_download_snapshot", fake_download)
    assert model_sources.resolve_verbalizer_model_path() == str(
        tmp_path / "verbalizer-snapshot"
    )
    assert calls == [
        ("TontaubeAI/TontaubeV1-Verbalizer", "v1.0.0", ("*",))
    ]


def test_verbalizer_local_override_avoids_hf(monkeypatch, tmp_path):
    _clear_model_env(monkeypatch)
    local = tmp_path / "verbalizer"
    monkeypatch.setattr(
        model_sources,
        "_download_snapshot",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected download")),
    )
    assert model_sources.resolve_verbalizer_model_path(str(local)) == str(local.resolve())


def test_runtime_models_default_to_hf_cache(monkeypatch, tmp_path):
    _clear_model_env(monkeypatch)
    calls = []

    def fake_download(repo_id, revision, allow_patterns):
        calls.append((repo_id, revision, allow_patterns))
        return str(tmp_path / repo_id.replace("/", "--") / revision)

    prepared = tmp_path / "prepared-vibevoice"
    monkeypatch.setattr(model_sources, "_download_snapshot", fake_download)
    monkeypatch.setattr(
        model_sources,
        "_prepared_vibevoice_path",
        lambda _snapshot: str(prepared),
    )

    paths = model_sources.resolve_runtime_model_paths(include_vibevoice=True)

    assert paths.dualcodec.endswith(model_sources.DEFAULT_DUALCODEC_REVISION)
    assert paths.w2vbert.endswith(model_sources.DEFAULT_W2VBERT_REVISION)
    assert paths.vibevoice == str(prepared.resolve())
    assert calls == [
        (
            model_sources.DEFAULT_DUALCODEC_REPO_ID,
            model_sources.DEFAULT_DUALCODEC_REVISION,
            model_sources.DUALCODEC_FILES,
        ),
        (
            model_sources.DEFAULT_W2VBERT_REPO_ID,
            model_sources.DEFAULT_W2VBERT_REVISION,
            model_sources.W2VBERT_FILES,
        ),
        (
            model_sources.DEFAULT_VIBEVOICE_REPO_ID,
            model_sources.DEFAULT_VIBEVOICE_REVISION,
            model_sources.VIBEVOICE_FILES,
        ),
    ]


def test_disabled_vibevoice_is_not_downloaded(monkeypatch, tmp_path):
    _clear_model_env(monkeypatch)
    calls = []

    def fake_download(repo_id, revision, allow_patterns):
        calls.append(repo_id)
        return str(tmp_path / repo_id.replace("/", "--") / revision)

    monkeypatch.setattr(model_sources, "_download_snapshot", fake_download)
    paths = model_sources.resolve_runtime_model_paths(include_vibevoice=False)

    assert paths.vibevoice is None
    assert model_sources.DEFAULT_VIBEVOICE_REPO_ID not in calls


def test_runtime_local_overrides_avoid_hf(monkeypatch, tmp_path):
    _clear_model_env(monkeypatch)
    monkeypatch.setenv("DUALCODEC_MODEL_PATH", str(tmp_path / "dualcodec"))
    monkeypatch.setenv("W2VBERT_MODEL_PATH", str(tmp_path / "w2vbert"))
    monkeypatch.setenv("VIBEVOICE_MODEL_PATH", str(tmp_path / "vibevoice"))
    monkeypatch.setattr(
        model_sources,
        "_download_snapshot",
        lambda *_args: (_ for _ in ()).throw(AssertionError("unexpected download")),
    )

    paths = model_sources.resolve_runtime_model_paths(include_vibevoice=True)

    assert paths.dualcodec == str((tmp_path / "dualcodec").resolve())
    assert paths.w2vbert == str((tmp_path / "w2vbert").resolve())
    assert paths.vibevoice == str((tmp_path / "vibevoice").resolve())


def test_vibevoice_extraction_is_reused(monkeypatch, tmp_path):
    _clear_model_env(monkeypatch)
    monkeypatch.setenv("TONTAUBE_CACHE_DIR", str(tmp_path / "cache"))
    snapshot = tmp_path / "snapshots" / "abc123"
    snapshot.mkdir(parents=True)
    calls = []

    def fake_extract(_source, output):
        calls.append(output)
        (output / "config.json").write_text("{}")
        (output / "model.safetensors").write_bytes(b"weights")

    monkeypatch.setattr(model_sources, "extract_acoustic_tokenizer", fake_extract)
    first = model_sources._prepared_vibevoice_path(str(snapshot))
    second = model_sources._prepared_vibevoice_path(str(snapshot))

    assert first == second
    assert len(calls) == 1
