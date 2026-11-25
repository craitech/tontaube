from __future__ import annotations

from types import SimpleNamespace

import pytest

from app import cli


def test_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    cli.main([])
    output = capsys.readouterr().out
    assert "{serve,ui,verbalize,preflight,quantize}" in output


def test_serve_routes_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str | None, int | None]] = []
    monkeypatch.setitem(
        __import__("sys").modules,
        "app.main",
        SimpleNamespace(main=lambda **kwargs: calls.append((kwargs["host"], kwargs["port"]))),
    )

    cli.main(["serve", "--host", "0.0.0.0", "--port", "9000"])

    assert calls == [("0.0.0.0", 9000)]


def test_serve_rate_sets_the_flash_sample_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLASH_SAMPLE_RATE", "48000")
    monkeypatch.setitem(__import__("sys").modules, "app.main", SimpleNamespace(main=lambda **kwargs: None))

    cli.main(["serve", "--rate", "24000"])

    assert __import__("os").environ["FLASH_SAMPLE_RATE"] == "24000"


def test_ui_routes_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int]] = []
    monkeypatch.setitem(
        __import__("sys").modules,
        "app.ui_server",
        SimpleNamespace(serve=lambda host, port: calls.append((host, port))),
    )

    cli.main(["ui", "--host", "0.0.0.0", "--port", "4000"])

    assert calls == [("0.0.0.0", 4000)]


def test_verbalize_routes_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    from app import torch_variant, verbalizer_server

    calls = []
    monkeypatch.setattr(torch_variant, "check_torch_variant", lambda argv: None)
    monkeypatch.setattr(verbalizer_server, "serve", lambda host, port: calls.append(("serve", host, port)))
    monkeypatch.setattr(verbalizer_server, "verbalize_file", lambda path: calls.append(("file", path)))

    cli.main(["verbalize", "report.txt"])
    cli.main(["verbalize", "--serve", "--port", "9000"])

    assert calls == [("file", "report.txt"), ("serve", None, 9000)]


def test_preflight_routes_command(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []
    monkeypatch.setitem(
        __import__("sys").modules,
        "app.preflight",
        SimpleNamespace(main=lambda: calls.append(True)),
    )

    cli.main(["preflight"])

    assert calls == [True]


def test_lite_is_the_default_backend():
    import os
    import subprocess
    import sys

    code = "import app.config as c, app.model_sources as m; print(c.TTS_BACKEND, m.DEFAULT_MODEL_REPO_ID)"
    env = {k: v for k, v in os.environ.items() if k != "TTS_BACKEND"}
    run = lambda env: subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,  # noqa: E731
                                     text=True, check=True).stdout.split()
    assert run(env) == ["lite", "TontaubeAI/TontaubeV1.1-int4"]
    assert run({**env, "TTS_BACKEND": "vllm"}) == ["vllm", "TontaubeAI/TontaubeV1.1"]
