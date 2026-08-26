from __future__ import annotations

from types import SimpleNamespace

import pytest

from app import cli


def test_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    cli.main([])
    output = capsys.readouterr().out
    assert "{serve,ui,preflight}" in output


def test_serve_routes_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str | None, int | None]] = []
    monkeypatch.setitem(
        __import__("sys").modules,
        "app.main",
        SimpleNamespace(main=lambda **kwargs: calls.append((kwargs["host"], kwargs["port"]))),
    )

    cli.main(["serve", "--host", "0.0.0.0", "--port", "9000"])

    assert calls == [("0.0.0.0", 9000)]


def test_ui_routes_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int]] = []
    monkeypatch.setitem(
        __import__("sys").modules,
        "app.ui_server",
        SimpleNamespace(serve=lambda host, port: calls.append((host, port))),
    )

    cli.main(["ui", "--host", "0.0.0.0", "--port", "4000"])

    assert calls == [("0.0.0.0", 4000)]


def test_preflight_routes_command(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[bool] = []
    monkeypatch.setitem(
        __import__("sys").modules,
        "app.preflight",
        SimpleNamespace(main=lambda: calls.append(True)),
    )

    cli.main(["preflight"])

    assert calls == [True]
