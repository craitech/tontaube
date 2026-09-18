import asyncio

import httpx
import pytest

import app.main as app_main
from app.main import create_app, server_bind_address
from app.ui_server import (
    UI_DIR,
    UI_ROUTES,
    available_ui_voices,
    resolve_ui_voice_file,
)


def _client(app):
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    )


@pytest.mark.asyncio
async def test_predict_limits_end_to_end_concurrency(monkeypatch):
    monkeypatch.setenv("MAX_INFLIGHT_REQUESTS", "2")
    monkeypatch.setattr(app_main, "_get_engine", lambda: asyncio.sleep(0))
    monkeypatch.setattr(app_main, "_unload_engine", lambda: None)

    active = 0
    peak = 0
    two_started = asyncio.Event()
    release = asyncio.Event()

    async def fake_synthesize(_req):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        if active == 2:
            two_started.set()
        try:
            await release.wait()
            return {"ok": True}
        finally:
            active -= 1

    monkeypatch.setattr(app_main, "synthesize", fake_synthesize)
    application = create_app()

    async with application.router.lifespan_context(application):
        client = _client(application)
        requests = [
            asyncio.create_task(client.post("/predict", json={"text": f"Request {i}"}))
            for i in range(5)
        ]
        await asyncio.wait_for(two_started.wait(), timeout=1)
        await asyncio.sleep(0)
        assert peak == 2
        release.set()
        responses = await asyncio.gather(*requests)
        assert all(response.status_code == 200 for response in responses)
        await client.aclose()


def test_standalone_ui_routes_reference_packaged_files():
    assert {filename for filename, _ in UI_ROUTES.values()} == {
        "index.html",
        "styles.css",
        "app.js",
        "tontaube-logo.png",
    }
    assert all((UI_DIR / filename).is_file() for filename, _ in UI_ROUTES.values())


def test_ui_voice_catalog_uses_local_files_without_exposing_paths(tmp_path, monkeypatch):
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "Amber.mp3").write_bytes(b"mp3")
    (samples / "cedar.FLAC").write_bytes(b"flac")
    (samples / "Sofia.wav").write_bytes(b"wav")
    (tmp_path / "manifest.json").write_text(
        '{"version":1,"voices":{'
        '"Amber.mp3":{"language":"en","style":"audiobook"},'
        '"Sofia.wav":{"language":"de","style":"audiobook"}'
        '}}',
        encoding="utf-8",
    )
    (tmp_path / "notes.txt").write_text("not audio", encoding="utf-8")
    (samples / "tokens.json").write_text('[[1], [2], [3]]')
    outside = tmp_path / "outside.mp3"
    outside.write_bytes(b"mp3")
    (samples / "escape.mp3").symlink_to(outside)
    (tmp_path / "nested").mkdir()
    monkeypatch.setenv("TTS_UI_VOICE_PATH", str(tmp_path))

    assert available_ui_voices("english") == [
        {"name": "Amber", "url": "/ui/voice-files/Amber.mp3", "style": "audiobook"},
        {"name": "cedar", "url": "/ui/voice-files/cedar.FLAC"},
    ]
    assert available_ui_voices("de") == [
        {"name": "cedar", "url": "/ui/voice-files/cedar.FLAC"},
        {"name": "Sofia", "url": "/ui/voice-files/Sofia.wav", "style": "audiobook"},
    ]
    assert available_ui_voices("spanish") == [
        {"name": "cedar", "url": "/ui/voice-files/cedar.FLAC"},
    ]
    assert resolve_ui_voice_file("Sofia.wav") == (samples / "Sofia.wav")
    assert resolve_ui_voice_file("Amber.mp3") == (samples / "Amber.mp3")
    assert resolve_ui_voice_file("../Amber.mp3") is None
    assert resolve_ui_voice_file("tokens.json") is None
    assert resolve_ui_voice_file("escape.mp3") is None
    with pytest.raises(ValueError):
        available_ui_voices("unknown")


@pytest.mark.asyncio
async def test_voice_catalog_requires_key_when_api_authentication_is_enabled(monkeypatch):
    monkeypatch.setattr("app.config.REQUIRE_API_KEY", True)
    monkeypatch.setattr("app.config.TTS_API_KEY", "test-secret")
    client = _client(create_app())

    assert (await client.get("/")).status_code == 200
    assert (await client.post("/predict", json={"text": "Hello"})).status_code == 401
    await client.aclose()


@pytest.mark.asyncio
async def test_inference_api_never_serves_ui_assets():
    client = _client(create_app())

    root = await client.get("/")
    assert root.status_code == 200
    assert root.json() == {"name": "Tontaube TTS API", "docs": "/docs"}
    assert (await client.get("/ui")).status_code == 404
    assert (await client.get("/ui/assets/app.js")).status_code == 404
    await client.aclose()


def test_server_bind_address_is_configurable(monkeypatch):
    monkeypatch.setenv("TTS_HOST", "127.0.0.1")
    monkeypatch.setenv("TTS_PORT", "9090")
    assert server_bind_address() == ("127.0.0.1", 9090)


def test_server_bind_address_defaults_to_loopback(monkeypatch):
    monkeypatch.delenv("TTS_HOST", raising=False)
    monkeypatch.delenv("TTS_PORT", raising=False)
    monkeypatch.delenv("PORT", raising=False)
    assert server_bind_address() == ("127.0.0.1", 8080)


def test_api_cli_runs_preflight_before_starting_uvicorn(monkeypatch):
    calls = []
    monkeypatch.setattr("app.preflight.main", lambda: calls.append("preflight"))
    monkeypatch.setattr(app_main, "server_bind_address", lambda: ("127.0.0.1", 8080))
    monkeypatch.setattr(
        "uvicorn.run",
        lambda application, *, host, port: calls.append(
            (application, host, port)
        ),
    )

    app_main.main()

    assert calls == ["preflight", (app_main.app, "127.0.0.1", 8080)]


@pytest.mark.asyncio
async def test_cors_is_disabled_by_default_and_scoped_when_configured(monkeypatch):
    origin = "http://127.0.0.1:3000"

    monkeypatch.setattr("app.config.CORS_ALLOWED_ORIGINS", ())
    default_client = _client(create_app())
    default_response = await default_client.options(
        "/predict",
        headers={"origin": origin, "access-control-request-method": "POST"},
    )
    assert "access-control-allow-origin" not in default_response.headers
    await default_client.aclose()

    monkeypatch.setattr("app.config.CORS_ALLOWED_ORIGINS", (origin,))
    cors_client = _client(create_app())
    allowed = await cors_client.options(
        "/predict",
        headers={
            "origin": origin,
            "access-control-request-method": "POST",
            "access-control-request-headers": "content-type,x-api-key",
        },
    )
    rejected = await cors_client.options(
        "/predict",
        headers={"origin": "https://untrusted.example", "access-control-request-method": "POST"},
    )

    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == origin
    assert "access-control-allow-origin" not in rejected.headers
    await cors_client.aclose()


@pytest.mark.asyncio
async def test_authenticated_cors_preflight_does_not_require_api_key(monkeypatch):
    origin = "http://127.0.0.1:3000"
    monkeypatch.setattr("app.config.CORS_ALLOWED_ORIGINS", (origin,))
    monkeypatch.setattr("app.config.REQUIRE_API_KEY", True)
    monkeypatch.setattr("app.config.TTS_API_KEY", "test-secret")
    client = _client(create_app())

    preflight = await client.options(
        "/predict",
        headers={
            "origin": origin,
            "access-control-request-method": "POST",
            "access-control-request-headers": "content-type,x-api-key",
        },
    )
    unauthorized = await client.post("/predict", json={"text": "Hello"})

    assert preflight.status_code == 200
    assert preflight.headers["access-control-allow-origin"] == origin
    assert unauthorized.status_code == 401
    await client.aclose()
