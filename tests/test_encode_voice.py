"""POST /encode_voice: request envelope, validation, and reuse as `voice_tokens`."""
import asyncio
import threading
from collections import OrderedDict
from unittest.mock import MagicMock

import httpx
import pytest
import torch

import app.main as app_main
from app import utils
from app.config import (
    DUALCODEC_CODEC_ID,
    DUALCODEC_HZ,
    DUALCODEC_MAX_CODEBOOKS,
    MAX_VOICE_PROMPT_TOKENS,
    MAX_VOICE_REFERENCE_B64_CHARS,
    MODEL_VERSION,
)
from app.inference import TTSEngine
from app.main import TTSRequest, create_app
from app.utils import VoicePromptError

SEMANTIC = list(range(40))
ACOUSTIC = [[100 + i] * 40 for i in range(DUALCODEC_MAX_CODEBOOKS - 1)]


class _FakeEngine:
    def __init__(self, error: Exception | None = None):
        self.calls = []
        self.error = error

    def encode_voice(self, voice_audio_b64, prompt_max_tokens=None, n_codebooks=None):
        self.calls.append((voice_audio_b64, prompt_max_tokens, n_codebooks))
        if self.error is not None:
            raise self.error
        cap = prompt_max_tokens or len(SEMANTIC)
        return [SEMANTIC[:cap], *(row[:cap] for row in ACOUSTIC[:(n_codebooks or 4) - 1])]


async def _post(monkeypatch, body, engine=None, *, ready=True):
    engine = engine or _FakeEngine()

    async def get_engine():
        return engine

    monkeypatch.setattr(app_main, "_get_engine", get_engine)
    monkeypatch.setattr(app_main, "_unload_engine", lambda: None)
    application = create_app()
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        if not ready:
            return await client.post("/encode_voice", json=body)
        async with application.router.lifespan_context(application):
            return await client.post("/encode_voice", json=body)


@pytest.mark.asyncio
async def test_returns_full_prompt_in_the_voice_tokens_format(monkeypatch):
    engine = _FakeEngine()
    response = await _post(monkeypatch, {"voice_audio_b64": "QUJD"}, engine)

    assert response.status_code == 200
    body = response.json()
    assert engine.calls == [("QUJD", None, None)]
    assert body["voice_tokens"] == [SEMANTIC, *ACOUSTIC[:3]]
    assert body["n_tokens"] == 40
    assert body["n_codebooks"] == 4
    assert body["duration_seconds"] == pytest.approx(40 / DUALCODEC_HZ)
    assert body["token_rate_hz"] == DUALCODEC_HZ
    assert body["model_version"] == MODEL_VERSION
    assert body["codec"] == DUALCODEC_CODEC_ID == "dualcodec_12hz_v1"
    # The output is accepted unchanged by /predict, /stream and /ws/stream.
    req = TTSRequest(text="Hello.", voice_tokens=body["voice_tokens"])
    assert req.voice_tokens == body["voice_tokens"]


@pytest.mark.asyncio
async def test_accepts_instances_envelope_alias_and_token_cap(monkeypatch):
    engine = _FakeEngine()
    response = await _post(
        monkeypatch,
        {"instances": [{"voice_opus_b64": "QUJD", "prompt_max_tokens": 10}]},
        engine,
    )

    assert response.status_code == 200
    assert engine.calls == [("QUJD", 10, None)]
    assert response.json()["n_tokens"] == 10
    assert all(len(row) == 10 for row in response.json()["voice_tokens"])


@pytest.mark.asyncio
@pytest.mark.parametrize("n_codebooks", [1, DUALCODEC_MAX_CODEBOOKS])
async def test_returns_the_requested_codebook_count(monkeypatch, n_codebooks):
    engine = _FakeEngine()
    response = await _post(monkeypatch, {"voice_audio_b64": "QUJD", "n_codebooks": n_codebooks}, engine)

    assert response.status_code == 200
    body = response.json()
    assert engine.calls == [("QUJD", None, n_codebooks)]
    assert body["n_codebooks"] == len(body["voice_tokens"]) == n_codebooks
    assert {len(row) for row in body["voice_tokens"]} == {body["n_tokens"]}
    # Generation routes take the first four rows.
    TTSRequest(text="Hello.", voice_tokens=body["voice_tokens"][:4])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {},
        {"voice_audio_b64": ""},
        {"voice_audio_b64": "QUJD", "prompt_max_tokens": 0},
        {"voice_audio_b64": "QUJD", "prompt_max_tokens": MAX_VOICE_PROMPT_TOKENS + 1},
        {"voice_audio_b64": "QUJD", "text": "not accepted here"},
        {"voice_audio_b64": "QUJD", "n_codebooks": 0},
        {"voice_audio_b64": "QUJD", "n_codebooks": DUALCODEC_MAX_CODEBOOKS + 1},
        {"voice_audio_b64": "QUJD", "n_codebooks": "all"},
        {"voice_audio_b64": "A" * (MAX_VOICE_REFERENCE_B64_CHARS + 1)},
        {"instances": [{"voice_audio_b64": "QUJD"}, {"voice_audio_b64": "QUJD"}]},
    ],
)
async def test_rejects_invalid_requests(monkeypatch, body):
    engine = _FakeEngine()
    response = await _post(monkeypatch, body, engine)

    assert response.status_code == 422
    assert engine.calls == []


@pytest.mark.asyncio
async def test_undecodable_audio_is_a_422_with_a_message(monkeypatch):
    engine = _FakeEngine(VoicePromptError("voice_audio_b64 is not valid base64-encoded audio"))
    response = await _post(monkeypatch, {"voice_audio_b64": "not audio"}, engine)

    assert response.status_code == 422
    assert response.json() == {"error": "voice_audio_b64 is not valid base64-encoded audio"}


@pytest.mark.asyncio
async def test_returns_503_while_the_model_loads(monkeypatch):
    engine = _FakeEngine()
    response = await _post(monkeypatch, {"voice_audio_b64": "QUJD"}, engine, ready=False)

    assert response.status_code == 503
    assert response.json() == {"error": "Model still loading"}
    assert engine.calls == []


@pytest.mark.asyncio
async def test_requires_the_api_key_like_predict(monkeypatch):
    monkeypatch.setattr("app.config.REQUIRE_API_KEY", True)
    monkeypatch.setattr("app.config.TTS_API_KEY", "test-secret")
    application = create_app()
    transport = httpx.ASGITransport(app=application)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.post("/encode_voice", json={"voice_audio_b64": "QUJD"})
    assert response.status_code == 401


def _engine_with_cache():
    engine = TTSEngine.__new__(TTSEngine)
    engine.config = MagicMock(prompt_seconds=60.0, prompt_max_tokens=300, generate_n_codebooks=4)
    engine.dc_inference = object()
    engine.prompt_semantic, engine.prompt_acoustic = [1], [[1], [1], [1]]
    engine._voice_prompt_cache = OrderedDict()
    engine._voice_prompt_cache_max = 8
    engine._voice_prompt_lock = threading.Lock()
    return engine


def test_engine_caches_each_codebook_count_separately(tmp_path, monkeypatch):
    def encode(_dc, _path, _seconds, _cap, n_codebooks):
        return list(range(50)), [[n_codebooks] * 50 for _ in range(n_codebooks - 1)]

    encode = MagicMock(side_effect=encode)
    monkeypatch.setattr(utils, "encode_voice_prompt", encode)

    def decode(*_args):
        path = tmp_path / "voice.wav"
        path.write_bytes(b"placeholder")
        return str(path)

    monkeypatch.setattr(utils, "decode_voice_audio_b64", decode)
    engine = _engine_with_cache()

    assert len(engine.encode_voice("QUJD")) == 4
    rows = engine.encode_voice("QUJD", n_codebooks=DUALCODEC_MAX_CODEBOOKS)
    assert len(rows) == DUALCODEC_MAX_CODEBOOKS and rows[1][0] == DUALCODEC_MAX_CODEBOOKS
    # Generation still gets its own 4-codebook entry, not the 8-row one.
    _semantic, acoustic = engine._resolve_voice_prompt("QUJD", None)
    assert len(acoustic) == 3 and acoustic[0][0] == 4
    assert [c.args[4] for c in encode.call_args_list] == [4, DUALCODEC_MAX_CODEBOOKS]


def test_engine_returns_the_full_prompt_and_shares_the_generation_cache(tmp_path, monkeypatch):
    full = (list(range(600)), [list(range(600)) for _ in range(3)])
    encode = MagicMock(return_value=full)
    monkeypatch.setattr(utils, "encode_voice_prompt", encode)

    def decode(*_args):  # resolve_voice_prompt deletes the decoded temporary file
        path = tmp_path / "voice.wav"
        path.write_bytes(b"placeholder")
        return str(path)

    monkeypatch.setattr(utils, "decode_voice_audio_b64", decode)
    engine = _engine_with_cache()

    rows = asyncio.run(asyncio.to_thread(engine.encode_voice, "QUJD"))
    assert rows == [full[0], *full[1]]  # not the server's 300-token default cap
    assert engine.encode_voice("QUJD", 5) == [full[0][:5], *(row[:5] for row in full[1])]
    # A later generation request with the same audio hits the cache.
    semantic, _acoustic = engine._resolve_voice_prompt("QUJD", None)
    assert semantic == full[0][:300]
    encode.assert_called_once()


def test_empty_reference_audio_is_a_voice_prompt_error(tmp_path, monkeypatch):
    path = tmp_path / "empty.wav"
    path.write_bytes(b"placeholder")
    monkeypatch.setattr(utils, "load_and_prep_audio", lambda _path: torch.zeros(1, 1, 100))
    with pytest.raises(VoicePromptError, match="contains no audio"):
        utils.encode_voice_prompt(object(), str(path), 60.0, None, 4)
