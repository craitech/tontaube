"""The OpenAI-compatible routes (POST /v1/audio/speech, GET /v1/models) with fake engines."""
import importlib
import io
import shutil
import sys

import httpx
import numpy as np
import pytest

import app.main as app_main
from app.stream_writers import Mp3StreamWriter, pcm16
from tests.test_flash_server import SR, FakeFlash, FakeFlash48, flash_engine  # noqa: F401

try:
    import opuslib  # noqa: F401
except Exception:                                       # no Opus library (libopus)
    opuslib = None


def _real(name):
    """The real module behind conftest's mock (soundfile, pydub)."""
    mock = sys.modules.pop(name)
    try:
        return importlib.import_module(name)
    finally:
        sys.modules[name] = mock


class FakeV11:
    """TontaubeV1.1's streaming interface: torch-free pieces at 24 kHz, the keyword arguments recorded."""

    can_stream = True

    def __init__(self):
        self.calls = []

    async def generate_streaming(self, text, **kwargs):
        self.calls.append({"text": text, **kwargs})
        for _ in range(3):
            yield np.full(1000, 0.1, np.float32)


def _app(monkeypatch, engine, model="flash"):
    async def get_engine():
        return engine

    monkeypatch.setattr(app_main, "TTS_MODEL", model)
    monkeypatch.setattr(app_main, "_get_engine", get_engine)
    monkeypatch.setattr(app_main, "_unload_engine", lambda: None)
    return app_main.create_app()


async def _call(monkeypatch, engine, path, body=None, headers=None, model="flash"):
    application = _app(monkeypatch, engine, model)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="http://testserver") as client:
        async with application.router.lifespan_context(application):
            if body is None:
                return await client.get(path, headers=headers)
            return await client.post(path, json=body, headers=headers)


def _speech(**fields):
    return {"model": "tts-1", "input": "Hello there.", "voice": "alloy", **fields}


def _spoken(fake):
    return fake.speak("x", seed=fake.calls[0]["seed"])


@pytest.mark.asyncio
async def test_mp3_streams_the_whole_utterance(monkeypatch, flash_engine):
    fake = FakeFlash48(frames=10, redraws={3, 8})
    response = await _call(monkeypatch, flash_engine(fake), "/v1/audio/speech", _speech())
    assert response.status_code == 200 and response.headers["content-type"] == "audio/mpeg"
    one_shot = Mp3StreamWriter(48000, bitrate=64)
    assert response.content == one_shot.write_chunk(_spoken(fake)) + one_shot.close()   # gap-free, every redraw in


@pytest.mark.asyncio
async def test_wav_has_the_engine_rate_and_samples(monkeypatch, flash_engine):
    sf = _real("soundfile")
    monkeypatch.setattr(app_main, "sf", sf)
    fake = FakeFlash48(frames=6, redraws={3})
    response = await _call(monkeypatch, flash_engine(fake), "/v1/audio/speech", _speech(response_format="wav"))
    assert response.status_code == 200 and response.headers["content-type"] == "audio/wav"
    info = sf.info(io.BytesIO(response.content))
    assert info.samplerate == 48000 and info.channels == 1 and info.subtype == "PCM_16"
    audio, _ = sf.read(io.BytesIO(response.content), dtype="int16")
    assert np.array_equal(audio, pcm16(_spoken(fake)))


@pytest.mark.asyncio
async def test_pcm_is_24_khz_from_a_48_khz_engine(monkeypatch, flash_engine):
    fake = FakeFlash48(frames=6)
    response = await _call(monkeypatch, flash_engine(fake), "/v1/audio/speech", _speech(response_format="pcm"))
    assert response.status_code == 200 and response.headers["content-type"] == "audio/pcm"
    assert len(response.content) == 2 * (6 * 6400 // 2)                # 16-bit samples, half the 48 kHz count


@pytest.mark.asyncio
async def test_pcm_from_a_24_khz_engine_is_its_audio(monkeypatch, flash_engine):
    fake = FakeFlash(frames=5)
    response = await _call(monkeypatch, flash_engine(fake), "/v1/audio/speech", _speech(response_format="pcm"))
    assert np.array_equal(np.frombuffer(response.content, "<i2"), pcm16(_spoken(fake)))


def test_pcm_low_pass_keeps_speech_band_and_removes_what_would_alias():
    t = np.arange(48000) / 48000
    for tone, expected in ((1000, True), (15000, False)):
        writer = app_main._PcmWriter(48000)
        audio = 0.5 * np.sin(2 * np.pi * tone * t).astype(np.float32)
        out = b"".join(writer.write_chunk(audio[i:i + 6400]) for i in range(0, len(audio), 6400)) + writer.close()
        pcm = np.frombuffer(out, "<i2") / 32767
        assert len(pcm) == 24000
        middle = pcm[200:-200]
        if expected:   # the 1 kHz tone, on the 24 kHz grid
            assert np.abs(middle - 0.5 * np.sin(2 * np.pi * tone * t[::2][200:-200])).max() < 2e-3
        else:          # 15 kHz would fold to 9 kHz
            assert np.sqrt(np.mean(middle ** 2)) < 1e-3


@pytest.mark.skipif(opuslib is None, reason="no Opus library (libopus)")
@pytest.mark.asyncio
async def test_opus_is_ogg_with_the_exact_length(monkeypatch, flash_engine):
    sf = _real("soundfile")
    fake = FakeFlash48(frames=7, tail_silence=77)
    response = await _call(monkeypatch, flash_engine(fake), "/v1/audio/speech", _speech(response_format="opus"))
    assert response.status_code == 200 and response.headers["content-type"] == "audio/ogg"
    assert response.content[:4] == b"OggS"
    audio, rate = sf.read(io.BytesIO(response.content))
    assert rate == 48000 and len(audio) == 7 * 6400 + 77


@pytest.mark.asyncio
async def test_flac_through_ffmpeg_is_lossless(monkeypatch, flash_engine):
    if shutil.which("ffmpeg") is None:
        pytest.skip("no FFmpeg")
    sf = _real("soundfile")
    monkeypatch.setattr(app_main, "sf", sf)
    monkeypatch.setattr(app_main, "AudioSegment", _real("pydub").AudioSegment)
    monkeypatch.setattr(app_main, "_ffmpeg_found", lambda: True)
    fake = FakeFlash48(frames=4)
    response = await _call(monkeypatch, flash_engine(fake), "/v1/audio/speech", _speech(response_format="flac"))
    assert response.status_code == 200 and response.headers["content-type"] == "audio/flac"
    audio, rate = sf.read(io.BytesIO(response.content), dtype="int16")
    assert rate == 48000 and np.array_equal(audio, pcm16(_spoken(fake)))


@pytest.mark.asyncio
@pytest.mark.parametrize("fields, message", [
    ({"speed": 1.5}, "speed is not supported"),
    ({"stream_format": "sse"}, "stream_format 'sse' is not supported; use 'audio'"),
    ({"response_format": "aac"}, "response_format 'aac' needs FFmpeg"),
    ({"response_format": "ogg"}, "response_format: Input should be"),
    ({"input": "  "}, "input: input must not be empty"),
    ({"input": "x" * 40_001}, "input: input must contain at most 40000 characters"),
    ({"model": None}, "model: Input should be a valid string"),
])
async def test_bad_requests_are_400s_in_openai_shape(monkeypatch, flash_engine, fields, message):
    monkeypatch.setattr(app_main, "_ffmpeg_found", lambda: False)
    fake = FakeFlash()
    response = await _call(monkeypatch, flash_engine(fake), "/v1/audio/speech", _speech(**fields))
    assert response.status_code == 400
    error = response.json()["error"]
    assert error["message"].startswith(message) and error["type"] == "invalid_request_error" and error["code"] is None
    assert fake.calls == []


@pytest.mark.asyncio
async def test_speed_1_and_instructions_are_accepted(monkeypatch, flash_engine):
    fake = FakeFlash(frames=2)
    response = await _call(monkeypatch, flash_engine(fake), "/v1/audio/speech",
                           _speech(speed=1.0, instructions="Cheerful.", stream_format="audio", response_format="pcm"))
    assert response.status_code == 200 and len(fake.calls) == 1


@pytest.mark.asyncio
async def test_bearer_and_x_api_key_both_work(monkeypatch, flash_engine):
    monkeypatch.setattr("app.config.REQUIRE_API_KEY", True)
    monkeypatch.setattr("app.config.TTS_API_KEY", "secret")
    engine = flash_engine(FakeFlash(frames=2))
    for headers, status in (({"authorization": "Bearer secret"}, 200), ({"x-api-key": "secret"}, 200),
                            ({"authorization": "Bearer wrong"}, 401), ({"x-api-key": "wrong"}, 401), ({}, 401)):
        for path, body in (("/v1/models", None), ("/v1/audio/speech", _speech(response_format="pcm"))):
            response = await _call(monkeypatch, engine, path, body, headers=headers)
            assert response.status_code == status, (headers, path)
            if status == 401:
                assert response.json() == {"error": {"message": "unauthorized", "type": "invalid_request_error",
                                                     "code": None}}
    assert (await _call(monkeypatch, engine, "/controls", headers={"authorization": "Bearer secret"})).status_code \
        == 401                                                  # the existing routes still want x-api-key


@pytest.mark.asyncio
async def test_v11_voice_maps_to_a_bundled_voice(monkeypatch):
    monkeypatch.delenv("VOICE_PATH", raising=False)
    fake = FakeV11()
    for voice in ("marcus", "CLARA", "alloy"):
        response = await _call(monkeypatch, fake, "/v1/audio/speech", _speech(voice=voice, response_format="pcm"),
                               model="v1.1")
        assert response.status_code == 200 and len(response.content) == 2 * 3000      # 24 kHz, passed through
    paths = [call["voice_path"] for call in fake.calls]
    assert paths[0].endswith("/voices/samples/Marcus.mp3") and paths[1].endswith("/voices/samples/Clara.wav")
    assert paths[2] is None                                     # no such voice: the default one


@pytest.mark.asyncio
@pytest.mark.parametrize("model, name", [("flash", "TontaubeV1.1 Flash"), ("v1.1", "TontaubeV1.1")])
async def test_models_lists_the_served_model(monkeypatch, flash_engine, model, name):
    engine = flash_engine(FakeFlash()) if model == "flash" else FakeV11()
    response = await _call(monkeypatch, engine, "/v1/models", model=model)
    assert response.json() == {"object": "list", "data": [{"id": name, "object": "model", "owned_by": "tontaube"}]}
