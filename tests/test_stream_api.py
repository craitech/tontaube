"""POST /stream and /ws/stream formats (default pcm16; mp3, opus) and the `verbalizer` field (rules / contextual /
off, the legacy use_verbalization) for TontaubeV1.1 and Flash, with fake engines."""
import base64
import json
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import app.main as app_main
from app.config import VERBALIZER_CONTEXTUAL, VERBALIZER_MODES, VERBALIZER_OFF, VERBALIZER_RULES
from app.main import TTSRequest, TTSRequestOptionError, _verbalizer_mode
from app.stream_writers import pcm16
from tests.test_flash_server import FakeFlash, flash_engine, needs_opus  # noqa: F401
from tests.test_openai_api import _call


def _events(response):
    return [json.loads(line[6:]) for line in response.text.split("\n\n") if line.startswith("data: ")]


def _payload(response) -> bytes:
    events = _events(response)
    assert events[-1] == {"done": True}
    assert [e["index"] for e in events[:-1]] == list(range(len(events) - 1))
    return b"".join(base64.b64decode(e["audio_b64"]) for e in events[:-1])


class FakeVerbalizer:
    def __init__(self):
        self.calls = []

    async def verbalize(self, text):
        self.calls.append(text)
        return text.replace("5", "five")

    async def verbalize_continuation(self, context, text):
        self.calls.append((context, text))
        return text.replace("5", "five")


class FakeV11:
    """TontaubeV1.1's streaming interface (24 kHz pieces, the keyword arguments recorded)."""

    can_stream = True
    sample_rate = 24000

    def __init__(self, verbalizer=None):
        self.verbalizer, self.calls = verbalizer, []
        self.config = SimpleNamespace(verbalize_prefix_context=True)

    async def generate_streaming(self, text, **kwargs):
        self.calls.append({"text": text, **kwargs})
        pre = kwargs.get("text_preprocessor")
        self.spoken = [await pre(t) for t in ([text] if isinstance(text, str) else text)] if pre else text
        for _ in range(3):
            yield np.full(480, 0.1, np.float32)


@pytest.fixture
def rules(monkeypatch):
    """The rule verbalizer replaced by a stand-in (the C++ front end is built on first use)."""
    from app import verbalize_rules

    seen = []
    monkeypatch.setattr(verbalize_rules, "front_end", lambda: None)
    monkeypatch.setattr(verbalize_rules, "verbalize", lambda text: (seen.append(text), text.replace("5", "RULE5"))[1])
    return seen


# ---------------------------------------------------------------- the stream formats
@pytest.mark.asyncio
async def test_stream_default_format_is_pcm16_for_the_base_model_and_flash(monkeypatch, flash_engine):
    engine = FakeV11()
    for body in ({"text": "Hi."}, {"text": "Hi.", "format": "pcm16"}):
        raw = _payload(await _call(monkeypatch, engine, "/stream", body, model="v1.1"))
        assert np.frombuffer(raw, "<i2").shape == (3 * 480,)
    fake = FakeFlash(frames=4)
    raw = _payload(await _call(monkeypatch, flash_engine(fake), "/stream", {"text": "Hi.", "seed": 2}))
    assert np.array_equal(np.frombuffer(raw, "<i2"), pcm16(fake.speak("Hi.", seed=2)))


@pytest.mark.asyncio
async def test_stream_mp3_is_selectable(monkeypatch, flash_engine):
    for engine, model in ((FakeV11(), "v1.1"), (flash_engine(FakeFlash(frames=6)), "flash")):
        response = await _call(monkeypatch, engine, "/stream", {"text": "Hi.", "format": "mp3"}, model=model)
        data = _payload(response)
        assert response.status_code == 200 and len(data) > 0
        assert data[:3] == b"ID3" or data[0] == 0xFF               # an MP3 stream, not PCM


@needs_opus
@pytest.mark.asyncio
async def test_stream_opus_sends_the_framed_opus_of_ws_stream(monkeypatch, flash_engine):
    for engine, model in ((FakeV11(), "v1.1"), (flash_engine(FakeFlash(frames=6)), "flash")):
        response = await _call(monkeypatch, engine, "/stream", {"text": "Hi.", "format": "opus"}, model=model)
        events = _events(response)
        assert response.status_code == 200 and events[-1] == {"done": True} and len(events) > 1
        for event in events[:-1]:
            blob = base64.b64decode(event["audio_b64"])
            count, = struct.unpack_from("<H", blob, 0)
            offset = 2
            for _ in range(count):
                size, = struct.unpack_from("<H", blob, offset)
                offset += 2 + size
            assert offset == len(blob)                              # [count][len data]... fills the event exactly


@pytest.mark.asyncio
async def test_stream_opus_without_libopus_is_a_422(monkeypatch):
    monkeypatch.setitem(sys.modules, "app.opus_stream", None)         # the import fails as without libopus
    response = await _call(monkeypatch, FakeV11(), "/stream", {"text": "Hi.", "format": "opus"}, model="v1.1")
    assert response.status_code == 422 and response.json()["code"] == "opus_unavailable"
    assert "libopus" in response.json()["error"]


@pytest.mark.asyncio
async def test_stream_refuses_wav(monkeypatch):
    response = await _call(monkeypatch, FakeV11(), "/stream", {"text": "Hi.", "format": "wav"}, model="v1.1")
    assert response.status_code == 422 and "pcm16 (default), mp3 or opus" in response.json()["error"]


# ---------------------------------------------------------------- the verbalizer field
def _mode(flash, monkeypatch, engine=None, **fields):
    monkeypatch.setattr(app_main, "TTS_MODEL", "flash" if flash else "v1.1")
    return _verbalizer_mode(TTSRequest(text="Today is 5.", **fields), engine)


def test_verbalizer_names_are_constants_in_one_place():
    assert VERBALIZER_MODES == (VERBALIZER_RULES, VERBALIZER_CONTEXTUAL, VERBALIZER_OFF) == ("rules", "contextual", "off")
    assert TTSRequest(text="x", verbalizer=" Rules ").verbalizer == "rules"
    with pytest.raises(ValueError, match="unknown verbalizer"):
        TTSRequest(text="x", verbalizer="syntactic")


@pytest.mark.parametrize("flash", [False, True])
def test_verbalizer_defaults_and_the_legacy_flag(flash, monkeypatch, rules):
    have = SimpleNamespace(verbalizer=FakeVerbalizer())
    default = VERBALIZER_RULES if flash else VERBALIZER_OFF
    legacy_true = VERBALIZER_RULES if flash else VERBALIZER_CONTEXTUAL
    assert _mode(flash, monkeypatch, have) == default
    assert _mode(flash, monkeypatch, have, use_verbalization=False) == VERBALIZER_OFF
    assert _mode(flash, monkeypatch, have, use_verbalization=True) == legacy_true
    assert _mode(flash, monkeypatch, have, verbalizer="off") == VERBALIZER_OFF
    assert _mode(flash, monkeypatch, have, verbalizer="rules") == VERBALIZER_RULES
    # both fields: they must agree
    assert _mode(flash, monkeypatch, have, verbalizer=legacy_true, use_verbalization=True) == legacy_true
    assert _mode(flash, monkeypatch, have, verbalizer="off", use_verbalization=False) == VERBALIZER_OFF
    for named, flag in (("off", True), (legacy_true, False), ("rules" if not flash else "contextual", True)):
        with pytest.raises(TTSRequestOptionError, match="contradicts") as exc:
            _mode(flash, monkeypatch, have, verbalizer=named, use_verbalization=flag)
        assert exc.value.code == "verbalizer_conflict"


def test_verbalizer_contextual_needs_the_loaded_model_verbalizer(monkeypatch):
    have, none = SimpleNamespace(verbalizer=FakeVerbalizer()), SimpleNamespace(verbalizer=None)
    assert _mode(False, monkeypatch, have, verbalizer="contextual") == VERBALIZER_CONTEXTUAL
    for flash, engine in ((False, none), (False, None), (True, none), (True, None)):
        with pytest.raises(TTSRequestOptionError, match="ENABLE_VERBALIZATION=1"):
            _mode(flash, monkeypatch, engine, verbalizer="contextual")
    with pytest.raises(TTSRequestOptionError, match="ENABLE_VERBALIZATION=1"):
        _mode(False, monkeypatch, none, use_verbalization=True)


@pytest.mark.parametrize("flash", [False, True])
@pytest.mark.parametrize("mode", VERBALIZER_MODES)
def test_every_verbalizer_mode_is_english_only(flash, mode, monkeypatch, rules):
    have = SimpleNamespace(verbalizer=FakeVerbalizer())
    if mode == VERBALIZER_OFF:
        assert _mode(flash, monkeypatch, have, verbalizer=mode, language="german") == VERBALIZER_OFF
        return
    with pytest.raises(TTSRequestOptionError, match="only for English"):
        _mode(flash, monkeypatch, have, verbalizer=mode, language="german")
    with pytest.raises(TTSRequestOptionError, match="only for English"):
        _mode(flash, monkeypatch, have, use_verbalization=True, language="german")


def test_rules_need_flash_front_end_on_the_base_model(monkeypatch):
    from app import verbalize_rules

    def broken():
        raise RuntimeError("no C++ compiler")

    monkeypatch.setattr(verbalize_rules, "front_end", broken)
    with pytest.raises(TTSRequestOptionError, match="no C\\+\\+ compiler") as exc:
        _mode(False, monkeypatch, SimpleNamespace(verbalizer=None), verbalizer="rules")
    assert exc.value.code == "verbalizer_unavailable"
    assert _mode(True, monkeypatch, None, verbalizer="rules") == VERBALIZER_RULES       # Flash: its own, no probe


@pytest.mark.asyncio
async def test_stream_verbalizes_by_mode_through_the_text_preprocessor(monkeypatch, rules):
    verbalizer = FakeVerbalizer()
    text = ["It is 5 now.", "Then 5 more."]
    fields = {}
    for body, expect in (
        ({"verbalizer": "off"}, text),
        ({"verbalizer": "rules"}, [t.replace("5", "RULE5") for t in text]),
        ({"verbalizer": "contextual"}, [t.replace("5", "five") for t in text]),
        ({"use_verbalization": True}, [t.replace("5", "five") for t in text]),
        ({}, text),
    ):
        engine = FakeV11(verbalizer=verbalizer)
        response = await _call(monkeypatch, engine, "/stream", {"text": text, **fields, **body}, model="v1.1")
        assert response.status_code == 200, (body, response.text)
        assert (engine.calls[0]["text_preprocessor"] is None) == (expect == text), body
        assert [s.strip() for s in engine.spoken] == [s.strip() for s in expect], body


@pytest.mark.asyncio
async def test_stream_refuses_unsupported_verbalizer_requests_with_422(monkeypatch):
    engine = FakeV11()           # no model verbalizer loaded
    for body in ({"verbalizer": "contextual"}, {"use_verbalization": True}, {"verbalizer": "off", "use_verbalization": True},
                 {"verbalizer": "syntactic"}):
        response = await _call(monkeypatch, engine, "/stream", {"text": "Hi.", **body}, model="v1.1")
        assert response.status_code == 422, body


@pytest.mark.asyncio
async def test_flash_stream_and_predict_take_the_verbalizer_field(monkeypatch, flash_engine):
    for body, want in (({}, True), ({"verbalizer": "rules"}, True), ({"verbalizer": "off"}, False),
                       ({"use_verbalization": False}, False), ({"use_verbalization": True}, True)):
        fake = FakeFlash(frames=3)
        response = await _call(monkeypatch, flash_engine(fake), "/stream", {"text": "Hi.", **body})
        assert response.status_code == 200 and fake.calls[0]["verbalize"] is want, body
    for body in ({"verbalizer": "contextual"}, {"verbalizer": "off", "use_verbalization": True}):
        response = await _call(monkeypatch, flash_engine(FakeFlash(frames=3)), "/stream", {"text": "Hi.", **body})
        assert response.status_code == 422, body
    fake = FakeFlash(frames=3)
    response = await _call(monkeypatch, flash_engine(fake), "/predict", {"text": "Hi.", "verbalizer": "off"})
    assert response.status_code == 200 and fake.calls[0]["verbalize"] is False


# ---------------------------------------------------------------- /ws/stream
def _ws_run(monkeypatch, engine, body):
    from fastapi.testclient import TestClient

    from tests.test_openai_api import _app

    with TestClient(_app(monkeypatch, engine, "v1.1")) as client:
        with client.websocket_connect("/ws/stream") as ws:
            ws.send_json(body)
            messages = []
            while True:
                message = ws.receive()
                if message.get("type") == "websocket.close":
                    return messages
                messages.append(json.loads(message["text"]) if message.get("text") else message["bytes"])
                if messages[-1] == {"done": True}:
                    return messages


@needs_opus
def test_ws_stream_takes_opus_only(monkeypatch):
    for fmt in ("pcm16", "mp3"):
        messages = _ws_run(monkeypatch, FakeV11(), {"text": "Hi.", "format": fmt})
        assert messages and "outputs opus" in messages[0]["error"]


@needs_opus
def test_ws_stream_verbalizer_errors_are_messages_not_failures(monkeypatch):
    messages = _ws_run(monkeypatch, FakeV11(), {"text": "Hi.", "verbalizer": "contextual"})
    assert messages and messages[0]["code"] == "verbalizer_unavailable" and "ENABLE_VERBALIZATION=1" in messages[0]["error"]


@pytest.mark.asyncio
async def test_stream_pcm16_is_at_the_engines_rate_and_predict_refuses_pcm16(monkeypatch, flash_engine):
    from tests.test_flash_server import FakeFlash48

    fake = FakeFlash48(frames=6)
    response = await _call(monkeypatch, flash_engine(fake), "/stream", {"text": "Hi.", "seed": 3})
    assert np.array_equal(np.frombuffer(_payload(response), "<i2"), pcm16(fake.speak("Hi.", seed=3)))   # not resampled
    assert {e["sample_rate"] for e in _events(response)[:-1]} == {48000}
    response = await _call(monkeypatch, FakeV11(), "/stream", {"text": "Hi."}, model="v1.1")
    assert {e["sample_rate"] for e in _events(response)[:-1]} == {24000}
    response = await _call(monkeypatch, FakeV11(), "/stream", {"text": "Hi.", "format": "mp3"}, model="v1.1")
    assert not any("sample_rate" in e for e in _events(response))        # the codec streams carry their own
    response = await _call(monkeypatch, flash_engine(FakeFlash(frames=3)), "/predict", {"text": "Hi.", "format": "pcm16"})
    assert response.status_code == 422 and "POST /stream only" in response.json()["error"]
