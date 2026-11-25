"""TontaubeV1.1 Flash in the server (TTS_MODEL=flash): FlashEngine (app/flash.py) behind /predict, /stream and
/ws/stream, with fake engines that speak like flash/gpu/engine.Engine (3,200-sample frames; redraws that replace the
last frames before the end)."""
import asyncio
import base64
import json
import os
import struct
import time

import httpx
import numpy as np
import pytest

import app.main as app_main
from app import flash
from app.flash import FlashEngine
from flash.common.append import HOLD, HOP, AppendOnly
try:
    from app.opus_stream import OpusStreamEncoder
except Exception:                                       # no Opus library (libopus): Windows, macOS without it
    OpusStreamEncoder = None
needs_opus = pytest.mark.skipif(OpusStreamEncoder is None, reason="no Opus library (libopus)")
from app.priority_gate import get_gate

SR = 24000
NO_BUDGET = lambda: -1e18  # noqa: E731


class FakeFlash:
    """Speaks like Engine.stream: one frame per piece; after frame i in `redraws` a replacement of the last `back`
    frames (start before the end), made only where played() or boundary() has not reached it (the engine's rule)."""

    sampler = ("midpoint", 8)
    sample_rate = SR

    def __init__(self, frames=12, redraws=(), back=3, tail_silence=0, fail=False, delay=0.0, pause=None):
        self.frames, self.redraws, self.back, self.tail_silence, self.fail = frames, set(redraws), back, tail_silence, fail
        self.delay, self.pause = delay, pause or {}
        self.temp, self.calls, self.replacements, self.closed = 0.7, [], [], 0
        self.stats = {"redraw": {"spots": len(self.redraws), "attempts": len(self.redraws)}}

    def set_sampler(self, solver, passes, temp):
        self.temp = temp

    def stream(self, text, seed=0, verbalize=True, played=None, boundary=None):
        self.calls.append({"text": text, "seed": seed, "verbalize": verbalize, "temp": self.temp})
        rng = np.random.default_rng(seed)
        hop = HOP * self.sample_rate // SR
        frame = lambda: rng.uniform(-0.5, 0.5, hop).astype(np.float32)  # noqa: E731
        end = 0
        try:
            for i in range(self.frames):
                if self.fail and i == 2:
                    raise RuntimeError("engine failure")
                time.sleep(self.delay + self.pause.get(i, 0.0))
                yield end, frame()
                end += hop
                if i in self.redraws:
                    where = played if played is not None else boundary
                    start, now = end - self.back * hop, where() if where is not None else 0
                    if now <= start:
                        self.replacements.append((start, now))
                        yield start, np.concatenate([frame() for _ in range(self.back)])
            if self.tail_silence:
                yield end, np.zeros(self.tail_silence, np.float32)
        finally:
            self.closed += 1

    def speak(self, text, seed=0, verbalize=True, played=NO_BUDGET):
        out = np.zeros(0, np.float32)
        for start, audio in self.stream(text, seed, verbalize, played):
            out = np.concatenate([out[:start], audio])
        return out


class FakeFlash48(FakeFlash):
    """The CPU engine's default: 48 kHz, 6,400-sample frames."""

    sample_rate = 48000


async def _collect(engine, text="Hello.", **kwargs):
    return [piece async for piece in engine.generate_streaming(text, **kwargs)]


@pytest.fixture
def flash_engine():
    """FlashEngine(fakes) for one test; its worker threads stop at the end."""
    made = []

    def make(*fakes, serialize=False):
        made.append(FlashEngine(list(fakes), serialize=serialize))
        return made[-1]

    yield make
    for engine in made:
        engine.shutdown()


# --------------------------------------------------------------------------------------------- the engine adapter
@pytest.mark.asyncio
async def test_stream_is_append_only_and_equals_speak(flash_engine):
    fake = FakeFlash(frames=14, redraws={6, 10, 13})
    pieces = await _collect(flash_engine(fake), seed=5)
    assert len(fake.replacements) == 3                       # every redraw happened, the last one in the final frame
    assert np.array_equal(np.concatenate(pieces), fake.speak("Hello.", seed=5))
    assert len(pieces[0]) == HOP                              # the first frame goes out at once
    assert all(now <= start for start, now in fake.replacements)


@pytest.mark.asyncio
async def test_a_48_khz_engine_streams_whole_frames_at_its_rate(flash_engine):
    fake = FakeFlash48(frames=14, redraws={6, 10, 13})
    engine = flash_engine(fake)
    assert engine.sample_rate == 48000
    pieces = await _collect(engine, seed=5)
    assert len(fake.replacements) == 3 and all(now <= start for start, now in fake.replacements)
    assert np.array_equal(np.concatenate(pieces), fake.speak("Hello.", seed=5))
    assert len(pieces[0]) == 2 * HOP                          # frame 0 at 48 kHz goes out at once


def test_append_only_counts_frames_at_the_engine_rate():
    clock = Clock()
    out = AppendOnly(clock=clock, frame_s=0.005, sample_rate=48000)
    sent = []
    for k in range(9):
        clock.t += 0.005
        sent.append(len(out.put(k * 2 * HOP, np.full(2 * HOP, k, np.float32))) / (2 * HOP))
    assert sent == [1, 0, 0, 0, 0, 1, 1, 1, 1] and out.boundary() == 5 * 2 * HOP


@pytest.mark.asyncio
@pytest.mark.parametrize("back, taken", [(4, True), (5, False)])
async def test_the_boundary_is_the_audio_sent(flash_engine, back, taken):
    """Audio goes out when it is 4 frames behind the newest sample: a redraw 4 frames back is still taken, one 5 frames
    back reaches audio already sent and the engine refuses it."""
    fake = FakeFlash(frames=16, redraws={8, 12}, back=back)
    pieces = await _collect(flash_engine(fake), seed=1)
    assert len(fake.replacements) == (2 if taken else 0)
    assert all(now <= start for start, now in fake.replacements)
    assert np.array_equal(np.concatenate(pieces), fake.speak("Hello.", seed=1, played=None if taken else lambda: 1e18))


@pytest.mark.asyncio
async def test_a_stalled_engine_gets_the_safety_net(flash_engine):
    """The engine stops for 1 s after 8 fast frames: the client would run out of audio, so frames go out before they
    are final, and a redraw that would reach them is refused (the stream stays seamless)."""
    fake = FakeFlash(frames=14, redraws={9}, pause={8: 1.0})
    pieces = await _collect(flash_engine(fake), seed=2)
    assert fake.replacements == []
    assert np.array_equal(np.concatenate(pieces), fake.speak("Hello.", seed=2, played=lambda: 1e18))


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _feed(out, clock, frames, p, first=0):
    """Frames first.. of an engine taking p seconds per frame; the frames each put() releases."""
    sent = []
    for k in range(first, first + frames):
        clock.t += p
        sent.append(len(out.put(k * HOP, np.full(HOP, k, np.float32))) / HOP)
    return sent


def test_a_fast_engine_releases_frame_0_at_once_then_final_audio():
    clock = Clock()
    out = AppendOnly(clock=clock, frame_s=0.005)
    assert _feed(out, clock, 9, 0.005) == [1, 0, 0, 0, 0, 1, 1, 1, 1]     # frame 0, then 4 frames behind the newest
    assert out.boundary() == 5 * HOP and out.early == 0
    assert len(out.put(5 * HOP, np.full(HOP, -1, np.float32))) == 0 and out.reached == 0   # a redraw at the boundary
    late = np.full(2 * HOP, -2, np.float32)                                # one before it: what was sent stays
    assert len(out.put(4 * HOP, late)) == 0 and out.reached == 1
    assert out.n == 6 * HOP and out.flush().tolist() == [-2.0] * HOP


def test_a_slower_engine_waits_for_its_lead_and_then_needs_no_safety_net():
    clock = Clock()
    out = AppendOnly(clock=clock)                                          # 3.5x real time, speed measured
    sent = _feed(out, clock, 40, 0.038)
    assert sent[:7] == [0, 0, 0, 0, 0, 2, 1]                              # first audio 5 frames later (190 ms): room for a redraw
    assert sum(sent) == 36 and out.early == 0                             # afterwards only final audio
    clock0 = Clock()
    assert _feed(AppendOnly(clock=clock0, redraw=0), clock0, 6, 0.038) == [0, 0, 0, 1, 0, 1]   # without that room: 3 frames
    clock.t += 1.0                                                         # a 1 s stall: the client still has audio
    assert _feed(out, clock, 1, 0.038, first=40) == [1] and out.early == 0
    clock.t += 4.0                                                         # a 4 s stall: it would run out
    assert _feed(out, clock, 1, 0.038, first=41) == [5] and out.early == 4 * HOP   # just in time: 4 frames early


def test_the_start_cut_shortens_frame_0():
    """2,400 samples cut before the first sound: only 800 samples of frame 0 are final at once, too little to start
    with; the first release waits for the first frame that is final (4 frames later on a fast engine)."""
    clock = Clock()
    out = AppendOnly(clock=clock, start_final=HOP - 2400, frame_s=0.005)
    sent = []
    for k in range(6):
        clock.t += 0.005
        sent.append(len(out.put(k * HOP, np.full(HOP, k, np.float32))))
    assert sent == [0, 0, 0, 0, HOP, HOP]


def test_a_whole_file_releases_nothing_before_the_end():
    whole = AppendOnly(hold=None)
    assert [len(whole.put(k * HOP, np.ones(HOP, np.float32))) for k in range(3)] == [0, 0, 0]
    assert len(whole.flush()) == 3 * HOP


@pytest.mark.asyncio
async def test_predict_gets_every_redraw_and_the_request_settings(flash_engine):
    fake = FakeFlash(frames=10, redraws={2, 9})
    engine = flash_engine(fake)
    audio = await engine.generate(["First part.", "Second part."], seed=3, temperature=0.4, verbalize=False)
    await engine.generate("Again.")
    assert np.array_equal(audio, fake.speak("x", seed=3))
    first, second = fake.calls[0], fake.calls[1]
    assert first == {"text": "First part.\nSecond part.", "seed": 3, "verbalize": False, "temp": 0.4}
    assert second["temp"] == flash.DEFAULT_TEMPERATURE and second["verbalize"] is True
    assert isinstance(second["seed"], int)                    # no seed: a random one


@pytest.mark.asyncio
async def test_requests_run_on_separate_engines_at_once(flash_engine):
    fakes = [FakeFlash(frames=30, delay=0.002), FakeFlash(frames=30, delay=0.002)]
    engine = flash_engine(*fakes, serialize=True)
    outs = await asyncio.gather(*(engine.generate("Hello.", seed=s) for s in (1, 2, 3)))
    streams = await asyncio.gather(*(_collect(engine, seed=s) for s in (4, 5)))
    for seed, audio in zip((1, 2, 3, 4, 5), outs + [np.concatenate(s) for s in streams]):
        assert np.array_equal(audio, FakeFlash(frames=30).speak("Hello.", seed=seed))
    assert all(f.calls for f in fakes)


@pytest.mark.asyncio
async def test_a_stream_left_early_frees_its_engine(flash_engine):
    fake = FakeFlash(frames=400, delay=0.001)
    engine = flash_engine(fake)
    async for _piece in engine.generate_streaming("Long text.", seed=0):
        break
    audio = await asyncio.wait_for(engine.generate("Next.", seed=2), timeout=10)
    assert fake.closed == 2 and len(audio) == 400 * HOP


@pytest.mark.asyncio
async def test_an_engine_error_fails_only_its_request(flash_engine):
    fake = FakeFlash(fail=True)
    engine = flash_engine(fake)
    with pytest.raises(RuntimeError, match="engine failure"):
        await engine.generate("Hello.")
    fake.fail = False
    assert len(await engine.generate("Hello.")) == 12 * HOP


@pytest.mark.asyncio
async def test_low_priority_waits_while_a_high_priority_request_runs(flash_engine):
    engine = flash_engine(FakeFlash())
    async with get_gate().high():
        low = asyncio.ensure_future(engine.generate("Hello.", priority="low"))
        await asyncio.sleep(0.1)
        assert not low.done()
    assert len(await asyncio.wait_for(low, timeout=10)) == 12 * HOP


@pytest.mark.asyncio
async def test_get_engine_builds_the_engines_once(monkeypatch):
    built = []
    monkeypatch.setenv("FLASH_ENGINES", "3")
    monkeypatch.setenv("FLASH_MODEL_PATH", "/models/flash")
    monkeypatch.setenv("FLASH_SAMPLE_RATE", "24000")
    monkeypatch.setattr(flash, "build_engines", lambda dev, weights, n, rate: built.append((dev, weights, n, rate)) or
                        [FakeFlash() for _ in range(n)])
    monkeypatch.setattr(flash, "_engine", None)
    engine = await flash.get_engine()
    try:
        assert built == [(flash.device(), "/models/flash", 3, 24000)] and len(engine.engines) == 3
        assert await flash.get_engine() is engine
    finally:
        flash.unload_engine()


# ------------------------------------------------------------------------------------------------------ the routes
class _Writer:
    chunks = []

    def __init__(self, *args, **kwargs):
        _Writer.chunks, _Writer.rate = [], args[0]

    def write_chunk(self, chunk):
        _Writer.chunks.append(chunk)
        return b"mp3"

    def close(self):
        return b""


def _app(monkeypatch, engine):
    async def get_engine():
        return engine

    monkeypatch.setattr(app_main, "TTS_MODEL", "flash")
    monkeypatch.setattr(app_main, "_get_engine", get_engine)
    monkeypatch.setattr(app_main, "_unload_engine", lambda: None)
    monkeypatch.setattr("app.stream_writers.Mp3StreamWriter", _Writer)
    return app_main.create_app()


async def _post(monkeypatch, engine, path, body, headers=None, ready=True):
    application = _app(monkeypatch, engine)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="http://testserver") as client:
        if not ready:
            return await client.post(path, json=body, headers=headers)
        async with application.router.lifespan_context(application):
            return await client.post(path, json=body, headers=headers)


@pytest.mark.asyncio
async def test_predict_returns_the_engine_audio(monkeypatch, flash_engine):
    fake = FakeFlash(frames=10, redraws={4}, tail_silence=SR)
    encoded = []
    monkeypatch.setattr(app_main, "_encode_audio", lambda audio, fmt, bitrate, sample_rate: encoded.append(
        (audio, fmt, sample_rate)) or (b"RIFF", "audio/wav"))
    response = await _post(monkeypatch, flash_engine(fake), "/predict",
                           {"text": "Hi.", "seed": 7, "trim_silence_padding_ms": None, "mossformer2_postprocess": False})
    assert response.status_code == 200
    body = response.json()
    assert body["sample_rate"] == SR and body["mime_type"] == "audio/wav" and base64.b64decode(body["audio_b64"]) == b"RIFF"
    audio, fmt, rate = encoded[0]
    assert fmt == "wav" and rate == SR and np.array_equal(audio, fake.speak("Hi.", seed=7))


@pytest.mark.asyncio
async def test_predict_trims_the_edges_like_v11(monkeypatch, flash_engine):
    fake = FakeFlash(frames=4, tail_silence=SR)
    encoded = []
    monkeypatch.setattr(app_main, "_encode_audio", lambda audio, *a, **k: encoded.append(audio) or (b"", "audio/wav"))
    response = await _post(monkeypatch, flash_engine(fake), "/predict",
                           {"text": "Hi.", "seed": 1, "mossformer2_postprocess": False})
    assert response.status_code == 200
    assert len(encoded[0]) == 13200 + SR // 4                # the last 20 ms window with sound ends at 13,200; + 250 ms


@pytest.mark.asyncio
async def test_predict_at_48_khz(monkeypatch, flash_engine):
    fake = FakeFlash48(frames=4, tail_silence=48000)
    encoded = []
    monkeypatch.setattr(app_main, "_encode_audio", lambda audio, fmt, bitrate, sample_rate: encoded.append(
        (audio, sample_rate)) or (b"RIFF", "audio/wav"))
    response = await _post(monkeypatch, flash_engine(fake), "/predict", {"text": "Hi.", "seed": 1, "mossformer2_postprocess": False})
    assert response.status_code == 200 and response.json()["sample_rate"] == 48000
    audio, rate = encoded[0]
    assert rate == 48000 and len(audio) == 26400 + 48000 // 4   # the 48 kHz trim: the last window with sound, + 250 ms


@pytest.mark.asyncio
async def test_sse_writer_uses_the_engine_rate(monkeypatch, flash_engine):
    response = await _post(monkeypatch, flash_engine(FakeFlash48(frames=3)), "/stream", {"text": "Hi."})
    assert response.status_code == 200 and _Writer.rate == 48000


def test_numpy_edge_trim_matches_the_tensor_trim():
    torch = pytest.importorskip("torch")          # Linux only (Windows and macOS install Flash's CPU set)
    rng = np.random.default_rng(0)
    audio = np.concatenate([np.zeros(5000), rng.uniform(-0.3, 0.3, 30000), 1e-4 * rng.standard_normal(9000),
                            0.01 * np.ones(250)]).astype(np.float32)
    for padding in (0, 100, 250):
        assert app_main._trim_bounds(audio, padding_ms=padding) == app_main._trim_bounds(
            torch.from_numpy(audio)[None], padding_ms=padding)


@pytest.mark.asyncio
@pytest.mark.parametrize("field", [
    {"voice_path": "Marcus.mp3"}, {"voice_tokens": [[1, 2]]}, {"tag": "conversational"}, {"controls": {"rate": 1}},
    {"include_timestamps": True}, {"prefix_text": "Before."}, {"max_new_tokens": 100}, {"top_p": 0.9},
    {"streaming_initial_seconds": 1.2}, {"vibevoice_postprocess": True}, {"language": "german"},
    {"mossformer2_postprocess": True},
])
async def test_v11_fields_are_refused(monkeypatch, flash_engine, field):
    fake = FakeFlash()
    for path in ("/predict", "/stream"):
        response = await _post(monkeypatch, flash_engine(fake), path, {"text": "Hi.", **field})
        assert response.status_code == 422, (path, response.text)
    assert fake.calls == []


@pytest.mark.asyncio
async def test_empty_speech_is_a_422(monkeypatch, flash_engine):
    response = await _post(monkeypatch, flash_engine(FakeFlash(frames=0)), "/predict", {"text": "..."})
    assert response.status_code == 422 and "nothing to speak" in response.json()["error"]


@pytest.mark.asyncio
async def test_sse_stream_sends_the_engine_audio_in_order(monkeypatch, flash_engine):
    fake = FakeFlash(frames=20, redraws={5, 12, 19})
    response = await _post(monkeypatch, flash_engine(fake), "/stream", {"text": "Hi.", "seed": 4, "temperature": 0.5})
    assert response.status_code == 200
    events = [json.loads(line[6:]) for line in response.text.split("\n\n") if line.startswith("data: ")]
    assert events[-1] == {"done": True}
    assert [e["index"] for e in events[:-1]] == list(range(len(events) - 1))
    assert np.array_equal(np.concatenate(_Writer.chunks), fake.speak("Hi.", seed=4))
    assert fake.calls[0]["temp"] == 0.5


def _opus_frames(blobs):
    frames = []
    for blob in filter(None, blobs):
        count, pos = struct.unpack_from("<H", blob)[0], 2
        for _ in range(count):
            size = struct.unpack_from("<H", blob, pos)[0]
            frames.append(blob[pos + 2:pos + 2 + size])
            pos += 2 + size
    return frames


@needs_opus
def test_ws_stream_sends_gap_free_opus(monkeypatch, flash_engine):
    from fastapi.testclient import TestClient

    fake = FakeFlash(frames=15, redraws={6}, tail_silence=77)   # a partial 20 ms frame at the end
    with TestClient(_app(monkeypatch, flash_engine(fake))) as client:
        with client.websocket_connect("/ws/stream") as ws:
            ws.send_json({"text": "Hi.", "seed": 2})
            blobs = []
            while True:
                message = ws.receive()
                if message.get("bytes") is not None:
                    blobs.append(message["bytes"])
                else:
                    assert json.loads(message["text"]) == {"done": True}
                    break
    expected = fake.speak("Hi.", seed=2)
    one_shot = OpusStreamEncoder(SR, bitrate=64000)
    reference = _opus_frames([one_shot.encode(expected), one_shot.flush()])
    assert _opus_frames(blobs) == reference                  # the same frames as the whole utterance at once
    assert len(reference) == -(-len(expected) // 480)


@needs_opus
def test_opus_encoder_carries_partial_frames():
    audio = np.random.default_rng(0).uniform(-0.5, 0.5, 5 * HOP + 77).astype(np.float32)
    pieces = OpusStreamEncoder(SR)
    blobs = [pieces.encode(audio[i:i + HOP]) for i in range(0, len(audio), HOP)] + [pieces.flush()]
    whole = OpusStreamEncoder(SR)
    assert _opus_frames(blobs) == _opus_frames([whole.encode(audio), whole.flush()])
    assert pieces.flush() == b""


@pytest.mark.asyncio
async def test_api_key_and_readiness(monkeypatch, flash_engine):
    monkeypatch.setattr("app.config.REQUIRE_API_KEY", True)
    monkeypatch.setattr("app.config.TTS_API_KEY", "secret")
    engine = flash_engine(FakeFlash())
    assert (await _post(monkeypatch, engine, "/predict", {"text": "Hi."})).status_code == 401
    assert (await _post(monkeypatch, engine, "/stream", {"text": "Hi."})).status_code == 401
    application = _app(monkeypatch, engine)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="http://testserver") as client:
        assert (await client.get("/readyz")).status_code == 503
        assert (await client.post("/predict", json={"text": "Hi."}, headers={"x-api-key": "secret"})).status_code == 503
        async with application.router.lifespan_context(application):
            assert (await client.get("/readyz")).status_code == 200
            ok = await client.post("/predict", json={"text": "Hi.", "mossformer2_postprocess": False},
                                   headers={"x-api-key": "secret"})
            assert ok.status_code == 200, ok.text
            catalog = (await client.get("/controls", headers={"x-api-key": "secret"})).json()
            assert catalog["controls"] == [] and catalog["model"] == "TontaubeV1.1 Flash"
            assert "text" in catalog["fields"] and "seed" in catalog["fields"]      # what the UI may send
            assert not {"language", "tag", "voice_audio_b64", "controls"} & set(catalog["fields"])
            voice = await client.post("/encode_voice", json={"voice_audio_b64": "QUJD"}, headers={"x-api-key": "secret"})
            assert voice.status_code == 422


def test_ws_needs_the_api_key(monkeypatch, flash_engine):
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    monkeypatch.setattr("app.config.REQUIRE_API_KEY", True)
    monkeypatch.setattr("app.config.TTS_API_KEY", "secret")
    with TestClient(_app(monkeypatch, flash_engine(FakeFlash()))) as client:
        with pytest.raises(WebSocketDisconnect) as closed:
            with client.websocket_connect("/ws/stream") as ws:
                ws.receive_json()
    assert closed.value.code == 4401


# ------------------------------------------------------------------------------------------- selection, preflight
def test_serve_selects_the_model_and_device(monkeypatch):
    import sys
    from types import SimpleNamespace

    from app import cli

    monkeypatch.setenv("TTS_MODEL", "v1.1")                  # restored after the test
    monkeypatch.setenv("LITE_DEVICE", "")
    monkeypatch.setitem(sys.modules, "app.main", SimpleNamespace(main=lambda **kwargs: None))
    cli.main(["serve", "--model", "flash", "--device", "cpu"])
    assert os.environ["TTS_MODEL"] == "flash" and os.environ["LITE_DEVICE"] == "cpu"
    cli.main(["serve", "--device", "gpu"])
    assert os.environ["LITE_DEVICE"] == "cuda"


def test_preflight_checks_flash(monkeypatch, capsys):
    from app import preflight

    monkeypatch.setattr(preflight, "TTS_MODEL", "flash")
    monkeypatch.setattr(flash, "preflight", lambda: "TontaubeV1.1 Flash int4 on the GPU: /w/a")
    preflight.main()
    assert "Model preflight passed: TontaubeV1.1 Flash int4" in capsys.readouterr().out

    def missing():
        raise FileNotFoundError("weight files not found: /w/a")

    monkeypatch.setattr(flash, "preflight", missing)
    with pytest.raises(SystemExit, match="weight files not found"):
        preflight.main()


def test_preflight_needs_a_compiler(monkeypatch, tmp_path):
    monkeypatch.setenv("TONTAUBE_CACHE_DIR", str(tmp_path))     # nothing cached
    monkeypatch.setenv("FLASH_BUILD_FROM_SOURCE", "1")          # no prebuilt library
    monkeypatch.setattr("shutil.which", lambda name: None)
    with pytest.raises(RuntimeError, match="C\\+\\+17 compiler"):
        flash.preflight()


@pytest.mark.asyncio
async def test_shutdown_stops_a_running_request(flash_engine):
    fake = FakeFlash(frames=100000, delay=0.001)
    engine = flash_engine(fake)
    stream = engine.generate_streaming("Long text.", seed=0)
    await stream.__anext__()
    started = time.perf_counter()
    await asyncio.to_thread(engine.shutdown)
    assert time.perf_counter() - started < 2 and fake.closed == 1
    await stream.aclose()


@pytest.mark.asyncio
async def test_a_busy_engine_is_a_503_before_any_audio(monkeypatch, flash_engine):
    engine = flash_engine(FakeFlash())
    engine.shutdown()                                         # refuses every request (EngineBusy)
    for path in ("/predict", "/stream"):
        response = await _post(monkeypatch, engine, path, {"text": "Hi."})
        assert response.status_code == 503 and response.json()["code"] == "engine_busy", (path, response.text)


@needs_opus
def test_a_busy_engine_closes_the_websocket_with_try_again_later(monkeypatch, flash_engine):
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect

    engine = flash_engine(FakeFlash())
    engine.shutdown()
    with TestClient(_app(monkeypatch, engine)) as client:
        with client.websocket_connect("/ws/stream") as ws:
            ws.send_json({"text": "Hi."})
            assert ws.receive_json()["code"] == "engine_busy"
            with pytest.raises(WebSocketDisconnect) as closed:
                ws.receive_json()
    assert closed.value.code == 1013


class FixedTemperatureFlash(FakeFlash):
    """An engine whose temperature is set when it is built (no set_sampler)."""

    set_sampler = sampler = property(lambda self: (_ for _ in ()).throw(AttributeError("no set_sampler")))


@pytest.mark.asyncio
async def test_a_fixed_temperature_engine_refuses_other_temperatures(monkeypatch, flash_engine):
    fake = FixedTemperatureFlash()
    engine = flash_engine(fake)
    assert engine.fixed_temperature
    for path in ("/predict", "/stream"):
        response = await _post(monkeypatch, engine, path, {"text": "Hi.", "temperature": 0.5})
        assert response.status_code == 422 and "fixed temperature" in response.json()["error"], (path, response.text)
    assert fake.calls == []
    response = await _post(monkeypatch, engine, "/predict", {"text": "Hi.", "temperature": 0.7, "mossformer2_postprocess": False})
    assert response.status_code == 200 and len(fake.calls) == 1


GiB = 1 << 30


@pytest.mark.parametrize("dev, n, quota, free, expected", [
    ("cpu", None, 0.0, None, [{}]),                          # one engine on the free cores (its own choice)
    ("cpu", 2, 0.0, None, [{"threads": 6}] * 2),            # FLASH_ENGINES=2: 12 physical cores shared
    ("cpu", 3, 4.0, None, [{"threads": 1}] * 3),            # a container quota of 4 CPUs
    ("gpu", None, 0.0, (20 * GiB,), [{"max_batch": 64}]),    # one batch engine: as many requests as fit, at most 64
    ("gpu", None, 0.0, (8 * GiB,), [{"max_batch": 24}]),     # 2 GiB stay free, 3 GiB base, 128 MiB per request
    ("gpu", 3, 0.0, None, [{"max_batch": 3}]),               # FLASH_ENGINES=3
])
def test_engine_count(monkeypatch, dev, n, quota, free, expected):
    from types import SimpleNamespace

    made, probes, rates = [], [], []
    if free is not None:
        torch = pytest.importorskip("torch")
        readings = iter(free)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (next(readings), 24 * GiB))

    class Engine:
        def __init__(self, variant, weights=None, sample_rate=None, **kwargs):
            rates.append(sample_rate)
            (probes if kwargs.get("threads") == 1 and not probes and (n or 1) > 1 and dev == "cpu" else made).append(kwargs)
            self.info = {"physical_cores": 12, "quota": quota}

        def close(self):
            pass

    monkeypatch.setattr(flash, "engine_module", lambda d: SimpleNamespace(Engine=Engine))
    monkeypatch.setattr(flash, "batch_module", lambda: SimpleNamespace(BatchEngine=Engine))
    assert len(flash.build_engines(dev, "/w", n, 24000)) == len(expected)
    assert made == expected and len(probes) == (dev == "cpu" and (n or 1) > 1) and set(rates) == {24000}
