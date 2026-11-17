"""Streaming start buffer: the calibration fit, the per-request choice, and the /stream wiring."""
import logging

import httpx
import pytest

import app.main as app_main
from app.config import DEFAULT_STREAMING_INITIAL_SECONDS, MAX_STREAMING_INITIAL_SECONDS
from app.stream_calibration import (
    CALIBRATION_TEXT,
    StreamCalibration,
    quantize_down,
    quantize_up,
)


def calibration(rate, overhead=0.3, first=0.3, cps=14.0):
    return StreamCalibration(rate=rate, segment_overhead_s=overhead, first_overhead_s=first,
                             chars_per_second=cps, took_s=1.0)


def synthetic_timings(rate, overhead, first, early, segments):
    """What generate_streaming(timings=...) records for a machine with these numbers."""
    t = first + early / rate
    timings, cumulative = [("early", t, early)], 0.0
    for k, duration in enumerate(segments):
        t += overhead + (duration - (early if k == 0 else 0.0)) / rate
        cumulative += duration
        timings.append(("segment", t, cumulative))
    timings.append(("end", t + 0.05, cumulative))
    return timings


def test_quantize_to_the_streaming_grid():
    assert quantize_up(2.81) == 3.2 and quantize_up(2.8) == 2.8 and quantize_up(0.1) == 0.4
    assert quantize_down(2.79) == 2.4 and quantize_down(6.0) == 6.0


def test_fit_recovers_rate_and_overheads():
    timings = synthetic_timings(rate=3.0, overhead=0.4, first=0.25, early=0.4, segments=[3.2, 15.0])
    fit = StreamCalibration.from_timings(timings, [44, 210], took_s=7.5)
    assert fit.rate == pytest.approx(3.0)
    assert fit.segment_overhead_s == pytest.approx(0.4)
    assert fit.first_overhead_s == pytest.approx(0.25)
    assert fit.chars_per_second == pytest.approx(254 / 18.2)
    assert fit.took_s == 7.5


def test_fit_without_a_usable_slope_falls_back_to_the_average_rate():
    timings = [("early", 0.5, 0.4), ("segment", 2.5, 4.4), ("segment", 4.0, 8.4)]  # second segment faster
    fit = StreamCalibration.from_timings(timings, [60, 60], took_s=4.0)
    assert fit.segment_overhead_s == 0.0
    assert fit.rate == pytest.approx((4.0 + 4.0) / (2.0 + 1.5))


def test_fit_without_an_early_chunk_counts_the_start_up_work_into_segment_zero():
    timings = synthetic_timings(rate=3.0, overhead=0.4, first=0.25, early=0.0, segments=[3.2, 15.0])[1:]
    fit = StreamCalibration.from_timings(timings, [44, 210], took_s=7.5)
    assert fit.rate == pytest.approx(3.0, rel=0.1) and fit.first_overhead_s == 0.0
    assert fit.segment_overhead_s > 0.4


def test_fit_refuses_incomplete_timings():
    with pytest.raises(ValueError):
        StreamCalibration.from_timings([("segment", 1.0, 3.0)], [40], took_s=1.0)


def test_fast_machine_keeps_the_floor():
    initial, pauses = calibration(rate=13.0).plan([350, 350, 350], floor=2.8)
    assert (initial, pauses) == (2.8, False)


def test_fast_machine_could_start_sooner_with_a_lower_floor():
    fast = calibration(rate=13.0)
    assert fast.plan([350, 350], floor=0.4) == (2.8, False)   # 2.8 s is just enough for a full first segment
    initial, pauses = fast.plan([150, 350], floor=0.4)
    assert initial < 2.8 and not pauses


def test_slow_machine_gets_a_larger_buffer_for_a_long_first_segment():
    slow = calibration(rate=5.0)
    long_first, pauses = slow.plan([350, 350], floor=2.8)
    short_text, _ = slow.plan([40], floor=2.8)
    assert 2.8 < long_first < MAX_STREAMING_INITIAL_SECONDS and not pauses and short_text == 2.8
    assert not slow._pauses(long_first, [350 / 14.0, 350 / 14.0], 1.25)
    assert slow._pauses(2.8, [350 / 14.0, 350 / 14.0], 1.25)


def test_below_about_four_times_real_time_a_full_first_segment_pauses_even_at_six_seconds():
    assert calibration(rate=4.5).plan([350], floor=2.8) == (6.0, False)
    assert calibration(rate=3.0).plan([350], floor=2.8) == (6.0, True)


def test_machine_slower_than_real_time_reports_pauses_at_the_largest_buffer():
    initial, pauses = calibration(rate=0.8).plan([350, 350, 350], floor=2.8)
    assert (initial, pauses) == (MAX_STREAMING_INITIAL_SECONDS, True)


def test_ceiling_from_max_new_tokens_is_respected():
    initial, pauses = calibration(rate=0.8).plan([350], floor=0.4, ceiling=2.4)
    assert initial == 2.4 and pauses


def test_summary_mentions_the_measured_numbers():
    text = calibration(rate=2.5).summary(2.8, 1.25)
    assert "2.50x real time" in text and "calibration took 1.0 s" in text


class _Engine:
    """Streams like a machine at `rate`x real time, recording the timings the real engine would."""

    vv_tokenizer = object()
    verbalizer = None

    def __init__(self, rate=3.0):
        self.rate = rate
        self.initial_seconds = []

    async def generate_streaming(self, text, timings=None, streaming_initial_seconds=None, **kwargs):
        self.initial_seconds.append(streaming_initial_seconds)
        pieces = [text] if isinstance(text, str) else list(text)
        if timings is not None:
            timings.extend(synthetic_timings(self.rate, 0.3, 0.2, streaming_initial_seconds,
                                             [len(piece) / 14.0 for piece in pieces]))
        yield b""


@pytest.fixture(autouse=True)
def app_logs(caplog):
    """The app's logger does not propagate; hand its records to caplog."""
    caplog.set_level(logging.INFO)
    app_main.logger.addHandler(caplog.handler)
    yield
    app_main.logger.removeHandler(caplog.handler)


class _Writer:
    def __init__(self, *args, **kwargs):
        pass

    def write_chunk(self, chunk):
        return b""

    def close(self):
        return b""


async def _stream(monkeypatch, engine, body, calibrate=True):
    monkeypatch.setenv("ENABLE_STREAMING", "1")
    monkeypatch.setattr(app_main, "STREAM_CALIBRATION", calibrate)

    async def get_engine():
        return engine

    monkeypatch.setattr(app_main, "_get_engine", get_engine)
    monkeypatch.setattr(app_main, "_unload_engine", lambda: None)

    monkeypatch.setattr("app.stream_writers.Mp3StreamWriter", _Writer)
    application = app_main.create_app()
    async with application.router.lifespan_context(application):
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="http://testserver")
        return await client.post("/stream", json=body)


LONG = " ".join(["The keeper climbed the stairs again and looked out over the quiet water."] * 4)


@pytest.mark.asyncio
async def test_startup_calibrates_and_a_stream_without_the_field_gets_the_planned_buffer(monkeypatch, caplog):
    engine = _Engine(rate=5.0)
    response = await _stream(monkeypatch, engine, {"text": LONG})
    assert response.status_code == 200
    calibration_run, request = engine.initial_seconds
    assert calibration_run == 1.2
    assert request > DEFAULT_STREAMING_INITIAL_SECONDS
    assert any("Stream calibration: 5.00x real time" in r.getMessage() for r in caplog.records)
    assert not any("set explicitly" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_an_explicit_value_is_used_and_logged_as_a_warning(monkeypatch, caplog):
    engine = _Engine(rate=5.0)
    response = await _stream(monkeypatch, engine, {"text": LONG, "streaming_initial_seconds": 1.2})
    assert response.status_code == 200 and engine.initial_seconds[-1] == 1.2
    [warning] = [r for r in caplog.records if "set explicitly" in r.getMessage()]
    assert warning.levelno == logging.WARNING and "playback may pause" in warning.getMessage()


OFF_WARNING = "Stream calibration is off"


def test_calibration_is_off_by_default():
    assert app_main.STREAM_CALIBRATION is False and app_main.STREAM_CALIBRATION_WARNING is True


@pytest.mark.asyncio
async def test_without_calibration_the_fixed_default_applies_and_the_first_stream_warns_once(monkeypatch, caplog):
    engine = _Engine(rate=5.0)
    monkeypatch.setenv("ENABLE_STREAMING", "1")
    monkeypatch.setattr(app_main, "STREAM_CALIBRATION", False)

    async def get_engine():
        return engine

    monkeypatch.setattr(app_main, "_get_engine", get_engine)
    monkeypatch.setattr(app_main, "_unload_engine", lambda: None)
    monkeypatch.setattr("app.stream_writers.Mp3StreamWriter", _Writer)
    application = app_main.create_app()
    async with application.router.lifespan_context(application):
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="http://testserver")
        for body in ({"text": LONG}, {"text": LONG}, {"text": LONG, "streaming_initial_seconds": 1.2}):
            assert (await client.post("/stream", json=body)).status_code == 200
    assert engine.initial_seconds == [DEFAULT_STREAMING_INITIAL_SECONDS, DEFAULT_STREAMING_INITIAL_SECONDS, 1.2]
    warnings = [r for r in caplog.records if OFF_WARNING in r.getMessage()]
    assert len(warnings) == 1 and warnings[0].levelno == logging.WARNING
    assert "STREAM_CALIBRATION_WARNING=0" in warnings[0].getMessage()
    assert not any("set explicitly" in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_the_warning_can_be_suppressed(monkeypatch, caplog):
    monkeypatch.setattr(app_main, "STREAM_CALIBRATION_WARNING", False)
    response = await _stream(monkeypatch, _Engine(rate=5.0), {"text": LONG}, calibrate=False)
    assert response.status_code == 200
    assert not any(OFF_WARNING in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_no_off_warning_when_calibrated(monkeypatch, caplog):
    response = await _stream(monkeypatch, _Engine(rate=5.0), {"text": LONG})
    assert response.status_code == 200
    assert not any(OFF_WARNING in r.getMessage() for r in caplog.records)


def test_calibration_text_is_one_short_and_one_longer_segment():
    short, longer = (len(piece) for piece in CALIBRATION_TEXT)
    assert short < 60 and 150 < longer < 350
