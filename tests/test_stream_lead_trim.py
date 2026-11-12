"""STREAM_LEAD_TRIM: /predict's leading edge trim on the first streamed chunk (default on, the release config).

The cut must equal /predict's leading cut (main._trim_boundary_silence, 250 ms padding, the headroom-matched
threshold), rounded down to the 480-sample Opus grid. Leads already within the padding, chunks without sound
and every later chunk are untouched; with the flag off the stream is the pre-flag stream object for object.
"""
import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest
import torch

import app.main as app_main
from app.config import ServerConfig
from app.inference import TTSEngine
from app.opus_stream import OpusStreamEncoder
from app.stream_lead_trim import OPUS_GRID_SAMPLES, stream_lead_cut, trim_stream_lead

SR = 24_000
GAIN_36 = 10.0 ** (-3.6 / 20.0)
EARLY_CHUNK = 67_200  # 2.8 s first packet = 21 VibeVoice frames of 3200 samples


def _speechy(n, seed):
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(n) / SR
    return 0.3 * torch.sin(2 * torch.pi * 180 * t) * (0.6 + 0.4 * torch.rand(n, generator=g))


def _chunk(lead_samples, n=EARLY_CHUNK, floor_dbfs=-70.0, seed=0):
    """lead_samples of low noise (floor_dbfs RMS), then speech to the end of the chunk (no trailing silence)."""
    g = torch.Generator().manual_seed(seed + 1)
    noise = torch.randn(lead_samples, generator=g) * 10.0 ** (floor_dbfs / 20.0)
    return torch.cat([noise, _speechy(n - lead_samples, seed)]).reshape(1, 1, -1)


def _predict_lead_cut(audio, headroom_db):
    """/predict's leading cut on the same audio, scaled by the output headroom first (as /predict does)."""
    scaled = audio * (10.0 ** (-headroom_db / 20.0)) if headroom_db else audio
    trimmed = app_main._trim_boundary_silence(scaled, padding_ms=250, headroom_db=headroom_db)
    assert trimmed.shape[-1] > 0
    return scaled.shape[-1] - trimmed.shape[-1]          # the chunk ends in speech: nothing is cut at the end


@pytest.mark.parametrize("headroom_db", [0.0, 3.6])
@pytest.mark.parametrize("lead_ms", [0, 120, 250, 260, 300, 333, 517, 640, 1000, 1733, 2500])
def test_cut_is_predicts_leading_cut_on_the_opus_grid(lead_ms, headroom_db):
    audio = _chunk(round(SR * lead_ms / 1000), seed=lead_ms)
    cut = stream_lead_cut(audio, padding_ms=250)
    ref = _predict_lead_cut(audio, headroom_db)
    assert cut % OPUS_GRID_SAMPLES == 0
    assert cut == ref // OPUS_GRID_SAMPLES * OPUS_GRID_SAMPLES     # /predict's cut, rounded down to the grid
    assert 0 <= ref - cut < OPUS_GRID_SAMPLES


@pytest.mark.parametrize("lead_ms", [0, 100, 240, 250])
def test_leads_within_the_padding_are_untouched(lead_ms):
    audio = _chunk(round(SR * lead_ms / 1000))
    assert stream_lead_cut(audio, padding_ms=250) == 0
    assert trim_stream_lead(audio, padding_ms=250) is audio


def test_silent_short_and_empty_chunks_are_untouched():
    silent = torch.zeros(1, 1, EARLY_CHUNK)
    quiet = torch.randn(1, 1, EARLY_CHUNK) * 10.0 ** (-60.0 / 20.0)
    short = torch.ones(1, 1, 300)
    empty = torch.zeros(1, 1, 0)
    for audio in (silent, quiet, short, empty):
        assert trim_stream_lead(audio, padding_ms=250) is audio


def test_keeps_the_padding_before_the_first_sound():
    lead = round(SR * 0.731)
    audio = _chunk(lead)
    out = trim_stream_lead(audio, padding_ms=250)
    kept_lead = lead - (audio.shape[-1] - out.shape[-1])
    # at least the padding; at most the padding + the 20 ms frame that reaches into the onset + the grid rounding
    assert round(SR * 0.25) <= kept_lead < round(SR * 0.25) + 2 * OPUS_GRID_SAMPLES
    assert torch.equal(out, audio[..., audio.shape[-1] - out.shape[-1]:])


def test_trimmed_early_chunk_needs_no_opus_padding():
    audio = _chunk(round(SR * 0.9))
    out = trim_stream_lead(audio, padding_ms=250)
    assert out.shape[-1] < audio.shape[-1]
    assert out.shape[-1] % OPUS_GRID_SAMPLES == 0
    enc = OpusStreamEncoder(SR)
    blob = enc.encode(out)
    assert int.from_bytes(blob[:2], "little") == out.shape[-1] // OPUS_GRID_SAMPLES   # whole frames, no zero tail


def test_flag_default_on_and_env():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("STREAM_LEAD_TRIM", None)
        os.environ.pop("STREAM_LEAD_TRIM_PADDING_MS", None)
        cfg = ServerConfig()
        assert cfg.stream_lead_trim is True and cfg.stream_lead_trim_padding_ms == 250
    with patch.dict(os.environ, {"STREAM_LEAD_TRIM": "0"}):
        assert ServerConfig().stream_lead_trim is False
    with patch.dict(os.environ, {"STREAM_LEAD_TRIM": "1", "STREAM_LEAD_TRIM_PADDING_MS": "300"}):
        cfg = ServerConfig()
        assert cfg.stream_lead_trim is True and cfg.stream_lead_trim_padding_ms == 300


# --- generate_streaming: the first non-empty chunk only ----------------------------------------------------------
def _stream_engine(trim, headroom_db):
    config = ServerConfig()
    config.output_headroom_db = headroom_db
    config.stream_lead_trim = trim
    engine = TTSEngine(config)
    engine.vv_tokenizer = object()
    engine.vv_engine = MagicMock()
    engine.vv_engine.alloc_slot.return_value = 0
    return engine


def _run_stream(engine, chunks_in):
    async def inner(**_kwargs):
        for c in chunks_in:
            yield c

    engine._generate_streaming_inner = inner

    async def collect():
        return [c async for c in engine.generate_streaming("Some text.")]

    return asyncio.run(collect())


def _chunks():
    first = _chunk(round(SR * 0.85), seed=3)
    later = [_chunk(round(SR * 0.6), n=48_000, seed=4), torch.zeros(1, 1, 48_000),
             torch.linspace(0.8, 0.0, 3201).reshape(1, 1, -1)]
    return [torch.zeros(1, 1, 0), first] + later


@pytest.mark.parametrize("headroom_db", [0.0, 3.6])
def test_flag_off_is_the_pre_flag_stream(headroom_db):
    chunks_in = _chunks()
    out = _run_stream(_stream_engine(False, headroom_db), chunks_in)
    assert len(out) == len(chunks_in)
    for o, c in zip(out, chunks_in):
        if headroom_db == 0.0:
            assert o is c
        else:
            assert torch.equal(o, c * GAIN_36)


@pytest.mark.parametrize("headroom_db", [0.0, 3.6])
def test_flag_on_trims_the_first_non_empty_chunk_only(headroom_db):
    chunks_in = _chunks()
    copies = [c.clone() for c in chunks_in]
    engine = _stream_engine(True, headroom_db)
    out = _run_stream(engine, chunks_in)
    assert len(out) == len(chunks_in)
    cut = stream_lead_cut(copies[1], padding_ms=250)
    assert cut > 0 and cut == _predict_lead_cut(copies[1], headroom_db) // OPUS_GRID_SAMPLES * OPUS_GRID_SAMPLES
    gain = 1.0 if headroom_db == 0.0 else GAIN_36
    assert out[0].shape[-1] == 0
    assert torch.equal(out[1], copies[1][..., cut:] * gain)
    for o, c, ref in zip(out[2:], chunks_in[2:], copies[2:]):        # later chunks: unchanged, even with a lead
        assert torch.equal(o, ref * gain)
        if headroom_db == 0.0:
            assert o is c
    for c, ref in zip(chunks_in, copies):
        assert torch.equal(c, ref)                                   # the inner chunks are never modified
    engine.vv_engine.free_slot.assert_called_once_with(0)


def test_flag_on_short_lead_first_chunk_is_untouched():
    first = _chunk(round(SR * 0.2), seed=5)
    out = _run_stream(_stream_engine(True, 0.0), [first, _chunk(round(SR * 0.9), n=48_000, seed=6)])
    assert out[0] is first
    assert out[1].shape[-1] == 48_000                                # the later chunk's lead is never trimmed
