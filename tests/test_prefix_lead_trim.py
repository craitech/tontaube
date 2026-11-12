"""PREFIX_LEAD_TRIM_PADDING_MS (default 80): a shorter lead before the first sound, for requests WITH a prefix only.
Off = the earlier behaviour exactly; /stream's first-chunk lead trim and /predict's leading edge use the shorter
padding; the tail, requests without a prefix and generation are untouched."""
import asyncio
import os
from unittest.mock import patch

import pytest
import torch

import app.main as app_main
from app.config import ServerConfig
from app.stream_lead_trim import OPUS_GRID_SAMPLES, stream_lead_cut
from tests.test_stream_lead_trim import SR, _chunk, _speechy, _stream_engine


def test_config_default_80_off_and_env():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("PREFIX_LEAD_TRIM_PADDING_MS", None)
        assert ServerConfig().prefix_lead_trim_padding_ms == 80          # default 80
    with patch.dict(os.environ, {"PREFIX_LEAD_TRIM_PADDING_MS": "120"}):
        assert ServerConfig().prefix_lead_trim_padding_ms == 120
    with patch.dict(os.environ, {"PREFIX_LEAD_TRIM_PADDING_MS": ""}):
        assert ServerConfig().prefix_lead_trim_padding_ms == 80
    for off in ("off", "none", "OFF"):
        with patch.dict(os.environ, {"PREFIX_LEAD_TRIM_PADDING_MS": off}):
            assert ServerConfig().prefix_lead_trim_padding_ms is None     # the old behaviour


def _edged(lead_s, tail_s, seed=0):
    g = torch.Generator().manual_seed(seed)
    n_lead, n_tail = round(SR * lead_s), round(SR * tail_s)
    return torch.cat([torch.randn(n_lead, generator=g) * 1e-4, _speechy(SR, seed), torch.randn(n_tail, generator=g) * 1e-4]).reshape(1, 1, -1)


@pytest.mark.parametrize("headroom_db", [0.0, 3.6])
def test_predict_trim_lead_padding_moves_only_the_start(headroom_db):
    audio = _edged(0.6, 0.6) * (10.0 ** (-headroom_db / 20.0))
    old = app_main._trim_boundary_silence(audio, padding_ms=250, headroom_db=headroom_db)
    same = app_main._trim_boundary_silence(audio, padding_ms=250, headroom_db=headroom_db, lead_padding_ms=None)
    assert torch.equal(old, same)                                           # None = the old function
    new = app_main._trim_boundary_silence(audio, padding_ms=250, headroom_db=headroom_db, lead_padding_ms=80)
    lead_cut = old.shape[-1] - new.shape[-1]
    assert lead_cut == round(SR * 0.170)                                    # 250 - 80 ms more at the start
    assert torch.equal(new, old[..., lead_cut:])                            # same end, same samples


def test_predict_lead_padding_never_cuts_into_sound():
    audio = _edged(0.05, 0.3)                                               # a lead already shorter than 80 ms
    new = app_main._trim_boundary_silence(audio, padding_ms=250, lead_padding_ms=80)
    old = app_main._trim_boundary_silence(audio, padding_ms=250)
    assert torch.equal(new, old)


def _run(engine, chunks_in, prefix_text):
    async def inner(**_kwargs):
        for c in chunks_in:
            yield c

    engine._generate_streaming_inner = inner

    async def collect():
        return [c async for c in engine.generate_streaming("Some text.", prefix_text=prefix_text)]

    return asyncio.run(collect())


@pytest.mark.parametrize("prefix_text, padding, want_ms", [
    (None, None, 250), (None, 80, 250),                                    # no prefix: never affected
    ("Before the join,", None, 250), ("Before the join,", 80, 80),
])
def test_stream_first_chunk_lead(prefix_text, padding, want_ms):
    first = _chunk(round(SR * 0.85), seed=3)
    later = _chunk(round(SR * 0.6), n=48_000, seed=4)
    copies = [first.clone(), later.clone()]
    engine = _stream_engine(True, 0.0)
    engine.config.prefix_start_fade = False                                 # isolate the trim from the 10-ms fade
    engine.config.prefix_lead_trim_padding_ms = padding
    out = _run(engine, [first, later], prefix_text)
    cut = stream_lead_cut(copies[0], padding_ms=want_ms)
    assert cut > 0 and cut % OPUS_GRID_SAMPLES == 0
    assert torch.equal(out[0], copies[0][..., cut:])
    assert torch.equal(out[1], copies[1])                                   # later chunks untouched
