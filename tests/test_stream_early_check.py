"""STREAM_EARLY_CHECK: the first streamed chunk is checked before it is played (default on, the release config).

The three failure modes of the early chunk (/predict unaffected): a SILENT early chunk (a 12794 run that the early
silence ratio did not count), a NEAR-SILENT early chunk (codes that pass, audio that does not), and RETRIES EXHAUSTED
(every attempt fails). With the flag a failed early chunk is re-sampled from scratch and is never forced into
segment 0; with every attempt failed the stream starts at segment 0 without an early chunk. With the flag off the
stream is the pre-flag stream.
"""
import asyncio
import math
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import torch

from app.config import ServerConfig
from app.inference import TTSEngine
from app.stream_early_check import EARLY_LEAD_SILENCE_TOKEN, active_fraction

BASE = 100
SR = 24_000
TEXT = "Some text for the stream, long enough to be checked."
GOOD = [3716] + [5000 + i for i in range(39)]                  # a normal start
SILENT = [3716] + [EARLY_LEAD_SILENCE_TOKEN] * 39               # a 12794 run: passes serving's ratio, fails with 12794 counted
QUIET = [3716] + [7000 + i for i in range(39)]                  # codes that pass, audio that is (near-)silent
SEG = [BASE + 6000 + i for i in range(60)]


def _audio_for(cb0, quiet_kind="near"):
    n = len(cb0) * 1920
    t = torch.arange(n) / SR
    if cb0[1] == EARLY_LEAD_SILENCE_TOKEN:
        return torch.zeros(1, 1, n)
    if cb0[1] == 7000:
        x = torch.zeros(n)
        if quiet_kind == "near":                                  # a 0.2 s burst: RMS above /predict's 0.002, active share < 0.10
            x[SR:SR + SR // 5] = 0.3 * torch.sin(2 * math.pi * 200 * t[SR:SR + SR // 5])
        else:
            x += 1e-3 * torch.sin(2 * math.pi * 200 * t)            # RMS 0.0007: /predict's silent check
        return x.reshape(1, 1, -1)
    return (0.3 * torch.sin(2 * math.pi * 200 * t)).reshape(1, 1, -1)


def _engine(flag, early_codes, quiet_kind="near", attempts=3):
    config = ServerConfig()
    config.stream_early_check = flag
    config.stream_early_check_attempts = attempts
    config.output_headroom_db = 0.0
    config.cb0_drop_bad_prev_audio = False
    engine = TTSEngine(config)
    engine.vv_tokenizer = object()
    engine.vv_engine = MagicMock()
    engine.vv_engine.alloc_slot.return_value = 0
    early_chunk = torch.full((1, 1, 67200), 0.25)
    engine.vv_engine.decode = AsyncMock(return_value=early_chunk)
    engine._resolve_params = MagicMock(return_value=(dict(temperature=0.8, top_p=0.95, top_k=50, frequency_penalty=0.0, seed=7),
                                                     (None, None), "", 0.0, -1))
    engine._resolve_prefix = AsyncMock(return_value=(None, {}))
    engine._n_codebooks = MagicMock(return_value=4)
    engine._cb_ctx = MagicMock(return_value=SimpleNamespace(dc_base_id=BASE))
    engine._generate_semantic_segment = AsyncMock(side_effect=[([BASE + c for c in e], False) for e in early_codes] + [(SEG, True)])

    async def fake_acoustic(state, **_kw):
        for cb in range(3):
            state.ac_segs[cb].append([0] * len(state.cb0_segments[-1]))
    engine._generate_all_acoustic_for_segment = AsyncMock(side_effect=fake_acoustic)
    seg_chunk = torch.full((1, 1, 48000), 0.5)
    engine._emit_vv_incremental = AsyncMock(side_effect=[(seg_chunk, 36), (None, 36)])
    engine._append_trailing_silence = MagicMock()
    return engine, early_chunk, seg_chunk


def _run(engine, quiet_kind="near"):
    async def collect():
        return [c async for c in engine.generate_streaming(TEXT, max_new_tokens=400)]
    with patch("app.inference.codes_to_audio_tensor", side_effect=lambda _dc, cb0, _ac: _audio_for(cb0, quiet_kind)) as dc, \
            patch("app.inference.vv_encode", return_value=torch.zeros(1, 30, 3)):
        out = asyncio.run(collect())
    return out, dc


def _calls(engine):
    """(early calls, the segment-0 call) of _generate_semantic_segment."""
    calls = engine._generate_semantic_segment.call_args_list
    return [c for c in calls if c.kwargs.get("early_mode")], [c for c in calls if not c.kwargs.get("early_mode")]


def test_silent_early_chunk_is_resampled_and_never_forced():
    engine, early_chunk, seg_chunk = _engine(True, [SILENT, GOOD])
    out, dc = _run(engine)
    early, seg = _calls(engine)
    assert len(early) == 2
    assert early[0].kwargs["sampling"]["seed"] == 7 and early[1].kwargs["sampling"]["seed"] == 8
    assert math.isclose(early[1].kwargs["sampling"]["temperature"], 0.8 + engine.config.vv_retry_temperature_step)
    assert dc.call_count == 1                                     # the silent codes never reach CB1-3 / the decode
    assert seg[0].kwargs["forced_prefix"] == GOOD                 # the accepted start, not the silent one
    assert out[0] is early_chunk and out[1] is seg_chunk


def test_near_silent_early_chunk_is_resampled():
    engine, early_chunk, seg_chunk = _engine(True, [QUIET, GOOD])
    out, dc = _run(engine, "near")
    early, seg = _calls(engine)
    assert len(early) == 2 and dc.call_count == 2                 # QUIET passes the codes check, fails on its audio
    assert seg[0].kwargs["forced_prefix"] == GOOD
    assert len(out) == 2 and out[0] is early_chunk


def test_silent_audio_with_passing_codes_is_resampled():
    engine, early_chunk, _ = _engine(True, [QUIET, GOOD])
    out, dc = _run(engine, "silent")
    _early, seg = _calls(engine)
    assert dc.call_count == 2 and seg[0].kwargs["forced_prefix"] == GOOD


def test_retries_exhausted_start_without_an_early_chunk():
    engine, early_chunk, seg_chunk = _engine(True, [SILENT, QUIET, SILENT])
    out, dc = _run(engine)
    early, seg = _calls(engine)
    assert len(early) == 3
    assert seg[0].kwargs["forced_prefix"] is None                 # no failed early prefix forced into segment 0
    engine.vv_engine.decode.assert_not_called()                   # nothing early was played
    assert out == [seg_chunk]


def test_flag_off_is_the_pre_flag_stream():
    engine, early_chunk, seg_chunk = _engine(False, [SILENT])
    out, dc = _run(engine)
    early, seg = _calls(engine)
    assert len(early) == 1 and early[0].kwargs["sampling"]["seed"] == 7
    assert seg[0].kwargs["forced_prefix"] == SILENT               # today: the silent start is played and forced
    assert out[0] is early_chunk and out[1] is seg_chunk


def test_flag_on_good_start_equals_flag_off():
    runs = []
    for flag in (False, True):
        engine, early_chunk, seg_chunk = _engine(flag, [GOOD])
        out, dc = _run(engine)
        early, seg = _calls(engine)
        runs.append((len(early), early[0].kwargs["sampling"], seg[0].kwargs["forced_prefix"], dc.call_count,
                     [o is c for o, c in zip(out, (early_chunk, seg_chunk))]))
    assert runs[0] == runs[1]


def test_12794_counts_only_in_early_mode_with_the_flag():
    for flag in (False, True):
        config = ServerConfig()
        config.stream_early_check = flag
        engine = TTSEngine(config)
        early = engine._check_cb0_generation(40, len(TEXT), False, 12.5, codes=SILENT, early_mode=True)
        full = engine._check_cb0_generation(80, 60, False, 12.5, codes=SILENT + [5000] * 40, early_mode=False)
        assert (early is not None) == flag and (early is None or early.startswith("too_much_silence"))
        assert full is None or not full.startswith("too_much_silence") or not flag


def test_active_fraction():
    t = torch.arange(SR * 2) / SR
    assert active_fraction(torch.zeros(1, 1, SR)) == 0.0
    assert active_fraction((0.3 * torch.sin(2 * math.pi * 200 * t)).reshape(1, 1, -1)) > 0.99
    assert active_fraction(torch.zeros(1, 1, 100)) == 0.0


def test_flag_default_on_and_env():
    with patch.dict(os.environ, {}, clear=False):
        for k in ("STREAM_EARLY_CHECK", "STREAM_EARLY_CHECK_ATTEMPTS", "STREAM_EARLY_MIN_ACTIVE"):
            os.environ.pop(k, None)
        c = ServerConfig()
        assert c.stream_early_check is True and c.stream_early_check_attempts == 3 and c.stream_early_min_active == 0.10
    with patch.dict(os.environ, {"STREAM_EARLY_CHECK": "0"}):
        assert ServerConfig().stream_early_check is False
    with patch.dict(os.environ, {"STREAM_EARLY_CHECK": "1", "STREAM_EARLY_CHECK_ATTEMPTS": "2", "STREAM_EARLY_MIN_ACTIVE": "0.2"}):
        c = ServerConfig()
        assert c.stream_early_check is True and c.stream_early_check_attempts == 2 and c.stream_early_min_active == 0.2
