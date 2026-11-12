"""STREAM_RETRY_FIX (streaming CB0 retries) and STREAM_EARLY_PACE_MIN_ACTIVE (the early pace floor): the fix default on, the floor
default off (the release config).

Without the fix: with the request's seed, retry 2 repeated retry 1 (the same temperature); streaming keeps the played early codes as the
prefix of every segment-0 attempt, so its retries rarely moved the chars/s, and after 5 failures the LAST (hottest) attempt was served.
With the flag every streaming CB0 retry draws a fresh sample (seed + attempt - 1) and, when all fail, the attempt closest to the chars/s
window is kept. /predict never passes the flag.
"""
import asyncio
import inspect
import math
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import torch

from app.config import ServerConfig
from app.inference import TTSEngine

TEXT = "a" * 80                                                       # 80 chars: the window 8-20 chars/s = 50-125 codes at 12.5 Hz


def _engine(retry_fix_cfg=False):
    engine = TTSEngine(ServerConfig())
    engine.config.force_first_silence = False
    engine.config.stream_retry_fix = retry_fix_cfg
    ctx = SimpleNamespace(tokenizer=MagicMock(), dc_base_id=100, pad_id=0, eos_id=2, audio_marker_id=3, hz=12.5, engine=MagicMock())
    engine._cb_ctx = MagicMock(return_value=ctx)
    return engine


def _run(engine, outputs, flag, early=False, forced_prefix=None):
    engine._generate_with_engine = AsyncMock(side_effect=outputs)
    sampling = dict(temperature=0.8, top_p=0.95, top_k=50, seed=123)

    async def go():
        with patch("app.inference.tokenize_text", return_value=[10]), \
                patch("app.inference.SamplingParams", side_effect=lambda **kw: SimpleNamespace(**kw)):
            return await engine._generate_semantic_segment(
                seg_idx=0, segments=[TEXT], prev_chunk_global=None, sampling=sampling, voice_prompt=(None, None), prefix=None,
                max_new_tokens=400, early_mode=early, forced_prefix=forced_prefix, stream_retry_fix=flag)
    codes, _eos = asyncio.run(go())
    params = [c.args[2] for c in engine._generate_with_engine.await_args_list]
    return codes, params, sampling


def _out(n, tok=120):
    return [tok] * n + [2]                                             # n codes, then EOS


# all five attempts fail the chars/s window (160, 40, 150, 300 and 200 codes); attempt 3 (150 codes) is the closest
# to it, and the last attempt has 200 codes
FAILS = [_out(160, 120), _out(40, 121), _out(150, 122), _out(300, 123), _out(200, 124)]


def test_flag_off_keeps_the_last_attempt_with_the_guard_fix_seeds():
    # with the flag off the guard fix reseeds every retry (retry_seed), so the old same-seed retries are gone on
    # every path; only the closest-attempt fallback stays the flag's.
    codes, params, sampling = _run(_engine(), FAILS, False)
    assert [p.seed for p in params] == [123 + k * 1_000_003 for k in range(5)]
    assert codes == [124 for _ in range(200)]                         # the last attempt
    assert sampling["seed"] == 123


def test_flag_on_fresh_seed_per_retry_and_the_closest_attempt():
    codes, params, sampling = _run(_engine(), FAILS, True)
    assert [p.seed for p in params] == [123, 124, 125, 126, 127]
    assert [round(p.temperature, 6) for p in params] == [0.8, 0.8, 0.85, 0.9, 0.95]   # the temperature schedule is unchanged
    assert codes == [122] * 150                                       # the attempt closest to the chars/s window
    assert sampling["seed"] == 123                                    # the caller's sampling dict is not mutated


def test_flag_on_a_passing_retry_is_kept():
    codes, params, _ = _run(_engine(), [_out(160, 120), _out(80, 125)], True)
    assert [p.seed for p in params] == [123, 124]
    assert codes == [125] * 80


def test_flag_on_early_mode_fresh_seeds_no_cps_fallback():
    silent = [100 + 12725] * 40 + [2]                                 # a collapse-silence run fails the early ratio
    codes, params, _ = _run(_engine(), [silent] * 5, True, early=True)
    assert [p.seed for p in params] == [123, 124, 125, 126, 127]
    assert codes == [100 + 12725] * 40                                # early mode: no chars/s fallback, the last attempt


def test_forced_prefix_is_kept_and_counted():
    prefix = [5000] * 40
    codes, params, _ = _run(_engine(), [_out(120, 126)] * 5, True, forced_prefix=prefix)
    assert codes[:40] == [100 + 5000] * 40 and len(codes) == 160    # every attempt starts with the prefix; cps counts it


def test_predict_path_never_passes_the_flag():
    sig = inspect.signature(TTSEngine._generate_semantic_segment)
    assert sig.parameters["stream_retry_fix"].default is False
    src = inspect.getsource(TTSEngine._generate_semantic)
    assert "stream_retry_fix" not in src


def test_streaming_passes_the_configured_flag():
    for flag in (False, True):
        config = ServerConfig()
        config.stream_retry_fix = flag
        config.stream_lead_trim = config.stream_early_check = False   # one early CB0 call (the mock's first side effect)
        config.output_headroom_db = 0.0
        config.cb0_drop_bad_prev_audio = False
        engine = TTSEngine(config)
        engine.vv_tokenizer = object()
        engine.vv_engine = MagicMock()
        engine.vv_engine.alloc_slot.return_value = 0
        engine.vv_engine.decode = AsyncMock(return_value=torch.zeros(1, 1, 67200))
        engine._resolve_params = MagicMock(return_value=(dict(temperature=0.8, top_p=0.95, top_k=50, frequency_penalty=0.0, seed=7),
                                                         (None, None), "", 0.0, -1))
        engine._resolve_prefix = AsyncMock(return_value=(None, {}))
        engine._n_codebooks = MagicMock(return_value=4)
        engine._cb_ctx = MagicMock(return_value=SimpleNamespace(dc_base_id=100))
        engine._generate_semantic_segment = AsyncMock(side_effect=[([100 + 5000] * 40, False), ([100 + 6000] * 60, True)])

        async def fake_acoustic(state, **_kw):
            for cb in range(3):
                state.ac_segs[cb].append([0] * len(state.cb0_segments[-1]))
        engine._generate_all_acoustic_for_segment = AsyncMock(side_effect=fake_acoustic)
        engine._emit_vv_incremental = AsyncMock(side_effect=[(torch.zeros(1, 1, 48000), 36), (None, 36)])
        engine._append_trailing_silence = MagicMock()

        async def collect():
            return [c async for c in engine.generate_streaming("Some text for the stream.", max_new_tokens=400)]
        with patch("app.inference.codes_to_audio_tensor", return_value=torch.zeros(1, 1, 76800)), \
                patch("app.inference.vv_encode", return_value=torch.zeros(1, 30, 3)):
            asyncio.run(collect())
        calls = engine._generate_semantic_segment.call_args_list
        assert len(calls) == 2 and all(c.kwargs["stream_retry_fix"] is flag for c in calls)


def test_pace_floor_rejects_a_slow_start_only_when_set():
    t = torch.arange(67200) / 24000
    x = torch.zeros(67200)
    x[:24000] = 0.3 * torch.sin(2 * math.pi * 200 * t[:24000])         # 1.0 s of speech in 2.8 s: active share ~0.36
    audio = torch.cat([x, torch.zeros(9600)]).reshape(1, 1, -1)
    for floor, rejected in ((0.0, False), (0.30, False), (0.40, True)):
        config = ServerConfig()
        config.stream_early_check = True
        config.stream_early_pace_min_active = floor
        engine = TTSEngine(config)
        reason = engine._early_chunk_rejection([3716] + [5000 + i for i in range(39)], audio, 21, 60)
        assert (reason is not None) == rejected
        assert reason is None or reason.startswith("slow_start")


def test_flag_defaults_and_env():
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("STREAM_RETRY_FIX", None)
        os.environ.pop("STREAM_EARLY_PACE_MIN_ACTIVE", None)
        c = ServerConfig()
        assert c.stream_retry_fix is True and c.stream_early_pace_min_active == 0.0
    with patch.dict(os.environ, {"STREAM_RETRY_FIX": "0"}):
        assert ServerConfig().stream_retry_fix is False
    with patch.dict(os.environ, {"STREAM_RETRY_FIX": "1", "STREAM_EARLY_PACE_MIN_ACTIVE": "0.4"}):
        c = ServerConfig()
        assert c.stream_retry_fix is True and c.stream_early_pace_min_active == 0.4
