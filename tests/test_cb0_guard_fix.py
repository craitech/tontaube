"""CB0 guard fix: (a) a fresh seed per CB0 retry attempt, (b) the whole-piece gap check,
(c) a kept attempt that failed a check is labelled FAIL-SHIPPED, never shipped silently."""
from unittest.mock import patch

import pytest

from app.cb0_guard import RETRY_SEED_STRIDE, longest_run, retry_seed
from app.config import COLLAPSE_SILENCE_TOKEN_IDS, DC_SILENCE_TOKEN, ORDINARY_SILENCE_TOKEN_IDS, ServerConfig
from app.inference import TTSEngine
from tests.test_cb0_guard import BASE, EOS, _config, _engine, _out, _recognizer, _segment

SIL = COLLAPSE_SILENCE_TOKEN_IDS[0]
SIL2 = ORDINARY_SILENCE_TOKEN_IDS[0]


async def _segment_warnings(engine):
    """Run one segment; return (codes, the formatted WARNING messages of app.inference's logger)."""
    with patch('app.inference.logger') as log:
        codes = await _segment(engine)
    return codes, [c.args[0] % c.args[1:] for c in log.warning.call_args_list]


def _gap_out(gap: int, code: int = 20, silence: int = SIL) -> list[int]:
    """80 speech codes, a silence run of ``gap`` codes after the first 50 (out of the early-silence check's reach), then
    30 more speech codes and EOS."""
    return [BASE + code] * 60 + [BASE + silence] * gap + [BASE + code] * 30 + [EOS]


# ---------------------------------------------------------------------------
# (a) one fresh, reproducible seed per retry attempt
# ---------------------------------------------------------------------------

def test_retry_seed_first_attempt_unchanged_retries_distinct_and_reproducible():
    assert retry_seed(123, 1) == 123
    seeds = [retry_seed(123, k) for k in range(1, 6)]
    assert len(set(seeds)) == 5
    assert seeds == [retry_seed(123, k) for k in range(1, 6)]            # deterministic
    assert retry_seed(123, 2) == 123 + RETRY_SEED_STRIDE
    assert all(0 <= s < 2 ** 31 for s in seeds + [retry_seed(2 ** 31 - 1, 5)])
    assert retry_seed(None, 3) is None                                   # no request seed: the engine draws one


@pytest.mark.asyncio
async def test_each_retry_attempt_gets_its_own_seed():
    engine = _engine(_config(cb0_ctc_guard=False), [_out(5, 20 + i) for i in range(4)] + [_out(80, 24)])
    await _segment(engine)                                               # request seed 1
    sent = [c.args[2].seed for c in engine._generate_with_engine.await_args_list]
    assert sent == [retry_seed(1, k) for k in range(1, 6)]
    assert sent[0] == 1 and len(set(sent)) == 5


@pytest.mark.asyncio
async def test_guard_retry_gets_a_new_seed():
    rec = _recognizer({20: 0.8, 21: 0.2})
    engine = _engine(_config(), [_out(80, 20), _out(80, 21)], rec)
    await _segment(engine)
    first, second = (c.args[2] for c in engine._generate_with_engine.await_args_list)
    assert first.seed == 1 and second.seed == retry_seed(1, 2) != 1
    assert first.temperature == second.temperature                       # the first retry's temperature is unchanged


# ---------------------------------------------------------------------------
# (b) whole-piece gap check
# ---------------------------------------------------------------------------

def test_longest_run():
    assert longest_run([], {1}) == 0
    assert longest_run([1, 1, 2, 1, 1, 1, 2], {1}) == 3
    assert longest_run([1, 3, 1, 3], {1, 3}) == 4


@pytest.mark.parametrize('gap, passes', [(31, True), (32, False), (60, False)])
def test_gap_check_over_the_whole_piece(gap, passes):
    engine = TTSEngine(_config(cb0_gap_check=True))
    codes = [20] * 60 + [SIL] * gap + [20] * 30                       # 31 codes = 2.48 s, 32 = 2.56 s at 12.5 Hz
    reason = engine._check_cb0_generation(len(codes), 30, False, 12.5, codes=codes)
    assert (reason is None) == passes
    if not passes:
        assert reason.startswith('long_gap (') and '> 2.5s' in reason


def test_gap_run_mixes_collapse_and_ordinary_silence_codes():
    engine = TTSEngine(_config(cb0_gap_check=True))
    codes = [20] * 60 + [SIL, SIL2] * 20 + [20] * 30
    assert engine._check_cb0_generation(len(codes), 30, False, 12.5, codes=codes).startswith('long_gap')


@pytest.mark.parametrize('count_digital', [False, True])
def test_gap_check_counts_digital_silence_only_with_its_flag(count_digital):
    engine = TTSEngine(_config(cb0_silence_count_digital=count_digital, cb0_gap_check=True))
    codes = [20] * 60 + [DC_SILENCE_TOKEN] * 40 + [20] * 30
    reason = engine._check_cb0_generation(len(codes), 30, False, 12.5, codes=codes)
    assert (reason is not None) == count_digital


def test_gap_check_skips_early_chunks_and_can_be_disabled():
    codes = [20] * 60 + [SIL] * 40 + [20] * 30
    assert TTSEngine(_config(cb0_gap_check=True))._check_cb0_generation(len(codes), 30, True, 12.5, codes=codes,
                                                                       early_mode=True) is None
    engine = TTSEngine(_config(cb0_gap_check=True))
    engine.config.cb0_max_gap_seconds = 0
    assert engine._check_cb0_generation(len(codes), 30, False, 12.5, codes=codes) is None



def test_gap_check_is_off_by_default_and_on_with_its_env_flag(monkeypatch):
    """(a) and (c) are on, (b) is OFF by default. Off, a long gap passes exactly as before the gap check; CB0_GAP_CHECK=1
    turns it on."""
    codes = [20] * 60 + [SIL] * 40 + [20] * 30
    monkeypatch.delenv('CB0_GAP_CHECK', raising=False)
    assert ServerConfig().cb0_gap_check is False
    assert TTSEngine(ServerConfig())._check_cb0_generation(len(codes), 30, False, 12.5, codes=codes) is None
    monkeypatch.setenv('CB0_GAP_CHECK', '1')
    assert ServerConfig().cb0_gap_check is True
    assert TTSEngine(ServerConfig())._check_cb0_generation(len(codes), 30, False, 12.5,
                                                           codes=codes).startswith('long_gap')

@pytest.mark.asyncio
async def test_gap_attempt_is_retried_like_the_other_checks():
    engine = _engine(_config(cb0_ctc_guard=False, cb0_gap_check=True), [_gap_out(40, 20), _gap_out(10, 21)])
    codes = await _segment(engine)
    assert engine._generate_with_engine.await_count == 2 and codes[0] == BASE + 21


# ---------------------------------------------------------------------------
# (c) no silent shipping of a rejected piece
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_all_attempts_fail_serving_check_ships_last_labelled():
    engine = _engine(_config(cb0_ctc_guard=False, cb0_gap_check=True), [_gap_out(40, 20 + i) for i in range(5)])
    codes, warnings = await _segment_warnings(engine)
    assert codes[0] == BASE + 24                                         # the design's fallback: the last attempt
    shipped = [m for m in warnings if 'FAIL-SHIPPED' in m]
    assert len(shipped) == 1 and 'after 5 attempts' in shipped[0] and 'long_gap (' in shipped[0]


@pytest.mark.asyncio
async def test_guard_exhausted_ships_lowest_cer_labelled():
    rec = _recognizer({20: 0.7, 21: 0.6, 22: 0.9})
    engine = _engine(_config(), [_out(80, c) for c in (20, 21, 22)], rec)
    codes, warnings = await _segment_warnings(engine)
    assert engine._generate_with_engine.await_count == 3 and codes[0] == BASE + 21   # design: lowest CER kept
    shipped = [m for m in warnings if 'FAIL-SHIPPED' in m]
    assert len(shipped) == 1 and 'ctc_cer (0.60 > 0.5)' in shipped[0]


@pytest.mark.asyncio
@pytest.mark.parametrize('guard_on', [False, True])
async def test_passing_piece_is_not_labelled(guard_on):
    rec = _recognizer({20: 0.8, 21: 0.1})
    outputs = [_out(5, 19), _out(80, 20), _out(80, 21)] if guard_on else [_out(5, 19), _out(80, 21)]
    engine = _engine(_config(cb0_ctc_guard=guard_on), outputs, rec if guard_on else None)
    codes, warnings = await _segment_warnings(engine)
    assert codes[0] == BASE + 21
    assert not any('FAIL-SHIPPED' in m for m in warnings)
