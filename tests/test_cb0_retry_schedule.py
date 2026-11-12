"""The CB0 retry loop with the guard fix, STREAM_RETRY_FIX and the /predict quality gate combined.

One reseed per attempt: the gate's pieces take the gate's schedule (seed_for), streaming with STREAM_RETRY_FIX keeps
seed + attempt - 1, every other call (gate pass-through pieces, /predict without the gate) takes the guard fix's
retry_seed. A piece the gate keeps without accepting it ships labelled FAIL-SHIPPED (check() returns None on every
stop, so a plain union of the branches never labels it).
"""
from unittest.mock import patch

import pytest

from app.cb0_guard import retry_seed
from tests.test_quality_gate import (BASE, GERMAN, PLAIN, ScriptedGate, _config, _engine, _out, _request, _segment,
                                     _sent)


async def _warnings(engine, **kwargs):
    with patch('app.inference.logger') as log:
        codes = await _segment(engine, **kwargs)
    return codes, [c.args[0] % c.args[1:] for c in log.warning.call_args_list]


def _shipped(warnings):
    return [m for m in warnings if 'FAIL-SHIPPED' in m]


@pytest.mark.asyncio
async def test_gate_pass_through_piece_takes_the_guard_fix_seeds():
    """A request gate is present but the piece is German (pass-through): serving's retries reseed with retry_seed."""
    gate = ScriptedGate({})
    engine = _engine(_config(), [_out(5, 20), _out(5, 21), _out(80, 22)], gate)
    codes = await _segment(engine, system_tag=GERMAN, gate_request=_request(gate, seed=1000))
    assert codes[0] == BASE + 22 and gate.calls == []
    assert [s.seed for s in _sent(engine)] == [retry_seed(1000, k) for k in (1, 2, 3)]


@pytest.mark.asyncio
async def test_gate_piece_takes_the_gate_seeds_even_with_the_stream_flag():
    """Defensive: the gate's schedule wins over STREAM_RETRY_FIX's if both were ever set on one call."""
    gate = ScriptedGate({20: 'R_len', 21: 'pass'})
    engine = _engine(_config(), [_out(80, 20), _out(80, 21)], gate)
    with patch('app.inference.tokenize_text', return_value=[10]), patch(
            'app.inference.SamplingParams', side_effect=lambda **kw: __import__('types').SimpleNamespace(**kw)):
        codes, _ = await engine._generate_semantic_segment(
            seg_idx=0, segments=['a' * 100], prev_chunk_global=None,
            sampling=dict(temperature=0.8, top_p=0.95, top_k=50, seed=1000), voice_prompt=([5, 6, 7], None),
            prefix=None, max_new_tokens=400, system_tag=PLAIN, quality_gate=_request(gate, seed=1000),
            stream_retry_fix=True)
    assert codes[0] == BASE + 21
    assert [s.seed for s in _sent(engine)] == [1000, 1000 + 7919]


@pytest.mark.asyncio
async def test_gate_kept_best_scored_piece_ships_labelled():
    script = {20: (['R_len', 'R_cer'], 0.1), 21: (['R_last'], 0.6), 22: (['R_cap'], 0.0), 23: (['R_cov'], 0.3)}
    gate = ScriptedGate(script)
    engine = _engine(_config(), [_out(80, c) for c in (20, 21, 22, 23)], gate)
    codes, warnings = await _warnings(engine, gate_request=_request(gate))
    assert codes[0] == BASE + 23                                          # the gate's ranking: 1 flag, lowest p
    shipped = _shipped(warnings)
    assert len(shipped) == 1
    assert 'quality_gate best_scored (stop: max scored attempts; fired: R_cov)' in shipped[0]


@pytest.mark.asyncio
async def test_gate_kept_best_of_serving_failures_ships_labelled():
    gate = ScriptedGate({20: 'R_len', 21: 'R_len'})
    engine = _engine(_config(cb0_max_retries=2), [_out(5, 20), _out(5, 21)], gate)
    codes, warnings = await _warnings(engine, gate_request=_request(gate, serving_max_retries=2))
    shipped = _shipped(warnings)
    assert len(shipped) == 1 and 'quality_gate best_of_serving_failed (stop: serving retries spent' in shipped[0]


@pytest.mark.asyncio
async def test_gate_accepted_piece_is_not_labelled():
    gate = ScriptedGate({20: 'R_len', 21: 'pass'})
    engine = _engine(_config(), [_out(80, 20), _out(80, 21)], gate)
    codes, warnings = await _warnings(engine, gate_request=_request(gate))
    assert codes[0] == BASE + 21 and _shipped(warnings) == []


@pytest.mark.asyncio
async def test_gate_off_unchanged_guard_fix_label():
    """No request gate: the guard fix's labelling (the last attempt's serving failure)."""
    engine = _engine(_config(predict_quality_gate=False, cb0_max_retries=2), [_out(5, 20), _out(5, 21)])
    codes, warnings = await _warnings(engine)
    assert codes[0] == BASE + 21
    shipped = _shipped(warnings)
    assert len(shipped) == 1 and 'after 2 attempts' in shipped[0]
    assert [s.seed for s in _sent(engine)] == [retry_seed(1000, k) for k in (1, 2)]


def test_stream_fix_label_names_the_kept_closest_attempt():
    """STREAM_RETRY_FIX keeps the attempt closest to the chars/s window; the FAIL-SHIPPED label names ITS failure."""
    from tests.test_stream_retry_fix import FAILS, _engine as _stream_engine, _run as _stream_run

    with patch('app.inference.logger') as log:
        codes, params, _ = _stream_run(_stream_engine(), FAILS, True)
    warnings = [c.args[0] % c.args[1:] for c in log.warning.call_args_list]
    retries = [m for m in warnings if ' retry 3/' in m]
    shipped = _shipped(warnings)
    assert codes == [122] * 150 and [p.seed for p in params] == [123, 124, 125, 126, 127]
    assert len(retries) == 1 and len(shipped) == 1
    assert shipped[0].endswith(retries[0].split(': ', 1)[1].rsplit(' (temperature', 1)[0])
