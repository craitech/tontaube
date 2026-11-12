"""CB0 retry check: rate-aware chars/s window and CB0-CTC guard (app/cb0_guard.py)."""
import asyncio
import dataclasses
import itertools
import json
import math
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from app import cb0_ctc
from app.cb0_guard import (
    RateWindow,
    SegmentGuard,
    interpolate_factor,
    load_factor_table,
    load_recognizer,
    parse_cb0_header,
)
from app.config import DEFAULT_CB0_RATE_CPS_FACTORS, ServerConfig
from app.inference import TTSEngine

PLAIN = '<|im_start|>english : audiobook<|im_end|>\n'
RATE0 = '<|im_start|>english : audiobook<|ctrl_rate_00|><|im_end|>\n'
RATE4 = '<|im_start|>english : audiobook<|ctrl_rate_04|><|ctrl_pause_02|><|im_end|>\n'
GERMAN = '<|im_start|>german : audiobook<|im_end|>\n'
TEXT = 'a' * 100           # normalises to 100 letters: hyp 'a' * k has CER (100 - k) / 100
BASE, EOS = 100, 2


def _config(**kwargs) -> ServerConfig:
    config = ServerConfig()
    config.force_first_silence = False
    # opt-in flags on, whatever the env says (tests that need them off say so)
    config.cb0_rate_aware_cps = config.cb0_ctc_guard = config.cb0_drop_bad_prev_audio = True
    # the values these tests were written for: guard CER 0.5, the built-in rate factors
    config.cb0_ctc_max_cer, config.cb0_rate_cps_factors_json = 0.5, None
    for key, value in kwargs.items():
        setattr(config, key, value)
    return config


def _engine(config: ServerConfig, outputs: list[list[int]], recognizer=None) -> TTSEngine:
    engine = TTSEngine(config)
    engine._cb_ctx = MagicMock(return_value=SimpleNamespace(
        tokenizer=MagicMock(), dc_base_id=BASE, pad_id=0, eos_id=EOS, audio_marker_id=3,
        hz=12.5, engine=MagicMock(),
    ))
    engine._generate_with_engine = AsyncMock(side_effect=outputs)
    engine.cb0_ctc = recognizer
    return engine


def _out(n: int, code: int = 20) -> list[int]:
    """n CB0 tokens of one (non-silence) local code, then EOS."""
    return [BASE + code] * n + [EOS]


def _recognizer(cer_by_code: dict[int, float]):
    """Fake CB0-CTC: the transcript's CER vs TEXT is looked up from the attempt's code."""
    rec = MagicMock()
    rec.recognize.side_effect = lambda codes_list: [
        'a' * round(100 * (1 - cer_by_code[codes[0]])) for codes in codes_list
    ]
    return rec


async def _segment(engine, system_tag=PLAIN, text=TEXT, early=False):
    with patch('app.inference.tokenize_text', return_value=[10]), patch(
            'app.inference.SamplingParams', side_effect=lambda **kwargs: SimpleNamespace(**kwargs)):
        codes, _ = await engine._generate_semantic_segment(
            seg_idx=0, segments=[text], prev_chunk_global=None,
            sampling=dict(temperature=0.8, top_p=0.95, top_k=50, seed=1),
            voice_prompt=(None, None), prefix=None, max_new_tokens=400,
            system_tag=system_tag, early_mode=early)
    return codes


# ---------------------------------------------------------------------------
# Rate-aware chars/s window
# ---------------------------------------------------------------------------

def test_header_parsing():
    assert parse_cb0_header(RATE4) == ('english', 'audiobook', 4)
    assert parse_cb0_header(PLAIN) == ('english', 'audiobook', None)
    assert parse_cb0_header('<|im_start|>german : conversational<|ctrl_pause_00|><|im_end|>\n') == (
        'german', 'conversational', None)
    assert parse_cb0_header('') == (None, None, None)


def test_default_factors_interpolate_log_linearly():
    window = RateWindow(DEFAULT_CB0_RATE_CPS_FACTORS)
    factors = [window.factor('audiobook', b) for b in range(5)]
    step = (math.log(1.30) - math.log(0.70)) / 4
    assert factors == pytest.approx([math.exp(math.log(0.70) + b * step) for b in range(5)])
    assert factors == pytest.approx([0.70, 0.817, 0.954, 1.114, 1.30], abs=1e-3)
    assert window.factor('audiobook', None) == 1.0          # no rate slider
    assert window.factor(None, 6) == pytest.approx(1.30)    # beyond the table: nearest end
    assert interpolate_factor({}, 2) is None


@pytest.mark.parametrize('n_tokens, scale, passes', [
    (50, 1.0, False),     # 25 chars/s: outside 8-20
    (50, 1.30, True),     # rate 4 window 10.4-26
    (45, 1.30, False),    # 27.8 chars/s
    (100, 1.0, True),     # 12.5 chars/s
    (100, 0.70, True),    # rate 0 window 5.6-14
    (80, 0.70, False),    # 15.6 chars/s
])
def test_window_scaling(n_tokens, scale, passes):
    engine = TTSEngine(_config())
    reason = engine._check_cb0_generation(n_tokens, 100, False, 12.5, cps_scale=scale)
    assert (reason is None) == passes
    if not passes:
        assert reason.startswith('cps_out_of_range')


def test_factor_table_json_per_style_with_fallbacks(tmp_path):
    path = tmp_path / 'rate_window_factors.json'
    path.write_text(json.dumps({
        '_meta': {'built_by': 'test'},
        'model_a': {
            'all': {'none': {'factor': 1.0, 'ci95': [1, 1], 'n': None},
                    'rate0': {'factor': 0.72}, 'rate4': {'factor': 1.28}},
            'audiobook': {'none': {'factor': 1.0}, 'rate0': {'factor': 0.65}, 'rate2': 0.9,
                          'rate4': {'factor': 1.35}},
            'various': {'none': {'factor': 1.0}},
        },
        '_checks': {},
    }))
    window = RateWindow(DEFAULT_CB0_RATE_CPS_FACTORS, load_factor_table(str(path)))
    assert window.factor('audiobook', 0) == 0.65
    assert window.factor('audiobook', 3) == pytest.approx(math.sqrt(0.9 * 1.35))
    assert window.factor('conversational', 4) == 1.28   # style missing: pooled "all"
    assert window.factor('various', 0) == 0.72          # style without bins: pooled "all"
    assert window.factor('audiobook', None) == 1.0
    config = _config(cb0_rate_cps_factors_json=str(path), cb0_rate_cps_factors_model='model_a')
    assert TTSEngine(config).rate_window.factor('audiobook', 4) == 1.35


def test_factor_table_needs_model_when_ambiguous(tmp_path):
    path = tmp_path / 'f.json'
    path.write_text(json.dumps({'m1': {'all': {'rate0': 0.7}}, 'm2': {'all': {'rate0': 0.8}}}))
    with pytest.raises(ValueError, match='CB0_RATE_CPS_FACTORS_MODEL'):
        load_factor_table(str(path))
    assert load_factor_table(str(path), 'm2') == {'all': {0: 0.8}}
    with pytest.raises(ValueError, match='not in'):
        load_factor_table(str(path), 'm3')


FLAGS = {'CB0_RATE_AWARE_CPS': 'cb0_rate_aware_cps', 'CB0_CTC_GUARD': 'cb0_ctc_guard',
         'CB0_RATE_AWARE_CHUNK_CHARS': 'cb0_rate_aware_chunk_chars',
         'CB0_DROP_BAD_PREV_AUDIO': 'cb0_drop_bad_prev_audio',
         'CB0_SILENCE_COUNT_DIGITAL': 'cb0_silence_count_digital'}


def test_flag_defaults_and_env(monkeypatch):
    for name in (*FLAGS, 'CB0_RATE_CPS_FACTORS', 'CB0_RATE_CPS_FACTORS_JSON', 'CB0_RATE_CPS_FACTORS_MODEL'):
        monkeypatch.delenv(name, raising=False)
    config = ServerConfig()
    assert [name for name, field in FLAGS.items() if not getattr(config, field)] == ['CB0_RATE_AWARE_CHUNK_CHARS']
    assert config.cb0_rate_cps_factors == {0: 0.70, 4: 1.30}
    assert config.cb0_rate_cps_factors_json.endswith('app/cb0_rate_window_factors.json')
    engine = TTSEngine(config)
    assert engine.rate_window.factor('audiobook', 0) == pytest.approx(0.92) and engine.rate_chunk_window is None
    for name in FLAGS:
        monkeypatch.setenv(name, '0')
    assert not any(getattr(ServerConfig(), field) for field in FLAGS.values())
    for name in FLAGS:
        monkeypatch.setenv(name, '1')
    monkeypatch.setenv('CB0_RATE_CPS_FACTORS', '0:0.6,2:0.9,4:1.4')
    config = ServerConfig()
    assert all(getattr(config, field) for field in FLAGS.values())
    assert config.cb0_rate_cps_factors == {0: 0.6, 2: 0.9, 4: 1.4}
    config.cb0_rate_cps_factors_json = None
    assert TTSEngine(config).rate_window.factor('audiobook', 2) == 0.9


def test_merge_flag_defaults():
    import app.config as config_module
    if os.environ.get('MERGE_SHORT_PIECES') or os.environ.get('MERGE_SHORT_CLIENT_SEGMENTS'):
        pytest.skip('merge flags set in the environment')
    assert config_module.MERGE_SHORT_PIECES is True
    assert config_module.MERGE_SHORT_CLIENT_SEGMENTS is False


@pytest.mark.parametrize('count_digital', [False, True])
def test_digital_silence_counts_only_with_its_flag(count_digital):
    from app.config import DC_SILENCE_TOKEN, SILENCE_TOKEN_IDS
    assert DC_SILENCE_TOKEN == 15478 and DC_SILENCE_TOKEN not in SILENCE_TOKEN_IDS
    engine = TTSEngine(_config(cb0_silence_count_digital=count_digital))
    codes = [3716] + [DC_SILENCE_TOKEN] * 44 + [20] * 55     # a runaway: 88 % digital silence early
    reason = engine._check_cb0_generation(len(codes), 120, False, 12.5, codes=codes)
    assert (reason is not None and reason.startswith('too_much_silence')) == count_digital
    # ordinary speech with a little digital silence passes either way
    speech = [DC_SILENCE_TOKEN] * 5 + [20] * 95
    assert engine._check_cb0_generation(len(speech), 120, False, 12.5, codes=speech) is None


@pytest.mark.asyncio
async def test_rate_slider_widens_window_in_generation():
    # 25 chars/s fails 8-20 but passes the rate-4 window 10.4-26.
    outputs = [_out(45), _out(50, 21), _out(100, 22)]
    engine = _engine(_config(), list(outputs))
    codes = await _segment(engine, RATE4)
    assert engine._generate_with_engine.await_count == 2 and codes[0] == BASE + 21
    engine = _engine(_config(), list(outputs))
    codes = await _segment(engine, PLAIN)                       # no slider: fixed window
    assert engine._generate_with_engine.await_count == 3 and codes[0] == BASE + 22


# ---------------------------------------------------------------------------
# CB0-CTC guard
# ---------------------------------------------------------------------------

def test_vendored_cer():
    assert cb0_ctc.cer('the 2 cats', 'The two cats!') == 0.0
    assert cb0_ctc.cer('a' * 40, TEXT) == pytest.approx(0.6)
    assert cb0_ctc.normalize_text('In 1984 Dr. Smith paid $3.50 & left.') == (
        'in nineteen eighty four dr smith paid three dollars fifty cents and left')
    # a number directly followed by a comma is kept
    assert cb0_ctc.normalize_text('In 1984, he') == 'in nineteen eighty four he'
    assert cb0_ctc.normalize_text('Pay $1,000, then 1,2,3.') == 'pay one thousand dollars then one two three'


@pytest.mark.asyncio
async def test_guard_accepts_first_attempt_below_threshold():
    rec = _recognizer({20: 0.1})
    engine = _engine(_config(), [_out(80)], rec)
    codes = await _segment(engine)
    assert engine._generate_with_engine.await_count == 1 and rec.recognize.call_count == 1
    assert codes == [BASE + 20] * 80


@pytest.mark.asyncio
async def test_guard_retries_until_cer_passes():
    rec = _recognizer({20: 0.8, 21: 0.2})
    engine = _engine(_config(), [_out(80, 20), _out(80, 21)], rec)
    codes = await _segment(engine)
    assert engine._generate_with_engine.await_count == 2 and codes[0] == BASE + 21


@pytest.mark.asyncio
async def test_guard_keeps_lowest_cer_after_two_guard_retries():
    rec = _recognizer({20: 0.7, 21: 0.6, 22: 0.9, 23: 0.0})
    engine = _engine(_config(), [_out(80, c) for c in (20, 21, 22, 23)], rec)
    codes = await _segment(engine)
    # attempt 3 fails the guard with its 2 retries spent: stop, keep attempt 2 (CER 0.6)
    assert engine._generate_with_engine.await_count == 3 and codes[0] == BASE + 21


@pytest.mark.asyncio
async def test_guard_only_scores_serving_ok_attempts_and_keeps_them_over_a_failed_last():
    rec = _recognizer({20: 0.7})
    outputs = [_out(80, 20)] + [_out(5, 21)] * 4                  # then 4 x too few tokens
    engine = _engine(_config(), outputs, rec)
    codes = await _segment(engine)
    assert engine._generate_with_engine.await_count == 5 and rec.recognize.call_count == 1
    assert codes[0] == BASE + 20                                  # serving's rule would keep the last


@pytest.mark.asyncio
async def test_guard_without_any_serving_ok_attempt_keeps_last():
    rec = _recognizer({})
    engine = _engine(_config(), [_out(5, 20 + i) for i in range(5)], rec)
    codes = await _segment(engine)
    assert rec.recognize.call_count == 0 and codes == [BASE + 24] * 5


@pytest.mark.asyncio
@pytest.mark.parametrize('tag, early, text, n_tokens', [
    (GERMAN, False, TEXT, 80),          # recogniser is English-only
    (PLAIN, True, TEXT, 30),            # early streaming chunk covers part of the text
    ('', False, TEXT, 80),              # no header: language unknown
    (PLAIN, False, '... ' * 10, 40),    # no letters
])
async def test_guard_skipped(tag, early, text, n_tokens):
    rec = _recognizer({20: 0.9})
    engine = _engine(_config(), [_out(n_tokens)], rec)
    await _segment(engine, tag, text=text, early=early)
    assert rec.recognize.call_count == 0 and engine._generate_with_engine.await_count == 1


@pytest.mark.asyncio
async def test_guard_fails_open_when_recogniser_raises():
    rec = MagicMock()
    rec.recognize.side_effect = RuntimeError('cuda error')
    engine = _engine(_config(), [_out(80)], rec)
    codes = await _segment(engine)
    assert engine._generate_with_engine.await_count == 1 and codes == [BASE + 20] * 80


def _reference_decide(cap, tries, att, max_attempts=5, guard_cer=0.5):
    """The guard's decision rule as a plain function (reference)."""
    k, ok, cer = tries[-1]
    if cap is None:
        return k if ok or att == max_attempts else None
    if ok and cer <= guard_cer:
        return k
    used = sum(1 for _, o, c in tries[:-1] if o and c > guard_cer)
    oks = [(c, kk) for kk, o, c in tries if o]
    if (ok and used >= cap) or att == max_attempts:
        return min(oks)[1] if oks else k
    return None


@pytest.mark.asyncio
async def test_guard_matches_reference_decision_rule_on_all_attempt_sequences():
    # per attempt: serving check fails, or passes with CTC CER 0.3 / 0.7 / 0.9
    options = [None, 0.3, 0.7, 0.9]
    for cap in (1, 2):
        for seq in itertools.product(options, repeat=5):
            rec = _recognizer({20 + i: c for i, c in enumerate(seq) if c is not None})
            outputs = [_out(80, 20 + i) if c is not None else _out(5, 20 + i) for i, c in enumerate(seq)]
            engine = _engine(_config(cb0_ctc_max_retries=cap), outputs, rec)
            codes = await _segment(engine)
            tries, expected = [], None
            for att, c in enumerate(seq, 1):
                tries.append((f'{att:02d}', c is not None, c if c is not None else 1.0))
                expected = _reference_decide(cap, tries, att)
                if expected is not None:
                    break
            assert engine._generate_with_engine.await_count == len(tries), (cap, seq)
            assert codes[0] - BASE - 20 == int(expected) - 1, (cap, seq)


# ---------------------------------------------------------------------------
# Flags off: today's behaviour
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_flags_off_parity():
    off = _config(cb0_rate_aware_cps=False, cb0_ctc_guard=False)
    rec = _recognizer({20: 0.9, 21: 0.9, 22: 0.9})
    outputs = [_out(45, 20), _out(50, 21), _out(100, 22)]
    flagged = _engine(off, list(outputs), rec)                  # recogniser present, flag off
    reference = _engine(_config(), list(outputs), None)         # today's path: no slider, no guard
    assert flagged.rate_window is None
    got = await _segment(flagged, RATE4)
    want = await _segment(reference, PLAIN)
    assert got == want == [BASE + 22] * 100
    assert rec.recognize.call_count == 0
    sent = [c.args[1:] for c in flagged._generate_with_engine.await_args_list]
    assert sent == [c.args[1:] for c in reference._generate_with_engine.await_args_list]
    for n in (40, 50, 80, 100, 200):
        assert flagged._check_cb0_generation(n, 100, False, 12.5) == \
            reference._check_cb0_generation(n, 100, False, 12.5, cps_scale=1.0)


def test_flags_off_does_not_load_recogniser(tmp_path):
    assert load_recognizer(_config(cb0_ctc_guard=False), str(tmp_path)) is None
    # flag on but no checkpoint anywhere: guard off with a warning, serving unchanged
    assert load_recognizer(_config(cb0_ctc_model_path=None), str(tmp_path)) is None


def test_recogniser_loads_from_cb0_bundle_dir(tmp_path):
    model = cb0_ctc.CB0CTCModel(cb0_ctc.CB0CTCConfig(d_model=32, n_layers=1, n_heads=2, d_ff=64, d_up=16))
    torch.save({'config': dataclasses.asdict(model.cfg), 'state_dict': model.state_dict(), 'charset': cb0_ctc.CHARSET},
               tmp_path / 'cb0_ctc.pt')
    rec = load_recognizer(_config(cb0_ctc_model_path=None, cb0_ctc_device='cpu'), str(tmp_path))
    assert rec.device == 'cpu'
    hyp = rec.recognize([[1, 2, 3, 4], [5, 6]])
    assert len(hyp) == 2 and all(isinstance(h, str) for h in hyp)
    assert asyncio.run(SegmentGuard(rec, 'hello there', 0.5, 2).score([1, 2, 3]))[0] >= 0.0


# ---------------------------------------------------------------------------
# Rate-aware chunk length (scales for the text splitter)
# ---------------------------------------------------------------------------

def _slider_engine(**kwargs) -> TTSEngine:
    from app.controls import ControlCatalog, ControlSpec
    engine = TTSEngine(_config(**{"cb0_rate_aware_chunk_chars": True, **kwargs}))
    engine.controls = ControlCatalog([ControlSpec("rate", 5), ControlSpec("pause", 5)])
    return engine


def test_chunk_char_scales_follow_each_segments_rate_bin():
    engine = _slider_engine()
    assert engine.chunk_char_scales(2, "english", "audiobook", True, controls={"rate": 0}) == \
        pytest.approx([0.70, 0.70])
    scales = engine.chunk_char_scales(
        3, "english", "audiobook", True, controls={"rate": 0},
        segment_controls=[{"rate": 4}, None, "<|ctrl_rate_01|>"])
    assert scales == pytest.approx([1.0, 0.70, 0.817], abs=1e-3)    # faster bins capped at x1
    assert engine.chunk_char_scales(1, "english", "audiobook", True, controls={"pause": 0}) == [1.0]
    assert engine.chunk_char_scales(2, "english", "audiobook", True) is None          # no sliders
    assert engine.chunk_char_scales(1, "english", "audiobook", True, controls={"rate": 9}) is None


def test_chunk_char_scales_flag_off_and_independent_of_cps_flag():
    assert _slider_engine(cb0_rate_aware_chunk_chars=False).chunk_char_scales(
        1, "english", "audiobook", True, controls={"rate": 0}) is None
    engine = _slider_engine(cb0_rate_aware_cps=False)
    assert engine.rate_window is None
    assert engine.chunk_char_scales(1, "english", "audiobook", True, controls={"rate": 0}) == \
        pytest.approx([0.70])


# ---------------------------------------------------------------------------
# Previous-audio hygiene (CB0_DROP_BAD_PREV_AUDIO)
# ---------------------------------------------------------------------------

WORDS = ('alpha', 'bravo', 'charlie', 'delta')
SEG_TEXTS = [w + ' ' + 'a' * 100 for w in WORDS]      # the word survives the 50-char lookahead cut


def _tokenize(text, _tokenizer):
    """Fake tokenizer: one marker id per segment word in the window (lookahead is cut to 50 chars)."""
    return [10] + [1000 + i for i, w in enumerate(WORDS) if w in text]


def _has_prev(call, code: int) -> bool:
    """Whether a CB0 call's input carries previous audio made of ``code``."""
    return BASE + code in call.args[1]


def _has_text(call, seg: int) -> bool:
    """Whether a CB0 call's text window holds segment ``seg``'s text."""
    return 1000 + seg in call.args[1]


def _sampling():
    return dict(temperature=0.8, top_p=0.95, top_k=50, seed=1)


async def _predict_semantic(engine, n_segments=3):
    with patch('app.inference.tokenize_text', side_effect=_tokenize) as tok, patch(
            'app.inference.SamplingParams', side_effect=lambda **kwargs: SimpleNamespace(**kwargs)):
        codes, _ = await engine._generate_semantic(
            SEG_TEXTS[:n_segments], sampling=_sampling(), voice_prompt=(None, None),
            max_new_tokens=400, system_tag=PLAIN, use_context=True)
    engine.windows = [c.args[0] for c in tok.call_args_list]   # CB0 text windows, in order
    return codes


async def _stream_semantic(engine, n_segments=3):
    engine._resolve_params = MagicMock(return_value=(_sampling(), (None, None), PLAIN, 0.0, -1))
    engine._resolve_prefix = AsyncMock(return_value=(None, {}))
    engine._generate_all_acoustic_for_segment = AsyncMock()
    engine._emit_vv_incremental = AsyncMock(return_value=(None, 0))
    engine.engines = [MagicMock() for _ in range(4)]
    with patch('app.inference.tokenize_text', side_effect=_tokenize) as tok, patch(
            'app.inference.SamplingParams', side_effect=lambda **kwargs: SimpleNamespace(**kwargs)):
        async for _ in engine._generate_streaming_inner(SEG_TEXTS[:n_segments], max_new_tokens=400):
            pass
    engine.windows = [c.args[0] for c in tok.call_args_list]


def _restart_window(engine, seg_idx, n_segments):
    return engine._text_window(seg_idx, SEG_TEXTS[:n_segments], use_context=True,
                               system_tag=PLAIN, restart=True)[0]


def test_restart_window_is_segment_0_shaped_at_i():
    from app.config import TEXT_MARKER
    engine = TTSEngine(_config())
    segs = ['First one.', 'Second one.', 'Third one is longer than fifty characters in total, really.',
            'Fourth.']
    restart, start, end = engine._text_window(1, segs, use_context=True, restart=True)
    assert (start, end) == (1, 3)
    # segment i + the trimmed lookahead of i+1, internal-chunk cue (leading space), nothing of i-1
    assert restart == ' Second one.' + TEXT_MARKER + ' Third one is longer than fifty characters in'
    context = engine._text_window(1, segs, use_context=True)[0]
    assert context == 'First one.' + TEXT_MARKER + restart                # the context window minus i-1
    # lookahead into the last segment keeps its end-of-request cue, as in the context window
    assert engine._text_window(2, segs, use_context=True, restart=True)[0] == \
        engine._text_window(2, segs, use_context=True)[0].split(TEXT_MARKER, 1)[1] == \
        ' ' + segs[2] + TEXT_MARKER + ' Fourth.\n'
    # last segment: no lookahead, end-of-request cue kept
    assert engine._text_window(3, segs, use_context=True, restart=True)[0] == ' Fourth.\n'
    # unchanged: plain independent segments, and segment 0
    assert engine._text_window(1, segs, use_context=False)[0] == ' Second one.'
    assert engine._text_window(1, segs, use_context=False, restart=True)[0] == ' Second one.'
    assert engine._text_window(0, segs, use_context=True, restart=True) == \
        engine._text_window(0, segs, use_context=True)
    tagged = engine._text_window(1, segs, use_context=True, system_tag=PLAIN, restart=True)[0]
    assert tagged == PLAIN + restart


def test_chain_has_no_earlier_segment_fallback():
    from app.cb0_guard import PrevAudioChain
    chain = PrevAudioChain(0.5)
    assert not chain.drops_context(0)
    chain.add([1], 0.1)
    assert chain.previous == [1] and not chain.drops_context(1)
    chain.add([2], 0.9)                       # failed: no previous audio, not segment 0's
    assert chain.previous is None and chain.drops_context(2)
    chain.add([3], None)                      # unscored counts as passed
    assert chain.previous == [3] and not chain.drops_context(3)
    chain.add([4], 0.5)
    assert chain.previous == [4]


# seg 0 reads fine; seg 1 runs away (empty transcript) on all 3 guard attempts; segs 2-3 read fine
RUNAWAY_CHAIN = ({30: 0.0, 40: 1.0, 41: 1.0, 42: 1.0, 50: 0.0, 60: 0.0},
                 [_out(80, 30), _out(80, 40), _out(80, 41), _out(80, 42), _out(80, 50), _out(80, 60)])


@pytest.mark.asyncio
@pytest.mark.parametrize('path', ['predict', 'stream'])
async def test_failed_segment_restarts_next_without_previous_audio_and_text(path):
    cers, outputs = RUNAWAY_CHAIN
    early = [[EOS]] if path == 'stream' else []                 # empty early streaming chunk
    engine = _engine(_config(), early + outputs, _recognizer(cers))
    await (_predict_semantic(engine, 4) if path == 'predict' else _stream_semantic(engine, 4))
    calls = engine._generate_with_engine.await_args_list[len(early):]
    assert len(calls) == 6
    seg1, seg2 = calls[1:4], calls[4]
    assert _has_prev(seg1[0], 30) and _has_text(seg1[0], 0)        # first attempt: full context
    for retry in seg1[1:]:                                         # non-speech retries: chunk start
        assert not _has_prev(retry, 30) and not _has_text(retry, 0)
        assert _has_text(retry, 1) and _has_text(retry, 2)         # segment i + lookahead of i+1
    # seg 2 after the failed seg 1: no previous audio (no fallback to seg 0), no seg-1 text,
    # its own text and seg 3's lookahead
    assert not any(_has_prev(seg2, c) for c in (30, 40, 41, 42))
    assert not _has_text(seg2, 1) and _has_text(seg2, 2) and _has_text(seg2, 3)
    # the windows: seg 1's retry rebuild and seg 2's drop path are both the restart window
    windows = engine.windows[1 if path == 'stream' else 0:]       # skip the early chunk's window
    assert windows[2] == _restart_window(engine, 1, 4)            # seg 1, rebuilt for the retry
    assert windows[3] == _restart_window(engine, 2, 4)            # seg 2, chain drop path
    assert len(windows) == 5 and windows[1] == engine._text_window(
        1, SEG_TEXTS[:4], use_context=True, system_tag=PLAIN)[0]  # seg 1 first attempt: context


@pytest.mark.asyncio
@pytest.mark.parametrize('path', ['predict', 'stream'])
async def test_prev_audio_flag_off_parity(path):
    cers, outputs = RUNAWAY_CHAIN
    early = [[EOS]] if path == 'stream' else []
    flagged = _engine(_config(cb0_drop_bad_prev_audio=False), early + outputs, _recognizer(cers))
    run = _predict_semantic if path == 'predict' else _stream_semantic
    await run(flagged)
    calls = flagged._generate_with_engine.await_args_list[len(early):]
    seg1, seg2 = calls[1:4], calls[4]
    assert all(_has_prev(c, 30) and _has_text(c, 0) for c in seg1)   # retries keep the context
    assert _has_prev(seg2, 40) and _has_text(seg2, 1)                # seg 2 gets seg 1's kept attempt
    # guard off as well: the same CB0 inputs as before this change
    off = _engine(_config(cb0_drop_bad_prev_audio=False, cb0_ctc_guard=False),
                  early + [_out(80, 30), _out(80, 40), _out(80, 50)], _recognizer(cers))
    ref = _engine(_config(), early + [_out(80, 30), _out(80, 40), _out(80, 50)], None)
    await run(off)
    await run(ref)
    assert [c.args[1:] for c in off._generate_with_engine.await_args_list] == \
        [c.args[1:] for c in ref._generate_with_engine.await_args_list]


@pytest.mark.asyncio
async def test_serving_failed_runaway_retries_like_a_chunk_start():
    # seg 1 attempt 1 runs to max tokens with no speech: caught by serving's check, not the guard
    cers = {30: 0.0, 40: 1.0, 41: 0.1, 50: 0.0}
    outputs = [_out(80, 30), [BASE + 40] * 400, _out(80, 41), _out(80, 50)]
    engine = _engine(_config(), outputs, _recognizer(cers))
    await _predict_semantic(engine)
    calls = engine._generate_with_engine.await_args_list
    assert _has_prev(calls[1], 30) and _has_text(calls[1], 0)
    assert not _has_prev(calls[2], 30) and not _has_text(calls[2], 0)
    assert _has_prev(calls[3], 41) and _has_text(calls[3], 1)       # seg 1's kept attempt passed


@pytest.mark.asyncio
async def test_guard_failure_with_speech_keeps_previous_audio_and_text():
    # CER 0.97 but a 3-char transcript: a misread, not non-speech -> retry with the same context
    cers = {30: 0.0, 40: 0.97, 41: 0.2, 50: 0.0}
    engine = _engine(_config(), [_out(80, c) for c in (30, 40, 41, 50)], _recognizer(cers))
    await _predict_semantic(engine)
    calls = engine._generate_with_engine.await_args_list
    assert _has_prev(calls[1], 30) and _has_prev(calls[2], 30) and _has_text(calls[2], 0)
    assert _has_prev(calls[3], 41)


@pytest.mark.asyncio
async def test_exhausted_runaway_segment_is_scored_and_next_restarts():
    # seg 1: 5 attempts that all fail serving's check; the kept last one is non-speech
    cers = {30: 0.0, **{40 + i: 1.0 for i in range(5)}, 50: 0.0}
    outputs = [_out(80, 30)] + [_out(5, 40 + i) for i in range(5)] + [_out(80, 50)]
    engine = _engine(_config(), outputs, _recognizer(cers))
    codes = await _predict_semantic(engine)
    calls = engine._generate_with_engine.await_args_list
    assert len(calls) == 7 and _has_prev(calls[1], 30) and not _has_prev(calls[2], 30)
    assert not any(_has_prev(calls[6], c) for c in (30, 44)) and not _has_text(calls[6], 1)
    assert codes.count(44) == 5                                    # seg 1 still keeps its last attempt


@pytest.mark.asyncio
async def test_first_segment_failing_restarts_second():
    cers = {30: 1.0, 31: 1.0, 32: 1.0, 50: 0.0}
    engine = _engine(_config(), [_out(80, c) for c in (30, 31, 32, 50)], _recognizer(cers))
    await _predict_semantic(engine, n_segments=2)
    seg1 = engine._generate_with_engine.await_args_list[3]
    assert not any(_has_prev(seg1, c) for c in (30, 31, 32))
    assert not _has_text(seg1, 0) and _has_text(seg1, 1)
    # the last segment restarts with its own text only and keeps the end-of-request cue
    assert engine.windows[-1] == PLAIN + ' ' + SEG_TEXTS[1] + '\n' == _restart_window(engine, 1, 2)


def test_guard_defaults():
    config = ServerConfig()
    assert (config.cb0_ctc_max_cer, config.cb0_ctc_max_retries, config.cb0_ctc_digit_max_cer,
            config.cb0_ctc_chain_max_cer) == (0.25, 2, 0.5, 0.5)


@pytest.mark.asyncio
@pytest.mark.parametrize("cers, calls, kept", [
    ([0.24], 1, 0),                # accepted
    ([0.35, 0.05], 2, 1),          # the retry is clean
    ([0.35, 0.30, 0.28], 3, 2),    # 2 retries, then keep the lowest
])
async def test_guard_budget_at_the_defaults(cers, calls, kept):
    rec = _recognizer({20 + i: c for i, c in enumerate(cers)})
    engine = _engine(_config(cb0_ctc_max_cer=0.25), [_out(80, 20 + i) for i in range(len(cers))], rec)
    codes = await _segment(engine)
    assert engine._generate_with_engine.await_count == calls
    assert codes[0] == BASE + 20 + kept


def test_digit_segments_use_the_digit_threshold():
    engine = TTSEngine(_config(cb0_ctc_max_cer=0.25))
    engine.cb0_ctc = MagicMock()
    assert engine._cb0_segment_checks(PLAIN, 'Room 4B is on the 3rd floor.', False)[1].max_cer == 0.5
    assert engine._cb0_segment_checks(PLAIN, 'Room four is on the third floor.', False)[1].max_cer == 0.25


@pytest.mark.asyncio
async def test_digits_count_four_chars_in_the_rate_check():
    text = 'In 1987, the population was 12,450; by 2023 it had grown to 31,200.'  # 67 chars, 18 digits
    engine = _engine(_config(cb0_ctc_guard=False), [_out(131)] * 2)  # 10.5 s: 6.4 raw, 11.5 weighted chars/s
    await _segment(engine, text=text)
    assert engine._generate_with_engine.await_count == 1


def test_control_catalog(tmp_path):
    from app.controls import ControlCatalog

    (tmp_path / 'controls.json').write_text(json.dumps({'specs': [
        {'name': 'rate', 'n_bins': 5, 'description': 'speaking rate', 'low_description': 'much slower',
         'high_description': 'much faster'},
        {'name': 'pitch_level', 'n_bins': 3},
    ]}))
    assert ControlCatalog.load(str(tmp_path)).catalog() == [
        {'name': 'rate', 'bins': 5, 'description': 'speaking rate', 'low': 'much slower', 'high': 'much faster'},
        {'name': 'pitch_level', 'bins': 3, 'description': '', 'low': '', 'high': ''},
    ]
