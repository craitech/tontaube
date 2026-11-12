"""/predict quality gate (app/quality_gate.py, PREDICT_QUALITY_GATE).

The checker's frozen constants and rules, the policy on a scripted fake engine, and flag-off parity.
"""
import dataclasses
import inspect
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from app import cb0_ctc
from app import quality_gate as Q
from app.cb0_guard import RateWindow, load_recognizer
from app.config import ServerConfig
from app.inference import TTSEngine

PLAIN = '<|im_start|>english : conversational<|im_end|>\n'
RATE4 = '<|im_start|>english : conversational<|ctrl_rate_04|><|im_end|>\n'
GERMAN = '<|im_start|>german : conversational<|im_end|>\n'
TEXT = 'a' * 100
BASE, EOS = 100, 2
GATE_ENV = ("PREDICT_QUALITY_GATE", "PREDICT_GATE_MAX_ATTEMPTS", "PREDICT_GATE_BUDGET_S", "PREDICT_GATE_POINT",
            "PREDICT_GATE_LOG")


# ---------------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------------
def _config(**kwargs) -> ServerConfig:
    config = ServerConfig()
    config.force_first_silence = False
    config.cb0_rate_aware_cps = config.cb0_ctc_guard = config.cb0_drop_bad_prev_audio = False
    config.predict_quality_gate = True
    for key, value in kwargs.items():
        setattr(config, key, value)
    return config


def _engine(config, outputs, gate=None, recognizer=None) -> TTSEngine:
    engine = TTSEngine(config)
    engine._cb_ctx = MagicMock(return_value=SimpleNamespace(
        tokenizer=MagicMock(), dc_base_id=BASE, pad_id=0, eos_id=EOS, audio_marker_id=3, hz=12.5, engine=MagicMock(),
    ))
    engine._generate_with_engine = AsyncMock(side_effect=list(outputs))
    engine.cb0_ctc = recognizer
    engine.quality_gate = gate
    return engine


def _out(n: int, code: int = 20) -> list[int]:
    return [BASE + code] * n + [EOS]


def _decision(spec) -> Q.Decision:
    """'pass' | rule name | [rules] | ([rules], p)"""
    if spec == 'pass':
        return Q.Decision(passed=True, flags={r: False for r in Q.RULES}, p=0.01)
    fired, p = (spec if isinstance(spec, tuple) else (spec, None))
    fired = [fired] if isinstance(fired, str) else list(fired)
    return Q.Decision(passed=False, flags={r: r in fired for r in Q.RULES}, values={r: 9.0 for r in fired},
                      p=p if p is not None else 0.5)


class ScriptedGate(Q.QualityGate):
    """QualityGate whose checker verdict is scripted per attempt (keyed by the attempt's first local code)."""

    def __init__(self, script, **kwargs):
        super().__init__(MagicMock(), **kwargs)
        self.script = script
        self.calls = []

    def score(self, codes, text, rate_factor=1.0, skip_ctc_on_cap=True):
        self.calls.append((codes[0], rate_factor, text))
        spec = self.script[codes[0]]
        if spec == 'raise':
            raise RuntimeError('cuda error')
        return _decision(spec)


async def _segment(engine, system_tag=PLAIN, text=TEXT, early=False, gate_request=None, seg_idx=0, segments=None,
                   seed=1000, prev_audio=None, prev=None):
    segments = segments or [text]
    kwargs = {'quality_gate': gate_request} if gate_request is not None else {}
    with patch('app.inference.tokenize_text', return_value=[10]), patch(
            'app.inference.SamplingParams', side_effect=lambda **kw: SimpleNamespace(**kw)):
        codes, _ = await engine._generate_semantic_segment(
            seg_idx=seg_idx, segments=segments, prev_chunk_global=prev,
            sampling=dict(temperature=0.8, top_p=0.95, top_k=50, seed=seed),
            voice_prompt=([5, 6, 7], None), prefix=None, max_new_tokens=400,
            system_tag=system_tag, early_mode=early, prev_audio=prev_audio, **kwargs)
    return codes


def _sent(engine):
    """SamplingParams of every CB0 call (SimpleNamespace)."""
    return [c.args[2] for c in engine._generate_with_engine.await_args_list]


def _request(gate, seed=1000, n_pieces=1, serving_max_retries=5):
    return gate.new_request(seed, ([5, 6, 7], None), n_pieces, serving_max_retries)


# ---------------------------------------------------------------------------------------------------------------------
# config / frozen constants
# ---------------------------------------------------------------------------------------------------------------------
def test_flags_default_off_and_env(monkeypatch):
    for name in GATE_ENV:
        monkeypatch.delenv(name, raising=False)
    c = ServerConfig()
    assert (c.predict_quality_gate, c.predict_gate_max_attempts, c.predict_gate_budget_s, c.predict_gate_point,
            c.predict_gate_log) == (False, 4, 2.0, "B", None)
    assert TTSEngine(c).quality_gate is None
    for name, value in zip(GATE_ENV, ("1", "3", "1.5", "c", "/tmp/g.jsonl")):
        monkeypatch.setenv(name, value)
    c = ServerConfig()
    assert (c.predict_quality_gate, c.predict_gate_max_attempts, c.predict_gate_budget_s, c.predict_gate_point,
            c.predict_gate_log) == (True, 3, 1.5, "C", "/tmp/g.jsonl")


def test_frozen_point_b_and_silence_set():
    assert Q.POINTS["B"] == {"R_cap": 0.0, "R_len": 2.2521739130434786, "R_unread": -1000000000.0,
                             "R_cer": 0.6984343434343433, "R_last": -2.4672347784042357, "R_first": -8.855300064086913,
                             "R_cov": 0.9284495192307685}
    assert len(Q.LOGIT_FEATS) == len(Q.LOGISTIC_MEAN) == len(Q.LOGISTIC_SCALE) == len(Q.LOGISTIC_COEF) == 19
    collapse = (3716, 12725, 3974, 4299, 10309, 5995, 6144, 15170, 13601, 398)
    ordinary = (7362, 17, 3710, 1531, 1384, 12293, 5169, 4544)
    assert Q.SIL == set(collapse + ordinary + (15478,))


def test_build_quality_gate(caplog):
    assert Q.build_quality_gate(_config(predict_quality_gate=False), MagicMock()) is None
    assert Q.build_quality_gate(_config(), None) is None                       # no checkpoint: gate off
    with pytest.raises(ValueError, match="PREDICT_GATE_POINT"):
        Q.build_quality_gate(_config(predict_gate_point="D"), MagicMock())
    g = Q.build_quality_gate(_config(predict_gate_point="C", predict_gate_max_attempts=3), MagicMock())
    assert g.point == "C" and g.thresholds == Q.POINTS["C"] and g.max_attempts == 3 and g.budget_s == 2.0


def test_recogniser_loads_for_the_gate_alone(tmp_path):
    model = cb0_ctc.CB0CTCModel(cb0_ctc.CB0CTCConfig(d_model=32, n_layers=1, n_heads=2, d_ff=64, d_up=16))
    torch.save({'config': dataclasses.asdict(model.cfg), 'state_dict': model.state_dict(), 'charset': cb0_ctc.CHARSET},
               tmp_path / 'cb0_ctc.pt')
    off = _config(predict_quality_gate=False, cb0_ctc_model_path=None, cb0_ctc_device='cpu')
    assert load_recognizer(off, str(tmp_path)) is None
    rec = load_recognizer(_config(cb0_ctc_model_path=None, cb0_ctc_device='cpu'), str(tmp_path))
    assert rec is not None and rec.device == 'cpu'
    assert load_recognizer(_config(cb0_ctc_model_path=None), str(tmp_path / 'missing')) is None


# ---------------------------------------------------------------------------------------------------------------------
# checker
# ---------------------------------------------------------------------------------------------------------------------
def test_forward_fp32_equals_the_fp32_forward():
    torch.manual_seed(1)
    m = cb0_ctc.CB0CTCModel(cb0_ctc.CB0CTCConfig(d_model=32, n_layers=2, n_heads=2, d_ff=64, d_up=16))
    codes = [list(range(5, 40)), [16383, 3716, 15478, 7]]
    a = cb0_ctc.CB0CTC(m, device='cpu', dtype=torch.float32)
    lp, ol, vad, _ = a._forward(codes)
    lp2, ol2, vad2 = cb0_ctc.CB0CTC(m, device='cpu', dtype=torch.bfloat16).forward_fp32(codes)
    assert torch.equal(lp, lp2) and torch.equal(ol, ol2) and torch.equal(vad, vad2)


def test_length_rule_covers_short_replies_and_scales_with_the_rate_factor():
    # "Yes." (4 chars): serving's chars/s window skips pieces of <= 30 chars, the gate's length rule does not
    engine = TTSEngine(_config())
    codes = [3716] + [20 + (i % 7) for i in range(99)]           # 8 s of speech codes for a 1-word reply
    assert engine._check_cb0_generation(len(codes), 4, False, 12.5, codes=codes) is None
    f = Q.code_features(codes, "Yes.")
    f.update(cer=0.0, fa_ok=True, hyp_letters=3)
    g = Q.derive(f)
    assert Q.rule_flags(g, Q.POINTS["B"])["R_len"]
    assert g["speech_codes_ratio"] == pytest.approx((99 / 12.5 + 0.5) / (1 + 3 * 3 / 14))
    assert Q.derive(f, 1.3)["speech_codes_ratio"] == pytest.approx(1.3 * g["speech_codes_ratio"])
    short = [3716] + [20] * 14 + [3716] * 3
    assert not Q.rule_flags(Q.derive({**Q.code_features(short, "Yes."), "cer": 0.0, "fa_ok": True}), Q.POINTS["B"])["R_len"]


def test_cap_is_rejected_without_a_ctc_forward():
    rec = MagicMock()
    gate = Q.QualityGate(rec)
    d = gate.score([20] * 400, "Hello there.")
    assert not d.passed and d.flags["R_cap"] and not d.ctc and rec.forward_fp32.call_count == 0


# ---------------------------------------------------------------------------------------------------------------------
# policy on a scripted engine
# ---------------------------------------------------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_accept_first():
    gate = ScriptedGate({20: 'pass'})
    engine = _engine(_config(), [_out(80, 20)], gate)
    codes = await _segment(engine, gate_request=_request(gate))
    assert engine._generate_with_engine.await_count == 1 and codes == [BASE + 20] * 80 and len(gate.calls) == 1


@pytest.mark.asyncio
async def test_retries_until_a_pass_with_fresh_seeds_at_base_temperature():
    gate = ScriptedGate({20: 'R_len', 21: 'R_last', 22: 'pass'})
    engine = _engine(_config(), [_out(80, c) for c in (20, 21, 22)], gate)
    codes = await _segment(engine, gate_request=_request(gate, seed=1000))
    assert codes[0] == BASE + 22
    sent = _sent(engine)
    assert [s.seed for s in sent] == [1000, 1000 + 7919, 1000 + 2 * 7919]
    assert [s.temperature for s in sent] == [0.8] * 3
    cfg = engine.config
    collapse = [s.logit_bias[BASE + 3716] for s in sent]
    assert collapse == [cfg.cb0_silence_logit_bias] * 3                 # a gate retry keeps the base silence bias


@pytest.mark.asyncio
async def test_stops_at_max_scored_attempts_and_keeps_the_best():
    script = {20: (['R_len', 'R_cer'], 0.1), 21: (['R_last'], 0.6), 22: (['R_cap'], 0.0), 23: (['R_cov'], 0.3), 24: 'pass'}
    gate = ScriptedGate(script)
    engine = _engine(_config(), [_out(80, c) for c in (20, 21, 22, 23, 24)], gate)
    codes = await _segment(engine, gate_request=_request(gate))
    assert engine._generate_with_engine.await_count == 4 and codes[0] == BASE + 23   # 1 flag, lowest p; R_cap last


@pytest.mark.asyncio
async def test_serving_failures_do_not_count_and_keep_servings_schedule():
    # too few tokens (serving), reject, serving, reject, reject, reject -> 4 scored after 6 CB0 attempts
    outs = [_out(5, 20), _out(80, 21), _out(5, 22), _out(80, 23), _out(80, 24), _out(80, 25), _out(80, 26)]
    gate = ScriptedGate({21: 'R_len', 23: 'R_len', 24: 'R_len', 25: ('R_len', 0.1), 26: 'pass'})
    engine = _engine(_config(), outs, gate)
    codes = await _segment(engine, gate_request=_request(gate))
    assert engine._generate_with_engine.await_count == 6 and len(gate.calls) == 4 and codes[0] == BASE + 25
    sent = _sent(engine)
    cfg = engine.config
    steps = [0, 1, 1, 2, 2, 2]          # serving's schedule moves only after its own failures (attempts 1 and 3)
    assert [s.temperature for s in sent] == pytest.approx(
        [0.8 + max(0, k - 1) * cfg.cb0_retry_temperature_step for k in steps])
    assert [s.logit_bias[BASE + 3716] for s in sent] == pytest.approx(
        [cfg.cb0_silence_logit_bias + k * cfg.cb0_silence_logit_bias_step for k in steps])
    assert [s.seed for s in sent] == [1000 + 7919 * k for k in range(6)]   # serving's retries get fresh seeds too


@pytest.mark.asyncio
async def test_at_most_eight_cb0_attempts_and_servings_own_cap():
    outs = [_out(5, 20), _out(80, 21), _out(5, 22), _out(80, 23), _out(5, 24), _out(80, 25), _out(5, 26), _out(80, 27),
            _out(80, 28)]
    gate = ScriptedGate({c: 'R_len' for c in range(20, 29)})
    engine = _engine(_config(), outs, gate)
    await _segment(engine, gate_request=_request(gate))
    assert engine._generate_with_engine.await_count == 8                 # cb0_max_retries + 4 - 1
    gate = ScriptedGate({})
    engine = _engine(_config(), [_out(5, 20 + i) for i in range(9)], gate)
    await _segment(engine, gate_request=_request(gate))
    assert engine._generate_with_engine.await_count == 5                 # serving's own failures: today's cap of 5


@pytest.mark.asyncio
async def test_piece_seed_rule_and_no_seed():
    assert Q.seed_for(1000, 0, 3) == 1000
    assert Q.seed_for(1000, 2, 3) == 1000 + 2 * 7919 + 3 * 104729
    assert Q.seed_for(2 ** 31 - 1, 1, 0) == (2 ** 31 - 1 + 7919) % 2 ** 31
    assert Q.seed_for(None, 2, 1) is None
    gate = ScriptedGate({20: 'R_len', 21: 'pass'})
    engine = _engine(_config(), [_out(80, 20), _out(80, 21)], gate)
    req = _request(gate, n_pieces=2)
    await _segment(engine, gate_request=req, seg_idx=1, segments=[TEXT, TEXT])
    assert [s.seed for s in _sent(engine)] == [1000, 1000 + 7919 + 104729]
    gate = ScriptedGate({20: 'R_len', 21: 'pass'})
    engine = _engine(_config(), [_out(80, 20), _out(80, 21)], gate)
    await _segment(engine, gate_request=_request(gate, seed=None), seed=None)
    assert [s.seed for s in _sent(engine)] == [None, None]
    req = _request(gate, seed=1000)
    assert req.begin_pass(1) == 1000 and req.begin_pass(3) == 1000 + 2 * 1299709


@pytest.mark.asyncio
async def test_budget_stops_new_gate_retries():
    gate = ScriptedGate({20: ('R_len', 0.9), 21: ('R_len', 0.2)}, budget_s=0.0)
    engine = _engine(_config(), [_out(80, 20), _out(80, 21)], gate)
    req = _request(gate)
    codes = await _segment(engine, gate_request=req)
    assert engine._generate_with_engine.await_count == 1 and codes[0] == BASE + 20 and req.budget_hit
    gate = ScriptedGate({20: 'R_len', 21: 'pass'}, budget_s=100.0)
    engine = _engine(_config(), [_out(80, 20), _out(80, 21)], gate)
    req = _request(gate)
    await _segment(engine, gate_request=req)
    assert engine._generate_with_engine.await_count == 2 and not req.budget_hit and req.added_s > 0


@pytest.mark.asyncio
async def test_fails_open_on_a_checker_exception():
    gate = ScriptedGate({20: 'raise', 21: 'pass'})
    engine = _engine(_config(), [_out(80, 20), _out(80, 21)], gate)
    codes = await _segment(engine, gate_request=_request(gate))
    assert engine._generate_with_engine.await_count == 1 and codes[0] == BASE + 20


@pytest.mark.asyncio
async def test_no_serving_ok_attempt_keeps_the_best_ranked_not_the_last():
    script = {20: (['R_len', 'R_cer'], 0.9), 21: (['R_cov'], 0.2), 22: (['R_len', 'R_last'], 0.1), 23: 'R_cap', 24: ('R_len', 0.5)}
    gate = ScriptedGate(script)
    engine = _engine(_config(), [_out(5, 20 + i) for i in range(5)], gate)
    codes = await _segment(engine, gate_request=_request(gate))
    assert engine._generate_with_engine.await_count == 5 and codes == [BASE + 21] * 5
    reference = _engine(_config(predict_quality_gate=False), [_out(5, 20 + i) for i in range(5)])
    assert await _segment(reference) == [BASE + 24] * 5                   # today: the last attempt


@pytest.mark.asyncio
async def test_gate_replaces_the_guards_retry_decision():
    rec = MagicMock()
    rec.recognize.side_effect = lambda codes_list: ['zzzz' for _ in codes_list]     # CER 1.0: the guard would retry
    gate = ScriptedGate({20: 'pass'})
    engine = _engine(_config(cb0_ctc_guard=True), [_out(80, 20), _out(80, 21)], gate, recognizer=rec)
    codes = await _segment(engine, gate_request=_request(gate))
    assert engine._generate_with_engine.await_count == 1 and codes[0] == BASE + 20 and rec.recognize.call_count == 0
    # the previous-audio chain still gets the guard's CER of the KEPT attempt
    from app.cb0_guard import PrevAudioChain
    gate = ScriptedGate({20: ('R_len', 0.1), 21: ('R_len', 0.9)}, max_attempts=2)
    rec2 = MagicMock()
    rec2.recognize.side_effect = lambda codes_list: ['a' * (100 if codes_list[0][0] == 20 else 10) for _ in codes_list]
    engine = _engine(_config(cb0_ctc_guard=True), [_out(80, 20), _out(80, 21)], gate, recognizer=rec2)
    chain = PrevAudioChain(0.5)
    codes = await _segment(engine, gate_request=_request(gate), prev_audio=chain)
    assert codes[0] == BASE + 20 and chain.previous == codes                 # kept attempt 1: CER 0 -> passed on


@pytest.mark.asyncio
@pytest.mark.parametrize("tag, early, text, n_tokens", [(GERMAN, False, TEXT, 80), (PLAIN, True, TEXT, 30),
                                                        ('', False, TEXT, 80), (PLAIN, False, '... ' * 10, 40)])
async def test_pass_through_pieces(tag, early, text, n_tokens):
    gate = ScriptedGate({20: 'R_len'})
    engine = _engine(_config(), [_out(n_tokens, 20), _out(n_tokens, 21)], gate)
    await _segment(engine, tag, text=text, early=early, gate_request=_request(gate))
    assert gate.calls == [] and engine._generate_with_engine.await_count == 1


@pytest.mark.asyncio
async def test_rate_slider_scales_the_expected_length():
    gate = ScriptedGate({20: 'pass'}, rate_window=RateWindow({0: 0.70, 4: 1.30}))
    engine = _engine(_config(), [_out(80, 20)], gate)
    await _segment(engine, RATE4, gate_request=_request(gate))
    assert gate.calls[0][1] == pytest.approx(1.30)
    gate.calls.clear()
    engine = _engine(_config(), [_out(80, 20)], gate)
    await _segment(engine, PLAIN, gate_request=_request(gate))
    assert gate.calls[0][1] == 1.0


@pytest.mark.asyncio
async def test_jsonl_log(tmp_path):
    path = tmp_path / "gate.jsonl"
    gate = ScriptedGate({20: ('R_len', 0.7), 21: 'pass'}, log_path=str(path))
    engine = _engine(_config(), [_out(80, 20), _out(80, 21)], gate)
    req = _request(gate)
    await _segment(engine, gate_request=req)
    req.finish(1, 1, [])
    rows = [json.loads(line) for line in open(path)]
    assert [r["type"] for r in rows] == ["piece", "request"]
    p = rows[0]
    assert p["text"] == TEXT and p["request_seed"] == 1000 and p["kept_attempt"] == 2 and p["gate_retries"] == 1
    assert [a["verdict"] for a in p["attempts"]] == ["rejected", "accepted"] and p["attempts"][0]["fired"] == ["R_len"]
    assert p["attempts"][0]["codes"] == [20] * 80 and p["attempts"][1]["seed"] == 1000 + 7919
    assert rows[1]["attempts_total"] == 2 and rows[1]["gate_retries"] == 1


# ---------------------------------------------------------------------------------------------------------------------
# request level (generate) and flag-off parity
# ---------------------------------------------------------------------------------------------------------------------
def _generate_engine(config, gate=None):
    engine = TTSEngine(config)
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.convert_tokens_to_ids = lambda t: {'<|dc_0_0|>': 100, '<|end_of_speech|>': 2, '<|audio_split|>': 3}.get(t, 99)
    engine.tokenizers = [tokenizer, tokenizer]
    engine.engines = [MagicMock(), MagicMock()]
    engine.vv_tokenizer = None
    engine.quality_gate = gate
    return engine


async def _generate(engine, audios):
    passes = iter(range(100))
    sem = AsyncMock(side_effect=lambda *a, **kw: ([1000 + next(passes)] * 30, []))
    engine._generate_semantic = sem
    engine._generate_acoustic = AsyncMock(return_value=[0] * 30)
    with patch("app.inference.decode_chunked_with_context", side_effect=list(audios)):
        audio, tokens = await engine.generate(text="Hello there.", seed=1000)
    return sem, audio, tokens


@pytest.mark.asyncio
async def test_request_level_keeps_the_best_pass_with_fresh_seeds():
    silent = [torch.full((1, 1, 2400), v) for v in (0.0005, 0.0015, 0.001)]     # all RMS <= 0.002 -> 3 failed passes
    gate = ScriptedGate({})
    engine = _generate_engine(_config(cb0_max_retries=3, output_headroom_db=0.0), gate)
    sem, audio, tokens = await _generate(engine, silent)
    assert [c.kwargs["sampling"]["seed"] for c in sem.await_args_list] == [1000, 1000 + 1299709, 1000 + 2 * 1299709]
    assert all(c.kwargs["quality_gate"] is not None for c in sem.await_args_list)
    assert tokens[0][0] == 1001 and torch.equal(audio, silent[1])          # pass 2: the highest RMS
    off = _generate_engine(_config(predict_quality_gate=False, cb0_max_retries=3, output_headroom_db=0.0))
    sem, audio, tokens = await _generate(off, [a.clone() for a in silent])
    assert tokens[0][0] == 1002 and [c.kwargs["sampling"]["seed"] for c in sem.await_args_list] == [1000] * 3


@pytest.mark.asyncio
async def test_flag_off_never_calls_gate_code():
    """PREDICT_QUALITY_GATE off: no gate object, no call into app.quality_gate, calls identical to the gate-free path."""
    boom = MagicMock(side_effect=AssertionError("quality gate code called with the flag off"))
    targets = [(Q, "build_quality_gate"), (Q.QualityGate, "new_request"), (Q.QualityGate, "score"),
               (Q.QualityGate, "score_safe"), (Q.RequestGate, "piece"), (Q.RequestGate, "begin_pass"),
               (Q.RequestGate, "finish"), (Q.PieceGate, "check"), (Q.PieceGate, "finish"), (Q, "seed_for")]
    with patch.multiple("app.inference", build_quality_gate=boom):
        patches = [patch.object(obj, name, boom) for obj, name in targets]
        for p in patches:
            p.start()
        try:
            off = _config(predict_quality_gate=False)
            assert TTSEngine(off).quality_gate is None
            # per piece: today's retries (cps window / too few tokens) with a recogniser present
            rec = MagicMock()
            outputs = [_out(5, 20), _out(45, 21), _out(100, 22)]
            got = _engine(off, outputs, None, recognizer=rec)
            codes = await _segment(got)
            assert codes == [BASE + 22] * 100 and rec.recognize.call_count == 0
            # request level: no quality_gate kwarg, request seed on every pass
            engine = _generate_engine(_config(predict_quality_gate=False, cb0_max_retries=2, output_headroom_db=0.0))
            sem, _, _ = await _generate(engine, [torch.zeros(1, 1, 2400), torch.ones(1, 1, 2400)])
            assert all("quality_gate" not in c.kwargs for c in sem.await_args_list)
        finally:
            for p in patches:
                p.stop()
    assert boom.call_count == 0


@pytest.mark.asyncio
async def test_flag_off_output_identical_to_gate_free_rules():
    """Byte-equal codes and identical CB0 calls (seed, temperature, biases) with the gate off vs the gate-free rules."""
    outputs = [_out(5, 20), _out(450 // 10, 21), _out(80, 22)]
    off = _engine(_config(predict_quality_gate=False), outputs)
    codes = await _segment(off)
    sent = _sent(off)
    cfg = off.config
    assert codes == [BASE + 22] * 80
    # guard fix: retry_seed per attempt, no longer the request seed on every one
    assert [s.seed for s in sent] == [1000 + k * 1_000_003 for k in range(3)]
    assert [s.temperature for s in sent] == pytest.approx([0.8, 0.8, 0.8 + cfg.cb0_retry_temperature_step])
    assert [s.logit_bias[BASE + 3716] for s in sent] == pytest.approx(
        [cfg.cb0_silence_logit_bias + k * cfg.cb0_silence_logit_bias_step for k in range(3)])
    # the same scripted engine through the gate-capable code with no request gate = the same calls
    again = _engine(_config(predict_quality_gate=True), outputs)
    assert await _segment(again) == codes
    assert [vars(s) for s in _sent(again)] == [vars(s) for s in sent]


def test_stream_never_builds_a_gate():
    for fn in (TTSEngine.generate_streaming, TTSEngine._generate_streaming_inner, TTSEngine._resolve_prefix):
        assert "quality_gate" not in inspect.getsource(fn)
    assert "quality_gate" in inspect.getsource(TTSEngine.generate)
    assert inspect.signature(TTSEngine._generate_semantic_segment).parameters["quality_gate"].default is None
