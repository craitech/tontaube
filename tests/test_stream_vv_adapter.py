"""STREAM_VV_ADAPTER: the VibeVoice adapter (VV_ADAPTER_CKPT, residual adapter + silence gate) on /stream.

correct_stream_frames: the frames [a, b) it returns equal frames [a, b) of ONE adapter pass over the whole request (/predict's
recipe, tail fix included), for early-chunk, segment, off-grid, long (chunked) and flush ranges, once the stream holds the
adapter's lookahead; a range without it raises, and a one-frame shift fails the comparison.
The stream (the real _generate_streaming_inner / _emit_vv_incremental with the fake models of test_decode_end_fix.py, whose
DualCodec decode, VibeVoice encode and latent are per code / per frame): with the adapter, the streamed audio equals /predict's
decode_chunked_with_context with the adapter on the same codes (early chunk or none, 1 to 3 segments, > 30 s, a continuation
prefix); without DECODE_END_FIX it keeps the adapter-off stream's length. A zero correction reproduces the adapter-off stream
exactly. STREAM_VV_ADAPTER=0 or no VV_ADAPTER_CKPT: the pre-flag stream, the adapter is never called. The release flags
(STREAM_LEAD_TRIM, STREAM_EARLY_CHECK, STREAM_RETRY_FIX) act as before on the corrected audio. Startup: the stream pad must
hold the adapter's lookahead.
"""
import asyncio
import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
import torch.nn.functional as F

import app.dualcodec_utils as du
import app.vv_adapter as va
from app.config import ServerConfig
from app.inference import TTSEngine
from app.stream_lead_trim import trim_stream_lead
from tests.test_decode_end_fix import HOP, HZ, SPT, fake_vv_decode, fake_vv_encode
from tests.test_vv_adapter import _engine, _tiny_ckpt, fake_dc_latent, fake_dc_levels, no_models, tiny_adapter

BASE = 100
EARLY = 40  # the early CB0 call's codes: 2.8 s + the 5-code pad


def _sem(i):
    return (i * 37 + 11) % 16000  # levels 0 to -90 dB in fake_dc_levels: the gate takes every value


def _ac(cb, i):
    return (i * (cb + 3) * 13 + 5 * cb) % 1024


def _codes(n, start=0):
    return [_sem(i) for i in range(start, start + n)], [[_ac(cb, i) for i in range(start, start + n)] for cb in range(3)]


# ---------------------------------------------------------------- correct_stream_frames vs one pass over the request
def one_pass(model, sem, ac):
    """/predict's recipe on the fakes (the flat reference decode up to the adapter): the request's K codes, the latent
    edge-padded to Kp, the audio cut / padded to K x 1920 and then to 3 Kp / 5 frames, one encode, one gate, ONE
    adapter pass."""
    k = len(sem)
    kp = -(-k // 5) * 5
    al = va.stream_tail_audio(fake_dc_levels(None, sem, ac), k)
    z_prod = fake_vv_encode(None, al)
    lat = F.pad(fake_dc_latent(None, sem, ac), (0, kp - k), mode="replicate")
    with torch.no_grad():
        return model(lat, z_prod, gate=va.silence_gate(al.reshape(1, -1), z_prod.shape[1]))


def stream_frames(model, sem, ac, n_held, a, b, final=False):
    """correct_stream_frames as the stream calls it when it holds the first n_held codes."""
    s, c = sem[:n_held], [x[:n_held] for x in ac]
    audio = fake_dc_levels(None, s, c)
    if final:
        audio = va.stream_tail_audio(audio, n_held)
    with patch("app.vv_adapter.dc_latent", side_effect=fake_dc_latent):
        return va.correct_stream_frames(model, object(), s, c, audio, fake_vv_encode(None, audio), a, b, final=final)


RANGES = [  # (request codes, codes held, frames a, b)
    (400, 40, 0, 21),       # the early chunk: 2.8 s = 21 frames, 40 codes
    (400, 65, 21, 36),      # the first segment chunk after it
    (400, 70, 22, 38),      # off the 3-frame grid on both sides
    (400, 36, 0, 21),       # exactly the lookahead held (1 code past frame 20's codes)
    (1600, 1030, 600, 615), # deep into a long request: the pass starts CONTEXT_CODES before frame 600
    (1600, 1035, 601, 616), # deep and off the 3-frame grid: the pass must still start on the 5-code grid
    (1600, 1505, 0, 900),   # a range longer than CHUNK_CODES: run_adapter's chunks
]


@pytest.mark.parametrize("k,held,a,b", RANGES)
def test_stream_frames_equal_one_pass_over_the_request(k, held, a, b):
    model = tiny_adapter()
    sem, ac = _codes(k)
    ref = one_pass(model, sem, ac)[:, a:b]
    out = stream_frames(model, sem, ac, held, a, b)
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)
    assert (out - fake_vv_encode(None, va.stream_tail_audio(fake_dc_levels(None, sem, ac), k))[:, a:b]).abs().max() > 1e-2


@pytest.mark.parametrize("k,a", [(103, 36), (100, 51), (1, 0), (4, 0), (751, 300)])
def test_the_flush_equals_one_pass_with_the_tail_fix(k, a):
    model = tiny_adapter()
    sem, ac = _codes(k)
    b = 3 * (-(-k // 5) * 5) // 5
    torch.testing.assert_close(stream_frames(model, sem, ac, k, a, b, final=True), one_pass(model, sem, ac)[:, a:b],
                               atol=1e-5, rtol=1e-5)
    with pytest.raises(RuntimeError, match="tail fix"):
        stream_frames(model, sem, ac, k, a, b - 1, final=True)


def test_a_range_past_the_lookahead_raises():
    model = tiny_adapter()
    sem, ac = _codes(100)
    with pytest.raises(RuntimeError, match="reads 36 codes, the stream holds 35"):
        stream_frames(model, sem, ac, 35, 0, 21)


@pytest.mark.parametrize("shift", [-1, 1])
def test_a_shift_by_one_frame_fails_the_comparison(shift):
    model = tiny_adapter()
    sem, ac = _codes(400)
    ref = one_pass(model, sem, ac)[:, 21 + shift:36 + shift]
    assert not torch.allclose(stream_frames(model, sem, ac, 65, 21, 36), ref, atol=1e-3)


def test_check_stream_lookahead():
    model = tiny_adapter()
    va.check_stream_lookahead(model, 5)
    va.check_stream_lookahead(model, 1)
    with pytest.raises(RuntimeError, match="lookahead"):
        va.check_stream_lookahead(model, 0)


# ---------------------------------------------------------------- the stream on the fake models
class FakeVVEngine:
    """VVDecodeEngine on fake_vv_decode: one causal decoder state per slot."""

    def __init__(self):
        self.state, self.frames = {}, []

    def alloc_slot(self):
        return 0

    def free_slot(self, slot):
        self.state.pop(slot, None)

    async def decode(self, frames, slot):
        self.frames.append(frames)
        y, self.state[slot] = fake_vv_decode(None, frames, self.state.get(slot))
        return y


def stream_engine(seg_lengths, early=True, prefix=False, adapter=None, flag=True, end_fix=True, n_codebooks=4,
                  release=False, early_codes=None):
    """A TTSEngine whose CB0 / CB1-3 calls return the fixed codes of _codes (segment 0 forced to start with the early
    codes, as the real call), with the real early chunk, _emit_vv_incremental and flush. release=False: trim, early check
    and headroom off, so the inner stream's audio is the decode itself."""
    config = ServerConfig()
    config.decode_end_fix, config.decode_end_fade_ms = end_fix, 15.0
    config.cb0_drop_bad_prev_audio = False
    config.stream_vv_adapter = flag
    if not release:
        config.stream_lead_trim = config.stream_early_check = False
        config.output_headroom_db = 0.0
    engine = TTSEngine(config)
    engine.vv_tokenizer, engine.dc_inference = object(), object()
    engine.vv_engine = FakeVVEngine()
    engine.vv_adapter = adapter
    engine._resolve_params = MagicMock(return_value=(dict(temperature=0.8, top_p=0.95, top_k=50, frequency_penalty=0.0,
                                                          seed=7), (None, None), "", 0.0, -1))
    pfx = None
    if prefix:
        pfx = {"semantic_codes": [5] * 12, "semantic_global": [BASE + 5] * 12}
        engine._resolve_prefix = AsyncMock(return_value=(pfx, {cb: [7] * 12 for cb in (1, 2, 3)}))
    else:
        engine._resolve_prefix = AsyncMock(return_value=(None, {}))
    engine._n_codebooks = MagicMock(return_value=n_codebooks)
    engine._cb_ctx = MagicMock(return_value=SimpleNamespace(dc_base_id=BASE))
    total = sum(seg_lengths)
    sem, _ = _codes(total)
    early_list = list(early_codes or []) + [sem[:EARLY] if early else sem[:3]]  # 3 codes: too short for a chunk
    starts = [sum(seg_lengths[:i]) for i in range(len(seg_lengths))]

    async def cb0(seg_idx=0, early_mode=False, forced_prefix=None, **_kw):
        if early_mode:
            return [BASE + c for c in early_list.pop(0)], False
        codes = sem[starts[seg_idx]:starts[seg_idx] + seg_lengths[seg_idx]]
        if forced_prefix:
            assert codes[:len(forced_prefix)] == list(forced_prefix)
        return [BASE + c for c in codes], seg_idx == len(seg_lengths) - 1

    async def acoustic(state, early_forced_prefixes=None, **_kw):
        for cb in range(state.n_codebooks - 1):
            at = len(state.accumulated_ac[cb])
            codes = [_ac(cb, at + i) for i in range(len(state.cb0_segments[-1]))]
            if early_forced_prefixes:
                assert codes[:len(early_forced_prefixes[cb])] == early_forced_prefixes[cb]
            state.ac_segs[cb].append(codes)
            state.accumulated_ac[cb].extend(codes)

    engine._generate_semantic_segment = AsyncMock(side_effect=cb0)
    engine._generate_all_acoustic_for_segment = AsyncMock(side_effect=acoustic)
    engine.test_texts = [f"Segment {i}, long enough for the early check." for i in range(len(seg_lengths))]
    return engine, _codes(total)


@contextlib.contextmanager
def stream_fakes():
    with patch("app.inference.codes_to_audio_tensor", side_effect=fake_dc_levels), \
            patch("app.inference.vv_encode", side_effect=fake_vv_encode), \
            patch("app.vv_adapter.dc_latent", side_effect=fake_dc_latent):
        yield


def run_stream(engine, inner=True):
    async def collect():
        if inner:
            return [c async for c in engine._generate_streaming_inner(engine.test_texts, max_new_tokens=4000, vv_slot=0)]
        return [c async for c in engine.generate_streaming(engine.test_texts, max_new_tokens=4000)]
    with stream_fakes():
        return asyncio.run(collect())


def predict_decode(sem, ac, adapter, end_fix=True):
    with patch("app.dualcodec_utils.codes_to_audio_tensor", side_effect=fake_dc_levels), \
            patch("app.dualcodec_utils.vv_encode", side_effect=fake_vv_encode), \
            patch("app.dualcodec_utils.vv_decode_streaming", side_effect=fake_vv_decode), \
            patch("app.vv_adapter.dc_latent", side_effect=fake_dc_latent):
        return du.decode_chunked_with_context(object(), object(), sem, ac, hz=HZ, context_seconds=6.0,
                                              end_fix=end_fix, end_fade_ms=15.0, vv_adapter=adapter)


LAYOUTS = [  # (segment lengths, early chunk, continuation prefix)
    ([100], True, False),
    ([100], False, False),
    ([60, 80, 45], True, False),
    ([63, 41, 7], False, False),
    ([300, 280], True, False),   # 46 s: two /predict decode windows
    ([90, 50], True, True),
]


@pytest.mark.parametrize("segs,early,prefix", LAYOUTS)
def test_the_stream_with_the_adapter_equals_predict_on_the_same_codes(segs, early, prefix):
    model = tiny_adapter()
    engine, (sem, ac) = stream_engine(segs, early=early, prefix=prefix, adapter=model)
    chunks = run_stream(engine)
    assert len(chunks) >= 2
    out = torch.cat(chunks, dim=-1)
    ref = predict_decode(sem, ac, model)
    assert out.shape == ref.shape == (1, 1, len(sem) * SPT)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
    off = torch.cat(run_stream(stream_engine(segs, early=early, prefix=prefix, adapter=None)[0]), dim=-1)
    assert (out - off).abs().max() > 1e-2     # the correction is not a no-op


@pytest.mark.parametrize("segs,early", [([100], True), ([63, 41, 7], False), ([300, 280], True)])
def test_without_end_fix_the_stream_keeps_its_adapter_off_length(segs, early):
    model = tiny_adapter()
    engine, (sem, ac) = stream_engine(segs, early=early, adapter=model, end_fix=False)
    out = torch.cat(run_stream(engine), dim=-1)
    off = torch.cat(run_stream(stream_engine(segs, early=early, adapter=None, end_fix=False)[0]), dim=-1)
    assert out.shape == off.shape
    ref = predict_decode(sem, ac, model, end_fix=False)
    n = min(out.shape[-1], ref.shape[-1])
    torch.testing.assert_close(out[..., :n], ref[..., :n], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("segs,early,end_fix", [([100], True, True), ([63, 41, 7], False, True), ([60, 80, 45], True, False)])
def test_a_zero_correction_reproduces_the_adapter_off_stream(segs, early, end_fix):
    on = run_stream(stream_engine(segs, early=early, adapter=tiny_adapter(zero=True), end_fix=end_fix)[0])
    off = run_stream(stream_engine(segs, early=early, adapter=None, end_fix=end_fix)[0])
    assert [c.shape for c in on] == [c.shape for c in off]
    assert all(torch.equal(a, b) for a, b in zip(on, off))


@pytest.mark.parametrize("why", ["flag off", "no adapter", "3-level codes"])
def test_off_is_the_pre_flag_stream(why):
    adapter = None if why == "no adapter" else tiny_adapter()
    n_cb = 3 if why == "3-level codes" else 4
    engine, _ = stream_engine([60, 80, 45], adapter=adapter, flag=why != "flag off", n_codebooks=n_cb)
    with patch("app.inference.correct_stream_frames", side_effect=AssertionError("adapter called")), \
            patch("app.inference.stream_tail_audio", side_effect=AssertionError("tail fix called")):
        out = run_stream(engine)
    ref_engine, _ = stream_engine([60, 80, 45], adapter=None, flag=False, n_codebooks=n_cb)
    ref = run_stream(ref_engine)
    assert len(out) == len(ref) and all(torch.equal(a, b) for a, b in zip(out, ref))
    # the decoder got the encoder's frames themselves (slices of the same frames, untouched)
    assert all(torch.equal(a, b) for a, b in zip(engine.vv_engine.frames, ref_engine.vv_engine.frames))


def test_the_release_flags_act_on_the_corrected_audio():
    """Release config (STREAM_LEAD_TRIM, STREAM_EARLY_CHECK, STREAM_RETRY_FIX and the headroom at their defaults): a silent
    early chunk is still rejected and re-sampled on the DualCodec audio, and the trim + headroom apply to the corrected
    chunks exactly as to the uncorrected ones."""
    model = tiny_adapter()
    silent = [3716] + [12794] * 39
    engine, _ = stream_engine([100, 60], adapter=model, release=True, early_codes=[silent])
    cfg = engine.config
    assert cfg.stream_lead_trim and cfg.stream_early_check and cfg.stream_retry_fix and cfg.stream_vv_adapter
    out = run_stream(engine, inner=False)
    early_calls = [c for c in engine._generate_semantic_segment.call_args_list if c.kwargs.get("early_mode")]
    assert len(early_calls) == 2                                        # the silent start was re-sampled
    plain, _ = stream_engine([100, 60], adapter=model)
    inner = run_stream(plain)
    gain = du.headroom_gain(cfg.output_headroom_db)
    expected = [trim_stream_lead(inner[0], padding_ms=cfg.stream_lead_trim_padding_ms)] + inner[1:]
    assert len(out) == len(expected)
    for a, b in zip(out, expected):
        torch.testing.assert_close(a, b * gain, atol=1e-6, rtol=1e-6)


def test_config_flag(monkeypatch):
    monkeypatch.delenv("STREAM_VV_ADAPTER", raising=False)
    assert ServerConfig().stream_vv_adapter is True
    monkeypatch.setenv("STREAM_VV_ADAPTER", "0")
    assert ServerConfig().stream_vv_adapter is False
    monkeypatch.setenv("STREAM_VV_ADAPTER", "1")
    assert ServerConfig().stream_vv_adapter is True


# ---------------------------------------------------------------- startup (TTSEngine.load)
@pytest.mark.asyncio
@pytest.mark.parametrize("flag", [True, False])
async def test_load_checks_the_stream_pad_against_the_lookahead(tmp_path, monkeypatch, flag):
    path, sha, _ = _tiny_ckpt(tmp_path)
    monkeypatch.setitem(va.APPROVED_SHA256, sha, "test checkpoint")
    engine = _engine(path, sha)
    engine.config.stream_vv_adapter = flag
    engine.config.streaming_dc_pad_tokens = 0
    if flag:
        with patch("app.inference.resolve_codebook_model_paths", side_effect=AssertionError("too late")):
            with pytest.raises(RuntimeError, match="lookahead"):
                await engine.load()
    else:
        with no_models():
            await engine.load()
        assert engine.vv_adapter is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", [True, False])
async def test_load_logs_where_the_adapter_runs(tmp_path, monkeypatch, flag):
    path, sha, _ = _tiny_ckpt(tmp_path)
    monkeypatch.setitem(va.APPROVED_SHA256, sha, "test checkpoint")
    engine = _engine(path, sha)
    engine.config.stream_vv_adapter = flag
    with no_models(), patch("app.vv_adapter.logger") as log:
        await engine.load()
    line = [c for c in log.info.call_args_list if c.args[0].startswith("VibeVoice adapter on")][0]
    assert line.args[-1] == ("/predict and /stream" if flag else "/predict only, /stream unchanged (STREAM_VV_ADAPTER=0)")


@pytest.mark.parametrize("final", [False, True])
@pytest.mark.parametrize("n,a,b", [(1030, 600, 615), (400, 210, 228)])
def test_a_decode_window_from_first_code_equals_the_whole_stream(n, a, b, final):
    """The lite backend's bounded streaming decode passes audio and z_prod from code first_code on: sliced from the
    same tensors, the corrected frames equal the whole-stream call's exactly."""
    model = tiny_adapter()
    sem, ac = _codes(n)
    if final:
        b = 3 * (-(-n // 5) * 5) // 5
    audio = fake_dc_levels(None, sem, ac)
    if final:
        audio = va.stream_tail_audio(audio, n)
    z_prod = fake_vv_encode(None, audio)
    s = max(0, va.GRID_CODES * (a // 3) - va.CONTEXT_CODES)

    def frames(first):
        return va.correct_stream_frames(model, object(), sem, ac, audio[..., first * va.DC_HOP:],
                                        z_prod[:, 3 * first // 5:], a, b, final=final, first_code=first)

    with patch("app.vv_adapter.dc_latent", side_effect=fake_dc_latent):
        full = va.correct_stream_frames(model, object(), sem, ac, audio, z_prod, a, b, final=final)
        for first in (0, s // 10 * 5, s):
            torch.testing.assert_close(frames(first), full, atol=0, rtol=0)
        with pytest.raises(RuntimeError, match="decode window starts"):
            frames(s + 5)
