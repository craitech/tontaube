"""VV_ADAPTER_CKPT: residual VibeVoice-latent adapter + silence gate.

Off (default, vv_adapter=None): decode_chunked_with_context is unchanged (test_decode_end_fix.py checks the
end_fix=False path against its frozen copy; here: None == omitted, and VibeVoice off ignores the adapter).
On, with the fake models of test_decode_end_fix.py: the output keeps the off length for every window layout, a
zero correction reproduces the off output exactly (so the frame bookkeeping and the tail fix are right), and the
chunked adapter pass equals one pass. A non-zero correction lands on the right frames: the windowed serving decode
equals the flat reference recipe (one DualCodec decode, one VibeVoice encode, one adapter pass) for 1 to 3003 codes
(1 to 9 windows), also when DualCodec decodes short; a gate or latent shifted by one frame fails that comparison.
Checkpoint identity: VV_ADAPTER_SHA256 is required and must match, the hash must be approved (or
VV_ADAPTER_ALLOW_UNLISTED=1), and TTSEngine.load checks all of it before any model loads.
"""
import contextlib
import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
import torch.nn.functional as F

import app.dualcodec_utils as du
import app.vv_adapter as va
from app.config import ServerConfig
from app.inference import TTSEngine
from tests.test_decode_end_fix import (
    HOP, HZ, LENGTHS, SPT, _codes, fake_dc_decode, fake_vv_decode, fake_vv_encode,
)

D_IN, D_Z = 8, 3  # the fake VibeVoice latent has D = 3
RELEASE_SHA256 = "80543a36669725947891dfe1002e24531f3816edd05c946b5e76861582d98181"
REFERENCE_GATE = va.silence_gate  # kept for the flat reference when a test patches the module's silence_gate
# K = 1 without DECODE_END_FIX has no audio with the adapter off (test_one_code_without_end_fix)
CASES = [(n, end_fix) for n in LENGTHS for end_fix in (True, False) if n > 1 or end_fix]


def fake_dc_latent(_dc, semantic_codes, acoustic_codes):
    """DualCodec latent stand-in: [1, D_IN, n], a function of the codes only (per code, like the real lookup)."""
    x = torch.tensor([semantic_codes] + list(acoustic_codes), dtype=torch.float32)  # [4, n]
    feats = [torch.sin(x[i % 4] * (0.001 * (i + 1))) for i in range(D_IN)]
    return torch.stack(feats)[None]


def fake_dc_levels(_dc, semantic_codes, acoustic_codes, short=0):
    """fake_dc_decode with a level per code from 0 to -90 dB (from the semantic code), so the silence gate takes every
    value from 0 to 1. short > 0: the decode comes back `short` samples short at the end, as the real DualCodec."""
    wav = fake_dc_decode(_dc, semantic_codes, acoustic_codes).reshape(-1, SPT)
    level = 10 ** (-(torch.tensor(semantic_codes, dtype=torch.long) % 91).float() / 20)
    wav = (wav * level[:, None]).reshape(1, 1, -1)
    return wav[..., :wav.shape[-1] - short] if short else wav


def tiny_adapter(zero=False, seed=0):
    torch.manual_seed(seed)
    m = va.ResAdapter(d_in=D_IN, d=16, z_dim=D_Z)
    if not zero:
        torch.nn.init.normal_(m.out.weight, std=0.3)
    return m.eval()


@pytest.fixture
def fakes():
    with patch("app.dualcodec_utils.codes_to_audio_tensor", side_effect=fake_dc_decode), \
            patch("app.dualcodec_utils.vv_encode", side_effect=fake_vv_encode), \
            patch("app.dualcodec_utils.vv_decode_streaming", side_effect=fake_vv_decode), \
            patch("app.vv_adapter.dc_latent", side_effect=fake_dc_latent):
        yield


@contextlib.contextmanager
def level_fakes(short=0):
    """The fakes, with fake_dc_levels as the DualCodec decode."""
    dc = lambda d, s, a: fake_dc_levels(d, s, a, short)  # noqa: E731
    with patch("app.dualcodec_utils.codes_to_audio_tensor", side_effect=dc), \
            patch("app.dualcodec_utils.vv_encode", side_effect=fake_vv_encode), \
            patch("app.dualcodec_utils.vv_decode_streaming", side_effect=fake_vv_decode), \
            patch("app.vv_adapter.dc_latent", side_effect=fake_dc_latent):
        yield


def _decode(sem, ac, context_seconds=6.0, **kw):
    return du.decode_chunked_with_context(object(), object(), sem, ac, hz=HZ, context_seconds=context_seconds, **kw)


def flat_reference_decode(model, sem, ac, short=0, end_fade_ms=15.0):
    """The flat reference recipe on the fakes, with no decode windows at all: one DualCodec decode of the K codes,
    zero-padded to K x 1920 samples and then to 3 * Kp / 5 VibeVoice frames (Kp = 5 * ceil(K / 5)); one VibeVoice
    encode = z_prod; the DualCodec latent of all K codes, edge-padded to Kp; the gate from the padded audio; one adapter
    pass; one streaming decode; cut to K x 1920 samples with the 15 ms fade."""
    k = len(sem)
    kp = -(-k // 5) * 5
    a = fake_dc_levels(None, sem, ac, short)[0, 0][:k * SPT]
    a = F.pad(a, (0, k * SPT - a.numel()))
    al = F.pad(a, (0, 3 * kp // 5 * HOP - a.numel()))[None]
    z_prod = fake_vv_encode(None, al)
    lat = F.pad(fake_dc_latent(None, sem, ac), (0, kp - k), mode="replicate")
    with torch.no_grad():
        z = model(lat, z_prod, gate=REFERENCE_GATE(al, z_prod.shape[1]))
    y, _ = fake_vv_decode(None, z)
    return du.finish_request_end(y, k * SPT, end_fade_ms)


@pytest.mark.parametrize("n,end_fix", CASES)
def test_none_is_off(fakes, n, end_fix):
    sem, ac = _codes(n)
    ref = _decode(sem, ac, end_fix=end_fix)
    assert torch.equal(_decode(sem, ac, end_fix=end_fix, vv_adapter=None), ref)
    pre = du.decode_chunked_with_context(object(), None, sem, ac, hz=HZ, context_seconds=6.0, end_fix=end_fix)
    with_ad = du.decode_chunked_with_context(object(), None, sem, ac, hz=HZ, context_seconds=6.0, end_fix=end_fix,
                                             vv_adapter=tiny_adapter())
    assert torch.equal(with_ad, pre)                      # VibeVoice off: the adapter is ignored


@pytest.mark.parametrize("n,end_fix", CASES)
def test_zero_correction_reproduces_off(fakes, n, end_fix):
    """dz = 0: z = z_prod, so the decode must equal the off decode sample for sample (same length too)."""
    sem, ac = _codes(n)
    off = _decode(sem, ac, end_fix=end_fix)
    on = _decode(sem, ac, end_fix=end_fix, vv_adapter=tiny_adapter(zero=True))
    assert on.shape == off.shape and torch.equal(on, off)
    assert on.shape[-1] == (n * SPT if end_fix else (n * 3 // 5) * HOP)


@pytest.mark.parametrize("n", [5, 13, 101, 104, 376, 452, 751, 1129])
def test_correction_changes_loud_frames_and_keeps_the_length(fakes, n):
    sem, ac = _codes(n)
    off, pre = _decode(sem, ac, end_fix=True, return_pre_vv=True)
    on, pre_on = _decode(sem, ac, end_fix=True, return_pre_vv=True, vv_adapter=tiny_adapter())
    assert on.shape == off.shape and not torch.equal(on, off)
    assert torch.equal(pre_on, pre)                       # the pre-VV audio is never changed


# 1 to 3003 codes = 1 to 9 decode windows: short requests (K < 5, under one VibeVoice frame), each tail case of the
# 5-code grid, both sides of the window edges (375, 750, ...) and the chunked adapter pass (> 750 codes)
FLAT_KS = [1, 2, 4, 5, 6, 374, 376, 749, 751, 752, 753, 754, 1124, 1126, 1501, 1876, 2252, 3003]
FLAT_SEED = 4  # its first codes are loud, so K = 1, 2 and 4 get a non-zero correction too


@pytest.mark.parametrize("k", FLAT_KS)
def test_correction_lands_on_the_right_frames(k):
    """The windowed serving decode with a non-zero correction equals the flat reference recipe: the latent, the gate
    (every level from 0 to 1) and z_prod are joined on one global frame grid, across windows and adapter chunks."""
    sem, ac = _codes(k, seed=FLAT_SEED)
    model = tiny_adapter()
    with level_fakes():
        on = _decode(sem, ac, end_fix=True, vv_adapter=model)
        off = _decode(sem, ac, end_fix=True)
    ref = flat_reference_decode(model, sem, ac)
    assert on.shape == ref.shape == (1, 1, k * SPT)
    torch.testing.assert_close(on, ref, atol=1e-5, rtol=0)
    assert (on - off).abs().max() > 1e-2              # the correction is not trivially zero


@pytest.mark.parametrize("k", [37, 68, 376, 751, 1129, 2252])
def test_correction_lands_on_the_right_frames_when_dualcodec_decodes_short(k):
    """The real DualCodec decode comes back a few samples short: serving zero-pads each window's content as the flat
    reference does."""
    sem, ac = _codes(k, seed=FLAT_SEED)
    model = tiny_adapter()
    with level_fakes(short=SPT):
        on = _decode(sem, ac, end_fix=True, vv_adapter=model)
    ref = flat_reference_decode(model, sem, ac, short=SPT)
    assert on.shape == ref.shape
    torch.testing.assert_close(on, ref, atol=1e-5, rtol=0)


@pytest.mark.parametrize("shift", ["gate", "latent"])
def test_a_shift_by_one_frame_fails_the_flat_comparison(shift):
    """The check above has teeth: a gate shifted by one VibeVoice frame, or a latent shifted by one code, in every
    window, keeps the length and changes the output (so test_correction_changes_loud_frames_and_keeps_the_length still
    passes), but no longer matches the flat reference recipe."""
    sem, ac = _codes(1129, seed=FLAT_SEED)
    model = tiny_adapter()
    ref = flat_reference_decode(model, sem, ac)
    if shift == "gate":
        bad = patch("app.vv_adapter.silence_gate", side_effect=lambda a, t: torch.roll(REFERENCE_GATE(a, t), 1, dims=1))
    else:
        bad = patch("app.vv_adapter.dc_latent", side_effect=lambda d, s, a: torch.roll(fake_dc_latent(d, s, a), 1, -1))
    with level_fakes(), bad:
        on = _decode(sem, ac, end_fix=True, vv_adapter=model)
    assert on.shape == ref.shape
    assert (on - ref).abs().max() > 1e-3


def test_a_context_off_the_5_code_grid_raises(fakes):
    """5.0 s of context = 62 codes: z_prod would slide 640 samples against the latent and the gate in every window
    after the first. The adapter refuses (check_grid); the off path is not affected."""
    sem, ac = _codes(451)
    with pytest.raises(RuntimeError, match="not a multiple of 5"):
        _decode(sem, ac, context_seconds=5.0, end_fix=True, vv_adapter=tiny_adapter())
    assert _decode(sem, ac, context_seconds=5.0, end_fix=True).shape[-1] == 451 * SPT
    for n in (0, 5, 75, 375):
        va.check_grid("x", n)
    for n in (1, 62, 374):
        with pytest.raises(RuntimeError):
            va.check_grid("x", n)


@pytest.mark.parametrize("end_fix", [True, False])
def test_zero_codes_fail_as_with_the_adapter_off(fakes, end_fix):
    """K = 0: the off decode raises (torch.cat of no VibeVoice frames); with the adapter it raises the same error."""
    with pytest.raises(ValueError) as off:
        _decode([], [[], [], []], end_fix=end_fix)
    with patch("app.vv_adapter.logger") as log, pytest.raises(ValueError) as on:
        _decode([], [[], [], []], end_fix=end_fix, vv_adapter=tiny_adapter())
    assert str(on.value) == str(off.value)
    assert not log.warning.called                       # no misleading "adapter skipped" warning


@pytest.mark.parametrize("k", [1, 4])
def test_one_and_four_codes(fakes, k):
    """K < 5, under one VibeVoice frame: the adapter still corrects, keeps the off length, and (end_fix) matches the
    flat recipe (test_correction_lands_on_the_right_frames); without end_fix, K = 4 keeps the off length (2 frames)."""
    sem, ac = _codes(k)
    on = _decode(sem, ac, end_fix=True, vv_adapter=tiny_adapter())
    assert on.shape[-1] == k * SPT
    if k == 4:
        off = _decode(sem, ac, end_fix=False)
        on = _decode(sem, ac, end_fix=False, vv_adapter=tiny_adapter())
        assert on.shape == off.shape and off.shape[-1] == 2 * HOP


def test_one_code_without_end_fix(fakes):
    """K = 1 with DECODE_END_FIX=0 (not the default): the off decode keeps int(0.6) = 0 VibeVoice
    frames and raises (torch.cat of no audio); with the adapter the decode keeps that off length, 0 samples."""
    sem, ac = _codes(1)
    with pytest.raises(ValueError):
        _decode(sem, ac, end_fix=False)
    assert _decode(sem, ac, end_fix=False, vv_adapter=tiny_adapter()).shape[-1] == 0


def test_silent_frames_stay_production(fakes):
    """Gate 0 below -65 dBFS: with a silent DualCodec decode the correction is switched off everywhere."""
    sem, ac = _codes(452)
    silent = lambda dc, s, a: fake_dc_decode(dc, s, a) * 1e-5   # ~ -110 dBFS
    with patch("app.dualcodec_utils.codes_to_audio_tensor", side_effect=silent):
        off = _decode(sem, ac, end_fix=True)
        on = _decode(sem, ac, end_fix=True, vv_adapter=tiny_adapter())
    assert torch.equal(on, off)


def test_silence_gate_levels():
    a = torch.zeros(1, 4 * HOP)
    for j, db in enumerate((-80.0, -65.0, -60.0, -40.0)):
        a[0, j * HOP:(j + 1) * HOP] = 10 ** (db / 20) * torch.sign(torch.randn(HOP))   # rms = db exactly
    g = va.silence_gate(a, 5)[0, :, 0]
    torch.testing.assert_close(g, torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]), atol=1e-4, rtol=0)


@pytest.mark.parametrize("k", [5, 105, 750, 755, 1600])
def test_chunked_pass_equals_one_pass(k):
    m = tiny_adapter()
    torch.manual_seed(k)
    lat, zp = torch.randn(1, D_IN, k), torch.randn(1, 3 * k // 5, D_Z)
    gate = torch.rand(1, 3 * k // 5, 1)
    with torch.no_grad():
        ref = m(lat, zp, gate=gate)
    out = va.run_adapter(m, lat, zp, gate, chunk_codes=150, context_codes=100)
    assert out.shape == ref.shape
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------- checkpoint identity
def _tiny_ckpt(tmp_path, name="g.pt", residual=True):
    m = va.ResAdapter(d=16)
    torch.nn.init.normal_(m.out.weight)
    path = tmp_path / name
    args = {"d": 16, "lookahead": 1, "residual": True} if residual else {"d": 16}
    torch.save({"model": m.state_dict(), "args": args}, path)
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest(), m


def test_the_approved_list_is_the_release_adapter():
    assert set(va.APPROVED_SHA256) == {RELEASE_SHA256}


def test_load_checkpoint(tmp_path, monkeypatch):
    path, sha, m = _tiny_ckpt(tmp_path)
    monkeypatch.setitem(va.APPROVED_SHA256, sha, "test checkpoint")
    loaded = va.load_vv_adapter(path, sha.upper())
    assert all(torch.equal(a, b) for a, b in zip(loaded.state_dict().values(), m.state_dict().values()))
    assert not loaded.training and loaded.sha256 == sha
    assert not any(p.requires_grad for p in loaded.parameters())
    assert va.activate_vv_adapter(loaded, "cpu") is loaded


def test_sha_mismatch_fails(tmp_path):
    path, sha, _ = _tiny_ckpt(tmp_path)
    wrong = RELEASE_SHA256  # an approved hash, but not this file's
    with pytest.raises(va.VVAdapterCheckpointError, match="has sha256 " + sha):
        va.load_vv_adapter(path, wrong)
    with pytest.raises(va.VVAdapterCheckpointError, match="has sha256"):
        va.load_vv_adapter(path, wrong, allow_unlisted=True)   # the override never skips the hash check


@pytest.mark.parametrize("bad", ["", "   ", "0123ab", "z" * 64])
def test_sha_is_required(tmp_path, bad):
    path, _, _ = _tiny_ckpt(tmp_path)
    with pytest.raises(va.VVAdapterCheckpointError, match="VV_ADAPTER_SHA256"):
        va.load_vv_adapter(path, bad, allow_unlisted=True)


def test_missing_file_fails(tmp_path):
    with pytest.raises(va.VVAdapterCheckpointError, match="no such file"):
        va.load_vv_adapter(str(tmp_path / "nope.pt"), RELEASE_SHA256)


def test_unlisted_checkpoint_needs_the_override(tmp_path):
    path, sha, _ = _tiny_ckpt(tmp_path)
    with pytest.raises(va.VVAdapterCheckpointError, match="not an approved"):
        va.load_vv_adapter(path, sha)
    with patch("app.vv_adapter.logger") as log:
        assert va.load_vv_adapter(path, sha, allow_unlisted=True).sha256 == sha
    assert "NOT an approved" in log.warning.call_args.args[0]


def test_non_residual_checkpoint_fails(tmp_path):
    path, sha, _ = _tiny_ckpt(tmp_path, residual=False)
    with pytest.raises(va.VVAdapterCheckpointError, match="not a residual"):
        va.load_vv_adapter(path, sha, allow_unlisted=True)


def test_config_flags(monkeypatch):
    for name in ("VV_ADAPTER_CKPT", "VV_ADAPTER_SHA256", "VV_ADAPTER_ALLOW_UNLISTED"):
        monkeypatch.delenv(name, raising=False)
    cfg = ServerConfig()
    assert (cfg.vv_adapter_ckpt, cfg.vv_adapter_sha256, cfg.vv_adapter_allow_unlisted) == ("", "", False)
    monkeypatch.setenv("VV_ADAPTER_CKPT", " /models/vv_adapter.pt ")
    monkeypatch.setenv("VV_ADAPTER_SHA256", " 0123AB ")
    monkeypatch.setenv("VV_ADAPTER_ALLOW_UNLISTED", "1")
    cfg = ServerConfig()
    assert (cfg.vv_adapter_ckpt, cfg.vv_adapter_sha256, cfg.vv_adapter_allow_unlisted) == (
        "/models/vv_adapter.pt", "0123ab", True)


# ---------------------------------------------------------------- startup wiring (TTSEngine.load)
def _engine(path="", sha="", allow_unlisted=False, vibevoice=True):
    engine = TTSEngine(ServerConfig())
    cfg = engine.config
    cfg.enable_verbalization, cfg.enable_vibevoice = False, vibevoice
    cfg.vv_adapter_ckpt, cfg.vv_adapter_sha256, cfg.vv_adapter_allow_unlisted = path, sha, allow_unlisted
    return engine


@contextlib.contextmanager
def no_models():
    """Every loader TTSEngine.load calls, stubbed: no model files, no GPU."""
    runtime = SimpleNamespace(dualcodec="dc", w2vbert="w2v", vibevoice="vv")
    with patch("app.inference.resolve_codebook_model_paths", return_value=[f"/models/cb{i}" for i in range(4)]), \
            patch("app.inference.resolve_runtime_model_paths", return_value=runtime), \
            patch("app.inference.read_codebook_metadata", side_effect=lambda p: (int(p[-1]), 4)), \
            patch("app.inference.load_tokenizer", return_value=MagicMock()), \
            patch("app.inference.ControlCatalog.load", return_value=None), \
            patch("app.inference.get_dc_inference", return_value=MagicMock()), \
            patch("app.inference.load_recognizer", return_value=None), \
            patch.object(TTSEngine, "_encode_default_prompt", return_value=(None, None)), \
            patch("app.inference.build_codebook_engines", new=AsyncMock(return_value=[MagicMock() for _ in range(4)])), \
            patch("app.inference.load_vibevoice_tokenizer", return_value=SimpleNamespace(device=torch.device("cpu"))), \
            patch("app.inference.VVDecodeEngine", return_value=MagicMock()), \
            patch("app.inference.free_memory"):
        yield


@pytest.mark.asyncio
@pytest.mark.parametrize("problem", ["missing file", "sha mismatch", "no sha", "unlisted", "not residual"])
async def test_engine_load_checks_the_checkpoint_before_any_model(tmp_path, problem):
    path, sha, _ = _tiny_ckpt(tmp_path, residual=problem != "not residual")
    engine = _engine(
        path=str(tmp_path / "missing.pt") if problem == "missing file" else path,
        sha={"sha mismatch": RELEASE_SHA256, "no sha": ""}.get(problem, sha),
        allow_unlisted=problem != "unlisted",
    )
    engine.config.enable_verbalization = True           # the verbalizer used to be the first thing load() loads
    too_late = AssertionError("a model started loading before the adapter checkpoint was checked")
    with patch("app.inference.resolve_verbalizer_model_path", side_effect=too_late), \
            patch("app.inference.resolve_codebook_model_paths", side_effect=too_late):
        with pytest.raises(va.VVAdapterCheckpointError):
            await engine.load()
    assert engine.vv_adapter is None


@pytest.mark.asyncio
async def test_engine_load_rejects_a_context_off_the_grid(tmp_path, monkeypatch):
    path, sha, _ = _tiny_ckpt(tmp_path)
    monkeypatch.setitem(va.APPROVED_SHA256, sha, "test checkpoint")
    engine = _engine(path, sha)
    engine.config.chunk_context_seconds = 5.0
    with patch("app.inference.resolve_codebook_model_paths", side_effect=AssertionError("too late")):
        with pytest.raises(RuntimeError, match="decode context"):
            await engine.load()


@pytest.mark.asyncio
@pytest.mark.parametrize("vibevoice", [True, False])
async def test_engine_load_wires_the_checked_adapter(tmp_path, monkeypatch, vibevoice):
    path, sha, m = _tiny_ckpt(tmp_path)
    monkeypatch.setitem(va.APPROVED_SHA256, sha, "test checkpoint")
    engine = _engine(path, sha, vibevoice=vibevoice)
    with no_models(), patch("app.inference.logger") as log:
        await engine.load()
    if vibevoice:
        assert isinstance(engine.vv_adapter, va.ResAdapter) and engine.vv_adapter.sha256 == sha
        assert not engine.vv_adapter.training
        assert all(torch.equal(a, b) for a, b in zip(engine.vv_adapter.state_dict().values(), m.state_dict().values()))
        with no_models():
            engine.unload()
        assert engine.vv_adapter is None
    else:
        assert engine.vv_adapter is None
        assert any("VibeVoice is disabled" in c.args[0] for c in log.warning.call_args_list)


@pytest.mark.asyncio
async def test_engine_load_without_the_flag_has_no_adapter():
    engine = _engine()
    with no_models(), patch("app.inference.load_vv_adapter", side_effect=AssertionError("loaded")):
        await engine.load()
    assert engine.vv_adapter is None


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", [None, "set"])
async def test_predict_passes_the_adapter(adapter):
    engine = TTSEngine(ServerConfig())
    engine.vv_tokenizer = object()
    engine.dc_inference = object()
    engine.vv_adapter = None if adapter is None else object()
    sampling = dict(temperature=0.6, top_p=0.95, top_k=50, seed=123)
    engine._resolve_params = MagicMock(return_value=(sampling, (None, None), "", 0.0, -1))
    engine._resolve_prefix = AsyncMock(return_value=(None, {}))
    engine._n_codebooks = MagicMock(return_value=4)
    engine._generate_semantic = AsyncMock(return_value=([1] * 100, []))
    engine._generate_acoustic = AsyncMock(return_value=[2] * 100)
    audio = torch.full((1, 1, 2400), 0.1)
    with patch("app.inference.free_memory"), patch(
            "app.inference.decode_chunked_with_context", return_value=audio) as decode:
        await engine.generate("Test text.", temperature=0.6)
    assert decode.call_args.kwargs["vv_adapter"] is engine.vv_adapter
