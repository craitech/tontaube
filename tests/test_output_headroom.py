"""OUTPUT_HEADROOM_DB: fixed output gain after the decode.

Default 3.6 dB (x0.6607). 0 must give the pre-flag output bit-exactly on /predict and /stream (the pre-flag
generate() returned the decode output unchanged, and generate_streaming() yielded the inner chunks unchanged).
The gain comes after DECODE_END_FIX and the silence retry check, and before trim / MossFormer2 / encoding.
"""
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

import app.dualcodec_utils as du
import app.main as app_main
import app.mossformer2_postprocessor as mossformer_module
from app.config import DUALCODEC_SAMPLE_RATE, ServerConfig
from app.inference import TTSEngine
from app.main import TTSRequest

HZ = 12.5
SPT = 1920  # samples per DualCodec token at 24 kHz
HOP = 3200  # samples per VibeVoice frame
GAIN_36 = 10.0 ** (-3.6 / 20.0)


# --- deterministic stand-ins for DualCodec and VibeVoice (shapes as the real models) -------------------------
def fake_dc_decode(_dc, semantic_codes, acoustic_codes):
    sem = torch.tensor(semantic_codes, dtype=torch.float32)
    ac = torch.tensor(acoustic_codes[0], dtype=torch.float32) if acoustic_codes else torch.zeros_like(sem)
    t = torch.arange(SPT, dtype=torch.float32) / SPT
    wav = torch.sin(2 * math.pi * (sem[:, None] * 0.013 + ac[:, None] * 0.007 + 3.0 * t[None, :]))
    return (0.9 * wav).reshape(1, 1, -1)


def fake_vv_encode(_vv, audio):
    x = audio.reshape(-1)
    n = -(-x.numel() // HOP)
    x = torch.nn.functional.pad(x, (0, n * HOP - x.numel()))
    fr = x.reshape(n, HOP)
    return torch.stack([fr.mean(1), fr.std(1), fr[:, -1]], dim=-1).unsqueeze(0)


def fake_vv_decode(_vv, frames, cache=None):
    """Causal decoder that overshoots full scale (like VibeVoice): peaks > 1 before the headroom gain."""
    state = 0.0 if cache is None else cache
    out = []
    for f in frames[0]:
        state = 0.9 * state + float(f[0])
        out.append(1.4 * torch.linspace(float(f[1]), float(f[2]), HOP) + state * 0.01)
    if not out:
        return torch.zeros(1, 1, 0), state
    return torch.cat(out).reshape(1, 1, -1), state


@pytest.fixture
def fakes():
    with patch("app.dualcodec_utils.codes_to_audio_tensor", side_effect=fake_dc_decode), \
            patch("app.dualcodec_utils.vv_encode", side_effect=fake_vv_encode), \
            patch("app.dualcodec_utils.vv_decode_streaming", side_effect=fake_vv_decode), \
            patch("app.inference.free_memory"):
        yield


def _codes(n, seed=0):
    g = torch.Generator().manual_seed(seed + n)
    sem = torch.randint(0, 16384, (n,), generator=g).tolist()
    ac = [torch.randint(0, 4096, (n,), generator=g).tolist() for _ in range(3)]
    return sem, ac


def _predict_engine(headroom_db, end_fix, n_tokens, vv=True):
    config = ServerConfig()
    config.output_headroom_db = headroom_db
    config.decode_end_fix, config.decode_end_fade_ms = end_fix, 15.0
    engine = TTSEngine(config)
    engine.vv_tokenizer = object() if vv else None
    engine.dc_inference = object()
    sem, ac = _codes(n_tokens)
    sampling = dict(temperature=0.6, top_p=0.95, top_k=50, seed=123)
    engine._resolve_params = MagicMock(return_value=(sampling, (None, None), "", 0.0, -1))
    engine._resolve_prefix = AsyncMock(return_value=(None, {}))
    engine._n_codebooks = MagicMock(return_value=4)
    engine._generate_semantic = AsyncMock(return_value=(list(sem), []))
    engine._generate_acoustic = AsyncMock(side_effect=[list(a) for a in ac])
    return engine, sem, ac


class _CaptureDecode:
    """Wraps the real decode_chunked_with_context and keeps a copy of what it returned (the pre-flag output)."""

    def __init__(self):
        self.returned, self.copy = None, None

    def __call__(self, *args, **kwargs):
        self.returned = du.decode_chunked_with_context(*args, **kwargs)
        self.copy = self.returned.clone()
        return self.returned


# --- gain and env parsing --------------------------------------------------------------------------------------
def test_gain_value():
    assert du.headroom_gain(3.6) == pytest.approx(0.6607, abs=5e-5)
    assert du.headroom_gain(0.0) == 1.0
    assert 20 * math.log10(du.headroom_gain(6.0)) == pytest.approx(-6.0)


def test_apply_output_headroom():
    x = torch.linspace(-1.5, 1.5, 4801).reshape(1, 1, -1)
    keep = x.clone()
    assert du.apply_output_headroom(x, 0.0) is x and du.apply_output_headroom(x, 0, inplace=True) is x
    assert torch.equal(x, keep)                           # 0 dB: untouched, also with inplace=True
    y = du.apply_output_headroom(x, 3.6)
    assert y is not x and torch.equal(x, keep) and y.dtype == x.dtype
    assert torch.equal(y, keep * GAIN_36)
    z = du.apply_output_headroom(x, 3.6, inplace=True)
    assert z is x and torch.equal(z, y)                   # in place == out of place, bit for bit


def test_config_env_parsing(monkeypatch):
    monkeypatch.delenv("OUTPUT_HEADROOM_DB", raising=False)
    assert ServerConfig().output_headroom_db == 3.6       # default ON
    for raw, value in [("0", 0.0), ("0.0", 0.0), ("", 3.6), ("  ", 3.6), ("6", 6.0), (" 2.5 ", 2.5)]:
        monkeypatch.setenv("OUTPUT_HEADROOM_DB", raw)
        assert ServerConfig().output_headroom_db == value, raw
    for raw in ["-1", "nan", "inf", "abc", "3.6dB"]:
        monkeypatch.setenv("OUTPUT_HEADROOM_DB", raw)
        with pytest.raises(ValueError):
            ServerConfig()


# --- /predict -------------------------------------------------------------------------------------------------
@pytest.mark.asyncio
@pytest.mark.parametrize("n", [13, 103, 451, 1129])
@pytest.mark.parametrize("vv", [True, False])
@pytest.mark.parametrize("end_fix", [True, False])
async def test_predict_zero_is_bit_identical_to_pre_flag(fakes, n, vv, end_fix):
    engine, _, _ = _predict_engine(0.0, end_fix, n, vv)
    capture = _CaptureDecode()
    with patch("app.inference.decode_chunked_with_context", side_effect=capture):
        audio, _ = await engine.generate("Test text.", temperature=0.6)
    assert audio is capture.returned                       # the decode output itself, as before the flag
    assert audio.dtype == capture.copy.dtype and torch.equal(audio, capture.copy)


@pytest.mark.asyncio
@pytest.mark.parametrize("n", [13, 103, 451, 1129])
@pytest.mark.parametrize("vv", [True, False])
@pytest.mark.parametrize("end_fix", [True, False])
async def test_predict_default_scales_after_the_end_fix(fakes, monkeypatch, n, vv, end_fix):
    monkeypatch.delenv("OUTPUT_HEADROOM_DB", raising=False)
    engine, sem, ac = _predict_engine(ServerConfig().output_headroom_db, end_fix, n, vv)
    assert engine.config.output_headroom_db == 3.6
    capture = _CaptureDecode()
    with patch("app.inference.decode_chunked_with_context", side_effect=capture):
        audio, _ = await engine.generate("Test text.", temperature=0.6)
    assert torch.equal(audio, capture.copy * GAIN_36)      # decode (incl. end fix) first, then the gain
    assert audio.abs().max() == pytest.approx(capture.copy.abs().max().item() * 0.6607, rel=1e-4)
    if end_fix:                                            # end fix ran inside the decode: codes' length, faded
        assert audio.shape[-1] == n * SPT and audio[..., -1].item() == 0.0
    # the decode itself got the same arguments as without the flag (VibeVoice off -> DualCodec scaled too)
    ref = du.decode_chunked_with_context(object(), engine.vv_tokenizer, sem, ac, hz=HZ,
                                         context_seconds=engine.config.chunk_context_seconds,
                                         end_fix=end_fix, end_fade_ms=15.0)
    assert torch.equal(capture.copy, ref)


@pytest.mark.asyncio
async def test_predict_gain_comes_after_the_silence_retry_check():
    """RMS 0.0025 passes the 0.002 silence check; scaled first (0.00165) it would retry. Decisions unchanged."""
    engine, _, _ = _predict_engine(3.6, True, 100)
    engine._generate_acoustic = AsyncMock(return_value=[2] * 100)
    audio = torch.full((1, 1, 2400), 0.0025)
    with patch("app.inference.free_memory"), patch(
            "app.inference.decode_chunked_with_context", return_value=audio) as decode:
        result, _ = await engine.generate("Test text.", temperature=0.6)
    assert decode.call_count == 1 and engine._generate_semantic.await_count == 1
    assert result is audio and torch.equal(result, torch.full((1, 1, 2400), 0.0025) * GAIN_36)


def _edge_case_audio():
    """Silence, a quiet edge between the trim threshold (-50 dBFS) and threshold / gain, speech, silence."""
    sr = DUALCODEC_SAMPLE_RATE
    edge = 0.004                                            # -48 dBFS; x0.6607 -> -51.6 dBFS (below -50)
    assert 10 ** (-50 / 20) < edge < 10 ** (-50 / 20) / GAIN_36
    t = torch.arange(sr // 2, dtype=torch.float32) / sr
    speech = 1.2 * torch.sin(2 * math.pi * 220 * t)        # overshoots full scale before the gain
    parts = [torch.zeros(sr // 2), torch.full((sr // 5,), edge), speech, torch.zeros(sr // 2)]
    return torch.cat(parts).reshape(1, 1, -1)


@pytest.mark.asyncio
@pytest.mark.parametrize("headroom_db", [0.0, 3.6])
async def test_predict_gain_comes_before_trim_and_mossformer(monkeypatch, headroom_db):
    """The gain comes before the trim; the trim (and the pause cap in MossFormer2) lower their thresholds by the
    same dB, so the quiet edge stays and only the level changes."""
    decoded = _edge_case_audio()
    engine, _, _ = _predict_engine(headroom_db, True, 100)
    engine._generate_acoustic = AsyncMock(return_value=[2] * 100)
    observed = {}

    class Postprocessor:
        def process(self, audio, input_sample_rate, *, headroom_db):
            observed["mossformer_in"] = audio.clone()
            observed["mossformer_headroom_db"] = headroom_db
            return audio

    def encode_audio(audio, _format, _bitrate, sample_rate):
        observed["encoded"] = audio.clone()
        return b"audio", "audio/wav"

    async def get_engine():
        return engine

    monkeypatch.setattr(app_main, "_get_engine", get_engine)
    monkeypatch.setattr(app_main, "_encode_audio", encode_audio)
    monkeypatch.setattr(mossformer_module, "get_mossformer2_postprocessor", lambda: Postprocessor())
    with patch("app.inference.free_memory"), patch(
            "app.inference.decode_chunked_with_context", return_value=decoded.clone()):
        await app_main._synthesize(TTSRequest(text="Test", trim_silence_padding_ms=0))

    pre_flag = app_main._trim_boundary_silence(decoded, padding_ms=0)     # today's /predict: decode -> trim
    if headroom_db == 0.0:
        expected = pre_flag
    else:                                                  # gain first, so the trim sees the scaled audio
        expected = app_main._trim_boundary_silence(decoded * GAIN_36, padding_ms=0, headroom_db=3.6)
        fixed = app_main._trim_boundary_silence(decoded * GAIN_36, padding_ms=0)       # the earlier fixed -50 dBFS
        assert fixed.shape[-1] < pre_flag.shape[-1]         # would drop the quiet edge (-48 -> -51.6 dBFS)
        assert torch.equal(expected, pre_flag * GAIN_36)    # the same trim, only the level changes
        assert expected.abs().max().item() < 1.0           # and the speech no longer exceeds full scale
    assert torch.equal(observed["mossformer_in"], expected)
    assert observed["mossformer_headroom_db"] == headroom_db   # the pause cap gets the headroom too
    assert torch.equal(observed["encoded"], expected)


# --- /stream --------------------------------------------------------------------------------------------------
def _stream_engine(headroom_db, end_fix=True):
    config = ServerConfig()
    config.output_headroom_db = headroom_db
    config.decode_end_fix, config.decode_end_fade_ms = end_fix, 15.0
    config.cb0_drop_bad_prev_audio = False
    config.stream_lead_trim = config.stream_early_check = False   # the pre-flag stream path (its own tests cover the flags)
    engine = TTSEngine(config)
    engine.vv_tokenizer = object()
    engine.vv_engine = MagicMock()
    engine.vv_engine.alloc_slot.return_value = 0
    return engine


@pytest.mark.asyncio
@pytest.mark.parametrize("headroom_db", [0.0, 3.6])
async def test_stream_gain_per_chunk(headroom_db):
    chunks_in = [torch.linspace(-1.3, 1.3, 9600).reshape(1, 1, -1), torch.zeros(1, 1, 0),
                 torch.full((1, 1, 48000), 1.1), torch.linspace(0.8, 0.0, 3201).reshape(1, 1, -1)]
    copies = [c.clone() for c in chunks_in]
    engine = _stream_engine(headroom_db)

    async def inner(**_kwargs):
        for c in chunks_in:
            yield c

    engine._generate_streaming_inner = inner
    out = [c async for c in engine.generate_streaming("Some text.")]
    assert len(out) == len(chunks_in)
    for o, c, ref in zip(out, chunks_in, copies):
        assert torch.equal(c, ref)                         # the inner chunks are never modified
        if headroom_db == 0.0:
            assert o is c                                  # pre-flag: the inner chunk itself
        else:
            assert o.shape == ref.shape and torch.equal(o, ref * GAIN_36)
    engine.vv_engine.free_slot.assert_called_once_with(0)


@pytest.mark.asyncio
@pytest.mark.parametrize("headroom_db", [0.0, 3.6])
@pytest.mark.parametrize("end_fix", [True, False])
async def test_stream_flush_gain_after_the_end_fix(headroom_db, end_fix):
    """Real streaming loop: segment chunk + flush (end fix trims and fades it), then the gain on each."""
    base, n_tokens, emitted = 100, 104, 30
    n_frames = -(-n_tokens * SPT // HOP)                 # 63 frames for 104 codes
    flush = torch.linspace(1.3, 0.2, (n_frames - emitted) * HOP).reshape(1, 1, -1)
    first = torch.full((1, 1, emitted * HOP), 1.2)
    engine = _stream_engine(headroom_db, end_fix)
    engine._resolve_params = MagicMock(return_value=(dict(temperature=0.8, top_p=0.95, top_k=50, seed=1),
                                                     (None, None), "", 0.0, -1))
    engine._resolve_prefix = AsyncMock(return_value=(None, {}))
    engine._n_codebooks = MagicMock(return_value=4)
    engine._cb_ctx = MagicMock(return_value=SimpleNamespace(dc_base_id=base))
    engine._generate_semantic_segment = AsyncMock(side_effect=[([], False), ([base + 20] * n_tokens, True)])
    engine._generate_all_acoustic_for_segment = AsyncMock()
    engine._emit_vv_incremental = AsyncMock(side_effect=[(first, emitted), (flush.clone(), n_frames)])
    chunks = [c async for c in engine.generate_streaming("Some text.", max_new_tokens=400)]
    assert len(chunks) == 2
    keep = n_tokens * SPT - emitted * HOP
    pre_flag_flush = du.finish_request_end(flush, keep, 15.0) if end_fix else flush   # today's flush chunk
    gain = 1.0 if headroom_db == 0.0 else GAIN_36
    if headroom_db == 0.0:
        assert chunks[0] is first
        assert torch.equal(chunks[1], pre_flag_flush)
    else:
        assert torch.equal(chunks[0], torch.full((1, 1, emitted * HOP), 1.2) * gain)
        assert torch.equal(chunks[1], pre_flag_flush * gain)
        assert max(c.abs().max().item() for c in chunks) < 1.0
    total = sum(c.shape[-1] for c in chunks)
    assert total == (n_tokens * SPT if end_fix else n_frames * HOP)
    if end_fix:
        assert chunks[1][..., -1].item() == 0.0
