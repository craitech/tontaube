"""DECODE_END_FIX: request-end decode fix.

Off (default): decode_chunked_with_context must stay bit-identical to the pre-flag code (frozen copy below).
On: the last partial VibeVoice frame is kept, the audio ends at the codes' length and its last
DECODE_END_FADE_MS are faded out; /predict and /stream pass the flag.
"""
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

import app.dualcodec_utils as du
from app.config import DUALCODEC_SAMPLE_RATE, ServerConfig
from app.inference import TTSEngine

HZ = 12.5
SPT = 1920  # samples per DualCodec token at 24 kHz
HOP = 3200  # samples per VibeVoice frame


# --- deterministic stand-ins for DualCodec and VibeVoice (shapes as the real models) -------------------------
def fake_dc_decode(_dc, semantic_codes, acoustic_codes):
    """DualCodec: 1920 samples per code, content depends on the codes and the position."""
    sem = torch.tensor(semantic_codes, dtype=torch.float32)
    ac = torch.tensor(acoustic_codes[0], dtype=torch.float32) if acoustic_codes else torch.zeros_like(sem)
    t = torch.arange(SPT, dtype=torch.float32) / SPT
    wav = torch.sin(2 * math.pi * (sem[:, None] * 0.013 + ac[:, None] * 0.007 + 3.0 * t[None, :]))
    return (0.3 * wav).reshape(1, 1, -1)


def fake_vv_encode(_vv, audio):
    """VibeVoice encoder: ceil(samples / 3200) frames (the real encoder pads the last one), D = 3."""
    x = audio.reshape(-1)
    n = -(-x.numel() // HOP)
    x = torch.nn.functional.pad(x, (0, n * HOP - x.numel()))
    fr = x.reshape(n, HOP)
    return torch.stack([fr.mean(1), fr.std(1), fr[:, -1]], dim=-1).unsqueeze(0)


def fake_vv_decode(_vv, frames, cache=None):
    """Causal streaming decoder: 3200 samples per frame, state carried in the cache."""
    state = 0.0 if cache is None else cache
    out = []
    for f in frames[0]:
        state = 0.9 * state + float(f[0])
        out.append(torch.linspace(float(f[1]), float(f[2]), HOP) + state * 0.01)
    if not out:
        return torch.zeros(1, 1, 0), state
    return torch.cat(out).reshape(1, 1, -1), state


@pytest.fixture
def fakes():
    with patch("app.dualcodec_utils.codes_to_audio_tensor", side_effect=fake_dc_decode), \
            patch("app.dualcodec_utils.vv_encode", side_effect=fake_vv_encode), \
            patch("app.dualcodec_utils.vv_decode_streaming", side_effect=fake_vv_decode):
        yield


def head_decode(dc_inference, vv_tokenizer, semantic_codes, acoustic_codes, hz, context_seconds,
                return_pre_vv=False):
    """Frozen copy of decode_chunked_with_context before the flag, logging removed."""
    total_tokens = len(semantic_codes)
    pad_seconds = context_seconds
    window_seconds = 30.0
    window_tokens = int(window_seconds * hz)
    pad_tokens = int(pad_seconds * hz)
    all_vv_frames = []
    pre_vv_parts = [] if return_pre_vv or vv_tokenizer is None else None
    offset_tokens = 0
    while offset_tokens < total_tokens:
        win_start = max(0, offset_tokens - pad_tokens)
        win_end_content = min(offset_tokens + window_tokens, total_tokens)
        win_end = min(win_end_content + pad_tokens, total_tokens)
        win_sem = semantic_codes[win_start:win_end]
        win_ac = [ac[win_start:win_end] for ac in acoustic_codes]
        with torch.no_grad():
            window_audio = du.codes_to_audio_tensor(dc_inference, win_sem, win_ac)
        left_pad_tokens = offset_tokens - win_start
        right_pad_tokens = win_end - win_end_content
        left_pad_samples = int(left_pad_tokens * DUALCODEC_SAMPLE_RATE / hz)
        right_pad_samples = int(right_pad_tokens * DUALCODEC_SAMPLE_RATE / hz)
        if pre_vv_parts is not None:
            if right_pad_samples > 0:
                pre_vv_parts.append(window_audio[..., left_pad_samples:-right_pad_samples].cpu())
            else:
                pre_vv_parts.append(window_audio[..., left_pad_samples:].cpu())
        if vv_tokenizer is not None:
            with torch.no_grad():
                frames = du.vv_encode(vv_tokenizer, window_audio)
            left_pad_vv = int(left_pad_tokens / hz * du.VV_HZ)
            content_vv = int((win_end_content - offset_tokens) / hz * du.VV_HZ)
            content_frames = frames[:, left_pad_vv:left_pad_vv + content_vv, :]
            all_vv_frames.append(content_frames.cpu())
            del frames
        del window_audio
        offset_tokens = win_end_content
    pre_vv_audio = torch.cat(pre_vv_parts, dim=-1) if pre_vv_parts else None
    if vv_tokenizer is None:
        return (pre_vv_audio, pre_vv_audio) if return_pre_vv else pre_vv_audio
    vv_frames = torch.cat(all_vv_frames, dim=1)
    vv_decode_chunk = int(30.0 * du.VV_HZ)
    cache = None
    audio_parts = []
    with torch.no_grad():
        for i in range(0, vv_frames.shape[1], vv_decode_chunk):
            chunk_frames = vv_frames[:, i:i + vv_decode_chunk, :]
            audio_chunk, cache = du.vv_decode_streaming(vv_tokenizer, chunk_frames, cache=cache)
            audio_parts.append(audio_chunk)
    post_vv_audio = torch.cat(audio_parts, dim=-1)
    if return_pre_vv:
        return post_vv_audio, pre_vv_audio
    return post_vv_audio


def _codes(n, seed=0):
    g = torch.Generator().manual_seed(seed + n)
    sem = torch.randint(0, 16384, (n,), generator=g).tolist()
    ac = [torch.randint(0, 4096, (n,), generator=g).tolist() for _ in range(3)]
    return sem, ac


LENGTHS = [1, 2, 3, 4, 5, 13, 100, 101, 102, 103, 104, 374, 375, 376, 377, 451, 452, 749, 750, 751, 1129]


def _as_tuple(x):
    return x if isinstance(x, tuple) else (x,)


@pytest.mark.parametrize("n", LENGTHS)
@pytest.mark.parametrize("vv", [True, False])
@pytest.mark.parametrize("context_seconds", [6.0, 0.0])
@pytest.mark.parametrize("return_pre_vv", [False, True])
def test_flag_off_is_bit_identical_to_pre_flag_code(fakes, n, vv, context_seconds, return_pre_vv):
    sem, ac = _codes(n)
    vv_tok = object() if vv else None
    try:
        ref = _as_tuple(head_decode(object(), vv_tok, sem, ac, HZ, context_seconds, return_pre_vv))
    except (RuntimeError, ValueError):                    # n = 1 with VibeVoice: 0 frames, torch.cat([]) fails
        ref = None
    for kwargs in ({}, {"end_fix": False}, {"end_fix": False, "end_fade_ms": 20.0}):
        if ref is None:
            with pytest.raises((RuntimeError, ValueError)):
                du.decode_chunked_with_context(object(), vv_tok, sem, ac, hz=HZ, context_seconds=context_seconds,
                                               return_pre_vv=return_pre_vv, **kwargs)
            continue
        out = _as_tuple(du.decode_chunked_with_context(
            object(), vv_tok, sem, ac, hz=HZ, context_seconds=context_seconds, return_pre_vv=return_pre_vv,
            **kwargs))
        assert len(out) == len(ref)
        for a, b in zip(out, ref):
            assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)


@pytest.mark.parametrize("n", LENGTHS)
@pytest.mark.parametrize("vv", [True, False])
def test_flag_on_ends_at_the_codes_length_with_a_fade(fakes, n, vv):
    sem, ac = _codes(n)
    vv_tok = object() if vv else None
    try:
        off = du.decode_chunked_with_context(object(), vv_tok, sem, ac, hz=HZ, context_seconds=6.0)
    except (RuntimeError, ValueError):                    # n = 1 with VibeVoice fails today (0 frames)
        off = torch.zeros(1, 1, 0)
    on = du.decode_chunked_with_context(object(), vv_tok, sem, ac, hz=HZ, context_seconds=6.0,
                                        end_fix=True, end_fade_ms=15.0)
    assert on.shape[-1] == n * SPT                       # the codes' length, no longer floor(0.6 n) frames
    if vv:
        assert off.shape[-1] == (n * 3 // 5) * HOP       # today: floored to whole VibeVoice frames
    n_fade = min(on.shape[-1], round(15.0 * DUALCODEC_SAMPLE_RATE / 1000))
    assert on[..., -1].item() == 0.0
    common = min(off.shape[-1], on.shape[-1] - n_fade)
    if common > 0:                                        # causal decoder: everything before is unchanged
        assert torch.equal(on[..., :common], off[..., :common])
    ramp = 0.5 * (1 + torch.cos(math.pi * torch.arange(1, n_fade + 1, dtype=torch.float64) / n_fade))
    full = du.decode_chunked_with_context(object(), vv_tok, sem, ac, hz=HZ, context_seconds=6.0,
                                          end_fix=True, end_fade_ms=0.0)
    torch.testing.assert_close(on[..., -n_fade:], full[..., -n_fade:] * ramp.float())


def test_flag_on_leaves_the_pre_vv_audio_alone(fakes):
    sem, ac = _codes(103)
    post_off, pre_off = du.decode_chunked_with_context(object(), object(), sem, ac, hz=HZ, context_seconds=6.0,
                                                       return_pre_vv=True)
    post_on, pre_on = du.decode_chunked_with_context(object(), object(), sem, ac, hz=HZ, context_seconds=6.0,
                                                     return_pre_vv=True, end_fix=True)
    assert torch.equal(pre_on, pre_off) and pre_on.shape[-1] == 103 * SPT
    assert post_on.shape[-1] == 103 * SPT and post_off.shape[-1] == 61 * HOP


def test_end_fade_shapes():
    x = torch.ones(1, 1, 1000)
    y = du.end_fade(x, 15.0)                              # 360 samples at 24 kHz
    assert torch.equal(y[..., :640], x[..., :640]) and y[..., -1].item() == 0.0
    assert torch.all(y[..., 640:-1] > 0) and torch.all(y[..., 641:] <= y[..., 640:-1])
    assert torch.equal(x, torch.ones(1, 1, 1000))         # input untouched
    assert du.end_fade(torch.ones(1, 1, 100), 15.0)[..., -1].item() == 0.0   # shorter than the fade
    assert du.end_fade(torch.ones(1, 1, 0), 15.0).shape[-1] == 0
    assert du.finish_request_end(torch.ones(1, 1, 500), 400, 0.0).shape[-1] == 400


def test_config_flag(monkeypatch):
    monkeypatch.delenv("DECODE_END_FIX", raising=False)
    monkeypatch.delenv("DECODE_END_FADE_MS", raising=False)
    config = ServerConfig()
    assert config.decode_end_fix is True and config.decode_end_fade_ms == 15.0
    monkeypatch.setenv("DECODE_END_FIX", "0")
    config = ServerConfig()
    assert config.decode_end_fix is False
    monkeypatch.setenv("DECODE_END_FIX", "1")
    monkeypatch.setenv("DECODE_END_FADE_MS", "20")
    config = ServerConfig()
    assert config.decode_end_fix is True and config.decode_end_fade_ms == 20.0


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", [False, True])
async def test_predict_passes_the_flag(flag):
    config = ServerConfig()
    config.decode_end_fix, config.decode_end_fade_ms = flag, 12.0
    engine = TTSEngine(config)
    engine.vv_tokenizer = object()
    engine.dc_inference = object()
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
    assert decode.call_args.kwargs["end_fix"] is flag
    assert decode.call_args.kwargs["end_fade_ms"] == 12.0


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", [False, True])
async def test_stream_flush_ends_at_the_codes_length(flag):
    """The flush emits every encoder frame (ceil); with the flag the stream stops at the codes' length, faded."""
    base, n_tokens, emitted = 100, 104, 30
    n_frames = -(-n_tokens * SPT // HOP)                 # 63 frames for 104 codes
    flush = torch.linspace(0.5, 0.2, (n_frames - emitted) * HOP).reshape(1, 1, -1)
    first = torch.full((1, 1, emitted * HOP), 0.3)
    config = ServerConfig()
    config.decode_end_fix, config.decode_end_fade_ms = flag, 15.0
    config.cb0_drop_bad_prev_audio = False
    config.stream_lead_trim = config.stream_early_check = False   # the early check would re-sample the empty early CB0
    engine = TTSEngine(config)
    engine._resolve_params = MagicMock(return_value=(dict(temperature=0.8, top_p=0.95, top_k=50, seed=1),
                                                     (None, None), "", 0.0, -1))
    engine._resolve_prefix = AsyncMock(return_value=(None, {}))
    engine._n_codebooks = MagicMock(return_value=4)
    engine._cb_ctx = MagicMock(return_value=SimpleNamespace(dc_base_id=base))
    engine._generate_semantic_segment = AsyncMock(side_effect=[
        ([], False),                                      # early CB0: too short for an early chunk
        ([base + 20] * n_tokens, True),
    ])
    engine._generate_all_acoustic_for_segment = AsyncMock()
    engine._emit_vv_incremental = AsyncMock(side_effect=[(first, emitted), (flush, n_frames)])
    chunks = [c async for c in engine._generate_streaming_inner("Some text.", max_new_tokens=400, vv_slot=0)]
    assert len(chunks) == 2 and chunks[0] is first
    if not flag:
        assert chunks[1] is flush
        assert sum(c.shape[-1] for c in chunks) == n_frames * HOP
    else:
        assert sum(c.shape[-1] for c in chunks) == n_tokens * SPT
        n_fade = round(15.0 * DUALCODEC_SAMPLE_RATE / 1000)
        keep = n_tokens * SPT - emitted * HOP
        assert torch.equal(chunks[1][..., :keep - n_fade], flush[..., :keep - n_fade])
        assert chunks[1][..., -1].item() == 0.0
