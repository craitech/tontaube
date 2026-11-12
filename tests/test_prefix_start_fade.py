"""PREFIX_START_FADE: fade-in at the start of a continuation.

A response to a request with a prefix (non-empty prefix_text) fades in over its first PREFIX_START_FADE_MS (default
10 ms) with a raised cosine: /predict on the final audio (after trim / pause cap / MossFormer2, at the output sample
rate), /stream and /ws/stream on the first emitted samples (TTSEngine.generate_streaming), carried across chunks
when the first chunk is shorter than the fade. Requests without a prefix, and every request with the flag off, are
unchanged (bit-identical). The client's half (fade the kept audio's last 10 ms out) is app/splice.py.
"""
import math
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import torch

import app.dualcodec_utils as du
import app.main as app_main
import app.mossformer2_postprocessor as mossformer_module
from app import splice
from app.config import DUALCODEC_SAMPLE_RATE, ServerConfig
from app.inference import TTSEngine
from app.main import TTSRequest

SR = DUALCODEC_SAMPLE_RATE
N24 = 240  # 10 ms at 24 kHz
N48 = 480  # 10 ms at 48 kHz (MossFormer2 output)
GAIN_36 = 10.0 ** (-3.6 / 20.0)
PREFIX = dict(prefix_text="It was Saturday morning,",
              prefix_tokens=[[5] * 40, [6] * 40, [7] * 40, [8] * 40])


def ramp_in(n, start=0, length=None):
    """The expected gain: 0.5 * (1 - cos(pi * j / n)) at response sample j (float64)."""
    j = torch.arange(start, start + (n - start if length is None else length), dtype=torch.float64)
    return 0.5 * (1.0 - torch.cos(math.pi * j / n))


def sounding(n_samples, seed=0, lead_silence=0):
    """Speech-like test audio that sounds from its first sample (or after `lead_silence` samples of a -70 dBFS
    noise floor, which the edge trim treats as silence)."""
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(n_samples, dtype=torch.float64) / SR
    x = 0.3 * torch.sin(2 * math.pi * 180 * t + 0.7) + 0.05 * torch.randn(n_samples, generator=g, dtype=torch.float64)
    x[:lead_silence] = 3e-4 * torch.randn(lead_silence, generator=g, dtype=torch.float64)
    return x.float().reshape(1, 1, -1)


# --- start_fade --------------------------------------------------------------------------------------------------
def test_start_fade_ramp_and_rest_unchanged():
    x = sounding(5000)
    y = du.start_fade(x, 10.0)
    assert y is not x and torch.equal(x, sounding(5000))              # a new tensor, the input untouched
    assert y[..., 0].item() == 0.0                                    # the first sample becomes 0
    torch.testing.assert_close(y[..., :N24], x[..., :N24] * ramp_in(N24).float(), rtol=0, atol=0)
    assert torch.equal(y[..., N24:], x[..., N24:])                    # sample n on: bit-identical
    g = ramp_in(N24)
    assert torch.all(g[1:] > g[:-1]) and g[0] == 0.0 and g[-1] < 1.0  # rises monotonically from 0 towards 1


def test_start_fade_is_end_fade_reversed_in_time():
    x = torch.ones(1, 1, 1000, dtype=torch.float64)
    y = du.start_fade(x, 10.0)
    e = du.end_fade(x, 10.0)
    torch.testing.assert_close(y[..., :N24], e[..., -N24:].flip(-1), rtol=0, atol=1e-12)   # equal up to rounding


@pytest.mark.parametrize("sr, n", [(24000, 240), (48000, 480), (16000, 160)])
def test_start_fade_length_follows_the_sample_rate(sr, n):
    x = torch.full((1, 1, 2 * n), 0.5)
    y = du.start_fade(x, 10.0, sr)
    assert torch.all(y[..., :n] < 0.5) and torch.equal(y[..., n:], x[..., n:])


def test_start_fade_degenerate_inputs():
    x = torch.full((1, 1, 100), 0.5)                                  # shorter than the fade: all of it on the ramp
    y = du.start_fade(x, 10.0)
    torch.testing.assert_close(y, x * ramp_in(N24, 0, 100).float(), rtol=0, atol=0)
    empty = torch.zeros(1, 1, 0)
    assert du.start_fade(empty, 10.0) is empty
    assert du.start_fade(x, 0.0) is x                                 # 0 ms: nothing to do
    assert du.start_fade(x, 10.0, offset=N24) is x                    # chunk entirely after the fade
    assert du.start_fade(x, 10.0, offset=10_000) is x


@pytest.mark.parametrize("sizes", [[3200], [100, 3200], [1, 1, 1, 5000], [0, 120, 0, 119, 1, 4000],
                                   [239, 2], [240, 240], [50] * 10, [9600, 9600]])
def test_start_fade_offset_continues_the_ramp_across_chunks(sizes):
    whole = sounding(sum(sizes), seed=len(sizes))
    ref = du.start_fade(whole, 10.0)
    out, emitted = [], 0
    for c in torch.split(whole, sizes, dim=-1):
        o = du.start_fade(c, 10.0, offset=emitted)
        if emitted >= N24 or c.shape[-1] == 0:
            assert o is c                                             # past the fade: the chunk itself
        out.append(o)
        emitted += c.shape[-1]
    assert torch.equal(torch.cat(out, dim=-1), ref)


def test_config_flag(monkeypatch):
    monkeypatch.delenv("PREFIX_START_FADE", raising=False)
    monkeypatch.delenv("PREFIX_START_FADE_MS", raising=False)
    config = ServerConfig()
    assert config.prefix_start_fade is True and config.prefix_start_fade_ms == 10.0
    monkeypatch.setenv("PREFIX_START_FADE", "0")
    monkeypatch.setenv("PREFIX_START_FADE_MS", "5")
    config = ServerConfig()
    assert config.prefix_start_fade is False and config.prefix_start_fade_ms == 5.0
    monkeypatch.setenv("PREFIX_START_FADE", "1")
    assert ServerConfig().prefix_start_fade is True


# --- /predict (app.main._synthesize, engine.generate mocked) -----------------------------------------------------
class IdentityMossFormer:
    """MossFormer2 stand-in: returns its input unchanged (main then reports 48 kHz, as with the real enhancer)."""

    def __init__(self, observed):
        self.observed = observed

    def process(self, audio, input_sample_rate, *, headroom_db):
        self.observed["mossformer_in"] = audio.clone()
        return audio.clone()


async def predict(monkeypatch, decoded, flag, fade_ms=10.0, lead_default=False, **req):
    config = ServerConfig()
    config.prefix_start_fade, config.prefix_start_fade_ms = flag, fade_ms
    if not lead_default:                       # these tests isolate the fade from the 80-ms prefix lead (its own tests)
        config.prefix_lead_trim_padding_ms = None
    engine = TTSEngine(config)
    engine.vv_tokenizer = object()
    engine.generate = AsyncMock(return_value=(decoded.clone(), [[1] * 10] * 4))
    observed = {}

    def encode_audio(audio, _format, _bitrate, sample_rate):
        observed["encoded"], observed["sample_rate"] = audio.clone(), sample_rate
        return b"audio", "audio/wav"

    async def get_engine():
        return engine

    monkeypatch.setattr(app_main, "_get_engine", get_engine)
    monkeypatch.setattr(app_main, "_encode_audio", encode_audio)
    monkeypatch.setattr(mossformer_module, "get_mossformer2_postprocessor", lambda: IdentityMossFormer(observed))
    await app_main._synthesize(TTSRequest(text="So we went out.", **req))
    observed["generate_kwargs"] = engine.generate.call_args.kwargs
    return observed


POST = [dict(mossformer2_postprocess=False, trim_silence_padding_ms=None),     # 24 kHz, no trim
        dict(mossformer2_postprocess=False),                                  # 24 kHz, default 250 ms trim
        dict(mossformer2_postprocess=True, trim_silence_padding_ms=None),      # "48 kHz" (identity enhancer)
        dict()]                                                               # the defaults: trim + MossFormer2
DECODES = [lambda: sounding(24000, seed=1),                                   # sounds from sample 0
           lambda: sounding(24000, seed=2, lead_silence=12000)]               # 0.5 s quiet lead: the trim cuts it


@pytest.mark.asyncio
@pytest.mark.parametrize("post", POST)
@pytest.mark.parametrize("make", DECODES)
async def test_predict_without_prefix_is_bit_identical_flag_on_vs_off(monkeypatch, post, make):
    decoded = make()
    on = await predict(monkeypatch, decoded, True, **post)
    off = await predict(monkeypatch, decoded, False, **post)
    tokens_only = await predict(monkeypatch, decoded, True, prefix_tokens=PREFIX["prefix_tokens"], **post)
    assert on["sample_rate"] == off["sample_rate"]
    assert torch.equal(on["encoded"], off["encoded"])
    assert torch.equal(tokens_only["encoded"], off["encoded"])        # prefix_tokens without prefix_text: no prefix
    assert on["generate_kwargs"]["prefix_text"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("post", POST)
@pytest.mark.parametrize("make", DECODES)
async def test_predict_with_prefix_fades_the_first_10_ms_in(monkeypatch, post, make):
    decoded = make()
    ref = (await predict(monkeypatch, decoded, False, **post))["encoded"]     # the old output (flag off)
    on = await predict(monkeypatch, decoded, True, **PREFIX, **post)
    assert on["generate_kwargs"]["prefix_text"] == PREFIX["prefix_text"]
    n = N48 if on["sample_rate"] == 48000 else N24
    y = on["encoded"]
    assert y.shape == ref.shape and y[..., 0].item() == 0.0
    torch.testing.assert_close(y[..., :n], ref[..., :n] * ramp_in(n).float(), rtol=0, atol=0)
    assert torch.equal(y[..., n:], ref[..., n:])                      # the rest is unchanged
    if post.get("trim_silence_padding_ms", 250) is not None and make is DECODES[1]:
        assert y.shape[-1] < decoded.shape[-1]                        # the trim cut the quiet lead; the fade
                                                                      # starts at the response's first sample
    if post.get("mossformer2_postprocess", True):                     # the fade comes after MossFormer2:
        assert on["sample_rate"] == 48000                             # it gets the unfaded audio (identity
        assert torch.equal(on["mossformer_in"], ref)                  # enhancer here, so = the old output)


@pytest.mark.asyncio
@pytest.mark.parametrize("post", POST)
async def test_predict_with_prefix_flag_off_is_the_old_output(monkeypatch, post):
    decoded = sounding(24000, seed=3)
    off = await predict(monkeypatch, decoded, False, **PREFIX, **post)
    old = await predict(monkeypatch, decoded, False, **post)
    assert torch.equal(off["encoded"], old["encoded"])
    expected = decoded                                               # the old pipeline, step by step
    if post.get("trim_silence_padding_ms", 250) is not None:
        expected = app_main._trim_boundary_silence(expected, padding_ms=post.get("trim_silence_padding_ms", 250),
                                                   headroom_db=ServerConfig().output_headroom_db)
    assert torch.equal(off["encoded"], expected)


@pytest.mark.asyncio
async def test_predict_fade_length_follows_the_flag_ms(monkeypatch):
    decoded = sounding(24000, seed=4)
    ref = (await predict(monkeypatch, decoded, False, mossformer2_postprocess=False))["encoded"]
    y = (await predict(monkeypatch, decoded, True, fade_ms=5.0, mossformer2_postprocess=False, **PREFIX))["encoded"]
    torch.testing.assert_close(y[..., :120], ref[..., :120] * ramp_in(120).float(), rtol=0, atol=0)
    assert torch.equal(y[..., 120:], ref[..., 120:])


# --- /stream and /ws/stream (TTSEngine.generate_streaming; the inner generator mocked) ----------------------------
def stream_engine(flag, headroom_db=0.0, fade_ms=10.0):
    config = ServerConfig()
    config.prefix_start_fade, config.prefix_start_fade_ms = flag, fade_ms
    config.output_headroom_db = headroom_db
    engine = TTSEngine(config)
    engine.vv_tokenizer = object()
    engine.vv_engine = MagicMock()
    engine.vv_engine.alloc_slot.return_value = 0
    return engine


async def stream(engine, chunks_in, **kw):
    seen = {}

    async def inner(**kwargs):
        seen.update(kwargs)
        for c in chunks_in:
            yield c

    engine._generate_streaming_inner = inner
    out = [c async for c in engine.generate_streaming("So we went out.", **kw)]
    engine.vv_engine.free_slot.assert_called_once_with(0)
    return out, seen


CHUNKINGS = [[67200], [100, 67200], [0, 239, 1, 9600], [17, 17, 17, 67200], [240, 3200], [4000, 0, 4000]]


def chunks_for(sizes, seed=0):
    return list(torch.split(sounding(sum(sizes), seed=seed), sizes, dim=-1))


@pytest.mark.asyncio
@pytest.mark.parametrize("sizes", CHUNKINGS)
@pytest.mark.parametrize("headroom_db", [0.0, 3.6])
async def test_stream_without_prefix_is_bit_identical_flag_on_vs_off(sizes, headroom_db):
    chunks_in = chunks_for(sizes)
    on, _ = await stream(stream_engine(True, headroom_db), chunks_in)
    off, _ = await stream(stream_engine(False, headroom_db), chunks_in)
    tok, _ = await stream(stream_engine(True, headroom_db), chunks_in, prefix_tokens=PREFIX["prefix_tokens"])
    assert len(on) == len(off) == len(tok) == len(chunks_in)
    for a, b, c, x in zip(on, off, tok, chunks_in):
        assert torch.equal(a, b) and torch.equal(c, b)
        if headroom_db == 0.0:
            assert a is x and b is x and c is x                    # the inner chunk itself, as before the flag


@pytest.mark.asyncio
@pytest.mark.parametrize("sizes", CHUNKINGS)
@pytest.mark.parametrize("headroom_db", [0.0, 3.6])
async def test_stream_with_prefix_fades_the_first_emitted_10_ms_in(sizes, headroom_db):
    chunks_in = chunks_for(sizes, seed=7)
    copies = [c.clone() for c in chunks_in]
    ref, _ = await stream(stream_engine(False, headroom_db), chunks_in, **PREFIX)       # the old output
    out, seen = await stream(stream_engine(True, headroom_db), chunks_in, **PREFIX)
    assert seen["prefix_text"] == PREFIX["prefix_text"] and seen["prefix_tokens"] == PREFIX["prefix_tokens"]
    assert [c.shape for c in out] == [c.shape for c in ref]           # chunking unchanged
    for c, cp in zip(chunks_in, copies):
        assert torch.equal(c, cp)                                     # the inner chunks are never modified
    y, r = torch.cat(out, dim=-1), torch.cat(ref, dim=-1)
    assert y[..., 0].item() == 0.0
    torch.testing.assert_close(y[..., :N24], r[..., :N24] * ramp_in(N24).float(), rtol=0, atol=0)
    assert torch.equal(y[..., N24:], r[..., N24:])
    emitted = 0
    for o, x in zip(out, ref):                                        # chunks after the fade: passed through
        if emitted >= N24 or x.shape[-1] == 0:
            assert o is x if headroom_db == 0.0 else torch.equal(o, x)
        emitted += x.shape[-1]


@pytest.mark.asyncio
async def test_stream_short_first_chunk_carries_the_ramp_into_the_next():
    chunks_in = chunks_for([100, 3200], seed=9)
    out, _ = await stream(stream_engine(True), chunks_in, **PREFIX)
    first, second = out
    torch.testing.assert_close(first, chunks_in[0] * ramp_in(N24, 0, 100).float(), rtol=0, atol=0)
    torch.testing.assert_close(second[..., :140], chunks_in[1][..., :140] * ramp_in(N24, 100, 140).float(),
                               rtol=0, atol=0)
    assert torch.equal(second[..., 140:], chunks_in[1][..., 140:])


@pytest.mark.asyncio
@pytest.mark.parametrize("sizes", CHUNKINGS)
async def test_stream_with_prefix_flag_off_is_the_old_output(sizes):
    chunks_in = chunks_for(sizes, seed=11)
    out, _ = await stream(stream_engine(False), chunks_in, **PREFIX)
    assert all(o is x for o, x in zip(out, chunks_in)) and len(out) == len(chunks_in)


# --- client side (app/splice.py) ----------------------------------------------------------------------------------
def test_cut_sample():
    assert splice.cut_sample(76) == 76 * 1920 == 145920
    assert splice.cut_sample(76, 48000) == 76 * 3840
    assert splice.cut_sample(0) == 0


def test_fade_out_tail_is_servings_end_fade():
    x = sounding(5000, seed=5).reshape(-1)
    y = splice.fade_out_tail(x.numpy())
    ref = du.end_fade(x.reshape(1, 1, -1), 10.0).reshape(-1).numpy()
    assert y.dtype == np.float32 and np.array_equal(y, ref)
    assert y[-1] == 0.0 and np.array_equal(y[:-N24], x.numpy()[:-N24])
    assert np.array_equal(x.numpy(), sounding(5000, seed=5).reshape(-1).numpy())    # the input untouched


def test_fade_out_tail_layouts_and_dtypes():
    stereo = np.ones((1000, 2), dtype=np.float32)
    y = splice.fade_out_tail(stereo)
    assert y.shape == (1000, 2) and np.array_equal(y[:, 0], y[:, 1]) and y[-1, 0] == 0.0
    pcm = np.full(1000, 1000, dtype=np.int16)
    y = splice.fade_out_tail(pcm)
    assert y.dtype == np.float32 and y[-1] == 0.0 and y[0] == 1000.0
    assert splice.fade_out_tail(np.ones(1000, dtype=np.float64)).dtype == np.float64
    short = splice.fade_out_tail(np.ones(10, dtype=np.float32))     # shorter than the fade: all of it, ends at 0
    assert short[-1] == 0.0 and np.all(np.diff(short) <= 0)
    assert splice.fade_out_tail(np.ones(1000), fade_ms=0.0).tolist() == [1.0] * 1000
    assert splice.fade_out_tail(np.zeros(0, dtype=np.float32)).shape == (0,)
    y48 = splice.fade_out_tail(np.ones(2000, dtype=np.float32), sample_rate=48000)
    assert np.all(y48[:-N48] == 1.0) and np.all(y48[-N48:] < 1.0)


def test_splice_meets_at_zero_and_removes_the_step():
    """A cut while the kept audio still sounds (a 200 Hz tone at about -28 dBFS) against a continuation that restarts
    at another phase: the hard splice steps; with both fades the join starts and ends at 0 and its largest
    sample step is no bigger than the tone's own."""
    t = np.arange(SR) / SR
    kept = (0.056 * np.sin(2 * np.pi * 200 * t + 0.3)).astype(np.float32)[:12345]
    cont = (0.056 * np.sin(2 * np.pi * 200 * t + 2.9)).astype(np.float32)
    hard = np.concatenate([kept, cont])
    tone_step = max(np.abs(np.diff(kept)).max(), np.abs(np.diff(cont)).max())
    assert abs(hard[12345] - hard[12344]) > 5 * tone_step             # the click of the old splice
    served = du.start_fade(torch.from_numpy(cont).reshape(1, 1, -1), 10.0).reshape(-1).numpy()
    joined = splice.splice_continuation(kept, served)
    assert joined.shape == hard.shape and joined.dtype == np.float32
    assert joined[12344] == 0.0 and joined[12345] == 0.0
    assert np.abs(np.diff(joined)).max() <= tone_step + 1e-7
    assert np.array_equal(joined[:12345 - N24], kept[:12345 - N24])
    assert np.array_equal(joined[12345 + N24:], cont[N24:])


@pytest.mark.asyncio
async def test_predict_with_prefix_default_lead_is_80_ms():
    """The DEFAULT config keeps 80 ms (not 250) before the first sound of a prefixed /predict (24 kHz, trim on)."""
    import pytest as _pytest
    mp = _pytest.MonkeyPatch()
    try:
        decoded = sounding(24000, seed=2, lead_silence=12000)
        old = await predict(mp, decoded, False, **PREFIX, mossformer2_postprocess=False)
        new = await predict(mp, decoded, False, lead_default=True, **PREFIX, mossformer2_postprocess=False)
        assert old["encoded"].shape[-1] - new["encoded"].shape[-1] == round(24000 * 0.170)
        assert torch.equal(new["encoded"], old["encoded"][..., round(24000 * 0.170):])
        plain = await predict(mp, decoded, False, lead_default=True, mossformer2_postprocess=False)   # no prefix: unchanged
        plain_old = await predict(mp, decoded, False, mossformer2_postprocess=False)
        assert torch.equal(plain["encoded"], plain_old["encoded"])
    finally:
        mp.undo()
