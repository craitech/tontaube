"""PREFIX_JOINT_DECODE: decode a prefix continuation with the prefix as the decoder's acoustic left context.

decode_with_prefix_context must return exactly the continuation's part of ONE decode of [prefix + continuation]
(the prefix's samples cut off; no level matching or crossfade). /predict uses it, with the last chunk_context_seconds of
the prefix codes, for a client splice (prefix_tokens with every codebook row, none shorter than the prefix) when the flag
is on; with the flag off, prefix_text alone, an incomplete prefix_tokens, or no prefix, the old decode call is unchanged.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

import app.dualcodec_utils as du
from app.config import ServerConfig
from app.inference import TTSEngine
from tests.test_decode_end_fix import HZ, SPT, fake_dc_decode, fake_vv_decode, fake_vv_encode


@pytest.fixture
def fakes():
    with patch("app.dualcodec_utils.codes_to_audio_tensor", side_effect=fake_dc_decode), \
            patch("app.dualcodec_utils.vv_encode", side_effect=fake_vv_encode), \
            patch("app.dualcodec_utils.vv_decode_streaming", side_effect=fake_vv_decode):
        yield


def codes(n, seed, n_cb=3):
    g = torch.Generator().manual_seed(seed)
    sem = torch.randint(0, 1000, (n,), generator=g).tolist()
    ac = [torch.randint(0, 1000, (n,), generator=g).tolist() for _ in range(n_cb)]
    return sem, ac


@pytest.mark.parametrize("n_pfx, n_cont", [(40, 63), (1, 10), (37, 500), (400, 413)])
@pytest.mark.parametrize("vv", [False, True])
@pytest.mark.parametrize("end_fix", [False, True])
def test_returns_the_continuation_part_of_one_joint_decode(fakes, n_pfx, n_cont, vv, end_fix):
    ps, pa = codes(n_pfx, 1)
    cs, ca = codes(n_cont, 2)
    tok = object() if vv else None
    kw = dict(hz=HZ, context_seconds=6.0, end_fix=end_fix, end_fade_ms=15.0)
    full = du.decode_chunked_with_context(None, tok, ps + cs, [p + c for p, c in zip(pa, ca)], **kw)
    out = du.decode_with_prefix_context(None, tok, ps, pa, cs, ca, **kw)
    assert torch.equal(out, full[..., n_pfx * SPT:])
    if end_fix:
        assert out.shape[-1] == n_cont * SPT          # ends at the continuation's codes, as without a prefix


def stateful_vv_decode(_vv, frames, cache=None):
    """Like fake_vv_decode, but each frame's output carries the previous frame's level (a decoder with memory;
    the shared fake's state stays ~0 because its frames have zero mean)."""
    prev = 0.0 if cache is None else cache
    out = []
    for f in frames[0]:
        out.append(torch.linspace(float(f[1]), float(f[2]), 3200) + 0.5 * prev)
        prev = float(f[1])
    if not out:
        return torch.zeros(1, 1, 0), prev
    return torch.cat(out).reshape(1, 1, -1), prev


def test_the_prefix_changes_the_continuation_start(fakes):
    """The prefix is real left context: with a decoder that has memory, the joint decode's first continuation
    samples differ from a cold decode, and the difference fades with the decoder's memory."""
    ps, pa = codes(40, 1)
    cs, ca = codes(63, 2)
    kw = dict(hz=HZ, context_seconds=6.0, end_fix=True, end_fade_ms=15.0)
    with patch("app.dualcodec_utils.vv_decode_streaming", side_effect=stateful_vv_decode):
        cold = du.decode_chunked_with_context(None, object(), cs, ca, **kw)
        joint = du.decode_with_prefix_context(None, object(), ps, pa, cs, ca, **kw)
    assert joint.shape == cold.shape
    assert not torch.allclose(joint[..., :SPT], cold[..., :SPT])
    assert torch.allclose(joint[..., 10 * SPT:], cold[..., 10 * SPT:])   # only the start depends on the prefix


def test_pre_vv_audio_is_cut_too(fakes):
    ps, pa = codes(25, 3)
    cs, ca = codes(50, 4)
    kw = dict(hz=HZ, context_seconds=6.0, end_fix=True, end_fade_ms=15.0, return_pre_vv=True)
    post, pre = du.decode_with_prefix_context(None, object(), ps, pa, cs, ca, **kw)
    full_post, full_pre = du.decode_chunked_with_context(None, object(), ps + cs, [p + c for p, c in zip(pa, ca)], **kw)
    assert torch.equal(post, full_post[..., 25 * SPT:]) and torch.equal(pre, full_pre[..., 25 * SPT:])


def test_misaligned_prefix_codes_are_refused():
    with pytest.raises(ValueError):
        du.decode_with_prefix_context(None, None, [1, 2, 3], [[1, 2, 3], [1, 2]], [4], [[4], [4]], hz=HZ,
                                      context_seconds=6.0)
    with pytest.raises(ValueError):
        du.decode_with_prefix_context(None, None, [1, 2, 3], [[1, 2, 3]], [4], [[4], [4]], hz=HZ,
                                      context_seconds=6.0)


def test_config_flag(monkeypatch):
    monkeypatch.delenv("PREFIX_JOINT_DECODE", raising=False)
    assert ServerConfig().prefix_joint_decode is True
    monkeypatch.setenv("PREFIX_JOINT_DECODE", "0")
    assert ServerConfig().prefix_joint_decode is False
    monkeypatch.setenv("PREFIX_JOINT_DECODE", "1")
    assert ServerConfig().prefix_joint_decode is True


def engine_for(flag, prefix_len, n_new):
    """A TTSEngine with generation mocked: CB0 = [7] * prefix_len + [1] * n_new, codebook k = k * 1000 + position."""
    config = ServerConfig()
    config.prefix_joint_decode = flag
    config.trailing_silence_tokens = 0
    engine = TTSEngine(config)
    engine.vv_tokenizer = object()
    engine.dc_inference = object()
    sampling = dict(temperature=0.6, top_p=0.95, top_k=50, seed=123)
    engine._resolve_params = MagicMock(return_value=(sampling, (None, None), "", 0.0, -1))
    prefix = None
    if prefix_len:
        prefix = {"text": "Before.", "semantic_codes": [7] * prefix_len, "semantic_global": [7] * prefix_len}
    engine._resolve_prefix = AsyncMock(return_value=(prefix, {}))
    engine._n_codebooks = MagicMock(return_value=4)
    sem = [7] * prefix_len + [1] * n_new
    engine._generate_semantic = AsyncMock(return_value=(sem, [prefix_len] if prefix_len else []))
    engine._generate_acoustic = AsyncMock(side_effect=lambda *a, codebook, **k: [codebook * 1000 + i for i in range(len(sem))])
    return engine


def rows(prefix_len, short=None):
    """prefix_tokens as a client sends them: 4 rows; `short` = (codebook, length) cuts one acoustic row."""
    r = [[7] * prefix_len] + [[cb * 1000 + i for i in range(prefix_len)] for cb in (1, 2, 3)]
    if short:
        r[short[0]] = r[short[0]][:short[1]]
    return r


async def run_generate(engine, n_new, **kw):
    audio = torch.full((1, 1, n_new * SPT), 0.1)
    with patch("app.inference.free_memory"), \
            patch("app.inference.decode_chunked_with_context", return_value=audio) as cold, \
            patch("app.inference.decode_with_prefix_context", return_value=audio.clone()) as joint:
        _, tokens = await engine.generate("After.", temperature=0.6, **kw)
    return cold, joint, tokens


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", [False, True])
async def test_predict_routes_a_client_splice(flag):
    engine = engine_for(flag, prefix_len=30, n_new=100)
    cold, joint, tokens = await run_generate(engine, 100, prefix_text="Before.", prefix_tokens=rows(30))
    assert tokens[0] == [1] * 100                                                         # prefix stripped
    assert all(t == [cb * 1000 + i for i in range(30, 130)] for cb, t in enumerate(tokens[1:], 1))
    if flag:
        cold.assert_not_called()
        args, kwargs = joint.call_args
        assert args[2] == [7] * 30                                                        # the whole short prefix
        assert args[3] == [[cb * 1000 + i for i in range(30)] for cb in (1, 2, 3)]
        assert args[4] == [1] * 100 and args[5] == [[cb * 1000 + i for i in range(30, 130)] for cb in (1, 2, 3)]
        assert kwargs["end_fix"] == engine.config.decode_end_fix and kwargs["vv_adapter"] is engine.vv_adapter
    else:
        joint.assert_not_called()
        assert cold.call_args.args[2] == [1] * 100


@pytest.mark.asyncio
async def test_a_long_prefix_gives_only_its_last_context_seconds():
    engine = engine_for(True, prefix_len=200, n_new=50)
    ctx = round(engine.config.chunk_context_seconds * 12.5)
    assert ctx == 75
    _, joint, _ = await run_generate(engine, 50, prefix_text="Before.", prefix_tokens=rows(200))
    args = joint.call_args.args
    assert args[2] == [7] * ctx
    assert args[3] == [[cb * 1000 + i for i in range(200 - ctx, 200)] for cb in (1, 2, 3)]
    assert args[4] == [1] * 50


@pytest.mark.asyncio
@pytest.mark.parametrize("kw", [dict(prefix_text="Before."),                                     # prefix_text only
                                dict(prefix_text="Before.", prefix_tokens=rows(30)[:2]),         # rows missing
                                dict(prefix_text="Before.", prefix_tokens=rows(30, short=(2, 20)))])   # a short row (padded)
async def test_no_joint_decode_without_a_complete_client_prefix(kw):
    engine = engine_for(True, prefix_len=30, n_new=100)
    cold, joint, tokens = await run_generate(engine, 100, **kw)
    joint.assert_not_called()
    assert cold.call_args.args[2] == [1] * 100 and tokens[0] == [1] * 100


@pytest.mark.asyncio
@pytest.mark.parametrize("flag", [False, True])
async def test_predict_without_prefix_is_unchanged(flag):
    engine = engine_for(flag, prefix_len=0, n_new=100)
    cold, joint, _ = await run_generate(engine, 100)
    joint.assert_not_called()
    assert cold.call_args.args[2] == [1] * 100
