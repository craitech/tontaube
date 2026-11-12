"""CPU (AVX2) tests for the lite backend on a tiny random TontaubeV1-shaped checkpoint."""
import asyncio
import json
import platform
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


def _has_avx2() -> bool:
    try:
        flags = Path("/proc/cpuinfo").read_text()
    except OSError:
        return False
    return platform.machine() == "x86_64" and all(f in flags for f in (" avx2", " fma", " f16c"))


pytestmark = pytest.mark.skipif(not _has_avx2(), reason="needs an x86-64 CPU with AVX2/FMA/F16C")

PAD, EOS, TM, AM, DC = 196, 197, 198, 199, 200
HID, LAYERS, NQ, NKV, HD, INTER, HEAD = 128, 2, 4, 2, 128, 256, 128


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    from safetensors.torch import save_file

    d = tmp_path_factory.mktemp("tiny_cpu_cb0")
    vocab = DC + 16384
    (d / "config.json").write_text(json.dumps({
        "hidden_size": HID, "num_hidden_layers": LAYERS, "num_attention_heads": NQ,
        "num_key_value_heads": NKV, "head_dim": HD, "intermediate_size": INTER,
        "rms_norm_eps": 1e-6, "rope_theta": 1e6, "vocab_size": vocab,
        "max_position_embeddings": 4096, "audio_head_hidden": HEAD, "target_codebook": 0,
        "n_codebooks": 4, "audio_token_start_id": DC, "pad_token_id": PAD, "eos_token_id": EOS,
        "text_marker_token_id": TM, "audio_marker_token_id": AM, "chunk_realign_offset": 25,
    }))
    g = torch.Generator().manual_seed(0)

    def r(*shape, s=0.08):
        return (torch.randn(*shape, generator=g) * s).to(torch.bfloat16)

    w = {"model.embed_tokens.weight": r(vocab, HID, s=1.0), "model.norm.weight": 1 + r(HID, s=0.1)}
    for i in range(LAYERS):
        p = f"model.layers.{i}."
        w |= {
            p + "input_layernorm.weight": 1 + r(HID, s=0.1),
            p + "post_attention_layernorm.weight": 1 + r(HID, s=0.1),
            p + "self_attn.q_norm.weight": 1 + r(HD, s=0.1),
            p + "self_attn.k_norm.weight": 1 + r(HD, s=0.1),
            p + "self_attn.q_proj.weight": r(NQ * HD, HID),
            p + "self_attn.k_proj.weight": r(NKV * HD, HID),
            p + "self_attn.v_proj.weight": r(NKV * HD, HID),
            p + "self_attn.o_proj.weight": r(HID, NQ * HD),
            p + "mlp.gate_proj.weight": r(INTER, HID),
            p + "mlp.up_proj.weight": r(INTER, HID),
            p + "mlp.down_proj.weight": r(HID, INTER),
        }
    w |= {
        "audio_head.fc1.weight": r(HEAD, HID, s=0.3).float(), "audio_head.fc1.bias": r(HEAD).float(),
        "audio_head.fc2.weight": r(16386, HEAD, s=0.3).float(), "audio_head.fc2.bias": r(16386).float(),
    }
    save_file(w, str(d / "model.safetensors"))
    return d


def _model(tiny_dir, plan="int4g64", max_len=1024):
    from app.lite.cpu.model import CpuCodebookModel
    from app.lite.plans import parse_plan
    from app.lite.weights import LazyWeights

    return CpuCodebookModel(tiny_dir, 0, parse_plan(plan), max_len=max_len,
                            weights=LazyWeights(tiny_dir / "model.safetensors", mmap=True))


def _voice(seed: int, n: int = 40) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return (DC + torch.randint(0, 16384, (n,), generator=g)).tolist()


def _text(seed: int, n: int = 30) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 190, (n,), generator=g).tolist()


def _kl(p_logits, q_logits) -> float:
    lp, lq = p_logits.log_softmax(-1), q_logits.log_softmax(-1)
    return float((lp.exp() * (lp - lq)).sum())


def test_cpu_kernels_match_reference():
    from app.lite.cpu.linear import CpuLinear
    from app.native import kernels
    from app.lite.quant import QLinear, QuantSpec, dequantize_ref

    ext = kernels()
    torch.manual_seed(0)
    for bits, group in ((4, 64), (4, 128), (8, 128), (8, 256)):
        w = torch.randn(300, 512) * 0.05
        b = torch.randn(300) * 0.1
        q = QLinear.from_weight(w, b, QuantSpec(bits, group), "cpu")
        ref = dequantize_ref(q.codes, q.scales, q.mins, bits, group, 512)
        lin = CpuLinear(q)
        assert lin.kind == ("q4" if bits == 4 else "q8")
        for m in (1, 3, 8, 9, 40):  # GEMV path up to 8 rows, blocked dequant + matmul above
            x = torch.randn(m, 512)
            y, yr = lin(x), x @ ref.t() + q.bias.float()  # the bias is stored in bf16
            tol = 5e-4 if m <= lin.GEMV_MAX_ROWS else 1e-6  # GEMV rounds activations to int16
            assert ((y - yr).norm() / yr.norm()) < tol, (bits, group, m)
    for L in (1, 7, 33, 300):
        qv = torch.randn(16, 128)
        kc = (torch.randn(8, 512, 128) * 2).bfloat16()
        vc = torch.randn(8, 512, 128).bfloat16()
        out = ext.decode_attn(qv, kc, vc, L, 128 ** -0.5).view(16, 128)
        ref = F.scaled_dot_product_attention(qv[None, :, None].double(), kc[None, :, :L].double(),
                                             vc[None, :, :L].double(), enable_gqa=True)[0, :, 0]
        assert (out.double() - ref).abs().max() < 1e-5, L


def test_cpu_decode_matches_prefill(tiny_dir):
    m = _model(tiny_dir)
    prompt = _voice(1) + _text(1) + [PAD]
    tokens = (DC + torch.arange(6) * 37).tolist()
    ids = prompt + tokens
    pos = m.logical_positions(ids)
    m.forward(prompt, pos[: len(prompt)], 0)
    for i in range(len(tokens)):
        t = len(prompt) + i
        dec = m.forward([ids[t]], [pos[t]], t)
    full = _model(tiny_dir).forward(ids, pos, 0)
    assert _kl(full, dec) < 1e-4


def test_cpu_fused_decoder_matches_per_op_path(tiny_dir):
    fused, ref = _model(tiny_dir), _model(tiny_dir)
    assert fused.decoder is not None
    ref.decoder = None  # per-op Python path
    prompt = _voice(3) + _text(2) + [PAD]
    pos = fused.logical_positions(prompt)
    for m in (fused, ref):
        m.forward(prompt, pos, 0)
    for i, tok in enumerate((DC + torch.arange(5) * 101).tolist()):
        t = len(prompt) + i
        a, b = fused.forward([tok], [t + 3], t), ref.forward([tok], [t + 3], t)
        assert (a - b).abs().max() < 1e-3 * b.abs().max() and int(a.argmax()) == int(b.argmax())
    n = len(fused.seq_ids)
    assert (fused.kv[..., :n, :].float() - ref.kv[..., :n, :].float()).abs().max() < 2e-2
    assert fused.seq_ids == ref.seq_ids and fused.seq_pos == ref.seq_pos


def test_cpu_prefix_reuse_matches_fresh_prefill(tiny_dir):
    m = _model(tiny_dir)
    voice = _voice(2)
    first = voice + _text(3, 25) + [PAD]
    m.forward(first, m.logical_positions(first), 0)
    m.forward([DC + 5], [len(first)], len(first))  # one decoded token, as the engine would

    second = voice + _text(4, 41) + [PAD]  # same voice block, different text length
    pos = m.logical_positions(second)
    k, offset = m.reusable_prefix(second, pos)
    assert k == len(voice) and offset != 0
    reused = m.forward(second[k:], [p + offset for p in pos[k:]], k)
    t = len(second)
    reused_step = m.forward([DC + 9], [t + offset], t)

    fresh = _model(tiny_dir)
    ref = fresh.forward(second, pos, 0)
    ref_step = fresh.forward([DC + 9], [t], t)
    assert _kl(ref, reused) < 1e-3 and _kl(ref_step, reused_step) < 1e-3

    # A retry of the same prompt reuses everything but the last token.
    k, offset = m.reusable_prefix(second, pos)
    assert k == len(second) - 1
    assert m.reusable_prefix(_voice(5) + _text(4) + [PAD], pos) == (0, 0)


def test_cpu_engine_generates_and_reuses(tiny_dir):
    from app.lite.cpu.engine import CpuEngine
    from app.lite.sampling import SamplingParams, TokensPrompt

    engine = CpuEngine(_model(tiny_dir), "cb0")
    voice = _voice(6)

    async def run(prompt, params):
        async for out in engine.generate(TokensPrompt(prompt_token_ids=prompt), params, "r"):
            return out.outputs[0]

    try:
        params = SamplingParams(temperature=0.0, max_tokens=12, min_tokens=12)
        a = asyncio.run(run(voice + _text(7) + [PAD], params))
        assert len(a.token_ids) == 12 and a.finish_reason == "length"
        assert all(DC <= t < DC + 16384 or t in (EOS, AM) for t in a.token_ids)
        b = asyncio.run(run(voice + _text(8) + [PAD], params))
        assert engine.model.seq_ids[: len(voice)] == voice and len(b.token_ids) == 12
        seeded = SamplingParams(temperature=0.9, top_p=0.9, max_tokens=10, seed=3)
        c1 = asyncio.run(run(voice + _text(9) + [PAD], seeded))
        c2 = asyncio.run(run(voice + _text(9) + [PAD], seeded))  # retry path: full-prompt reuse
        assert c1.token_ids == c2.token_ids
    finally:
        engine.shutdown()


# ---------------------------------------------------------------- VibeVoice tokenizer
VV_DIM, VV_NF, VV_DEPTHS = 16, 16, [1, 2, 1]


@pytest.fixture(scope="module")
def tiny_vv_dir(tmp_path_factory):
    """Random VibeVoice acoustic tokenizer, channels 16/32/64, strides 2 and 2."""
    from safetensors.torch import save_file

    d = tmp_path_factory.mktemp("tiny_vv")
    (d / "config.json").write_text(json.dumps({
        "vae_dim": VV_DIM, "fix_std": 0.5, "layernorm_eps": 1e-5, "encoder_depths": "-".join(map(str, VV_DEPTHS)),
        "decoder_depths": None, "encoder_n_filters": VV_NF, "decoder_n_filters": VV_NF,
    }))
    g = torch.Generator().manual_seed(0)

    def r(*shape, s=0.2):
        return (torch.randn(*shape, generator=g) * s).to(torch.bfloat16)

    def blocks(prefix, stage, dim, depth):
        out = {}
        for j in range(depth):
            p = f"{prefix}.stages.{stage}.{j}."
            out |= {p + "norm.weight": 1 + r(dim, s=0.1), p + "ffn_norm.weight": 1 + r(dim, s=0.1),
                    p + "mixer.conv.conv.conv.weight": r(dim, 1, 7), p + "mixer.conv.conv.conv.bias": r(dim),
                    p + "gamma": r(dim, s=0.5), p + "ffn_gamma": r(dim, s=0.5),
                    p + "ffn.linear1.weight": r(4 * dim, dim, s=0.1), p + "ffn.linear1.bias": r(4 * dim),
                    p + "ffn.linear2.weight": r(dim, 4 * dim, s=0.05), p + "ffn.linear2.bias": r(dim)}
        return out

    dims = [VV_NF * 2 ** i for i in range(len(VV_DEPTHS))]  # 16, 32, 64
    w = {"encoder.downsample_layers.0.0.conv.conv.weight": r(dims[0], 1, 7),
         "encoder.downsample_layers.0.0.conv.conv.bias": r(dims[0])}
    for i in range(1, len(dims)):
        w[f"encoder.downsample_layers.{i}.0.conv.conv.weight"] = r(dims[i], dims[i - 1], 4, s=0.1)
        w[f"encoder.downsample_layers.{i}.0.conv.conv.bias"] = r(dims[i])
    for i, (dim, depth) in enumerate(zip(dims, VV_DEPTHS)):
        w |= blocks("encoder", i, dim, depth)
    w |= {"encoder.head.conv.conv.weight": r(VV_DIM, dims[-1], 7, s=0.1), "encoder.head.conv.conv.bias": r(VV_DIM)}
    rdims, rdepths = dims[::-1], VV_DEPTHS[::-1]
    w |= {"decoder.upsample_layers.0.0.conv.conv.weight": r(rdims[0], VV_DIM, 7, s=0.1),
          "decoder.upsample_layers.0.0.conv.conv.bias": r(rdims[0])}
    for i in range(1, len(rdims)):
        w[f"decoder.upsample_layers.{i}.0.convtr.convtr.weight"] = r(rdims[i - 1], rdims[i], 4, s=0.1)
        w[f"decoder.upsample_layers.{i}.0.convtr.convtr.bias"] = r(rdims[i])
    for i, (dim, depth) in enumerate(zip(rdims, rdepths)):
        w |= blocks("decoder", i, dim, depth)
    w |= {"decoder.head.conv.conv.weight": r(1, rdims[-1], 7), "decoder.head.conv.conv.bias": r(1)}
    save_file({k: v.contiguous() for k, v in w.items()}, str(d / "model.safetensors"))
    return d


class _Cache:
    """Same get/set semantics as VibeVoiceTokenizerStreamingCache."""

    def __init__(self):
        self.d = {}

    def get(self, key, idx):
        states = [self.d.get((key, i)) for i in idx.tolist()]
        return None if any(s is None for s in states) else torch.stack(states)

    def set(self, key, idx, states):
        for j, i in enumerate(idx.tolist()):
            self.d[(key, i)] = states[j].clone()


def _rel(a, b):
    return float((a - b).norm() / b.norm())


@pytest.mark.parametrize("quant,fused_max", [(None, 256), ("int8g128", 16)])
def test_cpu_vibevoice_kernels_match_reference(tiny_vv_dir, monkeypatch, quant, fused_max):
    import app.codec.vibevoice_cpu as vv_cpu

    monkeypatch.setattr(vv_cpu, "FUSED_MAX_DIM", fused_max)  # 16: the 32/64 blocks take the wide path
    fast = vv_cpu.CpuVibeVoice(tiny_vv_dir, ffn_quant=quant, conv_quant=quant)
    ref = vv_cpu.CpuVibeVoice(tiny_vv_dir, ffn_quant=quant, conv_quant=quant, reference=True)
    audio = torch.randn(1, 1, 4 * 300 + 3, generator=torch.Generator().manual_seed(1)) * 0.3
    lat_fast, lat_ref = fast.encode(audio).mean, ref.encode(audio).mean
    assert lat_fast.shape == (1, 301, VV_DIM) and _rel(lat_fast, lat_ref) < 1e-5
    latents = torch.randn(2, 37, VV_DIM, generator=torch.Generator().manual_seed(2))
    out_fast, out_ref = fast.decode(latents.transpose(1, 2)), ref.decode(latents.transpose(1, 2))
    assert out_fast.shape == (2, 1, 37 * 4) and _rel(out_fast, out_ref) < 1e-5
    assert torch.equal(fast.decode(latents), out_fast)  # [B, T, D] is accepted as well


@pytest.mark.parametrize("reference", [False, True])
def test_cpu_vibevoice_streaming_matches_full_decode(tiny_vv_dir, reference):
    from app.codec.vibevoice_cpu import CpuVibeVoice

    vv = CpuVibeVoice(tiny_vv_dir, ffn_quant="int8g128", conv_quant="int8g128", reference=reference)
    latents = torch.randn(2, 23, VV_DIM, generator=torch.Generator().manual_seed(3))
    full = vv.decode(latents)
    cache, parts = _Cache(), [[], []]
    # Two streams in one cache: stream 0 in chunks of 1, 2, 3, ...; stream 1 in chunks of 5.
    starts = [0, 0]
    sizes = [iter([1, 2, 3, 4, 5, 8]), iter([5, 5, 5, 5, 3])]
    while starts[0] < 23 or starts[1] < 23:
        for s in (0, 1):
            if starts[s] >= 23:
                continue
            n = next(sizes[s])
            chunk = latents[s:s + 1, starts[s]:starts[s] + n]
            parts[s].append(vv.decode(chunk, cache=cache, sample_indices=torch.tensor([s]), use_cache=True))
            starts[s] += n
    for s in (0, 1):
        assert torch.allclose(torch.cat(parts[s], dim=-1), full[s:s + 1], atol=1e-5, rtol=1e-4)
    # Batched step over both slots continues both streams.
    cache = _Cache()
    a = vv.decode(latents[:, :10], cache=cache, sample_indices=torch.tensor([0, 1]), use_cache=True)
    b = vv.decode(latents[:, 10:], cache=cache, sample_indices=torch.tensor([0, 1]), use_cache=True)
    assert torch.allclose(torch.cat([a, b], dim=-1), full, atol=1e-5, rtol=1e-4)


def test_cpu_compact_text_table_matches_dense(tiny_dir):
    from app.lite.cpu.model import CpuCodebookModel
    from app.lite.plans import parse_plan
    from app.lite.weights import LazyWeights

    prompt = _voice(4) + _text(5) + [PAD]
    rows = sorted(set(_text(5)) | {PAD})  # the text rows this prompt needs
    compact = CpuCodebookModel(tiny_dir, 0, parse_plan("int4g64"), max_len=512, text_rows=rows,
                               weights=LazyWeights(tiny_dir / "model.safetensors", mmap=True))
    dense = _model(tiny_dir, max_len=512)
    assert compact.embed_cpu.shape[0] == len(rows) < dense.embed_cpu.shape[0]
    pos = dense.logical_positions(prompt)
    assert torch.equal(compact.forward(prompt, pos, 0), dense.forward(prompt, pos, 0))
    unseen = next(i for i in range(190) if i not in rows)
    with pytest.raises(ValueError, match="outside the model's text vocabulary"):
        compact.forward(prompt[:-1] + [unseen, PAD], pos, 0)


def test_build_cpu_engines_from_plans(tiny_dir):
    from types import SimpleNamespace

    from app.lite.cpu.engine import build_cpu_engines
    from app.lite.plans import parse_plan

    config = SimpleNamespace(max_model_len_per_cb=[256])
    engines = build_cpu_engines(config, [str(tiny_dir)], [parse_plan("int4g64")], bundle=None,
                                text_rows=[sorted(set(_text(7)) | {PAD})])
    try:
        assert len(engines) == 1 and engines[0].model.text_index is not None
    finally:
        for engine in engines:
            engine.shutdown()
