"""GPU tests for the lite backend on a tiny random TontaubeV1-shaped checkpoint."""
import asyncio
import json

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

PAD, EOS, TM, AM, DC = 196, 197, 198, 199, 200
HID, LAYERS, NQ, NKV, HD, INTER, HEAD = 64, 2, 4, 2, 64, 128, 128


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    from safetensors.torch import save_file

    d = tmp_path_factory.mktemp("tiny_cb0")
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


def _model(tiny_dir, plan="bf16", max_batch=3, kv_tokens=4096, graphs=True):
    from app.lite.gpu.model import LiteCodebookModel
    from app.lite.plans import parse_plan

    m = LiteCodebookModel(tiny_dir, 0, parse_plan(plan), max_len=1024, max_batch=max_batch,
                          kv_tokens=kv_tokens)
    if graphs:
        m.capture_graphs(list(range(1, max_batch + 1)))
    return m


def _prompt(seed: int, n_text: int = 30) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    voice = (DC + torch.randint(0, 16384, (20,), generator=g)).tolist()
    text = torch.randint(0, 190, (n_text,), generator=g).tolist()
    return voice + text + [PAD]


def test_kernels_match_reference():
    from app.lite.gpu.kernels import DecodeAttention, add_rmsnorm, dequantize, qgemv, silu_mul
    from app.lite.quant import dequantize_ref, quantize_rtn

    torch.manual_seed(0)
    for bits, group in ((4, 64), (8, 128), (8, 512)):
        w = torch.randn(300, 512, device="cuda") * 0.05
        codes, s, mn = quantize_rtn(w, bits, group)
        ref = dequantize_ref(codes, s, mn, bits, group, 512)
        assert torch.equal(dequantize(codes, s, mn, bits, group, 300, 512, dtype=torch.float32), ref)
        for m in (1, 5, 23, 64):
            x = torch.randn(m, 512, device="cuda", dtype=torch.bfloat16)
            y = qgemv(x, codes, s, mn, bits, group, 300).float()
            yr = x.float() @ ref.t()
            assert ((y - yr).norm() / yr.norm()) < 1e-2, (bits, group, m)
    x = torch.randn(3, 256, device="cuda", dtype=torch.bfloat16)
    res = torch.randn(3, 256, device="cuda", dtype=torch.bfloat16)
    wn = torch.rand(256, device="cuda", dtype=torch.bfloat16)
    h, r = add_rmsnorm(x, res, wn, 1e-6)
    rr = x + res
    rf = rr.float()
    href = wn * (rf * torch.rsqrt(rf.pow(2).mean(-1, keepdim=True) + 1e-6)).to(torch.bfloat16)
    assert torch.equal(r, rr)
    assert (h.float() - href.float()).abs().max() < 2e-2
    gu = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    g, u = gu.chunk(2, -1)
    assert (silu_mul(gu).float() - (torch.nn.functional.silu(g) * u).float()).abs().max() < 2e-2
    att = DecodeAttention(2, 4, 2, 64, 512, 64, "cuda")
    kp = torch.randn(10, 64, 2, 64, device="cuda", dtype=torch.bfloat16)
    vp = torch.randn(10, 64, 2, 64, device="cuda", dtype=torch.bfloat16)
    table = torch.tensor([[7, 2, 9, 0, 0, 0, 0, 0], [4, 0, 0, 0, 0, 0, 0, 0]], device="cuda", dtype=torch.int32)
    lens = torch.tensor([150, 1], device="cuda", dtype=torch.int32)
    q = torch.randn(2, 4, 64, device="cuda", dtype=torch.bfloat16)
    out = att(q, kp, vp, table, lens).float()
    for b, n in enumerate((150, 1)):
        blocks = table[b, : (n + 63) // 64].long()
        k = kp[blocks].reshape(-1, 2, 64)[:n].float().repeat_interleave(2, 1)
        v = vp[blocks].reshape(-1, 2, 64)[:n].float().repeat_interleave(2, 1)
        p = (torch.einsum("hd,nhd->hn", q[b].float(), k) / 8).softmax(-1)
        assert (out[b] - torch.einsum("hn,nhd->hd", p, v)).abs().max() < 2e-2


def test_decode_matches_prefill(tiny_dir):
    from app.lite.gpu.model import BLOCK_SIZE

    m = _model(tiny_dir, max_batch=1)
    prompt = _prompt(1)
    blocks = list(range(m.blocks_per_seq))
    table = torch.tensor([blocks], device="cuda", dtype=torch.int32)
    tokens = (DC + torch.arange(5) * 37).tolist()
    m.prefill(prompt, blocks)
    for i, tok in enumerate(tokens):
        t = len(prompt) + i
        dec = m.decode(torch.tensor([tok], device="cuda"), torch.tensor([t], device="cuda"), table,
                       torch.tensor([t + 1], device="cuda", dtype=torch.int32),
                       torch.tensor([blocks[t // BLOCK_SIZE] * BLOCK_SIZE + t % BLOCK_SIZE], device="cuda"))[0]
        ref = m.prefill(prompt + tokens[: i + 1], list(range(m.blocks_per_seq, 2 * m.blocks_per_seq)))
        # Different kernels (paged flash-decoding vs SDPA, GEMV vs GEMM): equal up to bf16 noise.
        lp, lq = ref.log_softmax(-1), dec.log_softmax(-1)
        kl = (lp.exp() * (lp - lq)).sum()
        assert kl < 2e-3, kl
        m.prefill(prompt + tokens[: i + 1], blocks)  # restore pool contents for the next step


def _run(engine, prompts, params, priorities=None):
    from app.lite.sampling import TokensPrompt

    async def one(p, prio):
        async for out in engine.generate(TokensPrompt(prompt_token_ids=p), params, "r", prio):
            return out.outputs[0].token_ids

    async def main():
        return await asyncio.gather(*[one(p, (priorities or [0] * len(prompts))[i])
                                      for i, p in enumerate(prompts)])

    return asyncio.run(main())


@pytest.mark.parametrize("plan", ["bf16", "int8g64", "int4g32"])
def test_engine_batching_is_consistent(tiny_dir, plan):
    from app.lite.gpu.engine import LiteEngine
    from app.lite.sampling import SamplingParams

    engine = LiteEngine(_model(tiny_dir, plan=plan), "cb0")
    try:
        params = SamplingParams(temperature=0.0, max_tokens=24, stop_token_ids=[EOS, AM],
                                logit_bias={EOS: -100.0, AM: -100.0})
        prompts = [_prompt(s, 20 + 7 * s) for s in range(3)]
        sequential = [_run(engine, [p], params)[0] for p in prompts]
        batched = _run(engine, prompts, params)
        assert all(len(s) == 24 for s in sequential)
        assert batched == sequential
    finally:
        engine.shutdown()


def test_engine_stops_min_tokens_and_seed(tiny_dir):
    from app.lite.gpu.engine import LiteEngine
    from app.lite.sampling import SamplingParams

    engine = LiteEngine(_model(tiny_dir, max_batch=2), "cb0")
    try:
        forced_eos = SamplingParams(temperature=0.0, max_tokens=50, min_tokens=4,
                                    stop_token_ids=[EOS, AM], logit_bias={EOS: 100.0})
        out = _run(engine, [_prompt(5)], forced_eos)[0]
        assert len(out) == 5 and out[-1] == EOS and EOS not in out[:-1]
        seeded = SamplingParams(temperature=0.9, top_k=50, top_p=0.95, max_tokens=30, seed=7,
                                logit_bias={EOS: -100.0, AM: -100.0})
        a = _run(engine, [_prompt(6)], seeded)[0]
        b = _run(engine, [_prompt(6)], seeded)[0]
        assert a == b and all(DC <= t < DC + 16384 for t in a)
    finally:
        engine.shutdown()


def test_engine_priority_and_kv_reuse(tiny_dir):
    from app.lite.gpu.engine import LiteEngine
    from app.lite.sampling import SamplingParams, TokensPrompt

    # One slot and a KV pool that fits one request at a time.
    engine = LiteEngine(_model(tiny_dir, max_batch=1, kv_tokens=1100), "cb0")
    short = SamplingParams(temperature=0.0, max_tokens=40, logit_bias={EOS: -100.0, AM: -100.0})
    long_ = SamplingParams(temperature=0.0, max_tokens=900, logit_bias={EOS: -100.0, AM: -100.0})
    done = []

    async def one(name, prio, params, wait_active):
        if wait_active:  # submit while "first" holds the only slot
            while not engine._active:
                await asyncio.sleep(0.001)
        async for _ in engine.generate(TokensPrompt(prompt_token_ids=_prompt(3)), params, name, prio):
            done.append(name)

    async def main():
        await asyncio.gather(one("first", 50, long_, False), one("low", 100, short, True),
                             one("high", 0, short, True))

    try:
        asyncio.run(main())
        assert done == ["first", "high", "low"]
        assert len(engine._free_blocks) == engine.model.n_blocks
    finally:
        engine.shutdown()


def test_greedy_fast_path_matches_stepwise(tiny_dir):
    from app.lite.gpu.engine import LiteEngine
    from app.lite.sampling import SamplingParams

    engine = LiteEngine(_model(tiny_dir), "cb0")
    try:
        # No stops, bias or penalty: runs device-side multi-step chunks.
        fast = SamplingParams(temperature=0.0, max_tokens=45, ignore_eos=True)
        # A zero bias changes nothing but forces one host round trip per step.
        stepwise = SamplingParams(temperature=0.0, max_tokens=45, ignore_eos=True, logit_bias={DC: 0.0})
        prompts = [_prompt(s, 15 + 9 * s) for s in range(3)]
        ref = [_run(engine, [p], stepwise)[0] for p in prompts]
        assert [_run(engine, [p], fast)[0] for p in prompts] == ref
        assert _run(engine, prompts, fast) == ref
        assert all(len(r) == 45 for r in ref)
    finally:
        engine.shutdown()


def test_causal_lm_matches_reference(tmp_path):
    """Plain causal LM: positions 0..n-1, tied embedding/LM head (verbalizer)."""
    import torch.nn.functional as F
    from safetensors.torch import save_file

    from app.lite.gpu.model import BLOCK_SIZE, LiteCausalLM
    from app.lite.plans import parse_plan

    V, H, L, NQ_, NKV_, D, I = 512, 64, 2, 4, 2, 64, 128
    (tmp_path / "config.json").write_text(json.dumps({
        "hidden_size": H, "num_hidden_layers": L, "num_attention_heads": NQ_, "num_key_value_heads": NKV_,
        "head_dim": D, "intermediate_size": I, "rms_norm_eps": 1e-6, "rope_theta": 1e6, "vocab_size": V,
        "max_position_embeddings": 1024, "tie_word_embeddings": True, "eos_token_id": 7}))
    g = torch.Generator().manual_seed(0)

    def r(*shape, s=0.08):
        return (torch.randn(*shape, generator=g) * s).to(torch.bfloat16)

    w = {"model.embed_tokens.weight": r(V, H, s=1.0), "model.norm.weight": 1 + r(H, s=0.1)}
    for i in range(L):
        p = f"model.layers.{i}."
        w |= {p + "input_layernorm.weight": 1 + r(H, s=0.1), p + "post_attention_layernorm.weight": 1 + r(H, s=0.1),
              p + "self_attn.q_norm.weight": 1 + r(D, s=0.1), p + "self_attn.k_norm.weight": 1 + r(D, s=0.1),
              p + "self_attn.q_proj.weight": r(NQ_ * D, H), p + "self_attn.k_proj.weight": r(NKV_ * D, H),
              p + "self_attn.v_proj.weight": r(NKV_ * D, H), p + "self_attn.o_proj.weight": r(H, NQ_ * D),
              p + "mlp.gate_proj.weight": r(I, H), p + "mlp.up_proj.weight": r(I, H),
              p + "mlp.down_proj.weight": r(H, I)}
    save_file(w, str(tmp_path / "model.safetensors"))

    def rms(x, wt):
        return wt.float() * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    def rot(x):
        a, b = x.chunk(2, -1)
        return torch.cat((-b, a), -1)

    ids = torch.randint(10, 500, (40,), generator=g).tolist()
    W = {k: v.float().cuda() for k, v in w.items()}
    x = W["model.embed_tokens.weight"][torch.tensor(ids, device="cuda")]
    pos = torch.arange(len(ids), device="cuda").float()
    inv = 1.0 / (1e6 ** (torch.arange(0, D, 2, device="cuda").float() / D))
    ang = torch.cat([pos[:, None] * inv] * 2, -1)
    cos, sin = ang.cos()[:, None], ang.sin()[:, None]
    for i in range(L):
        p = f"model.layers.{i}."
        h = rms(x, W[p + "input_layernorm.weight"])
        q = rms((h @ W[p + "self_attn.q_proj.weight"].t()).view(-1, NQ_, D), W[p + "self_attn.q_norm.weight"])
        k = rms((h @ W[p + "self_attn.k_proj.weight"].t()).view(-1, NKV_, D), W[p + "self_attn.k_norm.weight"])
        v = (h @ W[p + "self_attn.v_proj.weight"].t()).view(-1, NKV_, D)
        q, k = q * cos + rot(q) * sin, k * cos + rot(k) * sin
        a = F.scaled_dot_product_attention(q.transpose(0, 1)[None], k.transpose(0, 1)[None],
                                           v.transpose(0, 1)[None], is_causal=True, enable_gqa=True)
        x = x + a[0].transpose(0, 1).reshape(len(ids), -1) @ W[p + "self_attn.o_proj.weight"].t()
        h = rms(x, W[p + "post_attention_layernorm.weight"])
        x = x + (F.silu(h @ W[p + "mlp.gate_proj.weight"].t()) * (h @ W[p + "mlp.up_proj.weight"].t())) \
            @ W[p + "mlp.down_proj.weight"].t()
    ref = rms(x, W["model.norm.weight"]) @ W["model.embed_tokens.weight"].t()

    for plan, tol in (("bf16", 5e-3), ("int8g64", 2e-2)):
        m = LiteCausalLM(tmp_path, parse_plan(plan), max_len=256, max_batch=1, kv_tokens=256)
        m.capture_graphs([1])
        blocks = list(range(m.blocks_per_seq))
        got = [m.prefill(ids[:30], blocks)]
        table = torch.tensor([blocks], device="cuda", dtype=torch.int32)
        for t in range(30, 39):
            got.append(m.decode(torch.tensor([ids[t]], device="cuda"), torch.tensor([t], device="cuda"), table,
                                torch.tensor([t + 1], device="cuda", dtype=torch.int32),
                                torch.tensor([blocks[t // BLOCK_SIZE] * BLOCK_SIZE + t % BLOCK_SIZE],
                                             device="cuda")).clone()[0])
        lp, lq = ref[29:39].log_softmax(-1), torch.stack(got).log_softmax(-1)
        kl = (lp.exp() * (lp - lq)).sum(-1)
        assert kl.max() < tol, (plan, kl)


def test_build_gpu_engines_from_plans(tiny_dir):
    from types import SimpleNamespace

    from app.lite.gpu.engine import build_gpu_engines
    from app.lite.plans import parse_plan

    config = SimpleNamespace(max_model_len_per_cb=[256], max_num_seqs_per_cb=[2], lite_kv_tokens_per_cb=[1024])
    engines = build_gpu_engines(config, [str(tiny_dir)], [parse_plan("int8g64")], bundle=None)
    try:
        assert len(engines) == 1 and engines[0].model.max_batch == 2
    finally:
        for engine in engines:
            engine.shutdown()
