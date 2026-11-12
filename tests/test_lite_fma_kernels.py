"""FMA decode kernels (GPUs below compute capability 7.0) against an fp32 reference and the tensor-core kernels."""
import pytest
import torch

from tests import test_lite_engine as engine_tests

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
tiny_dir = engine_tests.tiny_dir

LINEARS = [
    ("cb0.qkv", 4, 64, 4096, 2048), ("cb0.o", 4, 64, 2048, 2048), ("cb0.gate_up", 4, 64, 12288, 2048),
    ("cb0.down", 4, 64, 2048, 6144), ("cb0.down.last", 8, 128, 2048, 6144), ("cb0.fc1", 8, 128, 4096, 2048),
    ("cb0.fc2", 8, 128, 16386, 4096),
    ("cb123.qkv", 4, 64, 4096, 1024), ("cb123.o", 4, 64, 1024, 2048), ("cb123.gate_up", 4, 64, 6144, 1024),
    ("cb123.down", 4, 64, 1024, 3072), ("cb123.down.last", 8, 128, 1024, 3072), ("cb123.fc1", 8, 128, 4096, 1024),
    ("cb123.fc2", 8, 128, 4096, 4096),
    ("int4g32", 4, 32, 300, 512), ("int8g64", 8, 64, 300, 512), ("int8g512", 8, 512, 300, 512),
    ("int8.per_row", 8, 2048, 129, 2048), ("int4g64.k192", 4, 64, 77, 192), ("int4g128", 4, 128, 50, 384),
]
ROWS = (1, 2, 3, 4, 5, 8, 16, 33, 64)
UNIT_ROUNDOFF = {torch.bfloat16: 2.0 ** -8, torch.float16: 2.0 ** -11, torch.float32: 2.0 ** -23}


@pytest.fixture(autouse=True)
def _gpu_supported_by_this_torch():
    from app.torch_variant import build_supports

    if not build_supports(torch.cuda.get_device_capability(), torch.cuda.get_arch_list()):
        pytest.skip("the installed PyTorch build has no kernels for this GPU")


@pytest.fixture
def kernels_setting(monkeypatch):
    """kernels_setting("on" | "off" | "auto") sets LITE_FMA_KERNELS for kernels launched afterwards."""
    from app.lite.gpu import arch

    def use(value: str):
        monkeypatch.setenv(arch.FMA_KERNELS_ENV, value)
        arch.reset()

    yield use
    arch.reset()


def _capability() -> tuple[int, int]:
    return tuple(torch.cuda.get_device_capability())


def _stock_qgemv(x, codes, scales, mins, bits, group, n, bias=None):
    from app.lite.gpu.kernels import _launch_qgemv
    from app.lite.gpu.tuning import heuristic

    block_n, split_k, warps = heuristic(n, x.shape[1], group)
    out = torch.empty((x.shape[0], n), device=x.device, dtype=x.dtype)
    part = torch.empty((split_k, x.shape[0], n), device=x.device, dtype=torch.float32) if split_k > 1 else out
    _launch_qgemv(x, codes, scales, mins, bias, out, part, bits, group, n, block_n, split_k, warps)
    return out


def _check_close(y, ref, dtype, rtol_factor=1.0):
    u = UNIT_ROUNDOFF[dtype]
    err = (y.float() - ref).abs()
    bound = rtol_factor * u * ref.abs() + 1e-5 * ref.abs().max()
    assert bool((err <= bound).all()), (err - bound).max().item()
    return (err.norm() / ref.norm()).item()


@pytest.mark.parametrize("name, bits, group, n, k", LINEARS, ids=[s[0] for s in LINEARS])
def test_fma_gemv_matches_fp32_reference_and_stock_kernel(name, bits, group, n, k):
    from app.lite.gpu import kernels_fma
    from app.lite.quant import dequantize_ref, quantize_rtn

    g = torch.Generator(device="cuda").manual_seed(n + k)
    codes, scales, mins = quantize_rtn(torch.randn(n, k, device="cuda", generator=g) * 0.05, bits, group)
    w = dequantize_ref(codes, scales, mins, bits, group, k)
    assert kernels_fma.qgemv_supported(1, k, bits, group)
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        bias = (torch.randn(n, device="cuda", generator=g) * 0.1).to(dtype) if name.endswith(("fc1", "fc2")) else None
        for m in ROWS:
            x = torch.randn(m, k, device="cuda", generator=g).to(dtype)
            ref = x.float() @ w.t() + (0 if bias is None else bias.float())
            y = kernels_fma.qgemv(x, codes, scales, mins, bits, group, n, bias)
            assert y.shape == (m, n) and y.dtype == dtype
            _check_close(y, ref, dtype)
            if dtype != torch.float32:
                stock = _stock_qgemv(x, codes, scales, mins, bits, group, n, bias).float()
                assert ((stock - ref).norm() / ref.norm()).item() < 1e-2
                assert ((y.float() - stock).norm() / ref.norm()).item() < 3 * UNIT_ROUNDOFF[dtype]


def test_fma_gemv_handles_every_launch_config():
    from app.lite.gpu import kernels_fma
    from app.lite.quant import dequantize_ref, quantize_rtn

    n, k = 200, 1536
    for bits, group in ((4, 64), (8, 128), (8, 64)):
        codes, scales, mins = quantize_rtn(torch.randn(n, k, device="cuda") * 0.05, bits, group)
        w = dequantize_ref(codes, scales, mins, bits, group, k)
        sub_k = kernels_fma.sub_k(bits, group)
        for m in (1, 3, 6):
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            ref = x.float() @ w.t()
            for config in ((1, 8, 1, sub_k, 1, 1), (2, 16, 2, sub_k, 3, 2), (4, 32, 8, sub_k, 8, 4),
                           (4, 8, 4, sub_k // 2, 5, 1), (1, 64, 2, sub_k, 64, 2)):
                y = kernels_fma.qgemv(x, codes, scales, mins, bits, group, n, config=config)
                _check_close(y, ref, torch.bfloat16)


def _paged_kv(n_blocks, block_size, n_kv, head_dim, dtype, lens, generator):
    """Pools filled with NaN except the tokens of each sequence; tables point unused entries at a NaN block."""
    k_pool = torch.full((n_blocks + 1, block_size, n_kv, head_dim), float("nan"), device="cuda", dtype=dtype)
    v_pool = k_pool.clone()
    per_seq = -(-max(lens) // block_size)
    table = torch.full((len(lens), per_seq), n_blocks, device="cuda", dtype=torch.int32)
    order = torch.randperm(n_blocks, generator=generator, device="cpu").tolist()
    keys, values = [], []
    for b, n in enumerate(lens):
        blocks = [order.pop() for _ in range(-(-n // block_size))]
        table[b, : len(blocks)] = torch.tensor(blocks, dtype=torch.int32)
        kk = torch.randn(n, n_kv, head_dim, generator=generator).to("cuda", dtype)
        vv = torch.randn(n, n_kv, head_dim, generator=generator).to("cuda", dtype)
        for t in range(0, n, block_size):
            blk = blocks[t // block_size]
            k_pool[blk, : min(block_size, n - t)] = kk[t: t + block_size]
            v_pool[blk, : min(block_size, n - t)] = vv[t: t + block_size]
        keys.append(kk)
        values.append(vv)
    return k_pool, v_pool, table, keys, values


def _attention_reference(q, keys, values, group, scale):
    out = []
    for b in range(q.shape[0]):
        k = keys[b].float().repeat_interleave(group, 1)
        v = values[b].float().repeat_interleave(group, 1)
        p = (torch.einsum("hd,nhd->hn", q[b].float(), k) * scale).softmax(-1)
        out.append(torch.einsum("hn,nhd->hd", p, v))
    return torch.stack(out)


SEQLENS = (1, 2, 63, 64, 65, 255, 256, 257, 700, 1023)
HEADS = [(16, 8, 128, 64), (4, 2, 64, 64), (8, 8, 64, 64), (16, 4, 128, 64), (12, 4, 64, 64),
         (16, 8, 128, 32), (16, 8, 128, 16)]


@pytest.mark.parametrize("n_heads, n_kv, head_dim, block_size", HEADS, ids=[str(h) for h in HEADS])
def test_fma_attention_matches_fp32_reference_and_stock_kernel(kernels_setting, n_heads, n_kv, head_dim, block_size):
    from app.lite.gpu.kernels import DecodeAttention
    from app.lite.gpu.kernels_fma import decode_attention

    g = torch.Generator(device="cuda").manual_seed(n_heads * head_dim + block_size)
    gc = torch.Generator().manual_seed(n_kv)
    max_len = 1024
    att = DecodeAttention(8, n_heads, n_kv, head_dim, max_len, block_size, "cuda")
    kernels_setting("off")
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for batch in (1, 2, 3, 4, 5, 8):
            lens = [SEQLENS[(batch + i) % len(SEQLENS)] for i in range(batch)]
            n_blocks = sum(-(-n // block_size) for n in lens) + 3
            k_pool, v_pool, table, keys, values = _paged_kv(n_blocks, block_size, n_kv, head_dim, dtype, lens, gc)
            q = torch.randn(batch, n_heads, head_dim, device="cuda", generator=g).to(dtype)
            seqlens = torch.tensor(lens, device="cuda", dtype=torch.int32)
            ref = _attention_reference(q, keys, values, n_heads // n_kv, head_dim ** -0.5)
            out = decode_attention(att, q, k_pool, v_pool, table, seqlens)
            assert out.shape == q.shape and out.dtype == dtype
            _check_close(out, ref, dtype, rtol_factor=1.0 if dtype != torch.float32 else 64.0)
            stock = att(q, k_pool, v_pool, table, seqlens).float()
            assert (stock - ref).abs().max().item() < 2e-2
            assert (out.float() - stock).abs().max().item() < 2e-2


def test_fma_attention_handles_every_launch_config(kernels_setting):
    from app.lite.gpu.kernels import DecodeAttention
    from app.lite.gpu.kernels_fma import decode_attention

    gc = torch.Generator().manual_seed(1)
    att = DecodeAttention(3, 16, 8, 128, 1024, 64, "cuda")
    lens = [1, 300, 1023]
    k_pool, v_pool, table, keys, values = _paged_kv(30, 64, 8, 128, torch.bfloat16, lens, gc)
    q = torch.randn(3, 16, 128, device="cuda", dtype=torch.bfloat16)
    ref = _attention_reference(q, keys, values, 2, 128 ** -0.5)
    seqlens = torch.tensor(lens, device="cuda", dtype=torch.int32)
    for split_tokens, sub, warps in ((16, 16, 1), (32, 8, 2), (64, 64, 4), (128, 32, 8), (256, 128, 2), (512, 16, 1),
                                     (8, 32, 1)):
        out = decode_attention(att, q, k_pool, v_pool, table, seqlens, split_tokens=split_tokens, sub=sub, warps=warps)
        _check_close(out, ref, torch.bfloat16)
    assert sorted(att._fma_partials) == [16, 32, 64, 128, 256, 512]


def _decode_logits(m, prompts, tokens):
    """Prefill each prompt into its own blocks, then teacher-force `tokens` as batched decode steps (CUDA graphs)."""
    from app.lite.gpu.model import BLOCK_SIZE

    per_seq = m.blocks_per_seq
    tables = torch.tensor([list(range(i * per_seq, (i + 1) * per_seq)) for i in range(len(prompts))],
                          device="cuda", dtype=torch.int32)
    for i, p in enumerate(prompts):
        m.prefill(p, tables[i].tolist())
    steps = []
    for j, tok in enumerate(tokens):
        pos = [len(p) + j for p in prompts]
        rows = [int(tables[i, t // BLOCK_SIZE]) * BLOCK_SIZE + t % BLOCK_SIZE for i, t in enumerate(pos)]
        steps.append(m.decode(torch.tensor([tok] * len(prompts), device="cuda"), torch.tensor(pos, device="cuda"),
                              tables, torch.tensor([t + 1 for t in pos], device="cuda", dtype=torch.int32),
                              torch.tensor(rows, device="cuda")).clone())
    return torch.stack(steps).float()


def _kl(p_logits, q_logits):
    lp, lq = p_logits.log_softmax(-1), q_logits.log_softmax(-1)
    return (lp.exp() * (lp - lq)).sum(-1)


@pytest.mark.parametrize("plan", ["bf16", "int4g32", "int8g64"])
def test_model_decode_with_fma_kernels_matches_prefill_like_stock(tiny_dir, kernels_setting, plan):
    """Batched decode steps (CUDA graphs) against prefill logits, which use dense matmuls and SDPA for these
    prompts of more than 64 tokens; the FMA kernels must stay within the same bound as the tensor-core kernels
    and be about as close on average."""
    prompts = [engine_tests._prompt(s, 50 + 9 * s) for s in range(3)]
    tokens = (engine_tests.DC + torch.arange(12) * 37).tolist()
    kl = {}
    for setting in ("off", "on"):
        kernels_setting(setting)
        m = engine_tests._model(tiny_dir, plan=plan, max_batch=3, kv_tokens=8192)
        logits = _decode_logits(m, prompts, tokens)
        blocks = list(range(3 * m.blocks_per_seq, 4 * m.blocks_per_seq))
        ref = torch.stack([torch.stack([m.prefill(p + tokens[: j + 1], blocks).float() for p in prompts])
                           for j in range(len(tokens))])
        kl[setting] = _kl(ref, logits)
        del m
    assert kl["off"].max() < 1e-2 and kl["on"].max() < 1e-2
    assert kl["on"].mean() < 2 * kl["off"].mean() + 1e-4


def test_auto_runs_fma_kernels_on_this_pre_volta_gpu(kernels_setting):
    if _capability() >= (7, 0):
        pytest.skip("tensor-core GPU: auto keeps the stock kernels")
    from app.lite.gpu import arch, kernels, kernels_fma
    from app.lite.quant import quantize_rtn

    kernels_setting("auto")
    assert arch.use_fma_kernels(torch.device("cuda", torch.cuda.current_device()))
    codes, scales, mins = quantize_rtn(torch.randn(256, 512, device="cuda"), 4, 64)
    x = torch.randn(2, 512, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(kernels.qgemv(x, codes, scales, mins, 4, 64, 256),
                       kernels_fma.qgemv(x, codes, scales, mins, 4, 64, 256))


def test_auto_keeps_stock_kernels_bit_exact_on_tensor_core_gpus(kernels_setting):
    if _capability() < (7, 0):
        pytest.skip("pre-Volta GPU: auto runs the FMA kernels")
    from app.lite.gpu import arch, kernels
    from app.lite.gpu.kernels import DecodeAttention
    from app.lite.quant import quantize_rtn

    gc = torch.Generator().manual_seed(2)
    codes, scales, mins = quantize_rtn(torch.randn(4096, 2048, device="cuda") * 0.05, 4, 64)
    att = DecodeAttention(4, 16, 8, 128, 1024, 64, "cuda")
    lens = [1, 65, 700, 1023]
    k_pool, v_pool, table, _, _ = _paged_kv(40, 64, 8, 128, torch.bfloat16, lens, gc)
    seqlens = torch.tensor(lens, device="cuda", dtype=torch.int32)
    q = torch.randn(4, 16, 128, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(4, 2048, device="cuda", dtype=torch.bfloat16)
    results = {}
    for setting in ("off", "auto"):
        kernels_setting(setting)
        assert not arch.use_fma_kernels(x.device)
        results[setting] = (kernels.qgemv(x, codes, scales, mins, 4, 64, 4096), att(q, k_pool, v_pool, table, seqlens))
    assert all(torch.equal(a, b) for a, b in zip(results["off"], results["auto"]))
