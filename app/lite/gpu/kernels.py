"""Triton kernels for the lite backend.

* Weight-only quantized GEMV for decode steps (few rows), int4 or int8 codes
  with one asymmetric (scale, min) pair per group of K.
* Dequantization to bf16 for prefill, followed by a cuBLAS matmul.
* Flash-decoding attention over a paged KV pool, so that batched decode
  steps with different lengths run inside one CUDA graph.

Packed weight layout ("QW"):
  codes:  uint8 [N, K]      for 8 bits
          uint8 [N, K // 2] for 4 bits; byte j holds k=2j (low nibble) and
                            k=2j+1 (high nibble)
  scales: fp16  [N, K // group]
  mins:   fp16  [N, K // group]       w = code * scale + min
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from app.lite.gpu.arch import use_fma_kernels


# ---------------------------------------------------------------------------
# Quantized GEMV (decode)
# ---------------------------------------------------------------------------

@triton.jit
def _qgemv_kernel(
    x_ptr, codes_ptr, scales_ptr, mins_ptr, bias_ptr, out_ptr, part_ptr,
    M, N, K,
    stride_xm, stride_cn, stride_sn, stride_om,
    BITS: tl.constexpr, GROUP: tl.constexpr, HAS_BIAS: tl.constexpr,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, CHUNKS_PER_SPLIT: tl.constexpr,
    SPLIT_K: tl.constexpr, BF16_BITCAST: tl.constexpr, M_PAD: tl.constexpr,
):
    """y[M, N] = x[M, K] @ (codes * scale + min)^T (+ bias) for M <= 64.

    Rows of x are padded to M_PAD (16/32/64) so the inner products run on
    tensor cores;
    each K chunk lies inside one quantization group, so the scale and min are
    applied once per (row, chunk): s * (x . q) + min * sum(x). Without a K
    split the result (plus bias) is stored directly in the output dtype;
    with one, fp32 partials go to part_ptr[split] for _qgemv_reduce.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_m = tl.arange(0, M_PAD)
    m_mask = offs_m < M
    acc = tl.zeros((M_PAD, BLOCK_N), dtype=tl.float32)
    k_first = pid_k * CHUNKS_PER_SPLIT * BLOCK_K
    for c in range(CHUNKS_PER_SPLIT):
        k0 = k_first + c * BLOCK_K
        if k0 < K:
            g = k0 // GROUP
            s = tl.load(scales_ptr + offs_n * stride_sn + g, mask=n_mask, other=0.0).to(tl.float32)
            mn = tl.load(mins_ptr + offs_n * stride_sn + g, mask=n_mask, other=0.0).to(tl.float32)
            if BITS == 8:
                offs_k = k0 + tl.arange(0, BLOCK_K)
                x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
                            mask=m_mask[:, None], other=0.0)
                q = tl.load(codes_ptr + offs_n[:, None] * stride_cn + offs_k[None, :],
                            mask=n_mask[:, None], other=0)
                dot = tl.dot(x, tl.trans(q.to(x.dtype)))
                xs = tl.sum(x.to(tl.float32), axis=1)
            else:
                HALF: tl.constexpr = BLOCK_K // 2
                offs_h = tl.arange(0, HALF)
                xe = tl.load(x_ptr + offs_m[:, None] * stride_xm + (k0 + 2 * offs_h)[None, :],
                             mask=m_mask[:, None], other=0.0)
                xo = tl.load(x_ptr + offs_m[:, None] * stride_xm + (k0 + 2 * offs_h + 1)[None, :],
                             mask=m_mask[:, None], other=0.0)
                b = tl.load(codes_ptr + offs_n[:, None] * stride_cn + (k0 // 2 + offs_h)[None, :],
                            mask=n_mask[:, None], other=0)
                if BF16_BITCAST:
                    # 0x4300 | q is the bf16 bit pattern of 128 + q (q < 16):
                    # no int->float conversion; the +128 offset is folded
                    # into the min term below.
                    b16 = b.to(tl.uint16)
                    lo = ((b16 & 0xF) | 0x4300).to(tl.bfloat16, bitcast=True)
                    hi = ((b16 >> 4) | 0x4300).to(tl.bfloat16, bitcast=True)
                    mn = mn - 128.0 * s
                else:
                    lo = (b & 0xF).to(xe.dtype)
                    hi = (b >> 4).to(xe.dtype)
                dot = tl.dot(xe, tl.trans(lo)) + tl.dot(xo, tl.trans(hi))
                xs = tl.sum(xe.to(tl.float32), axis=1) + tl.sum(xo.to(tl.float32), axis=1)
            acc += dot * s[None, :] + xs[:, None] * mn[None, :]
    mask = m_mask[:, None] & n_mask[None, :]
    if SPLIT_K == 1:
        if HAS_BIAS:
            acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)[None, :]
        tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :],
                 acc.to(out_ptr.dtype.element_ty), mask=mask)
    else:
        tl.store(part_ptr + pid_k * M * N + offs_m[:, None] * N + offs_n[None, :], acc, mask=mask)


@triton.jit
def _qgemv_reduce(part_ptr, bias_ptr, out_ptr, M, N, stride_om,
                  SPLIT_K: tl.constexpr, HAS_BIAS: tl.constexpr, BLOCK: tl.constexpr):
    m = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    acc = tl.zeros((BLOCK,), dtype=tl.float32)
    for k in tl.static_range(SPLIT_K):
        acc += tl.load(part_ptr + k * M * N + m * N + offs, mask=mask, other=0.0)
    if HAS_BIAS:
        acc += tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + m * stride_om + offs, acc.to(out_ptr.dtype.element_ty), mask=mask)


def _launch_qgemv(x, codes, scales, mins, bias, out, part, bits: int, group: int, n: int,
                  block_n: int, split_k: int, warps: int) -> None:
    M, K = x.shape
    block_k = min(128, group)
    n_chunks = K // block_k
    has_bias = bias is not None
    m_pad = 16 if M <= 16 else (32 if M <= 32 else 64)
    if m_pad > 16:
        block_n = min(block_n, 64)  # bound the accumulator tile
    _qgemv_kernel[(triton.cdiv(n, block_n), split_k)](
        x, codes, scales, mins, bias if has_bias else out, out, part,
        M, n, K,
        x.stride(0), codes.stride(0), scales.stride(0), out.stride(0),
        BITS=bits, GROUP=group, HAS_BIAS=has_bias, BLOCK_N=block_n, BLOCK_K=block_k,
        CHUNKS_PER_SPLIT=triton.cdiv(n_chunks, split_k), SPLIT_K=split_k,
        BF16_BITCAST=x.dtype == torch.bfloat16, M_PAD=m_pad, num_warps=warps,
    )
    if split_k > 1:
        _qgemv_reduce[(M, triton.cdiv(n, 1024))](
            part, bias if has_bias else part, out, M, n, out.stride(0),
            SPLIT_K=split_k, HAS_BIAS=has_bias, BLOCK=1024, num_warps=4,
        )


def qgemv(x: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor, mins: torch.Tensor,
          bits: int, group: int, n_out: int, bias: torch.Tensor | None = None) -> torch.Tensor:
    """y[M, N] = x[M, K] @ dequant(W)^T (+ bias) for M <= 64, in x.dtype."""
    if use_fma_kernels(x.device):
        from app.lite.gpu import kernels_fma

        if kernels_fma.qgemv_supported(x.shape[0], x.shape[1], bits, group):
            return kernels_fma.qgemv(x, codes, scales, mins, bits, group, n_out, bias)
    from app.lite.gpu.tuning import gemv_config

    M, K = x.shape
    assert M <= 64 and group % min(128, group) == 0
    block_n, split_k, warps = gemv_config(
        bits, group, n_out, K,
        lambda bn, sk, w, xx, oo, pp: _launch_qgemv(
            xx, codes, scales, mins, None, oo, pp, bits, group, n_out, bn, sk, w),
    )
    out = torch.empty((M, n_out), device=x.device, dtype=x.dtype)
    part = torch.empty((split_k, M, n_out), device=x.device, dtype=torch.float32) if split_k > 1 else out
    _launch_qgemv(x, codes, scales, mins, bias, out, part, bits, group, n_out, block_n, split_k, warps)
    return out


# ---------------------------------------------------------------------------
# Dequantize (prefill)
# ---------------------------------------------------------------------------

@triton.jit
def _dequant_kernel(
    codes_ptr, scales_ptr, mins_ptr, out_ptr, N, K,
    stride_cn, stride_sn, stride_on,
    BITS: tl.constexpr, GROUP: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.program_id(1) * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
    if BITS == 8:
        c = tl.load(codes_ptr + offs_n[:, None] * stride_cn + offs_k[None, :], mask=mask, other=0).to(tl.float32)
    else:
        byte = tl.load(codes_ptr + offs_n[:, None] * stride_cn + offs_k[None, :] // 2, mask=mask, other=0).to(tl.int32)
        c = ((byte >> ((offs_k[None, :] % 2) * 4)) & 0xF).to(tl.float32)
    g = offs_k // GROUP
    s = tl.load(scales_ptr + offs_n[:, None] * stride_sn + g[None, :], mask=mask, other=0.0).to(tl.float32)
    mn = tl.load(mins_ptr + offs_n[:, None] * stride_sn + g[None, :], mask=mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs_n[:, None] * stride_on + offs_k[None, :], (c * s + mn).to(out_ptr.dtype.element_ty), mask=mask)


def dequantize(codes, scales, mins, bits: int, group: int, n_out: int, k_in: int,
               out: torch.Tensor | None = None, dtype=torch.bfloat16) -> torch.Tensor:
    if out is None:
        out = torch.empty((n_out, k_in), device=codes.device, dtype=dtype)
    grid = (triton.cdiv(n_out, 64), triton.cdiv(k_in, 128))
    _dequant_kernel[grid](
        codes, scales, mins, out, n_out, k_in,
        codes.stride(0), scales.stride(0), out.stride(0),
        BITS=bits, GROUP=group, BLOCK_N=64, BLOCK_K=128,
    )
    return out


# ---------------------------------------------------------------------------
# Flash-decoding attention over a paged KV pool
# ---------------------------------------------------------------------------

@triton.jit
def _decode_attn_partial(
    q_ptr, k_ptr, v_ptr, table_ptr, len_ptr, po_ptr, plse_ptr,
    stride_qb, stride_qh,
    stride_kblk, stride_kt, stride_kh,
    stride_tb,
    stride_pob, stride_poh, stride_pos,
    stride_lb, stride_lh,
    sm_scale,
    GROUP: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_SPLIT: tl.constexpr, G_PAD: tl.constexpr,
):
    """Partial attention of one (sequence, kv head, split) over paged KV."""
    b = tl.program_id(0)
    hk = tl.program_id(1)
    sp = tl.program_id(2)
    seqlen = tl.load(len_ptr + b)
    start = sp * BLOCKS_PER_SPLIT * BLOCK_SIZE
    offs_g = tl.arange(0, G_PAD)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_t = tl.arange(0, BLOCK_SIZE)
    g_mask = offs_g < GROUP
    qh = hk * GROUP + offs_g
    q = tl.load(q_ptr + b * stride_qb + qh[:, None] * stride_qh + offs_d[None, :],
                mask=g_mask[:, None], other=0.0)
    m_i = tl.full((G_PAD,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((G_PAD,), dtype=tl.float32)
    acc = tl.zeros((G_PAD, HEAD_DIM), dtype=tl.float32)
    for j in range(BLOCKS_PER_SPLIT):
        t0 = start + j * BLOCK_SIZE
        if t0 < seqlen:
            blk = tl.load(table_ptr + b * stride_tb + t0 // BLOCK_SIZE).to(tl.int64)
            n_mask = (t0 + offs_t) < seqlen
            kv_off = blk * stride_kblk + hk * stride_kh + offs_t[:, None] * stride_kt + offs_d[None, :]
            k = tl.load(k_ptr + kv_off, mask=n_mask[:, None], other=0.0)
            qk = tl.dot(q, tl.trans(k)) * sm_scale
            qk = tl.where(n_mask[None, :], qk, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            v = tl.load(v_ptr + kv_off, mask=n_mask[:, None], other=0.0)
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
            m_i = m_new
    has = start < seqlen
    out = acc / tl.where(l_i > 0, l_i, 1.0)[:, None]
    lse = tl.where(has, m_i + tl.log(tl.where(l_i > 0, l_i, 1.0)), float("-inf"))
    tl.store(po_ptr + b * stride_pob + qh[:, None] * stride_poh + sp * stride_pos + offs_d[None, :],
             out, mask=g_mask[:, None])
    tl.store(plse_ptr + b * stride_lb + qh * stride_lh + sp, lse, mask=g_mask)


@triton.jit
def _decode_attn_combine(
    po_ptr, plse_ptr, out_ptr,
    stride_pob, stride_poh, stride_pos,
    stride_lb, stride_lh,
    stride_ob, stride_oh,
    NSPLIT: tl.constexpr, NSPLIT_PAD: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    offs_s = tl.arange(0, NSPLIT_PAD)
    s_mask = offs_s < NSPLIT
    lse = tl.load(plse_ptr + b * stride_lb + h * stride_lh + offs_s, mask=s_mask, other=float("-inf"))
    m = tl.max(lse, axis=0)
    w = tl.exp(lse - m)
    w = tl.where(s_mask, w, 0.0)
    offs_d = tl.arange(0, HEAD_DIM)
    po = tl.load(po_ptr + b * stride_pob + h * stride_poh + offs_s[:, None] * stride_pos + offs_d[None, :],
                 mask=s_mask[:, None], other=0.0)
    out = tl.sum(po * w[:, None], axis=0) / tl.sum(w, axis=0)
    tl.store(out_ptr + b * stride_ob + h * stride_oh + offs_d, out.to(out_ptr.dtype.element_ty))


class DecodeAttention:
    """Flash-decoding over a paged KV pool.

    KV pools are [n_blocks, BLOCK_SIZE, n_kv_heads, head_dim]; each sequence
    owns the blocks listed in its row of the block table. Partial buffers are
    sized for (max_batch, n_heads, max_len) and reused by every call.
    """

    BLOCKS_PER_SPLIT = 4

    def __init__(self, max_batch: int, n_heads: int, n_kv_heads: int, head_dim: int,
                 max_len: int, block_size: int, device):
        self.n_heads, self.n_kv_heads, self.head_dim = n_heads, n_kv_heads, head_dim
        self.block_size = block_size
        self.group = n_heads // n_kv_heads
        self.nsplit = triton.cdiv(max_len, self.BLOCKS_PER_SPLIT * block_size)
        self.po = torch.empty((max_batch, n_heads, self.nsplit, head_dim), device=device, dtype=torch.float32)
        self.plse = torch.empty((max_batch, n_heads, self.nsplit), device=device, dtype=torch.float32)
        self.sm_scale = head_dim ** -0.5

    def __call__(self, q, k_pool, v_pool, block_table, seqlens):
        """q [B, Hq, D]; pools [NB, BS, Hkv, D]; table int32 [B, MB]; seqlens int32 [B]."""
        if use_fma_kernels(q.device):
            from app.lite.gpu.kernels_fma import decode_attention

            return decode_attention(self, q, k_pool, v_pool, block_table, seqlens)
        B = q.shape[0]
        out = torch.empty_like(q)
        _decode_attn_partial[(B, self.n_kv_heads, self.nsplit)](
            q, k_pool, v_pool, block_table, seqlens, self.po, self.plse,
            q.stride(0), q.stride(1),
            k_pool.stride(0), k_pool.stride(1), k_pool.stride(2),
            block_table.stride(0),
            self.po.stride(0), self.po.stride(1), self.po.stride(2),
            self.plse.stride(0), self.plse.stride(1),
            self.sm_scale,
            GROUP=self.group, HEAD_DIM=self.head_dim, BLOCK_SIZE=self.block_size,
            BLOCKS_PER_SPLIT=self.BLOCKS_PER_SPLIT, G_PAD=16, num_warps=4,
        )
        _decode_attn_combine[(B, self.n_heads)](
            self.po, self.plse, out,
            self.po.stride(0), self.po.stride(1), self.po.stride(2),
            self.plse.stride(0), self.plse.stride(1),
            out.stride(0), out.stride(1),
            NSPLIT=self.nsplit, NSPLIT_PAD=triton.next_power_of_2(self.nsplit),
            HEAD_DIM=self.head_dim, num_warps=4,
        )
        return out


# ---------------------------------------------------------------------------
# Fused elementwise kernels (one launch instead of a chain of small ops)
# ---------------------------------------------------------------------------

@triton.jit
def _add_rmsnorm_kernel(x_ptr, r_ptr, w_ptr, h_ptr, rout_ptr, N, eps,
                        HAS_RES: tl.constexpr, BLOCK: tl.constexpr):
    """rout = x (+ r) rounded to the activation dtype; h = w * rmsnorm(rout).

    Rounding follows HF Qwen3: normalize in fp32, cast to the activation
    dtype, then multiply by the weight.
    """
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0)
    if HAS_RES:
        x = (x.to(tl.float32) + tl.load(r_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32))
        x = x.to(rout_ptr.dtype.element_ty)
        tl.store(rout_ptr + row * N + offs, x, mask=mask)
    xf = x.to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(xf * xf, axis=0) / N + eps)
    y = (xf * rstd).to(h_ptr.dtype.element_ty).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(h_ptr + row * N + offs, (y * w).to(h_ptr.dtype.element_ty), mask=mask)


def add_rmsnorm(x: torch.Tensor, residual: torch.Tensor | None, weight: torch.Tensor, eps: float):
    """Returns (normed, new_residual). With residual=None, new_residual is x."""
    x2 = x.reshape(-1, x.shape[-1])
    M, N = x2.shape
    h = torch.empty_like(x2)
    if residual is None:
        _add_rmsnorm_kernel[(M,)](x2, x2, weight, h, x2, N, eps, HAS_RES=False,
                                  BLOCK=triton.next_power_of_2(N), num_warps=8)
        return h.view(x.shape), x
    r2 = residual.reshape(-1, N)
    rout = torch.empty_like(x2)
    _add_rmsnorm_kernel[(M,)](x2, r2, weight, h, rout, N, eps, HAS_RES=True,
                              BLOCK=triton.next_power_of_2(N), num_warps=8)
    return h.view(x.shape), rout.view(x.shape)


@triton.jit
def _qk_norm_rope_store_kernel(
    qkv_ptr, qn_ptr, kn_ptr, pos_ptr, inv_freq_ptr, q_out_ptr, k_pool_ptr, v_pool_ptr, rows_ptr,
    stride_qkv, stride_qo_t, stride_qo_h, stride_pool_row, stride_pool_h,
    eps, pos_offset, max_pos,
    NQ: tl.constexpr, NKV: tl.constexpr, D: tl.constexpr,
):
    """Per (token, head): q/k head RMSNorm + neox RoPE; k and v go to the KV pool."""
    t = tl.program_id(0)
    h = tl.program_id(1)
    HALF: tl.constexpr = D // 2
    offs = tl.arange(0, HALF)
    base = qkv_ptr + t * stride_qkv + h * D
    row = tl.load(rows_ptr + t).to(tl.int64)
    if h >= NQ + NKV:  # value head: copy
        hv = h - NQ - NKV
        v1 = tl.load(base + offs)
        v2 = tl.load(base + HALF + offs)
        vdst = v_pool_ptr + row * stride_pool_row + hv * stride_pool_h
        tl.store(vdst + offs, v1)
        tl.store(vdst + HALF + offs, v2)
    else:
        x1 = tl.load(base + offs).to(tl.float32)
        x2 = tl.load(base + HALF + offs).to(tl.float32)
        rstd = 1.0 / tl.sqrt((tl.sum(x1 * x1, axis=0) + tl.sum(x2 * x2, axis=0)) / D + eps)
        if h < NQ:
            w1 = tl.load(qn_ptr + offs).to(tl.float32)
            w2 = tl.load(qn_ptr + HALF + offs).to(tl.float32)
        else:
            w1 = tl.load(kn_ptr + offs).to(tl.float32)
            w2 = tl.load(kn_ptr + HALF + offs).to(tl.float32)
        dt = q_out_ptr.dtype.element_ty
        y1 = ((x1 * rstd).to(dt).to(tl.float32) * w1).to(dt).to(tl.float32)
        y2 = ((x2 * rstd).to(dt).to(tl.float32) * w2).to(dt).to(tl.float32)
        p = tl.load(pos_ptr + t).to(tl.int64) + pos_offset
        p = tl.minimum(tl.maximum(p, 0), max_pos - 1).to(tl.float32)
        ang = p * tl.load(inv_freq_ptr + offs)
        c = tl.cos(ang).to(dt).to(tl.float32)
        s = tl.sin(ang).to(dt).to(tl.float32)
        o1 = (y1 * c - y2 * s).to(dt)
        o2 = (y2 * c + y1 * s).to(dt)
        if h < NQ:
            dst = q_out_ptr + t * stride_qo_t + h * stride_qo_h
        else:
            dst = k_pool_ptr + row * stride_pool_row + (h - NQ) * stride_pool_h
        tl.store(dst + offs, o1)
        tl.store(dst + HALF + offs, o2)


def qk_norm_rope_store(qkv, q_norm, k_norm, pos, inv_freq, k_pool, v_pool, rows,
                       n_q: int, n_kv: int, head_dim: int, eps: float,
                       pos_offset: int, max_pos: int):
    """qkv [T, (nq+2nkv)*D] -> q [T, nq, D]; k/v written to pool rows [T]."""
    T = qkv.shape[0]
    q = torch.empty((T, n_q, head_dim), device=qkv.device, dtype=qkv.dtype)
    kp = k_pool.view(-1, n_kv, head_dim)
    vp = v_pool.view(-1, n_kv, head_dim)
    _qk_norm_rope_store_kernel[(T, n_q + 2 * n_kv)](
        qkv, q_norm, k_norm, pos, inv_freq, q, kp, vp, rows,
        qkv.stride(0), q.stride(0), q.stride(1), kp.stride(0), kp.stride(1),
        eps, pos_offset, max_pos,
        NQ=n_q, NKV=n_kv, D=head_dim, num_warps=1,
    )
    return q


@triton.jit
def _silu_mul_kernel(gu_ptr, out_ptr, I, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < I
    g = tl.load(gu_ptr + row * 2 * I + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(gu_ptr + row * 2 * I + I + offs, mask=mask, other=0.0).to(tl.float32)
    s = (g / (1.0 + tl.exp(-g))).to(out_ptr.dtype.element_ty).to(tl.float32)
    tl.store(out_ptr + row * I + offs, (s * u).to(out_ptr.dtype.element_ty), mask=mask)


def silu_mul(gu: torch.Tensor) -> torch.Tensor:
    M, two_i = gu.shape
    inter = two_i // 2
    out = torch.empty((M, inter), device=gu.device, dtype=gu.dtype)
    _silu_mul_kernel[(M, triton.cdiv(inter, 1024))](gu, out, inter, BLOCK=1024, num_warps=4)
    return out
