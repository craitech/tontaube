"""Decode kernels for GPUs without tensor cores (compute capability below 7.0).

Drop-in variants of the quantized GEMV and the paged flash-decoding attention
in app.lite.gpu.kernels, selected by app.lite.gpu.arch. Those kernels pad the
decode rows (GEMV) or the query heads of a KV head (attention) to 16 for
tl.dot; on Pascal tl.dot becomes fp32 multiply-add loops, so one decode row
costs the work of 16 and the int4 GEMV reads its weights at a few percent of
the memory bandwidth. Here the products are explicit fp32 multiply-adds with
reductions, on the rows that exist.

Same contract as the tensor-core kernels: the same packed weight layout,
scales and mins, fp32 accumulation, output in the activation dtype, and the
same partial buffers and combine kernel for attention.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from app.lite.gpu.kernels import _decode_attn_combine, _qgemv_reduce


@triton.jit
def _qgemv_fma_kernel(
    x_ptr, codes_ptr, scales_ptr, mins_ptr, bias_ptr, out_ptr, part_ptr,
    M, N, K,
    stride_xm, stride_cn, stride_sn, stride_om,
    BITS: tl.constexpr, GROUP: tl.constexpr, HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, SUB_K: tl.constexpr, N_SUB: tl.constexpr,
    CHUNKS_PER_SPLIT: tl.constexpr, SPLIT_K: tl.constexpr,
):
    """y[M, N] = x[M, K] @ (codes * scale + min)^T (+ bias), without tl.dot.

    Program (n, split, m) reads BLOCK_N weight rows over its K range, N_SUB *
    SUB_K codes per row at a time (long contiguous reads), unpacks them once
    into K order and multiplies them with the rows m * BLOCK_M ... of x that
    exist. Each sub-chunk of SUB_K codes lies inside one quantization group, so
    scale and min are applied once per (row, sub-chunk): s * (x . q) + min * sum(x).
    Without a K split the result (plus bias) is stored in the output dtype;
    with one, fp32 partials go to part_ptr[split] for _qgemv_reduce.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    m0 = tl.program_id(2) * BLOCK_M
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    rows = tl.arange(0, BLOCK_M)
    offs_s = tl.arange(0, N_SUB)
    offs_k = tl.arange(0, SUB_K)
    offs_h = tl.arange(0, SUB_K // 2)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    BLOCK_K: tl.constexpr = N_SUB * SUB_K
    k_first = pid_k * CHUNKS_PER_SPLIT * BLOCK_K
    for c in range(CHUNKS_PER_SPLIT):
        k0 = k_first + c * BLOCK_K
        if k0 < K:
            ks = k0 + offs_s * SUB_K
            s_mask = ks < K
            ns_mask = n_mask[:, None] & s_mask[None, :]
            g = ks // GROUP
            s = tl.load(scales_ptr + offs_n[:, None] * stride_sn + g[None, :], mask=ns_mask, other=0.0).to(tl.float32)
            mn = tl.load(mins_ptr + offs_n[:, None] * stride_sn + g[None, :], mask=ns_mask, other=0.0).to(tl.float32)
            if BITS == 8:
                q = tl.load(codes_ptr + offs_n[:, None, None] * stride_cn + ks[None, :, None] + offs_k[None, None, :],
                            mask=ns_mask[:, :, None], other=0).to(tl.float32)
            else:
                b = tl.load(codes_ptr + offs_n[:, None, None] * stride_cn + (ks // 2)[None, :, None]
                            + offs_h[None, None, :], mask=ns_mask[:, :, None], other=0)
                q = tl.reshape(tl.join((b & 0xF).to(tl.float32), (b >> 4).to(tl.float32)), (BLOCK_N, N_SUB, SUB_K))
            for i in tl.static_range(BLOCK_M):
                if m0 + i < M:
                    xv = tl.load(x_ptr + (m0 + i) * stride_xm + ks[:, None] + offs_k[None, :],
                                 mask=s_mask[:, None], other=0.0).to(tl.float32)
                    dot = tl.sum(q * xv[None, :, :], axis=2)
                    part = tl.sum(dot * s + tl.sum(xv, axis=1)[None, :] * mn, axis=1)
                    acc = tl.where(rows[:, None] == i, acc + part[None, :], acc)
    offs_m = m0 + rows
    mask = (offs_m < M)[:, None] & n_mask[None, :]
    if SPLIT_K == 1:
        if HAS_BIAS:
            acc += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)[None, :]
        tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :],
                 acc.to(out_ptr.dtype.element_ty), mask=mask)
    else:
        tl.store(part_ptr + pid_k * M * N + offs_m[:, None] * N + offs_n[None, :], acc, mask=mask)


def _launch_qgemv(x, codes, scales, mins, bias, out, part, bits: int, group: int, n: int,
                  config: tuple[int, int, int, int, int, int]) -> None:
    block_m, block_n, n_sub, sub_k, split_k, warps = config
    M, K = x.shape
    has_bias = bias is not None
    n_chunks = triton.cdiv(K, n_sub * sub_k)
    _qgemv_fma_kernel[(triton.cdiv(n, block_n), split_k, triton.cdiv(M, block_m))](
        x, codes, scales, mins, bias if has_bias else out, out, part,
        M, n, K,
        x.stride(0), codes.stride(0), scales.stride(0), out.stride(0),
        BITS=bits, GROUP=group, HAS_BIAS=has_bias, BLOCK_M=block_m, BLOCK_N=block_n, SUB_K=sub_k, N_SUB=n_sub,
        CHUNKS_PER_SPLIT=triton.cdiv(n_chunks, split_k), SPLIT_K=split_k, num_warps=warps,
    )
    if split_k > 1:
        _qgemv_reduce[(M, triton.cdiv(n, 1024))](
            part, bias if has_bias else part, out, M, n, out.stride(0),
            SPLIT_K=split_k, HAS_BIAS=has_bias, BLOCK=1024, num_warps=4,
        )


def qgemv(x: torch.Tensor, codes: torch.Tensor, scales: torch.Tensor, mins: torch.Tensor,
          bits: int, group: int, n_out: int, bias: torch.Tensor | None = None,
          config: tuple[int, int, int, int, int, int] | None = None) -> torch.Tensor:
    """y[M, N] = x[M, K] @ dequant(W)^T (+ bias), in x.dtype, for shapes qgemv_supported() accepts."""
    M, K = x.shape
    config = config or gemv_config(bits, group, n_out, K, M)
    split_k = config[4]
    out = torch.empty((M, n_out), device=x.device, dtype=x.dtype)
    part = torch.empty((split_k, M, n_out), device=x.device, dtype=torch.float32) if split_k > 1 else out
    _launch_qgemv(x, codes, scales, mins, bias, out, part, bits, group, n_out, config)
    return out


def sub_k(bits: int, group: int) -> int:
    """Codes per sub-chunk: the largest power of two that divides the group, at most 64 (int4) or 128 (int8)."""
    return min(group & -group, 64 if bits == 4 else 128)


def gemv_config(bits: int, group: int, n: int, k: int, m: int) -> tuple[int, int, int, int, int, int]:
    """(BLOCK_M, BLOCK_N, N_SUB, SUB_K, SPLIT_K, num_warps) for an M x K by K x N product.

    A shape-only rule, no per-GPU tuning: about 32 dequantized weights per
    thread, weight rows read 128 to 512 bytes at a time, at most four x rows
    per program (more rows run as further programs), and a K split only while
    there are few programs. On a Pascal GPU it was within about 10% of the
    fastest configuration for every TontaubeV1.1 layer shape.
    """
    sk = sub_k(bits, group)
    if m == 1 and bits == 4:
        block_m, block_n, chunk, warps, programs, min_chunks = 1, 8, 512, 4, 512, 4
    elif m == 1:
        block_m, block_n, chunk, warps, programs, min_chunks = 1, 4, 256, 1, 0, 1
    else:
        block_m, block_n, chunk, warps, programs, min_chunks = min(triton.next_power_of_2(m), 4), 4, 64 * bits, 1, 2048, 2
    n_sub = max(1, chunk // sk)
    n_chunks = triton.cdiv(k, n_sub * sk)
    split_k = 1
    while triton.cdiv(n, block_n) * split_k < programs and n_chunks // (split_k * 2) >= min_chunks and split_k < 8:
        split_k *= 2
    return block_m, block_n, n_sub, sk, split_k, warps


def qgemv_supported(m: int, k: int, bits: int, group: int) -> bool:
    """Shapes the FMA GEMV handles; qgemv() falls back to the tensor-core kernel otherwise."""
    return bits in (4, 8) and group > 0 and k % group == 0 and sub_k(bits, group) >= 16


@triton.jit
def _decode_attn_partial_fma(
    q_ptr, k_ptr, v_ptr, table_ptr, len_ptr, po_ptr, plse_ptr,
    stride_qb, stride_qh,
    stride_kblk, stride_kt, stride_kh,
    stride_tb,
    stride_pob, stride_poh, stride_pos,
    stride_lb, stride_lh,
    sm_scale,
    GROUP: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    SPLIT_TOKENS: tl.constexpr, G_PAD: tl.constexpr, SUB: tl.constexpr,
):
    """Partial attention of one (sequence, kv head, split) over paged KV, without tl.dot.

    Like _decode_attn_partial, with the query heads of the KV head padded
    only to a power of two, splits of SPLIT_TOKENS tokens, and the keys and
    values visited SUB tokens at a time (SUB divides the KV block size).
    """
    b = tl.program_id(0)
    hk = tl.program_id(1)
    sp = tl.program_id(2)
    seqlen = tl.load(len_ptr + b)
    start = sp * SPLIT_TOKENS
    offs_g = tl.arange(0, G_PAD)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_t = tl.arange(0, SUB)
    g_mask = offs_g < GROUP
    qh = hk * GROUP + offs_g
    q = tl.load(q_ptr + b * stride_qb + qh[:, None] * stride_qh + offs_d[None, :],
                mask=g_mask[:, None], other=0.0).to(tl.float32)
    m_i = tl.full((G_PAD,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((G_PAD,), dtype=tl.float32)
    acc = tl.zeros((G_PAD, HEAD_DIM), dtype=tl.float32)
    for j in range(SPLIT_TOKENS // SUB):
        t0 = start + j * SUB
        if t0 < seqlen:
            blk = tl.load(table_ptr + b * stride_tb + t0 // BLOCK_SIZE).to(tl.int64)
            n_mask = (t0 + offs_t) < seqlen
            kv_off = (blk * stride_kblk + hk * stride_kh
                      + (t0 % BLOCK_SIZE + offs_t)[:, None] * stride_kt + offs_d[None, :])
            k = tl.load(k_ptr + kv_off, mask=n_mask[:, None], other=0.0).to(tl.float32)
            qk = tl.sum(q[:, None, :] * k[None, :, :], axis=2) * sm_scale
            qk = tl.where(n_mask[None, :], qk, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(qk, axis=1))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            v = tl.load(v_ptr + kv_off, mask=n_mask[:, None], other=0.0).to(tl.float32)
            acc = acc * alpha[:, None] + tl.sum(p[:, :, None] * v[None, :, :], axis=1)
            m_i = m_new
    has = start < seqlen
    out = acc / tl.where(l_i > 0, l_i, 1.0)[:, None]
    lse = tl.where(has, m_i + tl.log(tl.where(l_i > 0, l_i, 1.0)), float("-inf"))
    tl.store(po_ptr + b * stride_pob + qh[:, None] * stride_poh + sp * stride_pos + offs_d[None, :],
             out, mask=g_mask[:, None])
    tl.store(plse_ptr + b * stride_lb + qh * stride_lh + sp, lse, mask=g_mask)


ATTN_SPLIT_TOKENS = 32
ATTN_SUB = 16
ATTN_WARPS = 1


def _partials(att, split_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Partial output and log-sum-exp buffers for splits of split_tokens, kept on the DecodeAttention."""
    cache = att.__dict__.setdefault("_fma_partials", {})
    if split_tokens not in cache:
        max_batch, n_heads, nsplit, head_dim = att.po.shape
        n = triton.cdiv(nsplit * att.BLOCKS_PER_SPLIT * att.block_size, split_tokens)
        cache[split_tokens] = (
            torch.empty((max_batch, n_heads, n, head_dim), device=att.po.device, dtype=torch.float32),
            torch.empty((max_batch, n_heads, n), device=att.po.device, dtype=torch.float32),
        )
    return cache[split_tokens]


def decode_attention(att, q, k_pool, v_pool, block_table, seqlens, split_tokens: int | None = None,
                     sub: int | None = None, warps: int | None = None):
    """DecodeAttention.__call__ with the FMA partial kernel and the same combine kernel."""
    sub = min(sub or ATTN_SUB, att.block_size)
    split_tokens = max(split_tokens or ATTN_SPLIT_TOKENS, sub)
    po, plse = _partials(att, split_tokens)
    nsplit = po.shape[2]
    B = q.shape[0]
    out = torch.empty_like(q)
    _decode_attn_partial_fma[(B, att.n_kv_heads, nsplit)](
        q, k_pool, v_pool, block_table, seqlens, po, plse,
        q.stride(0), q.stride(1),
        k_pool.stride(0), k_pool.stride(1), k_pool.stride(2),
        block_table.stride(0),
        po.stride(0), po.stride(1), po.stride(2),
        plse.stride(0), plse.stride(1),
        att.sm_scale,
        GROUP=att.group, HEAD_DIM=att.head_dim, BLOCK_SIZE=att.block_size,
        SPLIT_TOKENS=split_tokens, G_PAD=triton.next_power_of_2(att.group), SUB=sub,
        num_warps=warps or ATTN_WARPS,
    )
    _decode_attn_combine[(B, att.n_heads)](
        po, plse, out,
        po.stride(0), po.stride(1), po.stride(2),
        plse.stride(0), plse.stride(1),
        out.stride(0), out.stride(1),
        NSPLIT=nsplit, NSPLIT_PAD=triton.next_power_of_2(nsplit),
        HEAD_DIM=att.head_dim, num_warps=4,
    )
    return out
