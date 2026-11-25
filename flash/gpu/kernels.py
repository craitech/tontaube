"""Triton kernels of the engine. Activations are fp16 where a matrix reads them, fp32 in every residual stream.

  linear        y[M, N] = x[M, K] @ W^T for M <= 64 rows straight from the stored weights (dense fp16, int8 or
                int4 codes with a scale and a minimum per group of inputs), with the bias, an activation (Mish, GELU,
                SiLU(gate) * up for Qwen's merged gate/up matrix) or a residual add fused in. Derived from Tontaube's
                lite backend (app/lite/gpu/kernels.py).
  dequantize    packed codes -> dense fp16 (text prefill of more than 64 tokens, then cuBLAS)
  attention     one query per head over rows [0, len) of the static key/value cache (flash decoding)
  add_rmsnorm, qk_norm_rope_store   the Qwen3 layer glue, positions are real numbers
  head_*, ae_*, vv_*   the small fused kernels of the flow head, the autoencoder decoder and the VibeVoice decoder
"""
import torch
import triton
import triton.language as tl

EPI_NONE, EPI_MISH, EPI_GELU, EPI_RES, EPI_GATED = 0, 1, 2, 3, 4


@triton.jit
def _mish(y):
    sp = tl.where(y > 20.0, y, tl.log(1.0 + tl.exp(tl.minimum(y, 20.0))))
    e = tl.exp(-2.0 * sp)
    return y * (1.0 - e) / (1.0 + e)


# ----------------------------------------------------------------------------------------------------------------
# linear for few rows
# ----------------------------------------------------------------------------------------------------------------

@triton.jit
def _acc_tile(x_ptr, w_ptr, s_ptr, mn_ptr, nw_ptr, offs_m, m_mask, offs_n, n_mask, k_first, K, stride_xm, stride_wn, stride_sn, eps,
              BITS: tl.constexpr, GROUP: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
              CHUNKS: tl.constexpr, M_PAD: tl.constexpr, NORM: tl.constexpr):
    """Partial products [M_PAD, BLOCK_N] over CHUNKS input chunks from k_first. The rows are padded to M_PAD so
    that the products run on tensor cores; each chunk lies inside one quantization group, so scale and minimum are
    applied once per (row, chunk): s * (x . q) + min * sum(x).
    NORM (> 0: the input width; one row): x is the fp32 residual stream and the layer's RMSNorm is applied here,
    x * rstd * weight rounded to fp16, so the norm costs no launch of its own."""
    acc = tl.zeros((M_PAD, BLOCK_N), dtype=tl.float32)
    if NORM > 0:
        xr = tl.load(x_ptr + tl.arange(0, NORM))
        rstd = 1.0 / tl.sqrt(tl.sum(xr * xr, axis=0) / NORM + eps)
    for c in range(CHUNKS):
        k0 = k_first + c * BLOCK_K
        if k0 < K:
            offs_k = k0 + tl.arange(0, BLOCK_K)
            if NORM > 0:
                x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :], mask=m_mask[:, None], other=0.0)
                x = (x * rstd * tl.load(nw_ptr + offs_k)[None, :]).to(tl.float16)
            else:
                x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :], mask=m_mask[:, None], other=0.0)
            if BITS == 16:
                w = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :], mask=n_mask[:, None], other=0.0)
                acc += tl.dot(x, tl.trans(w))
            else:
                g = k0 // GROUP
                s = tl.load(s_ptr + offs_n * stride_sn + g, mask=n_mask, other=0.0).to(tl.float32)
                mn = tl.load(mn_ptr + offs_n * stride_sn + g, mask=n_mask, other=0.0).to(tl.float32)
                if BITS == 8:
                    q = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :], mask=n_mask[:, None], other=0).to(tl.float16)
                else:
                    HALF: tl.constexpr = BLOCK_K // 2
                    b = tl.load(w_ptr + offs_n[:, None] * stride_wn + (k0 // 2 + tl.arange(0, HALF))[None, :], mask=n_mask[:, None], other=0)
                    # 0x4C00 | (q << 6) is the fp16 bit pattern of 16 + q (q < 16): no int -> float conversion; the 16
                    # goes into the minimum's term
                    b16 = b.to(tl.uint16)
                    q = tl.interleave((((b16 & 0xF) << 6) | 0x4C00).to(tl.float16, bitcast=True),
                                      (((b16 >> 4) << 6) | 0x4C00).to(tl.float16, bitcast=True))
                    mn = mn - 16.0 * s
                acc += tl.dot(x, tl.trans(q)) * s[None, :] + tl.sum(x.to(tl.float32), axis=1)[:, None] * mn[None, :]
    return acc


@triton.jit
def _finish(acc, acc2, bias_ptr, res_ptr, gate_ptr, out_ptr, offs_m, offs_n, mask, n_mask, stride_om, stride_rm, stride_gm, out_scale,
            HAS_BIAS: tl.constexpr, EPI: tl.constexpr, PAIR: tl.constexpr):
    """Bias, activation or residual, store. PAIR: acc = gate rows, acc2 = up rows -> SiLU(gate) * up * out_scale."""
    if PAIR:
        y = acc / (1.0 + tl.exp(-acc)) * acc2 * out_scale
        y = tl.minimum(tl.maximum(y, -65000.0), 65000.0)
    else:
        y = acc
        if HAS_BIAS:
            y += tl.load(bias_ptr + offs_n, mask=n_mask, other=0.0).to(tl.float32)[None, :]
        if EPI == 1:
            y = _mish(y)
        elif EPI == 2:
            y = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))
        elif EPI == 3:
            y = tl.load(res_ptr + offs_m[:, None] * stride_rm + offs_n[None, :], mask=mask, other=0.0).to(tl.float32) + y * out_scale
        elif EPI == 4:
            g = tl.load(gate_ptr + offs_m[:, None] * stride_gm + offs_n[None, :], mask=mask, other=0.0).to(tl.float32)
            y = tl.load(res_ptr + offs_m[:, None] * stride_rm + offs_n[None, :], mask=mask, other=0.0).to(tl.float32) + g * y
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :], y.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _lin_kernel(x_ptr, w_ptr, s_ptr, mn_ptr, bias_ptr, res_ptr, gate_ptr, out_ptr, part_ptr, nw_ptr,
                M, N, K, stride_xm, stride_wn, stride_sn, stride_om, stride_rm, stride_gm, out_scale, eps,
                BITS: tl.constexpr, GROUP: tl.constexpr, HAS_BIAS: tl.constexpr, EPI: tl.constexpr, PAIR: tl.constexpr,
                BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, CHUNKS_PER_SPLIT: tl.constexpr, SPLIT_K: tl.constexpr,
                M_PAD: tl.constexpr, NORM: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    n_mask = offs_n < N
    offs_m = tl.arange(0, M_PAD)
    m_mask = offs_m < M
    k_first = pid_k * CHUNKS_PER_SPLIT * BLOCK_K
    acc = _acc_tile(x_ptr, w_ptr, s_ptr, mn_ptr, nw_ptr, offs_m, m_mask, offs_n, n_mask, k_first, K, stride_xm, stride_wn, stride_sn, eps,
                    BITS, GROUP, BLOCK_N, BLOCK_K, CHUNKS_PER_SPLIT, M_PAD, NORM)
    if PAIR:
        acc2 = _acc_tile(x_ptr, w_ptr, s_ptr, mn_ptr, nw_ptr, offs_m, m_mask, offs_n + N, n_mask, k_first, K, stride_xm, stride_wn, stride_sn, eps,
                         BITS, GROUP, BLOCK_N, BLOCK_K, CHUNKS_PER_SPLIT, M_PAD, NORM)
    else:
        acc2 = acc
    mask = m_mask[:, None] & n_mask[None, :]
    if SPLIT_K == 1:
        _finish(acc, acc2, bias_ptr, res_ptr, gate_ptr, out_ptr, offs_m, offs_n, mask, n_mask, stride_om, stride_rm, stride_gm, out_scale,
                HAS_BIAS, EPI, PAIR)
    else:
        base = part_ptr + pid_k * (2 * M * N) + offs_m[:, None] * N + offs_n[None, :]
        tl.store(base, acc, mask=mask)
        if PAIR:
            tl.store(base + M * N, acc2, mask=mask)


@triton.jit
def _lin_reduce(part_ptr, bias_ptr, res_ptr, gate_ptr, out_ptr, M, N, stride_om, stride_rm, stride_gm, out_scale,
                SPLIT_K: tl.constexpr, HAS_BIAS: tl.constexpr, EPI: tl.constexpr, PAIR: tl.constexpr, BLOCK: tl.constexpr):
    offs_m = tl.program_id(0) + tl.arange(0, 1)
    offs_n = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    n_mask = offs_n < N
    mask = (offs_m < M)[:, None] & n_mask[None, :]
    acc = tl.zeros((1, BLOCK), dtype=tl.float32)
    acc2 = tl.zeros((1, BLOCK), dtype=tl.float32)
    for k in tl.static_range(SPLIT_K):
        base = part_ptr + k * (2 * M * N) + offs_m[:, None] * N + offs_n[None, :]
        acc += tl.load(base, mask=mask, other=0.0)
        if PAIR:
            acc2 += tl.load(base + M * N, mask=mask, other=0.0)
    _finish(acc, acc2, bias_ptr, res_ptr, gate_ptr, out_ptr, offs_m, offs_n, mask, n_mask, stride_om, stride_rm, stride_gm, out_scale,
            HAS_BIAS, EPI, PAIR)


def launch_linear(x, w, scales, mins, bias, res, gate, out, part, bits, group, n, epi, pair, out_scale, block_n, split_k, warps,
                  norm_w=None, eps=0.0):
    """x [M, K] fp16 -> out [M, n]. w: dense fp16 [n(*2), K] or packed codes. pair: n = half the rows of w.
    norm_w: x is one fp32 row of a residual stream, RMSNorm with this weight is applied first."""
    M, K = x.shape
    assert norm_w is None or M == 1
    block_k = min(128, group) if bits != 16 else (128 if K % 128 == 0 else 64)
    n_chunks = K // block_k
    m_pad = 16 if M <= 16 else (32 if M <= 32 else 64)
    if m_pad > 16:
        block_n = min(block_n, 64)
    d = out
    _lin_kernel[(triton.cdiv(n, block_n), split_k)](
        x, w, scales if scales is not None else d, mins if mins is not None else d, bias if bias is not None else d,
        res if res is not None else d, gate if gate is not None else d, out, part if part is not None else d,
        norm_w if norm_w is not None else d,
        M, n, K, x.stride(0), w.stride(0), scales.stride(0) if scales is not None else 0, out.stride(0),
        res.stride(0) if res is not None else 0, (gate.stride(0) if gate.dim() == 2 else 0) if gate is not None else 0, out_scale, eps,
        BITS=bits, GROUP=group, HAS_BIAS=bias is not None, EPI=epi, PAIR=pair, BLOCK_N=block_n, BLOCK_K=block_k,
        CHUNKS_PER_SPLIT=triton.cdiv(n_chunks, split_k), SPLIT_K=split_k, M_PAD=m_pad, NORM=K if norm_w is not None else 0,
        num_warps=warps)
    if split_k > 1:
        _lin_reduce[(M, triton.cdiv(n, 1024))](
            part, bias if bias is not None else d, res if res is not None else d, gate if gate is not None else d, out,
            M, n, out.stride(0), res.stride(0) if res is not None else 0,
            (gate.stride(0) if gate.dim() == 2 else 0) if gate is not None else 0, out_scale,
            SPLIT_K=split_k, HAS_BIAS=bias is not None, EPI=epi, PAIR=pair, BLOCK=1024, num_warps=4)


# ----------------------------------------------------------------------------------------------------------------
# dequantize (prefill of more than 64 tokens)
# ----------------------------------------------------------------------------------------------------------------

@triton.jit
def _dequant_kernel(codes_ptr, scales_ptr, mins_ptr, out_ptr, N, stride_cn, stride_sn, stride_on,
                    BITS: tl.constexpr, GROUP: tl.constexpr, BLOCK_N: tl.constexpr):
    """One program per (block of rows, group of inputs)."""
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    g = tl.program_id(1)
    mask = offs_n < N
    s = tl.load(scales_ptr + offs_n * stride_sn + g, mask=mask, other=0.0).to(tl.float32)[:, None]
    mn = tl.load(mins_ptr + offs_n * stride_sn + g, mask=mask, other=0.0).to(tl.float32)[:, None]
    offs_k = g * GROUP + tl.arange(0, GROUP)
    if BITS == 8:
        c = tl.load(codes_ptr + offs_n[:, None] * stride_cn + offs_k[None, :], mask=mask[:, None], other=0).to(tl.float32)
        w = c * s + mn
    else:
        HALF: tl.constexpr = GROUP // 2
        b = tl.load(codes_ptr + offs_n[:, None] * stride_cn + (g * HALF + tl.arange(0, HALF))[None, :], mask=mask[:, None], other=0)
        w = tl.interleave((b & 0xF).to(tl.float32) * s + mn, (b >> 4).to(tl.float32) * s + mn)
    tl.store(out_ptr + offs_n[:, None] * stride_on + offs_k[None, :], w.to(out_ptr.dtype.element_ty), mask=mask[:, None])


def dequantize(codes, scales, mins, bits: int, group: int, n: int, k: int, out=None, dtype=torch.float16):
    if out is None:
        out = torch.empty((n, k), device=codes.device, dtype=dtype)
    _dequant_kernel[(triton.cdiv(n, 64), k // group)](codes, scales, mins, out, n, codes.stride(0), scales.stride(0), out.stride(0),
                                                     BITS=bits, GROUP=group, BLOCK_N=64, num_warps=4)
    return out


# ----------------------------------------------------------------------------------------------------------------
# Qwen3 layer glue
# ----------------------------------------------------------------------------------------------------------------

@triton.jit
def _add_rmsnorm_kernel(y_ptr, r_ptr, w_ptr, h_ptr, rout_ptr, N, eps, y_scale, HAS_RES: tl.constexpr, BLOCK: tl.constexpr):
    """rout = r + y_scale * y (fp32; without a residual rout = y); h = w * rmsnorm(rout)."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(y_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    if HAS_RES:
        x = tl.load(r_ptr + row * N + offs, mask=mask, other=0.0) + x * y_scale
        tl.store(rout_ptr + row * N + offs, x, mask=mask)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0)
    tl.store(h_ptr + row * N + offs, (x * rstd * w).to(h_ptr.dtype.element_ty), mask=mask)


def add_rmsnorm(y, residual, weight, eps, h, y_scale=1.0):
    """y [M, N] (any float type) added to the fp32 residual in place (None: y is the stream) -> h [M, N] normed."""
    M, N = y.shape
    _add_rmsnorm_kernel[(M,)](y, residual if residual is not None else y, weight, h, residual if residual is not None else y,
                              N, eps, y_scale, HAS_RES=residual is not None, BLOCK=triton.next_power_of_2(N), num_warps=8)
    return h


@triton.jit
def _qk_norm_rope_store_kernel(qkv_ptr, qn_ptr, kn_ptr, pos_ptr, inv_freq_ptr, q_out_ptr, k_ptr, v_ptr, row_ptr,
                               stride_qkv, stride_qo_t, stride_qo_h, stride_row, stride_h, eps,
                               NQ: tl.constexpr, NKV: tl.constexpr, D: tl.constexpr, PER_TOKEN: tl.constexpr):
    """Per (token, head): q / k head RMSNorm and rotary position (a real number); k and v go to cache row row0 + t
    (PER_TOKEN: to row rows[t])."""
    t = tl.program_id(0)
    h = tl.program_id(1)
    HALF: tl.constexpr = D // 2
    offs = tl.arange(0, HALF)
    base = qkv_ptr + t * stride_qkv + h * D
    if PER_TOKEN:
        row = tl.load(row_ptr + t).to(tl.int64)
    else:
        row = tl.load(row_ptr).to(tl.int64) + t
    if h >= NQ + NKV:
        vdst = v_ptr + row * stride_row + (h - NQ - NKV) * stride_h
        tl.store(vdst + offs, tl.load(base + offs).to(v_ptr.dtype.element_ty))
        tl.store(vdst + HALF + offs, tl.load(base + HALF + offs).to(v_ptr.dtype.element_ty))
    else:
        x1 = tl.load(base + offs).to(tl.float32)
        x2 = tl.load(base + HALF + offs).to(tl.float32)
        rstd = 1.0 / tl.sqrt((tl.sum(x1 * x1, axis=0) + tl.sum(x2 * x2, axis=0)) / D + eps)
        if h < NQ:
            w1 = tl.load(qn_ptr + offs)
            w2 = tl.load(qn_ptr + HALF + offs)
        else:
            w1 = tl.load(kn_ptr + offs)
            w2 = tl.load(kn_ptr + HALF + offs)
        y1 = x1 * rstd * w1
        y2 = x2 * rstd * w2
        ang = tl.load(pos_ptr + t) * tl.load(inv_freq_ptr + offs)
        c = tl.cos(ang)
        s = tl.sin(ang)
        o1 = y1 * c - y2 * s
        o2 = y2 * c + y1 * s
        if h < NQ:
            dst = q_out_ptr + t * stride_qo_t + h * stride_qo_h
            tl.store(dst + offs, o1.to(q_out_ptr.dtype.element_ty))
            tl.store(dst + HALF + offs, o2.to(q_out_ptr.dtype.element_ty))
        else:
            dst = k_ptr + row * stride_row + (h - NQ) * stride_h
            tl.store(dst + offs, o1.to(k_ptr.dtype.element_ty))
            tl.store(dst + HALF + offs, o2.to(k_ptr.dtype.element_ty))


def qk_norm_rope_store(qkv, q_norm, k_norm, pos, inv_freq, k_cache, v_cache, row0, q_out, n_q, n_kv, head_dim, eps, per_token=False):
    """qkv [T, (nq + 2 nkv) * D] -> q_out [T, nq, D]; k / v written to cache rows row0[0] + t (per_token: row0[t]).
    pos [T] fp32."""
    T = qkv.shape[0]
    _qk_norm_rope_store_kernel[(T, n_q + 2 * n_kv)](
        qkv, q_norm, k_norm, pos, inv_freq, q_out, k_cache, v_cache, row0,
        qkv.stride(0), q_out.stride(0), q_out.stride(1), k_cache.stride(0), k_cache.stride(1), eps,
        NQ=n_q, NKV=n_kv, D=head_dim, PER_TOKEN=per_token, num_warps=1)
    return q_out


@triton.jit
def _attn_partial(q_ptr, k_ptr, v_ptr, len_ptr, po_ptr, plse_ptr, stride_qh, stride_kt, stride_kh, stride_poh, stride_pos,
                  stride_lh, sm_scale,
                  GROUP: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_SIZE: tl.constexpr, BLOCKS_PER_SPLIT: tl.constexpr,
                  G_PAD: tl.constexpr):
    """Partial attention of one (kv head, split) over cache rows [0, len)."""
    hk = tl.program_id(0)
    sp = tl.program_id(1)
    seqlen = tl.load(len_ptr).to(tl.int32)
    start = sp * BLOCKS_PER_SPLIT * BLOCK_SIZE
    offs_g = tl.arange(0, G_PAD)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_t = tl.arange(0, BLOCK_SIZE)
    g_mask = offs_g < GROUP
    qh = hk * GROUP + offs_g
    q = tl.load(q_ptr + qh[:, None] * stride_qh + offs_d[None, :], mask=g_mask[:, None], other=0.0)
    m_i = tl.full((G_PAD,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((G_PAD,), dtype=tl.float32)
    acc = tl.zeros((G_PAD, HEAD_DIM), dtype=tl.float32)
    for j in range(BLOCKS_PER_SPLIT):
        t0 = start + j * BLOCK_SIZE
        if t0 < seqlen:
            n_mask = (t0 + offs_t) < seqlen
            kv_off = (t0 + offs_t)[:, None] * stride_kt + hk * stride_kh + offs_d[None, :]
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
    tl.store(po_ptr + qh[:, None] * stride_poh + sp * stride_pos + offs_d[None, :], out, mask=g_mask[:, None])
    tl.store(plse_ptr + qh * stride_lh + sp, lse, mask=g_mask)


@triton.jit
def _attn_combine(po_ptr, plse_ptr, out_ptr, stride_poh, stride_pos, stride_lh, stride_oh, stride_pob, stride_lb, stride_ob,
                  NSPLIT: tl.constexpr, NSPLIT_PAD: tl.constexpr, HEAD_DIM: tl.constexpr):
    """The splits of one (head, row) combined; program axis 1: the row of a batch."""
    h = tl.program_id(0)
    b = tl.program_id(1)
    po_ptr += b * stride_pob
    plse_ptr += b * stride_lb
    out_ptr += b * stride_ob
    offs_s = tl.arange(0, NSPLIT_PAD)
    s_mask = offs_s < NSPLIT
    lse = tl.load(plse_ptr + h * stride_lh + offs_s, mask=s_mask, other=float("-inf"))
    w = tl.exp(lse - tl.max(lse, axis=0))
    w = tl.where(s_mask, w, 0.0)
    offs_d = tl.arange(0, HEAD_DIM)
    po = tl.load(po_ptr + h * stride_poh + offs_s[:, None] * stride_pos + offs_d[None, :], mask=s_mask[:, None], other=0.0)
    out = tl.sum(po * w[:, None], axis=0) / tl.sum(w, axis=0)
    tl.store(out_ptr + h * stride_oh + offs_d, out.to(out_ptr.dtype.element_ty))


@triton.jit
def _attn_partial_paged(q_ptr, k_ptr, v_ptr, len_ptr, tab_ptr, po_ptr, plse_ptr, stride_qb, stride_qh, stride_kt, stride_kh,
                        stride_tb, stride_pob, stride_poh, stride_pos, stride_lb, stride_lh, sm_scale,
                        GROUP: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_SIZE: tl.constexpr, BLOCKS_PER_SPLIT: tl.constexpr,
                        G_PAD: tl.constexpr):
    """_attn_partial for one row of a batch: its cache rows [0, len) lie in pages of BLOCK_SIZE rows, page j at pool
    row table[j] * BLOCK_SIZE."""
    b = tl.program_id(0)
    hk = tl.program_id(1)
    sp = tl.program_id(2)
    seqlen = tl.load(len_ptr + b).to(tl.int32)
    start = sp * BLOCKS_PER_SPLIT * BLOCK_SIZE
    offs_g = tl.arange(0, G_PAD)
    offs_d = tl.arange(0, HEAD_DIM)
    offs_t = tl.arange(0, BLOCK_SIZE)
    g_mask = offs_g < GROUP
    qh = hk * GROUP + offs_g
    q = tl.load(q_ptr + b * stride_qb + qh[:, None] * stride_qh + offs_d[None, :], mask=g_mask[:, None], other=0.0)
    m_i = tl.full((G_PAD,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((G_PAD,), dtype=tl.float32)
    acc = tl.zeros((G_PAD, HEAD_DIM), dtype=tl.float32)
    for j in range(BLOCKS_PER_SPLIT):
        t0 = start + j * BLOCK_SIZE
        if t0 < seqlen:
            page = tl.load(tab_ptr + b * stride_tb + t0 // BLOCK_SIZE).to(tl.int64)
            n_mask = (t0 + offs_t) < seqlen
            kv_off = (page * BLOCK_SIZE + offs_t)[:, None] * stride_kt + hk * stride_kh + offs_d[None, :]
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
    tl.store(po_ptr + b * stride_pob + qh[:, None] * stride_poh + sp * stride_pos + offs_d[None, :], out, mask=g_mask[:, None])
    tl.store(plse_ptr + b * stride_lb + qh * stride_lh + sp, lse, mask=g_mask)


class DecodeAttention:
    """One query token over rows [0, len) of a cache [S, n_kv, D]; len is read on the device."""

    def __init__(self, n_heads, n_kv_heads, head_dim, max_len, device, block_size=64, blocks_per_split=2):
        self.n_heads, self.n_kv, self.hd = n_heads, n_kv_heads, head_dim
        self.block, self.bps = block_size, blocks_per_split
        self.group = n_heads // n_kv_heads
        self.nsplit = triton.cdiv(max_len, blocks_per_split * block_size)
        self.po = torch.empty((n_heads, self.nsplit, head_dim), device=device, dtype=torch.float32)
        self.plse = torch.empty((n_heads, self.nsplit), device=device, dtype=torch.float32)
        self.scale = head_dim ** -0.5

    def __call__(self, q, k, v, seqlen, out):
        """q [nq, D] fp16, k / v [S, nkv, D], seqlen [1] -> out [nq, D]."""
        _attn_partial[(self.n_kv, self.nsplit)](
            q, k, v, seqlen, self.po, self.plse, q.stride(0), k.stride(0), k.stride(1), self.po.stride(0), self.po.stride(1),
            self.plse.stride(0), self.scale, GROUP=self.group, HEAD_DIM=self.hd, BLOCK_SIZE=self.block,
            BLOCKS_PER_SPLIT=self.bps, G_PAD=16, num_warps=4)
        _attn_combine[(self.n_heads, 1)](
            self.po, self.plse, out, self.po.stride(0), self.po.stride(1), self.plse.stride(0), out.stride(0), 0, 0, 0,
            NSPLIT=self.nsplit, NSPLIT_PAD=triton.next_power_of_2(self.nsplit), HEAD_DIM=self.hd, num_warps=4)
        return out


class PagedDecodeAttention:
    """One query token per row of a batch over the row's cache rows [0, len), kept in pages of `page` rows of a pool
    [pages * page, n_kv, D]; table [rows, pages per row] (int32) and len [rows] are read on the device."""

    def __init__(self, n_heads, n_kv_heads, head_dim, max_len, max_rows, device, page=64, blocks_per_split=2):
        self.n_heads, self.n_kv, self.hd, self.page, self.bps = n_heads, n_kv_heads, head_dim, page, blocks_per_split
        self.group = n_heads // n_kv_heads
        self.nsplit = triton.cdiv(max_len, blocks_per_split * page)
        self.po = torch.empty((max_rows, n_heads, self.nsplit, head_dim), device=device, dtype=torch.float32)
        self.plse = torch.empty((max_rows, n_heads, self.nsplit), device=device, dtype=torch.float32)
        self.scale = head_dim ** -0.5

    def __call__(self, q, k, v, table, seqlen, out):
        """q [B, nq, D] fp16, k / v [R, nkv, D], table [B, P], seqlen [B] -> out [B, nq * D]."""
        B = q.shape[0]
        po, pl = self.po, self.plse
        _attn_partial_paged[(B, self.n_kv, self.nsplit)](
            q, k, v, seqlen, table, po, pl, q.stride(0), q.stride(1), k.stride(0), k.stride(1), table.stride(0),
            po.stride(0), po.stride(1), po.stride(2), pl.stride(0), pl.stride(1), self.scale,
            GROUP=self.group, HEAD_DIM=self.hd, BLOCK_SIZE=self.page, BLOCKS_PER_SPLIT=self.bps, G_PAD=16, num_warps=4)
        _attn_combine[(self.n_heads, B)](
            po, pl, out, po.stride(1), po.stride(2), pl.stride(1), self.hd, po.stride(0), pl.stride(0), out.stride(0),
            NSPLIT=self.nsplit, NSPLIT_PAD=triton.next_power_of_2(self.nsplit), HEAD_DIM=self.hd, num_warps=4)
        return out


@triton.jit
def _causal_attn_kernel(q_ptr, k_ptr, v_ptr, out_ptr, T, stride_qt, stride_qh, stride_kt, stride_kh, sm_scale,
                        GROUP: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr):
    """Causal attention of prefix rows [0, T) over themselves (keys / values already in the cache), one program per
    (block of queries, head): the text prefill's attention in one launch."""
    qb = tl.program_id(0)
    h = tl.program_id(1)
    hk = h // GROUP
    offs_q = qb * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, HEAD_DIM)
    q_mask = offs_q < T
    q = tl.load(q_ptr + offs_q[:, None] * stride_qt + h * stride_qh + offs_d[None, :], mask=q_mask[:, None], other=0.0)
    m_i = tl.full((BLOCK_Q,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_Q, HEAD_DIM), dtype=tl.float32)
    hi = tl.minimum((qb + 1) * BLOCK_Q, T)
    for start in range(0, hi, BLOCK_K):
        offs_k = start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < hi
        kv_off = offs_k[:, None] * stride_kt + hk * stride_kh + offs_d[None, :]
        k = tl.load(k_ptr + kv_off, mask=k_mask[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k)) * sm_scale
        qk = tl.where(offs_k[None, :] <= offs_q[:, None], qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(v_ptr + kv_off, mask=k_mask[:, None], other=0.0)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    out = acc / tl.where(l_i > 0, l_i, 1.0)[:, None]
    tl.store(out_ptr + offs_q[:, None] * stride_qt + h * stride_qh + offs_d[None, :], out.to(out_ptr.dtype.element_ty), mask=q_mask[:, None])


def causal_attention(q, k, v, out, n_kv):
    """q [T, nq, D] fp16; k / v [S, nkv, D] (rows [0, T) are the prefix) -> out [T, nq, D]."""
    T, nq, D = q.shape
    _causal_attn_kernel[(triton.cdiv(T, 16), nq)](q, k, v, out, T, q.stride(0), q.stride(1), k.stride(0), k.stride(1), D ** -0.5,
                                                   GROUP=nq // n_kv, HEAD_DIM=D, BLOCK_Q=16, BLOCK_K=64, num_warps=4)
    return out


@triton.jit
def _silu_mul_kernel(gu_ptr, out_ptr, I, stride_g, out_scale, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < I
    g = tl.load(gu_ptr + row * stride_g + offs, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(gu_ptr + row * stride_g + I + offs, mask=mask, other=0.0).to(tl.float32)
    y = g / (1.0 + tl.exp(-g)) * u * out_scale
    y = tl.minimum(tl.maximum(y, -65000.0), 65000.0)
    tl.store(out_ptr + row * I + offs, y.to(out_ptr.dtype.element_ty), mask=mask)


def silu_mul(gu, out, out_scale):
    """gu [M, 2 I] (gate, up) -> out [M, I] = SiLU(gate) * up * out_scale (prefill; the decode step has it in the linear)."""
    M, two_i = gu.shape
    _silu_mul_kernel[(M, triton.cdiv(two_i // 2, 1024))](gu, out, two_i // 2, gu.stride(0), out_scale, BLOCK=1024, num_warps=4)
    return out


# ----------------------------------------------------------------------------------------------------------------
# small kernels: frame input, flow head, autoencoder, VibeVoice decoder. One program (or one row of programs) per row
# of a batch; a single request is a batch of one. The decoders' states are kept per request in rows of an arena: batch
# row b uses arena row sel[b] (without sel: the arena itself)
# ----------------------------------------------------------------------------------------------------------------

def _rows(t):
    """Rows of a [B, ...] tensor and its row stride (a 1-D tensor: one row)."""
    return (t.shape[0], t.stride(0)) if t.dim() == 2 else (1, 0)


@triton.jit
def _slot_base(arena_ptr, sel_ptr, b, astride, HAS_SEL: tl.constexpr):
    if HAS_SEL:
        return arena_ptr + tl.load(sel_ptr + b).to(tl.int64) * astride
    return arena_ptr


@triton.jit
def _small_lin_kernel(x_ptr, w_ptr, b_ptr, out_ptr, idx_ptr, x_scale, stride_x, stride_o, N: tl.constexpr, K: tl.constexpr,
                      ACT: tl.constexpr, INDEXED: tl.constexpr):
    """out[N] = act(W[N, K] x[K] * x_scale + b), W fp32, K small. INDEXED: x is row idx[0] of a table."""
    r = tl.program_id(0)
    offs_n = tl.arange(0, N)
    offs_k = tl.arange(0, K)
    if INDEXED:
        x = tl.load(x_ptr + tl.load(idx_ptr).to(tl.int64) * K + offs_k).to(tl.float32) * x_scale
    else:
        x = tl.load(x_ptr + r * stride_x + offs_k).to(tl.float32) * x_scale
    w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :])
    y = tl.sum(w * x[None, :], axis=1) + tl.load(b_ptr + offs_n)
    if ACT == 1:
        y = _mish(y)
    tl.store(out_ptr + r * stride_o + offs_n, y.to(out_ptr.dtype.element_ty))


def small_linear(x, w, b, out, act=0, idx=None, x_scale=1.0):
    """x [B, K] -> out [B, N]."""
    N, K = w.shape
    B, sx = _rows(x) if idx is None else (1, 0)
    _small_lin_kernel[(B,)](x, w, b, out, idx if idx is not None else x, x_scale, sx, _rows(out)[1], N=N, K=K, ACT=act,
                            INDEXED=idx is not None, num_warps=8)
    return out


@triton.jit
def _wide_out_kernel(y_ptr, w_ptr, b_ptr, out_ptr, K, stride_y, stride_o, N: tl.constexpr, BLOCK_K: tl.constexpr):
    """out[N] = W[N, K] y[K] + b for a few outputs (N <= 64), W fp32, y any float type."""
    offs_k = tl.arange(0, BLOCK_K)
    mask = offs_k < K
    n = tl.program_id(0)
    r = tl.program_id(1)
    y = tl.load(y_ptr + r * stride_y + offs_k, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + n * K + offs_k, mask=mask, other=0.0)
    tl.store(out_ptr + r * stride_o + n, (tl.sum(w * y, axis=0) + tl.load(b_ptr + n)).to(out_ptr.dtype.element_ty))


def wide_out(y, w, b, out):
    N, K = w.shape
    B, sy = _rows(y)
    _wide_out_kernel[(N, B)](y, w, b, out, K, sy, _rows(out)[1], N=N, BLOCK_K=triton.next_power_of_2(K), num_warps=4)
    return out


@triton.jit
def _head_pre_kernel(h_ptr, nw_ptr, nb_ptr, sw_ptr, sb_ptr, y_ptr, stop_ptr, N: tl.constexpr):
    """From the final state h [N] fp32: the stop value and the head's normed input y = LayerNorm(h) * w + b (fp16)."""
    r = tl.program_id(0)
    offs = tl.arange(0, N)
    h = tl.load(h_ptr + r * N + offs)
    tl.store(stop_ptr + r, tl.sum(h * tl.load(sw_ptr + offs), axis=0) + tl.load(sb_ptr))
    mean = tl.sum(h, axis=0) / N
    d = h - mean
    rs = 1.0 / tl.sqrt(tl.sum(d * d, axis=0) / N + 1e-5)
    tl.store(y_ptr + r * N + offs, (d * rs * tl.load(nw_ptr + offs) + tl.load(nb_ptr + offs)).to(y_ptr.dtype.element_ty))


def head_pre(h, nw, nb, sw, sb, y, stop):
    """h [B, N] (contiguous rows) -> y [B, N], stop [B]."""
    _head_pre_kernel[(h.shape[0],)](h, nw, nb, sw, sb, y, stop, N=h.shape[-1], num_warps=8)


@triton.jit
def _head_ws_kernel(c_ptr, tm_ptr, out_ptr, P, N: tl.constexpr):
    """ws[r, p] = Mish(c[r] + tm[p]) for every head pass p (program r * P + p)."""
    i = tl.program_id(0)
    offs = tl.arange(0, N)
    y = tl.load(c_ptr + (i // P) * N + offs).to(tl.float32) + tl.load(tm_ptr + (i % P) * N + offs)
    tl.store(out_ptr + i * N + offs, _mish(y).to(out_ptr.dtype.element_ty))


def head_ws(c, tm, out):
    """c [B, N], tm [P, N] -> out [B * P, N]."""
    _head_ws_kernel[(_rows(c)[0] * tm.shape[0],)](c, tm, out, tm.shape[0], N=tm.shape[1], num_warps=8)


@triton.jit
def _head_in_kernel(x_ptr, w_ptr, b_ptr, mod_ptr, hx_ptr, y_ptr, stride_mod, N: tl.constexpr, D: tl.constexpr):
    """hx = x_in(x) [N] (fp32) and the first block's input y = LayerNorm(hx) * (1 + scale) + shift (fp16);
    mod_ptr: this pass's and block's (shift, scale, gate)."""
    r = tl.program_id(0)
    mod_ptr += r * stride_mod
    offs = tl.arange(0, N)
    offs_d = tl.arange(0, D)
    x = tl.load(x_ptr + r * D + offs_d)
    hx = tl.sum(tl.load(w_ptr + offs[:, None] * D + offs_d[None, :]) * x[None, :], axis=1) + tl.load(b_ptr + offs)
    tl.store(hx_ptr + r * N + offs, hx)
    mean = tl.sum(hx, axis=0) / N
    d = hx - mean
    rs = 1.0 / tl.sqrt(tl.sum(d * d, axis=0) / N + 1e-5)
    y = d * rs * (1.0 + tl.load(mod_ptr + N + offs)) + tl.load(mod_ptr + offs)
    tl.store(y_ptr + r * N + offs, y.to(y_ptr.dtype.element_ty))


def head_in(x, w, b, mod, hx, y):
    """x [B, D] -> hx, y [B, N]; mod [B, ...] (one row: [...])."""
    _head_in_kernel[(_rows(x)[0],)](x, w, b, mod, hx, y, _rows(mod)[1], N=w.shape[0], D=w.shape[1], num_warps=8)


@triton.jit
def _head_mid_kernel(hx_ptr, mod_ptr, y_ptr, stride_mod, N: tl.constexpr):
    r = tl.program_id(0)
    mod_ptr += r * stride_mod
    offs = tl.arange(0, N)
    hx = tl.load(hx_ptr + r * N + offs)
    mean = tl.sum(hx, axis=0) / N
    d = hx - mean
    rs = 1.0 / tl.sqrt(tl.sum(d * d, axis=0) / N + 1e-5)
    y = d * rs * (1.0 + tl.load(mod_ptr + N + offs)) + tl.load(mod_ptr + offs)
    tl.store(y_ptr + r * N + offs, y.to(y_ptr.dtype.element_ty))


def head_mid(hx, mod, y):
    _head_mid_kernel[(_rows(hx)[0],)](hx, mod, y, _rows(mod)[1], N=hx.shape[-1], num_warps=8)


@triton.jit
def _head_out_kernel(hx_ptr, mod_ptr, w_ptr, b_ptr, base_ptr, dst_ptr, coef, stride_mod, N: tl.constexpr, D: tl.constexpr):
    """v = out(LayerNorm(hx) * (1 + scale) + shift) [D]; dst = base + coef * v (one solver update)."""
    r = tl.program_id(0)
    mod_ptr += r * stride_mod
    offs = tl.arange(0, N)
    offs_d = tl.arange(0, D)
    hx = tl.load(hx_ptr + r * N + offs)
    mean = tl.sum(hx, axis=0) / N
    d = hx - mean
    rs = 1.0 / tl.sqrt(tl.sum(d * d, axis=0) / N + 1e-5)
    y = d * rs * (1.0 + tl.load(mod_ptr + N + offs)) + tl.load(mod_ptr + offs)
    v = tl.sum(tl.load(w_ptr + offs_d[:, None] * N + offs[None, :]) * y[None, :], axis=1) + tl.load(b_ptr + offs_d)
    tl.store(dst_ptr + r * D + offs_d, tl.load(base_ptr + r * D + offs_d) + coef * v)


def head_out(hx, mod, w, b, base, dst, coef):
    _head_out_kernel[(_rows(hx)[0],)](hx, mod, w, b, base, dst, coef, _rows(mod)[1], N=w.shape[1], D=w.shape[0], num_warps=8)


@triton.jit
def _ae_pre_kernel(x_ptr, nw_ptr, nb_ptr, st_ptr, u_ptr, sel_ptr, astride, N: tl.constexpr, KB: tl.constexpr, HAS_SEL: tl.constexpr):
    """r = Mish(LayerNorm(x) * w + b); u = [the block's last KB - 1 values of r, r] (the convolution's input, oldest
    first, fp16); the state moves on by one frame."""
    b = tl.program_id(0)
    st_ptr = _slot_base(st_ptr, sel_ptr, b, astride, HAS_SEL)
    u_ptr += b * KB * N
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + b * N + offs)
    mean = tl.sum(x, axis=0) / N
    d = x - mean
    r = 1.0 / tl.sqrt(tl.sum(d * d, axis=0) / N + 1e-5)
    y = _mish(d * r * tl.load(nw_ptr + offs) + tl.load(nb_ptr + offs)).to(u_ptr.dtype.element_ty)
    if KB > 1:
        offs_s = tl.arange(0, KB - 1)
        st = tl.load(st_ptr + offs_s[:, None] * N + offs[None, :])
        tl.store(u_ptr + offs_s[:, None] * N + offs[None, :], st.to(u_ptr.dtype.element_ty))
        if KB > 2:
            offs_m = tl.arange(0, KB - 2)
            tl.store(st_ptr + offs_m[:, None] * N + offs[None, :], tl.load(st_ptr + (offs_m[:, None] + 1) * N + offs[None, :]))
        tl.store(st_ptr + (KB - 2) * N + offs, y.to(tl.float32))
    tl.store(u_ptr + (KB - 1) * N + offs, y)


def ae_pre(x, nw, nb, state, u, kb, sel=None, astride=0):
    """x [B, N] -> u [B, kb * N]; state: the arena from the block's ring on (row sel[b] for batch row b)."""
    _ae_pre_kernel[(_rows(x)[0],)](x, nw, nb, state if state is not None else x, u, sel if sel is not None else x, astride,
                                   N=x.shape[-1], KB=kb, HAS_SEL=sel is not None, num_warps=8)


@triton.jit
def _vv_stem_kernel(lat_ptr, arena_ptr, off, slot_stride, u_ptr, sel_ptr, astride, C: tl.constexpr, HAS_SEL: tl.constexpr):
    """u = [the last 6 latent frames, this one] (the stem convolution's input, fp16); the state moves on."""
    b = tl.program_id(0)
    arena_ptr = _slot_base(arena_ptr, sel_ptr, b, astride, HAS_SEL)
    lat_ptr += b * C
    u_ptr += b * 7 * C
    flip = tl.load(arena_ptr).to(tl.int64)
    st_in = arena_ptr + off + flip * slot_stride
    st_out = arena_ptr + off + (1 - flip) * slot_stride
    offs = tl.arange(0, C)
    offs_s = tl.arange(0, 8)
    x = tl.load(lat_ptr + offs)
    m6 = (offs_s < 6)[:, None] & (offs < C)[None, :]
    st = tl.load(st_in + offs_s[:, None] * C + offs[None, :], mask=m6, other=0.0)
    tl.store(u_ptr + offs_s[:, None] * C + offs[None, :], st.to(u_ptr.dtype.element_ty), mask=m6)
    tl.store(u_ptr + 6 * C + offs, x.to(u_ptr.dtype.element_ty))
    m5 = (offs_s < 5)[:, None] & (offs < C)[None, :]
    tl.store(st_out + offs_s[:, None] * C + offs[None, :], tl.load(st_in + (offs_s[:, None] + 1) * C + offs[None, :], mask=m5, other=0.0), mask=m5)
    tl.store(st_out + 5 * C + offs, x)


def vv_stem(lat, arena, off, slot_stride, u, c, sel=None, astride=0):
    _vv_stem_kernel[(_rows(lat)[0],)](lat, arena, off, slot_stride, u, sel if sel is not None else arena, astride, C=c,
                                      HAS_SEL=sel is not None, num_warps=4)


@triton.jit
def _vv_pre_kernel(xa_ptr, xb_ptr, phi_ptr, taps_ptr, dwb_ptr, fw_ptr, arena_ptr, off, slot_stride, xm_ptr, hn_ptr, T, C, eps,
                   sel_ptr, astride, HAS_B: tl.constexpr, BLOCK_C: tl.constexpr, ONE: tl.constexpr, HAS_SEL: tl.constexpr):
    """One block's first half, per frame t: x = xa (+ phi * xb); h = rmsnorm(x); xm = x + depthwise conv over the
    last 7 h (state: the 6 before this call; the taps carry the norm weight and the layer scale); hn = rmsnorm(xm) * fw
    (the feed-forward's input, fp16). The new state is the last 6 h. Program axis 1: the request (T frames each)."""
    t = tl.program_id(0)
    b = tl.program_id(1)
    arena_ptr = _slot_base(arena_ptr, sel_ptr, b, astride, HAS_SEL)
    xa_ptr += b * T * C
    xb_ptr += b * T * C
    xm_ptr += b * T * C
    hn_ptr += b * T * C
    flip = tl.load(arena_ptr).to(tl.int64)
    st_in = arena_ptr + off + flip * slot_stride
    st_out = arena_ptr + off + (1 - flip) * slot_stride
    offs = tl.arange(0, BLOCK_C)
    cm = offs < C
    offs_j = tl.arange(0, 8)                                   # 7 taps, padded to 8
    src = t - 6 + offs_j                                       # frame index of tap j; < 0: state row src + 6
    in_x = (src >= 0) & (offs_j < 7)
    in_s = (src < 0) & (offs_j < 7)
    x = tl.load(xa_ptr + src[:, None] * C + offs[None, :], mask=in_x[:, None] & cm[None, :], other=0.0).to(tl.float32)
    if HAS_B:
        xb = tl.load(xb_ptr + src[:, None] * C + offs[None, :], mask=in_x[:, None] & cm[None, :], other=0.0).to(tl.float32)
        x += xb * tl.load(phi_ptr + offs, mask=cm, other=0.0)[None, :]
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=1) / C + eps)
    h = x * rstd[:, None]
    hs = tl.load(st_in + (src[:, None] + 6) * C + offs[None, :], mask=in_s[:, None] & cm[None, :], other=0.0)
    h = tl.where(in_s[:, None], hs, h)
    taps = tl.load(taps_ptr + offs_j[:, None] * C + offs[None, :], mask=(offs_j < 7)[:, None] & cm[None, :], other=0.0)
    y = tl.sum(h * taps, axis=0) + tl.load(dwb_ptr + offs, mask=cm, other=0.0)
    xm = tl.sum(tl.where((offs_j == 6)[:, None], x, 0.0), axis=0) + y
    tl.store(xm_ptr + t * C + offs, xm, mask=cm)
    r2 = 1.0 / tl.sqrt(tl.sum(xm * xm, axis=0) / C + eps)
    tl.store(hn_ptr + t * C + offs, (xm * r2 * tl.load(fw_ptr + offs, mask=cm, other=0.0)).to(hn_ptr.dtype.element_ty), mask=cm)
    hcur = tl.sum(tl.where((offs_j == 6)[:, None], h, 0.0), axis=0)
    if ONE:                                                    # a single frame per call: shift the state by one
        offs_m = tl.arange(0, 8)
        mm = (offs_m < 5)[:, None] & cm[None, :]
        tl.store(st_out + offs_m[:, None] * C + offs[None, :], tl.load(st_in + (offs_m[:, None] + 1) * C + offs[None, :], mask=mm, other=0.0), mask=mm)
        tl.store(st_out + 5 * C + offs, hcur, mask=cm)
    else:
        if t >= T - 6:
            tl.store(st_out + (t - (T - 6)) * C + offs, hcur, mask=cm)


def _vv_warps(C):
    return 8 if C >= 1024 else (4 if C >= 256 else 1)


def vv_pre(xa, xb, phi, taps, dwb, fw, arena, off, slot_stride, xm, hn, eps, B=1, sel=None, astride=0):
    """xa [B * T, C]: T frames of each of B requests."""
    T, C = xa.shape[0] // B, xa.shape[1]
    assert T == 1 or T >= 6
    _vv_pre_kernel[(T, B)](xa, xb if xb is not None else xa, phi if phi is not None else taps, taps, dwb, fw, arena, off, slot_stride,
                           xm, hn, T, C, eps, sel if sel is not None else arena, astride, HAS_B=xb is not None,
                           BLOCK_C=triton.next_power_of_2(C), ONE=T == 1, HAS_SEL=sel is not None, num_warps=_vv_warps(C))


@triton.jit
def _vv_upcat_kernel(xa_ptr, xb_ptr, phi_ptr, arena_ptr, off, slot_stride, u_ptr, T, C, sel_ptr, astride, HAS_B: tl.constexpr,
                     BLOCK_C: tl.constexpr, HAS_SEL: tl.constexpr):
    """The stage output x = xa (+ phi * xb), laid out for the transposed convolution: u[t] = [x[t - 1], x[t]] (fp16),
    x[-1] from the state; the new state is the last frame."""
    t = tl.program_id(0)
    b = tl.program_id(1)
    arena_ptr = _slot_base(arena_ptr, sel_ptr, b, astride, HAS_SEL)
    xa_ptr += b * T * C
    xb_ptr += b * T * C
    u_ptr += b * T * 2 * C
    flip = tl.load(arena_ptr).to(tl.int64)
    st_in = arena_ptr + off + flip * slot_stride
    st_out = arena_ptr + off + (1 - flip) * slot_stride
    offs = tl.arange(0, BLOCK_C)
    cm = offs < C
    x = tl.load(xa_ptr + t * C + offs, mask=cm, other=0.0).to(tl.float32)
    if HAS_B:
        phi = tl.load(phi_ptr + offs, mask=cm, other=0.0)
        x += phi * tl.load(xb_ptr + t * C + offs, mask=cm, other=0.0).to(tl.float32)
    if t == 0:
        prev = tl.load(st_in + offs, mask=cm, other=0.0)
    else:
        prev = tl.load(xa_ptr + (t - 1) * C + offs, mask=cm, other=0.0).to(tl.float32)
        if HAS_B:
            prev += phi * tl.load(xb_ptr + (t - 1) * C + offs, mask=cm, other=0.0).to(tl.float32)
    tl.store(u_ptr + t * 2 * C + offs, prev.to(u_ptr.dtype.element_ty), mask=cm)
    tl.store(u_ptr + t * 2 * C + C + offs, x.to(u_ptr.dtype.element_ty), mask=cm)
    if t == T - 1:
        tl.store(st_out + offs, x, mask=cm)


def vv_upcat(xa, xb, phi, arena, off, slot_stride, u, B=1, sel=None, astride=0):
    T, C = xa.shape[0] // B, xa.shape[1]
    _vv_upcat_kernel[(T, B)](xa, xb if xb is not None else xa, phi if phi is not None else arena, arena, off, slot_stride, u, T, C,
                             sel if sel is not None else arena, astride, HAS_B=xb is not None, BLOCK_C=triton.next_power_of_2(C),
                             HAS_SEL=sel is not None, num_warps=_vv_warps(C))


@triton.jit
def _vv_head_kernel(xa_ptr, xb_ptr, phi_ptr, w_ptr, b_ptr, arena_ptr, off, slot_stride, out_ptr, T, sel_ptr, astride,
                    C: tl.constexpr, BLOCK_T: tl.constexpr, HAS_SEL: tl.constexpr):
    """audio[t] = b + sum over the last 7 frames of x = xa + phi * xb (state: the 6 before this call) times W [7, C];
    the new state is the last 6 frames."""
    bb = tl.program_id(1)
    arena_ptr = _slot_base(arena_ptr, sel_ptr, bb, astride, HAS_SEL)
    xa_ptr += bb * T * C
    xb_ptr += bb * T * C
    out_ptr += bb * T
    flip = tl.load(arena_ptr).to(tl.int64)
    st_in = arena_ptr + off + flip * slot_stride
    st_out = arena_ptr + off + (1 - flip) * slot_stride
    t = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    offs = tl.arange(0, C)
    phi = tl.load(phi_ptr + offs)
    acc = tl.zeros((BLOCK_T,), dtype=tl.float32) + tl.load(b_ptr)
    for j in tl.static_range(7):
        src = t - 6 + j
        x = tl.load(xa_ptr + src[:, None] * C + offs[None, :], mask=(src >= 0)[:, None], other=0.0).to(tl.float32) \
            + phi[None, :] * tl.load(xb_ptr + src[:, None] * C + offs[None, :], mask=(src >= 0)[:, None], other=0.0).to(tl.float32)
        xs = tl.load(st_in + (src[:, None] + 6) * C + offs[None, :], mask=(src < 0)[:, None], other=0.0)
        x = tl.where((src < 0)[:, None], xs, x)
        acc += tl.sum(x * tl.load(w_ptr + j * C + offs)[None, :], axis=1)
        if j == 6:
            tl.store(st_out + (t[:, None] - (T - 6)) * C + offs[None, :], x, mask=(t >= T - 6)[:, None])
    tl.store(out_ptr + t, acc, mask=t < T)


def vv_head(xa, xb, phi, w, b, arena, off, slot_stride, out, B=1, sel=None, astride=0):
    T, C = xa.shape[0] // B, xa.shape[1]
    _vv_head_kernel[(triton.cdiv(T, 64), B)](xa, xb, phi, w, b, arena, off, slot_stride, out, T, sel if sel is not None else arena,
                                             astride, C=C, BLOCK_T=64, HAS_SEL=sel is not None, num_warps=2)


@triton.jit
def _flip_kernel(arena_ptr, sel_ptr, astride):
    """The VibeVoice state slot flag of arena row sel[b] flips (the other slot holds the new state)."""
    p = arena_ptr + tl.load(sel_ptr + tl.program_id(0)).to(tl.int64) * astride
    tl.store(p, 1.0 - tl.load(p))


def vv_flip(arena, sel, B):
    """arena [rows, A]: row sel[b] for b < B."""
    _flip_kernel[(B,)](arena, sel, arena.stride(0), num_warps=1)


# ----------------------------------------------------------------------------------------------------------------
# 48 kHz upsampler (bwe.py): its network's rows [2 streams, B pieces, n frames, C]
# ----------------------------------------------------------------------------------------------------------------

@triton.jit
def _conv_norm_kernel(x_ptr, w_ptr, wb_ptr, nw_ptr, nb_ptr, out_ptr, rows, n, eps, C: tl.constexpr, CONV: tl.constexpr):
    """One row (stream k, frame t of a piece): CONV: d = wb + sum_j x[t + j - 3] w[:, j] (a depthwise convolution of
    7 frames over time, zeros outside the piece), else d = x[t]; out = LayerNorm(d) * nw + nb (each stream its weights)."""
    r = tl.program_id(0)
    k = r // rows
    t = r % n
    offs = tl.arange(0, C)
    if CONV:
        acc = tl.load(wb_ptr + k * C + offs)
        for j in tl.static_range(7):
            tt = t + j - 3
            ok = (tt >= 0) & (tt < n)
            xr = tl.load(x_ptr + (r + j - 3).to(tl.int64) * C + offs, mask=(offs < C) & ok, other=0.0).to(tl.float32)
            acc += xr * tl.load(w_ptr + k * C * 7 + offs * 7 + j)
    else:
        acc = tl.load(x_ptr + r.to(tl.int64) * C + offs).to(tl.float32)
    mean = tl.sum(acc, axis=0) / C
    d = acc - mean
    rs = 1.0 / tl.sqrt(tl.sum(d * d, axis=0) / C + eps)
    y = d * rs * tl.load(nw_ptr + k * C + offs) + tl.load(nb_ptr + k * C + offs)
    tl.store(out_ptr + r.to(tl.int64) * C + offs, y.to(out_ptr.dtype.element_ty))


def conv_norm(x, w, wb, nw, nb, out, eps=1e-6):
    """x [2, B, n, C] (contiguous) -> out (same shape, its dtype): LayerNorm (and before it, with w [2, C, 7] and
    wb [2, C], the depthwise convolution over time) with the stream's weights nw, nb [2, C]."""
    S, B, n, C = x.shape
    _conv_norm_kernel[(S * B * n,)](x, w if w is not None else nw, wb if wb is not None else nw, nw, nb, out, B * n, n, eps, C=C,
                                    CONV=w is not None, num_warps=4)
    return out
