// VibeVoice decoder blocks on x86-64 (AVX2 / AVX-512): RMS norm, causal depthwise convolution (7 taps), FFN with GELU,
// residuals; the small causal convolutions. Torch-free; used by V1.1's tokenizer (app/native) and the Flash engine.
//
// A block on rows x [T][C] (channels last) with the state = the last 6 normed rows before x:
//   h = rms(x) * norm_w;  x1 = x + gamma * (dwb + sum_j dw[j] * h[t - 6 + j]);  h2 = rms(x1) * ffn_norm_w
//   out = x1 + ffn_gamma * (gelu(h2 W1 + b1) W2 + b2)
// norm_w, gamma, ffn_norm_w, ffn_gamma may be null (folded into the other weights). Fused blocks run per tile of rows
// (each tile recomputes the 6 normed rows before it), so x is only read and out != x.
#ifndef KERN_PASS
#ifndef TT_KERN_VV_H
#define TT_KERN_VV_H
#include "gemm_f32.h"

namespace kern {
constexpr int DWK = 7;  // depthwise kernel

struct VVBlock {
    int64_t C = 0, H = 0;  // channels, FFN width
    const float *norm_w = nullptr, *dw = nullptr, *dwb = nullptr, *gamma = nullptr, *ffn_norm_w = nullptr;
    Panels w1, w2;
    const float *b1 = nullptr, *b2 = nullptr, *ffn_gamma = nullptr;
    float eps = 1e-5f;
    int gelu = EPI_GELU;  // EPI_GELU (the engine's) or EPI_GELU_AS (V1.1)
};
}  // namespace kern

#ifdef KERN_X86
#define KERN_PASS
KERN_TARGET_BEGIN(KERN_T_AVX2)
namespace kern { namespace avx2 {
#include "vv.h"
}}
KERN_TARGET_END
#define KERN_AVX512_PASS
KERN_TARGET_BEGIN(KERN_T_AVX512F)
namespace kern { namespace avx512 {
#include "vv.h"
}}
KERN_TARGET_END
#undef KERN_AVX512_PASS
#undef KERN_PASS
#elif defined(KERN_ARM)
#define KERN_PASS
#define KERN_NEON_PASS
namespace kern { namespace neon {
#include "vv.h"
}}
#undef KERN_NEON_PASS
#undef KERN_PASS
#endif

namespace kern {
// Default rows per tile of a fused block: the tile's FFN rows stay in L2.
inline int64_t vv_tile_rows(int64_t C) { return std::max<int64_t>(6, std::min<int64_t>(192, (65536 / (8 * C)) / 6 * 6)); }
// Floats of work memory per tile of tt rows (vv_block).
inline int64_t vv_tile_floats(const VVBlock &b, int64_t tt) { return (tt + DWK - 1) * b.C + 2 * tt * b.C + tt * b.H; }

// Whole blocks of B sequences x [B][T][C] (sequence q's state [6][C] at st[q]; st or st[q] null: zeros) -> out
// [B][T][C], parallel over tiles (fp32 FFN panels).
inline void vv_block(const Par &par, const VVBlock &b, const float *x, int64_t B, int64_t T, const float *const *st,
                     float *out, int64_t tt = 0) {
    if (tt <= 0) tt = vv_tile_rows(b.C);
    const int64_t tiles = (T + tt - 1) / tt;
    par_for(par, B * tiles, 1, [&](int64_t s0, int64_t s1) {
        std::vector<float> buf(vv_tile_floats(b, tt));
        for (int64_t s = s0; s < s1; s++) {
            const int64_t q = s / tiles, t0 = (s % tiles) * tt, t1 = std::min(T, t0 + tt);
            KERN_CALL(block_tile, b, x + q * T * b.C, st ? st[q] : nullptr, t0, t1, buf.data(), out + q * T * b.C);
        }
    });
}

// The mixer half of blocks: x1 and h2 [B][T][C] (FFN outside, e.g. quantized), parallel over tiles of tt rows.
inline void vv_mixer(const Par &par, const VVBlock &b, const float *x, int64_t B, int64_t T, const float *const *st,
                     float *x1, float *h2, int64_t tt = 16) {
    const int64_t tiles = (T + tt - 1) / tt;
    par_for(par, B * tiles, 1, [&](int64_t s0, int64_t s1) {
        std::vector<float> h((tt + DWK - 1) * b.C);
        for (int64_t s = s0; s < s1; s++) {
            const int64_t q = s / tiles, t0 = (s % tiles) * tt, t1 = std::min(T, t0 + tt), o = (q * T + t0) * b.C;
            KERN_CALL(mixer_tile, b, x + q * T * b.C, st ? st[q] : nullptr, t0, t1, h.data(), x1 + o, h2 + o);
        }
    });
}

// The state after x: its last 6 normed rows (from the old state st[q] when T < 6) into ns[q] (not st[q]).
inline void vv_next_state(const VVBlock &b, const float *x, int64_t B, int64_t T, const float *const *st,
                          float *const *ns) {
    for (int64_t q = 0; q < B; q++)
        KERN_CALL(normed_rows, x + q * T * b.C, st ? st[q] : nullptr, b.norm_w, b.C, b.eps, T, T, ns[q]);
}

// Causal stride-1 convolution with few channels, one sequence: y[t] = bias + sum_j x[t - k + 1 + j] @ w[j] for
// t0 <= t < t1; x [T][cin], state [k - 1][cin] rows before x (null: zeros), w [k][cin][cout]. cin % 8 == 0 if cout == 1,
// else cout % 8 == 0.
inline void conv_small(const float *x, const float *state, int64_t k, int64_t cin, int64_t cout, const float *w,
                       const float *bias, int64_t t0, int64_t t1, float *y) {
    KERN_CALL(conv_small_rows, x, state, k, cin, cout, w, bias, t0, t1, y);
}
}  // namespace kern
#endif  // TT_KERN_VV_H

#else  // ------------------------------------------------------------ body, once per ISA

// n rows of x1 = x + gamma * (dwb + sum_j dw[j] * h[i + j]) (gamma null: x + ...); h starts 6 rows before x's first.
// x1 may be x.
static inline void dw_residual(const float *x, const float *h, int64_t n, int64_t C, const float *dw, const float *dwb,
                               const float *gamma, float *x1) {
    for (int64_t i = 0; i < n; i++)
        for (int64_t c = 0; c < C; c += VW) {
            const int r = (int)std::min<int64_t>(VW, C - c);
            vf acc = vldn(dwb + c, r);
            for (int j = 0; j < DWK; j++) acc = vfma(vldn(dw + j * C + c, r), vldn(h + (i + j) * C + c, r), acc);
            const vf xv = vldn(x + i * C + c, r);
            const vf y = gamma ? vfma(vldn(gamma + c, r), acc, xv) : vadd(xv, acc);
            if (r == VW) vst(x1 + i * C + c, y); else vstn(x1 + i * C + c, y, r);
        }
}

// The normed rows t0 - 6 .. t1 - 1 of one sequence into h (rows before 0 from the state, or zeros).
static inline void normed_rows(const float *x, const float *state, const float *norm_w, int64_t C, float eps,
                               int64_t t0, int64_t t1, float *h) {
    for (int64_t t = t0 - (DWK - 1); t < t1; t++) {
        float *dst = h + (t - t0 + DWK - 1) * C;
        if (t >= 0) rms(x + t * C, norm_w, C, eps, dst);
        else if (state) std::memcpy(dst, state + (t + DWK - 1) * C, C * sizeof(float));
        else std::memset(dst, 0, C * sizeof(float));
    }
}

// Rows t0..t1-1: x1 and h2 = rms(x1) * ffn_norm_w (h: work, (t1 - t0 + 6) rows).
static inline void mixer_tile(const VVBlock &b, const float *x, const float *state, int64_t t0, int64_t t1, float *h,
                              float *x1, float *h2) {
    const int64_t C = b.C, n = t1 - t0;
    normed_rows(x, state, b.norm_w, C, b.eps, t0, t1, h);
    dw_residual(x + t0 * C, h, n, C, b.dw, b.dwb, b.gamma, x1);
    for (int64_t i = 0; i < n; i++) rms(x1 + i * C, b.ffn_norm_w, C, b.eps, h2 + i * C);
}

// A whole block on rows t0..t1-1 into out (buf: vv_tile_floats).
static inline void block_tile(const VVBlock &b, const float *x, const float *state, int64_t t0, int64_t t1, float *buf,
                              float *out) {
    const int64_t C = b.C, H = b.H, n = t1 - t0;
    float *h = buf, *x1 = h + (n + DWK - 1) * C, *h2 = x1 + n * C, *u = h2 + n * C;
    mixer_tile(b, x, state, t0, t1, h, x1, h2);
    gemm_tile(Rows(h2, C), n, b.w1.f32, C, H / 16, b.b1, u, H, Epi());
    if (b.gelu == EPI_GELU) gelu(u, n * H); else gelu_as(u, n * H);
    Epi e2;
    e2.op = b.ffn_gamma ? EPI_GADD : EPI_ADD;
    e2.r = x1;
    e2.ldr = C;
    e2.g = b.ffn_gamma;
    gemm_tile(Rows(u, H), n, b.w2.f32, H, C / 16, b.b2, out + t0 * C, C, e2);
}

static inline const float *conv_row(const float *x, const float *state, int64_t k, int64_t cin, int64_t t) {
    if (t >= 0) return x + t * cin;  // input row t (may be < 0: the state, or null for zeros)
    return state ? state + (t + k - 1) * cin : nullptr;
}
static inline void conv_small_rows(const float *x, const float *state, int64_t k, int64_t cin, int64_t cout,
                                   const float *w, const float *bias, int64_t t0, int64_t t1, float *y) {
    for (int64_t t = t0; t < t1; t++) {
        float *out = y + t * cout;
        if (cout == 1) {
            f8 acc = set8(0.f);
            for (int64_t j = 0; j < k; j++) {
                const float *r = conv_row(x, state, k, cin, t - k + 1 + j);
                if (!r) continue;
                for (int64_t c = 0; c < cin; c += 8) acc = fma8(ld8(r + c), ld8(w + j * cin + c), acc);
            }
            out[0] = hsum8(acc) + bias[0];
            continue;
        }
        for (int64_t o = 0; o < cout; o += 8) {
            f8 acc = ld8(bias + o);
            for (int64_t j = 0; j < k; j++) {
                const float *r = conv_row(x, state, k, cin, t - k + 1 + j);
                if (!r) continue;
                for (int64_t c = 0; c < cin; c++) acc = fma8(set8(r[c]), ld8(w + (j * cin + c) * cout + o), acc);
            }
            st8(out + o, acc);
        }
    }
}

#endif  // KERN_PASS
