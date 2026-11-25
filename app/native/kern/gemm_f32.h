// Float GEMM on packed weight panels with fused epilogues (x86-64, AVX2 / AVX-512; aarch64, NEON). Torch-free.
//
// C[m][n] = bias[n] + sum_k A_m[k] W[k][n] for M input rows A_m: strided (A + m * lda; rows may overlap, e.g. a causal
// convolution's windows with lda < K) or given as row pointers (rows of several streams). W is packed once into panels
// of 16 output columns, [N / 16][K][16] (fp32, or int8 codes dequantized one panel at a time). Each output is one FMA
// chain over k (bias first): every output row is bit-identical whatever the ISA, M, the row's position, the tiling or
// the thread count. The epilogue runs on each output tile while it is in L1: GELU, Mish, C = R + C or C = R + g * C
// (R may be C itself).
#ifndef KERN_PASS
#ifndef TT_KERN_GEMM_F32_H
#define TT_KERN_GEMM_F32_H
#include <vector>

#include "act.h"

namespace kern {
enum { EPI_NONE = 0, EPI_GELU = 1, EPI_GELU_AS = 2, EPI_MISH = 3, EPI_ADD = 4, EPI_GADD = 5 };
struct Epi {
    int op = EPI_NONE;
    const float *r = nullptr;  // EPI_ADD / EPI_GADD: rows ldr apart
    int64_t ldr = 0;
    const float *g = nullptr;  // EPI_GADD: [N]
};

// Input rows: row m at p[m] (row pointers) or a + m * lda.
struct Rows {
    const float *a = nullptr;
    int64_t lda = 0;
    const float *const *p = nullptr;
    Rows(const float *a_, int64_t lda_) : a(a_), lda(lda_) {}
    explicit Rows(const float *const *p_) : p(p_) {}
    const float *row(int64_t m) const { return p ? p[m] : a + m * lda; }
    Rows from(int64_t m) const { Rows r = *this; if (p) r.p += m; else r.a += m * lda; return r; }
};

// Panels of a weight matrix: fp32 [N/16][K][16], or int8 codes [N/16][K][16] with fp32 scale / min per
// (panel, group of QG rows of K, column): w = code * scale + min.
constexpr int QG = 128;
struct Panels {
    const float *f32 = nullptr;
    const int8_t *q = nullptr;
    const float *sc = nullptr, *mn = nullptr;
    int64_t K = 0, N = 0;
};

// w [N][K] (kn false) or [K][N] (kn true) -> panels [N/16][K][16]; N a multiple of 16.
inline void pack_panels(const float *w, int64_t N, int64_t K, bool kn, float *dst) {
    for (int64_t p = 0; p < N / 16; p++)
        for (int64_t k = 0; k < K; k++)
            for (int j = 0; j < 16; j++) {
                const int64_t n = p * 16 + j;
                dst[(p * K + k) * 16 + j] = kn ? w[k * N + n] : w[n * K + k];
            }
}
}  // namespace kern

#ifdef KERN_X86
#define KERN_PASS
KERN_TARGET_BEGIN(KERN_T_AVX2)
namespace kern { namespace avx2 {
#include "gemm_f32.h"
}}
KERN_TARGET_END
#define KERN_AVX512_PASS
KERN_TARGET_BEGIN(KERN_T_AVX512F)
namespace kern { namespace avx512 {
#include "gemm_f32.h"
}}
KERN_TARGET_END
#undef KERN_AVX512_PASS
#undef KERN_PASS
#elif defined(KERN_ARM)
#define KERN_PASS
#define KERN_NEON_PASS
namespace kern { namespace neon {
#include "gemm_f32.h"
}}
#undef KERN_NEON_PASS
#undef KERN_PASS
#endif

namespace kern {
// Single thread: rows 0..M-1 and the np panels at Wp (fp32 panels, K rows each); bias, C, e.r and e.g start at the
// first panel's column.
inline void gemm_f32_tile(const Rows &A, int64_t M, const float *Wp, int64_t K, int64_t np, const float *bias, float *C,
                          int64_t ldc, const Epi &e) {
    KERN_CALL(gemm_tile, A, M, Wp, K, np, bias, C, ldc, e);
}
// Panels p .. p + n - 1 of W as fp32 (int8 dequantized into buf).
inline const float *panels_get(const Panels &W, int64_t p, int64_t n, float *buf) {
    if (W.f32) return W.f32 + p * W.K * 16;
    for (int64_t i = 0; i < n; i++) KERN_CALL(dequant_panel, W, p + i, buf + i * W.K * 16);
    return buf;
}
// The parallel GEMM: tasks of (pg panels, rm rows); 0 picks a default.
inline void gemm_f32(const Par &par, const Rows &A, int64_t M, const Panels &W, const float *bias, float *C, int64_t ldc,
                     const Epi &e = Epi(), int64_t rm = 0, int64_t pg = 0) {
    if (M <= 0) return;
    const int64_t P = W.N / 16;
    if (pg <= 0) pg = 2;
    if (rm <= 0) rm = 96;
    const int64_t groups = (P + pg - 1) / pg, chunks = (M + rm - 1) / rm;
    par_for(par, groups * chunks, 1, [&](int64_t t0, int64_t t1) {
        std::vector<float> buf(W.q ? pg * W.K * 16 : 0);
        int64_t loaded = -1;
        const float *wp = nullptr;
        for (int64_t t = t0; t < t1; t++) {
            const int64_t gi = t / chunks, m0 = (t % chunks) * rm, n = std::min(rm, M - m0);
            const int64_t p0 = gi * pg, np = std::min(pg, P - p0);
            if (gi != loaded) { wp = panels_get(W, p0, np, buf.data()); loaded = gi; }
            Epi em = e;
            if (em.r) em.r += m0 * em.ldr + p0 * 16;
            if (em.g) em.g += p0 * 16;
            KERN_CALL(gemm_tile, A.from(m0), n, wp, W.K, np, bias ? bias + p0 * 16 : nullptr, C + m0 * ldc + p0 * 16,
                      ldc, em);
        }
    });
}
}  // namespace kern
#endif  // TT_KERN_GEMM_F32_H

#else  // ------------------------------------------------------------ body, once per ISA

#ifndef KERN_NEON_PASS
static inline void dequant_panel(const Panels &W, int64_t p, float *buf) {
    const int64_t K = W.K;
    const int8_t *c = W.q + p * K * 16;
    const float *s = W.sc + p * (K / QG) * 16, *m = W.mn + p * (K / QG) * 16;
    for (int64_t g = 0; g < K / QG; g++) {
        const f8 s0 = ld8(s + g * 16), s1 = ld8(s + g * 16 + 8), m0 = ld8(m + g * 16), m1 = ld8(m + g * 16 + 8);
        for (int64_t k = g * QG; k < (g + 1) * QG; k++) {
            const __m128i v = _mm_loadu_si128((const __m128i *)(c + k * 16));
            st8(buf + k * 16, fma8(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v)), s0, m0));
            st8(buf + k * 16 + 8, fma8(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v, 8))), s1, m1));
        }
    }
}

#endif

// the post-store epilogues of a tile of n rows x 16 columns
static inline void epi_post(int op, float *C, int64_t ldc, int n) {
    if (op == EPI_GELU) for (int i = 0; i < n; i++) gelu(C + i * ldc, 16);
    else if (op == EPI_GELU_AS) for (int i = 0; i < n; i++) gelu_as(C + i * ldc, 16);
    else if (op == EPI_MISH) for (int i = 0; i < n; i++) mish(C + i * ldc, 16);
}

#if defined(KERN_NEON_PASS)
#include "gemm_f32_neon.inc"
#elif !defined(KERN_AVX512_PASS)
// AVX2: MR (<= 6) rows x one panel (2 vectors), named accumulators (an array ends up on the stack).
constexpr int MR_MAX = 6;
static inline f8 epi_reg(f8 c, int op, const float *r, const float *g) {
    if (op == EPI_ADD) return _mm256_add_ps(ld8(r), c);
    if (op == EPI_GADD) return fma8(ld8(g), c, ld8(r));
    return c;
}
template <int MR>
static inline void mk(const float *const *a, const float *P, int64_t K, const float *b16, float *C, int64_t ldc, int op,
                      const float *R, int64_t ldr, const float *g16) {
    const f8 i0 = b16 ? ld8(b16) : set8(0.f), i1 = b16 ? ld8(b16 + 8) : set8(0.f);
    f8 c00 = i0, c01 = i1, c10 = i0, c11 = i1, c20 = i0, c21 = i1, c30 = i0, c31 = i1, c40 = i0, c41 = i1, c50 = i0, c51 = i1;
    const float *a0 = a[0], *a1 = a[MR > 1 ? 1 : 0], *a2 = a[MR > 2 ? 2 : 0], *a3 = a[MR > 3 ? 3 : 0];
    const float *a4 = a[MR > 4 ? 4 : 0], *a5 = a[MR > 5 ? 5 : 0];
#define KERN_ROW(m)                                    \
    if (MR > m) {                                      \
        const f8 v = _mm256_broadcast_ss(a##m + k);    \
        c##m##0 = fma8(v, b0, c##m##0);                \
        c##m##1 = fma8(v, b1, c##m##1);                \
    }
    for (int64_t k = 0; k < K; k++) {
        const f8 b0 = ld8(P + k * 16), b1 = ld8(P + k * 16 + 8);
        KERN_ROW(0) KERN_ROW(1) KERN_ROW(2) KERN_ROW(3) KERN_ROW(4) KERN_ROW(5)
    }
#undef KERN_ROW
#define KERN_STORE(m)                                                                       \
    if (MR > m) {                                                                           \
        st8(C + m * ldc, epi_reg(c##m##0, op, R + m * ldr, g16));                           \
        st8(C + m * ldc + 8, epi_reg(c##m##1, op, R + m * ldr + 8, g16 ? g16 + 8 : g16));   \
    }
    KERN_STORE(0) KERN_STORE(1) KERN_STORE(2) KERN_STORE(3) KERN_STORE(4) KERN_STORE(5)
#undef KERN_STORE
    epi_post(op, C, ldc, MR);
}
// one micro-tile: rows a[0..mr-1], panel P (np2 ignored)
static inline void mk_any(int mr, int, const float *const *a, const float *P, int64_t K, const float *b, float *C,
                          int64_t ldc, int op, const float *R, int64_t ldr, const float *g) {
    switch (mr) {
        case 6: mk<6>(a, P, K, b, C, ldc, op, R, ldr, g); break;
        case 5: mk<5>(a, P, K, b, C, ldc, op, R, ldr, g); break;
        case 4: mk<4>(a, P, K, b, C, ldc, op, R, ldr, g); break;
        case 3: mk<3>(a, P, K, b, C, ldc, op, R, ldr, g); break;
        case 2: mk<2>(a, P, K, b, C, ldc, op, R, ldr, g); break;
        default: mk<1>(a, P, K, b, C, ldc, op, R, ldr, g); break;
    }
}
constexpr int NP_MAX = 1;  // panels per micro-tile
#else
// AVX-512: MR (<= 8) rows x NP (1 or 2) panels, one vector per panel row.
constexpr int MR_MAX = 8;
static inline vf epi_reg(vf c, int op, const float *r, const float *g) {
    if (op == EPI_ADD) return _mm512_add_ps(vld(r), c);
    if (op == EPI_GADD) return vfma(vld(g), c, vld(r));
    return c;
}
template <int MR, int NP>
static inline void mk(const float *const *a, const float *P, int64_t K, const float *b, float *C, int64_t ldc, int op,
                      const float *R, int64_t ldr, const float *g) {
    const float *P1 = P + (NP > 1 ? K * 16 : 0);
    const vf i0 = b ? vld(b) : vzero(), i1 = b && NP > 1 ? vld(b + 16) : vzero();
    vf c00 = i0, c01 = i1, c10 = i0, c11 = i1, c20 = i0, c21 = i1, c30 = i0, c31 = i1;
    vf c40 = i0, c41 = i1, c50 = i0, c51 = i1, c60 = i0, c61 = i1, c70 = i0, c71 = i1;
    const float *a0 = a[0], *a1 = a[MR > 1 ? 1 : 0], *a2 = a[MR > 2 ? 2 : 0], *a3 = a[MR > 3 ? 3 : 0];
    const float *a4 = a[MR > 4 ? 4 : 0], *a5 = a[MR > 5 ? 5 : 0], *a6 = a[MR > 6 ? 6 : 0], *a7 = a[MR > 7 ? 7 : 0];
#define KERN_ROW(m)                                    \
    if (MR > m) {                                      \
        const vf v = _mm512_set1_ps(a##m[k]);          \
        c##m##0 = vfma(v, b0, c##m##0);                \
        if (NP > 1) c##m##1 = vfma(v, b1, c##m##1);    \
    }
    for (int64_t k = 0; k < K; k++) {
        const vf b0 = vld(P + k * 16), b1 = NP > 1 ? vld(P1 + k * 16) : b0;
        KERN_ROW(0) KERN_ROW(1) KERN_ROW(2) KERN_ROW(3) KERN_ROW(4) KERN_ROW(5) KERN_ROW(6) KERN_ROW(7)
    }
#undef KERN_ROW
#define KERN_STORE(m)                                                                                   \
    if (MR > m) {                                                                                       \
        vst(C + m * ldc, epi_reg(c##m##0, op, R + m * ldr, g));                                         \
        if (NP > 1) vst(C + m * ldc + 16, epi_reg(c##m##1, op, R + m * ldr + 16, g ? g + 16 : g));      \
    }
    KERN_STORE(0) KERN_STORE(1) KERN_STORE(2) KERN_STORE(3) KERN_STORE(4) KERN_STORE(5) KERN_STORE(6) KERN_STORE(7)
#undef KERN_STORE
    epi_post(op, C, ldc, MR);
    if (NP > 1) epi_post(op, C + 16, ldc, MR);
}
template <int NP>
static inline void mk_np(int mr, const float *const *a, const float *P, int64_t K, const float *b, float *C, int64_t ldc,
                         int op, const float *R, int64_t ldr, const float *g) {
    switch (mr) {
        case 8: mk<8, NP>(a, P, K, b, C, ldc, op, R, ldr, g); break;
        case 7: mk<7, NP>(a, P, K, b, C, ldc, op, R, ldr, g); break;
        case 6: mk<6, NP>(a, P, K, b, C, ldc, op, R, ldr, g); break;
        case 5: mk<5, NP>(a, P, K, b, C, ldc, op, R, ldr, g); break;
        case 4: mk<4, NP>(a, P, K, b, C, ldc, op, R, ldr, g); break;
        case 3: mk<3, NP>(a, P, K, b, C, ldc, op, R, ldr, g); break;
        case 2: mk<2, NP>(a, P, K, b, C, ldc, op, R, ldr, g); break;
        default: mk<1, NP>(a, P, K, b, C, ldc, op, R, ldr, g); break;
    }
}
static inline void mk_any(int mr, int np, const float *const *a, const float *P, int64_t K, const float *b, float *C,
                          int64_t ldc, int op, const float *R, int64_t ldr, const float *g) {
    if (np == 2) mk_np<2>(mr, a, P, K, b, C, ldc, op, R, ldr, g);
    else mk_np<1>(mr, a, P, K, b, C, ldc, op, R, ldr, g);
}
constexpr int NP_MAX = 2;
#endif

// Rows 0..M-1 x the np panels at Wp: for each group of panels (one or two), all row groups, so the panels stay in cache.
static inline void gemm_tile(const Rows &A, int64_t M, const float *Wp, int64_t K, int64_t np, const float *bias,
                             float *C, int64_t ldc, const Epi &e) {
    for (int64_t p = 0; p < np;) {
        const int n2 = p + NP_MAX <= np ? NP_MAX : 1;
        const float *P = Wp + p * K * 16, *b = bias ? bias + p * 16 : nullptr, *g = e.g ? e.g + p * 16 : nullptr;
        float *Cp = C + p * 16;
        const float *R = e.r ? e.r + p * 16 : Cp;  // (read only by EPI_ADD / EPI_GADD)
        const int64_t lr = e.r ? e.ldr : ldc;
        for (int64_t m = 0; m < M; m += MR_MAX) {
            const int mr = (int)std::min<int64_t>(MR_MAX, M - m);
            const float *a[MR_MAX];
            for (int i = 0; i < mr; i++) a[i] = A.row(m + i);
            mk_any(mr, n2, a, P, K, b, Cp + m * ldc, ldc, e.op, R + m * lr, lr, g);
        }
        p += n2;
    }
}

#endif  // KERN_PASS
