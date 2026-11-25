// Quantized linear layers on x86 and aarch64 (lin_q_neon.h) (torch-free; V1.1 app/native and the Flash engine).
//
// Weight formats (the GPU bundle / web .bin layout):
//   q4 (kind 0): codes uint8 [N][K/64][32]; byte i of a group holds input i (low nibble) and input i + 32 (high
//       nibble). fp16 scales / mins [N][K/64].
//   q8 (kind 1): codes int8 [N][K/128][128] = code - 128; fp16 scales [N][K/128] and mins [N][K/128] shifted by
//       128 * scale. In both, w = code * scale + min.
// Activations are int16 per 64 inputs (quant16): q, scale s and xs = s * sum(q), which carries the min term:
//   sum_k w_k x_k = sum_g ws_g * s_g * dot(code, q)_g + wm_g * xs_g
// The dot products are exact in int32 (madd / vpdpwssd); every variant (AVX2, AVX-VNNI, AVX-512) does the same
// float work per output, so all give bit-identical results, for any M, thread count or split.
#pragma once
#include "common.h"

#include <atomic>
#include <chrono>
#include <mutex>

namespace kern {

struct QW {  // quantized weights
    int kind = 0;  // 0 q4, 1 q8
    int64_t N = 0, K = 0;
    const uint8_t *codes = nullptr;
    const uint16_t *scales = nullptr, *mins = nullptr;
    const float *bias = nullptr;
    const uint8_t *packed = nullptr;  // aarch64, optional: the rows in quads (lin_q_neon.h), read for single rows
};

struct QX {  // M rows of int16 activations: row m at q + m * K, s / xs + m * (K / 64)
    const int16_t *q = nullptr;
    const float *s = nullptr, *xs = nullptr;
    int64_t M = 0, K = 0;
    const void *aux = nullptr;  // the activations in a variant's own layout (LinQImpl::prep), set by lin_q
};

struct LinQImpl {
    const char *name;
    // rows m0..m1-1 of x (rows ldx apart, K inputs) -> q [m][K], s / xs [m][K / 64]
    void (*quant)(const float *x, int64_t ldx, int64_t m0, int64_t m1, int64_t K, int16_t *q, float *s, float *xs);
    // outputs n0..n1-1 of activation rows m0..m0+mr-1 (mr <= 8) into yr[0..mr-1]; xsT: their xs transposed
    // [K / 64][8]
    void (*tile)(const QW &W, const QX &X, int64_t m0, int mr, const float *xsT, int64_t n0, int64_t n1, float *const *yr);
    // optional (aarch64): bytes of X.aux for M rows, and the rows of pairs p0..p1-1 written into it
    size_t (*aux_bytes)(int64_t M, int64_t K);
    void (*prep)(const QW &W, const QX &X, int64_t p0, int64_t p1, void *aux);
};

// ---------------------------------------------------------------- scalar reference
namespace ref {

static void quant(const float *x, int64_t ldx, int64_t m0, int64_t m1, int64_t K, int16_t *q, float *s, float *xs) {
    const int64_t G = K / 64;
    for (int64_t m = m0; m < m1; m++)
        for (int64_t g = 0; g < G; g++) {
            const float *v = x + m * ldx + g * 64;
            float amax = 0.f;
            for (int k = 0; k < 64; k++) amax = std::max(amax, std::fabs(v[k]));
            const float sc = amax / 32767.f, inv = amax > 0.f ? 32767.f / amax : 0.f;
            int64_t sum = 0;
            for (int k = 0; k < 64; k++) {
                int c = (int)std::nearbyint(v[k] * inv);
                c = std::max(-32767, std::min(32767, c));
                q[m * K + g * 64 + k] = (int16_t)c;
                sum += c;
            }
            s[m * G + g] = sc;
            xs[m * G + g] = sc * (float)sum;
        }
}

static void tile(const QW &W, const QX &X, int64_t m0, int mr, const float *, int64_t n0, int64_t n1, float *const *yr) {
    const int64_t K = W.K, GA = K / 64;
    for (int64_t m = m0; m < m0 + mr; m++) {
        const int16_t *q = X.q + m * K;
        const float *as = X.s + m * GA, *axs = X.xs + m * GA;
        for (int64_t n = n0; n < n1; n++) {
            float acc = 0.f, mt = 0.f;
            if (W.kind == 0) {
                const uint8_t *row = W.codes + n * GA * 32;
                for (int64_t g = 0; g < GA; g++) {
                    const uint8_t *c = row + g * 32;
                    const int16_t *xq = q + g * 64;
                    int32_t d = 0;
                    for (int k = 0; k < 32; k++) d += (c[k] & 15) * xq[k] + (c[k] >> 4) * xq[k + 32];
                    acc += (float)d * (h2f_sw(W.scales[n * GA + g]) * as[g]);
                    mt += h2f_sw(W.mins[n * GA + g]) * axs[g];
                }
            } else {
                const int64_t G = K / 128;
                const int8_t *row = (const int8_t *)W.codes + n * G * 128;
                for (int64_t g = 0; g < G; g++) {
                    const float ws = h2f_sw(W.scales[n * G + g]), wm = h2f_sw(W.mins[n * G + g]);
                    for (int h = 0; h < 2; h++) {
                        const int8_t *c = row + g * 128 + h * 64;
                        const int16_t *xq = q + (g * 2 + h) * 64;
                        int32_t d = 0;
                        for (int k = 0; k < 64; k++) d += c[k] * xq[k];
                        acc += (float)d * (ws * as[g * 2 + h]);
                        mt += wm * axs[g * 2 + h];
                    }
                }
            }
            yr[m - m0][n] = acc + mt + (W.bias ? W.bias[n] : 0.f);
        }
    }
}

}  // namespace ref

inline const LinQImpl &impl_scalar() {
    static const LinQImpl r{"scalar", ref::quant, ref::tile};
    return r;
}

#ifdef KERN_X86
// ---------------------------------------------------------------- AVX2 (+ AVX-VNNI)
KERN_TARGET_BEGIN(KERN_T_AVX2)
namespace avx2 {

// Same rounding as ref::quant (cvtps_epi32 rounds to nearest even, as nearbyint).
static void quant(const float *x, int64_t ldx, int64_t m0, int64_t m1, int64_t K, int16_t *q, float *s, float *xs) {
    const int64_t G = K / 64;
    const __m256 absm = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256i lim = _mm256_set1_epi32(32767), nlim = _mm256_set1_epi32(-32767);
    for (int64_t m = m0; m < m1; m++)
        for (int64_t g = 0; g < G; g++) {
            const float *v = x + m * ldx + g * 64;
            __m256 mx = _mm256_and_ps(_mm256_loadu_ps(v), absm);
            for (int k = 8; k < 64; k += 8) mx = _mm256_max_ps(mx, _mm256_and_ps(_mm256_loadu_ps(v + k), absm));
            __m128 h = _mm_max_ps(_mm256_castps256_ps128(mx), _mm256_extractf128_ps(mx, 1));
            h = _mm_max_ps(h, _mm_movehl_ps(h, h));
            h = _mm_max_ss(h, _mm_shuffle_ps(h, h, 1));
            const float amax = _mm_cvtss_f32(h);
            const float sc = amax / 32767.f, inv = amax > 0.f ? 32767.f / amax : 0.f;
            const __m256 iv = _mm256_set1_ps(inv);
            __m256i sum = _mm256_setzero_si256();
            int16_t *qd = q + m * K + g * 64;
            for (int k = 0; k < 64; k += 16) {
                __m256i c0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(v + k), iv));
                __m256i c1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(v + k + 8), iv));
                c0 = _mm256_max_epi32(nlim, _mm256_min_epi32(lim, c0));
                c1 = _mm256_max_epi32(nlim, _mm256_min_epi32(lim, c1));
                sum = _mm256_add_epi32(sum, _mm256_add_epi32(c0, c1));
                _mm256_storeu_si256((__m256i *)(qd + k), _mm256_permute4x64_epi64(_mm256_packs_epi32(c0, c1), 0xD8));
            }
            __m128i t = _mm_add_epi32(_mm256_castsi256_si128(sum), _mm256_extracti128_si256(sum, 1));
            t = _mm_add_epi32(t, _mm_shuffle_epi32(t, 0x4E));
            t = _mm_add_epi32(t, _mm_shuffle_epi32(t, 0xB1));
            s[m * G + g] = sc;
            xs[m * G + g] = sc * (float)_mm_cvtsi128_si32(t);
        }
}

#define KERN_DOT(d, w, x) _mm256_add_epi32(d, _mm256_madd_epi16(w, x))
#include "lin_q_ymm.inc"
#undef KERN_DOT

}  // namespace avx2
KERN_TARGET_END

KERN_TARGET_BEGIN(KERN_T_AVXVNNI)
namespace avxvnni {
#define KERN_DOT(d, w, x) _mm256_dpwssd_avx_epi32(d, w, x)
#include "lin_q_ymm.inc"
#undef KERN_DOT
}  // namespace avxvnni
KERN_TARGET_END

// The same 256-bit tiles with the EVEX vpdpwssd (AVX512-VL + VNNI): VNNI without 512-bit vectors (Zen 4, where those
// run at half rate).
KERN_TARGET_BEGIN(KERN_T_VNNI256)
namespace vnni256 {
#define KERN_DOT(d, w, x) _mm256_dpwssd_epi32(d, w, x)
#include "lin_q_ymm.inc"
#undef KERN_DOT
}  // namespace vnni256
KERN_TARGET_END

// ---------------------------------------------------------------- AVX-512 (BW + VNNI)
// Two weight rows per pass: each 64-input dot is two vpdpwssd on 32 lanes, folded to the 8 lanes of the AVX2 order
// (lanes j and j + 8 hold inputs 2j, 2j + 1 and 16 + 2j, 17 + 2j of each 32), and the pair's float work shares one
// zmm: [row n | row n + 1].
KERN_TARGET_BEGIN(KERN_T_AVX512)
namespace avx512 {

static inline __m512i fold2(__m512i a, __m512i b) {  // [a.lo + a.hi | b.lo + b.hi] (lanes of 8)
    return _mm512_add_epi32(_mm512_shuffle_i64x2(a, b, 0x44), _mm512_shuffle_i64x2(a, b, 0xEE));
}
static inline __m512 pair_ps(float a, float b) {  // [a x 8 | b x 8]
    return _mm512_mask_broadcastss_ps(_mm512_set1_ps(a), 0xFF00, _mm_set_ss(b));
}
static inline __m512 dup8(const float *p) {  // [p[0..8] | p[0..8]]
    return _mm512_castpd_ps(_mm512_broadcast_f64x4(_mm256_castps_pd(_mm256_loadu_ps(p))));
}
static inline void pair_out(const __m512 *acc, int mr, __m512 mt, const QW &W, int64_t n, int64_t nb, float *const *yr) {
    alignas(64) float mts[16];
    _mm512_store_ps(mts, mt);
    const float ba = W.bias ? W.bias[n] : 0.f, bb = W.bias ? W.bias[nb] : 0.f;
    for (int m = 0; m < mr; m++) {
        const __m256 lo = _mm512_castps512_ps256(acc[m]);
        const __m256 hi = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(acc[m]), 1));
        yr[m][n] = hsum(lo) + mts[m] + ba;
        if (nb != n) yr[m][nb] = hsum(hi) + mts[8 + m] + bb;
    }
}

template <int MR>
static void q4_tile(const QW &W, const QX &X, int64_t m0, const float *xsT, int64_t n0, int64_t n1, float *const *yr) {
    const int64_t K = W.K, G = K / 64;
    const __m256i m4 = _mm256_set1_epi8(0x0F);
    const int16_t *xq = X.q + m0 * K;
    const float *xsc = X.s + m0 * G;
    for (int64_t n = n0; n < n1; n += 2) {
        const int64_t nb = n + 1 < n1 ? n + 1 : n;  // odd tail: the row twice, written once
        __m512 acc[MR], mt = _mm512_setzero_ps();
        for (int m = 0; m < MR; m++) acc[m] = _mm512_setzero_ps();
        const uint8_t *ra = W.codes + n * G * 32, *rb = W.codes + nb * G * 32;
        const uint16_t *sa = W.scales + n * G, *sb = W.scales + nb * G, *ma = W.mins + n * G, *mb = W.mins + nb * G;
        for (int64_t g = 0; g < G; g++) {
            const __m256i ca = _mm256_loadu_si256((const __m256i *)(ra + g * 32));
            const __m256i cb = _mm256_loadu_si256((const __m256i *)(rb + g * 32));
            const __m512i a0 = _mm512_cvtepu8_epi16(_mm256_and_si256(ca, m4));
            const __m512i a1 = _mm512_cvtepu8_epi16(_mm256_and_si256(_mm256_srli_epi16(ca, 4), m4));
            const __m512i b0 = _mm512_cvtepu8_epi16(_mm256_and_si256(cb, m4));
            const __m512i b1 = _mm512_cvtepu8_epi16(_mm256_and_si256(_mm256_srli_epi16(cb, 4), m4));
            const __m512 ws = pair_ps(h2f(sa[g]), h2f(sb[g]));
            mt = _mm512_fmadd_ps(pair_ps(h2f(ma[g]), h2f(mb[g])), dup8(xsT + g * 8), mt);
            for (int m = 0; m < MR; m++) {
                const int16_t *x = xq + m * K + g * 64;
                const __m512i x0 = _mm512_loadu_si512(x), x1 = _mm512_loadu_si512(x + 32);
                const __m512i da = _mm512_dpwssd_epi32(_mm512_madd_epi16(a0, x0), a1, x1);
                const __m512i db = _mm512_dpwssd_epi32(_mm512_madd_epi16(b0, x0), b1, x1);
                acc[m] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(fold2(da, db)), _mm512_mul_ps(ws, _mm512_set1_ps(xsc[m * G + g])), acc[m]);
            }
        }
        pair_out(acc, MR, mt, W, n, nb, yr);
    }
}

template <int MR>
static void q8_tile(const QW &W, const QX &X, int64_t m0, const float *xsT, int64_t n0, int64_t n1, float *const *yr) {
    const int64_t K = W.K, G = K / 128, GA = K / 64;
    const int16_t *xq = X.q + m0 * K;
    const float *xsc = X.s + m0 * GA;
    for (int64_t n = n0; n < n1; n += 2) {
        const int64_t nb = n + 1 < n1 ? n + 1 : n;
        __m512 acc[MR], mt = _mm512_setzero_ps();
        for (int m = 0; m < MR; m++) acc[m] = _mm512_setzero_ps();
        const int8_t *ra = (const int8_t *)W.codes + n * G * 128, *rb = (const int8_t *)W.codes + nb * G * 128;
        const uint16_t *sa = W.scales + n * G, *sb = W.scales + nb * G, *ma = W.mins + n * G, *mb = W.mins + nb * G;
        for (int64_t g = 0; g < G; g++) {
            const __m512 ws = pair_ps(h2f(sa[g]), h2f(sb[g])), wm = pair_ps(h2f(ma[g]), h2f(mb[g]));
            for (int h = 0; h < 2; h++) {
                const int8_t *pa = ra + g * 128 + h * 64, *pb = rb + g * 128 + h * 64;
                const __m512i a0 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)pa));
                const __m512i a1 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)(pa + 32)));
                const __m512i b0 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)pb));
                const __m512i b1 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)(pb + 32)));
                const int64_t ga = g * 2 + h;
                mt = _mm512_fmadd_ps(wm, dup8(xsT + ga * 8), mt);
                for (int m = 0; m < MR; m++) {
                    const int16_t *x = xq + m * K + ga * 64;
                    const __m512i x0 = _mm512_loadu_si512(x), x1 = _mm512_loadu_si512(x + 32);
                    const __m512i da = _mm512_dpwssd_epi32(_mm512_madd_epi16(a0, x0), a1, x1);
                    const __m512i db = _mm512_dpwssd_epi32(_mm512_madd_epi16(b0, x0), b1, x1);
                    acc[m] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(fold2(da, db)), _mm512_mul_ps(ws, _mm512_set1_ps(xsc[m * GA + ga])), acc[m]);
                }
            }
        }
        pair_out(acc, MR, mt, W, n, nb, yr);
    }
}

static void tile(const QW &W, const QX &X, int64_t m0, int mr, const float *xsT, int64_t n0, int64_t n1, float *const *yr) {
    typedef void (*T)(const QW &, const QX &, int64_t, const float *, int64_t, int64_t, float *const *);
    static const T q4[8] = {q4_tile<1>, q4_tile<2>, q4_tile<3>, q4_tile<4>, q4_tile<5>, q4_tile<6>, q4_tile<7>, q4_tile<8>};
    static const T q8[8] = {q8_tile<1>, q8_tile<2>, q8_tile<3>, q8_tile<4>, q8_tile<5>, q8_tile<6>, q8_tile<7>, q8_tile<8>};
    (W.kind == 0 ? q4 : q8)[mr - 1](W, X, m0, xsT, n0, n1, yr);
}

}  // namespace avx512
KERN_TARGET_END

inline const LinQImpl &impl_avx2() { static const LinQImpl r{"avx2", avx2::quant, avx2::tile}; return r; }
inline const LinQImpl &impl_avxvnni() { static const LinQImpl r{"avxvnni", avx2::quant, avxvnni::tile}; return r; }
inline const LinQImpl &impl_vnni256() { static const LinQImpl r{"vnni256", avx2::quant, vnni256::tile}; return r; }
inline const LinQImpl &impl_avx512() { static const LinQImpl r{"avx512", avx2::quant, avx512::tile}; return r; }
#endif

}  // namespace kern
#ifdef KERN_ARM
#include "lin_q_neon.h"
#endif
namespace kern {

// The variants this CPU runs (within isa_limit), widest first.
inline std::vector<const LinQImpl *> lin_q_impls() {
    std::vector<const LinQImpl *> v;
#ifdef KERN_X86
    if (has_avx512()) v.push_back(&impl_avx512());
    if (has_avxvnni()) v.push_back(&impl_avxvnni());
    else if (has_vnni256()) v.push_back(&impl_vnni256());
    if (has_avx2()) v.push_back(&impl_avx2());
#elif defined(KERN_ARM)
    if (has_i8mm()) v.push_back(&impl_neon_mm());
    if (has_dotprod()) v.push_back(&impl_neon_dot());
#endif
    if (v.empty()) v.push_back(&impl_scalar());
    return v;
}

// The widest variant for this CPU (within isa_limit); lin_q without a variant measures which one is fastest.
inline const LinQImpl &lin_q_impl() {
#ifdef KERN_X86
    return has_avx512() ? impl_avx512() : has_avxvnni() ? impl_avxvnni() : has_avx2() ? impl_avx2() : impl_scalar();
#elif defined(KERN_ARM)
    return has_i8mm() ? impl_neon_mm() : has_dotprod() ? impl_neon_dot() : impl_scalar();
#else
    return impl_scalar();
#endif
}

// ---------------------------------------------------------------- drivers
// What a task does with its finished outputs while they are in cache (e.g. GELU, a residual add): rows m0..m1-1 (the
// call's row indices), outputs n0..n1-1; once per output, after it is final.
struct LinPost {
    void (*fn)(void *ctx, int64_t m0, int64_t m1, int64_t n0, int64_t n1) = nullptr;
    void *ctx = nullptr;
};
template <class F> inline LinPost lin_post(const F &f) {  // f(m0, m1, n0, n1); f must outlive the call
    LinPost p;
    p.fn = [](void *c, int64_t m0, int64_t m1, int64_t n0, int64_t n1) { (*(const F *)c)(m0, m1, n0, n1); };
    p.ctx = (void *)&f;
    return p;
}

namespace detail {
struct QuantCtx { const LinQImpl *I; const float *const *xr; int64_t K, G; int16_t *q; float *s, *xs; };
inline void quant_task(void *c, int64_t b, int64_t e) {  // items: (row, group); whole rows per call where possible
    const QuantCtx &t = *(const QuantCtx *)c;
    for (int64_t i = b; i < e;) {
        const int64_t m = i / t.G, g0 = i % t.G, g1 = std::min(t.G, g0 + (e - i));
        // a partial row: through a sub-row view (groups g0..g1-1 of row m)
        t.I->quant(t.xr[m] + g0 * 64, 0, 0, 1, (g1 - g0) * 64, t.q + m * t.K + g0 * 64, t.s + m * t.G + g0, t.xs + m * t.G + g0);
        i += g1 - g0;
    }
}

struct LinCtx { const LinQImpl *I; const QW *W; const QX *X; const float *xsT; float *const *yr; int64_t nb, nmb; const LinPost *post; };
inline void lin_task(void *c, int64_t b, int64_t e) {
    const LinCtx &t = *(const LinCtx *)c;
    const int64_t G = t.W->K / 64;
    for (int64_t i = b; i < e; i++) {
        const int64_t n0 = i * t.nb, n1 = std::min(t.W->N, n0 + t.nb);
        for (int64_t mb = 0; mb < t.nmb; mb++) {
            const int64_t m0 = mb * 8;
            const int mr = (int)std::min<int64_t>(8, t.X->M - m0);
            t.I->tile(*t.W, *t.X, m0, mr, t.xsT ? t.xsT + mb * G * 8 : nullptr, n0, n1, t.yr + m0);
            if (t.post) t.post->fn(t.post->ctx, m0, m0 + mr, n0, n1);
        }
    }
}

struct PrepCtx { const LinQImpl *I; const QW *W; const QX *X; void *aux; };
inline void prep_task(void *c, int64_t b, int64_t e) {  // items: pairs of rows
    const PrepCtx &t = *(const PrepCtx *)c;
    t.I->prep(*t.W, *t.X, b, e, t.aux);
}

template <typename T>
inline T *const *strided_rows(std::vector<T *> &v, T *p, int64_t ld, int64_t M) {
    v.resize((size_t)M);
    for (int64_t m = 0; m < M; m++) v[m] = p + m * ld;
    return v.data();
}
}  // namespace detail

// M float rows xr[m] (K inputs each, K a multiple of 64) -> q [M][K], s / xs [M][K / 64].
inline void quant16(const Par &par, const float *const *xr, int64_t M, int64_t K, int16_t *q, float *s, float *xs,
                    const LinQImpl &I = lin_q_impl()) {
    const int64_t G = K / 64;
    detail::QuantCtx c{&I, xr, K, G, q, s, xs};
    par(M * G, 256, detail::quant_task, &c);
}
inline void quant16(const Par &par, const float *x, int64_t ldx, int64_t M, int64_t K, int16_t *q, float *s, float *xs,
                    const LinQImpl &I = lin_q_impl()) {
    static thread_local std::vector<const float *> xr;
    quant16(par, detail::strided_rows(xr, x, ldx, M), M, K, q, s, xs, I);
}

inline void lin_q(const Par &par, const QW &W, const QX &X, float *const *yr, const LinQImpl &I, const LinPost *post = nullptr);

// Which variant is fastest differs by CPU, kind, rows and threads (e.g. Zen 4: 512-bit tiles slower at one row, faster
// at many; Sapphire Rapids: AVX-VNNI fastest at several rows and threads). All give the same bits, so the first real
// calls of each class (kind, rows 1 / 2-8 / more, thread count) take the variants in turn, each timed per weight as the
// real pass runs it (different layers, weights from memory), and from the fifth call of every variant on the class
// uses the one with the smallest median.
namespace detail {
struct Pick {
    std::atomic<int8_t> chosen{0};  // 1-based index into lin_q_impls(), 0 while measuring
    int n[4] = {0, 0, 0, 0};
    float t[4][5];
};
inline Pick &pick_slot(int kind, int64_t M, int threads) {
    static Pick p[2][3][65];
    return p[kind & 1][M == 1 ? 0 : M <= 8 ? 1 : 2][std::max(1, std::min(threads, 64))];
}
inline std::mutex &pick_mutex() { static std::mutex m; return m; }
inline const std::vector<const LinQImpl *> &pick_cands() { static const std::vector<const LinQImpl *> c = lin_q_impls(); return c; }
}  // namespace detail

// The measured choice of each class for this thread count, e.g. "q4 1:vnni256 2-8:avx512 9+:avx512; q8 ..." ("?": still
// measuring).
inline std::string lin_q_choices(int threads) {
    const auto &c = detail::pick_cands();
    std::string r;
    for (int kind = 0; kind < 2; kind++) {
        r += kind ? "; q8" : "q4";
        const char *cls[3] = {" 1:", " 2-8:", " 9+:"};
        const int64_t Ms[3] = {1, 8, 9};
        for (int k = 0; k < 3; k++) {
            const int v = c.size() == 1 ? 1 : detail::pick_slot(kind, Ms[k], threads).chosen.load();
            r += std::string(cls[k]) + (v > 0 ? c[v - 1]->name : "?");
        }
    }
    return r;
}

// yr[m][n] = W x_m (+ bias) for the M activation rows of X (any M). Work is split over outputs only; each output row
// is the same bits for any M, position in the batch, thread count or split, and for every variant.
// post (optional) on each finished tile.
inline void lin_q(const Par &par, const QW &W, const QX &X, float *const *yr, const LinPost *post = nullptr) {
    const auto &c = detail::pick_cands();
    if (c.size() == 1) { lin_q(par, W, X, yr, *c[0], post); return; }
    detail::Pick &p = detail::pick_slot(W.kind, X.M, par.threads);
    const int8_t v = p.chosen.load(std::memory_order_acquire);
    if (v > 0) { lin_q(par, W, X, yr, *c[v - 1], post); return; }
    size_t i = 0;
    {
        std::lock_guard<std::mutex> lock(detail::pick_mutex());
        for (size_t j = 1; j < c.size(); j++) if (p.n[j] < p.n[i]) i = j;
    }
    const auto t0 = std::chrono::steady_clock::now();
    lin_q(par, W, X, yr, *c[i], post);
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / ((double)W.N * W.K);
    std::lock_guard<std::mutex> lock(detail::pick_mutex());
    if (p.chosen.load() || p.n[i] >= 5) return;
    p.t[i][p.n[i]++] = (float)t;
    for (size_t j = 0; j < c.size(); j++) if (p.n[j] < 5) return;
    int8_t best = 1;
    float bm = 1e30f;
    for (size_t j = 0; j < c.size(); j++) {
        float u[5];
        std::copy(p.t[j], p.t[j] + 5, u);
        std::nth_element(u, u + 2, u + 5);
        if (u[2] < bm) { bm = u[2]; best = (int8_t)(j + 1); }
    }
    p.chosen.store(best, std::memory_order_release);
}
inline void lin_q(const Par &par, const QW &W, const QX &X, float *const *yr, const LinQImpl &I, const LinPost *post) {
    const int64_t M = X.M, G = W.K / 64, nmb = (M + 7) / 8;
    static thread_local std::vector<float> xsT;  // per block of 8 rows: xs transposed [G][8] (missing rows 0)
    static thread_local std::vector<uint8_t> aux;
    QX Xa = X;
    if (I.prep) {  // the variant's own activation layout, once per call
        aux.resize(I.aux_bytes(M, X.K));
        Xa.aux = aux.data();
        detail::PrepCtx pc{&I, &W, &Xa, aux.data()};
        par((M + 1) / 2, 4, detail::prep_task, &pc);
    } else {
        xsT.assign((size_t)(nmb * G * 8), 0.f);
        for (int64_t m = 0; m < M; m++)
            for (int64_t g = 0; g < G; g++) xsT[(size_t)(((m / 8) * G + g) * 8 + m % 8)] = X.xs[m * G + g];
    }
    // blocks of output rows: about 6 per thread (also even under a static split), a multiple of the tiles' row unit
    // (AVX-512: pairs; aarch64: quads)
    const int64_t want = std::max<int64_t>(1, (int64_t)par.threads * 6), unit = I.prep ? 4 : 2;
    const int64_t nb = std::max<int64_t>(4, ((W.N + want - 1) / want + unit - 1) / unit * unit);
    detail::LinCtx c{&I, &W, &Xa, I.prep ? nullptr : xsT.data(), yr, nb, nmb, post};
    par((W.N + nb - 1) / nb, 1, detail::lin_task, &c);
}
inline void lin_q(const Par &par, const QW &W, const QX &X, float *y, int64_t ldy, const LinQImpl *I = nullptr,
                  const LinPost *post = nullptr) {
    static thread_local std::vector<float *> yr;
    float *const *r = detail::strided_rows(yr, y, ldy, X.M);
    if (I) lin_q(par, W, X, r, *I, post);
    else lin_q(par, W, X, r, post);
}

// Activation codes of many rows (reused buffers).
struct QBuf {
    std::vector<int16_t> q;
    std::vector<float> s, xs;
};

// quant16 + lin_q: yr[m] = W xr[m] for M float rows given by pointers (rows may belong to different streams).
// I: a fixed variant (checks); default: the measured choice.
inline void lin_q_f32(const Par &par, const QW &W, const float *const *xr, int64_t M, float *const *yr, QBuf &b,
                      const LinQImpl *I = nullptr, const LinPost *post = nullptr) {
    const int64_t K = W.K, G = K / 64;
    if ((int64_t)b.q.size() < M * K) b.q.resize((size_t)(M * K));
    if ((int64_t)b.s.size() < M * G) { b.s.resize((size_t)(M * G)); b.xs.resize((size_t)(M * G)); }
    quant16(par, xr, M, K, b.q.data(), b.s.data(), b.xs.data(), I ? *I : lin_q_impl());
    const QX X{b.q.data(), b.s.data(), b.xs.data(), M, K};
    if (I) lin_q(par, W, X, yr, *I, post);
    else lin_q(par, W, X, yr, post);
}
// rows ldx / ldy apart (may overlap)
inline void lin_q_f32(const Par &par, const QW &W, const float *x, int64_t ldx, int64_t M, float *y, int64_t ldy,
                      QBuf &b, const LinQImpl *I = nullptr, const LinPost *post = nullptr) {
    static thread_local std::vector<const float *> xr;
    static thread_local std::vector<float *> yr;
    lin_q_f32(par, W, detail::strided_rows(xr, x, ldx, M), M, detail::strided_rows(yr, y, ldy, M), b, I, post);
}

}  // namespace kern
