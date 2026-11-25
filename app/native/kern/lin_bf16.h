// Quantized linears in bf16 (an option, off by default: it changes the numbers): weights dequantized to bf16 panels
// (w = code * scale + min in fp32, rounded), activations rounded to bf16, products summed in fp32 over the whole K. On
// AMX-BF16 (Sapphire Rapids and later) or AVX512-BF16 (Zen 4 and later); none elsewhere (the caller keeps lin_q).
//
// Per output the float work is fixed (k pairs in order), so each row is the same bits for any M, position in the batch,
// thread count or split within one variant; AMX and AVX512-BF16 differ from each other (their own orders).
// The scalar reference blocks follow the instructions' documented semantics (round to nearest even, denormal inputs
// and results flushed to zero): the hardware variants must equal them bit for bit.
#pragma once
#include "lin_q.h"

#define KERN_T_AVX512BF16 "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512bf16"
#define KERN_T_AMX "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512bf16,amx-tile,amx-bf16"

namespace kern {

// fp32 -> bf16 as vcvtneps2bf16: round to nearest even, denormals to signed zero, NaN quiet.
inline uint16_t f2bf(float f) {
    uint32_t u;
    memcpy(&u, &f, 4);
    if ((u & 0x7f800000u) == 0) return (uint16_t)((u >> 16) & 0x8000);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (uint16_t)((u >> 16) | 0x40);
    return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1)) >> 16);
}

// Activations of M rows in k-pair layout: xb[p][m][2] = bf16(x[m][2p]), bf16(x[m][2p + 1]); Mp = M rounded up to 32
// (rows from M on are zero).
// Weight panel: 32 rows n0..n0+31 of K bf16 each (rows from N on zero).
struct LinBF16Impl {
    const char *name;
    void (*act)(const float *const *xr, int64_t M, int64_t K, uint16_t *xb, int64_t Mp);
    void (*panel)(const QW &W, int64_t n0, uint16_t *p);
    // yr[m][n0..n0+31] = panel . x[m] (+ bias) for m < M
    void (*block)(const uint16_t *panel, int64_t K, const uint16_t *xb, int64_t Mp, int64_t M, const QW &W, int64_t n0,
                  float *const *yr);
};

// ---------------------------------------------------------------- scalar reference (tests)
namespace bfref {

inline float ftz(float v) { return std::fabs(v) < 1.17549435e-38f ? std::copysign(0.f, v) : v; }
inline float bfv(uint16_t h) { return ftz(bf2f(h)); }  // DAZ

static void act(const float *const *xr, int64_t M, int64_t K, uint16_t *xb, int64_t Mp) {
    for (int64_t p = 0; p < K / 2; p++)
        for (int64_t m = 0; m < Mp; m++)
            for (int j = 0; j < 2; j++) xb[(p * Mp + m) * 2 + j] = m < M ? f2bf(xr[m][2 * p + j]) : 0;
}

static void panel(const QW &W, int64_t n0, uint16_t *p) {
    const int64_t K = W.K;
    for (int64_t r = 0; r < 32; r++) {
        const int64_t n = n0 + r;
        for (int64_t k = 0; k < K; k++) {
            float w = 0.f;
            if (n < W.N) {
                if (W.kind == 0) {
                    const int64_t g = k / 64, i = k % 64;
                    const uint8_t b = W.codes[(n * (K / 64) + g) * 32 + i % 32];
                    const int c = i < 32 ? (b & 15) : (b >> 4);
                    w = std::fmaf((float)c, h2f_sw(W.scales[n * (K / 64) + g]), h2f_sw(W.mins[n * (K / 64) + g]));
                } else {
                    const int64_t g = k / 128;
                    const int c = (int8_t)W.codes[n * K + k];
                    w = std::fmaf((float)c, h2f_sw(W.scales[n * (K / 128) + g]), h2f_sw(W.mins[n * (K / 128) + g]));
                }
            }
            p[r * K + k] = f2bf(w);
        }
    }
}

// acc + a b, products exact (bf16 x bf16), each sum rounded and flushed
inline float acc2(float acc, float a, float b) { return ftz(acc + ftz(a * b)); }

template <bool AMX>
static void block(const uint16_t *pn, int64_t K, const uint16_t *xb, int64_t Mp, int64_t M, const QW &W, int64_t n0,
                  float *const *yr) {
    for (int64_t m = 0; m < M; m++)
        for (int64_t r = 0; r < 32 && n0 + r < W.N; r++) {
            float acc = 0.f;
            for (int64_t p = 0; p < K / 2; p++) {
                const float w0 = bfv(pn[r * K + 2 * p]), w1 = bfv(pn[r * K + 2 * p + 1]);
                const float x0 = bfv(xb[(p * Mp + m) * 2]), x1 = bfv(xb[(p * Mp + m) * 2 + 1]);
                if (AMX) { acc = acc2(acc, w0, x0); acc = acc2(acc, w1, x1); }   // tdpbf16ps: even, then odd
                else { acc = acc2(acc, w1, x1); acc = acc2(acc, w0, x0); }       // vdpbf16ps: odd, then even
            }
            yr[m][n0 + r] = acc + (W.bias ? W.bias[n0 + r] : 0.f);
        }
}

}  // namespace bfref

inline const LinBF16Impl &impl_bf16_ref_amx() { static const LinBF16Impl r{"ref-amx", bfref::act, bfref::panel, bfref::block<true>}; return r; }
inline const LinBF16Impl &impl_bf16_ref_dp() { static const LinBF16Impl r{"ref-avx512bf16", bfref::act, bfref::panel, bfref::block<false>}; return r; }

#ifdef KERN_X86
// ---------------------------------------------------------------- AVX512-BF16
KERN_TARGET_BEGIN(KERN_T_AVX512BF16)
namespace avx512bf16 {

// 16 x 16 dword transpose of r[16] in place
static inline void tr16(__m512i *r) {
    __m512i t[16];
    for (int i = 0; i < 16; i += 2) {
        t[i] = _mm512_unpacklo_epi32(r[i], r[i + 1]);
        t[i + 1] = _mm512_unpackhi_epi32(r[i], r[i + 1]);
    }
    for (int i = 0; i < 16; i += 4) {
        r[i] = _mm512_unpacklo_epi64(t[i], t[i + 2]);
        r[i + 1] = _mm512_unpackhi_epi64(t[i], t[i + 2]);
        r[i + 2] = _mm512_unpacklo_epi64(t[i + 1], t[i + 3]);
        r[i + 3] = _mm512_unpackhi_epi64(t[i + 1], t[i + 3]);
    }
    for (int i = 0; i < 8; i++) {
        const int a = (i & 3) + (i >> 2) * 8;  // rows 0..3, 8..11 pair with 4..7, 12..15
        t[a] = _mm512_shuffle_i32x4(r[a], r[a + 4], 0x88);
        t[a + 4] = _mm512_shuffle_i32x4(r[a], r[a + 4], 0xdd);
    }
    for (int i = 0; i < 8; i++) {
        const int a = i;
        r[a] = _mm512_shuffle_i32x4(t[a], t[a + 8], 0x88);
        r[a + 8] = _mm512_shuffle_i32x4(t[a], t[a + 8], 0xdd);
    }
}

// 16 floats -> 16 bf16 (vcvtneps2bf16); the barrier keeps clang 18 from merging two results into a bf16 vector it
// cannot select
static inline __m256i bf16x16(__m512 v) {
    __m256i r = (__m256i)_mm512_cvtneps_pbh(v);
    __asm__("" : "+x"(r));
    return r;
}

static void act(const float *const *xr, int64_t M, int64_t K, uint16_t *xb, int64_t Mp) {
    uint32_t *xd = (uint32_t *)xb;  // one k pair = 32 bits
    for (int64_t m0 = 0; m0 < Mp; m0 += 16)
        for (int64_t p0 = 0; p0 < K / 2; p0 += 16) {  // a 16 rows x 16 pairs block, transposed into [pair][row]
            __m512i r[16];
            for (int i = 0; i < 16; i++) {
                if (m0 + i >= M) { r[i] = _mm512_setzero_si512(); continue; }
                const float *x = xr[m0 + i] + 2 * p0;
                const __m256i lo = bf16x16(_mm512_loadu_ps(x)), hi = bf16x16(_mm512_loadu_ps(x + 16));
                r[i] = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
            }
            tr16(r);
            for (int i = 0; i < 16; i++) _mm512_storeu_si512(xd + (p0 + i) * Mp + m0, r[i]);
        }
}

static inline void put32(uint16_t *d, __m512 a, __m512 b) {  // 32 floats (a, then b) -> bf16
    _mm256_storeu_si256((__m256i *)d, bf16x16(a));
    _mm256_storeu_si256((__m256i *)(d + 16), bf16x16(b));
}

static void panel(const QW &W, int64_t n0, uint16_t *p) {
    const int64_t K = W.K;
    for (int64_t r = 0; r < 32; r++) {
        const int64_t n = n0 + r;
        uint16_t *d = p + r * K;
        if (n >= W.N) { memset(d, 0, (size_t)K * 2); continue; }
        if (W.kind == 0) {
            const int64_t G = K / 64;
            const uint8_t *c = W.codes + n * G * 32;
            for (int64_t g = 0; g < G; g++) {
                const __m512 s = _mm512_set1_ps(h2f(W.scales[n * G + g])), mn = _mm512_set1_ps(h2f(W.mins[n * G + g]));
                const __m128i b0 = _mm_loadu_si128((const __m128i *)(c + g * 32)), b1 = _mm_loadu_si128((const __m128i *)(c + g * 32 + 16));
                const __m512i m4 = _mm512_set1_epi32(15);
                const __m512i i0 = _mm512_cvtepu8_epi32(b0), i1 = _mm512_cvtepu8_epi32(b1);
                const __m512 w0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_and_si512(i0, m4)), s, mn);  // k 0..15
                const __m512 w1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_and_si512(i1, m4)), s, mn);  // 16..31
                const __m512 w2 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_srli_epi32(i0, 4)), s, mn);  // 32..47
                const __m512 w3 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_srli_epi32(i1, 4)), s, mn);  // 48..63
                put32(d + g * 64, w0, w1);
                put32(d + g * 64 + 32, w2, w3);
            }
        } else {
            const int64_t G = K / 128;
            const int8_t *c = (const int8_t *)W.codes + n * K;
            for (int64_t g = 0; g < G; g++) {
                const __m512 s = _mm512_set1_ps(h2f(W.scales[n * G + g])), mn = _mm512_set1_ps(h2f(W.mins[n * G + g]));
                for (int j = 0; j < 128; j += 32) {
                    const __m512 a = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(c + g * 128 + j)))), s, mn);
                    const __m512 b = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(c + g * 128 + j + 16)))), s, mn);
                    put32(d + g * 128 + j, a, b);
                }
            }
        }
    }
}

// lanes over m: acc[r][j] (rows n0 + nb + r, m 16 j ..) += pair of x . pair of w, k pairs in order
static void block(const uint16_t *pn, int64_t K, const uint16_t *xb, int64_t Mp, int64_t M, const QW &W, int64_t n0,
                  float *const *yr) {
    const uint32_t *xd = (const uint32_t *)xb;
    alignas(64) float c[4][32];
    for (int64_t m0 = 0; m0 < M; m0 += 32)
        for (int64_t nb = 0; nb < 32 && n0 + nb < W.N; nb += 4) {
            __m512 acc[4][2];
            for (int r = 0; r < 4; r++) acc[r][0] = acc[r][1] = _mm512_setzero_ps();
            const uint32_t *w[4];
            for (int r = 0; r < 4; r++) w[r] = (const uint32_t *)(pn + (nb + r) * K);
            for (int64_t p = 0; p < K / 2; p++) {
                const __m512bh x0 = (__m512bh)_mm512_loadu_si512(xd + p * Mp + m0);
                const __m512bh x1 = (__m512bh)_mm512_loadu_si512(xd + p * Mp + m0 + 16);
                for (int r = 0; r < 4; r++) {
                    const __m512bh wv = (__m512bh)_mm512_set1_epi32((int)w[r][p]);
                    acc[r][0] = _mm512_dpbf16_ps(acc[r][0], x0, wv);
                    acc[r][1] = _mm512_dpbf16_ps(acc[r][1], x1, wv);
                }
            }
            for (int r = 0; r < 4; r++) { _mm512_store_ps(c[r], acc[r][0]); _mm512_store_ps(c[r] + 16, acc[r][1]); }
            const int64_t mn = std::min<int64_t>(32, M - m0), rn = std::min<int64_t>(4, W.N - n0 - nb);
            for (int64_t m = 0; m < mn; m++)
                for (int64_t r = 0; r < rn; r++) yr[m0 + m][n0 + nb + r] = c[r][m] + (W.bias ? W.bias[n0 + nb + r] : 0.f);
        }
}

}  // namespace avx512bf16
KERN_TARGET_END

// ---------------------------------------------------------------- AMX-BF16
KERN_TARGET_BEGIN(KERN_T_AMX)
namespace amx {
#define KERN_TCFG(c) _tile_loadconfig(c)
#define KERN_TZERO(t) _tile_zero(t)
#define KERN_TLOAD(t, p, s) _tile_loadd(t, p, s)
#define KERN_TDP(c, a, b) _tile_dpbf16ps(c, a, b)
#define KERN_TSTORE(t, p, s) _tile_stored(t, p, s)
#define KERN_TREL() _tile_release()
#include "lin_bf16_amx.inc"
#undef KERN_TCFG
#undef KERN_TZERO
#undef KERN_TLOAD
#undef KERN_TDP
#undef KERN_TSTORE
#undef KERN_TREL
}  // namespace amx
KERN_TARGET_END

inline const LinBF16Impl &impl_bf16_dp() { static const LinBF16Impl r{"avx512bf16", avx512bf16::act, avx512bf16::panel, avx512bf16::block}; return r; }
inline const LinBF16Impl &impl_bf16_amx() { static const LinBF16Impl r{"amx-bf16", avx512bf16::act, avx512bf16::panel, amx::amx_block}; return r; }
#endif

inline bool has_avx512bf16() { return has_avx512() && isa().avx512bf16; }
inline bool has_amx_bf16() { return has_avx512bf16() && isa().amx_tile && isa().amx_bf16; }

// The bf16 variant of this CPU (within isa_limit), or null.
inline const LinBF16Impl *lin_bf16_impl() {
#ifdef KERN_X86
    return has_amx_bf16() ? &impl_bf16_amx() : has_avx512bf16() ? &impl_bf16_dp() : nullptr;
#else
    return nullptr;
#endif
}

struct BF16Buf {
    std::vector<uint16_t> xb;
};

namespace detail {
struct BF16Ctx { const LinBF16Impl *I; const QW *W; const uint16_t *xb; int64_t Mp, M; float *const *yr; };
inline void bf16_task(void *c, int64_t b, int64_t e) {
    const BF16Ctx &t = *(const BF16Ctx *)c;
    static thread_local std::vector<uint16_t> pn;
    if ((int64_t)pn.size() < 32 * t.W->K) pn.resize((size_t)(32 * t.W->K));
    for (int64_t i = b; i < e; i++) {
        t.I->panel(*t.W, i * 32, pn.data());
        t.I->block(pn.data(), t.W->K, t.xb, t.Mp, t.M, *t.W, i * 32, t.yr);
    }
}
}  // namespace detail

// yr[m] = W xr[m] in bf16 (any M); tasks of 32 output rows, each weight dequantized once per call.
inline void lin_bf16_f32(const Par &par, const QW &W, const float *const *xr, int64_t M, float *const *yr, BF16Buf &b,
                         const LinBF16Impl &I) {
    const int64_t K = W.K, Mp = (M + 31) / 32 * 32;
    if ((int64_t)b.xb.size() < K * Mp) b.xb.resize((size_t)(K * Mp));
    I.act(xr, M, K, b.xb.data(), Mp);
    detail::BF16Ctx c{&I, &W, b.xb.data(), Mp, M, yr};
    par((W.N + 31) / 32, 1, detail::bf16_task, &c);
}
inline void lin_bf16_f32(const Par &par, const QW &W, const float *x, int64_t ldx, int64_t M, float *y, int64_t ldy,
                         BF16Buf &b, const LinBF16Impl &I) {
    static thread_local std::vector<const float *> xr;
    static thread_local std::vector<float *> yr;
    lin_bf16_f32(par, W, detail::strided_rows(xr, x, ldx, M), M, detail::strided_rows(yr, y, ldy, M), b, I);
}

}  // namespace kern
