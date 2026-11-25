// Float elementwise kernels for x86-64 (AVX2+FMA+F16C, AVX-512) and aarch64 (NEON, neon.h): exp, GELU, Mish, SiLU,
// RMS / layer norms, rotary embedding, small float linears. Torch-free; x86 callers check has_avx2().
//
// Each kernel body below is written once over the vector type vf (VW lanes) and compiled once per instruction set: this
// header includes itself inside a target region per ISA (namespaces kern::avx2, kern::avx512); kern::f dispatches.
// Reductions use 8 lanes on every ISA and elementwise math is the same per lane, so results do not depend on the ISA.
#ifndef KERN_PASS
#ifndef TT_KERN_ACT_H
#define TT_KERN_ACT_H
#include "common.h"

// The float kernels' AVX-512 needs no VNNI (Skylake-SP has none).
#define KERN_T_AVX512F "avx2,fma,f16c,avx512f,avx512bw,avx512vl"
namespace kern {
inline bool has_avx512f() { const Isa &i = isa(); return has_avx2() && i.avx512f && i.avx512bw && i.avx512vl; }
// kern::avx512::f(...) or kern::avx2::f(...) (aarch64: kern::neon::f(...))
#ifdef KERN_ARM
#define KERN_CALL(f, ...) (::kern::neon::f(__VA_ARGS__))
#else
#define KERN_CALL(f, ...) (::kern::has_avx512f() ? ::kern::avx512::f(__VA_ARGS__) : ::kern::avx2::f(__VA_ARGS__))
#endif
// par over a lambda f(begin, end)
template <class F> inline void par_for(const Par &par, int64_t n, int64_t grain, const F &f) {
    par(n, grain, [](void *c, int64_t a, int64_t b) { (*(const F *)c)(a, b); }, (void *)&f);
}
}  // namespace kern

#ifdef KERN_X86
KERN_TARGET_BEGIN(KERN_T_AVX2)
namespace kern { namespace avx2 {
// 8-lane helpers (also the reductions of the AVX-512 pass)
typedef __m256 f8;
static inline f8 ld8(const float *p) { return _mm256_loadu_ps(p); }
static inline void st8(float *p, f8 v) { _mm256_storeu_ps(p, v); }
static inline f8 set8(float x) { return _mm256_set1_ps(x); }
static inline f8 fma8(f8 a, f8 b, f8 c) { return _mm256_fmadd_ps(a, b, c); }
static inline f8 fnma8(f8 a, f8 b, f8 c) { return _mm256_fnmadd_ps(a, b, c); }
static inline f8 mul8(f8 a, f8 b) { return _mm256_mul_ps(a, b); }
static inline f8 bc8(const float *p) { return _mm256_broadcast_ss(p); }
static inline float hsum8(f8 v) { return hsum(v); }  // ((v0 + v4) + (v2 + v6)) + ((v1 + v5) + (v3 + v7))
static inline __m256i mask8(int n) { return _mm256_cmpgt_epi32(_mm256_set1_epi32(n), _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7)); }
static inline f8 ld8n(const float *p, int n) { return _mm256_maskload_ps(p, mask8(n)); }  // n < 8 lanes, rest 0
static inline void st8n(float *p, f8 v, int n) { _mm256_maskstore_ps(p, mask8(n), v); }
static inline void dadd8(const float *p, __m256d &a, __m256d &b, __m256d m) {  // a, b += (x - m)^2 of 8 floats (doubles)
    const __m256 x = _mm256_loadu_ps(p);
    const __m256d d0 = _mm256_sub_pd(_mm256_cvtps_pd(_mm256_castps256_ps128(x)), m);
    const __m256d d1 = _mm256_sub_pd(_mm256_cvtps_pd(_mm256_extractf128_ps(x, 1)), m);
    a = _mm256_fmadd_pd(d0, d0, a);
    b = _mm256_fmadd_pd(d1, d1, b);
}
static inline void dsum8(const float *p, __m256d &a, __m256d &b) {
    const __m256 x = _mm256_loadu_ps(p);
    a = _mm256_add_pd(a, _mm256_cvtps_pd(_mm256_castps256_ps128(x)));
    b = _mm256_add_pd(b, _mm256_cvtps_pd(_mm256_extractf128_ps(x, 1)));
}
typedef __m256d dv4;
static inline dv4 dzero() { return _mm256_setzero_pd(); }
static inline dv4 dset(double x) { return _mm256_set1_pd(x); }
static inline double dhsum(__m256d a, __m256d b) {
    double t[4];
    _mm256_storeu_pd(t, _mm256_add_pd(a, b));
    return (t[0] + t[2]) + (t[1] + t[3]);
}

// native vectors: 8 lanes
typedef __m256 vf;
constexpr int VW = 8;
constexpr bool CHAINS4 = false;  // elementwise maps: two vector chains per step (four on aarch64)
static inline vf vld(const float *p) { return _mm256_loadu_ps(p); }
static inline void vst(float *p, vf v) { _mm256_storeu_ps(p, v); }
static inline vf vldn(const float *p, int n) { return ld8n(p, n); }
static inline void vstn(float *p, vf v, int n) { st8n(p, v, n); }
static inline vf vset(float x) { return _mm256_set1_ps(x); }
static inline vf vzero() { return _mm256_setzero_ps(); }
static inline vf vbc(const float *p) { return _mm256_broadcast_ss(p); }
static inline vf vadd(vf a, vf b) { return _mm256_add_ps(a, b); }
static inline vf vsub(vf a, vf b) { return _mm256_sub_ps(a, b); }
static inline vf vmul(vf a, vf b) { return _mm256_mul_ps(a, b); }
static inline vf vdiv(vf a, vf b) { return _mm256_div_ps(a, b); }
static inline vf vfma(vf a, vf b, vf c) { return _mm256_fmadd_ps(a, b, c); }   // a b + c
static inline vf vfnma(vf a, vf b, vf c) { return _mm256_fnmadd_ps(a, b, c); } // c - a b
static inline vf vmax(vf a, vf b) { return _mm256_max_ps(a, b); }
static inline vf vmin(vf a, vf b) { return _mm256_min_ps(a, b); }
static inline vf vand(vf a, vf b) { return _mm256_and_ps(a, b); }
static inline vf vxor(vf a, vf b) { return _mm256_xor_ps(a, b); }
static inline vf vabs(vf a) { return _mm256_andnot_ps(_mm256_set1_ps(-0.f), a); }
static inline vf vrnd(vf a) { return _mm256_round_ps(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); }
static inline vf vpow2i(vf n) {  // 2^n for integral n
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23));
}
static inline vf vsel_ge0(vf x, vf a, vf b) { return _mm256_blendv_ps(b, a, _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_GE_OQ)); }
}}  // namespace kern::avx2
KERN_TARGET_END

KERN_TARGET_BEGIN(KERN_T_AVX512F)
namespace kern { namespace avx512 {
using avx2::f8; using avx2::ld8; using avx2::st8; using avx2::set8; using avx2::fma8; using avx2::hsum8;
using avx2::ld8n; using avx2::st8n; using avx2::dadd8; using avx2::dsum8; using avx2::dhsum;
using avx2::fnma8; using avx2::mul8; using avx2::bc8; using avx2::dv4; using avx2::dzero; using avx2::dset;
// native vectors: 16 lanes
typedef __m512 vf;
constexpr int VW = 16;
constexpr bool CHAINS4 = false;
static inline vf vld(const float *p) { return _mm512_loadu_ps(p); }
static inline void vst(float *p, vf v) { _mm512_storeu_ps(p, v); }
static inline vf vldn(const float *p, int n) { return _mm512_maskz_loadu_ps((__mmask16)((1u << n) - 1), p); }
static inline void vstn(float *p, vf v, int n) { _mm512_mask_storeu_ps(p, (__mmask16)((1u << n) - 1), v); }
static inline vf vset(float x) { return _mm512_set1_ps(x); }
static inline vf vzero() { return _mm512_setzero_ps(); }
static inline vf vbc(const float *p) { return _mm512_set1_ps(*p); }
static inline vf vadd(vf a, vf b) { return _mm512_add_ps(a, b); }
static inline vf vsub(vf a, vf b) { return _mm512_sub_ps(a, b); }
static inline vf vmul(vf a, vf b) { return _mm512_mul_ps(a, b); }
static inline vf vdiv(vf a, vf b) { return _mm512_div_ps(a, b); }
static inline vf vfma(vf a, vf b, vf c) { return _mm512_fmadd_ps(a, b, c); }
static inline vf vfnma(vf a, vf b, vf c) { return _mm512_fnmadd_ps(a, b, c); }
static inline vf vmax(vf a, vf b) { return _mm512_max_ps(a, b); }
static inline vf vmin(vf a, vf b) { return _mm512_min_ps(a, b); }
static inline vf vand(vf a, vf b) { return _mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(a), _mm512_castps_si512(b))); }
static inline vf vxor(vf a, vf b) { return _mm512_castsi512_ps(_mm512_xor_si512(_mm512_castps_si512(a), _mm512_castps_si512(b))); }
static inline vf vabs(vf a) { return _mm512_abs_ps(a); }
static inline vf vrnd(vf a) { return _mm512_roundscale_ps(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); }
static inline vf vpow2i(vf n) {
    return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_add_epi32(_mm512_cvtps_epi32(n), _mm512_set1_epi32(127)), 23));
}
static inline vf vsel_ge0(vf x, vf a, vf b) { return _mm512_mask_blend_ps(_mm512_cmp_ps_mask(x, _mm512_setzero_ps(), _CMP_GE_OQ), b, a); }
}}  // namespace kern::avx512
KERN_TARGET_END

// Each kernel header ends its prologue with these passes (its body once per ISA).
#define KERN_PASS
KERN_TARGET_BEGIN(KERN_T_AVX2)
namespace kern { namespace avx2 {
#include "act.h"
}}
KERN_TARGET_END
#define KERN_AVX512_PASS
KERN_TARGET_BEGIN(KERN_T_AVX512F)
namespace kern { namespace avx512 {
#include "act.h"
}}
KERN_TARGET_END
#undef KERN_AVX512_PASS
#undef KERN_PASS
#elif defined(KERN_ARM)
#include "neon.h"
#define KERN_PASS
#define KERN_NEON_PASS
namespace kern { namespace neon {
#include "act.h"
}}
#undef KERN_NEON_PASS
#undef KERN_PASS
#endif

namespace kern {
inline void gelu(float *x, int64_t n) { KERN_CALL(gelu, x, n); }
inline void gelu_as(float *x, int64_t n) { KERN_CALL(gelu_as, x, n); }
inline void mish(float *x, int64_t n) { KERN_CALL(mish, x, n); }
inline void silu_mul(float *a, const float *b, int64_t n) { KERN_CALL(silu_mul, a, b, n); }
inline void rms(const float *x, const float *w, int64_t n, float eps, float *out) { KERN_CALL(rms, x, w, n, eps, out); }
inline void layernorm(const float *x, const float *w, const float *b, int64_t n, float eps, float *out) {
    KERN_CALL(layernorm, x, w, b, n, eps, out);
}
inline void rope(float *v, const float *cs, const float *sn, int64_t half) { KERN_CALL(rope, v, cs, sn, half); }
inline void linear_rows(const float *W, const float *bias, int64_t K, const float *x, int64_t xs, int64_t M, float *y,
                        int64_t ys, int64_t n0, int64_t n1) {
    KERN_CALL(linear_rows, W, bias, K, x, xs, M, y, ys, n0, n1);
}
}  // namespace kern
#endif  // TT_KERN_ACT_H

#else  // ------------------------------------------------------------ body, once per ISA (vf, VW, ...)

// The elementwise functions below take one vector or a pair (vf2): a pair's two dependency chains interleave, which
// keeps more of the core busy than one long chain (GELU ~20% faster). Same arithmetic per lane either way.
struct vf2 { vf a, b; };
#define KERN_OP2(name)                                                                       \
    static inline vf2 name(vf2 x, vf2 y) { return {name(x.a, y.a), name(x.b, y.b)}; }
KERN_OP2(vadd) KERN_OP2(vsub) KERN_OP2(vmul) KERN_OP2(vdiv) KERN_OP2(vmax) KERN_OP2(vand) KERN_OP2(vxor)
#undef KERN_OP2
static inline vf2 vfma(vf2 x, vf2 y, vf2 z) { return {vfma(x.a, y.a, z.a), vfma(x.b, y.b, z.b)}; }
static inline vf2 vfnma(vf2 x, vf2 y, vf2 z) { return {vfnma(x.a, y.a, z.a), vfnma(x.b, y.b, z.b)}; }
static inline vf2 vsel_ge0(vf2 x, vf2 y, vf2 z) { return {vsel_ge0(x.a, y.a, z.a), vsel_ge0(x.b, y.b, z.b)}; }
static inline vf2 vabs(vf2 x) { return {vabs(x.a), vabs(x.b)}; }
static inline vf2 vrnd(vf2 x) { return {vrnd(x.a), vrnd(x.b)}; }
static inline vf2 vpow2i(vf2 x) { return {vpow2i(x.a), vpow2i(x.b)}; }
template <class T> static inline T vc(float c);  // c in every lane
template <> inline vf vc<vf>(float c) { return vset(c); }
template <> inline vf2 vc<vf2>(float c) { return {vset(c), vset(c)}; }
static inline vf2 vld2(const float *p) { return {vld(p), vld(p + VW)}; }
static inline void vst2(float *p, vf2 v) { vst(p, v.a); vst(p + VW, v.b); }

// exp(x) for x <= 0 (Cephes polynomial, ~2 ulp); below -87 flushes towards 0.
template <class T> static inline T vexp_neg(T x) {
    x = vmax(x, vc<T>(-87.f));
    const T fx = vrnd(vmul(x, vc<T>(1.44269504088896341f)));
    x = vfnma(fx, vc<T>(0.693359375f), x);
    x = vfnma(fx, vc<T>(-2.12194440e-4f), x);
    T y = vc<T>(1.9875691500e-4f);
    y = vfma(y, x, vc<T>(1.3981999507e-3f));
    y = vfma(y, x, vc<T>(8.3334519073e-3f));
    y = vfma(y, x, vc<T>(4.1665795894e-2f));
    y = vfma(y, x, vc<T>(1.6666665459e-1f));
    y = vfma(y, x, vc<T>(5.0000001201e-1f));
    y = vfma(y, vmul(x, x), vadd(x, vc<T>(1.f)));
    return vmul(y, vpow2i(fx));
}

// GELU x Phi(x) with erfc(z) = t exp(-z^2 + P(t)), t = 1 / (1 + z / 2) (Numerical Recipes, |error| < 1.2e-7): the
// engine's GELU (web_gelu1); P by Estrin's scheme (a shorter chain than Horner's).
template <class T> static inline T vgelu(T v) {
    const T one = vc<T>(1.f), half = vc<T>(0.5f);
    const T z = vmul(vabs(v), vc<T>(0.70710678118654752f));
    const T t = vdiv(one, vfma(half, z, one)), t2 = vmul(t, t), t4 = vmul(t2, t2);
    const T a0 = vfma(vc<T>(1.00002368f), t, vc<T>(-1.26551223f)), a1 = vfma(vc<T>(0.09678418f), t, vc<T>(0.37409196f));
    const T a2 = vfma(vc<T>(0.27886807f), t, vc<T>(-0.18628806f)), a3 = vfma(vc<T>(1.48851587f), t, vc<T>(-1.13520398f));
    const T a4 = vfma(vc<T>(0.17087277f), t, vc<T>(-0.82215223f));
    const T p = vfma(a4, vmul(t4, t4), vfma(vfma(a3, t2, a2), t4, vfma(a1, t2, a0)));
    const T erfc = vmul(t, vexp_neg(vfnma(z, z, p)));
    return vsel_ge0(v, vmul(v, vfnma(half, erfc, one)), vmul(v, vmul(half, erfc)));
}

// GELU with erf from Abramowitz-Stegun 7.1.26 as in ATen's Vectorized<float>::erf (V1.1's VibeVoice tokenizer).
template <class T> static inline T vgelu_as(T x0) {
    const T x = vmul(x0, vc<T>(0.70710678118654752f));
    const T sign = vand(vc<T>(-0.f), x), ax = vxor(sign, x);
    const T t = vdiv(vc<T>(1.f), vfma(vc<T>(0.3275911f), ax, vc<T>(1.f)));
    T r = vfma(vc<T>(1.061405429f), t, vc<T>(-1.453152027f));
    r = vfma(r, t, vc<T>(1.421413741f));
    r = vfma(r, t, vc<T>(-0.284496736f));
    r = vfma(r, t, vc<T>(0.254829592f));
    const T e = vexp_neg(vsub(vc<T>(0.f), vmul(x, x)));
    const T erf = vxor(sign, vfnma(vmul(e, t), r, vc<T>(1.f)));
    return vmul(vmul(vc<T>(0.5f), x0), vadd(vc<T>(1.f), erf));
}

// Mish x tanh(softplus(x)) with e = exp(-|x|): x >= 0: (1 + 2e) / (1 + 2e + 2e^2); x < 0: n / (n + 2), n = e (e + 2).
template <class T> static inline T vmish(T x) {
    const T one = vc<T>(1.f), two = vc<T>(2.f);
    const T e = vexp_neg(vsub(vc<T>(0.f), vabs(x)));
    const T pn = vfma(two, e, one), pd = vfma(vmul(two, e), e, pn);
    const T nn = vmul(e, vadd(e, two)), nd = vadd(nn, two);
    return vmul(x, vdiv(vsel_ge0(x, pn, nn), vsel_ge0(x, pd, nd)));
}

// silu(g) * b
template <class T> static inline T vsilu_mul(T g, T b) {
    const T e = vexp_neg(vsub(vc<T>(0.f), vabs(g)));
    const T sig = vdiv(vsel_ge0(g, vc<T>(1.f), e), vadd(e, vc<T>(1.f)));
    return vmul(vmul(g, sig), b);
}

// x[i] = op(x[i]) over n values
#define KERN_MAP(name, op)                                                   \
    static inline void name(float *x, int64_t n) {                           \
        int64_t i = 0;                                                       \
        for (; CHAINS4 && i + 4 * VW <= n; i += 4 * VW) {                    \
            const vf2 p = op(vld2(x + i)), q = op(vld2(x + i + 2 * VW));     \
            vst2(x + i, p);                                                  \
            vst2(x + i + 2 * VW, q);                                         \
        }                                                                    \
        for (; i + 2 * VW <= n; i += 2 * VW) vst2(x + i, op(vld2(x + i)));   \
        for (; i + VW <= n; i += VW) vst(x + i, op(vld(x + i)));             \
        if (i < n) vstn(x + i, op(vldn(x + i, (int)(n - i))), (int)(n - i)); \
    }
KERN_MAP(gelu, vgelu)
KERN_MAP(gelu_as, vgelu_as)
KERN_MAP(mish, vmish)
#undef KERN_MAP

// a[i] = silu(a[i]) * b[i]
static inline void silu_mul(float *a, const float *b, int64_t n) {
    int64_t i = 0;
    for (; i + 2 * VW <= n; i += 2 * VW) vst2(a + i, vsilu_mul(vld2(a + i), vld2(b + i)));
    for (; i + VW <= n; i += VW) vst(a + i, vsilu_mul(vld(a + i), vld(b + i)));
    if (i < n) vstn(a + i, vsilu_mul(vldn(a + i, (int)(n - i)), vldn(b + i, (int)(n - i))), (int)(n - i));
}

// out = x / sqrt(mean(x^2) + eps) (* w if w)
static inline float rms_scale(const float *x, int64_t n, float eps) {
    f8 acc = set8(0.f);
    int64_t i = 0;
    for (; i + 8 <= n; i += 8) { const f8 v = ld8(x + i); acc = fma8(v, v, acc); }
    if (i < n) { const f8 v = ld8n(x + i, (int)(n - i)); acc = fma8(v, v, acc); }
    return 1.f / std::sqrt(hsum8(acc) / (float)n + eps);
}
static inline void rms(const float *x, const float *w, int64_t n, float eps, float *out) {
    const vf r = vset(rms_scale(x, n, eps));
    int64_t i = 0;
    if (w) {
        for (; i + VW <= n; i += VW) vst(out + i, vmul(vmul(vld(x + i), r), vld(w + i)));
        if (i < n) vstn(out + i, vmul(vmul(vldn(x + i, (int)(n - i)), r), vldn(w + i, (int)(n - i))), (int)(n - i));
    } else {
        for (; i + VW <= n; i += VW) vst(out + i, vmul(vld(x + i), r));
        if (i < n) vstn(out + i, vmul(vldn(x + i, (int)(n - i)), r), (int)(n - i));
    }
}

// torch LayerNorm (biased variance, statistics in double); w / b null: no affine. n a multiple of 8.
static inline void layernorm(const float *x, const float *w, const float *b, int64_t n, float eps, float *out) {
    dv4 a0 = dzero(), a1 = a0;
    for (int64_t i = 0; i < n; i += 8) dsum8(x + i, a0, a1);
    const double mean = dhsum(a0, a1) / (double)n;
    const dv4 m = dset(mean);
    a0 = a1 = dzero();
    for (int64_t i = 0; i < n; i += 8) dadd8(x + i, a0, a1, m);
    const vf r = vset(1.f / std::sqrt((float)(dhsum(a0, a1) / (double)n) + eps)), mu = vset((float)mean);
    if (w)
        for (int64_t i = 0; i < n; i += VW) {
            const int k = (int)std::min<int64_t>(VW, n - i);
            const vf y = vfma(vmul(vsub(vldn(x + i, k), mu), r), vldn(w + i, k), vldn(b + i, k));
            if (k == VW) vst(out + i, y); else vstn(out + i, y, k);
        }
    else
        for (int64_t i = 0; i < n; i += VW) {
            const int k = (int)std::min<int64_t>(VW, n - i);
            const vf y = vmul(vsub(vldn(x + i, k), mu), r);
            if (k == VW) vst(out + i, y); else vstn(out + i, y, k);
        }
}

// Rotary embedding in place, half a multiple of 8: [a, b] -> [a cos - b sin, b cos + a sin].
static inline void rope(float *v, const float *cs, const float *sn, int64_t half) {
    for (int64_t i = 0; i < half; i += 8) {
        const f8 a = ld8(v + i), b = ld8(v + half + i), c = ld8(cs + i), s = ld8(sn + i);
        st8(v + i, fnma8(b, s, mul8(a, c)));
        st8(v + half + i, fma8(a, s, mul8(b, c)));
    }
}

// Small float linear, W [N][K] row-major: y[m * ys + n] = bias[n] + W[n] . x[m * xs ..] for rows m < M, n0 <= n < n1.
// One 8-lane accumulator per dot product, four weight rows at a time.
static inline void linear_rows(const float *W, const float *bias, int64_t K, const float *x, int64_t xs, int64_t M,
                               float *y, int64_t ys, int64_t n0, int64_t n1) {
    const int64_t K8 = K / 8 * 8;
    for (int64_t m = 0; m < M; m++) {
        const float *xm = x + m * xs;
        int64_t n = n0;
        for (; n + 4 <= n1; n += 4) {
            const float *w0 = W + n * K, *w1 = w0 + K, *w2 = w1 + K, *w3 = w2 + K;
            f8 c0 = set8(0.f), c1 = c0, c2 = c0, c3 = c0;
            for (int64_t k = 0; k < K8; k += 8) {
                const f8 v = ld8(xm + k);
                c0 = fma8(ld8(w0 + k), v, c0); c1 = fma8(ld8(w1 + k), v, c1);
                c2 = fma8(ld8(w2 + k), v, c2); c3 = fma8(ld8(w3 + k), v, c3);
            }
            if (K8 < K) {
                const int r = (int)(K - K8);
                const f8 v = ld8n(xm + K8, r);
                c0 = fma8(ld8n(w0 + K8, r), v, c0); c1 = fma8(ld8n(w1 + K8, r), v, c1);
                c2 = fma8(ld8n(w2 + K8, r), v, c2); c3 = fma8(ld8n(w3 + K8, r), v, c3);
            }
            float *ym = y + m * ys + n;
            ym[0] = hsum8(c0) + (bias ? bias[n] : 0.f); ym[1] = hsum8(c1) + (bias ? bias[n + 1] : 0.f);
            ym[2] = hsum8(c2) + (bias ? bias[n + 2] : 0.f); ym[3] = hsum8(c3) + (bias ? bias[n + 3] : 0.f);
        }
        for (; n < n1; n++) {
            const float *w0 = W + n * K;
            f8 c0 = set8(0.f);
            for (int64_t k = 0; k < K8; k += 8) c0 = fma8(ld8(w0 + k), ld8(xm + k), c0);
            if (K8 < K) c0 = fma8(ld8n(w0 + K8, (int)(K - K8)), ld8n(xm + K8, (int)(K - K8)), c0);
            y[m * ys + n] = hsum8(c0) + (bias ? bias[n] : 0.f);
        }
    }
}

#endif  // KERN_PASS
