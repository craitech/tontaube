// Single-query attention over a 16-bit KV cache (bf16: V1.1, fp16: Flash), AVX2 and NEON (torch-free).
//
// Each query row i: q[i] [nq][128] fp32, query heads h * rep .. h * rep + rep - 1 belong to KV head h (rep = nq / nkv
// <= 8); its cache: KV head h's rows at K[i] + h * ks, V[i] + h * vs (128 values each; rows may use different caches,
// e.g. one per stream); attends to rows r0[i]..r1[i]-1; output out[i] [nq][128].
// Rows go in chunks of C per task (query row, KV head, chunk), each K / V row read once for all its query heads; the
// chunks' partial softmaxes are merged at the end. The result depends on C, not on the thread count or split.
#pragma once
#include "act.h"

namespace kern {

enum { KV_BF16 = 0, KV_F16 = 1 };

#ifdef KERN_X86
KERN_TARGET_BEGIN(KERN_T_AVX2)
namespace avx2 {

template <int T>
static inline __m256 kv8(const uint16_t *p) { return T == KV_BF16 ? bf16x8(p) : f16x8(p); }

struct AttnCtx {
    const float *const *q;
    const uint16_t *const *K, *const *V;
    float *const *out;
    int64_t ks, vs, rep, nkv, C;
    const int64_t *r0, *r1, *toff;  // per query row: range, first task
    const int32_t *trow;            // per task: query row
    float fs;
    float *part;  // per (task, query head): acc[128], max, sum
};

// scores of rows j..j+R-1 for query head qr: a = k[0] * q[0], then fma over the 16 vectors in order
template <int T, int R>
static inline void scores(const uint16_t *K, int64_t j, const float *qr, float fs, float *s) {
    __m256 a[R];
    for (int u = 0; u < R; u++) a[u] = _mm256_mul_ps(kv8<T>(K + (j + u) * 128), _mm256_loadu_ps(qr));
    for (int i = 1; i < 16; i++) {
        const __m256 qv = _mm256_loadu_ps(qr + 8 * i);
        for (int u = 0; u < R; u++) a[u] = _mm256_fmadd_ps(kv8<T>(K + (j + u) * 128 + 8 * i), qv, a[u]);
    }
    for (int u = 0; u < R; u++) s[u] = hsum(a[u]) * fs;
}

template <int T>
static void attn_task(void *cp, int64_t t0, int64_t t1) {
    const AttnCtx &c = *(const AttnCtx *)cp;
    const int64_t rep = c.rep, C = c.C;
    static thread_local std::vector<float> sbuf;
    if ((int64_t)sbuf.size() < rep * C + 8) sbuf.resize((size_t)(rep * C + 8));
    float *s = sbuf.data();
    for (int64_t t = t0; t < t1; t++) {
        const int64_t qi = c.trow[t], L = c.r1[qi] - c.r0[qi], nc = (L + C - 1) / C, k = t - c.toff[qi];
        const int64_t h = k / nc, j0 = (k % nc) * C, j1 = std::min(L, j0 + C), n = j1 - j0;
        const uint16_t *K = c.K[qi] + h * c.ks + c.r0[qi] * 128, *V = c.V[qi] + h * c.vs + c.r0[qi] * 128;
        for (int64_t r = 0; r < rep; r++) {
            const float *qr = c.q[qi] + (h * rep + r) * 128;
            int64_t j = j0;
            for (; j + 4 <= j1; j += 4) scores<T, 4>(K, j, qr, c.fs, s + r * C + (j - j0));
            for (; j < j1; j++) scores<T, 1>(K, j, qr, c.fs, s + r * C + (j - j0));
        }
        for (int64_t r = 0; r < rep; r++) {
            float *sr = s + r * C, mx = -INFINITY, sum = 0.f;
            for (int64_t j = 0; j < n; j++) mx = std::max(mx, sr[j]);
            int64_t j = 0;
            __m256 vsum = _mm256_setzero_ps();
            for (; j + 8 <= n; j += 8) {
                const __m256 e = vexp_neg(_mm256_sub_ps(_mm256_loadu_ps(sr + j), _mm256_set1_ps(mx)));
                _mm256_storeu_ps(sr + j, e);
                vsum = _mm256_add_ps(vsum, e);
            }
            sum = hsum(vsum);
            for (; j < n; j++) { sr[j] = std::exp(sr[j] - mx); sum += sr[j]; }
            float *dst = c.part + (t * rep + r) * 130;
            dst[128] = mx;
            dst[129] = sum;
        }
        for (int pass = 0; pass < 4; pass++) {  // 32 dims per pass: the accumulators stay in registers
            __m256 acc[8][4];
            for (int64_t r = 0; r < rep; r++)
                for (int i = 0; i < 4; i++) acc[r][i] = _mm256_setzero_ps();
            for (int64_t j = j0; j < j1; j++) {
                __m256 vv[4];
                for (int i = 0; i < 4; i++) vv[i] = kv8<T>(V + j * 128 + pass * 32 + 8 * i);
                for (int64_t r = 0; r < rep; r++) {
                    const __m256 p = _mm256_set1_ps(s[r * C + (j - j0)]);
                    for (int i = 0; i < 4; i++) acc[r][i] = _mm256_fmadd_ps(p, vv[i], acc[r][i]);
                }
            }
            for (int64_t r = 0; r < rep; r++) {
                float *dst = c.part + (t * rep + r) * 130 + pass * 32;
                for (int i = 0; i < 4; i++) _mm256_storeu_ps(dst + 8 * i, acc[r][i]);
            }
        }
    }
}

// out = sum_c e^(m_c - M) acc_c / sum_c e^(m_c - M) l_c
static void attn_merge(void *cp, int64_t h0, int64_t h1) {
    const AttnCtx &c = *(const AttnCtx *)cp;
    const int64_t rep = c.rep, nq = c.nkv * rep;
    for (int64_t ih = h0; ih < h1; ih++) {  // (query row, query head)
        const int64_t qi = ih / nq, hq = ih % nq, h = hq / rep, r = hq % rep;
        const int64_t nc = (c.r1[qi] - c.r0[qi] + c.C - 1) / c.C;
        const float *pb = c.part + (c.toff[qi] + h * nc) * rep * 130;
        float M = -INFINITY, den = 0.f;
        for (int64_t k = 0; k < nc; k++) M = std::max(M, pb[(k * rep + r) * 130 + 128]);
        __m256 o[16];
        for (int i = 0; i < 16; i++) o[i] = _mm256_setzero_ps();
        for (int64_t k = 0; k < nc; k++) {
            const float *src = pb + (k * rep + r) * 130;
            const float w = std::exp(src[128] - M);
            den = _mm_cvtss_f32(_mm_add_ss(_mm_set_ss(den), _mm_mul_ss(_mm_set_ss(w), _mm_set_ss(src[129]))));  // unfused
            const __m256 wv = _mm256_set1_ps(w);
            for (int i = 0; i < 16; i++) o[i] = _mm256_fmadd_ps(wv, _mm256_loadu_ps(src + 8 * i), o[i]);
        }
        const __m256 inv = _mm256_set1_ps(1.f / den);
        for (int i = 0; i < 16; i++) _mm256_storeu_ps(c.out[qi] + hq * 128 + 8 * i, _mm256_mul_ps(o[i], inv));
    }
}

}  // namespace avx2
KERN_TARGET_END
namespace attn_isa = avx2;
#endif
#ifdef KERN_ARM
#include "attn_neon.inc"
namespace attn_isa = neon;
#endif

#if defined(KERN_X86) || defined(KERN_ARM)
// x86: needs AVX2 + FMA + F16C (has_avx2()). T: KV_BF16 or KV_F16. Rows with r1 <= r0 are not allowed.
template <int T>
inline void attn_rows(const Par &par, int64_t rows, const float *const *q, const uint16_t *const *K,
                      const uint16_t *const *V, int64_t ks, int64_t vs, int64_t nq, int64_t nkv, const int64_t *r0,
                      const int64_t *r1, float fs, int64_t C, float *const *out) {
    const int64_t rep = nq / nkv;
    static thread_local std::vector<int64_t> toff;
    static thread_local std::vector<int32_t> trow;
    static thread_local std::vector<float> part;
    toff.resize((size_t)rows);
    int64_t nt = 0;
    for (int64_t i = 0; i < rows; i++) { toff[i] = nt; nt += nkv * ((r1[i] - r0[i] + C - 1) / C); }
    trow.resize((size_t)nt);
    for (int64_t i = 0; i < rows; i++)
        for (int64_t t = toff[i]; t < (i + 1 < rows ? toff[i + 1] : nt); t++) trow[t] = (int32_t)i;
    if ((int64_t)part.size() < nt * rep * 130) part.resize((size_t)(nt * rep * 130));
    attn_isa::AttnCtx c{q, K, V, out, ks, vs, rep, nkv, C, r0, r1, toff.data(), trow.data(), fs, part.data()};
    par(nt, 1, attn_isa::attn_task<T>, &c);
    par(rows * nq, nt >= 4 * nkv ? 1 : rows * nq, attn_isa::attn_merge, &c);
}

// Query / output rows ldq / ldo apart over one cache.
template <int T>
inline void attn_rows(const Par &par, int64_t rows, const float *q, int64_t ldq, const uint16_t *K, const uint16_t *V,
                      int64_t ks, int64_t vs, int64_t nq, int64_t nkv, const int64_t *r0, const int64_t *r1, float fs,
                      int64_t C, float *out, int64_t ldo) {
    static thread_local std::vector<const float *> qr;
    static thread_local std::vector<float *> orow;
    static thread_local std::vector<const uint16_t *> kr, vr;
    qr.resize((size_t)rows); orow.resize((size_t)rows); kr.assign((size_t)rows, K); vr.assign((size_t)rows, V);
    for (int64_t i = 0; i < rows; i++) { qr[i] = q + i * ldq; orow[i] = out + i * ldo; }
    attn_rows<T>(par, rows, qr.data(), kr.data(), vr.data(), ks, vs, nq, nkv, r0, r1, fs, C, orow.data());
}

// One query row over cache rows 0..L-1.
template <int T>
inline void attn(const Par &par, const float *q, const uint16_t *K, const uint16_t *V, int64_t ks, int64_t vs, int64_t nq,
                 int64_t nkv, int64_t L, float fs, int64_t C, float *out) {
    const int64_t r0 = 0, r1 = L;
    attn_rows<T>(par, 1, q, 0, K, V, ks, vs, nq, nkv, &r0, &r1, fs, C, out, 0);
}
#endif

}  // namespace kern
