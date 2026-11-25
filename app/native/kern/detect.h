// Kernels of the pitch-map detector (flash/engine/pitch_detect.h) on x86-64 and aarch64: the autocorrelation of the pitch map and
// the small convolutions (5x5 over time x pitch, the head's dilated taps) as sums of taps on channel-last rows.
// 8 lanes on every ISA; each output is one FMA chain (bias first), so results do not depend on the ISA.
#ifndef KERN_PASS
#ifndef TT_KERN_DETECT_H
#define TT_KERN_DETECT_H
#include "act.h"

#ifdef KERN_X86
#define KERN_PASS
KERN_TARGET_BEGIN(KERN_T_AVX2)
namespace kern { namespace avx2 {
#include "detect.h"
}}
KERN_TARGET_END
#define KERN_AVX512_PASS
KERN_TARGET_BEGIN(KERN_T_AVX512F)
namespace kern { namespace avx512 {
#include "detect.h"
}}
KERN_TARGET_END
#undef KERN_AVX512_PASS
#undef KERN_PASS
#elif defined(KERN_ARM)
#define KERN_PASS
#define KERN_NEON_PASS
namespace kern { namespace neon {
#include "detect.h"
}}
#undef KERN_NEON_PASS
#undef KERN_PASS
#endif

namespace kern {
// corr[tau] = sum_{i < win} x[i] x[i + tau] for tau = 0..tmax (x: win + tmax values).
inline void autocorr(const float *x, int win, int tmax, float *corr) { KERN_CALL(autocorr, x, win, tmax, corr); }
// out[p][o] = bias[o] + sum_t sum_{i < ci} in[t][p * lda + i] w[t][i][o] for p < P, o < co (co a multiple of 8);
// taps with in[t] null are skipped.
inline void taps_conv(int ntaps, const float *const *in, int64_t lda, int P, int ci, int co, const float *w,
                      const float *bias, float *out) {
    KERN_CALL(taps_conv, ntaps, in, lda, P, ci, co, w, bias, out);
}
}  // namespace kern
#endif  // TT_KERN_DETECT_H

#else  // ------------------------------------------------------------ body, once per ISA

// lags t0 .. t0 + 8 NV - 1 (the last vector: r lanes)
template <int NV>
static inline void ac_block(const float *x, int win, int t0, int r, float *corr) {
    f8 a0 = set8(0.f), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    const float *q = x + t0;
#define KERN_LAG(j) if (NV > j) a##j = fma8(v, j == NV - 1 && r < 8 ? ld8n(q + i + 8 * j, r) : ld8(q + i + 8 * j), a##j);
    for (int i = 0; i < win; i++) {
        const f8 v = bc8(x + i);
        KERN_LAG(0) KERN_LAG(1) KERN_LAG(2) KERN_LAG(3) KERN_LAG(4) KERN_LAG(5) KERN_LAG(6) KERN_LAG(7)
    }
#undef KERN_LAG
#define KERN_PUT(j) if (NV > j) { if (j == NV - 1 && r < 8) st8n(corr + t0 + 8 * j, a##j, r); else st8(corr + t0 + 8 * j, a##j); }
    KERN_PUT(0) KERN_PUT(1) KERN_PUT(2) KERN_PUT(3) KERN_PUT(4) KERN_PUT(5) KERN_PUT(6) KERN_PUT(7)
#undef KERN_PUT
}

static inline void autocorr(const float *x, int win, int tmax, float *corr) {
    const int n = tmax + 1;
    int t0 = 0;
    for (; t0 + 64 <= n; t0 += 64) ac_block<8>(x, win, t0, 8, corr);
    const int rest = n - t0, nv = (rest + 7) / 8, r = rest - 8 * (nv - 1);
    switch (nv) {
        case 7: ac_block<7>(x, win, t0, r, corr); break;
        case 6: ac_block<6>(x, win, t0, r, corr); break;
        case 5: ac_block<5>(x, win, t0, r, corr); break;
        case 4: ac_block<4>(x, win, t0, r, corr); break;
        case 3: ac_block<3>(x, win, t0, r, corr); break;
        case 2: ac_block<2>(x, win, t0, r, corr); break;
        case 1: ac_block<1>(x, win, t0, r, corr); break;
        default: break;
    }
}

// PB rows x CV vectors of outputs, named accumulators
template <int PB, int CV>
static inline void taps_block(int ntaps, const float *const *in, int64_t lda, int p0, int ci, int co, const float *w,
                              const float *bias, float *out) {
#define KERN_ACC(p) f8 a##p##0 = ld8(bias), a##p##1 = CV > 1 ? ld8(bias + 8) : a##p##0, \
                       a##p##2 = CV > 2 ? ld8(bias + 16) : a##p##0, a##p##3 = CV > 3 ? ld8(bias + 24) : a##p##0;
    KERN_ACC(0) KERN_ACC(1) KERN_ACC(2) KERN_ACC(3) KERN_ACC(4) KERN_ACC(5)
#undef KERN_ACC
    for (int t = 0; t < ntaps; t++) {
        if (!in[t]) continue;
        const float *x = in[t] + p0 * lda, *wt = w + (int64_t)t * ci * co;
        for (int i = 0; i < ci; i++) {
            const float *wi = wt + i * co;
            const f8 w0 = ld8(wi), w1 = CV > 1 ? ld8(wi + 8) : w0, w2 = CV > 2 ? ld8(wi + 16) : w0, w3 = CV > 3 ? ld8(wi + 24) : w0;
#define KERN_ROW(p)                                                    \
    if (PB > p) {                                                      \
        const f8 v = bc8(x + p * lda + i);                         \
        a##p##0 = fma8(v, w0, a##p##0);                                \
        if (CV > 1) a##p##1 = fma8(v, w1, a##p##1);                    \
        if (CV > 2) a##p##2 = fma8(v, w2, a##p##2);                    \
        if (CV > 3) a##p##3 = fma8(v, w3, a##p##3);                    \
    }
            KERN_ROW(0) KERN_ROW(1) KERN_ROW(2) KERN_ROW(3) KERN_ROW(4) KERN_ROW(5)
#undef KERN_ROW
        }
    }
#define KERN_PUT(p)                                                    \
    if (PB > p) {                                                      \
        float *o = out + (p0 + p) * co;                                \
        st8(o, a##p##0);                                               \
        if (CV > 1) st8(o + 8, a##p##1);                               \
        if (CV > 2) st8(o + 16, a##p##2);                              \
        if (CV > 3) st8(o + 24, a##p##3);                              \
    }
    KERN_PUT(0) KERN_PUT(1) KERN_PUT(2) KERN_PUT(3) KERN_PUT(4) KERN_PUT(5)
#undef KERN_PUT
}

template <int CV, int PB>
static inline void taps_rows(int ntaps, const float *const *in, int64_t lda, int P, int ci, int co, const float *w,
                             const float *bias, float *out) {
    int p = 0;
    for (; p + PB <= P; p += PB) taps_block<PB, CV>(ntaps, in, lda, p, ci, co, w, bias, out);
    for (; p < P; p++) taps_block<1, CV>(ntaps, in, lda, p, ci, co, w, bias, out);
}

static inline void taps_conv(int ntaps, const float *const *in, int64_t lda, int P, int ci, int co, const float *w,
                             const float *bias, float *out) {
    for (int o = 0; o < co; o += 32) {  // up to 4 vectors of outputs per pass (weights: column offset o, row stride co)
        switch (std::min(4, (co - o) / 8)) {
            case 4: taps_rows<4, 3>(ntaps, in, lda, P, ci, co, w + o, bias + o, out + o); break;
            case 3: taps_rows<3, 4>(ntaps, in, lda, P, ci, co, w + o, bias + o, out + o); break;
            case 2: taps_rows<2, 6>(ntaps, in, lda, P, ci, co, w + o, bias + o, out + o); break;
            default: taps_rows<1, 6>(ntaps, in, lda, P, ci, co, w + o, bias + o, out + o); break;
        }
    }
}

#endif  // KERN_PASS
