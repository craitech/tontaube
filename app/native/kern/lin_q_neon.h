// Quantized linear tiles on aarch64, included by lin_q.h: the same QW / QX, drivers and contract as on x86.
//
// The int16 activation codes are split into bytes, q = 256 * hi + lo (hi = q >> 8 signed, lo = q & 255), so the dot
// products run exactly on the 8-bit instructions (LinQImpl::prep writes the bytes once per call):
//   dot (dotprod): 4 weight rows at a time; SDOT on (codes, hi); q4: UDOT on (codes, lo); q8: SDOT on (codes,
//       lo - 128) plus 128 * sum(codes)
//   mm (i8mm): 2 activation rows x 2 weight rows per 8 inputs; SMMLA on hi, USMMLA on lo
// Both give the exact int32 dot of each (output, 64-input group) and then do the same float work per output:
//   acc = fma(float(dot_g), ws_g * s_g, acc), mt = fma(wm_g, xs_g, mt) over the groups in order, y = (acc + mt) + bias
// so an output row is the same bits for any M, position in the batch, split, thread count and either variant. The
// float order differs from x86 (one chain per output there too, but over 8 lane partials): ARM and x86 agree to
// rounding, not bit for bit.
#pragma once

namespace kern {
namespace neon {

constexpr int MAXG = 256;  // groups per row (K <= 16384)

static inline float h2f(uint16_t h) {
    __fp16 v;
    std::memcpy(&v, &h, 2);
    return (float)v;
}

// aux layouts. dot: per row [GA][hi 64 | lo 64] bytes (q8: lo - 128). mm: per pair of rows [GA][8 chunks of 8 inputs]
// [hi a 8, hi b 8 | lo a 8, lo b 8] bytes (the missing row of an odd M is zeros), then per pair and group
// {s_a, s_a, s_b, s_b, xs_a, xs_a, xs_b, xs_b}.
inline size_t aux_dot(int64_t M, int64_t K) { return (size_t)(M * K * 2); }
inline size_t aux_mm(int64_t M, int64_t K) { return (size_t)(((M + 1) / 2) * (K / 64) * (256 + 32)); }

// fp16 scales / mins of the 4 rows r[0..3] ([N][G]) as float [G][4]
static inline void quad_f16(const uint16_t *a, const int64_t *r, int64_t G, float *dst) {
    int64_t g = 0;
    for (; g + 4 <= G; g += 4) {
        const float32x4_t v0 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(a + r[0] * G + g)));
        const float32x4_t v1 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(a + r[1] * G + g)));
        const float32x4_t v2 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(a + r[2] * G + g)));
        const float32x4_t v3 = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(a + r[3] * G + g)));
        const float64x2_t t0 = vreinterpretq_f64_f32(vtrn1q_f32(v0, v1)), t1 = vreinterpretq_f64_f32(vtrn2q_f32(v0, v1));
        const float64x2_t t2 = vreinterpretq_f64_f32(vtrn1q_f32(v2, v3)), t3 = vreinterpretq_f64_f32(vtrn2q_f32(v2, v3));
        vst1q_f32(dst + g * 4, vreinterpretq_f32_f64(vtrn1q_f64(t0, t2)));
        vst1q_f32(dst + g * 4 + 4, vreinterpretq_f32_f64(vtrn1q_f64(t1, t3)));
        vst1q_f32(dst + g * 4 + 8, vreinterpretq_f32_f64(vtrn2q_f64(t0, t2)));
        vst1q_f32(dst + g * 4 + 12, vreinterpretq_f32_f64(vtrn2q_f64(t1, t3)));
    }
    for (; g < G; g++)
        for (int i = 0; i < 4; i++) dst[g * 4 + i] = h2f(a[r[i] * G + g]);
}

// fp16 scales / mins of rows ra, rb as float [G][2]
static inline void pair_f16(const uint16_t *a, int64_t ra, int64_t rb, int64_t G, float *dst) {
    int64_t g = 0;
    for (; g + 4 <= G; g += 4) {
        const uint16x4_t x = vld1_u16(a + ra * G + g), y = vld1_u16(a + rb * G + g);
        vst1q_f32(dst + 2 * g, vcvt_f32_f16(vreinterpret_f16_u16(vzip1_u16(x, y))));
        vst1q_f32(dst + 2 * g + 4, vcvt_f32_f16(vreinterpret_f16_u16(vzip2_u16(x, y))));
    }
    for (; g < G; g++) { dst[2 * g] = h2f(a[ra * G + g]); dst[2 * g + 1] = h2f(a[rb * G + g]); }
}

// out lanes (rows of a quad, or [a.ra, a.rb, b.ra, b.rb] of a pair) + bias -> yr
static inline void put_quad(float32x4_t acc, float32x4_t mt, const QW &W, int64_t n, int64_t n1, float *y) {
    float o[4];
    vst1q_f32(o, vaddq_f32(acc, mt));
    for (int i = 0; i < 4 && n + i < n1; i++) y[n + i] = o[i] + (W.bias ? W.bias[n + i] : 0.f);
}
static inline void put_pair(float32x4_t acc, float32x4_t mt, const QW &W, int64_t n, int64_t n1, float *ya, float *yb) {
    float o[4];
    vst1q_f32(o, vaddq_f32(acc, mt));
    const float ba = W.bias ? W.bias[n] : 0.f, bb = W.bias && n + 1 < n1 ? W.bias[n + 1] : 0.f;
    ya[n] = o[0] + ba;
    if (n + 1 < n1) ya[n + 1] = o[1] + bb;
    if (yb) {
        yb[n] = o[2] + ba;
        if (n + 1 < n1) yb[n + 1] = o[3] + bb;
    }
}

// sum of 4 rows' lane partials -> one lane per row
static inline int32x4_t rows4(const int32x4_t *t) { return vpaddq_s32(vpaddq_s32(t[0], t[1]), vpaddq_s32(t[2], t[3])); }

// = ref::quant (vcvtnq rounds to nearest even, as nearbyint)
static void quant(const float *x, int64_t ldx, int64_t m0, int64_t m1, int64_t K, int16_t *q, float *s, float *xs) {
    const int64_t G = K / 64;
    const int32x4_t lim = vdupq_n_s32(32767), nlim = vdupq_n_s32(-32767);
    for (int64_t m = m0; m < m1; m++)
        for (int64_t g = 0; g < G; g++) {
            const float *v = x + m * ldx + g * 64;
            float32x4_t mx = vabsq_f32(vld1q_f32(v));
            for (int k = 4; k < 64; k += 4) mx = vmaxq_f32(mx, vabsq_f32(vld1q_f32(v + k)));
            const float amax = vmaxvq_f32(mx);
            const float sc = amax / 32767.f, inv = amax > 0.f ? 32767.f / amax : 0.f;
            int32x4_t sum = vdupq_n_s32(0);
            int16_t *qd = q + m * K + g * 64;
            for (int k = 0; k < 64; k += 8) {
                const int32x4_t c0 = vmaxq_s32(nlim, vminq_s32(lim, vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(v + k), inv))));
                const int32x4_t c1 = vmaxq_s32(nlim, vminq_s32(lim, vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(v + k + 4), inv))));
                sum = vaddq_s32(sum, vaddq_s32(c0, c1));
                vst1q_s16(qd + k, vcombine_s16(vmovn_s32(c0), vmovn_s32(c1)));
            }
            s[m * G + g] = sc;
            xs[m * G + g] = sc * (float)vaddvq_s32(sum);
        }
}

KERN_TARGET_BEGIN(KERN_T_DOT)

static void prep_dot(const QW &W, const QX &X, int64_t p0, int64_t p1, void *aux) {
    const int64_t K = X.K, GA = K / 64;
    const uint8x16_t flip = vdupq_n_u8(W.kind == 1 ? 0x80 : 0);
    for (int64_t m = 2 * p0; m < std::min(X.M, 2 * p1); m++)
        for (int64_t g = 0; g < GA; g++) {
            const uint8_t *q = (const uint8_t *)(X.q + m * K + g * 64);
            uint8_t *d = (uint8_t *)aux + (m * GA + g) * 128;
            for (int i = 0; i < 4; i++) {
                const uint8x16_t a = vld1q_u8(q + 32 * i), c = vld1q_u8(q + 32 * i + 16);
                vst1q_u8(d + 16 * i, vuzp2q_u8(a, c));
                vst1q_u8(d + 64 + 16 * i, veorq_u8(vuzp1q_u8(a, c), flip));
            }
        }
}

template <int MR>
static void q4_dot(const QW &W, const QX &X, int64_t m0, int64_t n0, int64_t n1, float *const *yr) {
    const int64_t G = W.K / 64;
    const uint8_t *ab = (const uint8_t *)X.aux + m0 * G * 128;
    const float *xs = X.s + m0 * G, *xx = X.xs + m0 * G;
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    float ws[4 * MAXG], wm[4 * MAXG];
    for (int64_t n = n0; n < n1; n += 4) {
        int64_t r[4];
        for (int i = 0; i < 4; i++) r[i] = std::min(n + i, n1 - 1);
        quad_f16(W.scales, r, G, ws);
        quad_f16(W.mins, r, G, wm);
        const uint8_t *c[4];
        for (int i = 0; i < 4; i++) c[i] = W.codes + r[i] * G * 32;
        float32x4_t acc[MR], mt[MR];
        for (int m = 0; m < MR; m++) acc[m] = mt[m] = vdupq_n_f32(0.f);
        for (int64_t g = 0; g < G; g++) {
            uint8x16_t w[4][4];
            for (int i = 0; i < 4; i++) {
                const uint8x16_t x0 = vld1q_u8(c[i] + g * 32), x1 = vld1q_u8(c[i] + g * 32 + 16);
                w[i][0] = vandq_u8(x0, m4);
                w[i][1] = vandq_u8(x1, m4);
                w[i][2] = vshrq_n_u8(x0, 4);
                w[i][3] = vshrq_n_u8(x1, 4);
            }
            const float32x4_t wsv = vld1q_f32(ws + g * 4), wmv = vld1q_f32(wm + g * 4);
            for (int m = 0; m < MR; m++) {
                const uint8_t *a = ab + (m * G + g) * 128;
                int8x16_t h[4];
                uint8x16_t l[4];
                for (int j = 0; j < 4; j++) { h[j] = vld1q_s8((const int8_t *)a + 16 * j); l[j] = vld1q_u8(a + 64 + 16 * j); }
                int32x4_t t[4];
                for (int i = 0; i < 4; i++) {
                    int32x4_t ah = vdupq_n_s32(0);
                    uint32x4_t al = vdupq_n_u32(0);
                    for (int j = 0; j < 4; j++) {
                        ah = vdotq_s32(ah, vreinterpretq_s8_u8(w[i][j]), h[j]);
                        al = vdotq_u32(al, w[i][j], l[j]);
                    }
                    t[i] = vaddq_s32(vshlq_n_s32(ah, 8), vreinterpretq_s32_u32(al));
                }
                acc[m] = vfmaq_f32(acc[m], vcvtq_f32_s32(rows4(t)), vmulq_n_f32(wsv, xs[m * G + g]));
                mt[m] = vfmaq_n_f32(mt[m], wmv, xx[m * G + g]);
            }
        }
        for (int m = 0; m < MR; m++) put_quad(acc[m], mt[m], W, n, n1, yr[m]);
    }
}

// q8: weight group 128 = two activation groups; lo is stored minus 128, so dot = 256 hi.w + lo'.w + 128 sum(w)
template <int MR>
static void q8_dot(const QW &W, const QX &X, int64_t m0, int64_t n0, int64_t n1, float *const *yr) {
    const int64_t G = W.K / 128, GA = W.K / 64;
    const uint8_t *ab = (const uint8_t *)X.aux + m0 * GA * 128;
    const float *xs = X.s + m0 * GA, *xx = X.xs + m0 * GA;
    const int8x16_t one = vdupq_n_s8(1);
    float ws[4 * MAXG], wm[4 * MAXG];
    for (int64_t n = n0; n < n1; n += 4) {
        int64_t r[4];
        for (int i = 0; i < 4; i++) r[i] = std::min(n + i, n1 - 1);
        quad_f16(W.scales, r, G, ws);
        quad_f16(W.mins, r, G, wm);
        const int8_t *c[4];
        for (int i = 0; i < 4; i++) c[i] = (const int8_t *)W.codes + r[i] * G * 128;
        float32x4_t acc[MR], mt[MR];
        for (int m = 0; m < MR; m++) acc[m] = mt[m] = vdupq_n_f32(0.f);
        for (int64_t g = 0; g < G; g++) {
            const float32x4_t wsv = vld1q_f32(ws + g * 4), wmv = vld1q_f32(wm + g * 4);
            for (int hh = 0; hh < 2; hh++) {
                const int64_t ga = g * 2 + hh;
                int8x16_t w[4][4];
                int32x4_t sw[4];
                for (int i = 0; i < 4; i++) {
                    sw[i] = vdupq_n_s32(0);
                    for (int j = 0; j < 4; j++) {
                        w[i][j] = vld1q_s8(c[i] + g * 128 + hh * 64 + 16 * j);
                        sw[i] = vdotq_s32(sw[i], w[i][j], one);
                    }
                }
                const int32x4_t corr = vshlq_n_s32(rows4(sw), 7);
                for (int m = 0; m < MR; m++) {
                    const int8_t *a = (const int8_t *)ab + (m * GA + ga) * 128;
                    int8x16_t h[4], l[4];
                    for (int j = 0; j < 4; j++) { h[j] = vld1q_s8(a + 16 * j); l[j] = vld1q_s8(a + 64 + 16 * j); }
                    int32x4_t t[4];
                    for (int i = 0; i < 4; i++) {
                        int32x4_t ah = vdupq_n_s32(0), al = vdupq_n_s32(0);
                        for (int j = 0; j < 4; j++) {
                            ah = vdotq_s32(ah, w[i][j], h[j]);
                            al = vdotq_s32(al, w[i][j], l[j]);
                        }
                        t[i] = vaddq_s32(vshlq_n_s32(ah, 8), al);
                    }
                    const int32x4_t d = vaddq_s32(rows4(t), corr);
                    acc[m] = vfmaq_f32(acc[m], vcvtq_f32_s32(d), vmulq_n_f32(wsv, xs[m * GA + ga]));
                    mt[m] = vfmaq_n_f32(mt[m], wmv, xx[m * GA + ga]);
                }
            }
        }
        for (int m = 0; m < MR; m++) put_quad(acc[m], mt[m], W, n, n1, yr[m]);
    }
}

// M = 1 on the packed layout: per quad of rows and group {codes of the 4 rows, 4 fp16 scales, 4 fp16 mins}, one
// sequential stream per quad (rows past N are zeros); n0 a multiple of 4
static void q4_pk(const QW &W, const QX &X, int64_t n0, int64_t n1, float *y) {
    const int64_t G = W.K / 64;
    const uint8_t *ab = (const uint8_t *)X.aux;
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    for (int64_t n = n0; n < n1; n += 4) {
        const uint8_t *p = W.packed + (n / 4) * G * 144;
        float32x4_t acc = vdupq_n_f32(0.f), mt = vdupq_n_f32(0.f);
        for (int64_t g = 0; g < G; g++, p += 144) {
            const uint8_t *a = ab + g * 128;
            int8x16_t h[4];
            uint8x16_t l[4];
            for (int j = 0; j < 4; j++) { h[j] = vld1q_s8((const int8_t *)a + 16 * j); l[j] = vld1q_u8(a + 64 + 16 * j); }
            int32x4_t t[4];
            for (int i = 0; i < 4; i++) {
                const uint8x16_t c0 = vld1q_u8(p + i * 32), c1 = vld1q_u8(p + i * 32 + 16);
                const uint8x16_t w[4] = {vandq_u8(c0, m4), vandq_u8(c1, m4), vshrq_n_u8(c0, 4), vshrq_n_u8(c1, 4)};
                int32x4_t ah = vdupq_n_s32(0);
                uint32x4_t al = vdupq_n_u32(0);
                for (int j = 0; j < 4; j++) {
                    ah = vdotq_s32(ah, vreinterpretq_s8_u8(w[j]), h[j]);
                    al = vdotq_u32(al, w[j], l[j]);
                }
                t[i] = vaddq_s32(vshlq_n_s32(ah, 8), vreinterpretq_s32_u32(al));
            }
            const float32x4_t ws = vcvt_f32_f16(vld1_f16((const float16_t *)(p + 128)));
            const float32x4_t wm = vcvt_f32_f16(vld1_f16((const float16_t *)(p + 136)));
            acc = vfmaq_f32(acc, vcvtq_f32_s32(rows4(t)), vmulq_n_f32(ws, X.s[g]));
            mt = vfmaq_n_f32(mt, wm, X.xs[g]);
        }
        put_quad(acc, mt, W, n, n1, y);
    }
}

static void tile_dot(const QW &W, const QX &X, int64_t m0, int mr, const float *, int64_t n0, int64_t n1, float *const *yr) {
    typedef void (*T)(const QW &, const QX &, int64_t, int64_t, int64_t, float *const *);
    static const T q4[8] = {q4_dot<1>, q4_dot<2>, q4_dot<3>, q4_dot<4>, q4_dot<5>, q4_dot<6>, q4_dot<7>, q4_dot<8>};
    static const T q8[8] = {q8_dot<1>, q8_dot<2>, q8_dot<3>, q8_dot<4>, q8_dot<5>, q8_dot<6>, q8_dot<7>, q8_dot<8>};
    if (X.M == 1 && W.packed && W.kind == 0) { q4_pk(W, X, n0, n1, yr[0]); return; }
    (W.kind == 0 ? q4 : q8)[mr - 1](W, X, m0, n0, n1, yr);
}

KERN_TARGET_END

KERN_TARGET_BEGIN(KERN_T_I8MM)

static void prep_pairs(const QW &, const QX &X, int64_t p0, int64_t p1, void *aux) {
    const int64_t K = X.K, GA = K / 64, P = (X.M + 1) / 2;
    float *sx = (float *)((uint8_t *)aux + P * GA * 256);
    const uint8x16_t z = vdupq_n_u8(0);
    for (int64_t p = p0; p < p1; p++) {
        const int64_t ma = 2 * p, mb = 2 * p + 1;
        const bool two = mb < X.M;
        for (int64_t g = 0; g < GA; g++) {
            const uint8_t *qa = (const uint8_t *)(X.q + ma * K + g * 64), *qb = (const uint8_t *)(X.q + (two ? mb : ma) * K + g * 64);
            uint8_t *d = (uint8_t *)aux + (p * GA + g) * 256;
            for (int c = 0; c < 8; c++) {
                const uint8x16_t a = vld1q_u8(qa + 16 * c), b = two ? vld1q_u8(qb + 16 * c) : z;
                vst1q_u8(d + 32 * c, vuzp2q_u8(a, b));
                vst1q_u8(d + 32 * c + 16, vuzp1q_u8(a, b));
            }
            float *f = sx + (p * GA + g) * 8;
            const float sa = X.s[ma * GA + g], sb = two ? X.s[mb * GA + g] : 0.f;
            const float xa = X.xs[ma * GA + g], xb = two ? X.xs[mb * GA + g] : 0.f;
            f[0] = f[1] = sa;
            f[2] = f[3] = sb;
            f[4] = f[5] = xa;
            f[6] = f[7] = xb;
        }
    }
}

static inline int8x16_t zip_lo(uint8x16_t a, uint8x16_t b) {
    return vreinterpretq_s8_u64(vzip1q_u64(vreinterpretq_u64_u8(a), vreinterpretq_u64_u8(b)));
}
static inline int8x16_t zip_hi(uint8x16_t a, uint8x16_t b) {
    return vreinterpretq_s8_u64(vzip2q_u64(vreinterpretq_u64_u8(a), vreinterpretq_u64_u8(b)));
}

// exact dots of an activation pair (aux a, one group) with a weight pair (w[8]: chunks of 8 inputs)
static inline int32x4_t mm_dot(const uint8_t *a, const int8x16_t *w) {
    int32x4_t ch = vdupq_n_s32(0), cl = vdupq_n_s32(0);
    for (int c = 0; c < 8; c++) {
        ch = vmmlaq_s32(ch, vld1q_s8((const int8_t *)a + 32 * c), w[c]);
        cl = vusmmlaq_s32(cl, vld1q_u8(a + 32 * c + 16), w[c]);
    }
    return vmlaq_n_s32(cl, ch, 256);
}

// MP activation pairs (rows m0 .. m0 + mr - 1) x weight rows in pairs
template <int MP>
static void q4_mm(const QW &W, const QX &X, int64_t m0, int mr, int64_t n0, int64_t n1, float *const *yr) {
    const int64_t G = W.K / 64, P = (X.M + 1) / 2;
    const uint8_t *ab = (const uint8_t *)X.aux + (m0 / 2) * G * 256;
    const float *sx = (const float *)((const uint8_t *)X.aux + P * G * 256) + (m0 / 2) * G * 8;
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    float ws[2 * MAXG], wm[2 * MAXG];
    for (int64_t n = n0; n < n1; n += 2) {
        const int64_t rb = std::min(n + 1, n1 - 1);
        pair_f16(W.scales, n, rb, G, ws);
        pair_f16(W.mins, n, rb, G, wm);
        const uint8_t *ca = W.codes + n * G * 32, *cb = W.codes + rb * G * 32;
        float32x4_t acc[MP], mt[MP];
        for (int p = 0; p < MP; p++) acc[p] = mt[p] = vdupq_n_f32(0.f);
        for (int64_t g = 0; g < G; g++) {
            const uint8x16_t a0 = vld1q_u8(ca + g * 32), a1 = vld1q_u8(ca + g * 32 + 16);
            const uint8x16_t b0 = vld1q_u8(cb + g * 32), b1 = vld1q_u8(cb + g * 32 + 16);
            const int8x16_t z0 = zip_lo(a0, b0), z1 = zip_hi(a0, b0), z2 = zip_lo(a1, b1), z3 = zip_hi(a1, b1);
            const int8x16_t w[8] = {vandq_s8(z0, vreinterpretq_s8_u8(m4)), vandq_s8(z1, vreinterpretq_s8_u8(m4)),
                                    vandq_s8(z2, vreinterpretq_s8_u8(m4)), vandq_s8(z3, vreinterpretq_s8_u8(m4)),
                                    vreinterpretq_s8_u8(vshrq_n_u8(vreinterpretq_u8_s8(z0), 4)),
                                    vreinterpretq_s8_u8(vshrq_n_u8(vreinterpretq_u8_s8(z1), 4)),
                                    vreinterpretq_s8_u8(vshrq_n_u8(vreinterpretq_u8_s8(z2), 4)),
                                    vreinterpretq_s8_u8(vshrq_n_u8(vreinterpretq_u8_s8(z3), 4))};
            const float32x4_t wsv = vreinterpretq_f32_f64(vld1q_dup_f64((const double *)(ws + 2 * g)));
            const float32x4_t wmv = vreinterpretq_f32_f64(vld1q_dup_f64((const double *)(wm + 2 * g)));
            for (int p = 0; p < MP; p++) {
                const int32x4_t d = mm_dot(ab + (p * G + g) * 256, w);
                const float *f = sx + (p * G + g) * 8;
                acc[p] = vfmaq_f32(acc[p], vcvtq_f32_s32(d), vmulq_f32(wsv, vld1q_f32(f)));
                mt[p] = vfmaq_f32(mt[p], wmv, vld1q_f32(f + 4));
            }
        }
        for (int p = 0; p < MP; p++) put_pair(acc[p], mt[p], W, n, n1, yr[2 * p], 2 * p + 1 < mr ? yr[2 * p + 1] : nullptr);
    }
}

template <int MP>
static void q8_mm(const QW &W, const QX &X, int64_t m0, int mr, int64_t n0, int64_t n1, float *const *yr) {
    const int64_t G = W.K / 128, GA = W.K / 64, P = (X.M + 1) / 2;
    const uint8_t *ab = (const uint8_t *)X.aux + (m0 / 2) * GA * 256;
    const float *sx = (const float *)((const uint8_t *)X.aux + P * GA * 256) + (m0 / 2) * GA * 8;
    float ws[2 * MAXG], wm[2 * MAXG];
    for (int64_t n = n0; n < n1; n += 2) {
        const int64_t rb = std::min(n + 1, n1 - 1);
        pair_f16(W.scales, n, rb, G, ws);
        pair_f16(W.mins, n, rb, G, wm);
        const uint8_t *ca = W.codes + n * G * 128, *cb = W.codes + rb * G * 128;
        float32x4_t acc[MP], mt[MP];
        for (int p = 0; p < MP; p++) acc[p] = mt[p] = vdupq_n_f32(0.f);
        for (int64_t g = 0; g < G; g++) {
            const float32x4_t wsv = vreinterpretq_f32_f64(vld1q_dup_f64((const double *)(ws + 2 * g)));
            const float32x4_t wmv = vreinterpretq_f32_f64(vld1q_dup_f64((const double *)(wm + 2 * g)));
            for (int hh = 0; hh < 2; hh++) {
                const int64_t ga = g * 2 + hh;
                int8x16_t w[8];
                for (int j = 0; j < 4; j++) {
                    const uint8x16_t x = vld1q_u8(ca + g * 128 + hh * 64 + 16 * j), y = vld1q_u8(cb + g * 128 + hh * 64 + 16 * j);
                    w[2 * j] = zip_lo(x, y);
                    w[2 * j + 1] = zip_hi(x, y);
                }
                for (int p = 0; p < MP; p++) {
                    const int32x4_t d = mm_dot(ab + (p * GA + ga) * 256, w);
                    const float *f = sx + (p * GA + ga) * 8;
                    acc[p] = vfmaq_f32(acc[p], vcvtq_f32_s32(d), vmulq_f32(wsv, vld1q_f32(f)));
                    mt[p] = vfmaq_f32(mt[p], wmv, vld1q_f32(f + 4));
                }
            }
        }
        for (int p = 0; p < MP; p++) put_pair(acc[p], mt[p], W, n, n1, yr[2 * p], 2 * p + 1 < mr ? yr[2 * p + 1] : nullptr);
    }
}

// M = 1: the dot layout (lo unsigned) and 4 weight rows per pass, q8 lo through USDOT
static void prep_mm1(const QX &X, void *aux) {
    const int64_t GA = X.K / 64;
    for (int64_t g = 0; g < GA; g++) {
        const uint8_t *q = (const uint8_t *)(X.q + g * 64);
        uint8_t *d = (uint8_t *)aux + g * 128;
        for (int i = 0; i < 4; i++) {
            const uint8x16_t a = vld1q_u8(q + 32 * i), c = vld1q_u8(q + 32 * i + 16);
            vst1q_u8(d + 16 * i, vuzp2q_u8(a, c));
            vst1q_u8(d + 64 + 16 * i, vuzp1q_u8(a, c));
        }
    }
}

static void q8_usdot(const QW &W, const QX &X, int64_t n0, int64_t n1, float *y) {
    const int64_t G = W.K / 128;
    const uint8_t *ab = (const uint8_t *)X.aux;
    float ws[4 * MAXG], wm[4 * MAXG];
    for (int64_t n = n0; n < n1; n += 4) {
        int64_t r[4];
        for (int i = 0; i < 4; i++) r[i] = std::min(n + i, n1 - 1);
        quad_f16(W.scales, r, G, ws);
        quad_f16(W.mins, r, G, wm);
        const int8_t *c[4];
        for (int i = 0; i < 4; i++) c[i] = (const int8_t *)W.codes + r[i] * G * 128;
        float32x4_t acc = vdupq_n_f32(0.f), mt = vdupq_n_f32(0.f);
        for (int64_t g = 0; g < G; g++) {
            const float32x4_t wsv = vld1q_f32(ws + g * 4), wmv = vld1q_f32(wm + g * 4);
            for (int hh = 0; hh < 2; hh++) {
                const int64_t ga = g * 2 + hh;
                const int8_t *a = (const int8_t *)ab + ga * 128;
                int8x16_t h[4];
                uint8x16_t l[4];
                for (int j = 0; j < 4; j++) { h[j] = vld1q_s8(a + 16 * j); l[j] = vld1q_u8((const uint8_t *)a + 64 + 16 * j); }
                int32x4_t t[4];
                for (int i = 0; i < 4; i++) {
                    int32x4_t ah = vdupq_n_s32(0), al = vdupq_n_s32(0);
                    for (int j = 0; j < 4; j++) {
                        const int8x16_t w = vld1q_s8(c[i] + g * 128 + hh * 64 + 16 * j);
                        ah = vdotq_s32(ah, w, h[j]);
                        al = vusdotq_s32(al, l[j], w);
                    }
                    t[i] = vaddq_s32(vshlq_n_s32(ah, 8), al);
                }
                acc = vfmaq_f32(acc, vcvtq_f32_s32(rows4(t)), vmulq_n_f32(wsv, X.s[ga]));
                mt = vfmaq_n_f32(mt, wmv, X.xs[ga]);
            }
        }
        put_quad(acc, mt, W, n, n1, y);
    }
}

// M = 1, q8 on the packed layout
static void q8_pk(const QW &W, const QX &X, int64_t n0, int64_t n1, float *y) {
    const int64_t G = W.K / 128;
    const uint8_t *ab = (const uint8_t *)X.aux;
    for (int64_t n = n0; n < n1; n += 4) {
        const uint8_t *p = W.packed + (n / 4) * G * 528;
        float32x4_t acc = vdupq_n_f32(0.f), mt = vdupq_n_f32(0.f);
        for (int64_t g = 0; g < G; g++, p += 528) {
            const float32x4_t ws = vcvt_f32_f16(vld1_f16((const float16_t *)(p + 512)));
            const float32x4_t wm = vcvt_f32_f16(vld1_f16((const float16_t *)(p + 520)));
            for (int hh = 0; hh < 2; hh++) {
                const int64_t ga = g * 2 + hh;
                const uint8_t *a = ab + ga * 128;
                int8x16_t h[4];
                uint8x16_t l[4];
                for (int j = 0; j < 4; j++) { h[j] = vld1q_s8((const int8_t *)a + 16 * j); l[j] = vld1q_u8(a + 64 + 16 * j); }
                int32x4_t t[4];
                for (int i = 0; i < 4; i++) {
                    int32x4_t ah = vdupq_n_s32(0), al = vdupq_n_s32(0);
                    for (int j = 0; j < 4; j++) {
                        const int8x16_t w = vld1q_s8((const int8_t *)p + i * 128 + hh * 64 + 16 * j);
                        ah = vdotq_s32(ah, w, h[j]);
                        al = vusdotq_s32(al, l[j], w);
                    }
                    t[i] = vaddq_s32(vshlq_n_s32(ah, 8), al);
                }
                acc = vfmaq_f32(acc, vcvtq_f32_s32(rows4(t)), vmulq_n_f32(ws, X.s[ga]));
                mt = vfmaq_n_f32(mt, wm, X.xs[ga]);
            }
        }
        put_quad(acc, mt, W, n, n1, y);
    }
}

static void prep_mm(const QW &W, const QX &X, int64_t p0, int64_t p1, void *aux) {
    if (X.M == 1) prep_mm1(X, aux);
    else prep_pairs(W, X, p0, p1, aux);
}

static void tile_mm(const QW &W, const QX &X, int64_t m0, int mr, const float *, int64_t n0, int64_t n1, float *const *yr) {
    typedef void (*T)(const QW &, const QX &, int64_t, int, int64_t, int64_t, float *const *);
    static const T q4[4] = {q4_mm<1>, q4_mm<2>, q4_mm<3>, q4_mm<4>};
    static const T q8[4] = {q8_mm<1>, q8_mm<2>, q8_mm<3>, q8_mm<4>};
    if (X.M == 1) {
        if (W.packed) (W.kind == 0 ? q4_pk : q8_pk)(W, X, n0, n1, yr[0]);
        else if (W.kind == 0) q4_dot<1>(W, X, 0, n0, n1, yr);
        else q8_usdot(W, X, n0, n1, yr[0]);
        return;
    }
    (W.kind == 0 ? q4 : q8)[(mr + 1) / 2 - 1](W, X, m0, mr, n0, n1, yr);
}

KERN_TARGET_END

}  // namespace neon

// W's rows in quads for QW::packed: per quad and group {codes of rows 4q..4q+3, their fp16 scales, their fp16 mins}
inline size_t packed_bytes(const QW &W) {
    return (size_t)((W.N + 3) / 4) * (W.K / (W.kind == 0 ? 64 : 128)) * (W.kind == 0 ? 144 : 528);
}
inline void pack_quads(const QW &W, uint8_t *dst) {
    const int64_t cb = W.kind == 0 ? 32 : 128, G = W.K / (W.kind == 0 ? 64 : 128), st = 4 * cb + 16;
    for (int64_t q = 0; q < (W.N + 3) / 4; q++)
        for (int64_t g = 0; g < G; g++) {
            uint8_t *d = dst + (q * G + g) * st;
            for (int r = 0; r < 4; r++) {
                const int64_t n = q * 4 + r;
                const uint16_t s = n < W.N ? W.scales[n * G + g] : 0, m = n < W.N ? W.mins[n * G + g] : 0;
                if (n < W.N) std::memcpy(d + r * cb, W.codes + (n * G + g) * cb, cb);
                else std::memset(d + r * cb, 0, cb);
                std::memcpy(d + 4 * cb + 2 * r, &s, 2);
                std::memcpy(d + 4 * cb + 8 + 2 * r, &m, 2);
            }
        }
}

inline const LinQImpl &impl_neon_dot() {
    static const LinQImpl r{"neon-dot", neon::quant, neon::tile_dot, neon::aux_dot, neon::prep_dot};
    return r;
}
inline const LinQImpl &impl_neon_mm() {
    static const LinQImpl r{"neon-i8mm", neon::quant, neon::tile_mm, neon::aux_mm, neon::prep_mm};
    return r;
}

}  // namespace kern
