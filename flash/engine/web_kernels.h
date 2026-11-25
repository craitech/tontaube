// Kernels for the WebAssembly build of the engine (wasm SIMD128; scalar versions of the same math for the scalar
// build and native checks). Included by tt_engine.cpp after Lin / Act / quantize.
//
// q4 / q8 linears read the file layout directly (no packed copy): codes [N][G][32 | 128], fp16 scales / mins [N][G].
// Activations are the engine's int16 codes per 64 inputs; the dot products run on i32x4.dot_i16x8_s (8 multiply-adds
// per instruction, exact in int32). Rows are split over the pool in blocks of 4.
#pragma once
#include <cstdint>
#include <cstring>
#include <cmath>
#if defined(__wasm_simd128__) && !defined(WEB_SCALAR)
#include <wasm_simd128.h>
#define WEB_SIMD 1
#endif

// fp16 <-> fp32 without hardware help (exact for normal and subnormal values).
static inline float web_h2f(uint16_t h) {
    uint32_t m = (uint32_t)(h & 0x7fff) << 13, b;
    float f;
    memcpy(&f, &m, 4);
    f *= 0x1p112f;
    memcpy(&b, &f, 4);
    b |= (uint32_t)(h & 0x8000) << 16;
    if ((h & 0x7c00) == 0x7c00) b = ((uint32_t)(h & 0x8000) << 16) | 0x7f800000u | ((uint32_t)(h & 0x3ff) << 13);
    memcpy(&f, &b, 4);
    return f;
}
static inline uint16_t web_f2h(float f) {  // round to nearest even
    uint32_t x;
    memcpy(&x, &f, 4);
    const uint32_t s = (x >> 16) & 0x8000;
    x &= 0x7fffffffu;
    if (x >= 0x7f800000u) return (uint16_t)(s | 0x7c00 | (x > 0x7f800000u ? 0x200 : 0));
    if (x >= 0x477ff000u) return (uint16_t)(s | 0x7c00);  // overflow
    if (x < 0x38800000u) {                                  // subnormal half (or zero)
        float a;
        memcpy(&a, &x, 4);
        return (uint16_t)(s | (uint32_t)std::nearbyint(a * 0x1p24f));
    }
    const uint32_t r = x + 0xfff + ((x >> 13) & 1) - 0x38000000u;
    return (uint16_t)(s | (r >> 13));
}

#ifdef WEB_SIMD
// c + a * b: fused on CPUs with FMA when built with relaxed SIMD (-mrelaxed-simd), else multiply then add
#ifdef __wasm_relaxed_simd__
#define WEB_MADD(a, b, c) wasm_f32x4_relaxed_madd(a, b, c)
#else
#define WEB_MADD(a, b, c) wasm_f32x4_add(c, wasm_f32x4_mul(a, b))
#endif
static inline v128_t web_h2f4(v128_t x) {  // 4 halves in the low 16 bits of each lane
    const v128_t s = wasm_i32x4_shl(wasm_v128_and(x, wasm_i32x4_splat(0x8000)), 16);
    const v128_t m = wasm_i32x4_shl(wasm_v128_and(x, wasm_i32x4_splat(0x7fff)), 13);
    return wasm_v128_or(wasm_f32x4_mul(m, wasm_f32x4_splat(0x1p112f)), s);
}
static inline float web_hsum(v128_t v) {
    v = wasm_f32x4_add(v, wasm_i32x4_shuffle(v, v, 2, 3, 0, 1));
    v = wasm_f32x4_add(v, wasm_i32x4_shuffle(v, v, 1, 0, 3, 2));
    return wasm_f32x4_extract_lane(v, 0);
}
#endif

// n halves -> floats
static inline void web_h2f_n(const uint16_t *h, float *f, int64_t n) {
    int64_t i = 0;
#ifdef WEB_SIMD
    for (; i + 8 <= n; i += 8) {
        const v128_t x = wasm_v128_load(h + i);
        wasm_v128_store(f + i, web_h2f4(wasm_u32x4_extend_low_u16x8(x)));
        wasm_v128_store(f + i + 4, web_h2f4(wasm_u32x4_extend_high_u16x8(x)));
    }
#endif
    for (; i < n; i++) f[i] = web_h2f(h[i]);
}

// fp16 storage type of the KV cache (half the memory and traffic of float).
struct web_half {
    uint16_t v;
    web_half() = default;
    web_half(float f) : v(web_f2h(f)) {}
    operator float() const { return web_h2f(v); }
};

// a row of n floats into the cache (n a multiple of 8): float as is, fp16 rounded to nearest (subnormals flush to 0)
static inline void web_store_row(float *d, const float *x, int n) { memcpy(d, x, (size_t)n * sizeof(float)); }
static inline void web_store_row(web_half *d, const float *x, int n) {
    int i = 0;
#ifdef WEB_SIMD
    for (; i + 8 <= n; i += 8) {
        v128_t h[2];
        for (int k = 0; k < 2; k++) {
            const v128_t b = wasm_v128_load(x + i + 4 * k);
            const v128_t sgn = wasm_u32x4_shr(wasm_v128_and(b, wasm_i32x4_splat((int)0x80000000)), 16);
            const v128_t a = wasm_v128_and(b, wasm_i32x4_splat(0x7fffffff));
            v128_t r = wasm_i32x4_add(a, wasm_i32x4_splat(0xfff - 0x38000000));
            r = wasm_u32x4_shr(wasm_i32x4_add(r, wasm_v128_and(wasm_u32x4_shr(a, 13), wasm_i32x4_splat(1))), 13);
            r = wasm_v128_andnot(r, wasm_i32x4_lt(a, wasm_i32x4_splat(0x38800000)));
            r = wasm_i32x4_min(r, wasm_i32x4_splat(0x7bff));
            h[k] = wasm_v128_or(r, sgn);
        }
        wasm_v128_store(d + i, wasm_u16x8_narrow_i32x4(h[0], h[1]));
    }
#endif
    for (; i < n; i++) d[i] = web_half(x[i]);
}

#ifdef WEB_SIMD
// 8 cache values -> two f32x4
static inline void web_ld8(const float *p, v128_t &a, v128_t &b) { a = wasm_v128_load(p); b = wasm_v128_load(p + 4); }
static inline void web_ld8(const web_half *p, v128_t &a, v128_t &b) {
    const v128_t v = wasm_v128_load(p);
    a = web_h2f4(wasm_u32x4_extend_low_u16x8(v));
    b = web_h2f4(wasm_u32x4_extend_high_u16x8(v));
}
#endif

// ---------------------------------------------------------------- quantized linears
// One activation row in the engine's format: q [K] int16, s / xs [K / 64] (scale, scale x sum of codes).
struct WebAct { const int16_t *q; const float *s, *xs; };

static const int WEB_MAXG = 8192 / 64;  // largest K / 64 handled (scale buffers on the stack)
static bool g_web_pair = true;           // M >= 2: two weight rows per pass

// int8 activations for the engine's linears whose codes lie in one of these address ranges (a model file each; set
// by the engine, none by default)
struct WebRange { const uint8_t *lo, *hi; };
static WebRange g_web_act8[4];
static int g_web_n_act8 = 0;
static inline bool web_act8(const uint8_t *codes) {
    for (int i = 0; i < g_web_n_act8; i++) if (codes >= g_web_act8[i].lo && codes < g_web_act8[i].hi) return true;
    return false;
}

// Output rows per pool task: about four tasks per thread (small tasks cost more in claiming than they compute).
static int64_t web_rows(int64_t N, int threads, int M) {
    const int64_t r = N / (std::max(1, threads) * 4 * std::max(1, M / 2));
    return std::max<int64_t>(4, std::min<int64_t>(256, r / 4 * 4));
}

#ifdef WEB_SIMD
// The 64 weights of activation group h in weight group g of one row, as eight i16x8.
static inline void web_wload(int kind, const uint8_t *row, int64_t g, int64_t h, v128_t *w) {
    if (kind == 0) {
        const v128_t c0 = wasm_v128_load(row + g * 32), c1 = wasm_v128_load(row + g * 32 + 16);
        const v128_t m4 = wasm_i8x16_splat(15);
        const v128_t l0 = wasm_v128_and(c0, m4), l1 = wasm_v128_and(c1, m4);
        const v128_t h0 = wasm_u8x16_shr(c0, 4), h1 = wasm_u8x16_shr(c1, 4);
        w[0] = wasm_u16x8_extend_low_u8x16(l0); w[1] = wasm_u16x8_extend_high_u8x16(l0);
        w[2] = wasm_u16x8_extend_low_u8x16(l1); w[3] = wasm_u16x8_extend_high_u8x16(l1);
        w[4] = wasm_u16x8_extend_low_u8x16(h0); w[5] = wasm_u16x8_extend_high_u8x16(h0);
        w[6] = wasm_u16x8_extend_low_u8x16(h1); w[7] = wasm_u16x8_extend_high_u8x16(h1);
    } else {
        const uint8_t *c = row + g * 128 + h * 64;
        for (int i = 0; i < 4; i++) {
            const v128_t v = wasm_v128_load(c + 16 * i);
            w[2 * i] = wasm_i16x8_extend_low_i8x16(v);
            w[2 * i + 1] = wasm_i16x8_extend_high_i8x16(v);
        }
    }
}

// Rows n and n + 1 for M >= 2 activation rows: each activation load serves both rows.
static void web_gemm_q_pair(int kind, const uint8_t *codes, const uint16_t *scales, const uint16_t *mins, const float *bias,
                            int64_t G, const WebAct *a, int M, float *y, int64_t ys, int64_t n) {
    float wsA[WEB_MAXG], wmA[WEB_MAXG], wsB[WEB_MAXG], wmB[WEB_MAXG];
    const int64_t cb = kind == 0 ? 32 : 128, ag = kind == 0 ? 1 : 2;
    web_h2f_n(scales + n * G, wsA, G); web_h2f_n(mins + n * G, wmA, G);
    web_h2f_n(scales + (n + 1) * G, wsB, G); web_h2f_n(mins + (n + 1) * G, wmB, G);
    const uint8_t *rowA = codes + n * G * cb, *rowB = rowA + G * cb;
    v128_t accA[8], accB[8];
    for (int m = 0; m < M; m++) accA[m] = accB[m] = wasm_f32x4_splat(0.f);
    for (int64_t g = 0; g < G; g++)
        for (int64_t h = 0; h < ag; h++) {
            v128_t wA[8], wB[8];
            web_wload(kind, rowA, g, h, wA);
            web_wload(kind, rowB, g, h, wB);
            const int64_t ga = g * ag + h;
            for (int m = 0; m < M; m++) {
                const int16_t *xq = a[m].q + ga * 64;
                v128_t dA = wasm_i32x4_splat(0), dB = dA;
                for (int i = 0; i < 8; i++) {
                    const v128_t x = wasm_v128_load(xq + 8 * i);
                    dA = wasm_i32x4_add(dA, wasm_i32x4_dot_i16x8(wA[i], x));
                    dB = wasm_i32x4_add(dB, wasm_i32x4_dot_i16x8(wB[i], x));
                }
                const float s = a[m].s[ga];
                accA[m] = wasm_f32x4_add(accA[m], wasm_f32x4_mul(wasm_f32x4_convert_i32x4(dA), wasm_f32x4_splat(wsA[g] * s)));
                accB[m] = wasm_f32x4_add(accB[m], wasm_f32x4_mul(wasm_f32x4_convert_i32x4(dB), wasm_f32x4_splat(wsB[g] * s)));
            }
        }
    for (int m = 0; m < M; m++) {
        float tA = 0.f, tB = 0.f;
        for (int64_t g = 0; g < G; g++)
            for (int64_t h = 0; h < ag; h++) { tA += wmA[g] * a[m].xs[g * ag + h]; tB += wmB[g] * a[m].xs[g * ag + h]; }
        y[m * ys + n] = web_hsum(accA[m]) + tA + (bias ? bias[n] : 0.f);
        y[m * ys + n + 1] = web_hsum(accB[m]) + tB + (bias ? bias[n + 1] : 0.f);
    }
}
#endif

// y[m * ys + n] for rows n0..n1-1 and M activation rows; kind 0 = q4 (group 64), 1 = q8 (group 128).
static void web_gemm_q(int kind, const uint8_t *codes, const uint16_t *scales, const uint16_t *mins, const float *bias,
                       int64_t G, const WebAct *a, int M, float *y, int64_t ys, int64_t n0, int64_t n1) {
    float ws[WEB_MAXG], wm[WEB_MAXG];
    const int64_t cb = kind == 0 ? 32 : 128, ag = kind == 0 ? 1 : 2;  // code bytes and activation groups per group
#ifdef WEB_SIMD
    if (M >= 2 && g_web_pair)
        for (; n0 + 2 <= n1; n0 += 2) web_gemm_q_pair(kind, codes, scales, mins, bias, G, a, M, y, ys, n0);
#endif
    for (int64_t n = n0; n < n1; n++) {
        web_h2f_n(scales + n * G, ws, G);
        web_h2f_n(mins + n * G, wm, G);
        const uint8_t *row = codes + n * G * cb;
        float mt[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int m = 0; m < M; m++) {  // min terms: sum_g wm[g] * xs[m][g] (per activation group)
            float t = 0.f;
            for (int64_t g = 0; g < G; g++)
                for (int64_t h = 0; h < ag; h++) t += wm[g] * a[m].xs[g * ag + h];
            mt[m] = t;
        }
#ifdef WEB_SIMD
        v128_t acc[8];
        for (int m = 0; m < M; m++) acc[m] = wasm_f32x4_splat(0.f);
        for (int64_t g = 0; g < G; g++) {
            for (int64_t h = 0; h < ag; h++) {
                v128_t w[8];
                if (kind == 0) {
                    const v128_t c0 = wasm_v128_load(row + g * 32), c1 = wasm_v128_load(row + g * 32 + 16);
                    const v128_t m4 = wasm_i8x16_splat(15);
                    const v128_t l0 = wasm_v128_and(c0, m4), l1 = wasm_v128_and(c1, m4);
                    const v128_t h0 = wasm_u8x16_shr(c0, 4), h1 = wasm_u8x16_shr(c1, 4);
                    w[0] = wasm_u16x8_extend_low_u8x16(l0); w[1] = wasm_u16x8_extend_high_u8x16(l0);
                    w[2] = wasm_u16x8_extend_low_u8x16(l1); w[3] = wasm_u16x8_extend_high_u8x16(l1);
                    w[4] = wasm_u16x8_extend_low_u8x16(h0); w[5] = wasm_u16x8_extend_high_u8x16(h0);
                    w[6] = wasm_u16x8_extend_low_u8x16(h1); w[7] = wasm_u16x8_extend_high_u8x16(h1);
                } else {
                    const uint8_t *c = row + g * 128 + h * 64;
                    for (int i = 0; i < 4; i++) {
                        const v128_t v = wasm_v128_load(c + 16 * i);
                        w[2 * i] = wasm_i16x8_extend_low_i8x16(v);
                        w[2 * i + 1] = wasm_i16x8_extend_high_i8x16(v);
                    }
                }
                const int64_t ga = g * ag + h;
                for (int m = 0; m < M; m++) {
                    const int16_t *xq = a[m].q + ga * 64;
                    v128_t d0 = wasm_i32x4_dot_i16x8(w[0], wasm_v128_load(xq));
                    v128_t d1 = wasm_i32x4_dot_i16x8(w[1], wasm_v128_load(xq + 8));
                    d0 = wasm_i32x4_add(d0, wasm_i32x4_dot_i16x8(w[2], wasm_v128_load(xq + 16)));
                    d1 = wasm_i32x4_add(d1, wasm_i32x4_dot_i16x8(w[3], wasm_v128_load(xq + 24)));
                    d0 = wasm_i32x4_add(d0, wasm_i32x4_dot_i16x8(w[4], wasm_v128_load(xq + 32)));
                    d1 = wasm_i32x4_add(d1, wasm_i32x4_dot_i16x8(w[5], wasm_v128_load(xq + 40)));
                    d0 = wasm_i32x4_add(d0, wasm_i32x4_dot_i16x8(w[6], wasm_v128_load(xq + 48)));
                    d1 = wasm_i32x4_add(d1, wasm_i32x4_dot_i16x8(w[7], wasm_v128_load(xq + 56)));
                    const v128_t d = wasm_f32x4_convert_i32x4(wasm_i32x4_add(d0, d1));
                    acc[m] = wasm_f32x4_add(acc[m], wasm_f32x4_mul(d, wasm_f32x4_splat(ws[g] * a[m].s[ga])));
                }
            }
        }
        for (int m = 0; m < M; m++) y[m * ys + n] = web_hsum(acc[m]) + mt[m] + (bias ? bias[n] : 0.f);
#else
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int64_t g = 0; g < G; g++)
            for (int64_t h = 0; h < ag; h++) {
                const int64_t ga = g * ag + h;
                for (int m = 0; m < M; m++) {
                    const int16_t *xq = a[m].q + ga * 64;
                    int32_t d = 0;
                    if (kind == 0) {
                        const uint8_t *c = row + g * 32;
                        for (int k = 0; k < 32; k++) d += (c[k] & 15) * xq[k] + (c[k] >> 4) * xq[k + 32];
                    } else {
                        const int8_t *c = (const int8_t *)row + g * 128 + h * 64;
                        for (int k = 0; k < 64; k++) d += c[k] * xq[k];
                    }
                    acc[m] += (float)d * (ws[g] * a[m].s[ga]);
                }
            }
        for (int m = 0; m < M; m++) y[m * ys + n] = acc[m] + mt[m] + (bias ? bias[n] : 0.f);
#endif
    }
}

// The engine's activation quantization (same rounding as quantize() on x86: nearbyint, clamp to +-32767);
// qmax 127: int8 levels in the int16 codes (the int8 activations of builds without relaxed SIMD).
static void web_quantize16(const float *x, int64_t K, int16_t *q, float *s, float *xs, float qmax = 32767.f) {
    for (int64_t g = 0; g < K / 64; g++) {
        const float *v = x + g * 64;
        float amax = 0.f;
#ifdef WEB_SIMD
        v128_t mx = wasm_f32x4_splat(0.f);
        for (int k = 0; k < 64; k += 4) mx = wasm_f32x4_max(mx, wasm_f32x4_abs(wasm_v128_load(v + k)));
        mx = wasm_f32x4_max(mx, wasm_i32x4_shuffle(mx, mx, 2, 3, 0, 1));
        mx = wasm_f32x4_max(mx, wasm_i32x4_shuffle(mx, mx, 1, 0, 3, 2));
        amax = wasm_f32x4_extract_lane(mx, 0);
        const float sc = amax / qmax, inv = amax > 0.f ? qmax / amax : 0.f;
        v128_t sum = wasm_i32x4_splat(0);
        const v128_t lo = wasm_f32x4_splat(-qmax), hi = wasm_f32x4_splat(qmax), iv = wasm_f32x4_splat(inv);
        for (int k = 0; k < 64; k += 8) {
            const v128_t a0 = wasm_f32x4_nearest(wasm_f32x4_mul(wasm_v128_load(v + k), iv));
            const v128_t a1 = wasm_f32x4_nearest(wasm_f32x4_mul(wasm_v128_load(v + k + 4), iv));
            const v128_t c0 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_min(hi, wasm_f32x4_max(lo, a0)));
            const v128_t c1 = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_min(hi, wasm_f32x4_max(lo, a1)));
            sum = wasm_i32x4_add(sum, wasm_i32x4_add(c0, c1));
            wasm_v128_store(q + g * 64 + k, wasm_i16x8_narrow_i32x4(c0, c1));
        }
        sum = wasm_i32x4_add(sum, wasm_i32x4_shuffle(sum, sum, 2, 3, 0, 1));
        sum = wasm_i32x4_add(sum, wasm_i32x4_shuffle(sum, sum, 1, 0, 3, 2));
        s[g] = sc;
        xs[g] = sc * (float)wasm_i32x4_extract_lane(sum, 0);
#else
        for (int k = 0; k < 64; k++) amax = std::max(amax, std::fabs(v[k]));
        const float sc = amax / qmax, inv = amax > 0.f ? qmax / amax : 0.f;
        int64_t sum = 0;
        for (int k = 0; k < 64; k++) {
            int c = (int)std::nearbyint(v[k] * inv);
            c = std::max(-(int)qmax, std::min((int)qmax, c));
            q[g * 64 + k] = (int16_t)c;
            sum += c;
        }
        s[g] = sc;
        xs[g] = sc * (float)sum;
#endif
    }
}

// ---------------------------------------------------------------- int8 activations (relaxed SIMD dot product)
// Activations as int8 codes per 64 inputs (round to nearest, +-127), multiplied with the 4-bit codes (or the two
// nibbles of the 8-bit codes) by i32x4.relaxed_dot_i8x16_i7x16_add_s (SDOT on ARM): the weight operand stays in
// 0..127, where every implementation gives the exact sum. Builds without relaxed SIMD take the same int8 levels
// through the int16 kernels (web_quantize16 with qmax 127).
// One row: q [K], s / xs [K / 64] as WebAct, sq [K / 64] the integer sums of the codes.
struct WebAct8 { const int8_t *q; const float *s, *xs; const int32_t *sq; };

static void web_quantize8(const float *x, int64_t K, int8_t *q, float *s, float *xs, int32_t *sq) {
    for (int64_t g = 0; g < K / 64; g++) {
        const float *v = x + g * 64;
        float amax = 0.f;
        int32_t isum = 0;
#ifdef WEB_SIMD
        v128_t mx = wasm_f32x4_splat(0.f);
        for (int k = 0; k < 64; k += 4) mx = wasm_f32x4_max(mx, wasm_f32x4_abs(wasm_v128_load(v + k)));
        mx = wasm_f32x4_max(mx, wasm_i32x4_shuffle(mx, mx, 2, 3, 0, 1));
        mx = wasm_f32x4_max(mx, wasm_i32x4_shuffle(mx, mx, 1, 0, 3, 2));
        amax = wasm_f32x4_extract_lane(mx, 0);
        const float inv = amax > 0.f ? 127.f / amax : 0.f;
        v128_t sum = wasm_i32x4_splat(0);
        const v128_t lo = wasm_f32x4_splat(-127.f), hi = wasm_f32x4_splat(127.f), iv = wasm_f32x4_splat(inv);
        for (int k = 0; k < 64; k += 16) {
            v128_t c[4];
            for (int j = 0; j < 4; j++) {
                const v128_t a = wasm_f32x4_nearest(wasm_f32x4_mul(wasm_v128_load(v + k + 4 * j), iv));
                c[j] = wasm_i32x4_trunc_sat_f32x4(wasm_f32x4_min(hi, wasm_f32x4_max(lo, a)));
                sum = wasm_i32x4_add(sum, c[j]);
            }
            wasm_v128_store(q + g * 64 + k, wasm_i8x16_narrow_i16x8(wasm_i16x8_narrow_i32x4(c[0], c[1]), wasm_i16x8_narrow_i32x4(c[2], c[3])));
        }
        sum = wasm_i32x4_add(sum, wasm_i32x4_shuffle(sum, sum, 2, 3, 0, 1));
        sum = wasm_i32x4_add(sum, wasm_i32x4_shuffle(sum, sum, 1, 0, 3, 2));
        isum = wasm_i32x4_extract_lane(sum, 0);
#else
        for (int k = 0; k < 64; k++) amax = std::max(amax, std::fabs(v[k]));
        const float inv = amax > 0.f ? 127.f / amax : 0.f;
        for (int k = 0; k < 64; k++) {
            const int c = std::max(-127, std::min(127, (int)std::nearbyint(v[k] * inv)));
            q[g * 64 + k] = (int8_t)c;
            isum += c;
        }
#endif
        const float sc = amax / 127.f;
        s[g] = sc;
        sq[g] = isum;
        xs[g] = sc * (float)isum;
    }
}

#ifdef __wasm_relaxed_simd__
#define WEB_DOT8 1
// sum_i a[i] b[i] (the int8 kernels' min terms; summed in four lanes)
static inline float web_dot_f32(const float *a, const float *b, int64_t n) {
    v128_t acc = wasm_f32x4_splat(0.f);
    int64_t i = 0;
    for (; i + 4 <= n; i += 4) acc = WEB_MADD(wasm_v128_load(a + i), wasm_v128_load(b + i), acc);
    float t = web_hsum(acc);
    for (; i < n; i++) t += a[i] * b[i];
    return t;
}
// q4 rows n .. n + R - 1 (R = 1 or 2) against M int8 activation rows.
template <int R>
static inline void web_q4a8_rows(const uint8_t *codes, const uint16_t *scales, const uint16_t *mins, const float *bias, int64_t G,
                                 const WebAct8 *a, int M, float *y, int64_t ys, int64_t n) {
    float ws[R][WEB_MAXG], wm[R][WEB_MAXG];
    const uint8_t *row[R];
    for (int r = 0; r < R; r++) {
        web_h2f_n(scales + (n + r) * G, ws[r], G);
        web_h2f_n(mins + (n + r) * G, wm[r], G);
        row[r] = codes + (n + r) * G * 32;
    }
    v128_t acc[R][8];
    for (int r = 0; r < R; r++) for (int m = 0; m < M; m++) acc[r][m] = wasm_f32x4_splat(0.f);
    const v128_t m4 = wasm_i8x16_splat(15), z = wasm_i32x4_splat(0);
    for (int64_t g = 0; g < G; g++) {
        v128_t w[R][4];
        for (int r = 0; r < R; r++) {
            const v128_t c0 = wasm_v128_load(row[r] + g * 32), c1 = wasm_v128_load(row[r] + g * 32 + 16);
            w[r][0] = wasm_v128_and(c0, m4); w[r][1] = wasm_v128_and(c1, m4);
            w[r][2] = wasm_u8x16_shr(c0, 4); w[r][3] = wasm_u8x16_shr(c1, 4);
        }
        for (int m = 0; m < M; m++) {
            const int8_t *xq = a[m].q + g * 64;
            const v128_t x0 = wasm_v128_load(xq), x1 = wasm_v128_load(xq + 16), x2 = wasm_v128_load(xq + 32), x3 = wasm_v128_load(xq + 48);
            const float s = a[m].s[g];
            for (int r = 0; r < R; r++) {  // two chains of two dot products (exact integers in any order)
                const v128_t d0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(x1, w[r][1], wasm_i32x4_relaxed_dot_i8x16_i7x16_add(x0, w[r][0], z));
                const v128_t d1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(x3, w[r][3], wasm_i32x4_relaxed_dot_i8x16_i7x16_add(x2, w[r][2], z));
                const v128_t d = wasm_i32x4_add(d0, d1);
                acc[r][m] = wasm_f32x4_add(acc[r][m], wasm_f32x4_mul(wasm_f32x4_convert_i32x4(d), wasm_f32x4_splat(ws[r][g] * s)));
            }
        }
    }
    for (int r = 0; r < R; r++)
        for (int m = 0; m < M; m++) y[m * ys + n + r] = web_hsum(acc[r][m]) + web_dot_f32(wm[r], a[m].xs, G) + (bias ? bias[n + r] : 0.f);
}
// q8 row n against M int8 activation rows: code + 128 = 16 hi + lo (both 0..15), one relaxed dot per nibble; the
// 128 x sum of the activation codes comes off in integers (32 per lane).
static inline void web_q8a8_row(const uint8_t *codes, const uint16_t *scales, const uint16_t *mins, const float *bias, int64_t G,
                                const WebAct8 *a, int M, float *y, int64_t ys, int64_t n) {
    float ws[WEB_MAXG], wm[WEB_MAXG];
    web_h2f_n(scales + n * G, ws, G);
    web_h2f_n(mins + n * G, wm, G);
    const uint8_t *row = codes + n * G * 128;
    v128_t acc[8];
    for (int m = 0; m < M; m++) acc[m] = wasm_f32x4_splat(0.f);
    const v128_t m4 = wasm_i8x16_splat(15), flip = wasm_i8x16_splat((int8_t)0x80), z = wasm_i32x4_splat(0);
    for (int64_t g = 0; g < G; g++)
        for (int64_t h = 0; h < 2; h++) {
            v128_t uh[4], ul[4];
            for (int i = 0; i < 4; i++) {
                const v128_t u = wasm_v128_xor(wasm_v128_load(row + g * 128 + h * 64 + 16 * i), flip);
                uh[i] = wasm_u8x16_shr(u, 4);
                ul[i] = wasm_v128_and(u, m4);
            }
            const int64_t ga = 2 * g + h;
            for (int m = 0; m < M; m++) {
                const int8_t *xq = a[m].q + ga * 64;
                v128_t dh = z, dl = z;
                for (int i = 0; i < 4; i++) {
                    const v128_t x = wasm_v128_load(xq + 16 * i);
                    dh = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(x, uh[i], dh);
                    dl = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(x, ul[i], dl);
                }
                const v128_t d = wasm_i32x4_sub(wasm_i32x4_add(wasm_i32x4_shl(dh, 4), dl), wasm_i32x4_splat(32 * a[m].sq[ga]));
                acc[m] = wasm_f32x4_add(acc[m], wasm_f32x4_mul(wasm_f32x4_convert_i32x4(d), wasm_f32x4_splat(ws[g] * a[m].s[ga])));
            }
        }
    for (int m = 0; m < M; m++) {
        float t = 0.f;
        for (int64_t g = 0; g < G; g++) t += wm[g] * a[m].xs[2 * g] + wm[g] * a[m].xs[2 * g + 1];
        y[m * ys + n] = web_hsum(acc[m]) + t + (bias ? bias[n] : 0.f);
    }
}
// ---- exact int8 form of the int16 activations (x8): the same numbers as web_gemm_q on the 8-bit dot product
// q = 256 hi + lo (hi = q >> 8 signed, lo = q & 255); lo is stored as lo - 128 (signed) and 128 x the codes' sum comes
// back in integers. Within each block of 16 inputs the order is P = 0,1,8,9, 2,3,10,11, 4,5,12,13, 6,7,14,15 (codes
// and activations alike), so lane j of the relaxed dot product sums the same products as lane j of the int16 kernel's
// i32x4.dot_i16x8 sums: every int32 lane, and so every float operation after it, is the int16 kernel's.
static bool g_web_x8 = false;
struct WebActX { const int8_t *hi, *lo; const float *s, *xs; };
static inline v128_t web_p16(v128_t v) { return wasm_i8x16_shuffle(v, v, 0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15); }
// the int16 codes q [K] -> hi / lo [K] in the order P
static void web_split16(const int16_t *q, int64_t K, int8_t *hi, int8_t *lo) {
    const v128_t m255 = wasm_i16x8_splat(255), c128 = wasm_i16x8_splat(128);
    for (int64_t k = 0; k < K; k += 16) {
        const v128_t a = wasm_v128_load(q + k), b = wasm_v128_load(q + k + 8);
        const v128_t h = wasm_i8x16_narrow_i16x8(wasm_i16x8_shr(a, 8), wasm_i16x8_shr(b, 8));
        const v128_t l = wasm_i8x16_narrow_i16x8(wasm_i16x8_sub(wasm_v128_and(a, m255), c128), wasm_i16x8_sub(wasm_v128_and(b, m255), c128));
        wasm_v128_store(hi + k, web_p16(h));
        wasm_v128_store(lo + k, web_p16(l));
    }
}
// q4 rows n .. n + R - 1, M split activation rows; same float operations and order as web_gemm_q / web_gemm_q_pair
template <int R>
static inline void web_q4x8_rows(const uint8_t *codes, const uint16_t *scales, const uint16_t *mins, const float *bias, int64_t G,
                                 const WebActX *a, int M, float *y, int64_t ys, int64_t n) {
    float ws[R][WEB_MAXG], wm[R][WEB_MAXG];
    const uint8_t *row[R];
    for (int r = 0; r < R; r++) {
        web_h2f_n(scales + (n + r) * G, ws[r], G);
        web_h2f_n(mins + (n + r) * G, wm[r], G);
        row[r] = codes + (n + r) * G * 32;
    }
    v128_t acc[R][8];
    for (int r = 0; r < R; r++) for (int m = 0; m < M; m++) acc[r][m] = wasm_f32x4_splat(0.f);
    const v128_t m4 = wasm_i8x16_splat(15), z = wasm_i32x4_splat(0), one = wasm_i8x16_splat(1);
    for (int64_t g = 0; g < G; g++) {
        v128_t w[R][4], cs[R];
        for (int r = 0; r < R; r++) {
            const v128_t c0 = wasm_v128_load(row[r] + g * 32), c1 = wasm_v128_load(row[r] + g * 32 + 16);
            const v128_t l0 = wasm_v128_and(c0, m4), l1 = wasm_v128_and(c1, m4), h0 = wasm_u8x16_shr(c0, 4), h1 = wasm_u8x16_shr(c1, 4);
            w[r][0] = web_p16(l0); w[r][1] = web_p16(l1); w[r][2] = web_p16(h0); w[r][3] = web_p16(h1);
            // 128 x the codes' sum per lane (the low bytes' offset)
            const v128_t s0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(one, w[r][1], wasm_i32x4_relaxed_dot_i8x16_i7x16_add(one, w[r][0], z));
            const v128_t s1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(one, w[r][3], wasm_i32x4_relaxed_dot_i8x16_i7x16_add(one, w[r][2], z));
            cs[r] = wasm_i32x4_shl(wasm_i32x4_add(s0, s1), 7);
        }
        for (int m = 0; m < M; m++) {
            const int8_t *xh = a[m].hi + g * 64, *xl = a[m].lo + g * 64;
            const v128_t h0 = wasm_v128_load(xh), h1 = wasm_v128_load(xh + 16), h2 = wasm_v128_load(xh + 32), h3 = wasm_v128_load(xh + 48);
            const v128_t q0 = wasm_v128_load(xl), q1 = wasm_v128_load(xl + 16), q2 = wasm_v128_load(xl + 32), q3 = wasm_v128_load(xl + 48);
            const float s = a[m].s[g];
            for (int r = 0; r < R; r++) {
                const v128_t dh = wasm_i32x4_add(
                    wasm_i32x4_relaxed_dot_i8x16_i7x16_add(h1, w[r][1], wasm_i32x4_relaxed_dot_i8x16_i7x16_add(h0, w[r][0], z)),
                    wasm_i32x4_relaxed_dot_i8x16_i7x16_add(h3, w[r][3], wasm_i32x4_relaxed_dot_i8x16_i7x16_add(h2, w[r][2], z)));
                const v128_t dl = wasm_i32x4_add(
                    wasm_i32x4_relaxed_dot_i8x16_i7x16_add(q1, w[r][1], wasm_i32x4_relaxed_dot_i8x16_i7x16_add(q0, w[r][0], z)),
                    wasm_i32x4_relaxed_dot_i8x16_i7x16_add(q3, w[r][3], wasm_i32x4_relaxed_dot_i8x16_i7x16_add(q2, w[r][2], z)));
                const v128_t d = wasm_f32x4_convert_i32x4(wasm_i32x4_add(wasm_i32x4_add(wasm_i32x4_shl(dh, 8), dl), cs[r]));
                acc[r][m] = wasm_f32x4_add(acc[r][m], wasm_f32x4_mul(d, wasm_f32x4_splat(ws[r][g] * s)));
            }
        }
    }
    for (int r = 0; r < R; r++)
        for (int m = 0; m < M; m++) {
#pragma clang fp contract(off)
            float t = 0.f;  // (no fused multiply-add here: the int16 kernel's min terms are plain)
            for (int64_t g = 0; g < G; g++) t += wm[r][g] * a[m].xs[g];
            y[m * ys + n + r] = web_hsum(acc[r][m]) + t + (bias ? bias[n + r] : 0.f);
        }
}
// q8 (signed codes as the 8-bit operand): q + 32768 = 16384 u2 + 128 u1 + u0 (u0, u1 in 0..127, u2 in 0..3), each
// piece in the order P; the code sums per lane take the 32768 off.
static void web_split16x3(const int16_t *q, int64_t K, int8_t *u0, int8_t *u1, int8_t *u2) {
    const v128_t f = wasm_i16x8_splat((int16_t)0x8000), m7 = wasm_i16x8_splat(127);
    for (int64_t k = 0; k < K; k += 16) {
        const v128_t a = wasm_v128_xor(wasm_v128_load(q + k), f), b = wasm_v128_xor(wasm_v128_load(q + k + 8), f);
        wasm_v128_store(u0 + k, web_p16(wasm_i8x16_narrow_i16x8(wasm_v128_and(a, m7), wasm_v128_and(b, m7))));
        wasm_v128_store(u1 + k, web_p16(wasm_i8x16_narrow_i16x8(wasm_v128_and(wasm_u16x8_shr(a, 7), m7), wasm_v128_and(wasm_u16x8_shr(b, 7), m7))));
        wasm_v128_store(u2 + k, web_p16(wasm_i8x16_narrow_i16x8(wasm_u16x8_shr(a, 14), wasm_u16x8_shr(b, 14))));
    }
}
struct WebActX3 { const int8_t *u0, *u1, *u2; const float *s, *xs; };
// one q8 row n, M activation rows; the float operations of web_gemm_q (single-row form; pairs for M >= 2 there use
// the same per-row operations)
static void web_q8x8_row(const uint8_t *codes, const uint16_t *scales, const uint16_t *mins, const float *bias, int64_t G,
                         const WebActX3 *a, int M, float *y, int64_t ys, int64_t n) {
    float ws[WEB_MAXG], wm[WEB_MAXG];
    web_h2f_n(scales + n * G, ws, G);
    web_h2f_n(mins + n * G, wm, G);
    const uint8_t *row = codes + n * G * 128;
    float mt[8];
    for (int m = 0; m < M; m++) {
#pragma clang fp contract(off)
        float t = 0.f;
        for (int64_t g = 0; g < G; g++)
            for (int64_t h = 0; h < 2; h++) t += wm[g] * a[m].xs[g * 2 + h];
        mt[m] = t;
    }
    v128_t acc[8];
    for (int m = 0; m < M; m++) acc[m] = wasm_f32x4_splat(0.f);
    const v128_t z = wasm_i32x4_splat(0), one = wasm_i8x16_splat(1);
    for (int64_t g = 0; g < G; g++)
        for (int64_t h = 0; h < 2; h++) {
            v128_t c[4];
            for (int i = 0; i < 4; i++) c[i] = web_p16(wasm_v128_load(row + g * 128 + h * 64 + 16 * i));
            const v128_t cs = wasm_i32x4_shl(wasm_i32x4_add(
                wasm_i32x4_relaxed_dot_i8x16_i7x16_add(c[1], one, wasm_i32x4_relaxed_dot_i8x16_i7x16_add(c[0], one, z)),
                wasm_i32x4_relaxed_dot_i8x16_i7x16_add(c[3], one, wasm_i32x4_relaxed_dot_i8x16_i7x16_add(c[2], one, z))), 15);
            const int64_t ga = g * 2 + h;
            for (int m = 0; m < M; m++) {
                const int8_t *p0 = a[m].u0 + ga * 64, *p1 = a[m].u1 + ga * 64, *p2 = a[m].u2 + ga * 64;
                v128_t d0 = z, d1 = z, d2 = z;
                for (int i = 0; i < 4; i++) {
                    d0 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(c[i], wasm_v128_load(p0 + 16 * i), d0);
                    d1 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(c[i], wasm_v128_load(p1 + 16 * i), d1);
                    d2 = wasm_i32x4_relaxed_dot_i8x16_i7x16_add(c[i], wasm_v128_load(p2 + 16 * i), d2);
                }
                const v128_t d = wasm_i32x4_sub(wasm_i32x4_add(wasm_i32x4_add(wasm_i32x4_shl(d2, 14), wasm_i32x4_shl(d1, 7)), d0), cs);
                acc[m] = wasm_f32x4_add(acc[m], wasm_f32x4_mul(wasm_f32x4_convert_i32x4(d), wasm_f32x4_splat(ws[g] * a[m].s[ga])));
            }
        }
    for (int m = 0; m < M; m++) y[m * ys + n] = web_hsum(acc[m]) + mt[m] + (bias ? bias[n] : 0.f);
}
static void web_gemm_x8q8(const uint8_t *codes, const uint16_t *scales, const uint16_t *mins, const float *bias, int64_t G,
                          const WebActX3 *a, int M, float *y, int64_t ys, int64_t n0, int64_t n1) {
    for (int64_t n = n0; n < n1; n++) web_q8x8_row(codes, scales, mins, bias, G, a, M, y, ys, n);
}

// y[m * ys + n] for q4 rows n0..n1-1 (as web_gemm_q: pairs of rows for M >= 2, single rows else)
static void web_gemm_x8(const uint8_t *codes, const uint16_t *scales, const uint16_t *mins, const float *bias, int64_t G,
                        const WebActX *a, int M, float *y, int64_t ys, int64_t n0, int64_t n1) {
    int64_t n = n0;
    if (M >= 2 && g_web_pair)
        for (; n + 2 <= n1; n += 2) web_q4x8_rows<2>(codes, scales, mins, bias, G, a, M, y, ys, n);
    for (; n < n1; n++) web_q4x8_rows<1>(codes, scales, mins, bias, G, a, M, y, ys, n);
}

// y[m * ys + n] for rows n0..n1-1 and M <= 8 int8 activation rows; kind 0 = q4 (group 64), 1 = q8 (group 128).
static void web_gemm_a8(int kind, const uint8_t *codes, const uint16_t *scales, const uint16_t *mins, const float *bias,
                        int64_t G, const WebAct8 *a, int M, float *y, int64_t ys, int64_t n0, int64_t n1) {
    if (kind == 0) {
        int64_t n = n0;
        for (; n + 2 <= n1; n += 2) web_q4a8_rows<2>(codes, scales, mins, bias, G, a, M, y, ys, n);
        if (n < n1) web_q4a8_rows<1>(codes, scales, mins, bias, G, a, M, y, ys, n);
    } else {
        for (int64_t n = n0; n < n1; n++) web_q8a8_row(codes, scales, mins, bias, G, a, M, y, ys, n);
    }
}
#endif

// ---------------------------------------------------------------- float linears
// y[m * ys + n] = b[n] + sum_k W[n][k] x[m * xs + k], rows m0..m1-1, outputs n0..n1-1 (K a multiple of 4).
static void web_gemm_f32(const float *W, const float *b, int64_t K, const float *x, int64_t xs, float *y, int64_t ys,
                         int64_t m0, int64_t m1, int64_t n0, int64_t n1) {
#ifdef WEB_SIMD
    int64_t m = m0;
    for (; m + 4 <= m1; m += 4) {
        const float *x0 = x + m * xs, *x1 = x0 + xs, *x2 = x1 + xs, *x3 = x2 + xs;
        int64_t n = n0;
        for (; n + 2 <= n1; n += 2) {
            const float *w0 = W + n * K, *w1 = w0 + K;
            v128_t a00 = wasm_f32x4_splat(0.f), a01 = a00, a10 = a00, a11 = a00, a20 = a00, a21 = a00, a30 = a00, a31 = a00;
            for (int64_t k = 0; k < K; k += 4) {
                const v128_t v0 = wasm_v128_load(w0 + k), v1 = wasm_v128_load(w1 + k);
                const v128_t u0 = wasm_v128_load(x0 + k), u1 = wasm_v128_load(x1 + k);
                const v128_t u2 = wasm_v128_load(x2 + k), u3 = wasm_v128_load(x3 + k);
                a00 = WEB_MADD(u0, v0, a00); a01 = WEB_MADD(u0, v1, a01);
                a10 = WEB_MADD(u1, v0, a10); a11 = WEB_MADD(u1, v1, a11);
                a20 = WEB_MADD(u2, v0, a20); a21 = WEB_MADD(u2, v1, a21);
                a30 = WEB_MADD(u3, v0, a30); a31 = WEB_MADD(u3, v1, a31);
            }
            const float b0 = b ? b[n] : 0.f, b1 = b ? b[n + 1] : 0.f;
            y[m * ys + n] = web_hsum(a00) + b0; y[m * ys + n + 1] = web_hsum(a01) + b1;
            y[(m + 1) * ys + n] = web_hsum(a10) + b0; y[(m + 1) * ys + n + 1] = web_hsum(a11) + b1;
            y[(m + 2) * ys + n] = web_hsum(a20) + b0; y[(m + 2) * ys + n + 1] = web_hsum(a21) + b1;
            y[(m + 3) * ys + n] = web_hsum(a30) + b0; y[(m + 3) * ys + n + 1] = web_hsum(a31) + b1;
        }
        for (; n < n1; n++) {
            const float *w0 = W + n * K;
            v128_t a0 = wasm_f32x4_splat(0.f), a1 = a0, a2 = a0, a3 = a0;
            for (int64_t k = 0; k < K; k += 4) {
                const v128_t v0 = wasm_v128_load(w0 + k);
                a0 = WEB_MADD(wasm_v128_load(x0 + k), v0, a0);
                a1 = WEB_MADD(wasm_v128_load(x1 + k), v0, a1);
                a2 = WEB_MADD(wasm_v128_load(x2 + k), v0, a2);
                a3 = WEB_MADD(wasm_v128_load(x3 + k), v0, a3);
            }
            const float b0 = b ? b[n] : 0.f;
            y[m * ys + n] = web_hsum(a0) + b0; y[(m + 1) * ys + n] = web_hsum(a1) + b0;
            y[(m + 2) * ys + n] = web_hsum(a2) + b0; y[(m + 3) * ys + n] = web_hsum(a3) + b0;
        }
    }
    for (; m < m1; m++) {
        const float *x0 = x + m * xs;
        for (int64_t n = n0; n < n1; n++) {
            const float *w0 = W + n * K;
            v128_t a0 = wasm_f32x4_splat(0.f);
            for (int64_t k = 0; k < K; k += 4) a0 = WEB_MADD(wasm_v128_load(x0 + k), wasm_v128_load(w0 + k), a0);
            y[m * ys + n] = web_hsum(a0) + (b ? b[n] : 0.f);
        }
    }
#else
    for (int64_t m = m0; m < m1; m++)
        for (int64_t n = n0; n < n1; n++) {
            const float *w0 = W + n * K, *x0 = x + m * xs;
            float a[4] = {0, 0, 0, 0};
            for (int64_t k = 0; k < K; k += 4)
                for (int j = 0; j < 4; j++) a[j] += x0[k + j] * w0[k + j];
            y[m * ys + n] = (a[0] + a[2]) + (a[1] + a[3]) + (b ? b[n] : 0.f);
        }
#endif
}

#ifdef WEB_SIMD
// The same outputs as web_gemm_f32 (one f32x4 per output over k in steps of 4, then web_hsum), in register blocks of
// 4 rows x 4 outputs and blocks of 64 outputs: for CPUs with 32 vector registers (ARM); x86 has 16 and spills.
static bool g_web_f44 = false;
static void web_gemm_f32_44(const float *W, const float *b, int64_t K, const float *x, int64_t xs, float *y, int64_t ys,
                            int64_t m0, int64_t m1, int64_t n0, int64_t n1) {
    for (int64_t nb = n0; nb < n1; nb += 64) {
        const int64_t ne = std::min(n1, nb + 64);
        int64_t m = m0;
        for (; m + 4 <= m1; m += 4) {
            int64_t n = nb;
            for (; n + 4 <= ne; n += 4) {
                v128_t acc[4][4];
                for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) acc[r][c] = wasm_f32x4_splat(0.f);
                for (int64_t k = 0; k < K; k += 4) {
                    v128_t w[4];
                    for (int c = 0; c < 4; c++) w[c] = wasm_v128_load(W + (n + c) * K + k);
                    for (int r = 0; r < 4; r++) {
                        const v128_t u = wasm_v128_load(x + (m + r) * xs + k);
                        for (int c = 0; c < 4; c++) acc[r][c] = WEB_MADD(u, w[c], acc[r][c]);
                    }
                }
                for (int r = 0; r < 4; r++)
                    for (int c = 0; c < 4; c++) y[(m + r) * ys + n + c] = web_hsum(acc[r][c]) + (b ? b[n + c] : 0.f);
            }
            if (n < ne) web_gemm_f32(W, b, K, x, xs, y, ys, m, m + 4, n, ne);
        }
        if (m < m1) web_gemm_f32(W, b, K, x, xs, y, ys, m, m1, nb, ne);
    }
}
#endif

// ---------------------------------------------------------------- elementwise
// exp(x) for x <= 0 (Cephes polynomial, as the engine's NEON version); below -87 flushes towards 0.
#ifdef WEB_SIMD
static inline v128_t web_exp_neg(v128_t x) {
    x = wasm_f32x4_max(x, wasm_f32x4_splat(-87.f));
    const v128_t fx = wasm_f32x4_nearest(wasm_f32x4_mul(x, wasm_f32x4_splat(1.44269504088896341f)));
    x = wasm_f32x4_sub(x, wasm_f32x4_mul(fx, wasm_f32x4_splat(0.693359375f)));
    x = wasm_f32x4_sub(x, wasm_f32x4_mul(fx, wasm_f32x4_splat(-2.12194440e-4f)));
    v128_t y = wasm_f32x4_splat(1.9875691500e-4f);
    y = wasm_f32x4_add(wasm_f32x4_mul(y, x), wasm_f32x4_splat(1.3981999507e-3f));
    y = wasm_f32x4_add(wasm_f32x4_mul(y, x), wasm_f32x4_splat(8.3334519073e-3f));
    y = wasm_f32x4_add(wasm_f32x4_mul(y, x), wasm_f32x4_splat(4.1665795894e-2f));
    y = wasm_f32x4_add(wasm_f32x4_mul(y, x), wasm_f32x4_splat(1.6666665459e-1f));
    y = wasm_f32x4_add(wasm_f32x4_mul(y, x), wasm_f32x4_splat(5.0000001201e-1f));
    y = wasm_f32x4_add(wasm_f32x4_mul(y, wasm_f32x4_mul(x, x)), wasm_f32x4_add(x, wasm_f32x4_splat(1.f)));
    const v128_t e = wasm_i32x4_shl(wasm_i32x4_add(wasm_i32x4_trunc_sat_f32x4(fx), wasm_i32x4_splat(127)), 23);
    return wasm_f32x4_mul(y, e);
}
#endif

// Exact GELU, x * Phi(x), with erf from W. J. Cody's rational approximations folded into one form:
// erf(z) = 1 - t * exp(-z^2 + P(t)) (Numerical Recipes erfc, |error| < 1.2e-7).
static inline float web_gelu1(float x) {
    const float z = std::fabs(x) * 0.70710678118654752f, t = 1.f / (1.f + 0.5f * z);
    const float p = -1.26551223f + t * (1.00002368f + t * (0.37409196f + t * (0.09678418f + t * (-0.18628806f +
                    t * (0.27886807f + t * (-1.13520398f + t * (1.48851587f + t * (-0.82215223f + t * 0.17087277f))))))));
    const float erfc = t * std::exp(-z * z + p);
    return x >= 0.f ? x * (1.f - 0.5f * erfc) : x * 0.5f * erfc;
}
static void web_gelu(float *x, int64_t n) {
    int64_t i = 0;
#ifdef WEB_SIMD
    const v128_t one = wasm_f32x4_splat(1.f), half = wasm_f32x4_splat(0.5f);
    for (; i + 4 <= n; i += 4) {
        const v128_t v = wasm_v128_load(x + i);
        const v128_t z = wasm_f32x4_mul(wasm_f32x4_abs(v), wasm_f32x4_splat(0.70710678118654752f));
        const v128_t t = wasm_f32x4_div(one, wasm_f32x4_add(one, wasm_f32x4_mul(half, z)));
        v128_t p = wasm_f32x4_splat(0.17087277f);
        const float c[9] = {-0.82215223f, 1.48851587f, -1.13520398f, 0.27886807f, -0.18628806f, 0.09678418f, 0.37409196f, 1.00002368f, -1.26551223f};
        for (int j = 0; j < 9; j++) p = wasm_f32x4_add(wasm_f32x4_mul(p, t), wasm_f32x4_splat(c[j]));
        const v128_t erfc = wasm_f32x4_mul(t, web_exp_neg(wasm_f32x4_sub(p, wasm_f32x4_mul(z, z))));
        const v128_t pos = wasm_f32x4_mul(v, wasm_f32x4_sub(one, wasm_f32x4_mul(half, erfc)));
        const v128_t neg = wasm_f32x4_mul(v, wasm_f32x4_mul(half, erfc));
        wasm_v128_store(x + i, wasm_v128_bitselect(pos, neg, wasm_f32x4_ge(v, wasm_f32x4_splat(0.f))));
    }
#endif
    for (; i < n; i++) x[i] = web_gelu1(x[i]);
}

// out = x / sqrt(mean(x^2) + eps) (no weight)
static inline void web_rms(const float *x, int64_t n, float eps, float *out) {
    float ss = 0.f;
    int64_t i = 0;
#ifdef WEB_SIMD
    v128_t a0 = wasm_f32x4_splat(0.f), a1 = a0;
    for (; i + 8 <= n; i += 8) {
        const v128_t v0 = wasm_v128_load(x + i), v1 = wasm_v128_load(x + i + 4);
        a0 = wasm_f32x4_add(a0, wasm_f32x4_mul(v0, v0));
        a1 = wasm_f32x4_add(a1, wasm_f32x4_mul(v1, v1));
    }
    ss = web_hsum(wasm_f32x4_add(a0, a1));
#endif
    for (; i < n; i++) ss += x[i] * x[i];
    const float r = 1.f / std::sqrt(ss / (float)n + eps);
    i = 0;
#ifdef WEB_SIMD
    for (; i + 4 <= n; i += 4) wasm_v128_store(out + i, wasm_f32x4_mul(wasm_v128_load(x + i), wasm_f32x4_splat(r)));
#endif
    for (; i < n; i++) out[i] = x[i] * r;
}

// ---------------------------------------------------------------- attention
// Rows j0..j1-1 of one KV head for its rep (<= 4) query heads q [rep][128]: s[r][j] = q[r] . K[j] * fs (s rows ss
// apart); e = exp(s - max); o[r] = sum_j e[r][j] V[j] (unnormalized), mx[r], sum[r]. Each K / V row is read once.
template <int rep, typename KV>
static void web_attend_block(const float *q, const KV *K, const KV *V, int64_t j0, int64_t j1, float fs,
                             float *s, int64_t ss, float *o, float *mx, float *sum) {
    for (int r = 0; r < rep; r++) mx[r] = -INFINITY;
#ifdef WEB_SIMD
    for (int64_t j = j0; j < j1; j++) {
        const KV *k = K + j * 128;
        v128_t a[rep][2];
        for (int r = 0; r < rep; r++) a[r][0] = a[r][1] = wasm_f32x4_splat(0.f);
        for (int i = 0; i < 32; i += 2) {
            v128_t k0, k1;
            web_ld8(k + 4 * i, k0, k1);
            for (int r = 0; r < rep; r++) {
                a[r][0] = WEB_MADD(k0, wasm_v128_load(q + r * 128 + 4 * i), a[r][0]);
                a[r][1] = WEB_MADD(k1, wasm_v128_load(q + r * 128 + 4 * i + 4), a[r][1]);
            }
        }
        for (int r = 0; r < rep; r++) {
            s[r * ss + j] = web_hsum(wasm_f32x4_add(a[r][0], a[r][1])) * fs;
            mx[r] = std::max(mx[r], s[r * ss + j]);
        }
    }
    for (int r = 0; r < rep; r++) {
        float *sr = s + r * ss;
        v128_t vs = wasm_f32x4_splat(0.f);
        int64_t j = j0;
        for (; j + 4 <= j1; j += 4) {
            const v128_t e = web_exp_neg(wasm_f32x4_sub(wasm_v128_load(sr + j), wasm_f32x4_splat(mx[r])));
            wasm_v128_store(sr + j, e);
            vs = wasm_f32x4_add(vs, e);
        }
        sum[r] = web_hsum(vs);
        for (; j < j1; j++) { sr[j] = std::exp(sr[j] - mx[r]); sum[r] += sr[j]; }
    }
    for (int part = 0; part < 4; part++) {  // 32 dims per pass, all query heads: each V row read once
        v128_t acc[rep][8];
        for (int r = 0; r < rep; r++) for (int i = 0; i < 8; i++) acc[r][i] = wasm_f32x4_splat(0.f);
        for (int64_t jj = j0; jj < j1; jj++) {
            const KV *v = V + jj * 128 + part * 32;
            v128_t vv[8];
            for (int i = 0; i < 8; i += 2) web_ld8(v + 4 * i, vv[i], vv[i + 1]);
            for (int r = 0; r < rep; r++) {
                const v128_t pw = wasm_f32x4_splat(s[r * ss + jj]);
                for (int i = 0; i < 8; i++) acc[r][i] = WEB_MADD(pw, vv[i], acc[r][i]);
            }
        }
        for (int r = 0; r < rep; r++)
            for (int i = 0; i < 8; i++) wasm_v128_store(o + r * 130 + part * 32 + 4 * i, acc[r][i]);
    }
#else
    for (int64_t j = j0; j < j1; j++) {
        const KV *k = K + j * 128;
        for (int r = 0; r < rep; r++) {
            float a = 0.f;
            for (int i = 0; i < 128; i++) a += (float)k[i] * q[r * 128 + i];
            s[r * ss + j] = a * fs;
            mx[r] = std::max(mx[r], s[r * ss + j]);
        }
    }
    for (int r = 0; r < rep; r++) {
        sum[r] = 0.f;
        for (int64_t j = j0; j < j1; j++) { s[r * ss + j] = std::exp(s[r * ss + j] - mx[r]); sum[r] += s[r * ss + j]; }
        for (int i = 0; i < 128; i++) o[r * 130 + i] = 0.f;
        for (int64_t j = j0; j < j1; j++)
            for (int i = 0; i < 128; i++) o[r * 130 + i] += s[r * ss + j] * (float)V[j * 128 + i];
    }
#endif
}

// ---------------------------------------------------------------- weight files from memory
// The page puts each model file into the heap once; File::open(path) then uses that buffer instead of a mapping.
#ifdef __EMSCRIPTEN__
#include <map>
#include <string>
struct WebBuf { const uint8_t *p; size_t n; };
static std::map<std::string, WebBuf> &web_files() { static std::map<std::string, WebBuf> m; return m; }
static bool web_file_lookup(const char *path, const uint8_t **base, size_t *size) {
    auto it = web_files().find(path);
    if (it == web_files().end()) return false;
    *base = it->second.p;
    *size = it->second.n;
    return true;
}
#endif
