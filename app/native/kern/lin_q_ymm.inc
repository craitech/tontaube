// 256-bit quantized linear tiles, included by lin_q.h once per instruction set (AVX2, AVX-VNNI) inside its target
// region and namespace, with KERN_DOT(d, w, x) = d + dot pairs of int16 w and x (exact int32).
//
// Per output n and activation row m, the float work is the same in every variant: 8 lanes, lane j holding the
// int32 dot over inputs {16i + 2j, 16i + 2j + 1 : i < 4} of each 64-input group g, acc_j = fma(float(dot), ws_g *
// s_mg, acc_j) over g in order, then hsum(acc) + sum_g fma(wm_g, xs_mg) + bias.

// q4: rows n0..n1-1, activation rows m0..m0+MR-1 (xsT: their xs transposed, [G][8]); output of row m0 + m at yr[m].
template <int MR>
static void q4_tile(const QW &W, const QX &X, int64_t m0, const float *xsT, int64_t n0, int64_t n1, float *const *yr) {
    const int64_t K = W.K, G = K / 64;
    const __m128i m4 = _mm_set1_epi8(0x0F);
    const int16_t *xq = X.q + m0 * K;
    const float *xsc = X.s + m0 * G;
    for (int64_t n = n0; n < n1; n++) {
        __m256 acc[MR], mt = _mm256_setzero_ps();
        for (int m = 0; m < MR; m++) acc[m] = _mm256_setzero_ps();
        const uint8_t *row = W.codes + n * G * 32;
        const uint16_t *sc = W.scales + n * G, *mn = W.mins + n * G;
        for (int64_t g = 0; g < G; g++) {
            const __m128i c0 = _mm_loadu_si128((const __m128i *)(row + g * 32));
            const __m128i c1 = _mm_loadu_si128((const __m128i *)(row + g * 32 + 16));
            const __m256i w0 = _mm256_cvtepu8_epi16(_mm_and_si128(c0, m4));
            const __m256i w1 = _mm256_cvtepu8_epi16(_mm_and_si128(c1, m4));
            const __m256i w2 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(c0, 4), m4));
            const __m256i w3 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(c1, 4), m4));
            const __m256 ws = _mm256_set1_ps(h2f(sc[g]));
            mt = _mm256_fmadd_ps(_mm256_set1_ps(h2f(mn[g])), _mm256_loadu_ps(xsT + g * 8), mt);
            for (int m = 0; m < MR; m++) {
                const int16_t *x = xq + m * K + g * 64;
                __m256i d = _mm256_madd_epi16(w0, _mm256_loadu_si256((const __m256i *)x));
                d = KERN_DOT(d, w1, _mm256_loadu_si256((const __m256i *)(x + 16)));
                d = KERN_DOT(d, w2, _mm256_loadu_si256((const __m256i *)(x + 32)));
                d = KERN_DOT(d, w3, _mm256_loadu_si256((const __m256i *)(x + 48)));
                acc[m] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d), _mm256_mul_ps(ws, _mm256_broadcast_ss(xsc + m * G + g)), acc[m]);
            }
        }
        alignas(32) float mts[8];
        _mm256_store_ps(mts, mt);
        const float b = W.bias ? W.bias[n] : 0.f;
        for (int m = 0; m < MR; m++) yr[m][n] = hsum(acc[m]) + mts[m] + b;
    }
}

// q8: weight group 128 = two activation groups.
template <int MR>
static void q8_tile(const QW &W, const QX &X, int64_t m0, const float *xsT, int64_t n0, int64_t n1, float *const *yr) {
    const int64_t K = W.K, G = K / 128, GA = K / 64;
    const int16_t *xq = X.q + m0 * K;
    const float *xsc = X.s + m0 * GA;
    for (int64_t n = n0; n < n1; n++) {
        __m256 acc[MR], mt = _mm256_setzero_ps();
        for (int m = 0; m < MR; m++) acc[m] = _mm256_setzero_ps();
        const int8_t *row = (const int8_t *)W.codes + n * G * 128;
        const uint16_t *sc = W.scales + n * G, *mn = W.mins + n * G;
        for (int64_t g = 0; g < G; g++) {
            const __m256 ws = _mm256_set1_ps(h2f(sc[g])), wm = _mm256_set1_ps(h2f(mn[g]));
            for (int h = 0; h < 2; h++) {
                const int8_t *c = row + g * 128 + h * 64;
                const __m256i w0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)c));
                const __m256i w1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(c + 16)));
                const __m256i w2 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(c + 32)));
                const __m256i w3 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(c + 48)));
                const int64_t ga = g * 2 + h;
                mt = _mm256_fmadd_ps(wm, _mm256_loadu_ps(xsT + ga * 8), mt);
                for (int m = 0; m < MR; m++) {
                    const int16_t *x = xq + m * K + ga * 64;
                    __m256i d = _mm256_madd_epi16(w0, _mm256_loadu_si256((const __m256i *)x));
                    d = KERN_DOT(d, w1, _mm256_loadu_si256((const __m256i *)(x + 16)));
                    d = KERN_DOT(d, w2, _mm256_loadu_si256((const __m256i *)(x + 32)));
                    d = KERN_DOT(d, w3, _mm256_loadu_si256((const __m256i *)(x + 48)));
                    acc[m] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d), _mm256_mul_ps(ws, _mm256_broadcast_ss(xsc + m * GA + ga)), acc[m]);
                }
            }
        }
        alignas(32) float mts[8];
        _mm256_store_ps(mts, mt);
        const float b = W.bias ? W.bias[n] : 0.f;
        for (int m = 0; m < MR; m++) yr[m][n] = hsum(acc[m]) + mts[m] + b;
    }
}

static void tile(const QW &W, const QX &X, int64_t m0, int mr, const float *xsT, int64_t n0, int64_t n1, float *const *yr) {
    typedef void (*T)(const QW &, const QX &, int64_t, const float *, int64_t, int64_t, float *const *);
    static const T q4[8] = {q4_tile<1>, q4_tile<2>, q4_tile<3>, q4_tile<4>, q4_tile<5>, q4_tile<6>, q4_tile<7>, q4_tile<8>};
    static const T q8[8] = {q8_tile<1>, q8_tile<2>, q8_tile<3>, q8_tile<4>, q8_tile<5>, q8_tile<6>, q8_tile<7>, q8_tile<8>};
    (W.kind == 0 ? q4 : q8)[mr - 1](W, X, m0, xsT, n0, n1, yr);
}
