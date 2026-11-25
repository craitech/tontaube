// NEON part of gemm_f32.h's body (included in its kern::neon pass): int8 panel dequantization, epilogue on registers,
// MR (<= 5) rows x one panel (4 vectors) micro-kernel: 4 k per step, rows as vectors, FMLA by lane, named accumulators
// (gcc keeps arrays of them on the stack). Per output the same chain as on x86: bias, then fma over k.

static inline void dequant_panel(const Panels &W, int64_t p, float *buf) {
    const int64_t K = W.K;
    const int8_t *c = W.q + p * K * 16;
    const float *s = W.sc + p * (K / QG) * 16, *m = W.mn + p * (K / QG) * 16;
    for (int64_t g = 0; g < K / QG; g++) {
        float32x4_t sv[4], mv[4];
        for (int j = 0; j < 4; j++) { sv[j] = vld1q_f32(s + g * 16 + 4 * j); mv[j] = vld1q_f32(m + g * 16 + 4 * j); }
        for (int64_t k = g * QG; k < (g + 1) * QG; k++) {
            const int8x16_t v = vld1q_s8(c + k * 16);
            const int16x8_t lo = vmovl_s8(vget_low_s8(v)), hi = vmovl_high_s8(v);
            const int32x4_t q[4] = {vmovl_s16(vget_low_s16(lo)), vmovl_high_s16(lo), vmovl_s16(vget_low_s16(hi)), vmovl_high_s16(hi)};
            for (int j = 0; j < 4; j++) vst1q_f32(buf + k * 16 + 4 * j, vfmaq_f32(mv[j], vcvtq_f32_s32(q[j]), sv[j]));
        }
    }
}

// the post-store epilogue out of line: inlined, its elementwise code costs gcc registers in the k loop
static __attribute__((noinline)) void epi_post_tile(int op, float *C, int64_t ldc, int n) { epi_post(op, C, ldc, n); }

constexpr int MR_MAX = 5;
constexpr int NP_MAX = 1;
static inline vf epi_reg(vf c, int op, const float *r, const float *g) {
    if (op == EPI_ADD) return vaddq_f32(vld1q_f32(r), c);
    if (op == EPI_GADD) return vfmaq_f32(vld1q_f32(r), vld1q_f32(g), c);
    return c;
}
template <int MR>
static inline void mk(const float *const *a, const float *P, int64_t K, const float *b16, float *C, int64_t ldc, int op,
                      const float *R, int64_t ldr, const float *g16) {
    const float32x4_t z = vdupq_n_f32(0.f);
    const float32x4_t i0 = b16 ? vld1q_f32(b16) : z, i1 = b16 ? vld1q_f32(b16 + 4) : z, i2 = b16 ? vld1q_f32(b16 + 8) : z, i3 = b16 ? vld1q_f32(b16 + 12) : z;
    float32x4_t c00 = i0, c01 = i1, c02 = i2, c03 = i3, c10 = i0, c11 = i1, c12 = i2, c13 = i3, c20 = i0, c21 = i1, c22 = i2, c23 = i3;
    float32x4_t c30 = i0, c31 = i1, c32 = i2, c33 = i3, c40 = i0, c41 = i1, c42 = i2, c43 = i3;
    const float *a0 = a[0], *a1 = a[MR > 1 ? 1 : 0], *a2 = a[MR > 2 ? 2 : 0], *a3 = a[MR > 3 ? 3 : 0], *a4 = a[MR > 4 ? 4 : 0];
    const float *pk = P;
    int64_t k = 0;
#define KERN_R(m, l) if (MR > m) { c##m##0 = vfmaq_laneq_f32(c##m##0, w0, v##m, l); c##m##1 = vfmaq_laneq_f32(c##m##1, w1, v##m, l);                                    c##m##2 = vfmaq_laneq_f32(c##m##2, w2, v##m, l); c##m##3 = vfmaq_laneq_f32(c##m##3, w3, v##m, l); }
#define KERN_L(l) { const float32x4_t w0 = vld1q_f32(pk + 16 * l), w1 = vld1q_f32(pk + 16 * l + 4), w2 = vld1q_f32(pk + 16 * l + 8), w3 = vld1q_f32(pk + 16 * l + 12);                     KERN_R(0, l) KERN_R(1, l) KERN_R(2, l) KERN_R(3, l) KERN_R(4, l) }
    for (; k + 4 <= K; k += 4, pk += 64) {
        const float32x4_t v0 = vld1q_f32(a0 + k), v1 = MR > 1 ? vld1q_f32(a1 + k) : z, v2 = MR > 2 ? vld1q_f32(a2 + k) : z;
        const float32x4_t v3 = MR > 3 ? vld1q_f32(a3 + k) : z, v4 = MR > 4 ? vld1q_f32(a4 + k) : z;
        KERN_L(0) KERN_L(1) KERN_L(2) KERN_L(3)
    }
    for (; k < K; k++, pk += 16) {
        const float32x4_t v0 = vld1q_dup_f32(a0 + k), v1 = vld1q_dup_f32(a1 + k), v2 = vld1q_dup_f32(a2 + k), v3 = vld1q_dup_f32(a3 + k), v4 = vld1q_dup_f32(a4 + k);
        KERN_L(0)
    }
#undef KERN_L
#undef KERN_R
#define KERN_S(m) if (MR > m) { float *o = C + m * ldc; const float *r = R + m * ldr;         vst1q_f32(o, epi_reg(c##m##0, op, r, g16)); vst1q_f32(o + 4, epi_reg(c##m##1, op, r + 4, g16 ? g16 + 4 : g16));         vst1q_f32(o + 8, epi_reg(c##m##2, op, r + 8, g16 ? g16 + 8 : g16)); vst1q_f32(o + 12, epi_reg(c##m##3, op, r + 12, g16 ? g16 + 12 : g16)); }
    KERN_S(0) KERN_S(1) KERN_S(2) KERN_S(3) KERN_S(4)
#undef KERN_S
    if (op == EPI_GELU || op == EPI_GELU_AS || op == EPI_MISH) epi_post_tile(op, C, ldc, MR);
}
static inline void mk_any(int mr, int, const float *const *a, const float *P, int64_t K, const float *b, float *C,
                          int64_t ldc, int op, const float *R, int64_t ldr, const float *g) {
    switch (mr) {
        case 5: mk<5>(a, P, K, b, C, ldc, op, R, ldr, g); break;
        case 4: mk<4>(a, P, K, b, C, ldc, op, R, ldr, g); break;
        case 3: mk<3>(a, P, K, b, C, ldc, op, R, ldr, g); break;
        case 2: mk<2>(a, P, K, b, C, ldc, op, R, ldr, g); break;
        default: mk<1>(a, P, K, b, C, ldc, op, R, ldr, g); break;
    }
}
