// aarch64 (NEON) part of attn.h: the AVX2 tasks with the same float sequence (scores as 8-lane fma chains summed in the
// x86 order, the same exp polynomial, values one fma chain per output), so results match x86 bit for bit.
namespace neon {

template <int T>
static inline f8 kv8(const uint16_t *p) {
    const uint16x8_t h = vld1q_u16(p);
    if (T == KV_BF16)
        return {vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h), 16)), vreinterpretq_f32_u32(vshll_high_n_u16(h, 16))};
    const float16x8_t f = vreinterpretq_f16_u16(h);
    return {vcvt_f32_f16(vget_low_f16(f)), vcvt_high_f32_f16(f)};
}

struct AttnCtx {
    const float *const *q;
    const uint16_t *const *K, *const *V;
    float *const *out;
    int64_t ks, vs, rep, nkv, C;
    const int64_t *r0, *r1, *toff;
    const int32_t *trow;
    float fs;
    float *part;
};

template <int T>
static inline float score(const uint16_t *k, const float *qr, float fs) {
    f8 a = mul8(kv8<T>(k), ld8(qr));
    for (int i = 1; i < 16; i++) a = fma8(kv8<T>(k + 8 * i), ld8(qr + 8 * i), a);
    return hsum8(a) * fs;
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
            for (int64_t j = j0; j < j1; j++) s[r * C + (j - j0)] = score<T>(K + j * 128, qr, c.fs);
        }
        for (int64_t r = 0; r < rep; r++) {
            float *sr = s + r * C, mx = -INFINITY, sum = 0.f;
            for (int64_t j = 0; j < n; j++) mx = std::max(mx, sr[j]);
            int64_t j = 0;
            f8 vsum = set8(0.f);
            for (; j + 8 <= n; j += 8) {
                const f8 x = ld8(sr + j), m = set8(mx);
                const f8 e = {vexp_neg(vsubq_f32(x.a, m.a)), vexp_neg(vsubq_f32(x.b, m.b))};
                st8(sr + j, e);
                vsum = {vaddq_f32(vsum.a, e.a), vaddq_f32(vsum.b, e.b)};
            }
            sum = hsum8(vsum);
            for (; j < n; j++) { sr[j] = std::exp(sr[j] - mx); sum += sr[j]; }
            float *dst = c.part + (t * rep + r) * 130;
            dst[128] = mx;
            dst[129] = sum;
        }
        for (int pass = 0; pass < 4; pass++) {  // 32 dims per pass: the accumulators stay in registers
            float32x4_t acc[8][8];
            for (int64_t r = 0; r < rep; r++)
                for (int i = 0; i < 8; i++) acc[r][i] = vdupq_n_f32(0.f);
            for (int64_t j = j0; j < j1; j++) {
                float32x4_t vv[8];
                for (int i = 0; i < 4; i++) {
                    const f8 v = kv8<T>(V + j * 128 + pass * 32 + 8 * i);
                    vv[2 * i] = v.a;
                    vv[2 * i + 1] = v.b;
                }
                for (int64_t r = 0; r < rep; r++) {
                    const float32x4_t p = vdupq_n_f32(s[r * C + (j - j0)]);
                    for (int i = 0; i < 8; i++) acc[r][i] = vfmaq_f32(acc[r][i], p, vv[i]);
                }
            }
            for (int64_t r = 0; r < rep; r++) {
                float *dst = c.part + (t * rep + r) * 130 + pass * 32;
                for (int i = 0; i < 8; i++) vst1q_f32(dst + 4 * i, acc[r][i]);
            }
        }
    }
}

// out = sum_c e^(m_c - M) acc_c / sum_c e^(m_c - M) l_c
static void attn_merge(void *cp, int64_t h0, int64_t h1) {
    const AttnCtx &c = *(const AttnCtx *)cp;
    const int64_t rep = c.rep, nq = c.nkv * rep;
    for (int64_t ih = h0; ih < h1; ih++) {
        const int64_t qi = ih / nq, hq = ih % nq, h = hq / rep, r = hq % rep;
        const int64_t nc = (c.r1[qi] - c.r0[qi] + c.C - 1) / c.C;
        const float *pb = c.part + (c.toff[qi] + h * nc) * rep * 130;
        float M = -INFINITY, den = 0.f;
        for (int64_t k = 0; k < nc; k++) M = std::max(M, pb[(k * rep + r) * 130 + 128]);
        float32x4_t o[32];
        for (int i = 0; i < 32; i++) o[i] = vdupq_n_f32(0.f);
        for (int64_t k = 0; k < nc; k++) {
            const float *src = pb + (k * rep + r) * 130;
            const float w = std::exp(src[128] - M);
            volatile float wl = w * src[129];  // unfused, as x86
            den = den + wl;
            const float32x4_t wv = vdupq_n_f32(w);
            for (int i = 0; i < 32; i++) o[i] = vfmaq_f32(o[i], wv, vld1q_f32(src + 4 * i));
        }
        const float32x4_t inv = vdupq_n_f32(1.f / den);
        for (int i = 0; i < 32; i++) vst1q_f32(c.out[qi] + hq * 128 + 4 * i, vmulq_f32(o[i], inv));
    }
}

}  // namespace neon
