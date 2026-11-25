// aarch64 (NEON) vector primitives for the float kernel bodies of act.h, gemm_f32.h, vv.h and detect.h: namespace
// kern::neon, the same names and per-lane arithmetic as the x86 sets in act.h (vf: 4 lanes; the 8-lane reductions as
// pairs of vectors, summed in the x86 order). Included by act.h.
#pragma once
#include "common.h"

namespace kern { namespace neon {
// 8-lane helpers (the reductions)
struct f8 { float32x4_t a, b; };
static inline f8 ld8(const float *p) { return {vld1q_f32(p), vld1q_f32(p + 4)}; }
static inline void st8(float *p, f8 v) { vst1q_f32(p, v.a); vst1q_f32(p + 4, v.b); }
static inline f8 set8(float x) { return {vdupq_n_f32(x), vdupq_n_f32(x)}; }
static inline f8 bc8(const float *p) { const float32x4_t v = vld1q_dup_f32(p); return {v, v}; }
static inline f8 fma8(f8 a, f8 b, f8 c) { return {vfmaq_f32(c.a, a.a, b.a), vfmaq_f32(c.b, a.b, b.b)}; }   // a b + c
static inline f8 fnma8(f8 a, f8 b, f8 c) { return {vfmsq_f32(c.a, a.a, b.a), vfmsq_f32(c.b, a.b, b.b)}; } // c - a b
static inline f8 mul8(f8 a, f8 b) { return {vmulq_f32(a.a, b.a), vmulq_f32(a.b, b.b)}; }
static inline float hsum8(f8 v) {  // ((v0 + v4) + (v2 + v6)) + ((v1 + v5) + (v3 + v7))
    const float32x4_t t = vaddq_f32(v.a, v.b);
    return vaddv_f32(vadd_f32(vget_low_f32(t), vget_high_f32(t)));
}
static inline f8 ld8n(const float *p, int n) {  // n < 8 lanes, rest 0
    float t[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    std::memcpy(t, p, (size_t)n * 4);
    return ld8(t);
}
static inline void st8n(float *p, f8 v, int n) {
    float t[8];
    st8(t, v);
    std::memcpy(p, t, (size_t)n * 4);
}
struct dv4 { float64x2_t a, b; };  // 4 doubles
static inline dv4 dzero() { return {vdupq_n_f64(0.0), vdupq_n_f64(0.0)}; }
static inline dv4 dset(double x) { return {vdupq_n_f64(x), vdupq_n_f64(x)}; }
static inline void dsum8(const float *p, dv4 &a, dv4 &b) {
    const f8 x = ld8(p);
    a.a = vaddq_f64(a.a, vcvt_f64_f32(vget_low_f32(x.a)));
    a.b = vaddq_f64(a.b, vcvt_high_f64_f32(x.a));
    b.a = vaddq_f64(b.a, vcvt_f64_f32(vget_low_f32(x.b)));
    b.b = vaddq_f64(b.b, vcvt_high_f64_f32(x.b));
}
static inline void dadd8(const float *p, dv4 &a, dv4 &b, dv4 m) {  // a, b += (x - m)^2 of 8 floats (doubles)
    const f8 x = ld8(p);
    const float64x2_t d0 = vsubq_f64(vcvt_f64_f32(vget_low_f32(x.a)), m.a), d1 = vsubq_f64(vcvt_high_f64_f32(x.a), m.b);
    const float64x2_t d2 = vsubq_f64(vcvt_f64_f32(vget_low_f32(x.b)), m.a), d3 = vsubq_f64(vcvt_high_f64_f32(x.b), m.b);
    a.a = vfmaq_f64(a.a, d0, d0);
    a.b = vfmaq_f64(a.b, d1, d1);
    b.a = vfmaq_f64(b.a, d2, d2);
    b.b = vfmaq_f64(b.b, d3, d3);
}
static inline double dhsum(dv4 a, dv4 b) {  // (t0 + t2) + (t1 + t3), t = a + b
    const float64x2_t u = vaddq_f64(vaddq_f64(a.a, b.a), vaddq_f64(a.b, b.b));
    return vgetq_lane_f64(u, 0) + vgetq_lane_f64(u, 1);
}

// native vectors: 4 lanes
typedef float32x4_t vf;
constexpr int VW = 4;
constexpr bool CHAINS4 = true;  // elementwise maps: four vector chains per step (latency-bound exp / polynomials)
static inline vf vld(const float *p) { return vld1q_f32(p); }
static inline void vst(float *p, vf v) { vst1q_f32(p, v); }
static inline vf vldn(const float *p, int n) { float t[4] = {0, 0, 0, 0}; std::memcpy(t, p, (size_t)n * 4); return vld1q_f32(t); }
static inline void vstn(float *p, vf v, int n) { float t[4]; vst1q_f32(t, v); std::memcpy(p, t, (size_t)n * 4); }
static inline vf vset(float x) { return vdupq_n_f32(x); }
static inline vf vzero() { return vdupq_n_f32(0.f); }
static inline vf vbc(const float *p) { return vld1q_dup_f32(p); }
static inline vf vadd(vf a, vf b) { return vaddq_f32(a, b); }
static inline vf vsub(vf a, vf b) { return vsubq_f32(a, b); }
static inline vf vmul(vf a, vf b) { return vmulq_f32(a, b); }
static inline vf vdiv(vf a, vf b) { return vdivq_f32(a, b); }
static inline vf vfma(vf a, vf b, vf c) { return vfmaq_f32(c, a, b); }   // a b + c
static inline vf vfnma(vf a, vf b, vf c) { return vfmsq_f32(c, a, b); }  // c - a b
static inline vf vmax(vf a, vf b) { return vmaxq_f32(a, b); }
static inline vf vmin(vf a, vf b) { return vminq_f32(a, b); }
static inline vf vand(vf a, vf b) { return vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b))); }
static inline vf vxor(vf a, vf b) { return vreinterpretq_f32_u32(veorq_u32(vreinterpretq_u32_f32(a), vreinterpretq_u32_f32(b))); }
static inline vf vabs(vf a) { return vabsq_f32(a); }
static inline vf vrnd(vf a) { return vrndnq_f32(a); }
static inline vf vpow2i(vf n) {  // 2^n for integral n
    return vreinterpretq_f32_s32(vshlq_n_s32(vaddq_s32(vcvtnq_s32_f32(n), vdupq_n_s32(127)), 23));
}
static inline vf vsel_ge0(vf x, vf a, vf b) { return vbslq_f32(vcgezq_f32(x), a, b); }
}}  // namespace kern::neon
