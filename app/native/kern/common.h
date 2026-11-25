// Shared pieces of the x86 kernels: the parallel-for interface and small AVX2 helpers (torch-free).
#pragma once
#include "isa.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace kern {

// fn(ctx, begin, end) over a range of work items.
typedef void (*RangeFn)(void *ctx, int64_t begin, int64_t end);

// A parallel for: run(self, n, grain, fn, ctx) calls fn on disjoint ranges covering [0, n), at least `grain` items
// each where possible, and returns when all are done. Any split must give the same result (kernels split outputs,
// never a sum). run == nullptr: serial. Adapters: app/native/common.h (at::parallel_for), Flash tt_engine.cpp (Pool).
struct Par {
    void (*run)(void *self, int64_t n, int64_t grain, RangeFn fn, void *ctx) = nullptr;
    void *self = nullptr;
    int threads = 1;
    void operator()(int64_t n, int64_t grain, RangeFn fn, void *ctx) const {
        if (n <= 0) return;
        if (run && threads > 1 && n > grain) run(self, n, grain, fn, ctx);
        else fn(ctx, 0, n);
    }
};

// Software fp16 -> fp32 (exact for every half), for scalar code that must run without F16C.
inline float h2f_sw(uint16_t h) {
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
inline float bf2f(uint16_t h) {
    const uint32_t b = (uint32_t)h << 16;
    float f;
    memcpy(&f, &b, 4);
    return f;
}

#ifdef KERN_X86
KERN_TARGET_BEGIN(KERN_T_AVX2)

inline float hsum(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}

inline float h2f(uint16_t h) { return _cvtsh_ss(h); }
inline uint16_t f2h(float f) { return (uint16_t)_cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT); }

// 8 bf16 / fp16 values -> fp32
inline __m256 bf16x8(const uint16_t *p) {
    return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)p)), 16));
}
inline __m256 f16x8(const uint16_t *p) { return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)p)); }

// n floats -> fp16, round to nearest even (n a multiple of 8)
inline void store_f16(uint16_t *d, const float *x, int64_t n) {
    for (int64_t i = 0; i < n; i += 8)
        _mm_storeu_si128((__m128i *)(d + i), _mm256_cvtps_ph(_mm256_loadu_ps(x + i), _MM_FROUND_TO_NEAREST_INT));
}

KERN_TARGET_END
#endif

}  // namespace kern
