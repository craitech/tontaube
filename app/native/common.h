// Helpers shared by the AVX2 kernels (lm_kernels.cpp, vibevoice_kernels.cpp): torch glue over kern/.
#pragma once
#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <immintrin.h>

#include <cmath>
#include <cstring>
#include <vector>

#include "kern/common.h"

namespace {

using kern::h2f;
using kern::hsum;

inline const uint16_t *half_ptr(const torch::Tensor &t) {
    return reinterpret_cast<const uint16_t *>(t.data_ptr<at::Half>());
}

inline __m256 bf16x8(const at::BFloat16 *p) { return kern::bf16x8(reinterpret_cast<const uint16_t *>(p)); }

// kern::Par over at::parallel_for
inline void torch_run(void *, int64_t n, int64_t grain, kern::RangeFn fn, void *ctx) {
    at::parallel_for(0, n, grain, [&](int64_t b, int64_t e) { fn(ctx, b, e); });
}
inline kern::Par torch_par() {
    kern::Par p;
    p.run = torch_run;
    p.threads = at::get_num_threads();
    return p;
}

}  // namespace
