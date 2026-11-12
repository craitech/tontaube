// AVX2 kernels for the VibeVoice acoustic tokenizer on x86-64 CPUs (app/codec/vibevoice_cpu.py).
//
// Activations are channels-last [batch][time][C] fp32. Dense fp32 weights for the
// fused blocks are packed in panels of 16 output columns: [N/16][K][16].

#include "common.h"

namespace {

// erf as in ATen's AVX2 Vectorized<float>::erf (Abramowitz-Stegun 7.1.26), with exp_neg.
inline __m256 erf8(__m256 x) {
    const __m256 sign = _mm256_and_ps(_mm256_set1_ps(-0.f), x);
    const __m256 ax = _mm256_xor_ps(sign, x);
    const __m256 t = _mm256_div_ps(_mm256_set1_ps(1.f), _mm256_fmadd_ps(_mm256_set1_ps(0.3275911f), ax, _mm256_set1_ps(1.f)));
    __m256 r = _mm256_fmadd_ps(_mm256_set1_ps(1.061405429f), t, _mm256_set1_ps(-1.453152027f));
    r = _mm256_fmadd_ps(r, t, _mm256_set1_ps(1.421413741f));
    r = _mm256_fmadd_ps(r, t, _mm256_set1_ps(-0.284496736f));
    r = _mm256_fmadd_ps(r, t, _mm256_set1_ps(0.254829592f));
    const __m256 e = exp_neg(_mm256_sub_ps(_mm256_setzero_ps(), _mm256_mul_ps(x, x)));
    const __m256 y = _mm256_fnmadd_ps(_mm256_mul_ps(e, t), r, _mm256_set1_ps(1.f));
    return _mm256_xor_ps(sign, y);
}

// GELU (erf form), n a multiple of 8.
inline void gelu_row(float *v, int64_t n) {
    const __m256 half = _mm256_set1_ps(0.5f), rs2 = _mm256_set1_ps(0.70710678118654752f), one = _mm256_set1_ps(1.f);
    for (int64_t i = 0; i < n; i += 8) {
        const __m256 x = _mm256_loadu_ps(v + i);
        _mm256_storeu_ps(v + i, _mm256_mul_ps(_mm256_mul_ps(half, x), _mm256_add_ps(one, erf8(_mm256_mul_ps(x, rs2)))));
    }
}

// out = x * rsqrt(mean(x^2) + eps) * w, C a multiple of 8.
inline void rms_row(const float *x, const float *w, int64_t C, float eps, float *out) {
    __m256 acc = _mm256_setzero_ps();
    for (int64_t c = 0; c < C; c += 8) {
        const __m256 v = _mm256_loadu_ps(x + c);
        acc = _mm256_fmadd_ps(v, v, acc);
    }
    const __m256 r = _mm256_set1_ps(1.f / std::sqrt(hsum(acc) / (float)C + eps));
    for (int64_t c = 0; c < C; c += 8)
        _mm256_storeu_ps(out + c, _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(x + c), r), _mm256_loadu_ps(w + c)));
}

// C[MR x 16] = A[MR x K] (row stride lda) @ panel [K x 16] (+ bias16). Named accumulators:
// with an array GCC keeps them on the stack and reloads them every iteration (3x slower).
template <int MR>
inline void mk16(const float *A, int64_t lda, const float *P, int64_t K, const float *bias16, float *Cm, int64_t ldc) {
    const __m256 i0 = bias16 ? _mm256_loadu_ps(bias16) : _mm256_setzero_ps();
    const __m256 i1 = bias16 ? _mm256_loadu_ps(bias16 + 8) : _mm256_setzero_ps();
    __m256 c00 = i0, c01 = i1, c10 = i0, c11 = i1, c20 = i0, c21 = i1;
    __m256 c30 = i0, c31 = i1, c40 = i0, c41 = i1, c50 = i0, c51 = i1;
    const float *a0 = A, *a1 = A + lda, *a2 = A + 2 * lda, *a3 = A + 3 * lda, *a4 = A + 4 * lda, *a5 = A + 5 * lda;
#define TT_ROW(m)                                                     \
    if (MR > m) {                                                     \
        const __m256 a = _mm256_broadcast_ss(a##m + k);               \
        c##m##0 = _mm256_fmadd_ps(a, b0, c##m##0);                    \
        c##m##1 = _mm256_fmadd_ps(a, b1, c##m##1);                    \
    }
    for (int64_t k = 0; k < K; k++) {
        const __m256 b0 = _mm256_loadu_ps(P + k * 16), b1 = _mm256_loadu_ps(P + k * 16 + 8);
        TT_ROW(0) TT_ROW(1) TT_ROW(2) TT_ROW(3) TT_ROW(4) TT_ROW(5)
    }
#undef TT_ROW
#define TT_STORE(m)                                                   \
    if (MR > m) {                                                     \
        _mm256_storeu_ps(Cm + m * ldc, c##m##0);                      \
        _mm256_storeu_ps(Cm + m * ldc + 8, c##m##1);                  \
    }
    TT_STORE(0) TT_STORE(1) TT_STORE(2) TT_STORE(3) TT_STORE(4) TT_STORE(5)
#undef TT_STORE
}

// C[M x 16] = A rows (stride lda, may overlap) @ one panel [K x 16] (+ bias16).
inline void panel_rows(const float *A, int64_t lda, int64_t M, const float *P, int64_t K, const float *b,
                       float *Cp, int64_t ldc) {
    int64_t m = 0;
    for (; m + 6 <= M; m += 6) mk16<6>(A + m * lda, lda, P, K, b, Cp + m * ldc, ldc);
    switch (M - m) {
        case 5: mk16<5>(A + m * lda, lda, P, K, b, Cp + m * ldc, ldc); break;
        case 4: mk16<4>(A + m * lda, lda, P, K, b, Cp + m * ldc, ldc); break;
        case 3: mk16<3>(A + m * lda, lda, P, K, b, Cp + m * ldc, ldc); break;
        case 2: mk16<2>(A + m * lda, lda, P, K, b, Cp + m * ldc, ldc); break;
        case 1: mk16<1>(A + m * lda, lda, P, K, b, Cp + m * ldc, ldc); break;
        default: break;
    }
}

// C[M x N] = A[M x K] @ B + bias with B packed [N/16][K][16] (single thread).
void gemm_packed(const float *A, int64_t lda, int64_t M, const float *Bp, int64_t K, int64_t N,
                 const float *bias, float *Cm, int64_t ldc) {
    for (int64_t p = 0; p < N / 16; p++)
        panel_rows(A, lda, M, Bp + p * K * 16, K, bias ? bias + p * 16 : nullptr, Cm + p * 16, ldc);
}

// Weight panels: fp32 [N/16][K][16], or int8 codes [N/16][K][16] with fp32 scale/min per
// (panel, group of QG rows of K, column): w = code * scale + min.
constexpr int QG = 128;

struct Panels {
    const float *f32 = nullptr;
    const int8_t *q = nullptr;
    const float *sc = nullptr, *mn = nullptr;
    int64_t K = 0, N = 0;

    // Panel p as fp32 (dequantized into buf for int8).
    const float *get(int64_t p, float *buf) const {
        if (f32) return f32 + p * K * 16;
        const int8_t *c = q + p * K * 16;
        const float *s = sc + p * (K / QG) * 16, *m = mn + p * (K / QG) * 16;
        for (int64_t g = 0; g < K / QG; g++) {
            const __m256 s0 = _mm256_loadu_ps(s + g * 16), s1 = _mm256_loadu_ps(s + g * 16 + 8);
            const __m256 m0 = _mm256_loadu_ps(m + g * 16), m1 = _mm256_loadu_ps(m + g * 16 + 8);
            for (int64_t k = g * QG; k < (g + 1) * QG; k++) {
                const __m128i v = _mm_loadu_si128((const __m128i *)(c + k * 16));
                const __m256 f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v));
                const __m256 f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(v, 8)));
                _mm256_storeu_ps(buf + k * 16, _mm256_fmadd_ps(f0, s0, m0));
                _mm256_storeu_ps(buf + k * 16 + 8, _mm256_fmadd_ps(f1, s1, m1));
            }
        }
        return buf;
    }
};

Panels make_panels(const torch::Tensor &f32, const torch::Tensor &q, const torch::Tensor &sc, const torch::Tensor &mn) {
    Panels w;
    if (f32.numel()) {
        TORCH_CHECK(f32.dim() == 3 && f32.size(2) == 16 && f32.is_contiguous() && f32.scalar_type() == torch::kFloat32);
        w.f32 = f32.data_ptr<float>();
        w.K = f32.size(1);
        w.N = f32.size(0) * 16;
    } else {
        TORCH_CHECK(q.dim() == 3 && q.size(2) == 16 && q.is_contiguous() && q.scalar_type() == torch::kInt8);
        TORCH_CHECK(q.size(1) % QG == 0 && sc.is_contiguous() && mn.is_contiguous() && sc.size(1) == q.size(1) / QG);
        w.q = q.data_ptr<int8_t>();
        w.sc = sc.data_ptr<float>();
        w.mn = mn.data_ptr<float>();
        w.K = q.size(1);
        w.N = q.size(0) * 16;
    }
    return w;
}

enum { EPI_NONE = 0, EPI_GELU = 1, EPI_RESIDUAL = 2 };

// C[M x N] = A rows @ W + bias, parallel over (panel, chunk of rows); the epilogue runs on each
// [rows x 16] block while it is in L1: GELU, or C = R + g * C (R row stride ldr, g [N]).
void gemm_panels(const float *A, int64_t lda, int64_t M, const Panels &W, const float *bias, float *Cm,
                 int64_t ldc, int epi = EPI_NONE, const float *R = nullptr, int64_t ldr = 0, const float *g = nullptr) {
    if (M <= 0) return;
    const int64_t P = W.N / 16, RM = 96, chunks = (M + RM - 1) / RM;
    at::parallel_for(0, P * chunks, 1, [&](int64_t t0, int64_t t1) {
        std::vector<float> buf(W.q ? W.K * 16 : 0);
        int64_t loaded = -1;
        const float *panel = nullptr;
        for (int64_t t = t0; t < t1; t++) {
            const int64_t p = t / chunks, m0 = (t % chunks) * RM, n = std::min(RM, M - m0);
            if (p != loaded) {
                panel = W.get(p, buf.data());
                loaded = p;
            }
            float *Cp = Cm + m0 * ldc + p * 16;
            panel_rows(A + m0 * lda, lda, n, panel, W.K, bias ? bias + p * 16 : nullptr, Cp, ldc);
            if (epi == EPI_GELU) {
                for (int64_t i = 0; i < n; i++) gelu_row(Cp + i * ldc, 16);
            } else if (epi == EPI_RESIDUAL) {
                const __m256 g0 = _mm256_loadu_ps(g + p * 16), g1 = _mm256_loadu_ps(g + p * 16 + 8);
                for (int64_t i = 0; i < n; i++) {
                    const float *r = R + (m0 + i) * ldr + p * 16;
                    float *c = Cp + i * ldc;
                    _mm256_storeu_ps(c, _mm256_fmadd_ps(g0, _mm256_loadu_ps(c), _mm256_loadu_ps(r)));
                    _mm256_storeu_ps(c + 8, _mm256_fmadd_ps(g1, _mm256_loadu_ps(c + 8), _mm256_loadu_ps(r + 8)));
                }
            }
        }
    });
}

constexpr int DWK = 7;  // depthwise kernel; the conv state is the previous DWK - 1 normed rows

// Rows i = 0..n-1 of x1 = x + gamma * (dw_b + sum_j dw[j] * h[i + j]) with h = normed rows
// starting DWK - 1 rows before the first output row.
inline void dw_residual(const float *x, const float *h, int64_t n, int64_t C, const float *dw, const float *dwb,
                        const float *gamma, float *x1) {
    for (int64_t i = 0; i < n; i++)
        for (int64_t c = 0; c < C; c += 8) {
            __m256 acc = _mm256_loadu_ps(dwb + c);
            for (int j = 0; j < DWK; j++)
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(dw + j * C + c), _mm256_loadu_ps(h + (i + j) * C + c), acc);
            _mm256_storeu_ps(x1 + i * C + c,
                             _mm256_fmadd_ps(_mm256_loadu_ps(gamma + c), acc, _mm256_loadu_ps(x + i * C + c)));
        }
}

// The normed rows t0 - 6 .. t1 - 1 of one sequence (rows before 0 come from the state or are zero).
inline void normed_rows(const float *x, const float *state, const float *norm_w, int64_t C, float eps,
                        int64_t t0, int64_t t1, float *h) {
    for (int64_t t = t0 - (DWK - 1); t < t1; t++) {
        float *dst = h + (t - t0 + DWK - 1) * C;
        if (t >= 0) rms_row(x + t * C, norm_w, C, eps, dst);
        else if (state) std::memcpy(dst, state + (t + DWK - 1) * C, C * sizeof(float));
        else std::memset(dst, 0, C * sizeof(float));
    }
}

// New conv state: the last DWK - 1 normed rows of [state; rows of x].
torch::Tensor next_state(const torch::Tensor &x, const torch::Tensor &state, const float *norm_w, float eps) {
    const int64_t B = x.size(0), T = x.size(1), C = x.size(2);
    auto ns = torch::empty({B, DWK - 1, C}, torch::kFloat32);
    const float *xp = x.data_ptr<float>(), *sp = state.numel() ? state.data_ptr<float>() : nullptr;
    for (int64_t b = 0; b < B; b++)
        normed_rows(xp + b * T * C, sp ? sp + b * (DWK - 1) * C : nullptr, norm_w, C, eps, T, T,
                    ns.data_ptr<float>() + b * (DWK - 1) * C);
    return ns;
}

}  // namespace

// C = A @ B + bias on panels, one thread per tile of 48 rows (for benchmarks and tests).
torch::Tensor vv_gemm(torch::Tensor A, torch::Tensor Bp, torch::Tensor bias) {
    const int64_t M = A.size(0), K = A.size(1), N = Bp.size(0) * 16;
    auto C = torch::empty({M, N}, torch::kFloat32);
    const float *ap = A.data_ptr<float>(), *bp = Bp.data_ptr<float>(), *bias_p = bias.numel() ? bias.data_ptr<float>() : nullptr;
    float *cp = C.data_ptr<float>();
    at::parallel_for(0, (M + 47) / 48, 1, [&](int64_t t0, int64_t t1) {
        for (int64_t t = t0; t < t1; t++) {
            const int64_t m0 = t * 48, n = std::min<int64_t>(48, M - m0);
            gemm_packed(ap + m0 * K, K, n, bp, K, N, bias_p, cp + m0 * N, N);
        }
    });
    return C;
}

// Window GEMM over channels-last rows: out[b, j] = [xx[b, j*s - H] .. xx[b, j*s + s - 1]] @ W + bias,
// xx = [prev (H rows, zeros if empty); x]. x [B, T, Cin] with T % s == 0 -> [B, T / s, N].
// Covers the causal k-tap conv (s = 1, H = k - 1), the k = 2s strided conv (H = s) and, on
// one-row frames, the k = 2s transposed conv (s = 1, H = 1, W = [W_prev; W_cur]).
torch::Tensor vv_conv(torch::Tensor x, torch::Tensor prev, int64_t s, int64_t H, torch::Tensor wf, torch::Tensor wq,
                      torch::Tensor wsc, torch::Tensor wmn, torch::Tensor bias) {
    TORCH_CHECK(x.is_contiguous() && x.scalar_type() == torch::kFloat32 && x.dim() == 3);
    const int64_t B = x.size(0), T = x.size(1), Cin = x.size(2), J = T / s;
    TORCH_CHECK(T % s == 0 && (prev.numel() == 0 || (prev.is_contiguous() && prev.size(1) == H)));
    const Panels W = make_panels(wf, wq, wsc, wmn);
    TORCH_CHECK(W.K == (H + s) * Cin, "window size does not match the weights");
    auto out = torch::empty({B, J, W.N}, torch::kFloat32);
    const float *bp = bias.numel() ? bias.data_ptr<float>() : nullptr;
    const int64_t head = std::min(J, (H + s - 1) / s);  // output rows that reach into prev
    for (int64_t b = 0; b < B; b++) {
        const float *xb = x.data_ptr<float>() + b * T * Cin;
        float *ob = out.data_ptr<float>() + b * J * W.N;
        if (head > 0) {  // [prev; first rows] gathered for the first output rows
            std::vector<float> tmp((H + head * s) * Cin, 0.f);
            if (prev.numel()) std::memcpy(tmp.data(), prev.data_ptr<float>() + b * H * Cin, H * Cin * sizeof(float));
            std::memcpy(tmp.data() + H * Cin, xb, std::min(T, head * s) * Cin * sizeof(float));
            gemm_panels(tmp.data(), s * Cin, head, W, bp, ob, W.N);
        }
        gemm_panels(xb + (head * s - H) * Cin, s * Cin, J - head, W, bp, ob + head * W.N, W.N);
    }
    return out;
}

// Wide Block1D in one call: mixer rows, then the FFN as panel GEMMs over all rows (GELU and
// the gamma residual in the epilogues). FFN weights fp32 or int8 panels.
std::vector<torch::Tensor> vv_block_wide(torch::Tensor x, torch::Tensor state, torch::Tensor norm_w, torch::Tensor dw,
                                         torch::Tensor dwb, torch::Tensor gamma, torch::Tensor ffn_norm_w,
                                         torch::Tensor w1f, torch::Tensor w1q, torch::Tensor w1s, torch::Tensor w1m,
                                         torch::Tensor b1, torch::Tensor w2f, torch::Tensor w2q, torch::Tensor w2s,
                                         torch::Tensor w2m, torch::Tensor b2, torch::Tensor ffn_gamma, double eps);

torch::Tensor vv_pack(torch::Tensor w) {  // [K, N] -> [N/16][K][16]
    TORCH_CHECK(w.dim() == 2 && w.size(1) % 16 == 0);
    const int64_t K = w.size(0), N = w.size(1);
    return w.to(torch::kFloat32).contiguous().view({K, N / 16, 16}).permute({1, 0, 2}).contiguous();
}

// Whole Block1D for C <= 256: rows are processed in tiles that stay in L2.
// x [B, T, C]; state [B, 6, C] or empty (zeros). Returns (out, next state).
std::vector<torch::Tensor> vv_block(torch::Tensor x, torch::Tensor state, torch::Tensor norm_w, torch::Tensor dw,
                                    torch::Tensor dwb, torch::Tensor gamma, torch::Tensor ffn_norm_w,
                                    torch::Tensor w1p, torch::Tensor b1, torch::Tensor w2p, torch::Tensor b2,
                                    torch::Tensor ffn_gamma, double eps_d) {
    TORCH_CHECK(x.is_contiguous() && x.scalar_type() == torch::kFloat32 && x.dim() == 3);
    const int64_t B = x.size(0), T = x.size(1), C = x.size(2), H = 4 * C;
    TORCH_CHECK(C % 16 == 0 && w1p.size(0) == H / 16 && w1p.size(1) == C && w2p.size(0) == C / 16 && w2p.size(1) == H);
    TORCH_CHECK(state.numel() == 0 || (state.is_contiguous() && state.size(0) == B && state.size(1) == DWK - 1));
    const float eps = (float)eps_d;
    auto out = torch::empty_like(x);
    const int64_t TT = std::max<int64_t>(6, std::min<int64_t>(192, (65536 / (8 * C)) / 6 * 6));
    const int64_t tiles = (T + TT - 1) / TT;
    const float *xp = x.data_ptr<float>(), *sp = state.numel() ? state.data_ptr<float>() : nullptr;
    const float *nw = norm_w.data_ptr<float>(), *dwp = dw.data_ptr<float>(), *dbp = dwb.data_ptr<float>();
    const float *gp = gamma.data_ptr<float>(), *fnw = ffn_norm_w.data_ptr<float>(), *fgp = ffn_gamma.data_ptr<float>();
    const float *W1 = w1p.data_ptr<float>(), *B1 = b1.data_ptr<float>(), *W2 = w2p.data_ptr<float>(), *B2 = b2.data_ptr<float>();
    float *op = out.data_ptr<float>();
    at::parallel_for(0, B * tiles, 1, [&](int64_t s0, int64_t s1) {
        std::vector<float> buf((TT + DWK - 1) * C + 3 * TT * C + TT * H);
        float *h = buf.data(), *x1 = h + (TT + DWK - 1) * C, *h2 = x1 + TT * C, *v = h2 + TT * C, *u = v + TT * C;
        for (int64_t s = s0; s < s1; s++) {
            const int64_t b = s / tiles, t0 = (s % tiles) * TT, t1 = std::min(T, t0 + TT), n = t1 - t0;
            const float *xb = xp + b * T * C;
            normed_rows(xb, sp ? sp + b * (DWK - 1) * C : nullptr, nw, C, eps, t0, t1, h);
            dw_residual(xb + t0 * C, h, n, C, dwp, dbp, gp, x1);
            for (int64_t i = 0; i < n; i++) rms_row(x1 + i * C, fnw, C, eps, h2 + i * C);
            gemm_packed(h2, C, n, W1, C, H, B1, u, H);
            for (int64_t i = 0; i < n; i++) gelu_row(u + i * H, H);
            gemm_packed(u, H, n, W2, H, C, B2, v, C);
            float *ob = op + (b * T + t0) * C;
            for (int64_t i = 0; i < n * C; i += 8)
                _mm256_storeu_ps(ob + i, _mm256_fmadd_ps(_mm256_loadu_ps(fgp + i % C), _mm256_loadu_ps(v + i),
                                                         _mm256_loadu_ps(x1 + i)));
        }
    });
    return {out, next_state(x, state, nw, eps)};
}

// Mixer half of a wide Block1D: returns (x1, h2 = rms(x1) * ffn_norm_w, next state).
std::vector<torch::Tensor> vv_mixer(torch::Tensor x, torch::Tensor state, torch::Tensor norm_w, torch::Tensor dw,
                                    torch::Tensor dwb, torch::Tensor gamma, torch::Tensor ffn_norm_w, double eps_d) {
    TORCH_CHECK(x.is_contiguous() && x.scalar_type() == torch::kFloat32 && x.dim() == 3);
    const int64_t B = x.size(0), T = x.size(1), C = x.size(2);
    TORCH_CHECK(C % 8 == 0 && (state.numel() == 0 || (state.is_contiguous() && state.size(0) == B)));
    const float eps = (float)eps_d;
    auto x1 = torch::empty_like(x), h2 = torch::empty_like(x);
    const int64_t TT = 16, tiles = (T + TT - 1) / TT;
    const float *xp = x.data_ptr<float>(), *sp = state.numel() ? state.data_ptr<float>() : nullptr;
    const float *nw = norm_w.data_ptr<float>(), *fnw = ffn_norm_w.data_ptr<float>();
    at::parallel_for(0, B * tiles, 1, [&](int64_t s0, int64_t s1) {
        std::vector<float> h((TT + DWK - 1) * C);
        for (int64_t s = s0; s < s1; s++) {
            const int64_t b = s / tiles, t0 = (s % tiles) * TT, t1 = std::min(T, t0 + TT), n = t1 - t0;
            const float *xb = xp + b * T * C;
            normed_rows(xb, sp ? sp + b * (DWK - 1) * C : nullptr, nw, C, eps, t0, t1, h.data());
            float *x1p = x1.data_ptr<float>() + (b * T + t0) * C;
            dw_residual(xb + t0 * C, h.data(), n, C, dw.data_ptr<float>(), dwb.data_ptr<float>(),
                        gamma.data_ptr<float>(), x1p);
            for (int64_t i = 0; i < n; i++) rms_row(x1p + i * C, fnw, C, eps, h2.data_ptr<float>() + (b * T + t0 + i) * C);
        }
    });
    return {x1, h2, next_state(x, state, nw, eps)};
}

std::vector<torch::Tensor> vv_block_wide(torch::Tensor x, torch::Tensor state, torch::Tensor norm_w, torch::Tensor dw,
                                         torch::Tensor dwb, torch::Tensor gamma, torch::Tensor ffn_norm_w,
                                         torch::Tensor w1f, torch::Tensor w1q, torch::Tensor w1s, torch::Tensor w1m,
                                         torch::Tensor b1, torch::Tensor w2f, torch::Tensor w2q, torch::Tensor w2s,
                                         torch::Tensor w2m, torch::Tensor b2, torch::Tensor ffn_gamma, double eps) {
    auto mixed = vv_mixer(x, state, norm_w, dw, dwb, gamma, ffn_norm_w, eps);
    const int64_t M = x.size(0) * x.size(1), C = x.size(2);
    const Panels W1 = make_panels(w1f, w1q, w1s, w1m), W2 = make_panels(w2f, w2q, w2s, w2m);
    TORCH_CHECK(W1.K == C && W2.K == W1.N && W2.N == C);
    auto u = torch::empty({M, W1.N}, torch::kFloat32);
    auto out = torch::empty_like(x);
    gemm_panels(mixed[1].data_ptr<float>(), C, M, W1, b1.data_ptr<float>(), u.data_ptr<float>(), W1.N, EPI_GELU);
    gemm_panels(u.data_ptr<float>(), W1.N, M, W2, b2.data_ptr<float>(), out.data_ptr<float>(), C, EPI_RESIDUAL,
                mixed[0].data_ptr<float>(), C, ffn_gamma.data_ptr<float>());
    return {out, mixed[2]};
}

void vv_gelu_(torch::Tensor u) {
    TORCH_CHECK(u.is_contiguous() && u.scalar_type() == torch::kFloat32 && u.size(-1) % 8 == 0);
    const int64_t n = u.size(-1), rows = u.numel() / n;
    float *p = u.data_ptr<float>();
    at::parallel_for(0, rows, 16, [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; r++) gelu_row(p + r * n, n);
    });
}

// Causal stride-1 conv with few channels: y[t] = b + sum_j x[t - K + 1 + j] @ W[j].
// x [B, T, Cin]; state [B, K-1, Cin] or empty; w [K, Cin, Cout]. Cout % 8 == 0 or Cout == 1 (Cin % 8 == 0).
torch::Tensor vv_conv_small(torch::Tensor x, torch::Tensor state, torch::Tensor w, torch::Tensor bias) {
    TORCH_CHECK(x.is_contiguous() && x.scalar_type() == torch::kFloat32 && x.dim() == 3);
    const int64_t B = x.size(0), T = x.size(1), Cin = x.size(2), K = w.size(0), Cout = w.size(2);
    TORCH_CHECK(w.size(1) == Cin && (Cout % 8 == 0 || (Cout == 1 && Cin % 8 == 0)));
    auto y = torch::empty({B, T, Cout}, torch::kFloat32);
    const float *xp = x.data_ptr<float>(), *sp = state.numel() ? state.data_ptr<float>() : nullptr;
    const float *wp = w.data_ptr<float>(), *bp = bias.data_ptr<float>();
    float *yp = y.data_ptr<float>();
    auto row = [&](int64_t b, int64_t t) -> const float * {  // input row t (may be < 0)
        if (t >= 0) return xp + (b * T + t) * Cin;
        return sp ? sp + (b * (K - 1) + (t + K - 1)) * Cin : nullptr;
    };
    at::parallel_for(0, B * T, 256, [&](int64_t i0, int64_t i1) {
        for (int64_t i = i0; i < i1; i++) {
            const int64_t b = i / T, t = i % T;
            float *out = yp + i * Cout;
            if (Cout == 1) {
                __m256 acc = _mm256_setzero_ps();
                for (int64_t j = 0; j < K; j++) {
                    const float *r = row(b, t - K + 1 + j);
                    if (!r) continue;
                    for (int64_t c = 0; c < Cin; c += 8)
                        acc = _mm256_fmadd_ps(_mm256_loadu_ps(r + c), _mm256_loadu_ps(wp + j * Cin + c), acc);
                }
                out[0] = hsum(acc) + bp[0];
                continue;
            }
            for (int64_t o = 0; o < Cout; o += 8) {
                __m256 acc = _mm256_loadu_ps(bp + o);
                for (int64_t j = 0; j < K; j++) {
                    const float *r = row(b, t - K + 1 + j);
                    if (!r) continue;
                    for (int64_t c = 0; c < Cin; c++)
                        acc = _mm256_fmadd_ps(_mm256_set1_ps(r[c]), _mm256_loadu_ps(wp + (j * Cin + c) * Cout + o), acc);
                }
                _mm256_storeu_ps(out + o, acc);
            }
        }
    });
    return y;
}

void register_vibevoice(pybind11::module &m) {
    m.def("vv_pack", &vv_pack);
    m.def("vv_gemm", &vv_gemm);
    m.def("vv_conv", &vv_conv);
    m.def("vv_block_wide", &vv_block_wide);
    m.def("vv_block", &vv_block);
    m.def("vv_mixer", &vv_mixer);
    m.def("vv_gelu_", &vv_gelu_);
    m.def("vv_conv_small", &vv_conv_small);
}
