// Kernels for the lite codebook models on x86-64 CPUs: torch wrappers over kern/ (shared with the Flash
// engine). Weight and activation formats: kern/lin_q.h.
//
// Decoder runs a whole one-token decode step (all layers) in one call.
#include "common.h"
#include "kern/attn.h"
#include "kern/lin_q.h"

namespace {

constexpr int AG = 64;   // activation quantization group
constexpr int HD = 128;  // head dim of the attention kernel

struct Act {  // quantized activations, M rows of K inputs
    const int16_t *q;
    const float *s, *xs;
    int64_t M, K;
};

void quantize_rows(const float *xp, int64_t M, int64_t K, int16_t *qp, float *sp, float *xsp) {
    kern::quant16(torch_par(), xp, K, M, K, qp, sp, xsp);
}

// y[m][n] = sum_k w[n][k] x[m][k] (+ bias); kind 0 q4, 1 q8.
void gemv_raw(int kind, const Act &a, const void *cp, const uint16_t *scp, const uint16_t *mnp, const float *bp,
              int64_t N, float *yp) {
    kern::QW w;
    w.kind = kind;
    w.N = N;
    w.K = a.K;
    w.codes = (const uint8_t *)cp;
    w.scales = scp;
    w.mins = mnp;
    w.bias = bp;
    kern::lin_q(torch_par(), w, kern::QX{a.q, a.s, a.xs, a.M, a.K}, yp, N);
}

// One query token against a bf16 KV cache (kern::attn: chunks of positions per KV head in parallel, merged).
void attn_raw(const float *qp, const at::BFloat16 *kp, const at::BFloat16 *vp, int64_t ks, int64_t vs,
              int64_t nq, int64_t nkv, int64_t L, float fs, float *op) {
    const int64_t want = std::max<int64_t>(1, (3 * at::get_num_threads() + nkv - 1) / nkv);
    const int64_t C = std::max<int64_t>(32, (L + want - 1) / want);
    kern::attn<kern::KV_BF16>(torch_par(), qp, (const uint16_t *)kp, (const uint16_t *)vp, ks, vs, nq, nkv, L, fs, C, op);
}

// out = x * rsqrt(mean(x^2) + eps) * w
inline void rmsnorm(const float *x, const float *w, int64_t n, float eps, float *out) {
    float ss = 0.f;
    for (int64_t i = 0; i < n; i++) ss += x[i] * x[i];
    const float r = 1.f / std::sqrt(ss / (float)n + eps);
    for (int64_t i = 0; i < n; i++) out[i] = x[i] * r * w[i];
}

// Rotary embedding of one head in place: [a, b] -> [a cos - b sin, b cos + a sin].
inline void rope(float *v, const float *cs, const float *sn) {
    for (int i = 0; i < HD / 2; i++) {
        const float a = v[i], b = v[i + HD / 2];
        v[i] = a * cs[i] - b * sn[i];
        v[i + HD / 2] = b * cs[i] + a * sn[i];
    }
}

}  // namespace

std::vector<torch::Tensor> quantize_act(torch::Tensor x) {
    TORCH_CHECK(x.dim() == 2 && x.size(1) % AG == 0);
    x = x.contiguous().to(torch::kFloat32);
    const int64_t M = x.size(0), K = x.size(1), G = K / AG;
    auto q = torch::empty({M, K}, torch::kInt16);
    auto s = torch::empty({M, G}, torch::kFloat32);
    auto xs = torch::empty({M, G}, torch::kFloat32);
    quantize_rows(x.data_ptr<float>(), M, K, q.data_ptr<int16_t>(), s.data_ptr<float>(), xs.data_ptr<float>());
    return {q, s, xs};
}

torch::Tensor gemv_q4(torch::Tensor q, torch::Tensor s, torch::Tensor xs, torch::Tensor codes,
                      torch::Tensor scales, torch::Tensor mins, c10::optional<torch::Tensor> bias) {
    const int64_t M = q.size(0), K = q.size(1), N = codes.size(0);
    TORCH_CHECK(M <= 8 && codes.size(1) == K / AG && codes.size(2) == 32);
    auto y = torch::empty({M, N}, torch::kFloat32);
    gemv_raw(0, {q.data_ptr<int16_t>(), s.data_ptr<float>(), xs.data_ptr<float>(), M, K},
             codes.data_ptr<uint8_t>(), half_ptr(scales), half_ptr(mins),
             bias.has_value() ? bias->data_ptr<float>() : nullptr, N, y.data_ptr<float>());
    return y;
}

torch::Tensor gemv_q8(torch::Tensor q, torch::Tensor s, torch::Tensor xs, torch::Tensor codes,
                      torch::Tensor scales, torch::Tensor mins, c10::optional<torch::Tensor> bias) {
    const int64_t M = q.size(0), K = q.size(1), N = codes.size(0);
    TORCH_CHECK(M <= 8 && codes.size(1) == K / 128 && codes.size(2) == 128);
    auto y = torch::empty({M, N}, torch::kFloat32);
    gemv_raw(1, {q.data_ptr<int16_t>(), s.data_ptr<float>(), xs.data_ptr<float>(), M, K},
             codes.data_ptr<int8_t>(), half_ptr(scales), half_ptr(mins),
             bias.has_value() ? bias->data_ptr<float>() : nullptr, N, y.data_ptr<float>());
    return y;
}

// q [nq][128] fp32; kc, vc [nkv][S][128] bf16; attends to cache rows 0..L-1.
torch::Tensor decode_attn(torch::Tensor q, torch::Tensor kc, torch::Tensor vc, int64_t L, double scale) {
    const int64_t nq = q.size(0), nkv = kc.size(0);
    TORCH_CHECK(q.size(1) == HD && kc.size(2) == HD && kc.stride(1) == HD && vc.stride(1) == HD && nq / nkv <= 8);
    TORCH_CHECK(L >= 1 && L <= kc.size(1));
    q = q.contiguous();
    auto out = torch::empty({nq * HD}, torch::kFloat32);
    attn_raw(q.data_ptr<float>(), kc.data_ptr<at::BFloat16>(), vc.data_ptr<at::BFloat16>(), kc.stride(0),
             vc.stride(0), nq, nkv, L, (float)scale, out.data_ptr<float>());
    return out;
}

// Dense fp32 weights for prefill matmuls: rows r0..r1-1 into out [r1-r0][K].
void dequant_q4(torch::Tensor codes, torch::Tensor scales, torch::Tensor mins, torch::Tensor out,
                int64_t r0, int64_t r1) {
    const int64_t G = codes.size(1);
    TORCH_CHECK(out.is_contiguous() && out.size(0) >= r1 - r0 && out.size(1) == G * 64);
    const uint8_t *cp = codes.data_ptr<uint8_t>();
    const uint16_t *scp = half_ptr(scales), *mnp = half_ptr(mins);
    float *wp = out.data_ptr<float>();
    const __m256i m4 = _mm256_set1_epi32(0xF);
    at::parallel_for(r0, r1, 8, [&](int64_t n0, int64_t n1) {
        for (int64_t n = n0; n < n1; n++)
            for (int64_t g = 0; g < G; g++) {
                const __m256 s = _mm256_set1_ps(h2f(scp[n * G + g])), m = _mm256_set1_ps(h2f(mnp[n * G + g]));
                const uint8_t *c = cp + (n * G + g) * 32;
                float *o = wp + (n - r0) * G * 64 + g * 64;
                for (int j = 0; j < 32; j += 8) {
                    const __m256i v = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(c + j)));
                    _mm256_storeu_ps(o + j, _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_and_si256(v, m4)), s, m));
                    _mm256_storeu_ps(o + j + 32, _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_srli_epi32(v, 4)), s, m));
                }
            }
    });
}

void dequant_q8(torch::Tensor codes, torch::Tensor scales, torch::Tensor mins, torch::Tensor out,
                int64_t r0, int64_t r1) {
    const int64_t G = codes.size(1);
    TORCH_CHECK(out.is_contiguous() && out.size(0) >= r1 - r0 && out.size(1) == G * 128);
    const int8_t *cp = codes.data_ptr<int8_t>();
    const uint16_t *scp = half_ptr(scales), *mnp = half_ptr(mins);
    float *wp = out.data_ptr<float>();
    at::parallel_for(r0, r1, 8, [&](int64_t n0, int64_t n1) {
        for (int64_t n = n0; n < n1; n++)
            for (int64_t g = 0; g < G; g++) {
                const __m256 s = _mm256_set1_ps(h2f(scp[n * G + g])), m = _mm256_set1_ps(h2f(mnp[n * G + g]));
                const int8_t *c = cp + (n * G + g) * 128;
                float *o = wp + (n - r0) * G * 128 + g * 128;
                for (int j = 0; j < 128; j += 8) {
                    const __m256i v = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)(c + j)));
                    _mm256_storeu_ps(o + j, _mm256_fmadd_ps(_mm256_cvtepi32_ps(v), s, m));
                }
            }
    });
}

// All decoder layers of one one-token step: what CpuCodebookModel.forward does
// for T == 1, without per-op dispatch. Linears are q4/q8 (kind 0/1), given as
// (codes, scales, mins, bias-or-empty); norms per layer: in, post, q, k.
class Decoder {
  public:
    Decoder(std::vector<torch::Tensor> norms, std::vector<torch::Tensor> lins, std::vector<int64_t> kinds,
            torch::Tensor inv_freq, int64_t nq, int64_t nkv, double eps)
        : norms_(std::move(norms)), lins_(std::move(lins)), kinds_(std::move(kinds)),
          inv_freq_(inv_freq.contiguous().to(torch::kFloat32)), nq_(nq), nkv_(nkv), eps_((float)eps) {
        n_layers_ = norms_.size() / 4;
        TORCH_CHECK((int64_t)norms_.size() == 4 * n_layers_ && (int64_t)lins_.size() == 16 * n_layers_ &&
                    (int64_t)kinds_.size() == 4 * n_layers_ && inv_freq_.numel() == HD / 2);
        hidden_ = norms_[0].numel();
        inter_ = 0;
        for (int64_t l = 0; l < n_layers_; l++) inter_ = std::max(inter_, lins_[16 * l + 8].size(0) / 2);
        const int64_t width = std::max({hidden_, (nq_ + 2 * nkv_) * HD, 2 * inter_});
        buf_.resize(4 * width);
        qbuf_.resize(width);
        sbuf_.resize(2 * width / AG + 2);
    }

    // x_in [hidden]: embedded token; kv [layers][2][nkv][S][128] bf16; the token is
    // written at cache row t and rotated to position pos. Returns the hidden state.
    torch::Tensor step(torch::Tensor x_in, torch::Tensor kv, double pos, int64_t t) {
        TORCH_CHECK(kv.scalar_type() == torch::kBFloat16 && kv.is_contiguous() && kv.size(0) == n_layers_ &&
                    kv.size(2) == nkv_ && kv.size(4) == HD);
        TORCH_CHECK(t >= 0 && t < kv.size(3) && x_in.numel() == hidden_);
        auto x = x_in.to(torch::kFloat32).contiguous().clone();
        float *xp = x.data_ptr<float>();
        const int64_t H = hidden_, nq = nq_, nkv = nkv_, S = kv.size(3);
        const float scale = 1.f / std::sqrt((float)HD);
        float cs[HD / 2], sn[HD / 2];
        const float *invf = inv_freq_.data_ptr<float>();
        for (int i = 0; i < HD / 2; i++) {
            const float ang = (float)pos * invf[i];
            cs[i] = std::cos(ang);
            sn[i] = std::sin(ang);
        }
        const int64_t width = buf_.size() / 4;
        float *h = buf_.data(), *qkv = h + width, *a = qkv + width, *y = a + width;
        at::BFloat16 *kvp = kv.data_ptr<at::BFloat16>();
        for (int64_t l = 0; l < n_layers_; l++) {
            const float *in_norm = norms_[4 * l].data_ptr<float>(), *post_norm = norms_[4 * l + 1].data_ptr<float>();
            const float *q_norm = norms_[4 * l + 2].data_ptr<float>(), *k_norm = norms_[4 * l + 3].data_ptr<float>();
            at::BFloat16 *K = kvp + (l * 2) * nkv * S * HD, *V = kvp + (l * 2 + 1) * nkv * S * HD;

            rmsnorm(xp, in_norm, H, eps_, h);
            linear(l, 0, h, H, qkv);
            for (int64_t i = 0; i < nq + nkv; i++) {  // q heads, then k heads: per-head norm + rope
                float *v = qkv + i * HD;
                rmsnorm(v, i < nq ? q_norm : k_norm, HD, eps_, v);
                rope(v, cs, sn);
            }
            for (int64_t i = 0; i < nkv; i++) {
                const float *kr = qkv + (nq + i) * HD, *vr = qkv + (nq + nkv + i) * HD;
                at::BFloat16 *kd = K + (i * S + t) * HD, *vd = V + (i * S + t) * HD;
                for (int d = 0; d < HD; d++) { kd[d] = at::BFloat16(kr[d]); vd[d] = at::BFloat16(vr[d]); }
            }
            attn_raw(qkv, K, V, S * HD, S * HD, nq, nkv, t + 1, scale, a);
            linear(l, 1, a, nq * HD, y);
            for (int64_t i = 0; i < H; i++) xp[i] += y[i];

            rmsnorm(xp, post_norm, H, eps_, h);
            const int64_t I = lins_[16 * l + 8].size(0) / 2;
            linear(l, 2, h, H, a);
            at::parallel_for(0, I, 1024, [&](int64_t i0, int64_t i1) {
                for (int64_t i = i0; i < i1; i++) a[i] = a[i] / (1.f + std::exp(-a[i])) * a[I + i];
            });
            linear(l, 3, a, I, y);
            for (int64_t i = 0; i < H; i++) xp[i] += y[i];
        }
        return x;
    }

  private:
    // y = W x for linear j (0 qkv, 1 o, 2 gate_up, 3 down) of layer l; x has K inputs.
    void linear(int64_t l, int j, const float *x, int64_t K, float *y) {
        const torch::Tensor *w = &lins_[16 * l + 4 * j];
        float *s = sbuf_.data(), *xs = s + K / AG;
        quantize_rows(x, 1, K, qbuf_.data(), s, xs);
        const Act act{qbuf_.data(), s, xs, 1, K};
        const float *bp = w[3].numel() ? w[3].data_ptr<float>() : nullptr;
        gemv_raw((int)kinds_[4 * l + j], act, w[0].data_ptr(), half_ptr(w[1]), half_ptr(w[2]), bp, w[0].size(0), y);
    }

    std::vector<torch::Tensor> norms_, lins_;
    std::vector<int64_t> kinds_;
    torch::Tensor inv_freq_;
    int64_t nq_, nkv_, n_layers_, hidden_, inter_;
    float eps_;
    std::vector<float> buf_, sbuf_;
    std::vector<int16_t> qbuf_;
};

void register_lm(pybind11::module &m) {
    m.def("quantize_act", &quantize_act);
    m.def("gemv_q4", &gemv_q4);
    m.def("gemv_q8", &gemv_q8);
    m.def("dequant_q4", &dequant_q4);
    m.def("dequant_q8", &dequant_q8);
    m.def("decode_attn", &decode_attn);
    pybind11::class_<Decoder>(m, "Decoder")
        .def(pybind11::init<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<int64_t>,
                            torch::Tensor, int64_t, int64_t, double>())
        .def("step", &Decoder::step, pybind11::call_guard<pybind11::gil_scoped_release>());
}
