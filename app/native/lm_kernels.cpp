// AVX2 kernels for the lite codebook models on x86-64 CPUs.
//
// Weight formats (packed once at load time from the GPU bundle layout):
//   q4: codes uint8 [N][K/64][32]; byte i of a group holds input i (low
//       nibble) and input i + 32 (high nibble). fp16 scales/mins [N][K/64].
//   q8: codes int8 [N][K/128][128] = code - 128; fp16 scales [N][K/128] and
//       mins [N][K/128] already shifted by 128 * scale.
//   In both, w = code * scale + min.
// Activations are quantized to int16 per 64 inputs (rounding error ~1/32767,
// negligible next to the weights): q, scale, and scale * sum(q) ("xsum"),
// which carries the min term:
//   sum_k w_k x_k = sum_g scale_g * xscale_g * dot(code, q)_g + min_g * xsum_g
//
// Decoder runs a whole one-token decode step (all layers) in one call.
#include "common.h"

namespace {

constexpr int AG = 64;   // activation quantization group
constexpr int HD = 128;  // head dim of the attention kernel

struct Act {  // quantized activations, M rows of K inputs
    const int16_t *q;
    const float *s, *xs;
    int64_t M, K;
};

void quantize_rows(const float *xp, int64_t M, int64_t K, int16_t *qp, float *sp, float *xsp) {
    const int64_t G = K / AG;
    at::parallel_for(0, M * G, 64, [&](int64_t b, int64_t e) {
        for (int64_t i = b; i < e; i++) {
            const float *v = xp + i * AG;
            float amax = 0.f;
            for (int k = 0; k < AG; k++) amax = std::max(amax, std::fabs(v[k]));
            const float sc = amax / 32767.f, inv = amax > 0.f ? 32767.f / amax : 0.f;
            int64_t sum = 0;
            for (int k = 0; k < AG; k++) {
                int c = (int)std::nearbyint(v[k] * inv);
                c = std::max(-32767, std::min(32767, c));
                qp[i * AG + k] = (int16_t)c;
                sum += c;
            }
            sp[i] = sc;
            xsp[i] = sc * (float)sum;
        }
    });
}

// y[m][n] = sum_k w[n][k] x[m][k] (+ bias), m <= 8.
void gemv_q4_raw(const Act &a, const uint8_t *cp, const uint16_t *scp, const uint16_t *mnp,
                 const float *bp, int64_t N, float *yp) {
    const int64_t M = a.M, K = a.K, G = K / AG;
    const int16_t *qp = a.q;
    const float *sp = a.s, *xsp = a.xs;
    const __m128i m4 = _mm_set1_epi8(0x0F);
    at::parallel_for(0, N, 16, [&](int64_t n0, int64_t n1) {
        for (int64_t n = n0; n < n1; n++) {
            __m256 acc[8];
            float mterm[8];
            for (int m = 0; m < M; m++) { acc[m] = _mm256_setzero_ps(); mterm[m] = 0.f; }
            const uint8_t *row = cp + n * G * 32;
            for (int64_t g = 0; g < G; g++) {
                const __m128i c0 = _mm_loadu_si128((const __m128i *)(row + g * 32));
                const __m128i c1 = _mm_loadu_si128((const __m128i *)(row + g * 32 + 16));
                // inputs 0-15, 16-31 (low nibbles) and 32-47, 48-63 (high nibbles) as int16
                const __m256i w0 = _mm256_cvtepu8_epi16(_mm_and_si128(c0, m4));
                const __m256i w1 = _mm256_cvtepu8_epi16(_mm_and_si128(c1, m4));
                const __m256i w2 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(c0, 4), m4));
                const __m256i w3 = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(c1, 4), m4));
                const float ws = h2f(scp[n * G + g]), wm = h2f(mnp[n * G + g]);
                for (int m = 0; m < M; m++) {
                    const int16_t *xq = qp + m * K + g * AG;
                    __m256i d = _mm256_madd_epi16(w0, _mm256_loadu_si256((const __m256i *)xq));
                    d = _mm256_add_epi32(d, _mm256_madd_epi16(w1, _mm256_loadu_si256((const __m256i *)(xq + 16))));
                    d = _mm256_add_epi32(d, _mm256_madd_epi16(w2, _mm256_loadu_si256((const __m256i *)(xq + 32))));
                    d = _mm256_add_epi32(d, _mm256_madd_epi16(w3, _mm256_loadu_si256((const __m256i *)(xq + 48))));
                    acc[m] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d), _mm256_set1_ps(ws * sp[m * G + g]), acc[m]);
                    mterm[m] += wm * xsp[m * G + g];
                }
            }
            for (int m = 0; m < M; m++) yp[m * N + n] = hsum(acc[m]) + mterm[m] + (bp ? bp[n] : 0.f);
        }
    });
}

// int8 weights, group 128 (two activation groups per weight group).
void gemv_q8_raw(const Act &a, const int8_t *cp, const uint16_t *scp, const uint16_t *mnp,
                 const float *bp, int64_t N, float *yp) {
    const int64_t M = a.M, K = a.K, G = K / 128, GA = K / AG;
    const int16_t *qp = a.q;
    const float *sp = a.s, *xsp = a.xs;
    at::parallel_for(0, N, 16, [&](int64_t n0, int64_t n1) {
        for (int64_t n = n0; n < n1; n++) {
            __m256 acc[8];
            float mterm[8];
            for (int m = 0; m < M; m++) { acc[m] = _mm256_setzero_ps(); mterm[m] = 0.f; }
            const int8_t *row = cp + n * G * 128;
            for (int64_t g = 0; g < G; g++) {
                const float ws = h2f(scp[n * G + g]), wm = h2f(mnp[n * G + g]);
                for (int h = 0; h < 2; h++) {  // two 64-input halves (one activation group each)
                    const int8_t *c = row + g * 128 + h * 64;
                    __m256i w[4];
                    for (int j = 0; j < 4; j++)
                        w[j] = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(c + 16 * j)));
                    const int64_t ga = g * 2 + h;
                    for (int m = 0; m < M; m++) {
                        const int16_t *xq = qp + m * K + ga * AG;
                        __m256i d = _mm256_madd_epi16(w[0], _mm256_loadu_si256((const __m256i *)xq));
                        for (int j = 1; j < 4; j++)
                            d = _mm256_add_epi32(d, _mm256_madd_epi16(w[j], _mm256_loadu_si256((const __m256i *)(xq + 16 * j))));
                        acc[m] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d), _mm256_set1_ps(ws * sp[m * GA + ga]), acc[m]);
                        mterm[m] += wm * xsp[m * GA + ga];
                    }
                }
            }
            for (int m = 0; m < M; m++) yp[m * N + n] = hsum(acc[m]) + mterm[m] + (bp ? bp[n] : 0.f);
        }
    });
}

// One query token against a bf16 KV cache (flash-decoding: chunks of positions
// per KV head run in parallel, partial softmaxes are merged at the end).
// q [nq][128] fp32; K, V: head h at K + h * ks, rows of 128; attends to rows
// 0..L-1. out [nq * 128] fp32, head-major.
void attn_raw(const float *qp, const at::BFloat16 *kp, const at::BFloat16 *vp, int64_t ks, int64_t vs,
              int64_t nq, int64_t nkv, int64_t L, float fs, float *op) {
    const int64_t rep = nq / nkv;
    const int64_t want = std::max<int64_t>(1, (3 * at::get_num_threads() + nkv - 1) / nkv);
    const int64_t C = std::max<int64_t>(32, (L + want - 1) / want), nc = (L + C - 1) / C;
    std::vector<float> part(nkv * nc * rep * (HD + 2));  // per chunk: acc[HD], max, sum
    float *pp = part.data();
    at::parallel_for(0, nkv * nc, 1, [&](int64_t t0, int64_t t1) {
        std::vector<float> sbuf(rep * C + 8);
        for (int64_t t = t0; t < t1; t++) {
            const int64_t h = t / nc, c = t % nc, j0 = c * C, j1 = std::min(L, j0 + C), n = j1 - j0;
            const at::BFloat16 *K = kp + h * ks, *V = vp + h * vs;
            float *s = sbuf.data();
            for (int64_t j = j0; j < j1; j++) {
                __m256 kv[HD / 8];
                for (int i = 0; i < HD / 8; i++) kv[i] = bf16x8(K + j * HD + 8 * i);
                for (int r = 0; r < rep; r++) {
                    const float *qr = qp + (h * rep + r) * HD;
                    __m256 a = _mm256_mul_ps(kv[0], _mm256_loadu_ps(qr));
                    for (int i = 1; i < HD / 8; i++) a = _mm256_fmadd_ps(kv[i], _mm256_loadu_ps(qr + 8 * i), a);
                    s[r * C + (j - j0)] = hsum(a) * fs;
                }
            }
            for (int r = 0; r < rep; r++) {
                float *sr = s + r * C, mx = -INFINITY, sum = 0.f;
                for (int64_t j = 0; j < n; j++) mx = std::max(mx, sr[j]);
                int64_t j = 0;
                __m256 vsum = _mm256_setzero_ps();
                for (; j + 8 <= n; j += 8) {
                    const __m256 e = exp_neg(_mm256_sub_ps(_mm256_loadu_ps(sr + j), _mm256_set1_ps(mx)));
                    _mm256_storeu_ps(sr + j, e);
                    vsum = _mm256_add_ps(vsum, e);
                }
                sum = hsum(vsum);
                for (; j < n; j++) { sr[j] = std::exp(sr[j] - mx); sum += sr[j]; }
                float *dst = pp + (t * rep + r) * (HD + 2);
                dst[HD] = mx;
                dst[HD + 1] = sum;
            }
            __m256 acc[8][HD / 8 / 2];  // half of HD per pass keeps accumulators in registers for rep <= 2
            for (int half = 0; half < 2; half++) {
                for (int r = 0; r < rep; r++)
                    for (int i = 0; i < HD / 16; i++) acc[r][i] = _mm256_setzero_ps();
                for (int64_t j = j0; j < j1; j++) {
                    __m256 vv[HD / 16];
                    for (int i = 0; i < HD / 16; i++) vv[i] = bf16x8(V + j * HD + half * (HD / 2) + 8 * i);
                    for (int r = 0; r < rep; r++) {
                        const __m256 p = _mm256_set1_ps(s[r * C + (j - j0)]);
                        for (int i = 0; i < HD / 16; i++) acc[r][i] = _mm256_fmadd_ps(p, vv[i], acc[r][i]);
                    }
                }
                for (int r = 0; r < rep; r++) {
                    float *dst = pp + (t * rep + r) * (HD + 2) + half * (HD / 2);
                    for (int i = 0; i < HD / 16; i++) _mm256_storeu_ps(dst + 8 * i, acc[r][i]);
                }
            }
        }
    });
    // Merge chunks: out = sum_c e^(m_c - M) acc_c / sum_c e^(m_c - M) l_c
    at::parallel_for(0, nq, 1, [&](int64_t h0, int64_t h1) {
        for (int64_t hq = h0; hq < h1; hq++) {
            const int64_t h = hq / rep, r = hq % rep;
            float M = -INFINITY, den = 0.f;
            for (int64_t c = 0; c < nc; c++) M = std::max(M, pp[((h * nc + c) * rep + r) * (HD + 2) + HD]);
            float *o = op + hq * HD;
            for (int i = 0; i < HD; i++) o[i] = 0.f;
            for (int64_t c = 0; c < nc; c++) {
                const float *src = pp + ((h * nc + c) * rep + r) * (HD + 2);
                const float w = std::exp(src[HD] - M);
                den += w * src[HD + 1];
                for (int i = 0; i < HD; i++) o[i] += w * src[i];
            }
            const float inv = 1.f / den;
            for (int i = 0; i < HD; i++) o[i] *= inv;
        }
    });
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
    gemv_q4_raw({q.data_ptr<int16_t>(), s.data_ptr<float>(), xs.data_ptr<float>(), M, K},
                codes.data_ptr<uint8_t>(), half_ptr(scales), half_ptr(mins),
                bias.has_value() ? bias->data_ptr<float>() : nullptr, N, y.data_ptr<float>());
    return y;
}

torch::Tensor gemv_q8(torch::Tensor q, torch::Tensor s, torch::Tensor xs, torch::Tensor codes,
                      torch::Tensor scales, torch::Tensor mins, c10::optional<torch::Tensor> bias) {
    const int64_t M = q.size(0), K = q.size(1), N = codes.size(0);
    TORCH_CHECK(M <= 8 && codes.size(1) == K / 128 && codes.size(2) == 128);
    auto y = torch::empty({M, N}, torch::kFloat32);
    gemv_q8_raw({q.data_ptr<int16_t>(), s.data_ptr<float>(), xs.data_ptr<float>(), M, K},
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
        if (kinds_[4 * l + j] == 0)
            gemv_q4_raw(act, w[0].data_ptr<uint8_t>(), half_ptr(w[1]), half_ptr(w[2]), bp, w[0].size(0), y);
        else
            gemv_q8_raw(act, w[0].data_ptr<int8_t>(), half_ptr(w[1]), half_ptr(w[2]), bp, w[0].size(0), y);
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
