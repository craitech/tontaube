// Torch wrappers of the VibeVoice acoustic tokenizer kernels (kern/vv.h) for x86-64 CPUs (app/codec/vibevoice_cpu.py).
//
// Activations are channels-last [batch][time][C] fp32. Dense fp32 weights for the
// fused blocks are packed in panels of 16 output columns: [N/16][K][16].

#include "common.h"
#include "kern/vv.h"

namespace {

const float *fptr(const torch::Tensor &t) { return t.numel() ? t.data_ptr<float>() : nullptr; }

kern::Panels make_panels(const torch::Tensor &f32, const torch::Tensor &q, const torch::Tensor &sc, const torch::Tensor &mn) {
    kern::Panels w;
    if (f32.numel()) {
        TORCH_CHECK(f32.dim() == 3 && f32.size(2) == 16 && f32.is_contiguous() && f32.scalar_type() == torch::kFloat32);
        w.f32 = f32.data_ptr<float>();
        w.K = f32.size(1);
        w.N = f32.size(0) * 16;
    } else {
        TORCH_CHECK(q.dim() == 3 && q.size(2) == 16 && q.is_contiguous() && q.scalar_type() == torch::kInt8);
        TORCH_CHECK(q.size(1) % kern::QG == 0 && sc.is_contiguous() && mn.is_contiguous() && sc.size(1) == q.size(1) / kern::QG);
        w.q = q.data_ptr<int8_t>();
        w.sc = sc.data_ptr<float>();
        w.mn = mn.data_ptr<float>();
        w.K = q.size(1);
        w.N = q.size(0) * 16;
    }
    return w;
}

// The mixer weights of a block (eps, norms, depthwise conv, gammas).
kern::VVBlock make_block(const torch::Tensor &x, const torch::Tensor &norm_w, const torch::Tensor &dw,
                         const torch::Tensor &dwb, const torch::Tensor &gamma, const torch::Tensor &ffn_norm_w, double eps) {
    kern::VVBlock b;
    b.C = x.size(2);
    b.norm_w = norm_w.data_ptr<float>();
    b.dw = dw.data_ptr<float>();
    b.dwb = dwb.data_ptr<float>();
    b.gamma = gamma.data_ptr<float>();
    b.ffn_norm_w = ffn_norm_w.data_ptr<float>();
    b.eps = (float)eps;
    b.gelu = kern::EPI_GELU_AS;
    return b;
}

// Per-sequence row pointers into a [B][rows][C] tensor (empty: none).
std::vector<float *> seq_ptrs(const torch::Tensor &t, int64_t B) {
    std::vector<float *> p;
    if (t.numel())
        for (int64_t q = 0; q < B; q++) p.push_back(t.data_ptr<float>() + q * t.size(1) * t.size(2));
    return p;
}

// New conv state: the last DWK - 1 normed rows of [state; rows of x].
torch::Tensor next_state(const kern::VVBlock &b, const torch::Tensor &x, const torch::Tensor &state) {
    auto ns = torch::empty({x.size(0), kern::DWK - 1, b.C}, torch::kFloat32);
    const std::vector<float *> st = seq_ptrs(state, x.size(0)), np = seq_ptrs(ns, x.size(0));
    kern::vv_next_state(b, x.data_ptr<float>(), x.size(0), x.size(1), st.empty() ? nullptr : st.data(), np.data());
    return ns;
}

}  // namespace

// C = A @ B + bias on panels, one thread per tile of 48 rows (for benchmarks and tests).
torch::Tensor vv_gemm(torch::Tensor A, torch::Tensor Bp, torch::Tensor bias) {
    const int64_t M = A.size(0), K = A.size(1), N = Bp.size(0) * 16;
    auto C = torch::empty({M, N}, torch::kFloat32);
    const float *ap = A.data_ptr<float>(), *bp = Bp.data_ptr<float>(), *bias_p = fptr(bias);
    float *cp = C.data_ptr<float>();
    at::parallel_for(0, (M + 47) / 48, 1, [&](int64_t t0, int64_t t1) {
        for (int64_t t = t0; t < t1; t++) {
            const int64_t m0 = t * 48, n = std::min<int64_t>(48, M - m0);
            kern::gemm_f32_tile(kern::Rows(ap + m0 * K, K), n, bp, K, N / 16, bias_p, cp + m0 * N, N, kern::Epi());
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
    const kern::Panels W = make_panels(wf, wq, wsc, wmn);
    TORCH_CHECK(W.K == (H + s) * Cin, "window size does not match the weights");
    auto out = torch::empty({B, J, W.N}, torch::kFloat32);
    const float *bp = fptr(bias);
    const kern::Par par = torch_par();
    const int64_t head = std::min(J, (H + s - 1) / s);  // output rows that reach into prev
    for (int64_t b = 0; b < B; b++) {
        const float *xb = x.data_ptr<float>() + b * T * Cin;
        float *ob = out.data_ptr<float>() + b * J * W.N;
        if (head > 0) {  // [prev; first rows] gathered for the first output rows
            std::vector<float> tmp((H + head * s) * Cin, 0.f);
            if (prev.numel()) std::memcpy(tmp.data(), prev.data_ptr<float>() + b * H * Cin, H * Cin * sizeof(float));
            std::memcpy(tmp.data() + H * Cin, xb, std::min(T, head * s) * Cin * sizeof(float));
            kern::gemm_f32(par, kern::Rows(tmp.data(), s * Cin), head, W, bp, ob, W.N, kern::Epi(), 96, 1);
        }
        kern::gemm_f32(par, kern::Rows(xb + (head * s - H) * Cin, s * Cin), J - head, W, bp, ob + head * W.N, W.N,
                       kern::Epi(), 96, 1);
    }
    return out;
}

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
                                    torch::Tensor ffn_gamma, double eps) {
    TORCH_CHECK(x.is_contiguous() && x.scalar_type() == torch::kFloat32 && x.dim() == 3);
    const int64_t B = x.size(0), T = x.size(1), C = x.size(2), H = 4 * C;
    TORCH_CHECK(C % 16 == 0 && w1p.size(0) == H / 16 && w1p.size(1) == C && w2p.size(0) == C / 16 && w2p.size(1) == H);
    TORCH_CHECK(state.numel() == 0 || (state.is_contiguous() && state.size(0) == B && state.size(1) == kern::DWK - 1));
    kern::VVBlock b = make_block(x, norm_w, dw, dwb, gamma, ffn_norm_w, eps);
    b.H = H;
    b.w1.f32 = w1p.data_ptr<float>(); b.w1.K = C; b.w1.N = H;
    b.w2.f32 = w2p.data_ptr<float>(); b.w2.K = H; b.w2.N = C;
    b.b1 = b1.data_ptr<float>();
    b.b2 = b2.data_ptr<float>();
    b.ffn_gamma = ffn_gamma.data_ptr<float>();
    auto out = torch::empty_like(x);
    const std::vector<float *> st = seq_ptrs(state, B);
    kern::vv_block(torch_par(), b, x.data_ptr<float>(), B, T, st.empty() ? nullptr : st.data(), out.data_ptr<float>());
    return {out, next_state(b, x, state)};
}

// Mixer half of a wide Block1D: returns (x1, h2 = rms(x1) * ffn_norm_w, next state).
std::vector<torch::Tensor> vv_mixer(torch::Tensor x, torch::Tensor state, torch::Tensor norm_w, torch::Tensor dw,
                                    torch::Tensor dwb, torch::Tensor gamma, torch::Tensor ffn_norm_w, double eps) {
    TORCH_CHECK(x.is_contiguous() && x.scalar_type() == torch::kFloat32 && x.dim() == 3);
    TORCH_CHECK(x.size(2) % 8 == 0 && (state.numel() == 0 || (state.is_contiguous() && state.size(0) == x.size(0))));
    const kern::VVBlock b = make_block(x, norm_w, dw, dwb, gamma, ffn_norm_w, eps);
    auto x1 = torch::empty_like(x), h2 = torch::empty_like(x);
    const std::vector<float *> st = seq_ptrs(state, x.size(0));
    kern::vv_mixer(torch_par(), b, x.data_ptr<float>(), x.size(0), x.size(1), st.empty() ? nullptr : st.data(),
                   x1.data_ptr<float>(), h2.data_ptr<float>());
    return {x1, h2, next_state(b, x, state)};
}

// Wide Block1D in one call: mixer rows, then the FFN as panel GEMMs over all rows (GELU and
// the gamma residual in the epilogues). FFN weights fp32 or int8 panels.
std::vector<torch::Tensor> vv_block_wide(torch::Tensor x, torch::Tensor state, torch::Tensor norm_w, torch::Tensor dw,
                                         torch::Tensor dwb, torch::Tensor gamma, torch::Tensor ffn_norm_w,
                                         torch::Tensor w1f, torch::Tensor w1q, torch::Tensor w1s, torch::Tensor w1m,
                                         torch::Tensor b1, torch::Tensor w2f, torch::Tensor w2q, torch::Tensor w2s,
                                         torch::Tensor w2m, torch::Tensor b2, torch::Tensor ffn_gamma, double eps) {
    auto mixed = vv_mixer(x, state, norm_w, dw, dwb, gamma, ffn_norm_w, eps);
    const int64_t M = x.size(0) * x.size(1), C = x.size(2);
    const kern::Panels W1 = make_panels(w1f, w1q, w1s, w1m), W2 = make_panels(w2f, w2q, w2s, w2m);
    TORCH_CHECK(W1.K == C && W2.K == W1.N && W2.N == C);
    auto u = torch::empty({M, W1.N}, torch::kFloat32);
    auto out = torch::empty_like(x);
    const kern::Par par = torch_par();
    kern::Epi gelu;
    gelu.op = kern::EPI_GELU_AS;
    kern::gemm_f32(par, kern::Rows(mixed[1].data_ptr<float>(), C), M, W1, b1.data_ptr<float>(), u.data_ptr<float>(), W1.N,
                   gelu, 96, 1);
    kern::Epi res;
    res.op = kern::EPI_GADD;
    res.r = mixed[0].data_ptr<float>();
    res.ldr = C;
    res.g = ffn_gamma.data_ptr<float>();
    kern::gemm_f32(par, kern::Rows(u.data_ptr<float>(), W1.N), M, W2, b2.data_ptr<float>(), out.data_ptr<float>(), C, res,
                   96, 1);
    return {out, mixed[2]};
}

void vv_gelu_(torch::Tensor u) {
    TORCH_CHECK(u.is_contiguous() && u.scalar_type() == torch::kFloat32 && u.size(-1) % 8 == 0);
    const int64_t n = u.size(-1), rows = u.numel() / n;
    float *p = u.data_ptr<float>();
    at::parallel_for(0, rows, 16, [&](int64_t r0, int64_t r1) {
        for (int64_t r = r0; r < r1; r++) kern::gelu_as(p + r * n, n);
    });
}

// Causal stride-1 conv with few channels: y[t] = b + sum_j x[t - K + 1 + j] @ W[j].
// x [B, T, Cin]; state [B, K-1, Cin] or empty; w [K, Cin, Cout]. Cout % 8 == 0 or Cout == 1 (Cin % 8 == 0).
torch::Tensor vv_conv_small(torch::Tensor x, torch::Tensor state, torch::Tensor w, torch::Tensor bias) {
    TORCH_CHECK(x.is_contiguous() && x.scalar_type() == torch::kFloat32 && x.dim() == 3);
    const int64_t B = x.size(0), T = x.size(1), Cin = x.size(2), K = w.size(0), Cout = w.size(2);
    TORCH_CHECK(w.size(1) == Cin && (Cout % 8 == 0 || (Cout == 1 && Cin % 8 == 0)));
    auto y = torch::empty({B, T, Cout}, torch::kFloat32);
    const float *xp = x.data_ptr<float>(), *sp = fptr(state), *wp = w.data_ptr<float>(), *bp = bias.data_ptr<float>();
    float *yp = y.data_ptr<float>();
    at::parallel_for(0, B * T, 256, [&](int64_t i0, int64_t i1) {
        for (int64_t i = i0; i < i1;) {  // runs of rows of one sequence
            const int64_t b = i / T, t0 = i % T, t1 = std::min(T, t0 + (i1 - i));
            kern::conv_small(xp + b * T * Cin, sp ? sp + b * (K - 1) * Cin : nullptr, K, Cin, Cout, wp, bp, t0, t1,
                             yp + b * T * Cout);
            i += t1 - t0;
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
