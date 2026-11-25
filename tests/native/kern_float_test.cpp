// The float kernels of app/native/kern (act.h, gemm_f32.h, vv.h, detect.h) against scalar references, on every ISA
// this CPU has (AVX2; AVX-512 too, e.g. under Intel SDE), and the ISAs against each other (bit-identical by design).
// Error measure: |y - ref| / (sum of |terms| of ref) for sums; for exp-based functions relative to what rounding their
// argument alone may cost (exp(u) has relative condition |u|); limit 1e-6.
//   g++ -O2 -std=c++17 -I app/native -I flash/engine tests/native/kern_float_test.cpp -o kern_float_test
#include <array>
#include <cstdio>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "kern/detect.h"
#include "kern/vv.h"
#include "web_kernels.h"

static int g_fail = 0, g_checks = 0;
static std::mt19937 g_rng(1);
static std::vector<float> rnd(size_t n, float lo = -1.f, float hi = 1.f) {
    std::uniform_real_distribution<float> d(lo, hi);
    std::vector<float> v(n);
    for (auto &x : v) x = d(g_rng);
    return v;
}
// y vs ref, errors scaled by scale[i] (or |ref| + floor)
static void check(const std::string &what, const float *y, const double *ref, const double *scale, size_t n,
                  double tol = 1e-6, double floor = 1e-30) {
    double worst = 0;
    size_t at = 0;
    for (size_t i = 0; i < n; i++) {
        const double s = scale ? scale[i] : std::fabs(ref[i]) + floor, e = std::fabs((double)y[i] - ref[i]) / std::max(s, 1e-300);
        if (!(e <= worst)) { worst = e; at = i; }
    }
    g_checks++;
    const bool ok = worst <= tol;
    if (!ok) g_fail++;
    if (!ok || getenv("VERBOSE"))
        printf("%-4s %-44s err %.2e at %zu (y %.9g ref %.9g)\n", ok ? "ok" : "FAIL", what.c_str(), worst, at,
               n ? y[at] : 0.f, n ? ref[at] : 0.);
}
static void same(const std::string &what, const std::vector<float> &a, const std::vector<float> &b) {
    g_checks++;
    if (a.size() == b.size() && !memcmp(a.data(), b.data(), a.size() * 4)) return;
    g_fail++;
    printf("FAIL %-44s results differ\n", what.c_str());
}

// Runs f(isa) for each ISA present (0 AVX2, 1 AVX-512), checks each result with chk, and the ISAs against each other.
static void each_isa(const std::string &what, const std::function<std::vector<float>(int)> &f,
                     const std::function<void(const std::string &, const std::vector<float> &)> &chk) {
    const std::vector<float> a = f(0);
    chk(what + " avx2", a);
    if (kern::has_avx512f()) {
        const std::vector<float> b = f(1);
        chk(what + " avx512", b);
        same(what, a, b);
    }
}
#define ISA_CALL(isa, fn, ...) ((isa) ? kern::avx512::fn(__VA_ARGS__) : kern::avx2::fn(__VA_ARGS__))

// ---------------------------------------------------------------- scalar references
static double gelu_as_ref(float x0) {  // the A-S formula in float, as ATen
    const float x = x0 * 0.70710678118654752f, ax = std::fabs(x), t = 1.f / (1.f + 0.3275911f * ax);
    const float r = (((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f;
    const float e = 1.f - r * t * std::exp(-x * x);
    return 0.5 * x0 * (1.0 + (x < 0 ? -e : e));
}
static double mish_ref(float v) { return v * std::tanh(v > 20.f ? v : std::log1p(std::exp(v))); }  // fm.cpp mish1
static void layernorm_ref(const float *x, const float *w, const float *b, int64_t n, double *out) {
    double mean = 0, var = 0;
    for (int64_t i = 0; i < n; i++) mean += x[i];
    mean /= (double)n;
    for (int64_t i = 0; i < n; i++) var += (x[i] - mean) * (x[i] - mean);
    const double r = 1.0 / std::sqrt(var / (double)n + 1e-5);
    for (int64_t i = 0; i < n; i++) out[i] = w ? (x[i] - mean) * r * w[i] + b[i] : (x[i] - mean) * r;
}
static void rms_ref(const float *x, const float *w, int64_t n, double eps, double *out) {
    double ss = 0;
    for (int64_t i = 0; i < n; i++) ss += (double)x[i] * x[i];
    const double r = 1.0 / std::sqrt(ss / n + eps);
    for (int64_t i = 0; i < n; i++) out[i] = x[i] * r * (w ? w[i] : 1.0);
}

// ---------------------------------------------------------------- tests
static void test_elementwise() {
    for (int64_t n : {1, 7, 8, 15, 16, 17, 33, 1000, 1024}) {
        std::vector<float> x = rnd(n, -12.f, 12.f);
        if (n > 8) { x[0] = 0.f; x[1] = 1e-20f; x[2] = -1e-20f; x[3] = 40.f; x[4] = -30.f; x[5] = 25.f; x[6] = -0.5f; }
        std::vector<double> g(n), ga(n), m(n), s(n), gs(n), gas(n), ms(n), ss(n);
        const std::vector<float> b = rnd(n);
        for (int64_t i = 0; i < n; i++) {
            const double a = std::fabs(x[i]);
            g[i] = web_gelu1(x[i]);
            ga[i] = gelu_as_ref(x[i]);
            m[i] = mish_ref(x[i]);
            s[i] = (double)x[i] / (1.0 + std::exp(-(double)x[i])) * b[i];
            gs[i] = std::fabs(g[i]) * (1 + a * a / 2) + 1e-30;                  // exp(-x^2 / 2)
            gas[i] = std::fabs(ga[i]) * (1 + a * a / 2) + 0.5 * a + 1e-30;      // 0.5 x + 0.5 x erf(x): two terms
            ms[i] = std::fabs(m[i]) * (1 + a) + 1e-30;                          // exp(-|x|)
            ss[i] = std::fabs(s[i]) * (1 + a) + 1e-30;
        }
        const std::string sz = " n=" + std::to_string(n);
        auto run = [&](void (*a2)(float *, int64_t), void (*a5)(float *, int64_t)) {
            return [&, a2, a5](int isa) { std::vector<float> y = x; (isa ? a5 : a2)(y.data(), n); return y; };
        };
        each_isa("gelu" + sz, run(kern::avx2::gelu, kern::avx512::gelu),
                 [&](const std::string &w, const std::vector<float> &y) { check(w, y.data(), g.data(), gs.data(), n); });
        each_isa("gelu_as" + sz, run(kern::avx2::gelu_as, kern::avx512::gelu_as),
                 [&](const std::string &w, const std::vector<float> &y) { check(w, y.data(), ga.data(), gas.data(), n); });
        each_isa("mish" + sz, run(kern::avx2::mish, kern::avx512::mish),
                 [&](const std::string &w, const std::vector<float> &y) { check(w, y.data(), m.data(), ms.data(), n); });
        each_isa("silu_mul" + sz, [&](int isa) { std::vector<float> y = x; ISA_CALL(isa, silu_mul, y.data(), b.data(), n); return y; },
                 [&](const std::string &w, const std::vector<float> &y) { check(w, y.data(), s.data(), ss.data(), n); });
    }
}

static void test_norms() {
    for (int64_t n : {8, 13, 32, 64, 128, 1024, 2048}) {
        const std::vector<float> x = rnd(n, -3.f, 3.f), w = rnd(n, 0.5f, 1.5f), b = rnd(n);
        for (int wi = 0; wi < 2; wi++) {
            std::vector<double> r(n);
            rms_ref(x.data(), wi ? w.data() : nullptr, n, 1e-6, r.data());
            each_isa("rms" + std::string(wi ? "w" : "") + " n=" + std::to_string(n),
                     [&](int isa) { std::vector<float> y(n); ISA_CALL(isa, rms, x.data(), wi ? w.data() : nullptr, n, 1e-6f, y.data()); return y; },
                     [&](const std::string &s, const std::vector<float> &y) { check(s, y.data(), r.data(), nullptr, n); });
            if (n % 8) continue;
            std::vector<double> l(n);
            layernorm_ref(x.data(), wi ? w.data() : nullptr, b.data(), n, l.data());
            double mean = 0, var = 0;
            for (int64_t i = 0; i < n; i++) mean += x[i];
            mean /= n;
            for (int64_t i = 0; i < n; i++) var += (x[i] - mean) * (x[i] - mean);
            const double rs = 1.0 / std::sqrt(var / n + 1e-5);
            std::vector<double> sc(n);  // |terms|: (x - mean) r w + b
            for (int64_t i = 0; i < n; i++) sc[i] = (std::fabs(x[i]) + std::fabs(mean)) * rs * (wi ? std::fabs(w[i]) : 1.0) + (wi ? std::fabs(b[i]) : 0.0);
            each_isa("layernorm" + std::string(wi ? " affine" : "") + " n=" + std::to_string(n),
                     [&](int isa) { std::vector<float> y(n); ISA_CALL(isa, layernorm, x.data(), wi ? w.data() : nullptr, wi ? b.data() : nullptr, n, 1e-5f, y.data()); return y; },
                     [&](const std::string &s, const std::vector<float> &y) { check(s, y.data(), l.data(), sc.data(), n); });
        }
    }
    const std::vector<float> v = rnd(128), cs = rnd(64), sn = rnd(64);
    std::vector<double> r(128), sc(128);
    for (int i = 0; i < 64; i++) {
        r[i] = (double)v[i] * cs[i] - (double)v[i + 64] * sn[i];
        r[i + 64] = (double)v[i + 64] * cs[i] + (double)v[i] * sn[i];
        sc[i] = sc[i + 64] = std::fabs(v[i] * cs[i]) + std::fabs(v[i + 64] * sn[i]) + std::fabs(v[i + 64] * cs[i]) + std::fabs(v[i] * sn[i]);
    }
    each_isa("rope", [&](int isa) { std::vector<float> y = v; ISA_CALL(isa, rope, y.data(), cs.data(), sn.data(), 64); return y; },
             [&](const std::string &s, const std::vector<float> &y) { check(s, y.data(), r.data(), sc.data(), 128); });
}

static void test_linear_rows() {
    for (auto t : std::vector<std::array<int64_t, 3>>{{1024, 16, 1}, {16, 1024, 1}, {64, 1024, 2}, {1, 1024, 1}, {37, 20, 3}, {6, 5, 2}}) {
        const int64_t N = t[0], K = t[1], M = t[2], xs = K + 3, ys = N + 5;
        const std::vector<float> W = rnd(N * K), bias = rnd(N), x = rnd(M * xs);
        std::vector<double> r(M * ys, 0.0), sc(M * ys, 1.0);
        for (int64_t m = 0; m < M; m++)
            for (int64_t n = 0; n < N; n++) {
                double a = bias[n], s = std::fabs(bias[n]);
                for (int64_t k = 0; k < K; k++) { a += (double)W[n * K + k] * x[m * xs + k]; s += std::fabs(W[n * K + k] * x[m * xs + k]); }
                r[m * ys + n] = a; sc[m * ys + n] = s;
            }
        each_isa("linear_rows " + std::to_string(N) + "x" + std::to_string(K) + " M=" + std::to_string(M),
                 [&](int isa) { std::vector<float> y(M * ys, 0.f); ISA_CALL(isa, linear_rows, W.data(), bias.data(), K, x.data(), xs, M, y.data(), ys, 0, N); return y; },
                 [&](const std::string &s, const std::vector<float> &y) { check(s, y.data(), r.data(), sc.data(), M * ys); });
    }
}

static const char *epi_name[] = {"", " gelu", " gelu_as", " mish", " add", " gadd"};
static void test_gemm() {
    struct Sh { int64_t M, K, N, lda; };
    const std::vector<Sh> shapes = {{200, 256, 1024, 256}, {200, 1024, 256, 1024}, {800, 128, 512, 128}, {800, 512, 128, 512},
                                    {1600, 64, 256, 64}, {1600, 256, 64, 256}, {3200, 32, 128, 32}, {3200, 128, 32, 128},
                                    {1, 448, 2048, 64}, {800, 256, 128, 128}, {1600, 128, 64, 64}, {13, 40, 48, 40}, {7, 16, 16, 3}};
    for (const Sh &s : shapes) {
        const int64_t rows_in = (s.M - 1) * s.lda + s.K, ldc = s.N + 16;
        const std::vector<float> A = rnd(rows_in), W = rnd(s.N * s.K, -0.2f, 0.2f), bias = rnd(s.N), R = rnd(s.M * ldc), g = rnd(s.N);
        std::vector<float> Wp(W.size());
        kern::pack_panels(W.data(), s.N, s.K, false, Wp.data());
        std::vector<double> acc(s.M * ldc, 0.0), mag(s.M * ldc, 1.0);
        for (int64_t m = 0; m < s.M; m++)
            for (int64_t n = 0; n < s.N; n++) {
                double a = bias[n], t = std::fabs(bias[n]);
                for (int64_t k = 0; k < s.K; k++) { const double p = (double)A[m * s.lda + k] * W[n * s.K + k]; a += p; t += std::fabs(p); }
                acc[m * ldc + n] = a; mag[m * ldc + n] = t;
            }
        for (int op = 0; op <= kern::EPI_GADD; op++) {
            if (s.M > 1000 && op != kern::EPI_ADD && op != kern::EPI_GELU) continue;
            std::vector<double> r(s.M * ldc), sc(s.M * ldc);
            for (int64_t i = 0; i < s.M * ldc; i++) {
                const int64_t n = i % ldc;
                const double a = acc[i];
                if (n >= s.N) { r[i] = R[i]; sc[i] = 1; continue; }  // untouched columns
                switch (op) {
                    case kern::EPI_GELU: r[i] = web_gelu1((float)a); sc[i] = mag[i]; break;
                    case kern::EPI_GELU_AS: r[i] = gelu_as_ref((float)a); sc[i] = mag[i]; break;
                    case kern::EPI_MISH: r[i] = mish_ref((float)a); sc[i] = mag[i]; break;
                    case kern::EPI_ADD: r[i] = R[i] + a; sc[i] = std::fabs(R[i]) + mag[i]; break;
                    case kern::EPI_GADD: r[i] = R[i] + g[n] * a; sc[i] = std::fabs(R[i]) + std::fabs(g[n]) * mag[i]; break;
                    default: r[i] = a; sc[i] = mag[i];
                }
            }
            kern::Epi e;
            e.op = op;
            if (op >= kern::EPI_ADD) { e.r = R.data(); e.ldr = ldc; e.g = g.data(); }
            const std::string what = "gemm M=" + std::to_string(s.M) + " K=" + std::to_string(s.K) + " N=" + std::to_string(s.N) + epi_name[op];
            // GELU / Mish on an inexact argument: 2e-6 of the sum (their slope is at most 1.13)
            const double tol = op >= kern::EPI_GELU && op <= kern::EPI_MISH ? 2e-6 : 1e-6;
            each_isa(what, [&](int isa) {
                         std::vector<float> C = R;  // in place for ADD / GADD (C = R)
                         ISA_CALL(isa, gemm_tile, kern::Rows(A.data(), s.lda), s.M, Wp.data(), s.K, s.N / 16, bias.data(), C.data(), ldc, e);
                         return C;
                     },
                     [&](const std::string &w, const std::vector<float> &y) { check(w, y.data(), r.data(), sc.data(), s.M * ldc, tol); });
        }
        // the parallel driver (inline here), int8 panels: same as their dequantized fp32 panels
        if (s.K % kern::QG == 0 && s.M <= 200) {
            const int64_t P = s.N / 16, G = s.K / kern::QG;
            std::vector<int8_t> q(s.N * s.K);
            std::vector<float> sc = rnd(P * G * 16, 0.001f, 0.01f), mn = rnd(P * G * 16, -0.1f, 0.1f), deq(s.N * s.K);
            for (auto &c : q) c = (int8_t)(g_rng() % 256 - 128);
            kern::Panels Q;
            Q.q = q.data(); Q.sc = sc.data(); Q.mn = mn.data(); Q.K = s.K; Q.N = s.N;
            for (int64_t p = 0; p < P; p++)
                for (int64_t k = 0; k < s.K; k++)
                    for (int j = 0; j < 16; j++) {
                        const int64_t gi = (p * G + k / kern::QG) * 16 + j;
                        deq[(p * s.K + k) * 16 + j] = std::fma((float)q[(p * s.K + k) * 16 + j], sc[gi], mn[gi]);
                    }
            kern::Panels F;
            F.f32 = deq.data(); F.K = s.K; F.N = s.N;
            std::vector<float> c1(s.M * ldc, 0.f), c2(s.M * ldc, 0.f);
            kern::gemm_f32(kern::Par(), kern::Rows(A.data(), s.lda), s.M, Q, bias.data(), c1.data(), ldc);
            kern::gemm_f32(kern::Par(), kern::Rows(A.data(), s.lda), s.M, F, bias.data(), c2.data(), ldc, kern::Epi(), 7, 1);
            same("gemm int8 panels = dequantized M=" + std::to_string(s.M) + " K=" + std::to_string(s.K), c1, c2);
        }
    }
}

// The block in double: reference of vv_block / vv_mixer / vv_next_state.
struct BlockRef {
    int64_t C, H, T;
    std::vector<float> x, st, nw, dw, dwb, g, fnw, w1, b1, w2, b2, fg;  // w1 [H][C], w2 [C][H] (rows = outputs)
    bool fold;  // engine flavour: no norm weights / gammas
    std::vector<double> x1, h2, out, ns, sc_out, sc_x1, sc_h2;
    void run(int gelu) {
        const int64_t R = 6;
        std::vector<double> h((T + R) * C);
        for (int64_t t = -R; t < T; t++) {
            double *d = h.data() + (t + R) * C;
            if (t < 0) { for (int64_t c = 0; c < C; c++) d[c] = st.empty() ? 0.0 : st[(t + R) * C + c]; continue; }
            rms_ref(x.data() + t * C, fold ? nullptr : nw.data(), C, 1e-5, d);
        }
        x1.assign(T * C, 0); h2.assign(T * C, 0); out.assign(T * C, 0); sc_out.assign(T * C, 0); sc_x1.assign(T * C, 0);
        std::vector<float> x1f(T * C);
        for (int64_t t = 0; t < T; t++)
            for (int64_t c = 0; c < C; c++) {
                double a = dwb[c], s = std::fabs(dwb[c]);
                for (int j = 0; j < 7; j++) { a += (double)dw[j * C + c] * h[(t + j) * C + c]; s += std::fabs(dw[j * C + c] * h[(t + j) * C + c]); }
                const double gm = fold ? 1.0 : g[c];
                x1[t * C + c] = x[t * C + c] + gm * a;
                sc_x1[t * C + c] = std::fabs(x[t * C + c]) + std::fabs(gm) * s;
                x1f[t * C + c] = (float)x1[t * C + c];
            }
        sc_h2.assign(T * C, 0);
        for (int64_t t = 0; t < T; t++) {
            rms_ref(x1f.data() + t * C, fold ? nullptr : fnw.data(), C, 1e-5, h2.data() + t * C);
            double ssq = 0;
            for (int64_t c = 0; c < C; c++) ssq += (double)x1f[t * C + c] * x1f[t * C + c];
            const double r = 1.0 / std::sqrt(ssq / C + 1e-5);
            for (int64_t c = 0; c < C; c++) sc_h2[t * C + c] = sc_x1[t * C + c] * r * (fold ? 1.0 : fnw[c]);
        }
        std::vector<float> u(H);
        for (int64_t t = 0; t < T; t++) {
            double su = 0;
            for (int64_t f = 0; f < H; f++) {
                double a = b1[f];
                for (int64_t c = 0; c < C; c++) a += w1[f * C + c] * h2[t * C + c];
                u[f] = gelu == kern::EPI_GELU ? web_gelu1((float)a) : (float)gelu_as_ref((float)a);
                su += std::fabs(u[f]);
            }
            for (int64_t c = 0; c < C; c++) {
                double a = b2[c], s = std::fabs(b2[c]);
                for (int64_t f = 0; f < H; f++) { a += (double)w2[c * H + f] * u[f]; s += std::fabs(w2[c * H + f] * u[f]); }
                const double gm = fold ? 1.0 : fg[c];
                out[t * C + c] = x1[t * C + c] + gm * a;
                // the GELU inputs carry ~1e-7 relative errors: their effect is bounded by the |w2| . |u| terms
                sc_out[t * C + c] = sc_x1[t * C + c] + std::fabs(gm) * (s + 1e-30);
            }
        }
        ns.assign(R * C, 0);
        for (int64_t i = 0; i < R * C; i++) ns[i] = h[(T + i / C) * C + i % C];
    }
};

static void test_vv() {
    struct Sh { int64_t C, T; bool fold; int gelu; };
    for (const Sh &s : std::vector<Sh>{{32, 50, true, kern::EPI_GELU}, {64, 37, true, kern::EPI_GELU}, {256, 40, true, kern::EPI_GELU},
                                       {128, 3, false, kern::EPI_GELU_AS}, {16, 20, false, kern::EPI_GELU_AS}, {48, 9, true, kern::EPI_GELU}}) {
        for (int with_state = 0; with_state < 2; with_state++) {
            BlockRef r;
            r.C = s.C; r.H = 4 * s.C; r.T = s.T; r.fold = s.fold;
            r.x = rnd(s.T * s.C, -2.f, 2.f);
            if (with_state) r.st = rnd(6 * s.C);
            r.nw = rnd(s.C, .5f, 1.5f); r.dw = rnd(7 * s.C, -.3f, .3f); r.dwb = rnd(s.C, -.1f, .1f); r.g = rnd(s.C);
            r.fnw = rnd(s.C, .5f, 1.5f); r.w1 = rnd(r.H * s.C, -.1f, .1f); r.b1 = rnd(r.H, -.1f, .1f);
            r.w2 = rnd(s.C * r.H, -.1f, .1f); r.b2 = rnd(s.C, -.1f, .1f); r.fg = rnd(s.C);
            r.run(s.gelu);
            std::vector<float> p1(r.w1.size()), p2(r.w2.size());
            kern::pack_panels(r.w1.data(), r.H, s.C, false, p1.data());
            kern::pack_panels(r.w2.data(), s.C, r.H, false, p2.data());
            kern::VVBlock b;
            b.C = s.C; b.H = r.H; b.eps = 1e-5f; b.gelu = s.gelu;
            b.dw = r.dw.data(); b.dwb = r.dwb.data(); b.b1 = r.b1.data(); b.b2 = r.b2.data();
            if (!s.fold) { b.norm_w = r.nw.data(); b.gamma = r.g.data(); b.ffn_norm_w = r.fnw.data(); b.ffn_gamma = r.fg.data(); }
            b.w1.f32 = p1.data(); b.w1.K = s.C; b.w1.N = r.H;
            b.w2.f32 = p2.data(); b.w2.K = r.H; b.w2.N = s.C;
            const float *st0 = with_state ? r.st.data() : nullptr, *const st[1] = {st0};
            const std::string what = "vv C=" + std::to_string(s.C) + " T=" + std::to_string(s.T) + (s.fold ? " folded" : " weighted") +
                                     (with_state ? " state" : "");
            // GELU inputs are inexact (~1e-7 relative): block outputs within 2e-6 of their terms
            for (int64_t tt : {6, 12, 0}) {
                std::vector<float> o(s.T * s.C);
                kern::vv_block(kern::Par(), b, r.x.data(), 1, s.T, st, o.data(), tt);
                check(what + " block tt=" + std::to_string(tt), o.data(), r.out.data(), r.sc_out.data(), o.size(), 2e-6);
            }
            std::vector<float> x1(s.T * s.C), h2(s.T * s.C), ns(6 * s.C);
            float *const nsp[1] = {ns.data()};
            kern::vv_mixer(kern::Par(), b, r.x.data(), 1, s.T, st, x1.data(), h2.data(), 5);
            check(what + " mixer x1", x1.data(), r.x1.data(), r.sc_x1.data(), x1.size());
            check(what + " mixer h2", h2.data(), r.h2.data(), r.sc_h2.data(), h2.size());
            kern::vv_next_state(b, r.x.data(), 1, s.T, st, nsp);
            check(what + " next state", ns.data(), r.ns.data(), nullptr, ns.size(), 1e-6, 1e-6);
        }
    }
    // causal convolution with few channels (the head: cout 1)
    for (auto t : std::vector<std::array<int64_t, 4>>{{300, 32, 7, 1}, {20, 8, 7, 1}, {50, 1, 7, 32}, {9, 4, 3, 16}}) {
        const int64_t T = t[0], cin = t[1], k = t[2], cout = t[3];
        const std::vector<float> x = rnd(T * cin), st = rnd((k - 1) * cin), w = rnd(k * cin * cout), bias = rnd(cout);
        std::vector<double> r(T * cout), sc(T * cout);
        for (int64_t tt = 0; tt < T; tt++)
            for (int64_t o = 0; o < cout; o++) {
                double a = bias[o], s = std::fabs(bias[o]);
                for (int64_t j = 0; j < k; j++) {
                    const int64_t ti = tt - k + 1 + j;
                    const float *row = ti >= 0 ? x.data() + ti * cin : st.data() + (ti + k - 1) * cin;
                    for (int64_t c = 0; c < cin; c++) { a += (double)row[c] * w[(j * cin + c) * cout + o]; s += std::fabs(row[c] * w[(j * cin + c) * cout + o]); }
                }
                r[tt * cout + o] = a; sc[tt * cout + o] = s;
            }
        each_isa("conv_small T=" + std::to_string(T) + " cin=" + std::to_string(cin) + " cout=" + std::to_string(cout),
                 [&](int isa) { std::vector<float> y(T * cout); ISA_CALL(isa, conv_small_rows, x.data(), st.data(), k, cin, cout, w.data(), bias.data(), 0, T, y.data()); return y; },
                 [&](const std::string &s, const std::vector<float> &y) { check(s, y.data(), r.data(), sc.data(), y.size()); });
    }
}

static void test_detect() {
    for (auto t : std::vector<std::array<int, 2>>{{600, 343}, {600, 63}, {50, 10}, {17, 0}, {64, 70}}) {
        const int win = t[0], tmax = t[1];
        const std::vector<float> x = rnd(win + tmax);
        std::vector<double> r(tmax + 1), sc(tmax + 1);
        for (int tau = 0; tau <= tmax; tau++) {
            double a = 0, s = 0;
            for (int i = 0; i < win; i++) { a += (double)x[i] * x[i + tau]; s += std::fabs(x[i] * x[i + tau]); }
            r[tau] = a; sc[tau] = s + 1e-30;
        }
        each_isa("autocorr win=" + std::to_string(win) + " tmax=" + std::to_string(tmax),
                 [&](int isa) { std::vector<float> y(tmax + 1); ISA_CALL(isa, autocorr, x.data(), win, tmax, y.data()); return y; },
                 [&](const std::string &s, const std::vector<float> &y) { check(s, y.data(), r.data(), sc.data(), y.size()); });
    }
    // conv 5x5 on padded rows (P = 77 / 39 pitch rows), the head (5 dilated taps, one row)
    struct Sh { int ntaps, P, ci, co; bool holes; };
    for (const Sh &s : std::vector<Sh>{{25, 77, 2, 16, false}, {25, 39, 16, 24, true}, {25, 39, 24, 32, false}, {5, 1, 68, 32, true}, {9, 10, 3, 40, false}}) {
        const int kp = s.ntaps == 25 ? 5 : 1, ntime = s.ntaps / kp, rows = s.P + kp - 1;
        std::vector<std::vector<float>> in(ntime);
        for (auto &v : in) v = rnd((size_t)rows * s.ci);
        std::vector<const float *> taps(s.ntaps);
        for (int t = 0; t < s.ntaps; t++) taps[t] = s.holes && t / kp == 1 ? nullptr : in[t / kp].data() + (t % kp) * s.ci;
        const std::vector<float> w = rnd((size_t)s.ntaps * s.ci * s.co), bias = rnd(s.co);
        std::vector<double> r((size_t)s.P * s.co), sc((size_t)s.P * s.co);
        for (int p = 0; p < s.P; p++)
            for (int o = 0; o < s.co; o++) {
                double a = bias[o], m = std::fabs(bias[o]);
                for (int t = 0; t < s.ntaps; t++) {
                    if (!taps[t]) continue;
                    for (int i = 0; i < s.ci; i++) {
                        const double v = (double)taps[t][p * s.ci + i] * w[((size_t)t * s.ci + i) * s.co + o];
                        a += v; m += std::fabs(v);
                    }
                }
                r[p * s.co + o] = a; sc[p * s.co + o] = m;
            }
        each_isa("taps_conv taps=" + std::to_string(s.ntaps) + " P=" + std::to_string(s.P) + " ci=" + std::to_string(s.ci) + " co=" + std::to_string(s.co),
                 [&](int isa) { std::vector<float> y((size_t)s.P * s.co); ISA_CALL(isa, taps_conv, s.ntaps, taps.data(), s.ci, s.P, s.ci, s.co, w.data(), bias.data(), y.data()); return y; },
                 [&](const std::string &n, const std::vector<float> &y) { check(n, y.data(), r.data(), sc.data(), y.size()); });
    }
}

// Each output row the same bits whatever the batch: row i of M = 37 (strided or row pointers, any tiling) = the row
// alone; a batch of sequences = each sequence alone.
static void test_rows() {
    const int64_t K = 96, N = 64, M = 37;
    const std::vector<float> A = rnd(M * K), W = rnd(N * K, -.2f, .2f), bias = rnd(N), R = rnd(M * N);
    std::vector<float> Wp(W.size());
    kern::pack_panels(W.data(), N, K, false, Wp.data());
    kern::Panels P;
    P.f32 = Wp.data(); P.K = K; P.N = N;
    std::vector<const float *> rows(M);
    for (int64_t m = 0; m < M; m++) rows[m] = A.data() + ((m * 11) % M) * K;  // a permutation of the rows
    for (int op : {kern::EPI_NONE, kern::EPI_GELU, kern::EPI_ADD}) {
        kern::Epi e;
        e.op = op;
        std::vector<float> all = R, one(M * N);
        if (op == kern::EPI_ADD) { e.r = R.data(); e.ldr = N; }
        kern::gemm_f32(kern::Par(), kern::Rows(rows.data()), M, P, bias.data(), all.data(), N, e, 7, 1);
        for (int64_t m = 0; m < M; m++) {
            kern::Epi e1 = e;
            if (op == kern::EPI_ADD) e1.r = R.data() + m * N;
            std::copy(R.begin() + m * N, R.begin() + (m + 1) * N, one.begin() + m * N);
            kern::gemm_f32(kern::Par(), kern::Rows(rows[m], K), 1, P, bias.data(), one.data() + m * N, N, e1);
        }
        same(std::string("gemm rows: batch of 37 = rows alone") + epi_name[op], all, one);
    }
    // blocks: 3 sequences with their own states = one at a time
    const int64_t C = 32, H = 128, T = 20, B = 3;
    const std::vector<float> x = rnd(B * T * C), st = rnd(B * 6 * C), dw = rnd(7 * C, -.3f, .3f), dwb = rnd(C, -.1f, .1f);
    const std::vector<float> w1 = rnd(H * C, -.1f, .1f), b1 = rnd(H), w2 = rnd(C * H, -.1f, .1f), b2 = rnd(C);
    std::vector<float> p1(w1.size()), p2(w2.size());
    kern::pack_panels(w1.data(), H, C, false, p1.data());
    kern::pack_panels(w2.data(), C, H, false, p2.data());
    kern::VVBlock b;
    b.C = C; b.H = H; b.dw = dw.data(); b.dwb = dwb.data(); b.b1 = b1.data(); b.b2 = b2.data();
    b.w1.f32 = p1.data(); b.w1.K = C; b.w1.N = H; b.w2.f32 = p2.data(); b.w2.K = H; b.w2.N = C;
    const float *sts[3] = {st.data() + 12 * C, nullptr, st.data()};
    std::vector<float> out(B * T * C), sep(B * T * C), ns(B * 6 * C), nsep(B * 6 * C);
    float *nsp[3] = {ns.data(), ns.data() + 6 * C, ns.data() + 12 * C};
    kern::vv_block(kern::Par(), b, x.data(), B, T, sts, out.data(), 7);
    kern::vv_next_state(b, x.data(), B, T, sts, nsp);
    for (int64_t q = 0; q < B; q++) {
        float *n1[1] = {nsep.data() + q * 6 * C};
        kern::vv_block(kern::Par(), b, x.data() + q * T * C, 1, T, sts + q, sep.data() + q * T * C, 0);
        kern::vv_next_state(b, x.data() + q * T * C, 1, T, sts + q, n1);
    }
    same("vv_block: 3 sequences = each alone", out, sep);
    same("vv_next_state: 3 sequences = each alone", ns, nsep);
}

int main() {
    if (!kern::has_avx2()) { printf("no AVX2: nothing to test\n"); return 0; }
    printf("ISAs: AVX2%s\n", kern::has_avx512f() ? " AVX-512" : "");
    if (getenv("NEED_AVX512") && !kern::has_avx512f()) { printf("FAIL no AVX-512\n"); return 1; }
    test_elementwise();
    test_norms();
    test_linear_rows();
    test_gemm();
    test_vv();
    test_rows();
    test_detect();
    printf("%d checks, %d failed\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
