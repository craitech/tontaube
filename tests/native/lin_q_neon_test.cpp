// ARM lin_q checks: quant = ref bit for bit; dot / i8mm tiles = a scalar emulation of their float sequence with exact
// int64 dots (bit for bit); dot == i8mm; per-row identity (row alone, shuffled row pointers, other splits); vs double.
//   g++ -O2 -std=c++17 -fopenmp -I app/native tests/native/lin_q_neon_test.cpp (aarch64)
#include <omp.h>

#include <chrono>
#include <cstdio>
#include <random>

#include "kern/lin_q.h"

using namespace kern;
static std::mt19937 rng(7);

static void omp_run(void *, int64_t n, int64_t grain, RangeFn fn, void *ctx) {
    const int64_t chunk = std::max<int64_t>(grain, 1);
#pragma omp parallel for schedule(dynamic, 1)
    for (int64_t a = 0; a < n; a += chunk) fn(ctx, a, std::min(n, a + chunk));
}
static Par omp_par(int t) { Par p; p.run = omp_run; p.threads = t; return p; }
static void odd_run(void *, int64_t n, int64_t, RangeFn fn, void *ctx) {  // uneven serial split
    for (int64_t a = 0; a < n;) { const int64_t b = std::min(n, a + 1 + (a * 7) % 5); fn(ctx, a, b); a = b; }
}
static Par odd_par() { Par p; p.run = odd_run; p.threads = 3; return p; }

static uint16_t f2h(float f) { __fp16 v = (__fp16)f; uint16_t h; memcpy(&h, &v, 2); return h; }

struct Mat {
    int kind; int64_t N, K;
    std::vector<uint8_t> codes; std::vector<uint16_t> sc, mn; std::vector<float> bias;
    QW w(bool b) const { QW q; q.kind = kind; q.N = N; q.K = K; q.codes = codes.data(); q.scales = sc.data(); q.mins = mn.data(); q.bias = b ? bias.data() : nullptr; return q; }
};
static Mat rand_mat(int kind, int64_t N, int64_t K) {
    Mat m{kind, N, K, {}, {}, {}, {}};
    const int64_t G = K / (kind == 0 ? 64 : 128);
    m.codes.resize(N * K / (kind == 0 ? 2 : 1));
    for (auto &c : m.codes) c = rng() & 255;
    std::normal_distribution<float> nd(0.f, 1.f);
    m.sc.resize(N * G); m.mn.resize(N * G); m.bias.resize(N);
    for (auto &s : m.sc) s = f2h(0.01f * std::fabs(nd(rng)) + 1e-4f);
    for (auto &s : m.mn) s = f2h(0.05f * nd(rng));
    for (auto &b : m.bias) b = nd(rng);
    return m;
}
static int wcode(const Mat &W, int64_t n, int64_t k) {
    if (W.kind == 0) { const int64_t G = W.K / 64, g = k / 64, i = k % 64; const uint8_t b = W.codes[(n * G + g) * 32 + i % 32]; return i < 32 ? b & 15 : b >> 4; }
    return (int8_t)W.codes[n * W.K + k];
}
static float h2(uint16_t h) { __fp16 v; memcpy(&v, &h, 2); return v; }

// the ARM float sequence with exact dots
static float emulate(const Mat &W, const int16_t *q, const float *s, const float *xs, int64_t n, bool bias) {
    const int64_t GA = W.K / 64, gw = W.kind == 0 ? 1 : 2;
    float acc = 0.f, mt = 0.f;
    for (int64_t ga = 0; ga < GA; ga++) {
        int64_t d = 0;
        for (int k = 0; k < 64; k++) d += (int64_t)wcode(W, n, ga * 64 + k) * q[ga * 64 + k];
        const int64_t g = ga / gw;
        acc = fmaf((float)d, h2(W.sc[n * (GA / gw) + g]) * s[ga], acc);
        mt = fmaf(h2(W.mn[n * (GA / gw) + g]), xs[ga], mt);
    }
    return (acc + mt) + (bias ? W.bias[n] : 0.f);
}

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { fails++; if (fails < 20) { printf("FAIL " __VA_ARGS__); printf("\n"); } } } while (0)

static bool same(const float *a, const float *b, int64_t n) { return memcmp(a, b, n * 4) == 0; }

int main(int argc, char **argv) {
    printf("isa: dotprod %d i8mm %d sve2 %d; default %s\n", isa().dotprod, isa().i8mm, isa().sve2, lin_q_impl().name);
    std::normal_distribution<float> nd(0.f, 1.f);
    // quant: NEON = ref bit for bit (incl. a zero group and ties)
    for (int t = 0; t < 50; t++) {
        const int64_t M = 1 + rng() % 5, K = 64 * (1 + rng() % 40);
        std::vector<float> x(M * K);
        for (auto &v : x) v = nd(rng) * (t % 3 == 0 ? 1e-3f : 10.f);
        if (t == 1) std::fill(x.begin(), x.begin() + 64, 0.f);
        if (t == 2) for (int64_t i = 0; i < K; i++) x[i] = (float)(i % 7) - 3.f;
        std::vector<int16_t> q1(M * K), q2(M * K); std::vector<float> s1(M * K / 64), s2(s1), x1(s1), x2(s1);
        impl_scalar().quant(x.data(), K, 0, M, K, q1.data(), s1.data(), x1.data());
        impl_neon_dot().quant(x.data(), K, 0, M, K, q2.data(), s2.data(), x2.data());
        CHECK(q1 == q2 && same(s1.data(), s2.data(), s1.size()) && same(x1.data(), x2.data(), x1.size()), "quant t=%d", t);
    }
    std::vector<const LinQImpl *> impls{&impl_neon_dot()};
    if (has_i8mm()) impls.push_back(&impl_neon_mm());
    struct Shape { int kind; int64_t N, K, M; };
    std::vector<Shape> shapes;
    for (int t = 0; t < 60; t++) shapes.push_back({(int)(t & 1), 1 + (int64_t)(rng() % 70), (t & 1 ? 128 : 64) * (1 + (int64_t)(rng() % 12)), 1 + (int64_t)(rng() % 20)});
    for (int kind = 0; kind < 2; kind++) for (int64_t M : {1, 2, 3, 7, 8, 9, 16, 37}) shapes.push_back({kind, 260, 1024, M});
    shapes.push_back({1, 1024, 3712, 5});  // bwe K (29 q8 groups)
    shapes.push_back({0, 4096, 1024, 8});
    shapes.push_back({1, 2048, 8192, 3});
    int cases = 0;
    double maxrel = 0;
    for (const Shape &sh : shapes) {
        const Mat W = rand_mat(sh.kind, sh.N, sh.K);
        const int64_t M = sh.M, K = sh.K, N = sh.N, GA = K / 64;
        std::vector<float> x(M * K);
        for (auto &v : x) v = nd(rng) * (rng() % 4 == 0 ? 30.f : 1.f);
        std::vector<int16_t> q(M * K); std::vector<float> s(M * GA), xs(M * GA);
        impl_scalar().quant(x.data(), K, 0, M, K, q.data(), s.data(), xs.data());
        for (int b = 0; b < 2; b++) {
            const QW qw = W.w(b);
            std::vector<float> want(M * N);
            for (int64_t m = 0; m < M; m++)
                for (int64_t n = 0; n < N; n++) want[m * N + n] = emulate(W, q.data() + m * K, s.data() + m * GA, xs.data() + m * GA, n, b);
            for (int64_t m = 0; m < M; m++)  // float sequence vs double (dequantized activations)
                for (int64_t n = 0; n < N; n++) {
                    double ex = b ? W.bias[n] : 0, mag = std::fabs(ex);
                    for (int64_t k = 0; k < K; k++) {
                        const int64_t ga = k / 64, g = ga / (sh.kind == 0 ? 1 : 2), G = K / (sh.kind == 0 ? 64 : 128);
                        const double w = wcode(W, n, k) * (double)h2(W.sc[n * G + g]) + h2(W.mn[n * G + g]);
                        const double t = w * q[m * K + k] * (double)s[m * GA + ga];
                        ex += t; mag += std::fabs(t);
                    }
                    maxrel = std::max(maxrel, std::fabs(want[m * N + n] - ex) / mag);
                }
            for (const LinQImpl *I : impls) {
                QX X{q.data(), s.data(), xs.data(), M, K};
                std::vector<float> y(M * N, -1.f);
                lin_q(omp_par(8), qw, X, y.data(), N, I);
                CHECK(same(y.data(), want.data(), M * N), "%s kind %d N %lld K %lld M %lld bias %d: != emulation", I->name, sh.kind, (long long)N, (long long)K, (long long)M, b);
                // other split, row pointers in shuffled order, each row alone
                std::vector<int64_t> perm(M);
                for (int64_t m = 0; m < M; m++) perm[m] = m;
                std::shuffle(perm.begin(), perm.end(), rng);
                std::vector<int16_t> qp(M * K); std::vector<float> sp(M * GA), xp(M * GA), yp(M * N);
                std::vector<float *> yr(M);
                for (int64_t i = 0; i < M; i++) {
                    memcpy(qp.data() + i * K, q.data() + perm[i] * K, K * 2);
                    memcpy(sp.data() + i * GA, s.data() + perm[i] * GA, GA * 4);
                    memcpy(xp.data() + i * GA, xs.data() + perm[i] * GA, GA * 4);
                    yr[i] = yp.data() + perm[i] * N;
                }
                lin_q(odd_par(), qw, QX{qp.data(), sp.data(), xp.data(), M, K}, yr.data(), *I);
                CHECK(same(yp.data(), want.data(), M * N), "%s shuffled/odd split kind %d M %lld", I->name, sh.kind, (long long)M);
                const int64_t m1 = rng() % M;
                std::vector<float> y1(N);
                lin_q(omp_par(3), qw, QX{q.data() + m1 * K, s.data() + m1 * GA, xs.data() + m1 * GA, 1, K}, y1.data(), N, I);
                CHECK(same(y1.data(), want.data() + m1 * N, N), "%s row alone kind %d M %lld", I->name, sh.kind, (long long)M);
                std::vector<uint8_t> pk(packed_bytes(qw));
                pack_quads(qw, pk.data());
                QW wp = qw;
                wp.packed = pk.data();
                std::fill(y1.begin(), y1.end(), -1.f);
                lin_q(omp_par(5), wp, QX{q.data() + m1 * K, s.data() + m1 * GA, xs.data() + m1 * GA, 1, K}, y1.data(), N, I);
                CHECK(same(y1.data(), want.data() + m1 * N, N), "%s row alone packed kind %d N %lld", I->name, sh.kind, (long long)N);
                cases++;
            }
        }
    }
    printf("lin_q cases %d, float sequence vs double: max err / sum|terms| %.2e\n", cases, maxrel);
    // lin_q_f32 (quant + lin_q) row pointers
    {
        const Mat W = rand_mat(0, 300, 1024);
        const int64_t M = 13;
        std::vector<float> x(M * 1024), y(M * 300), y2(M * 300);
        for (auto &v : x) v = nd(rng);
        QBuf qb;
        lin_q_f32(omp_par(8), W.w(true), x.data(), 1024, M, y.data(), 300, qb);
        for (int64_t m = 0; m < M; m++) lin_q_f32(omp_par(2), W.w(true), x.data() + m * 1024, 1024, 1, y2.data() + m * 300, 300, qb);
        CHECK(same(y.data(), y2.data(), M * 300), "lin_q_f32 rows");
    }
    printf("%s (%d failures)\n", fails ? "FAILED" : "all passed", fails);
    return fails != 0;
}
