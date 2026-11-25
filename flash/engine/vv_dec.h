// VibeVoice decoder, streaming: one 64-value latent frame (133 ms) -> 3,200 samples at 24 kHz per call, with the
// exact causal state of every convolution (same result as decoding the whole sequence at once).
// Weights: vv.bin (blocks folded; wide linears int8 per 128 inputs, the rest float32).
// Include after tt_engine.cpp: uses File, Lin, Pool.
//
// Every causal convolution is a linear layer over a window of consecutive rows: with P = [state rows; new rows]
// contiguous, the window of output t starts at P + t * C, so rows overlap and nothing is copied.
#pragma once
#include "web_kernels.h"
#include "web_linear.h"

// Activation codes of many rows (int16 per 64 inputs; act8: int8, see web_quantize8).
struct WebRows { std::vector<int16_t> q; std::vector<float> s, xs; bool act8 = false; std::vector<int8_t> q8, lo, u2; std::vector<int32_t> sq; };

// What a task does with its finished output tile (rows m0..m1-1, outputs n0..n1-1) while it is in cache.
typedef std::function<void(int64_t, int64_t, int64_t, int64_t)> WebPost;

// y[m * ys ..] = L(x[m * xs ..]) for M rows (rows may overlap): float layers through web_gemm_f32, int8 / int4 layers
// through web_gemm_q in tasks of (8 rows x a block of outputs); post (if any) on each task's tile.
static void web_lin_rows(Pool *pool, const Lin &L, const float *x, int64_t xs, int64_t M, float *y, int64_t ys, WebRows &w,
                         const WebPost *post = nullptr) {
    if (L.kind == 2) {
        const int64_t mb = 16;
        if (M >= 32)
            web_rows_run(pool, (M + mb - 1) / mb, [&](int64_t a, int64_t b) {
#ifdef WEB_SIMD
                if (g_web_f44) web_gemm_f32_44(L.wf, L.bias, L.K, x, xs, y, ys, a * mb, std::min(M, b * mb), 0, L.N);
                else
#endif
                web_gemm_f32(L.wf, L.bias, L.K, x, xs, y, ys, a * mb, std::min(M, b * mb), 0, L.N);
                if (post) (*post)(a * mb, std::min(M, b * mb), 0, L.N);
            });
        else
            web_rows_run(pool, (L.N + 7) / 8, [&](int64_t a, int64_t b) {
                web_gemm_f32(L.wf, L.bias, L.K, x, xs, y, ys, 0, M, a * 8, std::min(L.N, b * 8));
                if (post) (*post)(0, M, a * 8, std::min(L.N, b * 8));
            });
        return;
    }
    const int64_t K = L.K, NG = K / 64;
    if ((int64_t)w.q.size() < M * K) { w.q.resize(M * K); w.s.resize(M * NG + 8); w.xs.resize(M * NG + 8); }
    const int64_t nb = web_rows(L.N, pool->n, (int)std::min<int64_t>(M, 8)), nmb = (M + 7) / 8, nnb = (L.N + nb - 1) / nb;
#ifdef WEB_DOT8
    if (w.act8) {
        if ((int64_t)w.q8.size() < M * K) { w.q8.resize(M * K); w.sq.resize(M * NG + 8); }
        pool->run(M, M >= 64 ? 8 : 1, [&](int64_t a, int64_t b) {
            for (int64_t m = a; m < b; m++)
                web_quantize8(x + m * xs, K, w.q8.data() + m * K, w.s.data() + m * NG, w.xs.data() + m * NG, w.sq.data() + m * NG);
        });
        web_rows_run(pool, nmb * nnb, [&](int64_t a, int64_t b) {
            for (int64_t task = a; task < b; task++) {
                const int64_t mi = task / nnb, ni = task % nnb, m0 = mi * 8, mn = std::min<int64_t>(8, M - m0);
                WebAct8 ac[8];
                for (int64_t m = 0; m < mn; m++)
                    ac[m] = WebAct8{w.q8.data() + (m0 + m) * K, w.s.data() + (m0 + m) * NG, w.xs.data() + (m0 + m) * NG, w.sq.data() + (m0 + m) * NG};
                web_gemm_a8(L.kind, L.codes, L.scales, L.mins, L.bias, L.G, ac, (int)mn, y + m0 * ys, ys, ni * nb, std::min(L.N, (ni + 1) * nb));
                if (post) (*post)(m0, m0 + mn, ni * nb, std::min(L.N, (ni + 1) * nb));
            }
        });
        return;
    }
#endif
    const float qmax = w.act8 ? 127.f : 32767.f;
#ifdef WEB_DOT8
    if (g_web_x8 && (L.kind == 0 || L.kind == 1) && !w.act8) {  // int16 numbers through the 8-bit dot product
        const bool q8 = L.kind == 1;
        if ((int64_t)w.q8.size() < M * K) { w.q8.resize(M * K); w.lo.resize(M * K); w.u2.resize(M * K); }
        pool->run(M, M >= 64 ? 8 : 1, [&](int64_t a, int64_t b) {
            for (int64_t m = a; m < b; m++) {
                web_quantize16(x + m * xs, K, w.q.data() + m * K, w.s.data() + m * NG, w.xs.data() + m * NG);
                if (q8) web_split16x3(w.q.data() + m * K, K, w.q8.data() + m * K, w.lo.data() + m * K, w.u2.data() + m * K);
                else web_split16(w.q.data() + m * K, K, w.q8.data() + m * K, w.lo.data() + m * K);
            }
        });
        web_rows_run(pool, nmb * nnb, [&](int64_t a, int64_t b) {
            for (int64_t task = a; task < b; task++) {
                const int64_t mi = task / nnb, ni = task % nnb, m0 = mi * 8, mn = std::min<int64_t>(8, M - m0);
                const int64_t o = m0 * K, sg = m0 * NG;
                if (q8) {
                    WebActX3 ac[8];
                    for (int64_t m = 0; m < mn; m++)
                        ac[m] = WebActX3{w.q8.data() + o + m * K, w.lo.data() + o + m * K, w.u2.data() + o + m * K, w.s.data() + sg + m * NG, w.xs.data() + sg + m * NG};
                    web_gemm_x8q8(L.codes, L.scales, L.mins, L.bias, L.G, ac, (int)mn, y + m0 * ys, ys, ni * nb, std::min(L.N, (ni + 1) * nb));
                } else {
                    WebActX ac[8];
                    for (int64_t m = 0; m < mn; m++)
                        ac[m] = WebActX{w.q8.data() + o + m * K, w.lo.data() + o + m * K, w.s.data() + sg + m * NG, w.xs.data() + sg + m * NG};
                    web_gemm_x8(L.codes, L.scales, L.mins, L.bias, L.G, ac, (int)mn, y + m0 * ys, ys, ni * nb, std::min(L.N, (ni + 1) * nb));
                }
                if (post) (*post)(m0, m0 + mn, ni * nb, std::min(L.N, (ni + 1) * nb));
            }
        });
        return;
    }
#endif
    pool->run(M, M >= 64 ? 8 : 1, [&](int64_t a, int64_t b) {
        for (int64_t m = a; m < b; m++) web_quantize16(x + m * xs, K, w.q.data() + m * K, w.s.data() + m * NG, w.xs.data() + m * NG, qmax);
    });
    web_rows_run(pool, nmb * nnb, [&](int64_t a, int64_t b) {
        for (int64_t task = a; task < b; task++) {
            const int64_t mi = task / nnb, ni = task % nnb, m0 = mi * 8, mn = std::min<int64_t>(8, M - m0);
            WebAct ac[8];
            for (int64_t m = 0; m < mn; m++)
                ac[m] = WebAct{w.q.data() + (m0 + m) * K, w.s.data() + (m0 + m) * NG, w.xs.data() + (m0 + m) * NG};
            web_gemm_q(L.kind, L.codes, L.scales, L.mins, L.bias, L.G, ac, (int)mn, y + m0 * ys, ys, ni * nb,
                       std::min(L.N, (ni + 1) * nb));
            if (post) (*post)(m0, m0 + mn, ni * nb, std::min(L.N, (ni + 1) * nb));
        }
    });
}
static WebRows g_bwe_rows;  // the upsampler's activation codes (one caller at a time)

struct VVDec {
    File f;
    Pool *pool = nullptr;
    int NS = 0, HOP = 3200, LAT = 64, DWK = 7, HK = 7;
    float eps = 1e-5f;
    struct Block { const float *dw = nullptr, *dwb = nullptr; Lin l1, l2; std::vector<float> st; };  // st [6][C]
    struct Stage { int C = 0, stride = 1, depth = 0, cin = 0, k = 0; Lin res; std::vector<Block> b; std::vector<float> st; };
    std::vector<Stage> S;
    Lin head;
    std::vector<float> hst;                 // head state [6][C of the last stage]
    std::vector<float> O, X, P, H, Q;       // work (see decode)
    WebRows wrows;                          // activation codes of a quantized linear's input rows
    std::vector<float> start;               // state after the fixed first frames (set_start)
    double t_stage[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
#ifdef TT_KERN
    // x86 (AVX2 / AVX-512): float linears as packed panels (blocks with float FFNs fused per tile of rows, or wide:
    // the FFN as GEMMs over all rows), quantized linears through kern::lin_q; one frame of several streams per pass
    bool x86 = false;
    int64_t wide_c = 256;  // blocks with C >= wide_c run wide
    int64_t cap = 1;       // streams per pass the work buffers hold
    struct X86Stage { kern::Panels res; std::vector<kern::VVBlock> b; int64_t piece = 0; };
    std::vector<X86Stage> xs;
    std::vector<std::vector<float>> packed;
    std::vector<float> ns;
    int64_t npiece = 0;  // a stream's state pieces (save_state order): per stage its window rows, then its blocks' rows
                         // (X86Stage::piece = the stage's first); last the head's
    std::vector<const float *> rin;
    std::vector<float *> rout, sout;
    kern::QBuf qbuf;
#endif
    int64_t calls = 0;

    bool load(const char *path, Pool *p) {
        pool = p;
        if (!f.open(path)) return false;
        const int32_t *c = (const int32_t *)f.get("cfg_vv").p;
        NS = c[0]; HOP = c[1]; LAT = c[2]; DWK = c[3]; HK = c[4];
        eps = ((const float *)f.get("cfg_vvf").p)[0];
        S.resize(NS);
        int64_t rows = 1, maxx = 0, maxh = 0, maxa = 0;
        for (int i = 0; i < NS; i++) {
            Stage &s = S[i];
            s.C = c[5 + 4 * i]; s.stride = c[6 + 4 * i]; s.depth = c[7 + 4 * i]; s.cin = c[8 + 4 * i];
            s.k = i == 0 ? DWK : 2;  // the stem is a causal conv (7 taps), the others transposed convs over 2 frames
            s.res.load(f, "s" + std::to_string(i) + ".res");
            const int64_t rin = rows;
            rows *= s.stride;
            s.b.resize(s.depth);
            for (int j = 0; j < s.depth; j++) {
                Block &b = s.b[j];
                const std::string p = "s" + std::to_string(i) + ".b" + std::to_string(j) + ".";
                b.dw = (const float *)f.get(p + "dw").p;
                b.dwb = (const float *)f.get(p + "dwb").p;
                b.l1.load(f, p + "l1");
                b.l2.load(f, p + "l2");
                if (b.l1.kind != 2) maxa = std::max(maxa, rows * b.l1.K);
                if (b.l2.kind != 2) maxa = std::max(maxa, rows * b.l2.K);
                maxh = std::max(maxh, rows * b.l1.N);
            }
            if (s.res.kind != 2) maxa = std::max(maxa, rin * s.res.K);
            maxx = std::max({maxx, rows * s.C, (rin + s.k) * s.cin, (rows + DWK) * s.C});
        }
        head.load(f, "head");
        O.assign(maxx, 0.f);
        X.assign(maxx, 0.f);
        P.assign(maxx, 0.f);
        H.assign(maxh, 0.f);
        Q.assign(maxx, 0.f);
        wrows.q.assign(maxa, 0);
        wrows.s.assign(maxa / 64 + 8, 0.f);
        wrows.xs.assign(maxa / 64 + 8, 0.f);
#ifdef TT_KERN
        x86_setup();
#endif
        reset();
        return true;
    }
    size_t weight_bytes() const {
        size_t b = 0;
        auto lb = [](const Lin &l) { return l.kind == 2 ? (size_t)l.N * l.K * 4 : (size_t)l.N * l.G * ((l.kind == 0 ? 32 : 128) + 4); };
        for (auto &s : S) { b += lb(s.res); for (auto &k : s.b) b += lb(k.l1) + lb(k.l2); }
        return b + lb(head);
    }

    void reset() {  // zero state = the zero padding of a decode from the start
        for (auto &s : S) {
            s.st.assign((size_t)(s.k - 1) * s.cin, 0.f);
            for (auto &b : s.b) b.st.assign((size_t)(DWK - 1) * s.C, 0.f);
        }
        hst.assign((size_t)(HK - 1) * S.back().C, 0.f);
    }
    // The whole streaming state as one flat block of state_size() floats (every causal window's rows): saving it
    // before a frame and loading it back makes the next decode() exactly as if nothing had been decoded since.
    size_t state_size() const {
        size_t n = hst.size();
        for (auto &s : S) { n += s.st.size(); for (auto &b : s.b) n += b.st.size(); }
        return n;
    }
    void save_state(float *dst) const {
        auto put = [&](const std::vector<float> &v) { memcpy(dst, v.data(), v.size() * sizeof(float)); dst += v.size(); };
        for (auto &s : S) { put(s.st); for (auto &b : s.b) put(b.st); }
        put(hst);
    }
    void load_state(const float *src) {
        auto get = [&](std::vector<float> &v) { memcpy(v.data(), src, v.size() * sizeof(float)); src += v.size(); };
        for (auto &s : S) { get(s.st); for (auto &b : s.b) get(b.st); }
        get(hst);
    }
    // The state after k fixed latent frames [k][64] (always the same audio, never played), for begin().
    void set_start(const float *lat, int64_t k) {
        std::vector<float> a((size_t)HOP);
        reset();
        for (int64_t t = 0; t < k; t++) decode(lat + t * LAT, a.data());
        start.clear();
        if (k > 0) { start.resize(state_size()); save_state(start.data()); }
        reset();
    }
    void begin() { if (start.empty()) reset(); else load_state(start.data()); }

    void linear(const Lin &L, const float *x, int64_t xs, int64_t M, float *y, int64_t ys, const WebPost *post = nullptr) {
        web_lin_rows(pool, L, x, xs, M, y, ys, wrows, post);
    }

#ifdef TT_KERN
    kern::Panels pack(const Lin &L) {
        kern::Panels w;
        if (L.kind != 2 || L.N % 16) return w;
        packed.emplace_back((size_t)L.N * L.K);
        kern::pack_panels(L.wf, L.N, L.K, false, packed.back().data());
        w.f32 = packed.back().data();
        w.K = L.K;
        w.N = L.N;
        return w;
    }
    void x86_setup() {
        x86 = g_kf && head.kind == 2 && head.N == 1 && S.back().C % 8 == 0;
        for (const Stage &st : S) {
            x86 = x86 && st.res.kind <= 2;
            for (const Block &b : st.b) x86 = x86 && b.l1.kind <= 2 && b.l2.kind <= 2;
        }
        if (!x86) return;
        xs.resize(NS);
        int64_t cmax = 0;
        for (int i = 0; i < NS; i++) {
            xs[i].res = pack(S[i].res);
            xs[i].piece = npiece++;
            cmax = std::max<int64_t>(cmax, S[i].C);
            for (Block &b : S[i].b) {
                kern::VVBlock k;  // norm weights and layer scales are folded into dw / l1 / l2
                k.C = S[i].C; k.H = b.l1.N; k.dw = b.dw; k.dwb = b.dwb; k.eps = eps; k.gelu = kern::EPI_GELU;
                k.b1 = b.l1.bias; k.b2 = b.l2.bias;
                if (b.l1.kind == 2 && b.l2.kind == 2) { k.w1 = pack(b.l1); k.w2 = pack(b.l2); }
                xs[i].b.push_back(k);
                npiece++;
            }
        }
        npiece++;
        ns.assign((size_t)(DWK - 1) * cmax, 0.f);
    }
    // y = L(x) for M rows (strided, may overlap, or row pointers) with the epilogue e (GELU, or y = e.r + L(x), rows ys
    // apart): float panels, or kern::lin_q with the epilogue on each finished output tile (an added layer goes
    // through P).
    void lin_x86(const kern::Par &par, const Lin &L, const kern::Panels &pw, const kern::Rows &x, int64_t M, float *y,
                 int64_t ys, const kern::Epi &e = kern::Epi()) {
        if (pw.f32) { kern::gemm_f32(par, x, M, pw, L.bias, y, ys, e, task_rows(M, L.N / 32), 2); return; }
        float *out = e.op == kern::EPI_ADD ? P.data() : y;
        const auto post = [&](int64_t m0, int64_t m1, int64_t n0, int64_t n1) {
            for (int64_t m = m0; m < m1; m++) {
                if (e.op == kern::EPI_GELU) kern::gelu(y + m * ys + n0, n1 - n0);
                else for (int64_t k = m * ys + n0; k < m * ys + n1; k++) y[k] = e.r[k] + out[k];
            }
        };
        const kern::LinPost lp = kern::lin_post(post);
        kern::QW w;
        w.kind = L.kind; w.N = L.N; w.K = L.K; w.codes = L.codes; w.scales = L.scales; w.mins = L.mins; w.bias = L.bias;
        if (x.p) {
            rout.resize((size_t)M);
            for (int64_t m = 0; m < M; m++) rout[m] = out + m * ys;
            kern::lin_q_f32(par, w, x.p, M, rout.data(), qbuf, nullptr, e.op ? &lp : nullptr);
        } else {
            kern::lin_q_f32(par, w, x.a, x.lda, M, out, ys, qbuf, nullptr, e.op ? &lp : nullptr);
        }
    }
    // rows per GEMM task: about four tasks per thread over (pairs of panels x rows)
    int64_t task_rows(int64_t M, int64_t groups) const {
        const int64_t chunks = std::max<int64_t>(1, (4 * pool->n + groups - 1) / std::max<int64_t>(1, groups));
        return std::max<int64_t>(6, ((M + chunks - 1) / chunks + 5) / 6 * 6);
    }
    // rows per tile of a fused block (M rows in all): a multiple of the thread count of tiles, at least two per thread
    int64_t tile_rows(int64_t M, int64_t C) const {
        const int64_t n = pool->n, def = kern::vv_tile_rows(C), k = std::max<int64_t>(2, (M + def * n - 1) / (def * n));
        return std::max<int64_t>(1, (M + k * n - 1) / (k * n));
    }
    // One frame of each of B streams through every stage together, so the quantized weights are read once for all
    // streams: lat[b] [64] -> audio[b] [HOP]. Stream b's state pieces are read at si[b * NP + p] and the new ones
    // written to so[b * NP + p] (NP = npiece; the same memory is fine). Every row has the same bits as for the
    // stream alone. Per block: X -> O (fused tiles, or the mixer x1 -> O, rms(x1) -> Q, then the FFN with GELU and the
    // residual add as the linears' epilogues), new states from X, swap.
    void pass_x86(int64_t B, const float *const *lat, const float *const *si, float *const *so, float *const *audio) {
        if (B > cap) {
            for (auto *v : {&O, &X, &P, &H, &Q, &ns}) v->resize(v->size() / cap * B);
            cap = B;
        }
        const kern::Par par = pool_par(pool);
        const int64_t NP = npiece;
        std::vector<const float *> stp((size_t)B);
        std::vector<float *> nsp((size_t)B);
        int64_t T = 1;
        for (int64_t b = 0; b < B; b++) memcpy(O.data() + b * LAT, lat[b], (size_t)LAT * sizeof(float));
        for (int i = 0; i < NS; i++) {
            const double t0 = now_s();
            Stage &s = S[i];
            const X86Stage &f = xs[i];
            // the stage's conv: per stream the window rows [state; new rows] in P, one linear over all streams' rows
            const int64_t Tin = T, nst = s.k - 1, win = (nst + Tin) * s.cin;
            rin.resize((size_t)(B * Tin));
            for (int64_t b = 0; b < B; b++) {
                float *pb = P.data() + b * win;
                memcpy(pb, si[b * NP + f.piece], (size_t)nst * s.cin * sizeof(float));
                memcpy(pb + nst * s.cin, O.data() + b * Tin * s.cin, (size_t)Tin * s.cin * sizeof(float));
                for (int64_t t = 0; t < Tin; t++) rin[b * Tin + t] = pb + t * s.cin;
            }
            lin_x86(par, s.res, f.res, kern::Rows(rin.data()), B * Tin, X.data(), s.res.N);
            for (int64_t b = 0; b < B; b++)
                memcpy(so[b * NP + f.piece], P.data() + b * win + Tin * s.cin, (size_t)nst * s.cin * sizeof(float));
            T = Tin * s.stride;
            const int64_t C = s.C, M = B * T;
            for (size_t j = 0; j < s.b.size(); j++) {
                Block &blk = s.b[j];
                const kern::VVBlock &kb = f.b[j];
                const int64_t pj = f.piece + 1 + (int64_t)j;
                for (int64_t b = 0; b < B; b++) { stp[b] = si[b * NP + pj]; nsp[b] = ns.data() + b * (DWK - 1) * C; }
                if (kb.w1.f32 && C < wide_c) {
                    kern::vv_block(par, kb, X.data(), B, T, stp.data(), O.data(), std::min(T, tile_rows(M, C)));
                } else {
                    kern::vv_mixer(par, kb, X.data(), B, T, stp.data(), O.data(), Q.data(),
                                   std::max<int64_t>((M + 2 * pool->n - 1) / (2 * pool->n), (8192 + C - 1) / C));
                    kern::Epi g, r;
                    g.op = kern::EPI_GELU;
                    r.op = kern::EPI_ADD; r.r = O.data(); r.ldr = C;
                    lin_x86(par, blk.l1, kb.w1, kern::Rows(Q.data(), C), M, H.data(), blk.l1.N, g);
                    lin_x86(par, blk.l2, kb.w2, kern::Rows(H.data(), blk.l1.N), M, O.data(), C, r);
                }
                kern::vv_next_state(kb, X.data(), B, T, stp.data(), nsp.data());
                for (int64_t b = 0; b < B; b++) memcpy(so[b * NP + pj], nsp[b], (size_t)(DWK - 1) * C * sizeof(float));
                std::swap(X, O);
            }
            std::swap(X, O);
            t_stage[i] += now_s() - t0;
        }
        // head: causal conv to one channel; its state is the last HK - 1 input rows (T = HOP >= HK - 1)
        const double t1 = now_s();
        const int64_t Ch = S.back().C, R = HK - 1;
        kern::par_for(par, B * T, 256, [&](int64_t a, int64_t e) {
            for (int64_t i = a; i < e;) {  // runs of rows of one stream
                const int64_t b = i / T, r0 = i % T, r1 = std::min(T, r0 + (e - i));
                kern::conv_small(O.data() + b * T * Ch, si[b * NP + NP - 1], HK, Ch, 1, head.wf, head.bias, r0, r1, audio[b]);
                i += r1 - r0;
            }
        });
        for (int64_t b = 0; b < B; b++)
            memcpy(so[b * NP + NP - 1], O.data() + (b * T + T - R) * Ch, (size_t)R * Ch * sizeof(float));
        t_stage[NS] += now_s() - t1;
        calls += B;
    }
#endif

    // One latent frame [64] -> HOP samples. O: the stage's input rows (the previous stage's output), P: window
    // rows, X: the stage's rows, Q: FFN input rows, H: FFN hidden rows. Per block: rms(x) into the window rows; the
    // depthwise conv added to x and rms(x) into Q in one pass; the FFN's GELU and residual add done by the linears'
    // tasks on their output tiles.
    void decode(const float *lat, float *audio) { VVDec *self = this; decode_rows(1, &self, &lat, &audio); }

    // One latent frame of each of B streams: lat[b] [LAT] -> audio[b] [HOP]. St[b]: stream b's decoder, of which only the
    // streaming state is read and written (the vectors of save_state / load_state; St[b] may be this); weights, work
    // buffers and timings are this decoder's. Every stream's audio and new state have the same bits as decode() on it
    // alone, for any B, order or thread count. On x86 the streams go through each stage together (the weights are read
    // once per pass); elsewhere one after the other.
    void decode_rows(int B, VVDec *const *St, const float *const *lat, float *const *audio) {
#ifdef TT_KERN
        if (x86 && !wrows.act8) {
            const int64_t NP = npiece;
            sout.resize((size_t)B * NP);
            for (int b = 0; b < B; b++) {
                float **o = sout.data() + b * NP;
                for (Stage &st : St[b]->S) { *o++ = st.st.data(); for (Block &k : st.b) *o++ = k.st.data(); }
                *o = St[b]->hst.data();
            }
            pass_x86(B, lat, sout.data(), sout.data(), audio);
            return;
        }
#endif
        for (int b = 0; b < B; b++) {
            if (St[b] != this) swap_state(*St[b]);
            decode_one(lat[b], audio[b]);
            if (St[b] != this) swap_state(*St[b]);
        }
    }
    void swap_state(VVDec &o) {
        for (int i = 0; i < NS; i++) {
            S[i].st.swap(o.S[i].st);
            for (size_t j = 0; j < S[i].b.size(); j++) S[i].b[j].st.swap(o.S[i].b[j].st);
        }
        hst.swap(o.hst);
    }
    // A stream's copy for decode_rows(): only its state is used, the rest is freed.
    void state_only() {
#ifdef TT_KERN
        packed = {}; xs = {}; ns = {};
#endif
        O = {}; X = {}; P = {}; H = {}; Q = {}; start = {};
        wrows = WebRows();
    }
    // stages < NS: only the first `stages` stages run (their states advance, no audio): the GPU fallback's state rebuild
    void decode_one(const float *lat, float *audio, int stages = 1 << 30) {
        int64_t T = 1;
        memcpy(O.data(), lat, (size_t)LAT * sizeof(float));
        for (int i = 0; i < std::min(NS, stages); i++) {
            const double t0 = now_s();
            Stage &s = S[i];
            const int64_t Tin = T, nst = s.k - 1;
            memcpy(P.data(), s.st.data(), (size_t)nst * s.cin * sizeof(float));
            memcpy(P.data() + nst * s.cin, O.data(), (size_t)Tin * s.cin * sizeof(float));
            linear(s.res, P.data(), s.cin, Tin, X.data(), s.res.N);
            memcpy(s.st.data(), P.data() + Tin * s.cin, (size_t)nst * s.cin * sizeof(float));
            T = Tin * s.stride;
            const int64_t C = s.C, R = DWK - 1, rc = std::max<int64_t>(1, 4096 / C);
            for (Block &b : s.b) {
                // depthwise causal conv over rms(x): P = [state; rms rows]
                memcpy(P.data(), b.st.data(), (size_t)R * C * sizeof(float));
                pool->run(T, rc, [&](int64_t a, int64_t e) {
                    for (int64_t t = a; t < e; t++) web_rms(X.data() + t * C, C, eps, P.data() + (R + t) * C);
                });
                pool->run(T, rc, [&](int64_t a, int64_t e) {
                    for (int64_t t = a; t < e; t++) {
                        float *x = X.data() + t * C;
                        for (int j = 0; j < DWK; j++) {
                            const float *w = b.dw + j * C, *p = P.data() + (t + j) * C;
                            if (j == 0) for (int64_t c = 0; c < C; c++) x[c] += b.dwb[c] + w[c] * p[c];
                            else for (int64_t c = 0; c < C; c++) x[c] += w[c] * p[c];
                        }
                        web_rms(x, C, eps, Q.data() + t * C);  // the FFN's input
                    }
                });
                memcpy(b.st.data(), P.data() + T * C, (size_t)R * C * sizeof(float));
                // FFN over rms(x): GELU on lin1's tiles, lin2's tiles added to x
                const int64_t F = b.l1.N;
                const WebPost gelu = [&](int64_t m0, int64_t m1, int64_t n0, int64_t n1) {
                    for (int64_t m = m0; m < m1; m++) web_gelu(H.data() + m * F + n0, n1 - n0);
                };
                const WebPost add = [&](int64_t m0, int64_t m1, int64_t n0, int64_t n1) {
                    for (int64_t m = m0; m < m1; m++)
                        for (int64_t k = m * C + n0; k < m * C + n1; k++) X[k] += P[k];
                };
                linear(b.l1, Q.data(), C, T, H.data(), F, &gelu);
                linear(b.l2, H.data(), F, T, P.data(), C, &add);
            }
            std::swap(X, O);
            t_stage[i] += now_s() - t0;
        }
        if (stages < NS) return;
        // head: causal conv to one channel
        const double t1 = now_s();
        const int64_t Ch = S.back().C, R = HK - 1;
        memcpy(P.data(), hst.data(), (size_t)R * Ch * sizeof(float));
        memcpy(P.data() + R * Ch, O.data(), (size_t)T * Ch * sizeof(float));
        linear(head, P.data(), Ch, T, audio, 1);
        memcpy(hst.data(), P.data() + T * Ch, (size_t)R * Ch * sizeof(float));
        t_stage[NS] += now_s() - t1;
        calls++;
    }
};
