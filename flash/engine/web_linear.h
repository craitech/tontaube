// Linear layers and attention of the WebAssembly build (included by tt_engine.cpp before Model; vv_dec.h uses
// web_rows_run).
#pragma once

// Another pool that takes the last share of a linear's row blocks while it has nothing else to do (pipelined: the
// decoder's pool for the frame model's linears, the frame model's pool for the decoder's; set per thread by the
// engine): start() hands it the job if it is free, wait() joins it. Every row is computed as on the caller's pool.
struct WebHelper {
    Pool *pool = nullptr;
    std::function<bool(const std::function<void()> &)> start;
    std::function<void()> wait;
};
static thread_local WebHelper *t_web_helper = nullptr;

// fn(t0, t1) over row blocks 0..nt-1 on the pool, the last part on the helper if it is free
static void web_rows_run(Pool *pool, int64_t nt, const std::function<void(int64_t, int64_t)> &fn) {
    WebHelper *h = t_web_helper;
    if (h && h->pool != pool && nt >= 4) {
        const int64_t split = nt * pool->n / (pool->n + h->pool->n);
        const std::function<void()> part = [&]() { h->pool->run(nt - split, 1, [&](int64_t a, int64_t b) { fn(split + a, split + b); }); };
        if (split > 0 && split < nt && h->start(part)) {
            pool->run(split, 1, fn);
            h->wait();
            return;
        }
    }
    pool->run(nt, 1, fn);
}

// Activations to the engine's int16 codes (int8 for the layers of g_web_act8), then web_gemm_q / web_gemm_a8 over
// the pool in blocks of output rows.
static bool web_linear(const Lin &L, const float *x, int64_t xs, float *y, int64_t ys, int M, Act *acts, Pool *pool) {
    if (L.kind > 1 || L.K / 64 > WEB_MAXG || M > 8) return false;
    const int64_t rows = web_rows(L.N, pool->n, M);
    const bool a8 = web_act8(L.codes);
#ifdef WEB_DOT8
    if (g_web_x8 && L.kind == 0 && !a8) {  // the int16 numbers through the 8-bit dot product (web_gemm_x8)
        WebActX b[8];
        for (int m = 0; m < M; m++) {
            web_quantize16(x + m * xs, L.K, acts[m].q.data(), acts[m].s.data(), acts[m].xs.data());
            web_split16(acts[m].q.data(), L.K, acts[m].hi.data(), (int8_t *)acts[m].lo.data());
            b[m] = WebActX{acts[m].hi.data(), (const int8_t *)acts[m].lo.data(), acts[m].s.data(), acts[m].xs.data()};
        }
        web_rows_run(pool, (L.N + rows - 1) / rows, [&](int64_t t0, int64_t t1) {
            web_gemm_x8(L.codes, L.scales, L.mins, L.bias, L.G, b, M, y, ys, t0 * rows, std::min(L.N, t1 * rows));
        });
        return true;
    }
    if (g_web_x8 && L.kind == 1 && !a8) {  // q8: the activation in three 7-bit pieces (web_gemm_x8q8)
        WebActX3 b[8];
        for (int m = 0; m < M; m++) {
            web_quantize16(x + m * xs, L.K, acts[m].q.data(), acts[m].s.data(), acts[m].xs.data());
            int8_t *u0 = acts[m].hi.data(), *u1 = (int8_t *)acts[m].lo.data(), *u2 = acts[m].hl.data();
            web_split16x3(acts[m].q.data(), L.K, u0, u1, u2);
            b[m] = WebActX3{u0, u1, u2, acts[m].s.data(), acts[m].xs.data()};
        }
        web_rows_run(pool, (L.N + rows - 1) / rows, [&](int64_t t0, int64_t t1) {
            web_gemm_x8q8(L.codes, L.scales, L.mins, L.bias, L.G, b, M, y, ys, t0 * rows, std::min(L.N, t1 * rows));
        });
        return true;
    }
    if (a8) {
        static int32_t sq[8 * WEB_MAXG];
        WebAct8 b[8];
        for (int m = 0; m < M; m++) {
            web_quantize8(x + m * xs, L.K, acts[m].hi.data(), acts[m].s.data(), acts[m].xs.data(), sq + m * WEB_MAXG);
            b[m] = WebAct8{acts[m].hi.data(), acts[m].s.data(), acts[m].xs.data(), sq + m * WEB_MAXG};
        }
        web_rows_run(pool, (L.N + rows - 1) / rows, [&](int64_t t0, int64_t t1) {
            web_gemm_a8(L.kind, L.codes, L.scales, L.mins, L.bias, L.G, b, M, y, ys, t0 * rows, std::min(L.N, t1 * rows));
        });
        return true;
    }
#endif
    WebAct a[8];
    for (int m = 0; m < M; m++) {
        web_quantize16(x + m * xs, L.K, acts[m].q.data(), acts[m].s.data(), acts[m].xs.data(), a8 ? 127.f : 32767.f);
        a[m] = WebAct{acts[m].q.data(), acts[m].s.data(), acts[m].xs.data()};
    }
    web_rows_run(pool, (L.N + rows - 1) / rows, [&](int64_t t0, int64_t t1) {
        web_gemm_q(L.kind, L.codes, L.scales, L.mins, L.bias, L.G, a, M, y, ys, t0 * rows, std::min(L.N, t1 * rows));
    });
    return true;
}

// One query token against cache rows r0 .. r0 + L - 1 of every KV head (K / V: [kv head][S][128]): one task per KV
// head and block of 128 rows (its query heads together, so each cache row is read once); the blocks' partial softmax
// sums are combined exactly afterwards.
template <typename KV>
static void web_attention(const float *q, const KV *K, const KV *V, int64_t S, int64_t nq, int64_t nkv, int64_t L,
                          int64_t r0, float fs, float *out, float *sbuf, Pool *pool) {
    const int64_t rep = nq / nkv, BR = 128, nb = (L + BR - 1) / BR, P = 130;  // P: 128 values, max, sum
    static std::vector<float> part;
    if ((int64_t)part.size() < nq * nb * P) part.resize(nq * nb * P);
    pool->run(nkv * nb, 1, [&](int64_t a0, int64_t a1) {
        for (int64_t task = a0; task < a1; task++) {
            const int64_t hh = task / nb, blk = task % nb, j0 = blk * BR, j1 = std::min(L, j0 + BR);
            float mx[4], sm[4];
            float *p = part.data() + task * rep * P;
            const float *qh = q + hh * rep * 128;
            const KV *Kh = K + (hh * S + r0) * 128, *Vh = V + (hh * S + r0) * 128;
            float *sh = sbuf + hh * rep * (S + 8);
            if (rep == 2) web_attend_block<2>(qh, Kh, Vh, j0, j1, fs, sh, S + 8, p, mx, sm);
            else if (rep == 1) web_attend_block<1>(qh, Kh, Vh, j0, j1, fs, sh, S + 8, p, mx, sm);
            else web_attend_block<4>(qh, Kh, Vh, j0, j1, fs, sh, S + 8, p, mx, sm);
            for (int64_t r = 0; r < rep; r++) { p[r * P + 128] = mx[r]; p[r * P + 129] = sm[r]; }
        }
    });
    for (int64_t hq = 0; hq < nq; hq++) {
        const int64_t hh = hq / rep, r = hq % rep;
        float mx = -INFINITY, sum = 0.f;
        for (int64_t b = 0; b < nb; b++) mx = std::max(mx, part[((hh * nb + b) * rep + r) * P + 128]);
        float *o = out + hq * 128;
        for (int i = 0; i < 128; i++) o[i] = 0.f;
        for (int64_t b = 0; b < nb; b++) {
            const float *p = part.data() + ((hh * nb + b) * rep + r) * P, w = std::exp(p[128] - mx);
            sum += w * p[129];
            for (int i = 0; i < 128; i++) o[i] += w * p[i];
        }
        const float inv = 1.f / sum;
        for (int i = 0; i < 128; i++) o[i] *= inv;
    }
}
