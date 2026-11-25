// The pitch-map detector of "autotune" spots on the decoded audio, streaming: a pitch map (normalised
// cross-correlation per 5 ms step on a log-pitch axis, quantised to 8 bits), a small convolutional network over it,
// and the spot measure (100 ms mean of the voiced score). Weights: detector.bin. Include after tt_engine.cpp
// (File, Pool).
//
// Step t (5 ms) is centred at sample 120 t of the utterance. Every stage of a step (map, conv1, conv2, conv3, head) is
// computed once, as soon as the inputs it reads are final, so streaming gives the same numbers as one pass over the
// whole utterance. cut(n) drops audio from sample n on (a redraw) and everything that read it; finish() ends the
// utterance (zero padding after its end, as in a whole-utterance pass); tentative() scores as if it ended now.
#pragma once

struct PitchDetect {
    static const int HOP = 120, GUARD = 48, PIECE = 256;
    int64_t RING = 1024;   // steps kept (5 s): a redraw goes back well under a second
    int WIN = 600, TMAX = 343, CEN = 383, P = 77, P2 = 39, C1 = 16, C2 = 24, C3 = 32, HH = 32, NF = 68;
    File file;
    const int32_t *lo = nullptr;
    const float *lw = nullptr;
    std::vector<float> w1, w2, w3, wh1;   // [kt][kp][ci][co] (conv), [k][ci][co] (head1)
    const float *b1 = nullptr, *b2 = nullptr, *b3 = nullptr, *bh1 = nullptr, *wh2 = nullptr;
    float bh2 = 0.f;
    Pool *pool = nullptr;
    double t_det = 0;  // seconds spent

    // utterance state
    std::vector<float> x;    // audio from sample x0 on
    int64_t x0 = 0, N = 0;   // N: samples so far
    bool ended = false;
    int64_t T = 0;           // steps of the ended utterance
    int64_t v[5] = {0, 0, 0, 0, 0};  // steps done per stage: map, conv1 (+ pool), conv2, conv3 (+ max / mean), head
    std::vector<float> rmap, rclar, ra1, ra2, rfeat, rpv;   // stage outputs, ring of RING steps

    bool load(const char *path, Pool *p) {
        pool = p;
        if (!file.open(path)) return false;
        const int32_t *c = (const int32_t *)file.get("cfg_det").p;
        WIN = c[0]; TMAX = c[1]; CEN = c[2]; P = c[3]; C1 = c[4]; C2 = c[5]; C3 = c[6]; HH = c[7];
        P2 = (P + 1) / 2; NF = 2 * C3 + 4;
        lo = (const int32_t *)file.get("pm.lo").p;
        lw = (const float *)file.get("pm.w").p;
        auto F = [&](const char *n) { return (const float *)file.get(n).p; };
        auto conv = [&](const char *n, int co, int ci, std::vector<float> &out) {   // [co][ci][5][5] -> [kt][kp][ci][co]
            const float *w = F(n);
            out.assign((size_t)25 * ci * co, 0.f);
            for (int o = 0; o < co; o++)
                for (int i = 0; i < ci; i++)
                    for (int kt = 0; kt < 5; kt++)
                        for (int kp = 0; kp < 5; kp++) out[(((size_t)kt * 5 + kp) * ci + i) * co + o] = w[(((size_t)o * ci + i) * 5 + kt) * 5 + kp];
        };
        conv("conv1.w", C1, 2, w1); conv("conv2.w", C2, C1, w2); conv("conv3.w", C3, C2, w3);
        const float *h = F("head1.w");   // [HH][NF][5] -> [k][ci][co]
        wh1.assign((size_t)5 * NF * HH, 0.f);
        for (int o = 0; o < HH; o++)
            for (int i = 0; i < NF; i++)
                for (int k = 0; k < 5; k++) wh1[((size_t)k * NF + i) * HH + o] = h[((size_t)o * NF + i) * 5 + k];
        b1 = F("conv1.b"); b2 = F("conv2.b"); b3 = F("conv3.b"); bh1 = F("head1.b"); wh2 = F("head2.w"); bh2 = F("head2.b")[0];
        rmap.assign((size_t)RING * P, 0.f); rclar.assign(RING, 0.f); ra1.assign((size_t)RING * P2 * C1, 0.f);
        ra2.assign((size_t)RING * P2 * C2, 0.f); rfeat.assign((size_t)RING * NF, 0.f); rpv.assign(RING, 0.f);
        reset();
        return true;
    }

    void reset() {
        x.clear(); x0 = 0; N = 0; ended = false; T = 0;
        for (int64_t &k : v) k = 0;
    }
    void push(const float *a, int64_t n) {
        x.insert(x.end(), a, a + n);
        N += n;
        update();
    }
    // Keep only the first n samples (n >= what was dropped from memory: a few seconds back at most).
    void cut(int64_t n) {
        if (n >= N) return;
        if (n < x0) { fprintf(stderr, "pitch_detect: cut before the kept audio\n"); exit(2); }
        x.resize((size_t)(n - x0));
        N = n;
        ended = false;
        bound();
    }
    void finish() {
        ended = true;
        T = (N + HOP - 1) / HOP;
        update();
    }
    // Steps t < final_end() have their final measure (decoded samples / 120 - 48; all once finished).
    int64_t final_end() const { return ended ? T : std::max<int64_t>(0, std::min<int64_t>(N / HOP - GUARD, v[4] - 9)); }
    // The spot measure (100 ms mean of the voiced score) at step t < final_end() (or any step after tentative()).
    float measure(int64_t t) const {
        float s = 0.f;
        for (int64_t k = t - 10; k < t + 10; k++)
            if (k >= 0 && !(ended && k >= T)) s += rpv[k % RING];
        return s / 20.f;
    }
    // The measure of steps lo..hi-1 as if the utterance ended now (whole-utterance numbers of the audio so far).
    void tentative(int64_t lo_, int64_t hi, std::vector<float> &out) {
        int64_t keep[5];
        memcpy(keep, v, sizeof keep);
        const bool e = ended;
        if (!e) finish();
        out.clear();
        for (int64_t t = lo_; t < hi; t++) out.push_back(t < T ? measure(t) : 0.f);
        if (!e) { memcpy(v, keep, sizeof keep); ended = false; }
    }

    // ------------------------------------------------------------ internals
    static float mish(float a) { return a * std::tanh(a > 20.f ? a : std::log1p(std::exp(a))); }
    static void mish_n(float *x, int64_t n) {
#ifdef TT_KERN
        if (g_kf) { kern::mish(x, n); return; }
#endif
        for (int64_t i = 0; i < n; i++) x[i] = mish(x[i]);
    }
    int64_t map_steps() const {   // steps whose analysis window lies in the audio
        if (ended) return T;
        const int64_t need = WIN + TMAX - CEN;   // samples after the step's centre
        return N >= need ? (N - need) / HOP + 1 : 0;
    }
    void bound() {   // stage counts after a cut
        const int r[5] = {0, 2, 4, 8, 16};
        v[0] = std::min(v[0], map_steps());
        for (int k = 1; k < 5; k++) v[k] = std::min(v[k], std::max<int64_t>(0, v[k - 1] - r[k]));
    }
    void par(int64_t n, const std::function<void(int64_t, int64_t)> &fn) {
        if (pool && n > 1) pool->run(n, 1, fn); else fn(0, n);
    }
    void update() {
        const double t0 = now_s();
        const int r[5] = {0, 2, 4, 8, 16};
        for (;;) {
            const int64_t n0 = std::min(map_steps(), v[0] + PIECE);
            bool moved = n0 > v[0];
            stage0(v[0], n0); v[0] = n0;
            for (int k = 1; k < 5; k++) {
                const int64_t nk = ended && v[k - 1] >= T ? T : std::max<int64_t>(v[k], v[k - 1] - r[k]);
                if (nk <= v[k]) continue;
                moved = true;
                if (k == 1) stage1(v[k], nk); else if (k == 2) stage2(v[k], nk); else if (k == 3) stage3(v[k], nk); else stage4(v[k], nk);
                v[k] = nk;
            }
            if (!moved) break;
        }
        const int64_t keep_from = std::max<int64_t>(0, (v[0] - RING + 64) * HOP - CEN);   // audio no stage can need again
        if (keep_from > x0 + 48000) { x.erase(x.begin(), x.begin() + (keep_from - x0)); x0 = keep_from; }
        t_det += now_s() - t0;
    }
    float sample(int64_t i) const { return i < 0 || i >= N ? 0.f : x[(size_t)(i - x0)]; }

    void stage0(int64_t a, int64_t b) {   // pitch map, 8-bit, and its clarity
        const int L = WIN + TMAX;
        par(b - a, [&](int64_t i0, int64_t i1) {
            std::vector<float> seg(L), corr(TMAX + 1), eb(TMAX + 1), c2(L + 1);
            for (int64_t s = a + i0; s < a + i1; s++) {
                const int64_t st = s * HOP - CEN;
                for (int i = 0; i < L; i++) seg[i] = sample(st + i);
#ifdef TT_KERN
                if (g_kf) kern::autocorr(seg.data(), WIN, TMAX, corr.data());
                else
#endif
                for (int tau = 0; tau <= TMAX; tau++) {
                    float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                    const float *p = seg.data(), *q = seg.data() + tau;
                    int i = 0;
                    for (; i + 8 <= WIN; i += 8)
                        for (int j = 0; j < 8; j++) acc[j] += p[i + j] * q[i + j];
                    float t = ((acc[0] + acc[4]) + (acc[2] + acc[6])) + ((acc[1] + acc[5]) + (acc[3] + acc[7]));
                    for (; i < WIN; i++) t += p[i] * q[i];
                    corr[tau] = t;
                }
                double run = 0;   // torch's float cumsum accumulates in double
                c2[0] = 0.f;
                for (int i = 0; i < L; i++) { run += (double)seg[i] * seg[i]; c2[i + 1] = (float)run; }
                for (int tau = 0; tau <= TMAX; tau++) eb[tau] = c2[tau + WIN] - c2[tau];
                const float ea = eb[0], fl = (float)WIN * 1e-7f;
                float *m = rmap.data() + (s % RING) * P, best = 0.f;
                for (int k = 0; k <= TMAX; k++) corr[k] = corr[k] / std::sqrt(ea * eb[k] + fl * fl);
                for (int k = 0; k < P; k++) {
                    const float n0 = corr[lo[k]], n1 = corr[lo[k] + 1], wk = lw[k];   // torch.lerp
                    float y = std::max(0.f, wk < 0.5f ? n0 + wk * (n1 - n0) : n1 - (n1 - n0) * (1.f - wk));
                    y = std::nearbyint(std::min(1.f, y) * 255.f) / 255.f;
                    m[k] = y;
                    best = std::max(best, y);
                }
                rclar[s % RING] = best;
            }
        });
    }
    bool in_time(int64_t s) const { return s >= 0 && !(ended && s >= T); }
    void stage1(int64_t a, int64_t b) {   // conv1 (map, clarity) 5x5, Mish, max over pitch pairs
        par(b - a, [&](int64_t i0, int64_t i1) {
            std::vector<float> o((size_t)P * C1);
            for (int64_t s = a + i0; s < a + i1; s++) {
#ifdef TT_KERN
                if (g_kf) {  // (map, clarity) rows of the 5 time taps, 2 zero rows each side
                    std::vector<float> pad((size_t)5 * (P + 4) * 2, 0.f);
                    const float *taps[25];
                    for (int kt = 0; kt < 5; kt++) {
                        const int64_t si = s + kt - 2;
                        float *d = pad.data() + (size_t)kt * (P + 4) * 2;
                        for (int kp = 0; kp < 5; kp++) taps[kt * 5 + kp] = in_time(si) ? d + kp * 2 : nullptr;
                        if (!in_time(si)) continue;
                        const float *m = rmap.data() + (si % RING) * P, cl = rclar[si % RING];
                        for (int p = 0; p < P; p++) { d[(p + 2) * 2] = m[p]; d[(p + 2) * 2 + 1] = cl; }
                    }
                    kern::taps_conv(25, taps, 2, P, 2, C1, w1.data(), b1, o.data());
                    kern::mish(o.data(), (int64_t)P * C1);
                    float *out = ra1.data() + (s % RING) * P2 * C1;
                    for (int q = 0; q < P2; q++)
                        for (int c = 0; c < C1; c++)
                            out[(size_t)q * C1 + c] = 2 * q + 1 < P ? std::max(o[(size_t)2 * q * C1 + c], o[(size_t)(2 * q + 1) * C1 + c])
                                                                    : o[(size_t)2 * q * C1 + c];
                    continue;
                }
#endif
                for (int p = 0; p < P; p++) std::copy(b1, b1 + C1, o.begin() + (size_t)p * C1);
                for (int kt = 0; kt < 5; kt++) {
                    const int64_t si = s + kt - 2;
                    if (!in_time(si)) continue;
                    const float *m = rmap.data() + (si % RING) * P, cl = rclar[si % RING];
                    for (int p = 0; p < P; p++) {
                        float *op = o.data() + (size_t)p * C1;
                        for (int kp = 0; kp < 5; kp++) {
                            const int pi = p + kp - 2;
                            if (pi < 0 || pi >= P) continue;
                            const float *w = w1.data() + ((size_t)kt * 5 + kp) * 2 * C1;
                            const float a0 = m[pi];
                            for (int c = 0; c < C1; c++) op[c] += a0 * w[c] + cl * w[C1 + c];
                        }
                    }
                }
                float *out = ra1.data() + (s % RING) * P2 * C1;
                for (int q = 0; q < P2; q++)
                    for (int c = 0; c < C1; c++) {
                        float y = mish(o[(size_t)2 * q * C1 + c]);
                        if (2 * q + 1 < P) y = std::max(y, mish(o[(size_t)(2 * q + 1) * C1 + c]));
                        out[(size_t)q * C1 + c] = y;
                    }
            }
        });
    }
    // 5x5 convolution over (time, pitch) with time dilation dil on channel-last rows: out[p][co] for one step.
    void conv_step(const std::vector<float> &ring, int ci, int co, const std::vector<float> &w, const float *bias, int dil,
                   int64_t s, float *out) const {
#ifdef TT_KERN
        if (g_kf) {  // the 5 time taps' rows with 2 zero rows each side
            std::vector<float> pad((size_t)5 * (P2 + 4) * ci, 0.f);
            const float *taps[25];
            for (int kt = 0; kt < 5; kt++) {
                const int64_t si = s + (int64_t)dil * (kt - 2);
                float *d = pad.data() + (size_t)kt * (P2 + 4) * ci;
                for (int kp = 0; kp < 5; kp++) taps[kt * 5 + kp] = in_time(si) ? d + kp * ci : nullptr;
                if (in_time(si)) memcpy(d + 2 * ci, ring.data() + (si % RING) * P2 * ci, (size_t)P2 * ci * sizeof(float));
            }
            kern::taps_conv(25, taps, ci, P2, ci, co, w.data(), bias, out);
            return;
        }
#endif
        for (int p = 0; p < P2; p++) std::copy(bias, bias + co, out + (size_t)p * co);
        for (int kt = 0; kt < 5; kt++) {
            const int64_t si = s + (int64_t)dil * (kt - 2);
            if (!in_time(si)) continue;
            const float *in = ring.data() + (si % RING) * P2 * ci;
            for (int p = 0; p < P2; p++) {
                float *op = out + (size_t)p * co;
                for (int kp = 0; kp < 5; kp++) {
                    const int pi = p + kp - 2;
                    if (pi < 0 || pi >= P2) continue;
                    const float *xr = in + (size_t)pi * ci, *wr = w.data() + ((size_t)kt * 5 + kp) * ci * co;
                    for (int i = 0; i < ci; i++) {
                        const float a0 = xr[i];
                        const float *wi = wr + (size_t)i * co;
                        for (int c = 0; c < co; c++) op[c] += a0 * wi[c];
                    }
                }
            }
        }
    }
    void stage2(int64_t a, int64_t b) {
        par(b - a, [&](int64_t i0, int64_t i1) {
            for (int64_t s = a + i0; s < a + i1; s++) {
                float *out = ra2.data() + (s % RING) * P2 * C2;
                conv_step(ra1, C1, C2, w2, b2, 2, s, out);
                mish_n(out, P2 * C2);
            }
        });
    }
    void stage3(int64_t a, int64_t b) {   // conv3, Mish, max and mean over pitch
        par(b - a, [&](int64_t i0, int64_t i1) {
            std::vector<float> o((size_t)P2 * C3);
            for (int64_t s = a + i0; s < a + i1; s++) {
                conv_step(ra2, C2, C3, w3, b3, 4, s, o.data());
                mish_n(o.data(), P2 * C3);
                float *f = rfeat.data() + (s % RING) * NF;
                for (int c = 0; c < C3; c++) {
                    float mx = -INFINITY, sum = 0.f;
                    for (int p = 0; p < P2; p++) {
                        const float y = o[(size_t)p * C3 + c];
                        mx = std::max(mx, y);
                        sum += y;
                    }
                    f[c] = mx;
                    f[C3 + c] = sum / (float)P2;
                }
                const float ph = (float)(std::fmod((double)s * HOP, 3200.0) / 3200.0 * 2.0 * M_PI);   // frame boundaries at 3200 k
                f[2 * C3] = std::sin(ph); f[2 * C3 + 1] = std::cos(ph); f[2 * C3 + 2] = std::sin(2 * ph); f[2 * C3 + 3] = std::cos(2 * ph);
            }
        });
    }
    void stage4(int64_t a, int64_t b) {   // head: conv1d k5 dilation 8, Mish, 1x1, sigmoid; times voiced
        for (int64_t s = a; s < b; s++) {
            float h[64];
            const float *taps[5];
            for (int k = 0; k < 5; k++) {
                const int64_t si = s + 8 * (k - 2);
                taps[k] = in_time(si) ? rfeat.data() + (si % RING) * NF : nullptr;
            }
#ifdef TT_KERN
            if (g_kf) kern::taps_conv(5, taps, NF, 1, NF, HH, wh1.data(), bh1, h);
            else
#endif
            {
                std::copy(bh1, bh1 + HH, h);
                for (int k = 0; k < 5; k++) {
                    if (!taps[k]) continue;
                    const float *f = taps[k], *w = wh1.data() + (size_t)k * NF * HH;
                    for (int i = 0; i < NF; i++)
                        for (int c = 0; c < HH; c++) h[c] += f[i] * w[(size_t)i * HH + c];
                }
            }
            mish_n(h, HH);
            float lg = bh2;
            for (int c = 0; c < HH; c++) lg += wh2[c] * h[c];
            rpv[s % RING] = rclar[s % RING] > 0.5f ? 1.f / (1.f + std::exp(-lg)) : 0.f;
        }
    }
};
