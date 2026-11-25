// AP-BWE 24 kHz -> 48 kHz for streaming (include after tt_engine.cpp; weights: bwe.bin).
//
// Our audio is upsampled 2x (torchaudio's sinc kernel) and kept as it is; the model's change of the spectrum is added
// only from ~11.6 kHz up (g per bin). The same result piece by piece as at once: every layer keeps the frames its
// 7-frame filters still need, so nothing is computed twice. Look-ahead ~52 ms: of the first 133 ms in, ~82 ms come
// out at once, then all of each piece. Each layer runs over all new frames at once: one parallel region per linear
// layer, every thread keeps its slice of the weights in cache for all frames.
#pragma once

#ifdef __aarch64__
// int8 weights (Lin::pack_mm layout) x int8 activations, two activation rows per SMMLA (32 multiply-adds each).
// pa: per row pair p (rows 2p, 2p + 1) and 8 inputs 16 bytes (8 of row 2p, 8 of row 2p + 1); as / axs: per row and
// 64 inputs the activation scale and scale x sum of codes; rows >= nrows are not written.
TT_I8MM static void gemm_q8a8(const Lin &L, const int8_t *pa, const float *as, const float *axs, int64_t npairs, int64_t nrows,
                      float *y, int64_t ys, int64_t q0, int64_t q1) {
    const int64_t G = L.G, KG = L.K / AG;
    for (int64_t q = q0; q < q1; q++) {
        const uint8_t *pq = L.packed + q * G * 528;
        for (int64_t p = 0; p < npairs; p++) {
            const int8_t *b = pa + p * L.K * 2;
            const float *s0 = as + 2 * p * KG, *s1 = s0 + KG, *x0 = axs + 2 * p * KG, *x1 = x0 + KG;
            float32x4_t y0 = vdupq_n_f32(0.f), y1 = y0, m0 = y0, m1 = y0;
            const uint8_t *pp = pq;
            for (int64_t g = 0; g < G; g++, pp += 528) {
                const float32x4_t ws = vcvt_f32_f16(vld1_f16((const float16_t *)(pp + 512)));
                const float32x4_t wm = vcvt_f32_f16(vld1_f16((const float16_t *)(pp + 520)));
                for (int hh = 0; hh < 2; hh++) {
                    const int64_t ga = g * 2 + hh;
                    const int8_t *w01 = (const int8_t *)pp + hh * 128, *w23 = (const int8_t *)pp + 256 + hh * 128, *bb = b + ga * 128;
                    int32x4_t a01 = vdupq_n_s32(0), a23 = a01;
                    for (int c = 0; c < 8; c++) {
                        const int8x16_t bv = vld1q_s8(bb + 16 * c);
                        a01 = vmmlaq_s32(a01, vld1q_s8(w01 + 16 * c), bv);
                        a23 = vmmlaq_s32(a23, vld1q_s8(w23 + 16 * c), bv);
                    }
                    y0 = vfmaq_f32(y0, vcvtq_f32_s32(vuzp1q_s32(a01, a23)), vmulq_n_f32(ws, s0[ga]));
                    y1 = vfmaq_f32(y1, vcvtq_f32_s32(vuzp2q_s32(a01, a23)), vmulq_n_f32(ws, s1[ga]));
                    m0 = vfmaq_n_f32(m0, wm, x0[ga]);
                    m1 = vfmaq_n_f32(m1, wm, x1[ga]);
                }
            }
            float o[2][4];
            vst1q_f32(o[0], vaddq_f32(y0, m0));
            vst1q_f32(o[1], vaddq_f32(y1, m1));
            for (int h = 0; h < 2; h++) {
                const int64_t row = 2 * p + h;
                if (row >= nrows) break;
                for (int r = 0; r < 4; r++) {
                    const int64_t n = q * 4 + r;
                    if (n < L.N) y[row * ys + n] = o[h][r] + (L.bias ? L.bias[n] : 0.f);
                }
            }
        }
    }
}
#endif

struct BWE {
    File f;
    Model *lm = nullptr;
    Pool *pool = nullptr;
    int NF = 1024, HOP = 80, WN = 320, C = 512, NL = 8, KB0 = 248, KP = 3712, NB = 265, OFF = 352, W0 = 513;
    const float *upk = nullptr, *win = nullptr, *g = nullptr;
    struct Block { const float *dwb, *nw, *nb, *gam; std::vector<float> dw; Lin pw1, pw2; };  // dw: [7][C]
    struct Stream { Lin pre; const float *pnw, *pnb, *qnw, *qnb; Block b[8]; } s[2];
    Lin post_m, post_r, post_i;
    std::vector<float> ct, st;   // FFT twiddles
    std::vector<int> rev;
    std::vector<Act> acts;
#ifdef TT_KERN
    kern::QBuf kq;  // activation codes of the kern kernels
#endif
    std::vector<int8_t> pairs;
    std::vector<float> asc, axs;
    bool a8 = true;   // int8 activations (two rows per SMMLA); false: the engine's 16-bit activations
    double t_net = 0, t_fft = 0, t_lin = 0;

    // streaming state (absolute sample / frame indices)
    std::vector<float> in24, x48, ola, env, xin, buf[9][2], X, Hb, Y, Y2, fr, D;
    int64_t a24 = 0, n24 = 0, np = 0;               // in24[0] is sample a24; received; upsampled pairs
    int64_t a48 = 0, nf = 0;                        // x48[0] is sample a48; STFT frames made
    int64_t b0[9] = {}, nin[9] = {}, nxt[9] = {};   // per layer: first kept input frame, inputs, next output frame
    int64_t ax = 0, npost = 0, aout = 0, nout = 0;  // xin[0] is frame ax; frames done; ola[0] is sample aout; samples out
    int64_t T = -1;                                 // frames in all (known at the end)

    bool load(const char *path, Model *m) {
        if (!f.open(path)) return false;
        lm = m;
        pool = m->pool;
        const int32_t *c = (const int32_t *)f.get("cfg").p;
        NF = c[0]; HOP = c[1]; WN = c[2]; C = c[3]; NL = c[4]; KB0 = c[5]; KP = c[6];
        W0 = NF / 2 + 1; NB = W0 - KB0; OFF = (NF - WN) / 2;
        auto F = [&](const std::string &n) { return (const float *)f.get(n).p; };
        auto L = [&](Lin &l, const std::string &n) {
            l.load(f, n);
            if (lm->opt.neon) { if (lm->opt.mm) l.pack_mm(); else l.pack(); }
            else l.NQ = (l.N + 3) / 4;
        };
        upk = F("up_k"); win = F("window"); g = F("g");
        for (int k = 0; k < 2; k++) {
            const std::string p = k ? "p." : "m.";
            Stream &S = s[k];
            L(S.pre, p + "pre"); S.pnw = F(p + "pre_norm.w"); S.pnb = F(p + "pre_norm.b");
            S.qnw = F(p + "post_norm.w"); S.qnb = F(p + "post_norm.b");
            for (int b = 0; b < NL; b++) {
                const std::string q = p + "b" + std::to_string(b) + ".";
                Block &B_ = S.b[b];
                const float *dw = F(q + "dw.w");
                B_.dw.resize(7 * C);
                for (int j = 0; j < 7; j++) for (int cc = 0; cc < C; cc++) B_.dw[j * C + cc] = dw[cc * 7 + j];
                B_.dwb = F(q + "dw.b"); B_.nw = F(q + "norm.w"); B_.nb = F(q + "norm.b"); B_.gam = F(q + "gamma");
                L(B_.pw1, q + "pw1"); L(B_.pw2, q + "pw2");
            }
        }
        L(post_m, "post_m"); L(post_r, "post_r"); L(post_i, "post_i");
        ct.resize(NF / 2); st.resize(NF / 2); rev.resize(NF);
        for (int k = 0; k < NF / 2; k++) { ct[k] = (float)std::cos(2 * M_PI * k / NF); st[k] = (float)std::sin(2 * M_PI * k / NF); }
        int bits = 0;
        while ((1 << bits) < NF) bits++;
        for (int i = 0; i < NF; i++) { int r = 0; for (int b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b); rev[i] = r; }
        reset();
        return true;
    }
    size_t bytes() const {
        size_t b = post_m.packed_bytes() + post_r.packed_bytes() + post_i.packed_bytes();
        for (auto &S : s) { b += S.pre.packed_bytes(); for (int i = 0; i < NL; i++) b += S.b[i].pw1.packed_bytes() + S.b[i].pw2.packed_bytes(); }
        return b;
    }
    // The streaming state as a value (the redraw replaces audio: back to the state before a frame, then on again).
    struct State {
        std::vector<float> in24, x48, ola, env, xin, buf[9][2];
        int64_t a24, n24, np, a48, nf, b0[9], nin[9], nxt[9], ax, npost, aout, nout, T;
    };
    void save(State &s) const {
        s.in24 = in24; s.x48 = x48; s.ola = ola; s.env = env; s.xin = xin;
        for (int l = 0; l < 9; l++) { s.buf[l][0] = buf[l][0]; s.buf[l][1] = buf[l][1]; s.b0[l] = b0[l]; s.nin[l] = nin[l]; s.nxt[l] = nxt[l]; }
        s.a24 = a24; s.n24 = n24; s.np = np; s.a48 = a48; s.nf = nf; s.ax = ax; s.npost = npost; s.aout = aout; s.nout = nout; s.T = T;
    }
    void load(const State &s) {
        in24 = s.in24; x48 = s.x48; ola = s.ola; env = s.env; xin = s.xin;
        for (int l = 0; l < 9; l++) { buf[l][0] = s.buf[l][0]; buf[l][1] = s.buf[l][1]; b0[l] = s.b0[l]; nin[l] = s.nin[l]; nxt[l] = s.nxt[l]; }
        a24 = s.a24; n24 = s.n24; np = s.np; a48 = s.a48; nf = s.nf; ax = s.ax; npost = s.npost; aout = s.aout; nout = s.nout; T = s.T;
    }
    void reset() {
        in24.clear(); x48.clear(); ola.clear(); env.clear(); xin.clear();
        for (auto &l : buf) { l[0].clear(); l[1].clear(); }
        a24 = n24 = np = a48 = nf = ax = npost = aout = nout = 0;
        for (int l = 0; l < 9; l++) b0[l] = nin[l] = nxt[l] = 0;
        T = -1;
    }

    // in-place complex FFT of size NF (sign -1 forward, +1 inverse, unscaled)
    void fft(float *re, float *im, int sign) const {
        for (int i = 0; i < NF; i++) if (i < rev[i]) { std::swap(re[i], re[rev[i]]); std::swap(im[i], im[rev[i]]); }
        for (int len = 2; len <= NF; len <<= 1) {
            const int h = len / 2, step = NF / len;
            for (int i = 0; i < NF; i += len)
                for (int j = 0; j < h; j++) {
                    const float wr = ct[j * step], wi = sign * st[j * step];
                    float *ar = re + i + j, *ai = im + i + j, *br = re + i + j + h, *bi = im + i + j + h;
                    const float tr = *br * wr - *bi * wi, ti = *br * wi + *bi * wr;
                    *br = *ar - tr; *bi = *ai - ti; *ar += tr; *ai += ti;
                }
        }
    }

    float x48at(int64_t i, int64_t L48) const {   // with torch.stft's reflect padding at both ends (L48 < 0: end not known)
        if (i < 0) i = -i;
        if (L48 >= 0 && i >= L48) i = 2 * (L48 - 1) - i;
        return x48[i - a48];
    }

    static void layernorm6(const float *x, const float *w, const float *b, int n, float *out) {
        float mean = 0.f, var = 0.f;
        for (int i = 0; i < n; i++) mean += x[i];
        mean /= n;
        for (int i = 0; i < n; i++) var += (x[i] - mean) * (x[i] - mean);
        const float r = 1.f / std::sqrt(var / n + 1e-6f);
        for (int i = 0; i < n; i++) out[i] = (x[i] - mean) * r * w[i] + b[i];
    }
    static void gelu(float *x, int64_t n) {   // erf form (torch nn.GELU): rational erf, max error ~1e-7
        for (int64_t i = 0; i < n; i++) {
            const float v = x[i], z = std::min(4.f, std::max(-4.f, v * 0.70710678f)), z2 = z * z;
            float p = -2.72614225801306e-10f;
            p = p * z2 + 2.77068142495902e-08f; p = p * z2 - 2.10102402082508e-06f; p = p * z2 - 5.69250639462346e-05f;
            p = p * z2 - 7.34990630326855e-04f; p = p * z2 - 2.95459980854025e-03f; p = p * z2 - 1.60960333262415e-02f;
            float q = -1.45660718464996e-05f;
            q = q * z2 - 2.13374055278905e-04f; q = q * z2 - 1.68282697438203e-03f; q = q * z2 - 7.37332916720468e-03f;
            q = q * z2 - 1.42647390514189e-02f;
            x[i] = 0.5f * v * (1.f + z * p / q);
        }
    }

    // y[r][0..N) = L(x[r]) for n rows (rows xs / ys apart): one parallel region, each thread a slice of the weights
    void blin(const Lin &L, const float *x, int64_t xs, int64_t n, float *y, int64_t ys) {
        const double t0 = now_s();
#ifdef __EMSCRIPTEN__
        web_lin_rows(pool, L, x, xs, n, y, ys, g_bwe_rows);
        t_lin += now_s() - t0;
        return;
#endif
#ifdef TT_KERN
#ifdef TT_X86
        if (g_kf && L.kind == 2) {
            pool->run(L.N, 8, [&](int64_t a, int64_t b) { kern::linear_rows(L.wf, L.bias, L.K, x, xs, n, y, ys, a, b); });
            t_lin += now_s() - t0;
            return;
        }
        if (g_kf && L.kind != 2) {
#else
        if (g_kf && L.kind != 2) {
#endif
            kern::QW w;
            w.kind = L.kind; w.N = L.N; w.K = L.K; w.codes = L.codes; w.scales = L.scales; w.mins = L.mins; w.bias = L.bias;
            kern::lin_q_f32(pool_par(pool), w, x, xs, n, y, ys, kq);
            t_lin += now_s() - t0;
            return;
        }
#endif
        if (L.kind == 2) {
            pool->run(L.N, 8, [&](int64_t a, int64_t b) {
                for (int64_t o = a; o < b; o++) {
                    const float *w = L.wf + o * L.K;
                    for (int64_t r = 0; r < n; r++) {
                        const float *xr = x + r * xs;
                        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                        int64_t k = 0;
                        for (; k + 8 <= L.K; k += 8)
                            for (int j = 0; j < 8; j++) acc[j] += w[k + j] * xr[k + j];
                        float t = ((acc[0] + acc[4]) + (acc[2] + acc[6])) + ((acc[1] + acc[5]) + (acc[3] + acc[7]));
                        for (; k < L.K; k++) t += w[k] * xr[k];
                        y[r * ys + o] = t + (L.bias ? L.bias[o] : 0.f);
                    }
                }
            });
            t_lin += now_s() - t0;
            return;
        }
        if ((int64_t)acts.size() < n) acts.resize(n);
        for (int64_t r = 0; r < n; r++) if ((int64_t)acts[r].q.size() < KP) acts[r].resize(KP);
        const bool mm = lm->opt.mm;
#ifdef __aarch64__
        if (mm && a8) {
            const int64_t KG = L.K / AG, npairs = (n + 1) / 2;
            pairs.resize((size_t)npairs * L.K * 2);
            asc.assign((size_t)npairs * 2 * KG, 0.f);
            axs.assign((size_t)npairs * 2 * KG, 0.f);
            pool->run(npairs, 2, [&](int64_t a, int64_t b) {
                for (int64_t p = a; p < b; p++) {
                    int8_t *d = pairs.data() + p * L.K * 2;
                    for (int h = 0; h < 2; h++) {
                        const int64_t r = 2 * p + h;
                        if (r >= n) { for (int64_t c = 0; c < L.K / 8; c++) memset(d + c * 16 + h * 8, 0, 8); continue; }
                        quantize(x + r * xs, L.K, acts[r], 8, false);
                        for (int64_t c = 0; c < L.K / 8; c++) memcpy(d + c * 16 + h * 8, acts[r].hi.data() + c * 8, 8);
                        memcpy(asc.data() + r * KG, acts[r].s.data(), KG * sizeof(float));
                        memcpy(axs.data() + r * KG, acts[r].xs.data(), KG * sizeof(float));
                    }
                }
            });
            pool->run(L.NQ, 4, [&](int64_t a, int64_t b) {
                gemm_q8a8(L, pairs.data(), asc.data(), axs.data(), npairs, n, y, ys, a, b);
            }, true);
            t_lin += now_s() - t0;
            return;
        }
#endif
        pool->run(n, 4, [&](int64_t a, int64_t b) { for (int64_t r = a; r < b; r++) quantize(x + r * xs, L.K, acts[r], 16, mm); });
#ifdef __aarch64__
        if (mm) {
            pool->run(L.NQ, 4, [&](int64_t a, int64_t b) {
                for (int64_t r = 0; r < n; r += MB) gemv_q8_mm(L, acts.data() + r, (int)std::min<int64_t>(MB, n - r), y + r * ys, ys, a, b);
            }, true);
            t_lin += now_s() - t0;
            return;
        }
#endif
        pool->run(L.NQ, 4, [&](int64_t a, int64_t b) {
            for (int64_t r = 0; r < n; r++) gemv_scalar(L, acts[r], y + r * ys, a * 4, std::min<int64_t>(b * 4, L.N));
        });
        t_lin += now_s() - t0;
    }

    // ---- the pieces of the pipeline
    void upsample(bool fin) {
        const int64_t end = fin ? n24 : n24 - 7;
        if (end <= np) return;
        const size_t o = x48.size();
        x48.resize(o + 2 * (size_t)(end - np));
        for (int64_t n = np; n < end; n++)
            for (int i = 0; i < 2; i++) {
                float acc = 0.f;
                for (int k = 0; k < 15; k++) {
                    const int64_t j = n + k - 7;
                    if (j >= 0 && j < n24) acc += in24[j - a24] * upk[i * 15 + k];
                }
                x48[o + 2 * (n - np) + i] = acc;
            }
        np = end;
        const int64_t keep = std::max<int64_t>(0, np - 7);   // pair np needs samples from np - 7
        in24.erase(in24.begin(), in24.begin() + (keep - a24));
        a24 = keep;
    }

    void stft(bool fin) {
        const int64_t L48 = 2 * np;
        const int64_t end = fin ? 1 + L48 / HOP : L48 > WN / 2 ? (L48 - WN / 2 - 1) / HOP + 1 : 0;   // frame t reads up to t HOP + WN / 2 - 1
        if (end <= nf) return;
        const int64_t n = end - nf;
        for (int k = 0; k < 2; k++) buf[0][k].resize(buf[0][k].size() + (size_t)n * W0);
        const size_t o0 = (size_t)(nin[0] - b0[0]) * W0, ox = xin.size();
        xin.resize(ox + (size_t)n * 3 * NB);
        fr.resize((size_t)n * 2 * NF);
        const double t0 = now_s();
        pool->run(n, 2, [&](int64_t a, int64_t b) {
            for (int64_t q = a; q < b; q++) {
                float *re = fr.data() + q * 2 * NF, *im = re + NF;
                std::fill(re, re + 2 * NF, 0.f);
                const int64_t t = nf + q;
                for (int k = 0; k < WN; k++) re[OFF + k] = x48at(t * HOP - WN / 2 + k, fin ? L48 : -1) * win[k];
                fft(re, im, -1);
                float *la = buf[0][0].data() + o0 + q * W0, *ph = buf[0][1].data() + o0 + q * W0, *xs = xin.data() + ox + q * 3 * NB;
                for (int k = 0; k < W0; k++) {
                    la[k] = std::log(std::sqrt(re[k] * re[k] + im[k] * im[k]) + 1e-4f);
                    ph[k] = std::atan2(im[k], re[k]);
                }
                for (int k = 0; k < NB; k++) { xs[k] = re[KB0 + k]; xs[NB + k] = im[KB0 + k]; xs[2 * NB + k] = la[KB0 + k]; }
            }
        });
        t_fft += now_s() - t0;
        nf = end;
        nin[0] = nf;
        trim48();
    }
    void trim48() {   // x48 is still needed from the next STFT frame's first sample and from the next sample out
        const int64_t keep = std::max<int64_t>(0, std::min(nf * HOP - WN / 2, nout));
        if (keep > a48) { x48.erase(x48.begin(), x48.begin() + (keep - a48)); a48 = keep; }
    }

    // frames [t0, t1) of layer l (0: conv_pre, 1..NL: blocks), both streams -> out[k] [n][C]
    void layer(int l, int64_t t0, int64_t t1, std::vector<float> out[2]) {
        const int64_t n = t1 - t0;
        const int wi = l == 0 ? W0 : C;
        for (int k = 0; k < 2; k++) {
            const Stream &S = s[k];
            const std::vector<float> &in = buf[l][k];
            auto at = [&](int64_t t) -> const float * { return (t < 0 || (T >= 0 && t >= T) || t >= nin[l]) ? nullptr : in.data() + (size_t)(t - b0[l]) * wi; };
            out[k].resize((size_t)n * C);
            if (l == 0) {
                X.resize((size_t)n * KP);
                pool->run(n, 4, [&](int64_t a, int64_t b) {
                    for (int64_t m = a; m < b; m++) {
                        float *row = X.data() + m * KP;
                        std::fill(row, row + KP, 0.f);
                        for (int j = 0; j < 7; j++) if (const float *p = at(t0 + m + j - 3)) memcpy(row + j * wi, p, wi * sizeof(float));
                    }
                });
                Y.resize((size_t)n * C);
                blin(S.pre, X.data(), KP, n, Y.data(), C);
                pool->run(n, 8, [&](int64_t a, int64_t b) {
                    for (int64_t m = a; m < b; m++) layernorm6(Y.data() + m * C, S.pnw, S.pnb, C, out[k].data() + m * C);
                });
                continue;
            }
            const Block &Bk = S.b[l - 1];
            const int I = (int)Bk.pw1.N;
            X.resize((size_t)n * C); Hb.resize((size_t)n * I); Y.resize((size_t)n * C);
            pool->run(n, 4, [&](int64_t a, int64_t b) {   // depthwise filter over 7 frames, then the norm
                std::vector<float> d(C);
                for (int64_t m = a; m < b; m++) {
                    for (int c = 0; c < C; c++) d[c] = Bk.dwb[c];
                    for (int j = 0; j < 7; j++)
                        if (const float *p = at(t0 + m + j - 3)) {
                            const float *w = Bk.dw.data() + j * C;
                            for (int c = 0; c < C; c++) d[c] += w[c] * p[c];
                        }
                    layernorm6(d.data(), Bk.nw, Bk.nb, C, X.data() + m * C);
                }
            });
            blin(Bk.pw1, X.data(), C, n, Hb.data(), I);
            pool->run(n, 2, [&](int64_t a, int64_t b) { gelu(Hb.data() + a * I, (b - a) * I); });
            blin(Bk.pw2, Hb.data(), I, n, Y.data(), C);
            pool->run(n, 8, [&](int64_t a, int64_t b) {
                for (int64_t m = a; m < b; m++) {
                    const float *r = at(t0 + m), *h = Y.data() + m * C;
                    float *y = out[k].data() + m * C;
                    for (int c = 0; c < C; c++) y[c] = r[c] + Bk.gam[c] * h[c];
                }
            });
        }
    }

    void network(bool fin) {
        const double t0 = now_s();
        for (int l = 0; l <= NL; l++) {
            const int64_t end = fin ? T : std::max<int64_t>(nxt[l], nin[l] - 3);
            if (end > nxt[l]) {
                std::vector<float> out[2];
                layer(l, nxt[l], end, out);
                const int64_t n = end - nxt[l];
                if (l < NL) {
                    for (int64_t q = 0; q < n; q++) {   // x_m = x_m + x_p; x_p = x_p + x_m (before every block)
                        float *xm = out[0].data() + q * C, *xp = out[1].data() + q * C;
                        for (int c = 0; c < C; c++) { xm[c] += xp[c]; xp[c] += xm[c]; }
                    }
                    for (int k = 0; k < 2; k++) buf[l + 1][k].insert(buf[l + 1][k].end(), out[k].begin(), out[k].end());
                    nin[l + 1] = end;
                } else {
                    post(out, nxt[l], end);
                }
                nxt[l] = end;
            }
            const int wi = l == 0 ? W0 : C;
            const int64_t keep = std::max<int64_t>(b0[l], nxt[l] - 3);
            for (int k = 0; k < 2; k++) buf[l][k].erase(buf[l][k].begin(), buf[l][k].begin() + (size_t)(keep - b0[l]) * wi);
            b0[l] = keep;
        }
        t_net += now_s() - t0;
    }

    // last layers for frames [t0, t1): spectrum change times g -> inverse STFT, overlap-add
    void post(std::vector<float> out[2], int64_t t0, int64_t t1) {
        const int64_t n = t1 - t0;
        X.resize((size_t)n * C); Y.resize((size_t)n * C);
        pool->run(n, 8, [&](int64_t a, int64_t b) {
            for (int64_t m = a; m < b; m++) {
                layernorm6(out[0].data() + m * C, s[0].qnw, s[0].qnb, C, X.data() + m * C);
                layernorm6(out[1].data() + m * C, s[1].qnw, s[1].qnb, C, Y.data() + m * C);
            }
        });
        Hb.resize((size_t)n * 3 * NB);
        blin(post_m, X.data(), C, n, Hb.data(), 3 * NB);
        blin(post_r, Y.data(), C, n, Hb.data() + NB, 3 * NB);
        blin(post_i, Y.data(), C, n, Hb.data() + 2 * NB, 3 * NB);
        fr.resize((size_t)n * 2 * NF);
        const double ta = now_s();
        pool->run(n, 2, [&](int64_t a, int64_t b) {
            for (int64_t q = a; q < b; q++) {
                const int64_t t = t0 + q;
                const float *xs = xin.data() + (size_t)(t - ax) * 3 * NB, *ml = Hb.data() + q * 3 * NB, *pr = ml + NB, *pi = ml + 2 * NB;
                float *re = fr.data() + q * 2 * NF, *im = re + NF;
                std::fill(re, re + 2 * NF, 0.f);
                for (int k = 0; k < NB; k++) {
                    const float amp = std::exp(xs[2 * NB + k] + ml[k]), nn = std::sqrt(pr[k] * pr[k] + pi[k] * pi[k]);
                    const float cr = nn > 0.f ? pr[k] / nn : 1.f, ci = nn > 0.f ? pi[k] / nn : 0.f;
                    const float dr = (amp * cr - xs[k]) * g[k], di = (amp * ci - xs[NB + k]) * g[k];
                    const int bin = KB0 + k;
                    re[bin] = dr; im[bin] = bin == NF / 2 ? 0.f : di;
                    if (bin > 0 && bin < NF / 2) { re[NF - bin] = dr; im[NF - bin] = -di; }
                }
                fft(re, im, +1);
            }
        });
        t_fft += now_s() - ta;
        // overlap-add: frame t's sample OFF + j is x48 sample t HOP - WN / 2 + j
        const int64_t need = (t1 - 1) * HOP + WN / 2 - aout;
        if ((int64_t)ola.size() < need) { ola.resize(need, 0.f); env.resize(need, 0.f); }
        for (int64_t q = 0; q < n; q++) {
            const int64_t base = (t0 + q) * HOP - WN / 2;
            const float *re = fr.data() + q * 2 * NF + OFF;
            for (int j = 0; j < WN; j++) {
                const int64_t i = base + j;
                if (i < aout) continue;
                ola[i - aout] += re[j] * (1.f / NF) * win[j];
                env[i - aout] += win[j] * win[j];
            }
        }
        npost = t1;
        xin.erase(xin.begin(), xin.begin() + (size_t)(npost - ax) * 3 * NB);
        ax = npost;
    }

    void emit(bool fin, std::vector<float> &out48) {
        const int64_t L48 = 2 * np;
        const int64_t end = fin ? L48 : std::min(L48, npost * HOP - WN / 2);
        if (end <= nout) return;
        for (int64_t i = nout; i < end; i++) {
            const float e = i - aout < (int64_t)env.size() ? env[i - aout] : 0.f;
            const float h = e > 1e-11f ? ola[i - aout] / e : 0.f;
            out48.push_back(x48[i - a48] + h);
        }
        nout = end;
        const int64_t drop = std::min<int64_t>(nout - aout, (int64_t)ola.size());
        ola.erase(ola.begin(), ola.begin() + drop); env.erase(env.begin(), env.begin() + drop);
        aout += drop;
        trim48();
    }

    // n new samples at 24 kHz; appends every finished 48 kHz sample to out48
    void push(const float *x, int64_t n, std::vector<float> &out48) {
        in24.insert(in24.end(), x, x + n);
        n24 += n;
        upsample(false);
        stft(false);
        network(false);
        emit(false, out48);
    }
    void finish(std::vector<float> &out48) {
        upsample(true);
        T = 1 + 2 * np / HOP;
        stft(true);
        network(true);
        emit(true, out48);
        reset();
    }
};
