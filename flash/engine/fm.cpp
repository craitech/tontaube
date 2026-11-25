// Frame model (a Qwen3-0.6B backbone over the autoencoder's code frames, flow head) on the CPU engine:
// text chunks -> code frames -> VibeVoice latents, frame by frame (streaming autoencoder decoder).
//
// A case file (optional, for checks): the chunks' text, the sampler settings, start noise, frames, hidden states and
// latents of another implementation; forced: its frames are fed back (every state is comparable); free: the own frames.
//
// Chunk scheme: [previous chunk text] TM [chunk text] TM [next chunk, first 50 characters] START
// [previous chunk frames] AM [chunk frames]; text one position per character, frames `rate` positions apart.
#include "tt_engine.cpp"

struct OutFile {  // TTLM writer (float32 / int32 tensors)
    struct E { std::string name; int dtype; std::vector<uint64_t> dims; std::vector<uint8_t> data; };
    std::vector<E> e;
    void put(const std::string &name, int dtype, const void *p, std::vector<uint64_t> dims) {
        size_t n = 4;
        for (uint64_t d : dims) n *= d;
        E x{name, dtype, dims, std::vector<uint8_t>(n)};
        memcpy(x.data.data(), p, n);
        e.push_back(std::move(x));
    }
    void f32(const std::string &name, const float *p, std::vector<uint64_t> dims) { put(name, 3, p, dims); }
    void i32(const std::string &name, const int32_t *p, std::vector<uint64_t> dims) { put(name, 4, p, dims); }
    bool save(const char *path) const {
        FILE *f = fopen(path, "wb");
        if (!f) return false;
        const uint32_t n = (uint32_t)e.size(), z = 0;
        fwrite("TTLM0001", 1, 8, f);
        fwrite(&n, 4, 1, f);
        fwrite(&z, 4, 1, f);
        uint64_t off = (16 + (uint64_t)n * 104 + 63) / 64 * 64;
        for (const E &x : e) {
            char name[48] = {0};
            strncpy(name, x.name.c_str(), 47);
            const uint32_t dt = x.dtype, nd = (uint32_t)x.dims.size();
            uint64_t dims[4] = {0, 0, 0, 0};
            for (size_t i = 0; i < x.dims.size(); i++) dims[i] = x.dims[i];
            const uint64_t nb = x.data.size();
            fwrite(name, 1, 48, f); fwrite(&dt, 4, 1, f); fwrite(&nd, 4, 1, f); fwrite(dims, 8, 4, f);
            fwrite(&off, 8, 1, f); fwrite(&nb, 8, 1, f);
            off = (off + nb + 63) / 64 * 64;
        }
        for (const E &x : e) {
            const long pad = (64 - ftell(f) % 64) % 64;
            for (long i = 0; i < pad; i++) fputc(0, f);
            fwrite(x.data.data(), 1, x.data.size(), f);
        }
        fclose(f);
        return true;
    }
};

static inline float mish1(float v) { return v * std::tanh(v > 20.f ? v : std::log1p(std::exp(v))); }
static void mish(float *x, int64_t n) {
#ifdef TT_KERN
    if (g_kf) { kern::mish(x, n); return; }
#endif
    for (int64_t i = 0; i < n; i++) x[i] = mish1(x[i]);
}
// torch LayerNorm (eps 1e-5, biased variance); w / b null: no affine.
static void layernorm5(const float *x, const float *w, const float *b, int64_t n, float *out) {
#ifdef TT_KERN
    if (g_kf) { kern::layernorm(x, w, b, n, 1e-5f, out); return; }
#endif
    double mean = 0, var = 0;
    for (int64_t i = 0; i < n; i++) mean += x[i];
    mean /= (double)n;
    for (int64_t i = 0; i < n; i++) var += (x[i] - mean) * (x[i] - mean);
    const float r = 1.f / std::sqrt((float)(var / (double)n) + 1e-5f), mu = (float)mean;
    if (w) for (int64_t i = 0; i < n; i++) out[i] = (x[i] - mu) * r * w[i] + b[i];
    else for (int64_t i = 0; i < n; i++) out[i] = (x[i] - mu) * r;
}

typedef std::vector<int32_t> Text;  // code points

// Standard normal floats exactly as libc++'s std::normal_distribution<float> draws them from the generator (the
// browser and phone builds), so a seed gives the same noise in every build (libstdc++ hands out each pair of the polar
// method in the other order).
struct Gauss {
    float saved = 0.f;
    bool hot = false;
    void reset() { hot = false; }
    static float canonical(std::mt19937_64 &g) { return (float)g() / 18446744073709551616.f; }  // generate_canonical<float, 24>
    float operator()(std::mt19937_64 &g) {
        if (hot) { hot = false; return saved * 1.f + 0.f; }
        float u, v, s;
        do {
            u = 2.f * canonical(g) + -1.f;
            v = 2.f * canonical(g) + -1.f;
            s = u * u + v * v;
        } while (s > 1.f || s == 0.f);
        const float f = std::sqrt(-2.f * std::log(s) / s);
        saved = v * f;
        hot = true;
        return u * f * 1.f + 0.f;
    }
};

static std::vector<Text> read_chunks(const char *path) {  // one chunk per line, UTF-8
    std::vector<Text> out;
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot read %s\n", path); exit(2); }
    Text cur;
    int c;
    auto flush = [&]() { if (!cur.empty()) out.push_back(cur); cur.clear(); };
    while ((c = fgetc(f)) != EOF) {
        if (c == '\n') { flush(); continue; }
        if (c == '\r') continue;
        int32_t cp = c;
        int more = 0;
        if (c >= 0xF0) { cp = c & 7; more = 3; } else if (c >= 0xE0) { cp = c & 15; more = 2; } else if (c >= 0xC0) { cp = c & 31; more = 1; }
        for (; more > 0; more--) cp = (cp << 6) | (fgetc(f) & 63);
        cur.push_back(cp);
    }
    flush();
    fclose(f);
    return out;
}

struct FM {
    Model lm;
    File fl, fm;
    int64_t D = 16, H = 1024, FW = 1024, NB = 3, TD = 256, AW = 1024, AB = 4, NCH = 0, LOOK = 50, OFFR = 10, MINF = 2, W = 0;
    float rate = 2.f, text_off = 75.f;
    const int32_t *chars = nullptr, *ak = nullptr;
    const float *char_emb, *start, *tmark, *amark, *cnw, *cnb, *cmu, *csd, *amu, *asd, *anw, *anb, *bnw[8], *bnb[8];
    Lin fin0, fin2, x_in, c_in, t0, t2, mod[8], ma[8], mb[8], out_mod, out, stop, ae_in, ae_conv[8], ae_mix[8], ae_out;
    std::vector<float> b0, b1, ws, tm, modb, outm, ring[8];
    bool euler = false;
    int passes = 6;
    int64_t unknown = 0;
    double t_prefill = 0, t_lm = 0, t_head = 0, t_ae = 0, t_embed = 0;
    // the text reading of a chunk's prefix elsewhere (emb, pos, n, cache row base, out: last hidden state): writes the
    // cache rows base..base+n-1 and returns true, or false to read it here
    std::function<bool(const float *, const float *, int64_t, int64_t, float *)> prefill_hook;
    // the next chunk's prefix text read elsewhere while this chunk's first frames are made (started at the first
    // frame step with enough audio ahead; ahead_wait_hook returns when its cache rows are in place, false if they
    // could not be made: then the rows are read here in pieces as without it)
    std::function<bool(const float *, const float *, int64_t, int64_t)> ahead_start_hook;
    std::function<bool()> ahead_wait_hook;
    std::function<bool()> ahead_ready_hook;  // without waiting: has it finished (done or failed)?
    bool g_ahead_pending = false, g_ahead_due = false;
    void ahead_wait() {
        g_ahead_pending = false;
        if (!ahead_wait_hook()) g_s_fed = 0;
    }

    void lin(const Lin &L, const float *x, float *y, int M = 1) { lm.linear(L, x, y, M); }
    // mish over the pool (the same scalar function per element: the result does not depend on the threads)
    void mish_p(float *x, int64_t n) {
#ifdef TT_KERN
        if (g_kf) {  // vector Mish: a few thousand values take about a microsecond, less than a pool region
            lm.pool->run((n + 4095) / 4096, 1, [&](int64_t a, int64_t b) { mish(x + a * 4096, std::min(n, b * 4096) - a * 4096); });
            return;
        }
#endif
        lm.pool->run((n + 127) / 128, 1, [&](int64_t a, int64_t b) {
            for (int64_t i = a * 128; i < std::min(n, b * 128); i++) x[i] = mish1(x[i]);
        });
    }

    void load(const char *pl, const char *pm, Pool *pool, const Options &opt, int64_t max_len) {
        if (!fl.open(pl) || !fm.open(pm)) { fprintf(stderr, "cannot read the model files\n"); exit(2); }
        lm.pool = pool;
        lm.opt = opt;
        lm.load(fl, max_len);
        if (opt.neon) lm.pack_all();
        const int32_t *c = (const int32_t *)fm.get("cfg_misc").p;
        D = c[0]; H = c[1]; FW = c[2]; NB = c[3]; TD = c[4]; AW = c[5]; AB = c[6]; NCH = c[7]; LOOK = c[8]; OFFR = c[9]; MINF = c[10];
        const float *cf = (const float *)fm.get("cfg_miscf").p;
        rate = cf[0]; text_off = cf[1];
        if (H != lm.H || NB > 8 || AB > 8) { fprintf(stderr, "model files do not fit together\n"); exit(2); }
        auto F = [&](const std::string &n) { return (const float *)fm.get(n).p; };
        chars = (const int32_t *)fm.get("chars").p;
        ak = (const int32_t *)fm.get("ae.k").p;
        char_emb = F("char_emb"); start = F("start"); tmark = F("text_marker"); amark = F("audio_marker");
        cnw = F("fl.c_norm.w"); cnb = F("fl.c_norm.b"); cmu = F("cmu"); csd = F("csd"); amu = F("ae.mu"); asd = F("ae.sd");
        anw = F("ae.norm.w"); anb = F("ae.norm.b");
        auto L = [&](Lin &l, const std::string &n) {
            l.load(fm, n);
            if (opt.neon) { if (opt.mm) l.pack_mm(); else l.pack(); }
        };
        L(fin0, "frame_in0"); L(fin2, "frame_in2"); L(x_in, "fl.x_in"); L(c_in, "fl.c_in"); L(t0, "fl.t0"); L(t2, "fl.t2");
        L(out_mod, "fl.out_mod"); L(out, "fl.out"); L(stop, "stop"); L(ae_in, "ae.in"); L(ae_out, "ae.out");
        for (int b = 0; b < NB; b++) {
            const std::string s = std::to_string(b);
            L(mod[b], "fl.mod" + s); L(ma[b], "fl.mlp" + s + "a"); L(mb[b], "fl.mlp" + s + "b");
        }
        for (int b = 0; b < AB; b++) {
            const std::string p = "ae.b" + std::to_string(b) + ".";
            bnw[b] = F(p + "norm.w"); bnb[b] = F(p + "norm.b");
            L(ae_conv[b], p + "conv"); L(ae_mix[b], p + "mix");
            ring[b].assign((size_t)ak[b] * AW, 0.f);
        }
        W = lm.W;
        for (auto *v : {&b0, &b1, &ws}) v->assign((size_t)MB * W, 0.f);
        if (fm.has("first")) set_first((const float *)fm.get("first").p, (int64_t)fm.get("first").dims[0]);
        if (fm.has("prime")) {
            const Tensor &pt = fm.get("prime");
            prime.assign((const float *)pt.p, (const float *)pt.p + pt.dims[0] * 64);
        }
    }
    size_t weight_bytes() const {
        size_t b = 0;
        for (auto &l : lm.layers) b += l.qkv.packed_bytes() + l.o.packed_bytes() + l.gate_up.packed_bytes() + l.down.packed_bytes();
        return b;
    }
    size_t misc_bytes() const {
        size_t b = 0;
        for (const Lin *l : {&fin0, &fin2, &x_in, &c_in, &out_mod, &out, &ae_in, &ae_out}) b += l->packed_bytes();
        for (int i = 0; i < NB; i++) b += mod[i].packed_bytes() + ma[i].packed_bytes() + mb[i].packed_bytes();
        for (int i = 0; i < AB; i++) b += ae_conv[i].packed_bytes() + ae_mix[i].packed_bytes();
        return b;
    }

    // The sampler's time vectors t_in(sinusoid(t)), one per head pass: Euler t = i / steps; midpoint t = i / steps and
    // t + 1 / (2 steps).
    void set_sampler(bool euler_, int passes_) {
        euler = euler_;
        passes = euler ? passes_ : 2 * std::max(1, passes_ / 2);
        tm.assign((size_t)passes * FW, 0.f);
        modb.assign((size_t)NB * passes * 3 * FW, 0.f);
        outm.assign((size_t)passes * 2 * FW, 0.f);
        const int steps = euler ? passes : passes / 2;
        const int64_t half = TD / 2;
        std::vector<float> e(TD), y(FW);
        for (int p = 0; p < passes; p++) {
            const float t = euler ? (float)((double)p / steps)
                                  : (float)((double)(p / 2) / steps) + (p % 2 ? (float)(1.0 / (2 * steps)) : 0.f);
            for (int64_t k = 0; k < half; k++) {
                const float a = 1000.f * t * std::exp(-(float)std::log(10000.0) * (float)k / (float)half);
                e[k] = std::cos(a);
                e[half + k] = std::sin(a);
            }
            lin(t0, e.data(), y.data());
            mish(y.data(), FW);
            lin(t2, y.data(), tm.data() + (size_t)p * FW);
        }
    }

    // One code frame [D] from the final-norm state hn and the start noise (already times the temperature).
    void sample(const float *hn, const float *x0, float *x) {
        std::vector<float> c0(FW), hx(FW), y(FW), z(FW), v(D), xm(D);
        layernorm5(hn, cnw, cnb, H, y.data());
        lin(c_in, y.data(), c0.data());
        for (int i0 = 0; i0 < passes; i0 += MB) {  // the modulations do not depend on x: all passes in batched rows
            const int M = std::min<int>(MB, passes - i0);
#ifdef TT_KERN
            if (g_kf)
                for (int r = 0; r < M; r++) {
                    float *wr = ws.data() + r * W;
                    for (int64_t k = 0; k < FW; k++) wr[k] = c0[k] + tm[(size_t)(i0 + r) * FW + k];
                    mish(wr, FW);
                }
            else
#endif
            lm.pool->run((int64_t)M * FW / 128, 1, [&](int64_t a, int64_t b) {
                for (int64_t i = a * 128; i < b * 128; i++) {
                    const int64_t r = i / FW, k = i % FW;
                    ws[r * W + k] = mish1(c0[k] + tm[(size_t)(i0 + r) * FW + k]);
                }
            });
            for (int b = 0; b < NB; b++) {
                lin(mod[b], ws.data(), b1.data(), M);
                for (int r = 0; r < M; r++)
                    memcpy(modb.data() + ((size_t)b * passes + i0 + r) * 3 * FW, b1.data() + r * W, 3 * FW * sizeof(float));
            }
            lin(out_mod, ws.data(), b1.data(), M);
            for (int r = 0; r < M; r++) memcpy(outm.data() + (size_t)(i0 + r) * 2 * FW, b1.data() + r * W, 2 * FW * sizeof(float));
        }
        auto vel = [&](const float *xin, int p) {
            lin(x_in, xin, hx.data());
            for (int b = 0; b < NB; b++) {
                const float *a = modb.data() + ((size_t)b * passes + p) * 3 * FW;  // shift, scale, gate
                layernorm5(hx.data(), nullptr, nullptr, FW, y.data());
                for (int64_t k = 0; k < FW; k++) y[k] = y[k] * (1.f + a[FW + k]) + a[k];
                lin(ma[b], y.data(), b0.data());
                mish_p(b0.data(), FW);
                lin(mb[b], b0.data(), z.data());
                for (int64_t k = 0; k < FW; k++) hx[k] += a[2 * FW + k] * z[k];
            }
            const float *a = outm.data() + (size_t)p * 2 * FW;
            layernorm5(hx.data(), nullptr, nullptr, FW, y.data());
            for (int64_t k = 0; k < FW; k++) y[k] = y[k] * (1.f + a[FW + k]) + a[k];
            lin(out, y.data(), v.data());
        };
        memcpy(x, x0, D * sizeof(float));
        const int steps = euler ? passes : passes / 2;
        for (int i = 0; i < steps; i++) {
            if (euler) {
                vel(x, i);
                for (int64_t k = 0; k < D; k++) x[k] += v[k] / (float)steps;
            } else {
                vel(x, 2 * i);
                for (int64_t k = 0; k < D; k++) xm[k] = x[k] + v[k] / (float)(2 * steps);
                vel(xm.data(), 2 * i + 1);
                for (int64_t k = 0; k < D; k++) x[k] += v[k] / (float)steps;
            }
        }
    }

    void frame_embed(const float *x, float *e) {  // frame_in: Linear, Mish, Linear
        std::vector<float> y(H);
        lin(fin0, x, y.data());
        mish_p(y.data(), H);
        lin(fin2, y.data(), e);
    }

    // Streaming autoencoder decoder: one code frame (the model's units) -> one VibeVoice latent [64]. Each block's
    // convolution reads its last ak[b] input frames (zeros before the first).
    void ae_reset() {
        for (int b = 0; b < AB; b++) std::fill(ring[b].begin(), ring[b].end(), 0.f);
    }
    void ae_step(const float *code, float *lat) {
        std::vector<float> raw(D), x(AW), h(AW), u(2 * AW + 8), y(AW);
        for (int64_t k = 0; k < D; k++) raw[k] = code[k] * csd[k] + cmu[k];
        lin(ae_in, raw.data(), x.data());
        for (int b = 0; b < AB; b++) {
            float *r = ring[b].data();
            const int64_t k = ak[b];
            if (k > 1) memmove(r, r + AW, (size_t)(k - 1) * AW * sizeof(float));
            layernorm5(x.data(), bnw[b], bnb[b], AW, r + (k - 1) * AW);
            mish_p(r + (k - 1) * AW, AW);
            lin(ae_conv[b], r, u.data());
            mish_p(u.data(), ae_conv[b].N);
            lin(ae_mix[b], u.data(), y.data());
            for (int64_t i = 0; i < AW; i++) x[i] += y[i];
        }
        layernorm5(x.data(), anw, anb, AW, h.data());
        mish_p(h.data(), AW);
        float o[64];
        lin(ae_out, h.data(), o);
        for (int k = 0; k < 64; k++) lat[k] = o[k] * asd[k] + amu[k];
    }

    const float *char_row(int32_t cp) {
        const int32_t *e = chars + NCH, *p = std::lower_bound(chars, e, cp);
        if (p == e || *p != cp) { unknown++; p = std::lower_bound(chars, e, (int32_t)' '); }
        return char_emb + (p - chars) * H;
    }

    // Text segments around chunk c with the boundary cues (lm_chunks.segments).
    std::vector<Text> segments(const std::vector<Text> &chunks, size_t c, int &cur) const {
        const size_t n = chunks.size();
        auto cue = [&](size_t i, Text s) {
            if (i > 0) s.insert(s.begin(), ' ');
            if (i == n - 1) s.push_back('\n');
            return s;
        };
        std::vector<Text> segs;
        if (c > 0) segs.push_back(cue(c - 1, chunks[c - 1]));
        segs.push_back(cue(c, chunks[c]));
        if (c + 1 < n) {
            Text s = chunks[c + 1];
            if ((int64_t)s.size() > LOOK) {
                int64_t cut = -1;
                for (int64_t i = LOOK - 1; i >= 0; i--)
                    if (s[i] == ' ') { cut = i; break; }
                s.resize(cut > 0 ? cut : LOOK);
            }
            segs.push_back(cue(c + 1, s));
        }
        cur = c > 0 ? 1 : 0;
        return segs;
    }

    // n tokens through the model, MBP at a time, at cache rows row0.. of the region that starts at row a0. Only the
    // last token's output is computed (want_last) or none.
    void feed(const float *emb, const float *pos, int64_t n, int64_t row0, int64_t a0, bool want_last = true) {
        for (int64_t i = 0; i < n; i += MBP) {
            const int M = (int)std::min<int64_t>(MBP, n - i);
            int64_t wrow[MBP], r0[MBP], r1[MBP];
            for (int m = 0; m < M; m++) { wrow[m] = row0 + i + m; r0[m] = a0; r1[m] = wrow[m] + 1; }
            lm.last_skip = i + M < n || !want_last ? M : M - 1;
            lm.step(emb + i * H, pos + i, 0, M, wrow, r0, r1);
        }
        lm.last_skip = 0;
    }

    // Start positions of the text segments (lm_chunks.layout): each reserves max(characters + text_off, rate * frames
    // of the previous chunk for the segments before the current one) + 1.
    std::vector<float> seg_starts(const std::vector<Text> &segs, int cur, int64_t n_prev) const {
        std::vector<float> st;
        float p = 1.f;
        for (size_t s = 0; s < segs.size(); s++) {
            st.push_back(p);
            p += std::max((float)segs[s].size() + text_off, (int)s < cur ? rate * (float)n_prev : 0.f) + 1.f;
        }
        return st;
    }
    // The text part of a prefix: the segments' characters, the text markers between them, START.
    void text_part(const std::vector<Text> &segs, const std::vector<float> &st, std::vector<float> &emb, std::vector<float> &pos) {
        auto put = [&](const float *row, float at) { emb.insert(emb.end(), row, row + H); pos.push_back(at); };
        for (size_t s = 0; s < segs.size(); s++) {
            for (size_t j = 0; j < segs[s].size(); j++) put(char_row(segs[s][j]), st[s] + (float)j);
            if (s + 1 < segs.size()) put(tmark, st[s + 1] - 1.f);
        }
        put(start, 0.f);
    }

    struct Result {
        std::vector<float> codes, lat;  // [T][D], [T][64]
        std::vector<int64_t> n_prefix, n_frames;
        double t_first = -1, t_total = 0, t_boundary = 0;  // t_boundary: longest pause between two chunks' frames
        int64_t n_ahead = 0, n_redo = 0;                   // prefix tokens prepared ahead; prefixes that had to be redone
        int64_t n_first = 0;                               // fixed frames at the start (in codes / lat, not sent to on_latent)
        std::vector<float> hn;                             // with keep_snaps: [T][H] the normed state each frame was drawn from
    };

    // Prefix of the next chunk prepared while the current one is generated (ahead): its text part does not depend on
    // the current chunk's frames as long as they fit the text's reserve, and each frame can follow as it is made. The
    // two chunks live in the two halves of the cache; a step carries up to MB - 1 tokens of the next prefix in front
    // of the current frame (one weight pass). lead (seconds of audio ahead of the playback, or null): below
    // ahead_margin only the frame is computed.
    bool ahead = true;
    double ahead_margin = 0.4;
    std::function<double()> lead;
    // Code frames (model units) taken as the first frames of the first chunk, as if generated (misc.bin "first",
    // optional): a fixed silent start, after which the model goes on to the first word. They are read together
    // with the text and never reach on_latent; first_lat are their latents (always the same), for the caller to
    // bring its VibeVoice decoder to the state after them.
    std::vector<float> first, first_lat;
    // VibeVoice latents [k][64] (misc.bin "prime") the caller decodes before every utterance, audio dropped: the
    // decoder's start state the model expects.
    std::vector<float> prime;
    void set_first(const float *codes, int64_t k) {
        first.assign(codes, codes + k * D);
        first_lat.assign((size_t)k * 64, 0.f);
        ae_reset();
        for (int64_t t = 0; t < k; t++) ae_step(first.data() + t * D, first_lat.data() + t * 64);
    }

    // ------------------------------------------------------------ one utterance, frame by frame
    // begin(), then per chunk begin_chunk() and step() until it returns false, then finish(); generate() runs exactly
    // that. A frame's latent comes out of step() before the frame is fed back into the model (done at the start of
    // the next step(), or by flush()), so the caller can decode it at once. With keep_snaps the state before each of
    // the chunk's last frames is kept: rewind() goes back to it, save_take() / load_take() keep and restore a stretch
    // of frames with everything they changed (redraw.h).
    struct Snap {                      // the state before frame t of the current chunk
        int64_t t = -1, s_fed = 0;
        std::vector<float> hid, ring;  // hidden state the frame is drawn from; autoencoder decoder context
    };
    struct Take {                      // frames lo..end-1 of the current chunk and the state after them
        int64_t lo = 0, end = 0;
        bool ended = false;            // the chunk stopped after them
        std::vector<float> codes, lat, fe, hn;
        std::vector<Snap> snaps;       // before frames lo..end
        std::vector<uint8_t> kv;       // cache rows of the frames
    };
    static const int NSNAP = 24;
    bool keep_snaps = false;
    int64_t chunk_now = -1;            // the chunk being generated
    Result R;
    const std::vector<Text> *g_chunks = nullptr;
    const File *g_cs = nullptr;
    bool g_forced = false, g_ready = false, g_open = false, g_ended = true, g_pending = false, g_sh = false;
    float g_temp = 0.8f;
    std::mt19937_64 *g_rng = nullptr;
    OutFile *g_tr = nullptr;
    Gauss g_gauss;
    double g_t_start = 0, g_t_end_prev = -1;
    // cache half, its first row, prefix tokens of the chunk, first frame of the chunk in R, frame index in the chunk
    int64_t g_RS = 0, g_base = 0, g_n = 0, g_c0 = 0, g_t = 0, g_max_f = 0, g_n_ref = 0, g_n_h = 0;
    int64_t g_sbase = 0, g_s_text = 0, g_s_fed = 0;   // the next chunk's prefix: its rows, text tokens, tokens in the cache
    int g_cur = 0, g_scur = 0;
    size_t g_slen0 = 0;
    const float *g_rnoise = nullptr, *g_rx = nullptr;
    std::vector<Text> g_segs;
    std::vector<float> g_starts, g_sstarts, g_prev, g_fe, g_emb, g_pos, g_semb, g_spos, g_hs, g_stops, g_xs, g_bemb, g_hid,
        g_hn, g_e, g_x, g_x0, g_last;
    std::vector<Snap> g_snaps;

    // cs: a case with its noise (and with forced its frames); else noise from rng.
    void begin(const std::vector<Text> &chunks, const File *cs, bool forced, float temp, std::mt19937_64 &rng, OutFile *tr) {
        if (g_ahead_pending) ahead_wait();
        g_chunks = &chunks; g_cs = cs; g_forced = forced; g_temp = temp; g_rng = &rng; g_tr = tr;
        g_gauss.reset();
        R = Result();
        g_t_start = now_s();
        g_t_end_prev = -1;
        g_RS = ahead && chunks.size() > 1 ? lm.S / 2 : lm.S;
        g_base = g_n = 0;
        g_ready = g_open = g_pending = false;
        g_ended = true;
        chunk_now = -1;
        g_prev.clear();
        g_hid.assign(H, 0.f); g_hn.assign(H, 0.f); g_e.assign(H, 0.f); g_x.assign(D, 0.f); g_x0.assign(D, 0.f); g_last.assign(D, 0.f);
        g_bemb.assign((size_t)MB * H, 0.f);
        g_snaps.assign(NSNAP, Snap());
        ae_reset();
    }

    // The next chunk: closes the current one, reads the new one's text (unless prepared ahead). False after the last.
    bool begin_chunk() {
        if (g_open) close_chunk();
        if (++chunk_now >= (int64_t)g_chunks->size()) return false;
        const std::vector<Text> &chunks = *g_chunks;
        const size_t c = (size_t)chunk_now;
        g_segs = segments(chunks, c, g_cur);
        const int64_t n_prev = g_cur ? (int64_t)g_prev.size() / D : 0;
        double t0 = now_s();
        if (!g_ready) {
            g_starts = seg_starts(g_segs, g_cur, n_prev);
            g_emb.clear(); g_pos.clear();
            text_part(g_segs, g_starts, g_emb, g_pos);
            if (g_cur) {
                for (int64_t t = 0; t < n_prev; t++) {
                    frame_embed(g_prev.data() + t * D, g_e.data());
                    g_emb.insert(g_emb.end(), g_e.begin(), g_e.end()); g_pos.push_back(g_starts[0] + rate * (float)t);
                }
                g_emb.insert(g_emb.end(), amark, amark + H); g_pos.push_back(g_starts[1] - 1.f);
            }
            g_n = (int64_t)g_pos.size();
            for (int64_t t = 0; t < (c == 0 ? (int64_t)first.size() / D : 0); t++) {   // the fixed frames ride along
                frame_embed(first.data() + t * D, g_e.data());
                g_emb.insert(g_emb.end(), g_e.begin(), g_e.end()); g_pos.push_back(g_starts[g_cur] + rate * (float)t);
            }
            if ((int64_t)g_pos.size() > g_RS) { fprintf(stderr, "chunk %zu: %lld positions do not fit the cache (%lld)\n", c, (long long)g_n, (long long)g_RS); exit(2); }
            t_embed += now_s() - t0;
            t0 = now_s();
            if (!(prefill_hook && prefill_hook(g_emb.data(), g_pos.data(), (int64_t)g_pos.size(), g_base, g_hid.data()))) {
                feed(g_emb.data(), g_pos.data(), (int64_t)g_pos.size(), g_base, g_base);
                memcpy(g_hid.data(), lm.hid, H * sizeof(float));
            }
            t_prefill += now_s() - t0;
        }
        g_max_f = (int64_t)g_segs[g_cur].size() + OFFR;
        if (g_n + g_max_f > g_RS) { fprintf(stderr, "chunk %zu: %lld positions do not fit the cache (%lld)\n", c, (long long)(g_n + g_max_f), (long long)g_RS); exit(2); }
        // the next chunk's prefix, assuming this chunk's frames fit its text's reserve
        g_sh = ahead && c + 1 < chunks.size();
        g_sbase = g_RS - g_base;
        g_scur = 0;
        g_s_text = g_s_fed = 0;
        g_slen0 = 0;
        if (g_sh) {
            const std::vector<Text> segs2 = segments(chunks, c + 1, g_scur);
            g_slen0 = segs2[0].size();
            g_sstarts = seg_starts(segs2, g_scur, 0);
            g_semb.clear(); g_spos.clear();
            text_part(segs2, g_sstarts, g_semb, g_spos);
            g_s_text = (int64_t)g_spos.size();
            g_ahead_due = ahead_start_hook != nullptr;
        }
        g_rnoise = g_rx = nullptr;
        g_n_ref = 0;
        g_n_h = g_max_f;
        if (g_cs) {
            const std::string k = "c" + std::to_string(c) + ".";
            g_rnoise = (const float *)g_cs->get(k + "noise").p;
            g_rx = (const float *)g_cs->get(k + "x").p;
            g_n_ref = (int64_t)g_cs->get(k + "x").dims[0];
            if (g_forced) g_n_h = (int64_t)g_cs->get(k + "h").dims[0];
        }
        g_hs.clear(); g_stops.clear(); g_xs.clear(); g_prev.clear(); g_fe.clear();
        g_c0 = (int64_t)R.codes.size() / D;
        g_t = 0;
        std::vector<float> lat(64);
        for (; g_t < (c == 0 ? (int64_t)first.size() / D : 0); g_t++) {   // the fixed frames: already in the cache
            const float *fed = first.data() + g_t * D;
            ae_step(fed, lat.data());
            R.codes.insert(R.codes.end(), fed, fed + D);
            R.lat.insert(R.lat.end(), lat.begin(), lat.end());
            g_prev.insert(g_prev.end(), fed, fed + D);
            frame_embed(fed, g_e.data());
            g_fe.insert(g_fe.end(), g_e.begin(), g_e.end());
        }
        R.n_first = c == 0 ? g_t : R.n_first;
        for (auto &s : g_snaps) s.t = -1;
        g_open = true;
        g_ended = g_pending = false;
        return true;
    }

    // One frame of the current chunk: its latent [64] into lat. False when the chunk has ended (stop value or reserve).
    bool step(float *lat) {
        if (g_ended) return false;
        flush();
        if (keep_snaps) snap_save();
        double t0 = now_s();
        if (g_t >= g_n_h) { g_ended = true; return false; }
        rmsnorm(g_hid.data(), lm.norm, H, lm.eps, g_hn.data());
        float sv;
        lin(stop, g_hn.data(), &sv);
        if (g_tr) { g_hs.insert(g_hs.end(), g_hn.begin(), g_hn.end()); g_stops.push_back(sv); }
        if (g_forced ? g_t >= g_n_ref : (g_t >= MINF && sv > 0.f)) { g_ended = true; return false; }
        if (keep_snaps) { R.hn.resize((size_t)(g_c0 + g_t) * H); R.hn.insert(R.hn.end(), g_hn.begin(), g_hn.end()); }
        for (int64_t k = 0; k < D; k++) g_x0[k] = g_temp * (g_rnoise && g_t < g_n_ref ? g_rnoise[g_t * D + k] : g_gauss(*g_rng));
        sample(g_hn.data(), g_x0.data(), g_x.data());
        if (g_tr) g_xs.insert(g_xs.end(), g_x.begin(), g_x.end());
        memcpy(g_last.data(), g_forced ? g_rx + g_t * D : g_x.data(), D * sizeof(float));
        t_head += now_s() - t0;
        t0 = now_s();
        ae_step(g_last.data(), lat);
        t_ae += now_s() - t0;
        R.codes.insert(R.codes.end(), g_last.begin(), g_last.end());
        R.lat.insert(R.lat.end(), lat, lat + 64);
        if (R.t_first < 0) R.t_first = now_s() - g_t_start;
        if (g_t == 0 && g_t_end_prev >= 0) R.t_boundary = std::max(R.t_boundary, now_s() - g_t_end_prev);
        g_prev.insert(g_prev.end(), g_last.begin(), g_last.end());
        g_pending = true;
        return true;
    }

    // The frame just made into the model: one step with tokens of the next prefix first (nobody reads their output),
    // this frame last.
    void flush() {
        if (!g_pending) return;
        g_pending = false;
        const double t0 = now_s();
        const int64_t t = g_t;
        frame_embed(g_last.data(), g_e.data());
        g_fe.insert(g_fe.end(), g_e.begin(), g_e.end());
        int M = 0;
        int64_t wrow[MB], r0[MB], r1[MB];
        float bpos[MB];
        if (g_ahead_due && g_s_fed == 0 && g_sh && (!lead || lead() > ahead_margin)) {  // after this frame's decode was handed on
            g_ahead_due = false;
            if (ahead_start_hook(g_semb.data(), g_spos.data(), g_s_text, g_sbase)) { g_s_fed = g_s_text; g_ahead_pending = true; }
        }
        if (g_ahead_pending && ahead_ready_hook()) ahead_wait();  // done (or failed: then read here)
        if (g_sh && !g_ahead_pending && (!lead || lead() > ahead_margin))
            for (; M < MB - 1 && g_s_fed < g_s_text + t + 1; M++, g_s_fed++) {
                const bool text = g_s_fed < g_s_text;
                memcpy(g_bemb.data() + (size_t)M * H, text ? g_semb.data() + g_s_fed * H : g_fe.data() + (g_s_fed - g_s_text) * H, H * sizeof(float));
                bpos[M] = text ? g_spos[g_s_fed] : g_sstarts[0] + rate * (float)(g_s_fed - g_s_text);
                wrow[M] = g_sbase + g_s_fed; r0[M] = g_sbase; r1[M] = wrow[M] + 1;
                R.n_ahead++;
            }
        memcpy(g_bemb.data() + (size_t)M * H, g_e.data(), H * sizeof(float));
        bpos[M] = g_starts[g_cur] + rate * (float)t;
        wrow[M] = g_base + g_n + t; r0[M] = g_base; r1[M] = wrow[M] + 1;
        lm.last_skip = M;
        lm.step(g_bemb.data(), bpos, 0, M + 1, wrow, r0, r1);
        lm.last_skip = 0;
        memcpy(g_hid.data(), lm.hid, H * sizeof(float));
        g_t++;
        t_lm += now_s() - t0;
    }

    void close_chunk() {
        flush();
        g_ahead_due = false;
        if (g_ahead_pending) ahead_wait();
        g_open = false;
        const size_t c = (size_t)chunk_now;
        const int64_t nf = (int64_t)g_prev.size() / D;
        R.n_prefix.push_back(g_n);
        R.n_frames.push_back(nf);
        if (g_tr) {
            const std::string k = "c" + std::to_string(c) + ".";
            g_tr->f32(k + "h", g_hs.data(), {(uint64_t)g_stops.size(), (uint64_t)H});
            g_tr->f32(k + "stop", g_stops.data(), {(uint64_t)g_stops.size()});
            g_tr->f32(k + "x", g_xs.data(), {(uint64_t)(g_xs.size() / D), (uint64_t)D});
        }
        g_t_end_prev = now_s();
        g_ready = false;
        if (g_sh && rate * (float)nf <= (float)g_slen0 + text_off) {  // finish the prepared prefix: rest, then the audio marker
            const double t0 = now_s();
            const int64_t total = g_s_text + nf;
            if (total + 1 > g_RS) { fprintf(stderr, "chunk %zu: prefix does not fit the cache\n", c + 1); exit(2); }
            g_emb.clear(); g_pos.clear();
            for (int64_t k = g_s_fed; k < total; k++) {
                const bool text = k < g_s_text;
                const float *row = text ? g_semb.data() + k * H : g_fe.data() + (k - g_s_text) * H;
                g_emb.insert(g_emb.end(), row, row + H);
                g_pos.push_back(text ? g_spos[k] : g_sstarts[0] + rate * (float)(k - g_s_text));
            }
            g_emb.insert(g_emb.end(), amark, amark + H); g_pos.push_back(g_sstarts[1] - 1.f);
            feed(g_emb.data(), g_pos.data(), (int64_t)g_pos.size(), g_sbase + g_s_fed, g_sbase);
            memcpy(g_hid.data(), lm.hid, H * sizeof(float));
            t_prefill += now_s() - t0;
            g_n = total + 1;
            g_starts = g_sstarts;
            g_base = g_sbase;
            g_ready = true;
        } else if (g_sh) {
            R.n_redo++;  // the frames ran past the text's reserve: the positions after the first segment move
        }
    }

    Result finish() {
        if (g_open) close_chunk();
        R.t_total = now_s() - g_t_start;
        return R;
    }

    Result generate(const std::vector<Text> &chunks, const File *cs, bool forced, float temp, std::mt19937_64 &rng, OutFile *tr,
                    const std::function<void(const float *)> &on_latent) {
        begin(chunks, cs, forced, temp, rng, tr);
        float lat[64];
        while (begin_chunk())
            while (step(lat))
                if (on_latent) on_latent(lat);
        return finish();
    }

    // ------------------------------------------------------------ going back (redraw)
    int64_t chunk_first() const { return g_c0; }          // the chunk's first frame (index into R.codes)
    int64_t frames() const { return (int64_t)R.codes.size() / D; }
    bool chunk_ended() const { return g_ended; }
    void snap_save() {
        Snap &s = g_snaps[g_t % NSNAP];
        s.t = g_t; s.s_fed = g_s_fed; s.hid = g_hid;
        s.ring.clear();
        for (int b = 0; b < AB; b++) s.ring.insert(s.ring.end(), ring[b].begin(), ring[b].end());
    }
    void snap_load(const Snap &s, int64_t fed_max) {
        g_t = s.t; g_s_fed = std::min(s.s_fed, fed_max); g_hid = s.hid;
        size_t o = 0;
        for (int b = 0; b < AB; b++) { std::copy(s.ring.begin() + o, s.ring.begin() + o + ring[b].size(), ring[b].begin()); o += ring[b].size(); }
        g_pending = false;
    }
    // Back to just before frame b of the current chunk (b >= lo: frames from lo on may differ from what the next
    // chunk's prepared prefix has read, so those prefix rows are made again).
    bool rewind(int64_t b, int64_t lo) {
        const Snap &s = g_snaps[b % NSNAP];
        if (b < 0 || s.t != b) return false;
        snap_load(s, g_s_text + lo);
        g_ended = false;
        R.codes.resize((size_t)(g_c0 + b) * D); R.lat.resize((size_t)(g_c0 + b) * 64);
        g_prev.resize((size_t)b * D); g_fe.resize((size_t)b * H);
        return true;
    }
    // KV cache row (keys, values and the int8 copies the engine keeps) to / from bytes.
    size_t kv_row_bytes() const {
        return (size_t)lm.NL * lm.nkv * (2 * HD * sizeof(kv_t) + (lm.v8 ? HD + 4 : 0) + (lm.k8 ? HD + 4 : 0));
    }
    void kv_row(int64_t row, uint8_t *p, bool save) {
        const int64_t S = lm.S;
        auto mv = [&](void *a, size_t n) { if (save) memcpy(p, a, n); else memcpy(a, p, n); p += n; };
        for (int64_t l = 0; l < lm.NL; l++)
            for (int64_t i = 0; i < lm.nkv; i++) {
                mv(lm.kv + ((l * 2) * lm.nkv * S + i * S + row) * HD, HD * sizeof(kv_t));
                mv(lm.kv + ((l * 2 + 1) * lm.nkv * S + i * S + row) * HD, HD * sizeof(kv_t));
                if (lm.v8) {
                    int8_t *vh = lm.v8 + (size_t)(l * lm.nkv + i) * S * HD;
                    for (int d = 0; d < HD; d++) mv(vh + v8_index(S, d, row), 1);
                    mv(lm.v8s + (l * lm.nkv + i) * S + row, 4);
                }
                if (lm.k8) {
                    mv(lm.k8 + ((l * lm.nkv + i) * S + row) * HD, HD);
                    mv(lm.k8s + (l * lm.nkv + i) * S + row, 4);
                }
            }
    }
    // Frames lo.. of the current chunk (to now) and the state after them.
    void save_take(int64_t lo, Take &k) {
        flush();
        snap_save();
        k.lo = lo; k.end = g_t; k.ended = g_ended;
        k.codes.assign(R.codes.begin() + (g_c0 + lo) * D, R.codes.end());
        k.lat.assign(R.lat.begin() + (g_c0 + lo) * 64, R.lat.end());
        k.fe.assign(g_fe.begin() + lo * H, g_fe.end());
        k.hn.assign(R.hn.begin() + std::min<size_t>(R.hn.size(), (size_t)(g_c0 + lo) * H), R.hn.end());
        k.snaps.clear();
        for (int64_t j = lo; j <= k.end; j++) {
            if (g_snaps[j % NSNAP].t != j) { fprintf(stderr, "save_take: frame %lld not kept\n", (long long)j); exit(2); }
            k.snaps.push_back(g_snaps[j % NSNAP]);
        }
        const size_t rb = kv_row_bytes();
        k.kv.resize(rb * (k.end - lo));
        for (int64_t j = lo; j < k.end; j++) kv_row(g_base + g_n + j, k.kv.data() + rb * (j - lo), true);
    }
    void load_take(const Take &k) {
        R.codes.resize((size_t)(g_c0 + k.lo) * D); R.codes.insert(R.codes.end(), k.codes.begin(), k.codes.end());
        R.lat.resize((size_t)(g_c0 + k.lo) * 64); R.lat.insert(R.lat.end(), k.lat.begin(), k.lat.end());
        g_prev.resize((size_t)k.lo * D); g_prev.insert(g_prev.end(), k.codes.begin(), k.codes.end());
        g_fe.resize((size_t)k.lo * H); g_fe.insert(g_fe.end(), k.fe.begin(), k.fe.end());
        R.hn.resize((size_t)(g_c0 + k.lo) * H); R.hn.insert(R.hn.end(), k.hn.begin(), k.hn.end());
        for (int64_t j = k.lo; j <= k.end; j++) g_snaps[j % NSNAP] = k.snaps[j - k.lo];
        const size_t rb = kv_row_bytes();
        for (int64_t j = k.lo; j < k.end; j++) kv_row(g_base + g_n + j, const_cast<uint8_t *>(k.kv.data()) + rb * (j - k.lo), false);
        snap_load(k.snaps.back(), g_s_text + k.lo);
        g_ended = k.ended;
    }
};
