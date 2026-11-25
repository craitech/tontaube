// The VibeVoice decoder and the text reading of chunk prefixes on the GPU (WebGPU in ext/gpu_worker.js), switched on
// by tts_gpu(); off (the default) nothing here runs and the engine is the CPU engine. Every call goes through
// Module.ttsHooks (engine_worker.js), which hands it to the GPU worker and waits where an answer is needed.
// A failed GPU call (device lost) moves the rest of the request and every later one to the CPU decoder: its state is
// rebuilt by decoding again the latents that led to it (the decoder's state depends on its last 58 latents only).
// Included by tts_web.cpp (browser build only).

EM_JS(int, js_gpu_begin, (const float *lat, int n), { return Module.ttsHooks.gpuBegin(lat, n); });
EM_JS(int, js_gpu_decode, (const float *lat, float *audio), { return Module.ttsHooks.gpuDecode(lat, audio); });
EM_JS(void, js_gpu_submit, (const float *lat, float *audio), { Module.ttsHooks.gpuSubmit(lat, audio); });
EM_JS(int, js_gpu_wait, (), { return Module.ttsHooks.gpuWait(); });
EM_JS(void, js_gpu_save, (double id), { Module.ttsHooks.gpuSave(id); });
EM_JS(int, js_gpu_load, (double id), { return Module.ttsHooks.gpuLoad(id); });
EM_JS(int, js_gpu_prefill, (const float *emb, const float *pos, int n, int base, float *hid, void *kv, int S), {
    return Module.ttsHooks.gpuPrefill(emb, pos, n, base, hid, kv, S);
});
EM_JS(int, js_gpu_prefill_start, (const float *emb, const float *pos, int n, int base, void *kv, int S), {
    return Module.ttsHooks.gpuPrefillStart(emb, pos, n, base, kv, S);
});
EM_JS(int, js_gpu_prefill_wait, (), { return Module.ttsHooks.gpuPrefillWait(); });
EM_JS(int, js_gpu_prefill_ready, (), { return Module.ttsHooks.gpuPrefillReady(); });
// a model file kept out of the heap (tts_lazy_vv): its size, and the file read into p
EM_JS(double, js_file_size, (const char *name), { return Module.ttsHooks.fileSize(UTF8ToString(name)); });
EM_JS(int, js_file_read, (const char *name, void *p, double n), { return Module.ttsHooks.fileRead(UTF8ToString(name), p, n); });

// The decoder on the GPU behind the redraw's hook. Its state bytes are an id; the GPU keeps a snapshot per id, the
// CPU reaches any recent state by decoding the latents that led to it again (from the request's start state when the
// chain reaches it, else from a zero state: the last 64 latents give the state exactly; stage 0 alone for all but
// the last 4).
struct GpuDec : FrameDecoder {
    bool overlap = true;   // submit / wait: the next backbone step runs on the CPU while the GPU decodes
    bool gpu_ok = true;    // false after a failed GPU call: the CPU decoder from then on
    bool pending = false;  // a submitted frame not waited for (a request stopped in between)
    int64_t cur = 0, next = 1, root = 0, fallbacks = 0, replayed = 0, stale = 0;
    double t_fallback = 0;
    struct Node { int64_t parent = -1; std::array<float, 64> lat{}; };
    std::map<int64_t, Node> nodes;                          // the last 1024 states: parent state and latent
    mutable std::map<int64_t, std::vector<float>> snaps;    // CPU state snapshots after a fallback (the last 64)
    float pend_lat[64];
    float *pend_audio = nullptr;

    void add(const float *lat) {
        Node n;
        n.parent = cur;
        memcpy(n.lat.data(), lat, sizeof pend_lat);
        cur = next++;
        nodes[cur] = n;
        while (nodes.size() > 1024) nodes.erase(nodes.begin());
    }
    void begin() {  // a request starts from the decoder's start state (priming latents, as the CPU decoder)
        if (pending) { js_gpu_wait(); pending = false; }
        root = cur = next++;
        nodes[root] = Node{};
        if (!gpu_ok) { if (g_eng.ensure_vv()) g_eng.vv.begin(); return; }
        std::vector<float> st0(g_eng.m.prime);
        st0.insert(st0.end(), g_eng.m.first_lat.begin(), g_eng.m.first_lat.end());
        if (!js_gpu_begin(st0.data(), (int)(st0.size() / 64))) fail_now();
    }
    bool to_cpu(int64_t s) {  // the CPU decoder to state s
        if (!g_eng.ensure_vv()) return false;
        auto it = snaps.find(s);
        if (it != snaps.end()) { g_eng.vv.load_state(it->second.data()); return true; }
        std::vector<const float *> chain;
        int64_t k = s;
        while (k != root && chain.size() < 64) {
            auto n = nodes.find(k);
            if (n == nodes.end()) break;
            chain.push_back(n->second.lat.data());
            k = n->second.parent;
        }
        if (k == root) g_eng.vv.begin(); else g_eng.vv.reset();
        // stage 0 alone for all but the last 4 latents (the later stages' states depend on those 4 only: exact)
        std::vector<float> a((size_t)g_eng.vv.HOP);
        const int64_t n = (int64_t)chain.size();
        for (int64_t i = 0; i < n; i++) g_eng.vv.decode_one(chain[(size_t)(n - 1 - i)], a.data(), i < n - 4 ? 1 : 1 << 30);
        replayed += n;
        return true;
    }
    void fail_now() {  // the GPU is gone: the CPU decoder takes over at the current state
        const double t0 = now_s();
        gpu_ok = false;
        fallbacks++;
        to_cpu(cur);
        t_fallback += now_s() - t0;
    }
    bool decode(const float *lat, float *audio) override {
        if (gpu_ok && !js_gpu_decode(lat, audio)) fail_now();
        if (!gpu_ok) g_eng.vv.decode(lat, audio);
        add(lat);
        return true;
    }
    bool submit(const float *lat, float *audio) override {
        if (!overlap || !gpu_ok) return false;
        memcpy(pend_lat, lat, sizeof pend_lat);
        pend_audio = audio;
        pending = true;
        js_gpu_submit(lat, audio);
        return true;
    }
    void wait() override {
        pending = false;
        if (!js_gpu_wait()) {
            fail_now();
            g_eng.vv.decode(pend_lat, pend_audio);
        }
        add(pend_lat);
    }
    size_t state_size() const override { return sizeof(int64_t); }
    void get_state(uint8_t *dst) const override {
        memcpy(dst, &cur, sizeof cur);
        if (gpu_ok) { js_gpu_save((double)cur); return; }
        std::vector<float> v(g_eng.vv.state_size());
        g_eng.vv.save_state(v.data());
        snaps[cur] = std::move(v);
        while (snaps.size() > 64) snaps.erase(snaps.begin());
    }
    void set_state(const uint8_t *src) override {
        int64_t s;
        memcpy(&s, src, sizeof s);
        cur = s;
        if (gpu_ok && !js_gpu_load((double)s)) { stale++; fail_now(); }  // older than the GPU's snapshots
        else if (!gpu_ok) to_cpu(s);
    }
};
static GpuDec g_gpu_dec;
static const char *g_gpu_refused = "";  // why tts_gpu kept a part on the CPU (speed options), or ""

extern "C" {
// Before tts_load: vv.bin is not put into the heap (the GPU decodes); the engine reads it through the hooks only
// when it needs the CPU decoder (a GPU failure, tts_redecode).
EMSCRIPTEN_KEEPALIVE void tts_lazy_vv(int on) {
    g_eng.lazy_vv = on != 0;
    g_eng.load_hook = [](const char *name) {
        const double n = js_file_size(name);
        if (n <= 0) return false;
        void *p = aligned_alloc(64, ((size_t)n + 63) / 64 * 64);
        if (!p || !js_file_read(name, p, n)) { free(p); return false; }
        web_files()[std::string("/m/") + name] = WebBuf{(const uint8_t *)p, (size_t)n};
        return true;
    };
}
// decoder: the decoder on the GPU; text: chunk prefixes read on the GPU (the next chunk's while this one plays)
EMSCRIPTEN_KEEPALIVE void tts_gpu(int decoder, int text) {
    // pipelining (its own decoder threads) is for the CPU decoder: no GPU with it; int8 activations in the decoder
    // keep the decoder on the CPU (the GPU's CPU fallback must be the same function as the GPU)
    g_gpu_refused = g_cfg.pipe() ? "pipe" : g_cfg.act8_vv ? "act8=vv" : "";
    if (g_cfg.pipe()) decoder = text = 0;
    else if (g_cfg.act8_vv) decoder = 0;
    g_gpu_dec.gpu_ok = true;
    g_gpu_dec.snaps.clear();
    g_eng.ext_dec = decoder ? &g_gpu_dec : nullptr;
    g_eng.ext_begin = decoder ? std::function<void()>([]() { g_gpu_dec.begin(); }) : nullptr;
    FM &m = g_eng.m;
    m.prefill_hook = nullptr;
    m.ahead_start_hook = nullptr;
    m.ahead_wait_hook = nullptr;
    m.ahead_ready_hook = nullptr;
    if (!text) return;
    m.prefill_hook = [](const float *emb, const float *pos, int64_t n, int64_t base, float *hid) {
        return js_gpu_prefill(emb, pos, (int)n, (int)base, hid, (void *)g_eng.m.lm.kv, (int)g_eng.m.lm.S) != 0;
    };
    m.ahead_start_hook = [](const float *emb, const float *pos, int64_t n, int64_t base) {
        return js_gpu_prefill_start(emb, pos, (int)n, (int)base, (void *)g_eng.m.lm.kv, (int)g_eng.m.lm.S) != 0;
    };
    m.ahead_wait_hook = []() { return js_gpu_prefill_wait() != 0; };
    m.ahead_ready_hook = []() { return js_gpu_prefill_ready() != 0; };
}
// Auto's check: a short sentence through the engine (redraw off), then its latents from the decoder's start state
// through the CPU decoder (its file loaded if kept out) and the GPU decoder, one frame at a time (at most n): JSON
// {frames, snr_db, max_abs, cpu_ms, gpu_ms} (ms per frame, the first frame not timed). Each request begins with its
// start state, so nothing of this stays.
EMSCRIPTEN_KEEPALIVE const char *tts_gpu_check(int n) {
    static std::string out;
    if (!g_eng.ext_dec || !g_eng.ensure_vv()) { out = "{\"snr_db\":0,\"cpu_ms\":0,\"gpu_ms\":1e9,\"error\":\"no decoder\"}"; return out.c_str(); }
    Engine::Stats sst;
    Sink none;
    g_eng.speak_chunks({Text{'H', 'e', 'l', 'l', 'o', ',', ' ', 'h', 'o', 'w', ' ', 'a', 'r', 'e', ' ', 'y', 'o', 'u', '?'}}, false, 1, 0.0, none, sst,
                       now_s(), false);
    const std::vector<float> lat = g_eng.m.R.lat;
    VVDec &v = g_eng.vv;
    std::vector<float> st0(g_eng.m.prime);
    st0.insert(st0.end(), g_eng.m.first_lat.begin(), g_eng.m.first_lat.end());
    n = std::min<int>(n, (int)(lat.size() / 64));
    std::vector<float> a((size_t)v.HOP), g((size_t)v.HOP);
    double se = 0, ss = 0, mx = 0, tc = 0, tg = 0;
    v.begin();
    bool ok = n > 1 && js_gpu_begin(st0.data(), (int)(st0.size() / 64)) != 0;
    for (int t = 0; ok && t < n; t++) {
        const float *l = lat.data() + (size_t)t * 64;
        double t0 = now_s();
        v.decode(l, a.data());
        if (t > 0) tc += now_s() - t0;
        t0 = now_s();
        ok = js_gpu_decode(l, g.data()) != 0;
        if (t > 0) tg += now_s() - t0;
        for (size_t i = 0; i < a.size(); i++) { const double d = (double)a[i] - g[i]; se += d * d; ss += (double)a[i] * a[i]; mx = std::max(mx, std::fabs(d)); }
    }
    char b[220];
    snprintf(b, sizeof b, "{\"frames\":%d,\"snr_db\":%.1f,\"max_abs\":%.3g,\"cpu_ms\":%.1f,\"gpu_ms\":%.1f%s}", n, ok ? 10 * std::log10(ss / std::max(se, 1e-30)) : 0.0, mx,
             1e3 * tc / std::max(1, n - 1), ok ? 1e3 * tg / std::max(1, n - 1) : 1e9, ok ? "" : ",\"error\":\"GPU call failed\"");
    out = b;
    return out.c_str();
}
// JSON: the GPU decoder's state and fallback counters
EMSCRIPTEN_KEEPALIVE const char *tts_gpu_status() {
    static std::string out;
    char b[256];
    snprintf(b, sizeof b, "{\"refused\":\"%s\",\"decoder\":%s,\"gpu_ok\":%s,\"fallbacks\":%lld,\"stale\":%lld,\"replayed\":%lld,\"fallback_ms\":%.1f,\"cpu_decoder\":%s}",
             g_gpu_refused, g_eng.ext_dec ? "true" : "false", g_gpu_dec.gpu_ok ? "true" : "false", (long long)g_gpu_dec.fallbacks,
             (long long)g_gpu_dec.stale, (long long)g_gpu_dec.replayed, 1e3 * g_gpu_dec.t_fallback, g_eng.vv.NS > 0 ? "true" : "false");
    out = b;
    return out.c_str();
}
}
