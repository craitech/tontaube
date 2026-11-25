// The browser engine: text -> rule verbalizer -> chunker -> frame model (fm.cpp) -> VibeVoice decoder (vv_dec.h)
// -> detector and redraw of flagged spots (redraw.h) -> start at the first sound (lead_cut.h) -> optional AP-BWE to
// 48 kHz (bwe.h) -> audio, one 133 ms frame at a time.
//
// The engine worker calls the tts_* functions; audio, pacing state and the stop flag go through JS functions
// (Module.ttsHooks) that work on a SharedArrayBuffer ring read by an AudioWorklet. Model files: lm.bin, misc.bin,
// vv.bin, detector.bin (without it the redraw is off), bwe.bin (48 kHz only).
#include "fm.cpp"
#include "redraw.h"
#include "lead_cut.h"
#include "vv_dec.h"
#include "bwe.h"
#include "chunker.h"
#include "rv/verbalizer.h"
#include "sha256.h"

#include <array>
#include <functional>
#include <map>
#include <memory>
#include <sstream>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#endif

struct Cancelled {};

// Speed options (tts_config before tts_load), all off by default (then the arithmetic and the schedule are the plain
// engine's). Same numbers: pipe=A,B the decoder on its own pool of B threads next to the frame model's pool of A
// threads, frame k + 1 made while frame k is decoded; share=1 (pipelined) each pool takes part of the other's linears
// while it would wait; x8=1 the q4 / q8 linears' int16 activations through the 8-bit dot product (relaxed SIMD);
// f44=1 the decoder's float linears in 4 x 4 register blocks (SIMD; for CPUs with 32 vector registers). Other
// numbers: act8=lm,vv int8 activations for the linears of lm.bin and the decoder's quantized linears (relaxed SIMD);
// bf16=lm the quantized linears of lm.bin and misc.bin in bf16 (native x86 with AMX or AVX512-BF16), bf16=prefill only
// in the text reading (steps of more than 8 rows; frames stay int16). weights=mmap: read the model files from the page
// cache instead of a copy in huge pages (native Linux builds; same numbers).
struct Cfg {
    bool act8_lm = false, act8_vv = false;
    int bf16 = 0;  // 1 lm, 2 prefill
    int pipe_fm = 0, pipe_vv = 0;
    bool share = false, x8 = false, f44 = false;
    bool pipe() const { return pipe_fm > 0 && pipe_vv > 0; }
    // "pipe=3,3 share=1 x8=1 f44=1 act8=lm,vv" (any order, any subset); returns "" or an error
    std::string parse(const std::string &spec) {
        std::istringstream in(spec);
        std::string kv;
        while (in >> kv) {
            const size_t e = kv.find('=');
            const std::string k = kv.substr(0, e), v = e == std::string::npos ? "" : kv.substr(e + 1);
            if (k == "act8") {
                std::istringstream parts(v);
                std::string p;
                while (std::getline(parts, p, ',')) {
                    if (p == "lm") act8_lm = true;
                    else if (p == "vv") act8_vv = true;
                    else if (!p.empty()) return "unknown act8 part " + p;
                }
            } else if (k == "pipe") {
                pipe_fm = atoi(v.c_str());
                const size_t c = v.find(',');
                pipe_vv = c == std::string::npos ? 0 : atoi(v.c_str() + c + 1);
            } else if (k == "share") {
                share = v != "0";
            } else if (k == "x8") {
                x8 = v != "0";
            } else if (k == "weights") {  // "weights=mmap": read the model files from the page cache (no huge-page copy)
                if (v == "mmap") g_weights_mmap = true;
                else return "unknown weights mode " + v;
            } else if (k == "bf16") {
                if (v == "lm") bf16 = 1;
                else if (v == "prefill") bf16 = 2;
                else if (!v.empty()) return "unknown bf16 part " + v;
            } else if (k == "f44") {
                f44 = v != "0";
            } else if (!k.empty()) {
                return "unknown option " + k;
            }
        }
        return "";
    }
    std::string str() const {  // the options that are on ("" if none)
        std::string s;
        if (act8_lm || act8_vv) s += std::string(" act8=") + (act8_lm ? "lm" : "") + (act8_lm && act8_vv ? "," : "") + (act8_vv ? "vv" : "");
        if (bf16) s += bf16 == 1 ? " bf16=lm" : " bf16=prefill";
        if (g_weights_mmap) s += " weights=mmap";
        if (pipe()) s += " pipe=" + std::to_string(pipe_fm) + "," + std::to_string(pipe_vv);
        if (share) s += " share=1";
        if (x8) s += " x8=1";
        if (f44) s += " f44=1";
        return s.empty() ? s : s.substr(1);
    }
};
static Cfg g_cfg;
// The kernel switches of the options; int8 activations for the layers whose codes lie in lm.bin.
static void apply_act8(const FM &m) {
#ifdef WEB_DOT8
    g_web_x8 = g_cfg.x8;
#endif
#ifdef WEB_SIMD
    g_web_f44 = g_cfg.f44;
#endif
    g_web_n_act8 = 0;
    if (g_cfg.act8_lm) g_web_act8[g_web_n_act8++] = WebRange{m.fl.base, m.fl.base + m.fl.size};
}

// One thread that runs the caller's jobs one at a time (the decoder next to the frame model): run() hands a job over
// and returns, wait() returns when it is done. While the caller waits it runs what the thread asks of it (ask():
// part of the decoder's linears on the caller's pool). Both sides spin a little, then sleep on a futex.
struct JobThread {
    plat::Thread th;
    bool on = false;
    std::atomic<uint32_t> go{0}, done{0}, ev{0}, hreq{0}, hack{0};  // ev: done or a request (what a waiting caller sleeps on)
    std::atomic<bool> stop{false}, waiting{false};
    const std::function<void()> *hfn = nullptr;
    WebHelper *helper = nullptr;  // the thread's own helper (the caller's pool), if any
    std::function<void()> job;
    double t_job = 0;  // seconds inside jobs (counted ones)
    bool count = true;
    static void *entry(void *p) { ((JobThread *)p)->loop(); return nullptr; }
    static void await(std::atomic<uint32_t> &a, uint32_t seen) {
        const double t0 = now_s();
        for (uint32_t spins = 0; a.load(std::memory_order_acquire) == seen; spins++)
            if ((spins & 255) == 255 && now_s() - t0 > 1e-3) plat::wait(a, seen);
    }
    void loop() {
        t_web_helper = helper;
        uint32_t seen = 0;
        for (;;) {
            await(go, seen);
            seen = go.load(std::memory_order_acquire);
            if (stop.load()) return;
            const double t0 = now_s();
            job();
            if (count) t_job += now_s() - t0;
            done.store(seen, std::memory_order_release);
            ev.fetch_add(1, std::memory_order_release);
            plat::wake_all(ev);
        }
    }
    // from the thread: the waiting caller runs f (false: the caller is not waiting); asked_wait() joins it
    bool ask(const std::function<void()> &f) {
        if (!waiting.load()) return false;
        hfn = &f;
        hreq.fetch_add(1, std::memory_order_release);
        ev.fetch_add(1, std::memory_order_release);
        plat::wake_all(ev);
        return true;
    }
    void asked_wait() {
        const uint32_t r = hreq.load(std::memory_order_acquire);
        for (uint32_t a = hack.load(std::memory_order_acquire); a != r; a = hack.load(std::memory_order_acquire)) await(hack, a);
    }
    void start() { if (!on) { th.start(entry, this); on = true; } }
    bool idle() const { return done.load(std::memory_order_acquire) == go.load(std::memory_order_relaxed); }
    void run(std::function<void()> f, bool counted = true) {
        count = counted;
        job = std::move(f);
        go.fetch_add(1, std::memory_order_release);
        plat::wake_all(go);
    }
    void wait() {
        const uint32_t want = go.load(std::memory_order_acquire);
        waiting.store(true);
        for (;;) {
            const uint32_t e = ev.load(std::memory_order_acquire);
            if (done.load(std::memory_order_acquire) == want) break;
            const uint32_t r = hreq.load(std::memory_order_acquire);
            if (r != hack.load(std::memory_order_relaxed)) {
                (*hfn)();
                hack.store(r, std::memory_order_release);
                plat::wake_all(hack);
                continue;
            }
            await(ev, e);
        }
        waiting.store(false);
    }
    ~JobThread() {
        if (!on) return;
        stop.store(true);
        go.fetch_add(1);
        plat::wake_all(go);
        th.join();
    }
};

// Output gain, -1.5 dB (the decoder's peaks reach about 1.1): applied only to what goes to the sink; the detector, the
// redraw and the start cut work on the unscaled audio.
static const float OUT_GAIN = 0.84139514f;

// Where the audio goes and what the player reports; the browser binds these to its ring buffer.
struct Sink {
    std::function<void(const float *, int64_t)> push;  // samples at the output rate
    std::function<double()> buffered;                  // seconds queued and not yet played
    std::function<bool()> cancelled;
    std::function<void(const std::string &)> status;   // JSON
    std::function<void(const std::string &)> sources;  // JSON [[begin, end], ...]: each chunk's input bytes, before the audio
    std::function<void(int64_t, double)> chunk;        // chunk c starts at this output sample
    // redraw: the player's read position (output samples of the request it has
    // consumed), and the replacement of the audio from output sample `at` on by n new samples, only if `at` is at
    // least `guard` samples past the read position (1 done, 0 refused, 2 done but the player passed `at` meanwhile)
    std::function<int64_t()> played;
    std::function<int(int64_t, const float *, int64_t, int64_t)> replace;
    // played() is a boundary, not a playing position: the samples that can no longer be replaced (a host that holds
    // audio back before it sends it on). A redraw may then swap from any point at or after it: nothing is added for
    // the time the redraw takes, and no margin
    bool boundary = false;
};

typedef std::vector<std::array<uint32_t, 2>> Ranges;

// The input bytes each chunk was read from: its first and last spoken character (whitespace skipped on both sides,
// a character the chunker added matches nothing), through the verbalizer's map: copied text exactly, a reading its
// whole input span.
static Ranges chunk_sources(const std::string &spoken, const std::vector<chunker::S> &cs, const std::vector<rv::Piece> &map) {
    const chunker::S sp = chunker::from_utf8(spoken);
    std::vector<uint32_t> off;  // byte offset of each code point (decoded as chunker::from_utf8 does)
    for (size_t i = 0; i < spoken.size();) {
        const unsigned char c = spoken[i];
        off.push_back((uint32_t)i);
        i = std::min(spoken.size(), i + 1 + (c >= 0xF0 ? 3 : c >= 0xE0 ? 2 : c >= 0xC0 ? 1 : 0));
    }
    off.push_back((uint32_t)spoken.size());
    auto in_at = [&](uint32_t byte, bool end) -> uint32_t {  // input offset of a spoken byte (end: of the byte before)
        auto it = std::upper_bound(map.begin(), map.end(), byte, [](uint32_t v, const rv::Piece &p) { return v < p.out_begin; });
        if (it == map.begin()) return 0;
        const rv::Piece &p = *(it - 1);
        if (!p.copied) return end ? p.in_end : p.in_begin;
        return std::min(p.in_end, p.in_begin + (byte - p.out_begin) + (end ? 1 : 0));
    };
    Ranges out;
    size_t j = 0;
    for (const chunker::S &c : cs) {
        int64_t first = -1, last = -1;
        for (char32_t ch : c) {
            if (chunker::is_space(ch)) continue;
            size_t k = j;
            while (k < sp.size() && chunker::is_space(sp[k])) k++;
            if (k < sp.size() && sp[k] == ch) { if (first < 0) first = (int64_t)k; last = (int64_t)k; j = k + 1; }
        }
        if (first < 0) { const uint32_t e = out.empty() ? 0 : out.back()[1]; out.push_back({e, e}); continue; }
        const uint32_t b = in_at(off[first], false), e = in_at(off[last + 1] - 1, true);
        out.push_back({b, std::max(b, e)});
    }
    return out;
}

static std::string ranges_json(const Ranges &r) {
    std::string j = "[";
    for (size_t i = 0; i < r.size(); i++) j += (i ? ",[" : "[") + std::to_string(r[i][0]) + "," + std::to_string(r[i][1]) + "]";
    return j + "]";
}

static std::string jstr(const std::string &s) {  // JSON string literal
    std::string o = "\"";
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += (char)c; }
        else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += (char)c;
    }
    return o + "\"";
}

struct Engine {
    std::unique_ptr<Pool> pool, vpool, apool;  // pipelined: vpool the decoder's threads, apool all threads (text reading)
    JobThread vth;                      // runs the decoder on vpool next to the frame model
    WebHelper helper, vhelper;          // share: vpool takes part of the frame model's linears while vth is idle, the
                                        // frame model's pool part of the decoder's while the caller waits for it
    FM m;
    VVDec vv;
    BWE bwe;
    PitchDetect det;  // the redraw's detector (detector.bin); without it the engine speaks without redraw
    bool loaded = false, has_bwe = false, has_det = false;
    int threads = 1;
    float temp = 0.7f;  // sampling temperature and flow head passes (midpoint) of the model
    int passes = 8;
    float cutoff = 0.85f;  // redraw: the spot measure that flags a spot, redraws per spot, seconds kept ahead of the player
    int max_redraws = 3;
    double margin = 0.05;
    std::vector<int> cpus, little;  // native builds: the cores of the pool's threads, and which of them are little cores
    std::string load_msg;
    // set (tts_gpu): requests decode through it and ext_begin starts each request there, else the CPU decoder
    FrameDecoder *ext_dec = nullptr;
    std::function<void()> ext_begin;
    // lazy_vv (tts_lazy_vv): vv.bin is loaded only when the CPU decoder is needed (ensure_vv), through load_hook,
    // which puts the file into the heap and registers it (false: not available)
    bool lazy_vv = false;
    std::function<bool(const char *)> load_hook;
    std::string dir_;
    bool ensure_vv() {
        if (vv.NS > 0) return true;
        if (!load_hook || !load_hook("vv.bin") || !vv.load((dir_ + "/vv.bin").c_str(), pool.get())) return false;
        std::vector<float> st0(m.prime);
        st0.insert(st0.end(), m.first_lat.begin(), m.first_lat.end());
        vv.set_start(st0.data(), (int64_t)st0.size() / 64);
        return true;
    }
    std::vector<int> cores(int from, int n) const {  // the cores of n threads from thread `from` on (none: not pinned)
        return (int)cpus.size() >= from + n ? std::vector<int>(cpus.begin() + from, cpus.begin() + from + n) : std::vector<int>();
    }

    // threads: pool size; dir: where lm.bin, misc.bin, vv.bin (bwe.bin) are (paths or registered buffers)
    std::string load(const std::string &dir, int threads_, bool want_bwe) {
        const double t0 = now_s();
        threads = std::max(1, threads_);
        pool.reset(new Pool());
        pool->start(g_cfg.pipe() ? g_cfg.pipe_fm : threads, cores(0, g_cfg.pipe() ? g_cfg.pipe_fm : threads), little);
        if (g_cfg.pipe()) {
            vpool.reset(new Pool());
            vpool->start(g_cfg.pipe_vv, cores(g_cfg.pipe_fm, g_cfg.pipe_vv), little);
            threads = g_cfg.pipe_fm + g_cfg.pipe_vv;
            apool.reset(new Pool());
            apool->start(threads, cores(0, threads), little);
            if (g_cfg.share) {
                helper.pool = vpool.get();
                helper.start = [this](const std::function<void()> &f) {
                    if (!vth.idle()) return false;
                    vth.run([&f]() { f(); }, false);
                    return true;
                };
                helper.wait = [this]() { vth.wait(); };
                t_web_helper = &helper;
                vhelper.pool = pool.get();
                vhelper.start = [this](const std::function<void()> &f) { return vth.ask(f); };
                vhelper.wait = [this]() { vth.asked_wait(); };
                vth.helper = &vhelper;
            }
            vth.start();
            const std::vector<int> vc = cores(g_cfg.pipe_fm, 1);  // the decoder thread is the first of vpool
            if (!vc.empty()) { vth.run([&vc]() { Pool::pin(vc[0]); }, false); vth.wait(); }
        }
        g_spin_s = 3e-3;
        Options opt;
        native_kernels(opt);
        opt.bf16 = g_cfg.bf16;
        m.load((dir + "/lm.bin").c_str(), (dir + "/misc.bin").c_str(), pool.get(), opt, 4096);
        apply_act8(m);
        m.set_sampler(false, passes);
        Pool *dp = g_cfg.pipe() ? vpool.get() : pool.get();  // pipelined: the decoder and the detector run on vth
        dir_ = dir;
        if (!lazy_vv && !vv.load((dir + "/vv.bin").c_str(), dp)) return "cannot load " + dir + "/vv.bin";
        vv.wrows.act8 = g_cfg.act8_vv;
        has_det = det.load((dir + "/detector.bin").c_str(), dp);
        has_bwe = want_bwe && bwe.load((dir + "/bwe.bin").c_str(), &m.lm);
        const double t1 = now_s();
        warm_up();
        char b[320];
        const std::string opts = g_cfg.str();
        snprintf(b, sizeof b, "ready in %.1f s (warm-up %.1f s), %d threads, redraw %s, upsampler %s%s%s", now_s() - t0, now_s() - t1,
                 threads, has_det ? "on" : "off (no detector.bin)", has_bwe ? "loaded" : "off", opts.empty() ? "" : ", options ",
                 opts.c_str());
        load_msg = b;
        loaded = true;
        return b;
    }
    void set_threads(int n) {  // a new pool of n threads (benchmarks); one pool from here on, not pipelined
        t_web_helper = nullptr;
        vth.helper = nullptr;
        g_cfg.pipe_fm = g_cfg.pipe_vv = 0;
        vpool.reset();  // first: their threads go back to the page's thread pool
        apool.reset();
        pool.reset();
        pool.reset(new Pool());
        pool->start(n, cpus, little);
        threads = n;
        m.lm.pool = pool.get();
        vv.pool = pool.get();
        bwe.pool = pool.get();
        det.pool = pool.get();
    }
    // The decoder's start state (the priming latents of misc.bin "prime" decoded, as the model expects, then
    // the latents of the model's fixed first frames if it has any; their audio is never played), then one short
    // request through every part, so the first real one is not slower.
    void warm_up() {
        (void)rv::verbalize("Warm up at 10:30 on 5/9, $5.");  // the verbalizer builds its tables on the first call
        std::vector<float> st0(m.prime);
        st0.insert(st0.end(), m.first_lat.begin(), m.first_lat.end());
        if (vv.NS == 0) return;  // lazy_vv: no CPU decoder (the GPU decodes)
        vv.set_start(st0.data(), (int64_t)st0.size() / 64);
        Stats st;
        Sink none;
        speak_chunks({Text{'H', 'i', '.'}}, has_bwe, 0, 0.0, none, st, now_s(), true);
    }

    struct Stats {
        double t_verb = 0, t_first = -1, t_gen = 0, t_wait = 0, t_vv = 0, t_bwe = 0, t_det = 0;
        int64_t ahead = 0, undone = 0;  // pipelined: frames made ahead, of those undone
        int64_t frames = 0, samples = 0, chars = 0, nchunks = 0, n_wait = 0;
        std::string unknown;
        bool redraw = false;
        RedrawStats rd;                               // the redraw's counters
        std::string rd_log;                           // the redraw's decisions, one JSON object per spot
        int64_t replaced = 0, player_refused = 0, late = 0;  // swaps done; refused by the player's own check; faults
        int64_t cut = -1, sound = -1;                         // start cut and first sound (utterance samples; -1: none)
    };

    // Text (UTF-8) -> chunks of code points; characters the model does not know are collected (read as spaces).
    // sources: each chunk's input bytes (chunk_sources).
    std::vector<Text> prepare(const std::string &text, bool verbalize, Stats &st, Ranges *sources = nullptr) {
        const double t0 = now_s();
        std::vector<rv::Piece> map;
        const std::string spoken = !verbalize ? text : sources ? rv::verbalize(text, map) : rv::verbalize(text);
        if (!verbalize) map.push_back({0, (uint32_t)text.size(), 0, (uint32_t)text.size(), true});
        const std::vector<chunker::S> cs = chunker::chunks(chunker::from_utf8(spoken));
        if (sources) *sources = chunk_sources(spoken, cs, map);
        std::vector<Text> chunks;
        std::u32string unk;
        for (auto &c : cs) {
            chunks.emplace_back(c.begin(), c.end());
            st.chars += (int64_t)c.size();
            for (char32_t ch : c)
                if (!std::binary_search(m.chars, m.chars + m.NCH, (int32_t)ch) && unk.find(ch) == std::u32string::npos) unk.push_back(ch);
        }
        st.unknown = chunker::to_utf8(unk);
        st.nchunks = (int64_t)chunks.size();
        st.t_verb = now_s() - t0;
        return chunks;
    }

    std::string status_json(const char *state, const Stats &st, double elapsed) const {
        const double rate = 1.0 / 0.133333;
        const int64_t F = std::max<int64_t>(1, st.frames);
        const RedrawStats &r = st.rd;
        char b[2048];
        snprintf(b, sizeof b,
                 "{\"state\":\"%s\",\"chunks\":%lld,\"chars\":%lld,\"unknown\":%s,\"verbalize_ms\":%.1f,\"first_audio_ms\":%.0f,"
                 "\"audio_s\":%.2f,\"busy_s\":%.2f,\"speed\":%.2f,\"prefill_ms\":%.0f,\"lm_ms\":%.2f,\"head_ms\":%.2f,"
                 "\"ae_ms\":%.2f,\"vv_ms\":%.2f,\"bwe_ms\":%.2f,\"det_ms\":%.2f,\"frames\":%lld,\"threads\":%d,\"paused_s\":%.2f,"
                 "\"pauses\":%lld,\"redraw\":%s,\"spots\":%lld,\"attempts\":%lld,\"clean\":[%lld,%lld,%lld],\"kept_best\":%lld,"
                 "\"budget\":%lld,\"previous_chunk\":%lld,\"extra_frames\":%lld,\"early_stop\":%lld,\"played_stop\":%lld,"
                 "\"refused\":%lld,\"replaced\":%lld,\"player_refused\":%lld,\"late\":%lld,\"det_off\":%lld,\"cut_ms\":%.1f,"
                 "\"first_sound_ms\":%.1f,\"frame_ms\":%.2f,\"ahead\":%lld,\"undone\":%lld,\"cfg\":\"%s\"}",
                 state, (long long)st.nchunks, (long long)st.chars, jstr(st.unknown).c_str(), 1e3 * st.t_verb,
                 st.t_first < 0 ? -1.0 : 1e3 * st.t_first, st.frames / rate, elapsed - st.t_wait,
                 st.frames > 0 ? (st.frames / rate) / std::max(1e-3, elapsed - st.t_wait) : 0.0, 1e3 * m.t_prefill,
                 1e3 * m.t_lm / F, 1e3 * m.t_head / F, 1e3 * m.t_ae / F, 1e3 * st.t_vv / F, 1e3 * st.t_bwe / F, 1e3 * st.t_det / F,
                 (long long)st.frames, threads, st.t_wait, (long long)st.n_wait, st.redraw ? "true" : "false", (long long)r.spots,
                 (long long)r.attempts, (long long)r.clean[0], (long long)r.clean[1], (long long)r.clean[2], (long long)r.kept_best,
                 (long long)r.budget, (long long)r.previous_chunk, (long long)r.extra_frames, (long long)r.early_stop,
                 (long long)r.played_stop, (long long)r.refused, (long long)st.replaced, (long long)st.player_refused,
                 (long long)st.late, (long long)r.det_off, st.cut < 0 ? -1.0 : st.cut / 24.0, st.sound < 0 ? -1.0 : st.sound / 24.0,
                 1e3 * std::max(0.0, elapsed - st.t_wait - m.t_prefill) / F, (long long)st.ahead, (long long)st.undone, g_cfg.str().c_str());
        return b;
    }

    // The decoder behind the redraw's hook: its streaming state as bytes (vv_dec.h).
    struct Dec : FrameDecoder {
        VVDec &v;
        JobThread *th = nullptr;  // pipelined: the decoder's thread
        explicit Dec(VVDec &d) : v(d) {}
        bool decode(const float *lat, float *audio) override { v.decode(lat, audio); return true; }
        bool start(const float *lat, float *audio, const std::function<void()> &after) override {
            if (!th) return false;
            th->run([this, lat, audio, after]() { v.decode(lat, audio); if (after) after(); });
            return true;
        }
        void wait() override { th->wait(); }
        size_t state_size() const override { return v.state_size() * sizeof(float); }
        void get_state(uint8_t *dst) const override { v.save_state(reinterpret_cast<float *>(dst)); }
        void set_state(const uint8_t *src) override { v.load_state(reinterpret_cast<const float *>(src)); }
    };
    // The redraw's player side, behind the start cut (lead_cut.h: positions here are player samples = utterance
    // samples - cut): the request's sink and the per-frame duties of a request (stop, chunk starts, status, pacing).
    // write() appends; replace() is a kept redraw's swap in one step, checked inside the player against its real read
    // position: refused (false, nothing changed) if the swap point is played or less than the guard
    // (one 50 ms output block) ahead, and the redraw then restores the original take in the generator. At 48 kHz the
    // writes go through the upsampler, whose state and output count are kept before each recent write: a swap restarts
    // it there, and a refused swap puts back the state the player's audio came from.
    struct Out : Playback {
        Engine &E;
        const Sink &sink;
        Stats &st;
        const Redraw *rd = nullptr;
        double t0, pace_hi;
        bool bwe;
        int64_t written = 0, out_n = 0, chunk_sent = -1;  // 24 kHz samples written; samples given to the sink (its rate)
        int64_t guard;                                    // 50 ms at the sink's rate
        struct Slot { int64_t pos = -1, out = 0; BWE::State s; };  // upsampler state before the write at player sample pos
        std::vector<Slot> slots;                                      // the recent writes', oldest first
        std::vector<float> o48;
        const LeadCut *lc = nullptr;
        Out(Engine &e, const Sink &s, Stats &st_, double t0_, double pace, bool bwe_)
            : E(e), sink(s), st(st_), t0(t0_), pace_hi(pace), bwe(bwe_), guard(bwe_ ? 2400 : 1200) {}
        std::vector<float> og;  // the sink's copy with the output gain
        const float *gained(const float *x, int64_t n) {
            og.resize((size_t)n);
            for (int64_t i = 0; i < n; i++) og[(size_t)i] = x[i] * OUT_GAIN;
            return og.data();
        }
        void emit(const float *x, int64_t n) {
            if (n <= 0) return;
            if (sink.push) sink.push(gained(x, n), n);
            if (st.t_first < 0) st.t_first = now_s() - t0;
            st.samples += n;
            out_n += n;
        }
        double vth0 = 0;  // the decoder thread's time at the start of the request
        void sync() {
            if (!rd) return;
            st.rd = rd->st;
            st.rd_log = rd->log;
            st.t_vv = rd->ahead ? E.vth.t_job - vth0 - E.det.t_det : rd->st.dec_s;  // pipelined: the decoder thread's own time
            st.ahead = rd->n_ahead;
            st.undone = rd->n_undone;
            st.t_det = E.det.t_det;
            if (lc) { st.cut = lc->c; st.sound = lc->s; }
        }
        int64_t utt(int64_t p) const { return p + (lc && lc->c > 0 ? lc->c : 0); }  // player sample -> utterance sample
        // a write's 24 kHz samples (one piece; pieces end at the utterance's frame boundaries) through the upsampler
        // into o48, its state before the write kept in `sl` (out: the output position it starts at)
        void upsample(int64_t at, const float *x, int64_t n, int64_t base, std::vector<Slot> &sl) {
            o48.clear();
            for (int64_t o = 0; o < n; o += 3200) {
                Slot s;
                s.pos = at + o;
                s.out = base + (int64_t)o48.size();
                E.bwe.save(s.s);
                sl.push_back(std::move(s));
                const double tb = now_s();
                E.bwe.push(x + o, std::min<int64_t>(3200, n - o), o48);
                st.t_bwe += now_s() - tb;
            }
        }
        // new slots from player sample `from` on: those of the audio they replace go (no stale state can be found later)
        void keep_slots(int64_t from, std::vector<Slot> &sl) {
            size_t k = 0;
            while (k < slots.size() && slots[k].pos < from) k++;
            slots.resize(k);
            for (Slot &x : sl) slots.push_back(std::move(x));
            if (slots.size() > (size_t)FM::NSNAP) slots.erase(slots.begin(), slots.end() - FM::NSNAP);
        }
        void write(int64_t at, const float *x, int64_t n) override {
            if (at != written) { fprintf(stderr, "write at %lld, end %lld\n", (long long)at, (long long)written); st.late++; return; }
            if (lc && st.cut < 0) { st.cut = lc->c; st.sound = lc->s; }  // known from the first write on
            if (sink.chunk && E.m.chunk_now != chunk_sent) {
                chunk_sent = E.m.chunk_now;
                sink.chunk(chunk_sent, (double)written * (bwe ? 2 : 1));
            }
            if (bwe) {
                std::vector<Slot> sl;
                upsample(at, x, n, out_n, sl);
                keep_slots(at, sl);
                emit(o48.data(), (int64_t)o48.size());
            } else {
                emit(x, n);
            }
            written += n;
            st.frames = utt(written) / E.vv.HOP;
            if (sink.status && st.frames % 8 == 0) { sync(); sink.status(E.status_json("speaking", st, now_s() - t0)); }
            if (pace_hi > 0 && sink.buffered && sink.buffered() >= pace_hi) {  // a player's pacing, in bursts
                const double w0 = now_s();
                while (sink.buffered() > pace_hi - 2.0 && !(sink.cancelled && sink.cancelled())) plat::sleep_ms(10);
                st.t_wait += now_s() - w0;
                st.n_wait++;
            }
        }
        bool replace(int64_t at, const float *x, int64_t n) override {
            if (!sink.replace) { st.player_refused++; return false; }
            int64_t rep = at;
            const float *y = x;
            int64_t ny = n;
            BWE::State now_st;   // 48 kHz: the upsampler as the player's audio left it, back if the swap is refused
            std::vector<Slot> sl;
            if (bwe) {
                const Slot *s0 = nullptr;
                for (const Slot &k : slots) if (k.pos == at) s0 = &k;
                if (!s0) { st.player_refused++; return false; }  // no state there (cannot happen: the redraw keeps fewer frames)
                E.bwe.save(now_st);
                rep = s0->out;
                E.bwe.load(s0->s);
                upsample(at, x, n, rep, sl);
                y = o48.data();
                ny = (int64_t)o48.size();
            }
            const int r = sink.replace(rep, gained(y, ny), ny, guard);
            if (!r) {
                st.player_refused++;
                if (bwe) E.bwe.load(now_st);
                return false;
            }
            if (bwe) keep_slots(at, sl);
            st.replaced++;
            st.late += r == 2;  // the player passed the swap point while it was written (a fault; counted)
            out_n = rep + ny;
            written = at + n;
            st.frames = utt(written) / E.vv.HOP;
            return true;
        }
        // 24 kHz player samples played dt seconds from now (no player: none ever); at 48 kHz plus the upsampler's
        // look-ahead (LeadCut adds the start cut: utterance samples for the redraw)
        double played(double dt) override {
            const int64_t p = sink.played ? sink.played() : -1;
            if (p < 0) return -1e18;
            const double lag = bwe ? (double)std::max<int64_t>(0, E.bwe.n24 - E.bwe.nout / 2) : 0.0;
            if (sink.boundary) return (double)p / (bwe ? 2.0 : 1.0) + lag - 0.5;  // half a sample below: a swap at the boundary passes
            return ((double)p / (bwe ? 48000.0 : 24000.0) + dt) * 24000.0 + lag;
        }
        void charge(int64_t frames, int64_t) override {
            if (frames && sink.cancelled && sink.cancelled()) throw Cancelled();
        }
    };

    // One request. pace_hi > 0: once that many seconds are queued, wait until pace_hi - 2 (in bursts: a sleep after
    // every frame lets the cores clock down). Throws nothing; a stop ends it early.
    Stats speak(const std::string &text, bool verbalize, bool use_bwe, uint64_t seed, double pace_hi, const Sink &sink, bool redraw = true,
                bool lead_cut = true) {
        Stats st;
        const double t0 = now_s();
        Ranges sources;
        const std::vector<Text> chunks = prepare(text, verbalize, st, sink.sources ? &sources : nullptr);
        if (chunks.empty()) { if (sink.status) sink.status(status_json("nothing to say", st, 0)); return st; }
        if (sink.sources) sink.sources(ranges_json(sources));
        speak_chunks(chunks, use_bwe, seed, pace_hi, sink, st, t0, redraw, lead_cut);
        return st;
    }
    // The generation part of a request: ready chunks (code points) -> audio, with the redraw of flagged spots
    // unless redraw is false. t0: when the request began.
    void speak_chunks(const std::vector<Text> &chunks, bool use_bwe, uint64_t seed, double pace_hi, const Sink &sink, Stats &st,
                      double t0, bool redraw, bool lead_cut = true) {
        use_bwe = use_bwe && has_bwe;
        st.redraw = redraw && has_det;
        if (!ext_dec) ensure_vv();
        if (vv.NS > 0) vv.begin();
        if (ext_begin) ext_begin();
        if (use_bwe) bwe.reset();
        std::mt19937_64 rng(seed);
        m.lead = sink.buffered ? std::function<double()>(sink.buffered) : nullptr;
        m.t_prefill = m.t_lm = m.t_head = m.t_ae = m.t_embed = 0;
        det.t_det = 0;
        Dec dec(vv);
        if (g_cfg.pipe()) dec.th = &vth;
        Out out(*this, sink, st, t0, pace_hi, use_bwe);
        out.vth0 = vth.t_job;
        LeadCut lc(out);  // the player starts at the request's first sound, unless lead_cut is off
        Redraw rd(m, ext_dec ? *ext_dec : (FrameDecoder &)dec, lead_cut ? (Playback &)lc : (Playback &)out, st.redraw ? &det : nullptr);
        rd.cut = cutoff;
        rd.max_redraws = max_redraws;
        rd.margin = margin;
        rd.ahead = g_cfg.pipe();
        if (g_cfg.pipe())  // a chunk's text is read on all threads, its frames on the two pools
            rd.reading = [this](bool on) {
                m.lm.pool = on ? apool.get() : pool.get();
                t_web_helper = on || !g_cfg.share ? nullptr : &helper;
            };
        out.rd = &rd;
        out.lc = lead_cut ? &lc : nullptr;
        if (sink.status) sink.status(status_json("speaking", st, now_s() - t0));
        try {
            rd.speak(chunks, temp, rng);
            if (lead_cut) lc.finish();
            if (use_bwe) { out.o48.clear(); bwe.finish(out.o48); out.emit(out.o48.data(), (int64_t)out.o48.size()); }
            st.t_gen = now_s() - t0;
            out.sync();
            if (sink.status) sink.status(status_json("generated", st, st.t_gen));
        } catch (const Cancelled &) {
            st.t_gen = now_s() - t0;
            out.sync();
            if (sink.status) sink.status(status_json("stopped", st, st.t_gen));
        }
        if (g_cfg.pipe()) { m.lm.pool = pool.get(); t_web_helper = g_cfg.share ? &helper : nullptr; }
        m.lead = nullptr;
    }
};

// ---------------------------------------------------------------- browser API
#ifdef __EMSCRIPTEN__
static Engine g_eng;

EM_JS(void, js_push, (const float *p, int n), { Module.ttsHooks.push(p, n); });
EM_JS(double, js_buffered, (), { return Module.ttsHooks.buffered(); });
EM_JS(int, js_cancelled, (), { return Module.ttsHooks.cancelled() ? 1 : 0; });
EM_JS(void, js_status, (const char *s), { Module.ttsHooks.status(UTF8ToString(s)); });
EM_JS(void, js_sources, (const char *s), { if (Module.ttsHooks.sources) Module.ttsHooks.sources(UTF8ToString(s)); });
EM_JS(void, js_chunk, (int c, double at), { if (Module.ttsHooks.chunk) Module.ttsHooks.chunk(c, at); });
EM_JS(double, js_played, (), { return Module.ttsHooks.played(); });
EM_JS(int, js_replace, (double at, const float *p, int n, int guard), { return Module.ttsHooks.replace(at, p, n, guard); });
EM_JS(void, js_result, (const float *p, int n), { Module.ttsHooks.result(p, n); });

#include "gpu_web.h"

extern "C" {
// The page copies each model file into a buffer from tts_alloc, then registers it under its name.
EMSCRIPTEN_KEEPALIVE void *tts_alloc(double bytes) { return aligned_alloc(64, ((size_t)bytes + 63) / 64 * 64); }
EMSCRIPTEN_KEEPALIVE void tts_register(const char *name, const uint8_t *p, double bytes) {
    web_files()[std::string("/m/") + name] = WebBuf{p, (size_t)bytes};
}
EMSCRIPTEN_KEEPALIVE const char *tts_load(int threads, int want_bwe) {
    static std::string msg;
    msg = g_eng.load("/m", threads, want_bwe != 0);
    return msg.c_str();
}
// Speed options before tts_load (Cfg::parse); returns "" or an error.
EMSCRIPTEN_KEEPALIVE const char *tts_config(const char *spec) {
    static std::string msg;
    msg = g_cfg.parse(spec);
    return msg.c_str();
}
EMSCRIPTEN_KEEPALIVE int tts_rate(int use_bwe) { return use_bwe && g_eng.has_bwe ? 48000 : 24000; }
// Runs until the text is spoken (generation side) or stopped; status JSON through js_status.
// realtime 1: the audio streams into the player's ring as it is made (pacing, the redraw's budget by the player's
// read position); 0: no player: every redraw gets its takes (no budget), and the whole audio is handed over at the end
// (Module.ttsHooks.result). lead_cut 0: the request is not started at its first sound.
EMSCRIPTEN_KEEPALIVE void tts_speak(const char *text, int verbalize, int use_bwe, double seed, double pace_hi, int redraw, int lead_cut,
                                    int realtime) {
    static std::vector<float> all;  // realtime 0: the request's audio
    Sink s;
    s.cancelled = []() { return js_cancelled() != 0; };
    s.status = [](const std::string &j) { js_status(j.c_str()); };
    s.sources = [](const std::string &j) { js_sources(j.c_str()); };
    s.chunk = [](int64_t c, double at) { js_chunk((int)c, at); };
    if (realtime) {
        s.push = [](const float *p, int64_t n) { js_push(p, (int)n); };
        s.buffered = []() { return js_buffered(); };
        s.played = []() { return (int64_t)js_played(); };
        s.replace = [](int64_t at, const float *p, int64_t n, int64_t guard) { return js_replace((double)at, p, (int)n, (int)guard); };
    } else {
        all.clear();
        s.push = [](const float *p, int64_t n) { all.insert(all.end(), p, p + n); };
        s.played = []() { return (int64_t)-1; };
        s.replace = [](int64_t at, const float *p, int64_t n, int64_t) {
            if (at > (int64_t)all.size()) return 0;
            all.resize((size_t)at);
            all.insert(all.end(), p, p + n);
            return 1;
        };
    }
    g_eng.speak(text, verbalize != 0, use_bwe != 0, (uint64_t)seed, realtime ? pace_hi : 0.0, s, redraw != 0, lead_cut != 0);
    if (!realtime) { js_result(all.data(), (int)all.size()); all.clear(); all.shrink_to_fit(); }
}
EMSCRIPTEN_KEEPALIVE const char *tts_sha256(const uint8_t *p, double n) {
    static std::string h;
    h = sha256_hex(p, (size_t)n);
    return h.c_str();
}
// Verbalizer + chunker only: JSON {chunks, sources (input bytes of each chunk), unknown}.
EMSCRIPTEN_KEEPALIVE const char *tts_prepare(const char *text, int verbalize) {
    static std::string out;
    Engine::Stats st;
    Ranges src;
    const std::vector<Text> ch = g_eng.prepare(text, verbalize != 0, st, &src);
    std::string j = "{\"chunks\":[";
    for (size_t i = 0; i < ch.size(); i++) {
        chunker::S s(ch[i].begin(), ch[i].end());
        j += (i ? "," : "") + jstr(chunker::to_utf8(s));
    }
    j += "],\"sources\":" + ranges_json(src) + ",\"unknown\":" + jstr(st.unknown) + ",\"verbalize_ms\":" + std::to_string(1e3 * st.t_verb) + "}";
    out = j;
    return out.c_str();
}
}
#endif
