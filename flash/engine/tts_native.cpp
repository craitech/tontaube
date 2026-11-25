// The engine as a native shared library (Linux, Windows, macOS; x86-64 or aarch64) with a C API, built by flash/cpu/engine.py on
// first use together with the text front end (flash_prepare / flash_verbalize of flash/text/text_front.cpp).
//
// Several engines per process: each has its own pool of threads, pinned to its own physical cores (no two threads
// on one core, no two engines on one core, the cores of one NUMA node, those sharing a last-level cache together);
// the model files are mapped once per process and shared. A request runs on the calling thread (the pool's first
// worker, pinned to the engine's first core while it runs) and hands its audio to C callbacks as it is made.
//
// Player modes (where the audio is consumed; positions in player samples at the output rate, the request starting at
// its first sound unless lead_cut is 0):
//   0  none: nothing is played before the request ends; every redraw gets its takes, no pacing
//   1  clock: a player that starts with the first audio and plays at real time
//   2  callback: sink.played() gives the samples the player has consumed
//   3  boundary: sink.played() gives the samples that can no longer be replaced (a host that holds audio back before
//      it sends it on); a redraw may replace from any point at or after it
// A redraw replaces audio from a point on only if that point is at least 50 ms past the player's position, the time
// the redraw takes counted as playing time (modes 1 and 2), or as the host's sink.replace decides.
#include "tts_web.cpp"

#include <mutex>
#include <set>

extern "C" {
typedef struct flash_sink {
    void *user;
    // samples for player positions start .. start + n - 1: appended (start = the end so far) or, after a redraw,
    // everything from start on replaced (start before the end)
    void (*audio)(void *user, int64_t start, const float *x, int64_t n);
    // optional: the host takes a replacement itself (1) or refuses it (0); audio() is then not called for it
    int (*replace)(void *user, int64_t at, const float *x, int64_t n, int64_t guard);
    int64_t (*played)(void *user);  // mode 2: samples the player has consumed; mode 3: samples that are final
    int (*cancelled)(void *user);   // optional: nonzero stops the request at the next frame
} flash_sink;
}

namespace {

using plat::Cpu;
using plat::parse_list;

// Cores claimed by the engines of this process, and how many engines there are (the speed options of tts_web.cpp,
// Cfg, are one setting per process).
std::mutex g_cores_mu;
std::multiset<int> g_claimed;
int g_engines = 0, g_threads = 0;  // engines alive, and their threads together

const int AUTO_MAX = 12;  // automatic thread count: at most this many (on a 12-core desktop CPU one stream still gained from 8 to 12)

// Automatic pinning: n free physical cores, big cores before little ones, on the NUMA node with the most free cores,
// grouped by last-level cache (the cache with the most free cores first). Fewer if there are not that many.
std::vector<int> pick_cores(const std::vector<Cpu> &cpus, int n) {
    double cmax = 0;
    for (const Cpu &c : cpus) cmax = std::max(cmax, c.cap);
    std::vector<Cpu> free;
    for (const Cpu &c : cpus)
        if (c.id == c.core && !g_claimed.count(c.id)) free.push_back(c);
    if (free.empty()) return {};
    auto little = [&](const Cpu &c) { return cmax > 0 && c.cap < 0.75 * cmax; };
    std::map<int, int> per_node, per_llc;
    for (const Cpu &c : free) { per_node[c.node] += !little(c); per_llc[c.llc] += !little(c); }
    int node = free[0].node;
    for (auto &kv : per_node) if (kv.second > per_node[node]) node = kv.first;
    std::stable_sort(free.begin(), free.end(), [&](const Cpu &a, const Cpu &b) {
        if ((a.node == node) != (b.node == node)) return a.node == node;
        if (little(a) != little(b)) return !little(a);
        if (a.llc != b.llc) return per_llc[a.llc] != per_llc[b.llc] ? per_llc[a.llc] > per_llc[b.llc] : a.llc < b.llc;
        return a.id < b.id;
    });
    std::vector<int> out;
    for (int i = 0; i < n && i < (int)free.size(); i++) out.push_back(free[i].id);
    return out;
}

bool model_file(const std::string &path) {  // a readable TTLM file
    FILE *f = plat::open_file(path.c_str(), "rb");
    if (!f) return false;
    char m[8] = {0};
    const bool ok = fread(m, 1, 8, f) == 8 && !memcmp(m, "TTLM0001", 8);
    fclose(f);
    return ok;
}

std::string jnum_list(const std::vector<int64_t> &v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); i++) s += (i ? "," : "") + std::to_string(v[i]);
    return s + "]";
}

}  // namespace

struct flash_engine {
    Engine E;
    std::atomic<int64_t> cut{-1};  // the running request's start cut (utterance samples) once known, else -1
    bool bwe = false;          // output at 48 kHz (AP-BWE)
    std::vector<int> claimed;  // cores this engine holds
    int threads = 0;           // its part of g_threads
    std::string info, stats, err;
    std::mutex busy;           // one request at a time
};

namespace {

// The player side of one request (see the player modes above).
struct Player {
    const flash_sink *cb;
    int mode;
    double rate;
    const Engine::Stats *st;
    std::atomic<int64_t> *cut;
    double t0 = -1;
    int64_t end = 0;
    bool stopped = false;
    int64_t played() {
        if (mode == 1) return t0 < 0 ? 0 : (int64_t)((now_s() - t0) * rate);
        if (mode >= 2 && cb->played) return cb->played(cb->user);
        return -1;
    }
    Sink sink() {
        Sink s;
        s.push = [this](const float *x, int64_t n) {
            if (t0 < 0) t0 = now_s();
            if (st->cut >= 0) cut->store(st->cut);  // before the host sees the first audio
            cb->audio(cb->user, end, x, n);
            end += n;
        };
        s.played = [this]() { return played(); };
        s.replace = [this](int64_t at, const float *x, int64_t n, int64_t guard) {
            if (cb->replace) {
                const int r = cb->replace(cb->user, at, x, n, guard);
                if (r) end = at + n;
                return r;
            }
            if (at > end) return 0;
            const int64_t p = played();
            if (p >= 0 && at < p + (mode == 3 ? 0 : guard)) return 0;
            cb->audio(cb->user, at, x, n);
            end = at + n;
            return 1;
        };
        if (mode == 1 || mode == 2) s.buffered = [this]() { const int64_t p = played(); return p < 0 ? 0.0 : (double)(end - p) / rate; };
        s.boundary = mode == 3;
        s.cancelled = [this]() {
            if (!stopped && cb->cancelled && cb->cancelled(cb->user)) stopped = true;
            return stopped;
        };
        return s;
    }
};

// Rows of the key/value cache chunk c may need (fm.cpp layout, frames at most text + OFFR): text of the previous,
// own and next chunk with their cues and markers, the previous chunk's frames and its own.
bool chunks_fit(const FM &m, const std::vector<Text> &ch) {
    const int64_t RS = ch.size() > 1 ? m.lm.S / 2 : m.lm.S;
    for (size_t c = 0; c < ch.size(); c++) {
        const int64_t prev = c ? (int64_t)ch[c - 1].size() + 2 : 0, cur = (int64_t)ch[c].size() + 2;
        const int64_t next = c + 1 < ch.size() ? std::min<int64_t>((int64_t)ch[c + 1].size(), m.LOOK) + 2 : 0;
        const int64_t first = c == 0 ? (int64_t)m.first.size() / m.D : 0;
        if (2 * prev + 2 * cur + next + 2 * m.OFFR + first + 4 > RS) return false;
    }
    return true;
}

std::string stats_json(flash_engine *e, const Engine::Stats &st, const char *state) {
    Engine &E = e->E;
    std::vector<int64_t> frames(E.m.R.n_frames.begin(), E.m.R.n_frames.end()), prefix(E.m.R.n_prefix.begin(), E.m.R.n_prefix.end());
    if (!frames.empty()) frames[0] -= E.m.R.n_first;  // the fixed first frames are not the request's audio
    char b[256];
    snprintf(b, sizeof b, ",\"t_gen\":%.6f,\"t_first\":%.6f,\"cut\":%lld,\"sound\":%lld,\"samples\":%lld,\"stopped\":%s", st.t_gen,
             st.t_first, (long long)st.cut, (long long)st.sound, (long long)st.samples, !strcmp(state, "stopped") ? "true" : "false");
    return "{\"status\":" + E.status_json(state, st, st.t_gen) + ",\"frames\":" + jnum_list(frames) + ",\"prefix\":" + jnum_list(prefix) + b +
           ",\"spots\":[" + st.rd_log + "]}";
}

int run(flash_engine *e, const std::vector<Text> &chunks, Engine::Stats &st, double t0, uint64_t seed, int redraw, int lead_cut,
        int mode, double pace, const flash_sink *sink) {
    if (!e || !sink || !sink->audio) return -3;
    std::unique_lock<std::mutex> lock(e->busy, std::try_to_lock);
    if (!lock.owns_lock()) return -1;
    if (!chunks_fit(e->E.m, chunks)) return -2;
    plat::Affinity keep;  // the calling thread is the pool's first worker: on the engine's first core while it runs
    if (!e->E.cpus.empty()) Pool::pin(e->E.cpus[0]);
    t_web_helper = g_cfg.pipe() && g_cfg.share ? &e->E.helper : nullptr;  // this thread's share of the decoder's pool
    e->cut.store(-1);
    Player pl{sink, mode, e->bwe ? 48000.0 : 24000.0, &st, &e->cut};
    const Sink s = pl.sink();
    const FM &m = e->E.m;
    std::u32string unk;  // characters the model does not know (read as spaces)
    st.nchunks = (int64_t)chunks.size();
    st.chars = 0;
    for (auto &c : chunks) {
        st.chars += (int64_t)c.size();
        for (int32_t ch : c)
            if (!std::binary_search(m.chars, m.chars + m.NCH, ch) && unk.find((char32_t)ch) == std::u32string::npos) unk.push_back((char32_t)ch);
    }
    st.unknown = chunker::to_utf8(unk);
    e->E.speak_chunks(chunks, e->bwe, seed, s.buffered && pace > 0 ? pace : 0.0, s, st, t0, redraw != 0, lead_cut != 0);
    e->stats = stats_json(e, st, pl.stopped ? "stopped" : "done");
    return pl.stopped ? 1 : 0;
}

}  // namespace

extern "C" {

// A new engine on the model files in dir (lm.bin, misc.bin, vv.bin; detector.bin for the redraw; bwe.bin for 48 kHz).
// threads <= 0: automatic; more than the process can run at once (physical cores it may use, cgroup quota, less the
// other engines' threads) are cut to that. cpus: "" or "auto" (free physical cores, see above), "none" (no pinning),
// or a list such as "0-5,12" (one thread per entry the process may use, in that order). options: "key=value" pairs separated by spaces: temp, passes, solver
// (midpoint | euler), cutoff, max_takes, margin_ms, bwe (0 | 1), and the speed options of tts_config (Cfg; the same
// for every engine of a process; pipe=A,B sets the threads to A + B). Null on failure, the reason in err.
flash_engine *flash_engine_create(const char *dir, int threads, const char *cpus, const char *options, char *err, int err_len) {
    auto fail = [&](const std::string &m) -> flash_engine * {
        if (err && err_len > 0) snprintf(err, (size_t)err_len, "%s", m.c_str());
        return nullptr;
    };
    const std::string d = dir ? dir : "";
    for (const char *f : {"lm.bin", "misc.bin", "vv.bin"})
        if (!model_file(d + "/" + f)) return fail("not a model file: " + d + "/" + f);
    std::unique_ptr<flash_engine> e(new flash_engine());
    Engine &E = e->E;
    bool euler = false, bwe = false;
    std::string speed;  // the options for Cfg
    std::istringstream in(options ? options : "");
    for (std::string kv; in >> kv;) {
        const size_t eq = kv.find('=');
        const std::string key = kv.substr(0, eq), val = eq == std::string::npos ? "" : kv.substr(eq + 1);
        const double v = atof(val.c_str());
        if (key == "temp") E.temp = (float)v;
        else if (key == "passes") E.passes = std::max(1, (int)v);
        else if (key == "solver" && (val == "midpoint" || val == "euler")) euler = val == "euler";
        else if (key == "cutoff") E.cutoff = (float)v;
        else if (key == "max_takes") E.max_redraws = std::max(0, (int)v - 1);
        else if (key == "margin_ms") E.margin = v / 1000.0;
        else if (key == "bwe") bwe = v != 0;
        else speed += " " + kv;
    }
    Cfg cfg;
    const std::string bad = cfg.parse(speed);
    if (!bad.empty()) return fail(bad);
    if (cfg.pipe()) threads = cfg.pipe_fm + cfg.pipe_vv;
    const std::vector<Cpu> topo = plat::cpus();
    int ncores = 0;
    for (const Cpu &c : topo) ncores += c.id == c.core;
    const double quota = plat::cpu_quota();
    const std::string spec = cpus ? cpus : "";
    std::unique_lock<std::mutex> lock(g_cores_mu);  // held while the engine loads (Cfg is read there)
    if (g_engines > 0 && cfg.str() != g_cfg.str()) return fail("the speed options differ from those of the other engines in this process");
    g_cfg = cfg;
    // Never more threads than the process can run at once (the pool's workers spin: with more threads than CPUs a
    // frame takes many times longer): at most the physical cores it may use and its cgroup quota, less what the
    // other engines of the process hold.
    const int asked = threads;
    int nfree = 0;
    for (const Cpu &c : topo) nfree += c.id == c.core && !g_claimed.count(c.id);
    int usable = std::min(nfree, ncores - g_threads);
    if (quota > 0) usable = std::min(usable, (int)quota - g_threads);
    usable = std::max(1, usable);
    if (cfg.pipe() && threads > usable) return fail("pipe=A,B needs A + B usable cores, there are " + std::to_string(usable));
    if (spec.empty() || spec == "auto" || spec == "none") {
        threads = threads <= 0 ? std::min(usable, AUTO_MAX) : std::min(threads, usable);
        if (spec != "none") E.cpus = pick_cores(topo, threads);
        if ((int)E.cpus.size() < threads) E.cpus.clear();  // no free cores left (one thread): not pinned
    } else {
        for (int c : parse_list(spec)) {  // the listed CPUs the process may use, each once
            const bool ok = std::any_of(topo.begin(), topo.end(), [&](const Cpu &x) { return x.id == c; });
            if (g_claimed.count(c)) return fail("cpu " + std::to_string(c) + " is used by another engine of this process");
            if (ok && !std::count(E.cpus.begin(), E.cpus.end(), c)) E.cpus.push_back(c);
        }
        if (E.cpus.empty()) return fail("no usable cpu in the list: " + spec);
        if (cfg.pipe() && (int)E.cpus.size() != threads) return fail("pipe=A,B needs A + B cpus");
        if (quota > 0 && (int)E.cpus.size() > std::max(1, (int)quota - g_threads)) E.cpus.resize(std::max(1, (int)quota - g_threads));
        threads = (int)E.cpus.size();
    }
    double cmax = 0;
    for (const Cpu &c : topo) cmax = std::max(cmax, c.cap);
    for (const Cpu &c : topo)
        if (cmax > 0 && c.cap < 0.75 * cmax && std::count(E.cpus.begin(), E.cpus.end(), c.id)) E.little.push_back(c.id);
    std::string msg;
    {
        plat::Affinity keep;  // Pool::start pins the calling thread to the first core
        msg = E.load(d, threads, bwe);
        if (E.loaded && euler) E.m.set_sampler(true, E.passes);
    }
    if (!E.loaded) return fail(msg);
    for (int c : E.cpus) g_claimed.insert(c);
    e->claimed = E.cpus;
    e->threads = E.threads;
    g_threads += E.threads;
    g_engines++;
    lock.unlock();
    e->bwe = bwe && E.has_bwe;
    std::string cl = "[";
    for (size_t i = 0; i < E.cpus.size(); i++) cl += (i ? "," : "") + std::to_string(E.cpus[i]);
#ifdef TT_KERN
    // the linears' kernels: the measured choice per kind and rows at this thread count (the warm-up request measured
    // the one-row classes; "?" while a class is still being measured)
    const std::string ks = kern::lin_q_choices(E.threads);
    const char *kernels = ks.c_str();
#else
    const char *kernels = "scalar";
#endif
    char b[768];
    snprintf(b, sizeof b, "{\"threads\":%d,\"asked\":%d,\"usable\":%d,\"cpus\":%s],\"physical_cores\":%d,\"quota\":%.2f,\"redraw\":%s,"
             "\"rate\":%d,\"kernels\":\"%s\",\"message\":", E.threads, std::max(0, asked), usable, cl.c_str(), ncores, quota,
             E.has_det ? "true" : "false", e->bwe ? 48000 : 24000, kernels);
    e->info = b + jstr(msg) + "}";
    return e.release();
}

// Engine facts as JSON: threads (what it runs with), asked (what was asked for, 0 = automatic), usable (the most it
// could get), the cores it runs on, output rate, whether it can redraw, the kernels of the linears, its load message.
const char *flash_engine_info(flash_engine *e) { return e ? e->info.c_str() : "{}"; }

// One request on chunks of code points (n chunks; lens[i] code points each, one after another in cps), e.g. from
// flash_prepare. mode: the player mode (0 to 3, see above). pace > 0 (modes 1 and 2): once that many seconds are queued
// ahead of the player, wait until pace - 2 (in bursts). Returns 0 when spoken, 1 when stopped (cancelled), -1 if the engine is busy with another
// request, -2 if a chunk is too long for the cache, -3 without an audio callback.
int flash_engine_speak_chunks(flash_engine *e, const int32_t *cps, const int32_t *lens, int n, uint64_t seed, int redraw, int lead_cut,
                              int mode, double pace, const flash_sink *sink) {
    const double t0 = now_s();
    std::vector<Text> chunks;
    for (int i = 0, o = 0; i < n; o += lens[i], i++) chunks.emplace_back(cps + o, cps + o + lens[i]);
    Engine::Stats st;
    if (chunks.empty()) {
        if (e) e->stats = stats_json(e, st, "nothing to say");
        return 0;
    }
    return run(e, chunks, st, t0, seed, redraw, lead_cut, mode, pace, sink);
}

// One request on text (UTF-8, len bytes) through the engine's own front end (verbalize 0: read as written).
int flash_engine_speak(flash_engine *e, const char *text, int64_t len, int verbalize, uint64_t seed, int redraw, int lead_cut, int mode,
                       double pace, const flash_sink *sink) {
    if (!e) return -3;
    const double t0 = now_s();
    Engine::Stats st;
    const std::vector<Text> chunks = e->E.prepare(std::string(text, (size_t)len), verbalize != 0, st);
    if (chunks.empty()) {
        e->stats = stats_json(e, st, "nothing to say");
        return 0;
    }
    return run(e, chunks, st, t0, seed, redraw, lead_cut, mode, pace, sink);
}

// The flow head's sampler for the next requests: euler (1) or midpoint (0), head passes per frame (midpoint: 2 per
// step), temperature of the start noise. Returns 0, or -1 while a request runs.
int flash_engine_set_sampler(flash_engine *e, int euler, int passes, double temp) {
    if (!e) return -3;
    std::unique_lock<std::mutex> lock(e->busy, std::try_to_lock);
    if (!lock.owns_lock()) return -1;
    e->E.temp = (float)temp;
    const int P = euler ? std::max(1, passes) : 2 * std::max(1, passes / 2);
    if (e->E.m.euler != (euler != 0) || e->E.m.passes != P) e->E.m.set_sampler(euler != 0, P);
    e->E.passes = P;
    return 0;
}

// The running request's start cut in utterance samples (player sample p = utterance sample p + cut), known from its
// first audio on; -1 before that or without the start cut. May be called from the sink's callbacks or another thread.
int64_t flash_engine_cut(flash_engine *e) { return e ? e->cut.load() : -1; }

// The last request's numbers as JSON: status (the engine's status line), frames and prefix tokens per chunk, t_gen,
// t_first (first audio), cut and sound (utterance samples), spots (the redraw's decisions).
const char *flash_engine_stats(flash_engine *e) { return e ? e->stats.c_str() : "{}"; }

void flash_engine_destroy(flash_engine *e) {
    if (!e) return;
    Model &lm = e->E.m.lm;  // the key / value cache (Model::load)
    big_free(lm.kv, (size_t)lm.NL * 2 * lm.nkv * lm.S * HD * sizeof(kv_t));
    big_free(lm.k8, (size_t)lm.NL * lm.nkv * lm.S * HD);
    big_free(lm.k8s, (size_t)lm.NL * lm.nkv * lm.S * sizeof(float));
    big_free(lm.v8, (size_t)lm.NL * lm.nkv * lm.S * HD);
    big_free(lm.v8s, (size_t)lm.NL * lm.nkv * lm.S * sizeof(float));
    {
        std::lock_guard<std::mutex> g(g_cores_mu);
        for (int c : e->claimed) g_claimed.erase(g_claimed.find(c));
        g_threads -= e->threads;
        g_engines--;
    }
    delete e;
}

}  // extern "C"
