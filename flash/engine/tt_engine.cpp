// Transformer decode steps on a CPU (NEON on aarch64, WebAssembly SIMD in the browser build, AVX2 / AVX-512 on x86
// through app/native/kern, chosen at run time).
//
// Weight formats:
//   q4: codes uint8 [N][K/64][32] (byte i: input i low nibble, input i + 32 high nibble), fp16 scale/min per group
//   q8: codes int8 [N][K/128][128] = code - 128, fp16 scale and (shifted) min per group
//   w = code * scale + min. Activations are quantized per 64 inputs to int16 (or int8):
//   sum_k w_k x_k = sum_g scale_g * xscale_g * dot(code, q)_g + min_g * xsum_g
// The int16 activation is split into a signed high and an unsigned low byte so the dot products run on
// SDOT/UDOT (USDOT for q8): dot = 256 * dot(code, hi) + dot(code, lo), exact.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <vector>

#include "platform.h"
#include "web_kernels.h"

#ifdef __aarch64__
#include <arm_neon.h>
typedef __fp16 kv_t;
// NEON kernels by the instructions they need beyond ARMv8.0; native_kernels() picks them by the CPU's hwcaps, so one
// build runs on every aarch64 CPU
#ifdef __clang__
#define TT_DOT __attribute__((target("dotprod")))
#define TT_I8MM __attribute__((target("dotprod,i8mm")))
#define TT_FHM __attribute__((target("fp16fml")))
#else
#define TT_DOT __attribute__((target("+dotprod")))
#define TT_I8MM __attribute__((target("+dotprod+i8mm")))
#define TT_FHM __attribute__((target("+fp16fml")))
#endif
#ifndef __EMSCRIPTEN__
#include "kern/attn.h"
#include "kern/detect.h"
#include "kern/lin_q.h"
#include "kern/vv.h"
#define TT_ARM 1
#endif
#elif defined(__EMSCRIPTEN__) && defined(WEB_KV32)
typedef float kv_t;
#elif defined(__EMSCRIPTEN__)
typedef web_half kv_t;
#else
#include <immintrin.h>
#include "web_kernels.h"
#include "kern/act.h"
#include "kern/attn.h"
#include "kern/detect.h"
#include "kern/lin_bf16.h"
#include "kern/lin_q.h"
#include "kern/vv.h"
#define TT_X86 1
typedef web_half kv_t;  // fp16, as in the browser build: half the cache traffic of float
#endif
#if defined(TT_X86) || defined(TT_ARM)
#define TT_KERN 1  // native builds: the shared kernels of app/native/kern, instruction set chosen at run time
// the kern kernels (instruction set chosen at run time); false: the portable reference code (checks)
#ifdef TT_X86
static bool g_kf = kern::has_avx2();
#else
static bool g_kf = true;
#endif
#endif

static const int AG = 64;   // activation quantization group
static const int HD = 128;  // head dim
#ifndef __aarch64__
static const int MB = 8;  // largest prompt-processing batch
#endif
#ifdef TT_KERN
static const int MBP = 64;  // largest batch of feed() (prefill): the kern kernels read the weights once per batch
#else
static const int MBP = 8;  // = MB
#endif

using plat::now_s;

static inline float h2f(uint16_t h) {
#ifdef __aarch64__
    __fp16 v;
    memcpy(&v, &h, 2);
    return (float)v;
#elif defined(__F16C__)
    return _cvtsh_ss(h);
#else
    return web_h2f(h);
#endif
}

// Position of value element (column d, row j) in a head's transposed int8 value cache (see attn_values8).
static inline size_t v8_index(int64_t S, int d, int64_t j) {
    return ((((size_t)(d >> 4) * (S / 8) + (j >> 3)) * 8 + ((d >> 1) & 7)) * 16) + (d & 1) * 8 + (j & 7);
}

// ---------------------------------------------------------------- thread pool (dynamic chunks, spinning workers)
static double g_spin_s = 3e-3;  // workers spin this long after a region before they sleep

// Work items 0..total-1 are claimed in chunks from the front by the fast threads. Slow threads (little cores) join
// only regions that allow it and claim small chunks from the back, so the end of a region never waits long for them.
struct Pool {
    int n = 1;
    std::vector<plat::Thread> th;
    std::vector<int> cpus, little;  // the threads' CPUs (thread i on cpus[i]); those of little cores
    std::atomic<uint32_t> epoch{0}, epoch_slow{0};  // slow threads watch epoch_slow (bumped for regions they join)
    std::atomic<uint64_t> range{0};  // (front << 32) | back: unclaimed items are front..back-1
    std::atomic<int> pending{0}, sleepers{0}, active{1 << 20};  // threads with id >= active sit regions out
    std::atomic<bool> stop{false}, slow_on{true};
    std::atomic<int> slow_div{8};  // slow threads claim chunk / slow_div items at a time
    int64_t chunk = 1;
    int n_slow = 0;
    const std::function<void(int64_t, int64_t)> *fn = nullptr;

    static void pin(int cpu) { if (cpu >= 0) plat::pin_thread(cpu); }
    // the pool whose region this thread works in: a run() from inside one of its own regions runs inline
    static const Pool *&region() { static thread_local const Pool *p = nullptr; return p; }
    bool is_slow(int id) const { return id < (int)cpus.size() && std::count(little.begin(), little.end(), cpus[id]) > 0; }
    struct Arg { Pool *p; int id; };
    static void *entry(void *a) {
        Arg *arg = (Arg *)a;
        Pool *p = arg->p;
        pin(arg->id < (int)p->cpus.size() ? p->cpus[arg->id] : -1);
        const int arg_id = arg->id;
        const bool slow = p->is_slow(arg_id);
        delete arg;
        region() = p;
        std::atomic<uint32_t> &ep = slow ? p->epoch_slow : p->epoch;
        uint32_t seen = 0;
        for (;;) {
            // Decode has a parallel region every few hundred microseconds: spin, sleep only when idle for ms.
            double t_idle = 0;
            for (uint32_t spins = 0; ep.load(std::memory_order_acquire) == seen; spins++) {
                if (p->stop.load(std::memory_order_relaxed)) return nullptr;
                if ((spins & 1023) == 1023) {
                    const double t = now_s();
                    if (t_idle == 0) t_idle = t;
                    else if (t - t_idle > g_spin_s) {
                        p->sleepers.fetch_add(1);
                        plat::wait(ep, seen);
                        p->sleepers.fetch_sub(1);
                        t_idle = 0;
                    }
                }
            }
            seen = ep.load(std::memory_order_acquire);
            if (p->stop.load()) return nullptr;
            if (slow) p->work_back();
            else if (arg_id < p->active.load(std::memory_order_relaxed)) p->work_front();
            p->pending.fetch_sub(1, std::memory_order_release);
        }
    }
    void start(int threads, const std::vector<int> &cpu_list, const std::vector<int> &little_cpus = {}) {
        n = std::max(1, threads);
        cpus = cpu_list;
        little = little_cpus;
        for (int i = 1; i < n; i++) n_slow += is_slow(i);
        pin(cpus.empty() ? -1 : cpus[0]);
        for (int i = 1; i < n; i++) {
            plat::Thread t;
            t.start(entry, new Arg{this, i});
            th.push_back(t);
        }
    }
    void work_front() {
        uint64_t st = range.load(std::memory_order_relaxed);
        for (;;) {
            const uint64_t f = st >> 32, b = st & 0xffffffffu;
            if (f >= b) break;
            const uint64_t e = std::min<uint64_t>(f + (uint64_t)chunk, b);
            if (!range.compare_exchange_weak(st, (e << 32) | b, std::memory_order_relaxed)) continue;
            (*fn)((int64_t)f, (int64_t)e);
            st = range.load(std::memory_order_relaxed);
        }
    }
    void work_back() {
        const uint64_t c = (uint64_t)std::max<int64_t>(1, chunk / slow_div.load(std::memory_order_relaxed));
        uint64_t st = range.load(std::memory_order_relaxed);
        for (;;) {
            const uint64_t f = st >> 32, b = st & 0xffffffffu;
            if (f >= b) break;
            const uint64_t nb = b - f > c ? b - c : f;
            if (!range.compare_exchange_weak(st, (f << 32) | nb, std::memory_order_relaxed)) continue;
            (*fn)((int64_t)nb, (int64_t)b);
            st = range.load(std::memory_order_relaxed);
        }
    }
    // slow: little cores may help (large regions of many small independent items only)
    void run(int64_t total_, int64_t chunk_, const std::function<void(int64_t, int64_t)> &f, bool slow = false) {
        if (n == 1 || total_ <= chunk_ || region() == this) { f(0, total_); return; }
        const Pool *outer = region();
        region() = this;
        fn = &f;
        chunk = chunk_;
        const bool with_slow = slow && n_slow > 0 && slow_on.load(std::memory_order_relaxed);
        range.store((uint64_t)total_);
        pending.store(n - 1 - (with_slow ? 0 : n_slow));
        epoch.fetch_add(1, std::memory_order_release);
        if (with_slow) epoch_slow.fetch_add(1, std::memory_order_release);
        if (sleepers.load() > 0) {
            plat::wake_all(epoch);
            if (with_slow) plat::wake_all(epoch_slow);
        }
        work_front();
        while (pending.load(std::memory_order_acquire) > 0) {}
        region() = outer;
    }
    ~Pool() {
        stop.store(true);
        epoch.fetch_add(1);
        epoch_slow.fetch_add(1);
        plat::wake_all(epoch);
        plat::wake_all(epoch_slow);
        for (auto &t : th) t.join();
    }
};

// Native builds: the model files are copied into huge pages at load where possible (File::huge_copy); true keeps the
// page-cache mapping (option weights=mmap).
static bool g_weights_mmap = false;
#define TT_WEIGHTS_COPY 1
// ---------------------------------------------------------------- model file
struct Tensor { int dtype = 0, ndim = 0; uint64_t dims[4] = {0, 0, 0, 0}; const uint8_t *p = nullptr; uint64_t nbytes = 0; };

struct File {
    std::map<std::string, Tensor> t;
    const uint8_t *base = nullptr;
    size_t size = 0;
    bool open(const char *path) {
#ifdef __EMSCRIPTEN__
        if (web_file_lookup(path, &base, &size)) return index();
#endif
        base = plat::map_file(path, size, !g_weights_mmap);  // natively copied into huge pages where possible
        return base && index();
    }
    bool index() {
        if (memcmp(base, "TTLM0001", 8)) return false;
        uint32_t n;
        memcpy(&n, base + 8, 4);
        const uint8_t *e = base + 16;
        for (uint32_t i = 0; i < n; i++, e += 104) {
            Tensor x;
            uint32_t dt, nd;
            uint64_t off;
            memcpy(&dt, e + 48, 4);
            memcpy(&nd, e + 52, 4);
            memcpy(x.dims, e + 56, 32);
            memcpy(&off, e + 88, 8);
            memcpy(&x.nbytes, e + 96, 8);
            x.dtype = dt;
            x.ndim = nd;
            x.p = base + off;
            t[std::string((const char *)e)] = x;
        }
        return true;
    }
    const Tensor &get(const std::string &name) const {
        auto it = t.find(name);
        if (it == t.end()) { fprintf(stderr, "missing tensor %s\n", name.c_str()); exit(2); }
        return it->second;
    }
    bool has(const std::string &name) const { return t.count(name) > 0; }
};

// Zeroed memory for large buffers (natively 2 MiB aligned with transparent huge pages where the system allows them).
static void *big_alloc(size_t bytes) {
    void *a = plat::big_alloc(bytes);
    if (!a) { fprintf(stderr, "alloc of %zu MiB failed\n", bytes >> 20); exit(2); }
    return a;
}
using plat::big_free;  // bytes: as given to big_alloc

// ---------------------------------------------------------------- linear layers
struct Lin {
    int kind = 0;  // 0 q4, 1 q8, 2 dense fp32 (file entry <name>.w [N][K])
    int64_t N = 0, K = 0, G = 0, NQ = 0;
    const float *wf = nullptr;
    const uint8_t *codes = nullptr;
    const uint16_t *scales = nullptr, *mins = nullptr;
    const float *bias = nullptr;
    uint8_t *packed = nullptr;  // [quad][group]{codes[4][32 or 128], fp16 scales[4], fp16 mins[4]}
    int64_t stride() const { return kind == 0 ? 144 : 528; }

    void load(const File &f, const std::string &name) {
        if (f.has(name + ".w")) {
            const Tensor &w = f.get(name + ".w");
            kind = 2;
            N = w.dims[0];
            K = w.dims[1];
            wf = (const float *)w.p;
            bias = f.has(name + ".bias") ? (const float *)f.get(name + ".bias").p : nullptr;
            return;
        }
        const Tensor &c = f.get(name + ".codes");
        kind = c.dtype == 0 ? 0 : 1;
        N = c.dims[0];
        G = c.dims[1];
        K = G * (kind == 0 ? 64 : 128);
        codes = c.p;
        scales = (const uint16_t *)f.get(name + ".scales").p;
        mins = (const uint16_t *)f.get(name + ".mins").p;
        bias = f.has(name + ".bias") ? (const float *)f.get(name + ".bias").p : nullptr;
    }
    // One packed copy per matrix and layout in the process, shared by every engine (the codes are mapped once);
    // shared_packed() holds the lock while a new copy is filled.
    static std::mutex &pack_mu() { static std::mutex m; return m; }
    bool shared_packed(int layout, std::unique_lock<std::mutex> &lock) {
        static std::map<std::pair<const void *, int>, uint8_t *> copies;
        lock = std::unique_lock<std::mutex>(pack_mu());
        uint8_t *&c = copies[{codes, layout}];
        const bool have = c != nullptr;
        if (!have) c = (uint8_t *)big_alloc(NQ * G * stride());
        packed = c;
        return have;
    }
    // Rows in quads, one contiguous stream per quad: every core reads one sequential stream.
    void pack() {
        NQ = (N + 3) / 4;
        if (kind == 2) return;
        const int64_t cb = kind == 0 ? 32 : 128, st = stride();
        std::unique_lock<std::mutex> lock;
        if (shared_packed(0, lock)) return;
        for (int64_t q = 0; q < NQ; q++)
            for (int64_t g = 0; g < G; g++) {
                uint8_t *d = packed + (q * G + g) * st;
                for (int r = 0; r < 4; r++) {
                    const int64_t n = q * 4 + r;
                    uint16_t s = 0, m = 0;
                    if (n < N) {
                        memcpy(d + r * cb, codes + (n * G + g) * cb, cb);
                        s = scales[n * G + g];
                        m = mins[n * G + g];
                    } else {
                        memset(d + r * cb, 0, cb);
                    }
                    memcpy(d + 4 * cb + 2 * r, &s, 2);
                    memcpy(d + 4 * cb + 8 + 2 * r, &m, 2);
                }
            }
    }
    // Layout of the matrix-multiply kernels: per quad and group, for each row pair (r, r + 1) chunks of 16 bytes =
    // 8 code bytes of row r then the same 8 of row r + 1 (one USMMLA/SMMLA operand); then scales[4], mins[4].
    void pack_mm() {
        NQ = (N + 3) / 4;
        if (kind == 2) return;
        const int64_t cb = kind == 0 ? 32 : 128, st = stride();
        std::unique_lock<std::mutex> lock;
        if (shared_packed(1, lock)) return;
        for (int64_t q = 0; q < NQ; q++)
            for (int64_t g = 0; g < G; g++) {
                uint8_t *d = packed + (q * G + g) * st;
                for (int pr = 0; pr < 2; pr++)
                    for (int64_t c = 0; c < cb / 8; c++)
                        for (int k = 0; k < 2; k++) {
                            const int64_t n = q * 4 + pr * 2 + k;
                            uint8_t *dst = d + pr * 2 * cb + c * 16 + k * 8;
                            if (n < N) memcpy(dst, codes + (n * G + g) * cb + c * 8, 8);
                            else memset(dst, 0, 8);
                        }
                for (int r = 0; r < 4; r++) {
                    const int64_t n = q * 4 + r;
                    const uint16_t sc = n < N ? scales[n * G + g] : 0, mn = n < N ? mins[n * G + g] : 0;
                    memcpy(d + 4 * cb + 2 * r, &sc, 2);
                    memcpy(d + 4 * cb + 8 + 2 * r, &mn, 2);
                }
            }
    }
    size_t packed_bytes() const { return kind == 2 ? (size_t)N * K * 4 : NQ * G * stride(); }

    // Coarse copy of a q8 layer in the q4 format: code4 = (code8 + 128) >> 4, so w ~ code4 * (16 s) + (min - 120 s).
    // Owns its arrays (never freed: lives as long as the model).
    static uint16_t f2h(float f) {
#ifdef __aarch64__
        const __fp16 v = (__fp16)f;
        uint16_t h;
        memcpy(&h, &v, 2);
        return h;
#elif defined(__F16C__)
        return _cvtss_sh(f, 0);
#else
        return web_f2h(f);
#endif
    }
    void coarse_of(const Lin &src) {
        kind = 0;
        N = src.N;
        K = src.K;
        G = K / 64;
        bias = src.bias;
        uint8_t *c = (uint8_t *)big_alloc((size_t)N * G * 32);
        uint16_t *sc = (uint16_t *)big_alloc((size_t)N * G * 2), *mn = (uint16_t *)big_alloc((size_t)N * G * 2);
        for (int64_t n = 0; n < N; n++)
            for (int64_t g = 0; g < G; g++) {
                const int8_t *s8 = (const int8_t *)src.codes + (n * src.G + g / 2) * 128 + (g & 1) * 64;
                for (int k = 0; k < 32; k++)
                    c[(n * G + g) * 32 + k] = (uint8_t)(((s8[k] + 128) >> 4) | (((s8[k + 32] + 128) >> 4) << 4));
                const float s = h2f(src.scales[n * src.G + g / 2]), m = h2f(src.mins[n * src.G + g / 2]);
                sc[n * G + g] = f2h(16.f * s);
                mn[n * G + g] = f2h(m - 120.f * s);
            }
        codes = c;
        scales = sc;
        mins = mn;
    }
};

struct Act {  // quantized activations of one row
    std::vector<int16_t> q;
    std::vector<int8_t> hi;
    std::vector<uint8_t> lo;
    std::vector<int8_t> hl;  // mm kernels: per 8 inputs 8 high bytes then 8 low bytes, q = 256 * hi + lo, both signed
    std::vector<float> s, xs;
    void resize(int64_t K) {
        q.resize(K); hi.resize(K + 64); lo.resize(K + 64); hl.resize(2 * K + 64); s.resize(K / AG); xs.resize(K / AG);
    }
};

// mm: balanced split for USMMLA/SMMLA (hi = round(q / 256) must fit int8, so |q| <= 32639).
static void quantize(const float *x, int64_t K, Act &a, int bits, bool mm = false) {
#ifdef TT_X86
    if (bits == 16 && !mm) { kern::lin_q_impl().quant(x, 0, 0, 1, K, a.q.data(), a.s.data(), a.xs.data()); return; }
#endif
    const float qmax = bits == 8 ? 127.f : mm ? 32639.f : 32767.f;
    for (int64_t g = 0; g < K / AG; g++) {
        const float *v = x + g * AG;
#ifdef __aarch64__
        float32x4_t mx = vdupq_n_f32(0.f);
        for (int k = 0; k < AG; k += 4) mx = vmaxq_f32(mx, vabsq_f32(vld1q_f32(v + k)));
        const float amax = vmaxvq_f32(mx);
        const float inv = amax > 0.f ? qmax / amax : 0.f;
        int32x4_t sum = vdupq_n_s32(0);
        for (int k = 0; k < AG; k += 8) {
            const int32x4_t c0 = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(v + k), inv));
            const int32x4_t c1 = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(v + k + 4), inv));
            sum = vaddq_s32(sum, vaddq_s32(c0, c1));
            const int16x8_t c = vcombine_s16(vqmovn_s32(c0), vqmovn_s32(c1));
            vst1q_s16(a.q.data() + g * AG + k, c);
            if (mm) {
                vst1_s8(a.hl.data() + (g * AG + k) * 2, vrshrn_n_s16(c, 8));
                vst1_s8(a.hl.data() + (g * AG + k) * 2 + 8, vmovn_s16(c));
            } else if (bits == 8) {
                vst1_s8(a.hi.data() + g * AG + k, vmovn_s16(c));
            } else {
                vst1_s8(a.hi.data() + g * AG + k, vshrn_n_s16(c, 8));
                vst1_u8(a.lo.data() + g * AG + k, vreinterpret_u8_s8(vmovn_s16(c)));
            }
        }
        const float sc = amax / qmax;
        a.s[g] = sc;
        a.xs[g] = sc * (float)vaddvq_s32(sum);
#else
        float amax = 0.f;
        for (int k = 0; k < AG; k++) amax = std::max(amax, std::fabs(v[k]));
        const float sc = amax / qmax, inv = amax > 0.f ? qmax / amax : 0.f;
        int64_t sum = 0;
        for (int k = 0; k < AG; k++) {
            int c = (int)std::nearbyint(v[k] * inv);
            c = std::max(-(int)qmax, std::min((int)qmax, c));
            a.q[g * AG + k] = (int16_t)c;
            sum += c;
        }
        a.s[g] = sc;
        a.xs[g] = sc * (float)sum;
#endif
    }
}

// Reference kernels on the file layout (also the scalar build's kernels).
static void gemv_scalar(const Lin &L, const Act &a, float *y, int64_t n0, int64_t n1) {
    const int16_t *q = a.q.data();
    for (int64_t n = n0; n < n1; n++) {
        float acc = 0.f, mt = 0.f;
        if (L.kind == 0) {
            const uint8_t *row = L.codes + n * L.G * 32;
            for (int64_t g = 0; g < L.G; g++) {
                const uint8_t *c = row + g * 32;
                const int16_t *xq = q + g * AG;
                int32_t d = 0;
                for (int k = 0; k < 32; k++) d += (c[k] & 15) * xq[k] + (c[k] >> 4) * xq[k + 32];
                acc += (float)d * (h2f(L.scales[n * L.G + g]) * a.s[g]);
                mt += h2f(L.mins[n * L.G + g]) * a.xs[g];
            }
        } else {
            const int8_t *row = (const int8_t *)L.codes + n * L.G * 128;
            for (int64_t g = 0; g < L.G; g++) {
                const float ws = h2f(L.scales[n * L.G + g]), wm = h2f(L.mins[n * L.G + g]);
                for (int h = 0; h < 2; h++) {
                    const int8_t *c = row + g * 128 + h * 64;
                    const int16_t *xq = q + (g * 2 + h) * AG;
                    int32_t d = 0;
                    for (int k = 0; k < 64; k++) d += c[k] * xq[k];
                    acc += (float)d * (ws * a.s[g * 2 + h]);
                    mt += wm * a.xs[g * 2 + h];
                }
            }
        }
        y[n] = acc + mt + (L.bias ? L.bias[n] : 0.f);
    }
}

#ifdef __aarch64__
static int g_pf = 0;  // software prefetch distance in bytes (0: off)

// q4, quads q0..q1-1 of the packed stream. bits 16: hi/lo bytes; bits 8: hi only.
TT_DOT static void gemv_q4_neon(const Lin &L, const Act &a, float *y, int64_t q0, int64_t q1, int bits) {
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    const int8_t *hi = a.hi.data();
    const uint8_t *lo = a.lo.data();
    const float *as = a.s.data(), *xs = a.xs.data();
    const int64_t G = L.G;
    for (int64_t q = q0; q < q1; q++) {
        const uint8_t *p = L.packed + q * G * 144;
        float32x4_t yv = vdupq_n_f32(0.f), ym = vdupq_n_f32(0.f);
        for (int64_t g = 0; g < G; g++, p += 144) {
            __builtin_prefetch(p + g_pf);
            __builtin_prefetch(p + g_pf + 64);
            __builtin_prefetch(p + g_pf + 128);
            const int8x16_t h0 = vld1q_s8(hi + g * 64), h1 = vld1q_s8(hi + g * 64 + 16);
            const int8x16_t h2 = vld1q_s8(hi + g * 64 + 32), h3 = vld1q_s8(hi + g * 64 + 48);
            int32x4_t t[4];
            if (bits == 16) {
                const uint8x16_t l0 = vld1q_u8(lo + g * 64), l1 = vld1q_u8(lo + g * 64 + 16);
                const uint8x16_t l2 = vld1q_u8(lo + g * 64 + 32), l3 = vld1q_u8(lo + g * 64 + 48);
                for (int r = 0; r < 4; r++) {
                    const uint8x16_t c0 = vld1q_u8(p + r * 32), c1 = vld1q_u8(p + r * 32 + 16);
                    const uint8x16_t w0 = vandq_u8(c0, m4), w1 = vandq_u8(c1, m4);
                    const uint8x16_t w2 = vshrq_n_u8(c0, 4), w3 = vshrq_n_u8(c1, 4);
                    int32x4_t ah = vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(w0), h0);
                    uint32x4_t al = vdotq_u32(vdupq_n_u32(0), w0, l0);
                    ah = vdotq_s32(ah, vreinterpretq_s8_u8(w1), h1);
                    al = vdotq_u32(al, w1, l1);
                    ah = vdotq_s32(ah, vreinterpretq_s8_u8(w2), h2);
                    al = vdotq_u32(al, w2, l2);
                    ah = vdotq_s32(ah, vreinterpretq_s8_u8(w3), h3);
                    al = vdotq_u32(al, w3, l3);
                    t[r] = vaddq_s32(vshlq_n_s32(ah, 8), vreinterpretq_s32_u32(al));
                }
            } else {
                for (int r = 0; r < 4; r++) {
                    const uint8x16_t c0 = vld1q_u8(p + r * 32), c1 = vld1q_u8(p + r * 32 + 16);
                    int32x4_t ah = vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(vandq_u8(c0, m4)), h0);
                    ah = vdotq_s32(ah, vreinterpretq_s8_u8(vandq_u8(c1, m4)), h1);
                    ah = vdotq_s32(ah, vreinterpretq_s8_u8(vshrq_n_u8(c0, 4)), h2);
                    ah = vdotq_s32(ah, vreinterpretq_s8_u8(vshrq_n_u8(c1, 4)), h3);
                    t[r] = ah;
                }
            }
            const int32x4_t d = vpaddq_s32(vpaddq_s32(t[0], t[1]), vpaddq_s32(t[2], t[3]));
            const float32x4_t ws = vcvt_f32_f16(vld1_f16((const float16_t *)(p + 128)));
            const float32x4_t wm = vcvt_f32_f16(vld1_f16((const float16_t *)(p + 136)));
            yv = vfmaq_f32(yv, vcvtq_f32_s32(d), vmulq_n_f32(ws, as[g]));
            ym = vfmaq_n_f32(ym, wm, xs[g]);
        }
        float out[4];
        vst1q_f32(out, vaddq_f32(yv, ym));
        for (int r = 0; r < 4; r++) {
            const int64_t n = q * 4 + r;
            if (n < L.N) y[n] = out[r] + (L.bias ? L.bias[n] : 0.f);
        }
    }
}

// q8 (weight group 128 = two activation groups).
TT_I8MM static void gemv_q8_neon(const Lin &L, const Act &a, float *y, int64_t q0, int64_t q1, int bits) {
    const int8_t *hi = a.hi.data();
    const uint8_t *lo = a.lo.data();
    const float *as = a.s.data(), *xs = a.xs.data();
    const int64_t G = L.G;
    for (int64_t q = q0; q < q1; q++) {
        const uint8_t *p = L.packed + q * G * 528;
        float32x4_t yv = vdupq_n_f32(0.f), ym = vdupq_n_f32(0.f);
        for (int64_t g = 0; g < G; g++, p += 528) {
            const float32x4_t ws = vcvt_f32_f16(vld1_f16((const float16_t *)(p + 512)));
            const float32x4_t wm = vcvt_f32_f16(vld1_f16((const float16_t *)(p + 520)));
            for (int h = 0; h < 2; h++) {
                const int64_t ga = g * 2 + h;
                __builtin_prefetch(p + g_pf + h * 256);
                __builtin_prefetch(p + g_pf + h * 256 + 64);
                __builtin_prefetch(p + g_pf + h * 256 + 128);
                __builtin_prefetch(p + g_pf + h * 256 + 192);
                int32x4_t t[4];
                for (int r = 0; r < 4; r++) {
                    const int8_t *c = (const int8_t *)p + r * 128 + h * 64;
                    int32x4_t ah = vdupq_n_s32(0), al = vdupq_n_s32(0);
                    for (int j = 0; j < 4; j++) {
                        const int8x16_t w = vld1q_s8(c + 16 * j);
                        ah = vdotq_s32(ah, w, vld1q_s8(hi + ga * 64 + 16 * j));
                        if (bits == 16) al = vusdotq_s32(al, vld1q_u8(lo + ga * 64 + 16 * j), w);
                    }
                    t[r] = bits == 16 ? vaddq_s32(vshlq_n_s32(ah, 8), al) : ah;
                }
                const int32x4_t d = vpaddq_s32(vpaddq_s32(t[0], t[1]), vpaddq_s32(t[2], t[3]));
                yv = vfmaq_f32(yv, vcvtq_f32_s32(d), vmulq_n_f32(ws, as[ga]));
                ym = vfmaq_n_f32(ym, wm, xs[ga]);
            }
        }
        float out[4];
        vst1q_f32(out, vaddq_f32(yv, ym));
        for (int r = 0; r < 4; r++) {
            const int64_t n = q * 4 + r;
            if (n < L.N) y[n] = out[r] + (L.bias ? L.bias[n] : 0.f);
        }
    }
}

// Several activation rows per pass (prompt processing): weights are unpacked once per group and reused.
static const int MB = 8;

TT_DOT static void gemv_q4_neon_m(const Lin &L, const Act *acts, int M, float *y, int64_t ystride, int64_t q0, int64_t q1) {
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    const int64_t G = L.G;
    for (int64_t q = q0; q < q1; q++) {
        const uint8_t *p = L.packed + q * G * 144;
        float32x4_t yv[MB], ym[MB];
        for (int m = 0; m < M; m++) yv[m] = ym[m] = vdupq_n_f32(0.f);
        for (int64_t g = 0; g < G; g++, p += 144) {
            __builtin_prefetch(p + g_pf);
            __builtin_prefetch(p + g_pf + 64);
            __builtin_prefetch(p + g_pf + 128);
            uint8x16_t w[4][4];
            for (int r = 0; r < 4; r++) {
                const uint8x16_t c0 = vld1q_u8(p + r * 32), c1 = vld1q_u8(p + r * 32 + 16);
                w[r][0] = vandq_u8(c0, m4);
                w[r][1] = vandq_u8(c1, m4);
                w[r][2] = vshrq_n_u8(c0, 4);
                w[r][3] = vshrq_n_u8(c1, 4);
            }
            const float32x4_t ws = vcvt_f32_f16(vld1_f16((const float16_t *)(p + 128)));
            const float32x4_t wm = vcvt_f32_f16(vld1_f16((const float16_t *)(p + 136)));
            for (int m = 0; m < M; m++) {
                const int8_t *hi = acts[m].hi.data() + g * 64;
                const uint8_t *lo = acts[m].lo.data() + g * 64;
                int8x16_t h[4];
                uint8x16_t l[4];
                for (int i = 0; i < 4; i++) { h[i] = vld1q_s8(hi + 16 * i); l[i] = vld1q_u8(lo + 16 * i); }
                int32x4_t t[4];
                for (int r = 0; r < 4; r++) {
                    int32x4_t ah = vdupq_n_s32(0);
                    uint32x4_t al = vdupq_n_u32(0);
                    for (int i = 0; i < 4; i++) {
                        ah = vdotq_s32(ah, vreinterpretq_s8_u8(w[r][i]), h[i]);
                        al = vdotq_u32(al, w[r][i], l[i]);
                    }
                    t[r] = vaddq_s32(vshlq_n_s32(ah, 8), vreinterpretq_s32_u32(al));
                }
                const int32x4_t d = vpaddq_s32(vpaddq_s32(t[0], t[1]), vpaddq_s32(t[2], t[3]));
                yv[m] = vfmaq_f32(yv[m], vcvtq_f32_s32(d), vmulq_n_f32(ws, acts[m].s[g]));
                ym[m] = vfmaq_n_f32(ym[m], wm, acts[m].xs[g]);
            }
        }
        for (int m = 0; m < M; m++) {
            float out[4];
            vst1q_f32(out, vaddq_f32(yv[m], ym[m]));
            for (int r = 0; r < 4; r++) {
                const int64_t n = q * 4 + r;
                if (n < L.N) y[m * ystride + n] = out[r] + (L.bias ? L.bias[n] : 0.f);
            }
        }
    }
}

TT_I8MM static void gemv_q8_neon_m(const Lin &L, const Act *acts, int M, float *y, int64_t ystride, int64_t q0, int64_t q1) {
    const int64_t G = L.G;
    for (int64_t q = q0; q < q1; q++) {
        const uint8_t *p = L.packed + q * G * 528;
        float32x4_t yv[MB], ym[MB];
        for (int m = 0; m < M; m++) yv[m] = ym[m] = vdupq_n_f32(0.f);
        for (int64_t g = 0; g < G; g++, p += 528) {
            const float32x4_t ws = vcvt_f32_f16(vld1_f16((const float16_t *)(p + 512)));
            const float32x4_t wm = vcvt_f32_f16(vld1_f16((const float16_t *)(p + 520)));
            for (int hh = 0; hh < 2; hh++) {
                const int64_t ga = g * 2 + hh;
                for (int i = 0; i < 4; i++) __builtin_prefetch(p + g_pf + hh * 256 + 64 * i);
                int8x16_t w[4][4];
                for (int r = 0; r < 4; r++)
                    for (int i = 0; i < 4; i++) w[r][i] = vld1q_s8((const int8_t *)p + r * 128 + hh * 64 + 16 * i);
                for (int m = 0; m < M; m++) {
                    const int8_t *hi = acts[m].hi.data() + ga * 64;
                    const uint8_t *lo = acts[m].lo.data() + ga * 64;
                    int32x4_t t[4];
                    for (int r = 0; r < 4; r++) {
                        int32x4_t ah = vdupq_n_s32(0), al = vdupq_n_s32(0);
                        for (int i = 0; i < 4; i++) {
                            ah = vdotq_s32(ah, w[r][i], vld1q_s8(hi + 16 * i));
                            al = vusdotq_s32(al, vld1q_u8(lo + 16 * i), w[r][i]);
                        }
                        t[r] = vaddq_s32(vshlq_n_s32(ah, 8), al);
                    }
                    const int32x4_t d = vpaddq_s32(vpaddq_s32(t[0], t[1]), vpaddq_s32(t[2], t[3]));
                    yv[m] = vfmaq_f32(yv[m], vcvtq_f32_s32(d), vmulq_n_f32(ws, acts[m].s[ga]));
                    ym[m] = vfmaq_n_f32(ym[m], wm, acts[m].xs[ga]);
                }
            }
        }
        for (int m = 0; m < M; m++) {
            float out[4];
            vst1q_f32(out, vaddq_f32(yv[m], ym[m]));
            for (int r = 0; r < 4; r++) {
                const int64_t n = q * 4 + r;
                if (n < L.N) y[m * ystride + n] = out[r] + (L.bias ? L.bias[n] : 0.f);
            }
        }
    }
}

// Matrix-multiply kernels (i8mm): one USMMLA takes 8 inputs of two weight rows against the high and the low
// activation bytes: [r.hi, r.lo, (r+1).hi, (r+1).lo]. Half the dot-product instructions of the SDOT/UDOT kernels.
static inline int32x4_t mm_reduce(int32x4_t a01, int32x4_t a23) {
    static const int32_t sh[4] = {8, 0, 8, 0};  // [hi, lo] pairs: 256 * hi + lo
    const int32x4_t shv = vld1q_s32(sh);
    return vpaddq_s32(vshlq_s32(a01, shv), vshlq_s32(a23, shv));
}

TT_I8MM static void gemv_q4_mm(const Lin &L, const Act *acts, int M, float *y, int64_t ystride, int64_t q0, int64_t q1) {
    const uint8x16_t m4 = vdupq_n_u8(0x0F);
    const int64_t G = L.G;
    for (int64_t q = q0; q < q1; q++) {
        const uint8_t *p = L.packed + q * G * 144;
        float32x4_t yv[MB], ym[MB];
        for (int m = 0; m < M; m++) yv[m] = ym[m] = vdupq_n_f32(0.f);
        for (int64_t g = 0; g < G; g++, p += 144) {
            if (g_pf) {
                __builtin_prefetch(p + g_pf);
                __builtin_prefetch(p + g_pf + 64);
                __builtin_prefetch(p + g_pf + 128);
            }
            uint8x16_t w01[8], w23[8];  // segments 0-3 (low nibbles), 4-7 (high nibbles)
            for (int c = 0; c < 4; c++) {
                const uint8x16_t x = vld1q_u8(p + c * 16), z = vld1q_u8(p + 64 + c * 16);
                w01[c] = vandq_u8(x, m4);
                w01[c + 4] = vshrq_n_u8(x, 4);
                w23[c] = vandq_u8(z, m4);
                w23[c + 4] = vshrq_n_u8(z, 4);
            }
            const float16x8_t sm = vld1q_f16((const float16_t *)(p + 128));  // scales[4], mins[4]
            const float32x4_t ws = vcvt_f32_f16(vget_low_f16(sm)), wm = vcvt_high_f32_f16(sm);
            for (int m = 0; m < M; m++) {
                const int8_t *b = acts[m].hl.data() + g * 128;
                int32x4_t a01 = vdupq_n_s32(0), a23 = vdupq_n_s32(0);
                for (int sgm = 0; sgm < 8; sgm++) {
                    const int8x16_t bv = vld1q_s8(b + 16 * sgm);
                    a01 = vusmmlaq_s32(a01, w01[sgm], bv);
                    a23 = vusmmlaq_s32(a23, w23[sgm], bv);
                }
                yv[m] = vfmaq_f32(yv[m], vcvtq_f32_s32(mm_reduce(a01, a23)), vmulq_n_f32(ws, acts[m].s[g]));
                ym[m] = vfmaq_n_f32(ym[m], wm, acts[m].xs[g]);
            }
        }
        for (int m = 0; m < M; m++) {
            float out[4];
            vst1q_f32(out, vaddq_f32(yv[m], ym[m]));
            for (int r = 0; r < 4; r++) {
                const int64_t n = q * 4 + r;
                if (n < L.N) y[m * ystride + n] = out[r] + (L.bias ? L.bias[n] : 0.f);
            }
        }
    }
}

TT_I8MM static void gemv_q8_mm(const Lin &L, const Act *acts, int M, float *y, int64_t ystride, int64_t q0, int64_t q1) {
    const int64_t G = L.G;
    for (int64_t q = q0; q < q1; q++) {
        const uint8_t *p = L.packed + q * G * 528;
        float32x4_t yv[MB], ym[MB];
        for (int m = 0; m < M; m++) yv[m] = ym[m] = vdupq_n_f32(0.f);
        for (int64_t g = 0; g < G; g++, p += 528) {
            const float32x4_t ws = vcvt_f32_f16(vld1_f16((const float16_t *)(p + 512)));
            const float32x4_t wm = vcvt_f32_f16(vld1_f16((const float16_t *)(p + 520)));
            for (int hh = 0; hh < 2; hh++) {
                const int64_t ga = g * 2 + hh;
                const int8_t *w01 = (const int8_t *)p + hh * 128, *w23 = (const int8_t *)p + 256 + hh * 128;
                for (int i = 0; i < 4; i++) __builtin_prefetch(p + g_pf + hh * 256 + 64 * i);
                for (int m = 0; m < M; m++) {
                    const int8_t *b = acts[m].hl.data() + ga * 128;
                    int32x4_t a01 = vdupq_n_s32(0), a23 = vdupq_n_s32(0);
                    for (int sgm = 0; sgm < 8; sgm++) {
                        const int8x16_t bv = vld1q_s8(b + 16 * sgm);
                        a01 = vmmlaq_s32(a01, vld1q_s8(w01 + 16 * sgm), bv);
                        a23 = vmmlaq_s32(a23, vld1q_s8(w23 + 16 * sgm), bv);
                    }
                    yv[m] = vfmaq_f32(yv[m], vcvtq_f32_s32(mm_reduce(a01, a23)), vmulq_n_f32(ws, acts[m].s[ga]));
                    ym[m] = vfmaq_n_f32(ym[m], wm, acts[m].xs[ga]);
                }
            }
        }
        for (int m = 0; m < M; m++) {
            float out[4];
            vst1q_f32(out, vaddq_f32(yv[m], ym[m]));
            for (int r = 0; r < 4; r++) {
                const int64_t n = q * 4 + r;
                if (n < L.N) y[m * ystride + n] = out[r] + (L.bias ? L.bias[n] : 0.f);
            }
        }
    }
}

// exp for x <= 0 (Cephes polynomial); inputs below -87 flush towards 0.
static inline float32x4_t vexp_neg(float32x4_t x) {
    x = vmaxq_f32(x, vdupq_n_f32(-87.f));
    const float32x4_t fx = vrndnq_f32(vmulq_n_f32(x, 1.44269504088896341f));
    x = vfmsq_n_f32(x, fx, 0.693359375f);
    x = vfmsq_n_f32(x, fx, -2.12194440e-4f);
    float32x4_t y = vdupq_n_f32(1.9875691500e-4f);
    y = vfmaq_f32(vdupq_n_f32(1.3981999507e-3f), y, x);
    y = vfmaq_f32(vdupq_n_f32(8.3334519073e-3f), y, x);
    y = vfmaq_f32(vdupq_n_f32(4.1665795894e-2f), y, x);
    y = vfmaq_f32(vdupq_n_f32(1.6666665459e-1f), y, x);
    y = vfmaq_f32(vdupq_n_f32(5.0000001201e-1f), y, x);
    y = vfmaq_f32(vaddq_f32(x, vdupq_n_f32(1.f)), y, vmulq_f32(x, x));
    const int32x4_t e = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(fx), vdupq_n_s32(127)), 23);
    return vmulq_f32(y, vreinterpretq_f32_s32(e));
}
#endif

// a[i] = silu(a[i]) * a[I + i]
static void silu_mul(float *a, int64_t i0, int64_t i1, int64_t I) {
#ifdef TT_KERN
    if (g_kf) { kern::silu_mul(a + i0, a + I + i0, i1 - i0); return; }
#endif
    int64_t i = i0;
#ifdef __aarch64__
    for (; i + 4 <= i1; i += 4) {
        const float32x4_t g = vld1q_f32(a + i);
        const float32x4_t e = vexp_neg(vnegq_f32(vabsq_f32(g)));  // exp(-|g|)
        const float32x4_t den = vaddq_f32(e, vdupq_n_f32(1.f));
        const float32x4_t sig = vdivq_f32(vbslq_f32(vcgezq_f32(g), vdupq_n_f32(1.f), e), den);
        vst1q_f32(a + i, vmulq_f32(vmulq_f32(g, sig), vld1q_f32(a + I + i)));
    }
#endif
    for (; i < i1; i++) a[i] = a[i] / (1.f + std::exp(-a[i])) * a[I + i];
}

static inline void rmsnorm(const float *x, const float *w, int64_t n, float eps, float *out) {
#ifdef TT_KERN
    if (g_kf) { kern::rms(x, w, n, eps, out); return; }
#endif
    float ss = 0.f;
#ifdef __aarch64__
    float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f);
    for (int64_t i = 0; i < n; i += 8) {  // n is a multiple of 8 here
        const float32x4_t v0 = vld1q_f32(x + i), v1 = vld1q_f32(x + i + 4);
        a0 = vfmaq_f32(a0, v0, v0);
        a1 = vfmaq_f32(a1, v1, v1);
    }
    ss = vaddvq_f32(vaddq_f32(a0, a1));
#else
    for (int64_t i = 0; i < n; i++) ss += x[i] * x[i];
#endif
    const float r = 1.f / std::sqrt(ss / (float)n + eps);
    for (int64_t i = 0; i < n; i++) out[i] = x[i] * r * w[i];
}

static inline void rope(float *v, const float *cs, const float *sn) {
#ifdef TT_KERN
    if (g_kf) { kern::rope(v, cs, sn, HD / 2); return; }
#endif
    for (int i = 0; i < HD / 2; i++) {
        const float a = v[i], b = v[i + HD / 2];
        v[i] = a * cs[i] - b * sn[i];
        v[i + HD / 2] = b * cs[i] + a * sn[i];
    }
}

// One KV head: rep query heads (q, rep * HD) against K, V rows 0..L-1 (fp16 on aarch64); out rep * HD.
static void attend_head(const float *q, const kv_t *K, const kv_t *V, int rep, int64_t L, float fs, float *out,
                        float *sbuf, float vskip = 0.f) {
    for (int r = 0; r < rep; r++) {
        const float *qr = q + r * HD;
        float *s = sbuf + r * (L + 8);
        float mx = -INFINITY;
#ifdef __aarch64__
        float32x4_t qv[HD / 4];
        for (int i = 0; i < HD / 4; i++) qv[i] = vld1q_f32(qr + 4 * i);
        for (int64_t j = 0; j < L; j++) {
            const float16_t *k = (const float16_t *)(K + j * HD);
            float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f), a2 = vdupq_n_f32(0.f), a3 = vdupq_n_f32(0.f);
            for (int i = 0; i < HD / 16; i++) {
                const float16x8_t k0 = vld1q_f16(k + 16 * i), k1 = vld1q_f16(k + 16 * i + 8);
                a0 = vfmaq_f32(a0, vcvt_f32_f16(vget_low_f16(k0)), qv[4 * i]);
                a1 = vfmaq_f32(a1, vcvt_high_f32_f16(k0), qv[4 * i + 1]);
                a2 = vfmaq_f32(a2, vcvt_f32_f16(vget_low_f16(k1)), qv[4 * i + 2]);
                a3 = vfmaq_f32(a3, vcvt_high_f32_f16(k1), qv[4 * i + 3]);
            }
            s[j] = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3))) * fs;
            mx = std::max(mx, s[j]);
        }
        float sum = 0.f;
        int64_t j = 0;
        float32x4_t vs = vdupq_n_f32(0.f);
        for (; j + 4 <= L; j += 4) {
            const float32x4_t e = vexp_neg(vsubq_f32(vld1q_f32(s + j), vdupq_n_f32(mx)));
            vst1q_f32(s + j, e);
            vs = vaddq_f32(vs, e);
        }
        sum = vaddvq_f32(vs);
        for (; j < L; j++) { s[j] = std::exp(s[j] - mx); sum += s[j]; }
        const float inv = 1.f / sum;
        float *o = out + r * HD;
        for (int part = 0; part < 4; part++) {  // 32 dims per pass: accumulators stay in registers
            float32x4_t acc[8];
            for (int i = 0; i < 8; i++) acc[i] = vdupq_n_f32(0.f);
            for (int64_t jj = 0; jj < L; jj++) {
                const float16_t *v = (const float16_t *)(V + jj * HD) + part * 32;
                const float32x4_t pw = vdupq_n_f32(s[jj]);
                for (int i = 0; i < 4; i++) {
                    const float16x8_t v8 = vld1q_f16(v + 8 * i);
                    acc[2 * i] = vfmaq_f32(acc[2 * i], pw, vcvt_f32_f16(vget_low_f16(v8)));
                    acc[2 * i + 1] = vfmaq_f32(acc[2 * i + 1], pw, vcvt_high_f32_f16(v8));
                }
            }
            for (int i = 0; i < 8; i++) vst1q_f32(o + part * 32 + 4 * i, vmulq_n_f32(acc[i], inv));
        }
#else
        for (int64_t j = 0; j < L; j++) {
            const kv_t *k = K + j * HD;
            float a = 0.f;
            for (int i = 0; i < HD; i++) a += (float)k[i] * qr[i];
            s[j] = a * fs;
            mx = std::max(mx, s[j]);
        }
        float sum = 0.f;
        for (int64_t j = 0; j < L; j++) { s[j] = std::exp(s[j] - mx); sum += s[j]; }
        float *o = out + r * HD;
        for (int i = 0; i < HD; i++) o[i] = 0.f;
        for (int64_t j = 0; j < L; j++) {
            if (s[j] < vskip) continue;
            const kv_t *v = V + j * HD;
            for (int i = 0; i < HD; i++) o[i] += s[j] * (float)v[i];
        }
        for (int i = 0; i < HD; i++) o[i] /= sum;
#endif
    }
}

#ifdef __aarch64__
static const int64_t BR = 128;  // rows per attention task (32 KiB of V: stays in L1 across the passes)
static int g_apf = 0;           // attention: prefetch K / V rows this many rows ahead (0: off)

static inline void prefetch_row(const kv_t *row) {
    const char *p = (const char *)row;
    __builtin_prefetch(p);
    __builtin_prefetch(p + 64);
    __builtin_prefetch(p + 128);
    __builtin_prefetch(p + 192);
}

// Raw scores of rows j0..j1-1 of one KV head for its rep (<= 2) query heads; each K row is read once.
static void attn_scores(const float *q, const kv_t *K, int rep, int64_t j0, int64_t j1, float fs, float *s,
                        int64_t stride) {
    for (int64_t j = j0; j < j1; j++) {
        const float16_t *k = (const float16_t *)(K + j * HD);
        if (g_apf) prefetch_row(K + (j + g_apf) * HD);
        float32x4_t a[2][4];
        for (int r = 0; r < 2; r++)
            for (int i = 0; i < 4; i++) a[r][i] = vdupq_n_f32(0.f);
        for (int i = 0; i < HD / 16; i++) {
            const float16x8_t k0 = vld1q_f16(k + 16 * i), k1 = vld1q_f16(k + 16 * i + 8);
            const float32x4_t f0 = vcvt_f32_f16(vget_low_f16(k0)), f1 = vcvt_high_f32_f16(k0);
            const float32x4_t f2 = vcvt_f32_f16(vget_low_f16(k1)), f3 = vcvt_high_f32_f16(k1);
            for (int r = 0; r < rep; r++) {
                const float *qr = q + r * HD + 16 * i;
                a[r][0] = vfmaq_f32(a[r][0], f0, vld1q_f32(qr));
                a[r][1] = vfmaq_f32(a[r][1], f1, vld1q_f32(qr + 4));
                a[r][2] = vfmaq_f32(a[r][2], f2, vld1q_f32(qr + 8));
                a[r][3] = vfmaq_f32(a[r][3], f3, vld1q_f32(qr + 12));
            }
        }
        for (int r = 0; r < rep; r++)
            s[r * stride + j] = vaddvq_f32(vaddq_f32(vaddq_f32(a[r][0], a[r][1]), vaddq_f32(a[r][2], a[r][3]))) * fs;
    }
}

// Scores with FMLAL: keys and query stay fp16, products accumulate exactly in fp32. q16 [rep][HD] is the query
// rounded to fp16 (the keys are fp16 already). One query head per pass; the block's keys stay in L1.
TT_FHM static void attn_scores_fml(const float16_t *q16, const kv_t *K, int rep, int64_t j0, int64_t j1, float fs, float *s,
                            int64_t stride) {
    for (int r = 0; r < rep; r++) {
        float16x8_t qv[HD / 8];
        for (int i = 0; i < HD / 8; i++) qv[i] = vld1q_f16(q16 + r * HD + 8 * i);
        float *sr = s + r * stride;
        for (int64_t j = j0; j < j1; j++) {
            const float16_t *k = (const float16_t *)(K + j * HD);
            float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f), a2 = vdupq_n_f32(0.f), a3 = vdupq_n_f32(0.f);
            for (int i = 0; i < HD / 8; i += 2) {
                const float16x8_t k0 = vld1q_f16(k + 8 * i), k1 = vld1q_f16(k + 8 * i + 8);
                a0 = vfmlalq_low_f16(a0, k0, qv[i]);
                a1 = vfmlalq_high_f16(a1, k0, qv[i]);
                a2 = vfmlalq_low_f16(a2, k1, qv[i + 1]);
                a3 = vfmlalq_high_f16(a3, k1, qv[i + 1]);
            }
            sr[j] = vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3))) * fs;
        }
    }
}

// part[r][HD] = sum over rows j0..j1-1 of p[r][j] * V[j] (p: softmax numerators, the largest is 1).
// Rows where every query head's numerator is below thr are not read (thr 0: all rows). Returns rows read.
static int64_t attn_values(const float *p, const kv_t *V, int rep, int64_t j0, int64_t j1, int64_t stride,
                           float *part, float thr) {
    int32_t idx[BR];
    int n = 0;
    for (int64_t j = j0; j < j1; j++)
        if (p[j] >= thr || (rep > 1 && p[stride + j] >= thr)) idx[n++] = (int32_t)j;
    for (int pt = 0; pt < 4; pt++) {  // 32 dims per pass: accumulators of both query heads stay in registers
        float32x4_t acc[2][8];
        for (int r = 0; r < 2; r++)
            for (int i = 0; i < 8; i++) acc[r][i] = vdupq_n_f32(0.f);
        for (int k = 0; k < n; k++) {
            const int64_t j = idx[k];
            if (g_apf && pt == 0) prefetch_row(V + (j + g_apf) * HD);
            const float16_t *v = (const float16_t *)(V + j * HD) + pt * 32;
            float32x4_t f[8];
            for (int i = 0; i < 4; i++) {
                const float16x8_t v8 = vld1q_f16(v + 8 * i);
                f[2 * i] = vcvt_f32_f16(vget_low_f16(v8));
                f[2 * i + 1] = vcvt_high_f32_f16(v8);
            }
            for (int r = 0; r < rep; r++) {
                const float32x4_t pw = vdupq_n_f32(p[r * stride + j]);
                for (int i = 0; i < 8; i++) acc[r][i] = vfmaq_f32(acc[r][i], pw, f[i]);
            }
        }
        for (int r = 0; r < rep; r++)
            for (int i = 0; i < 8; i++) vst1q_f32(part + r * HD + pt * 32 + 4 * i, acc[r][i]);
    }
    return n;
}

// Coarse scores of rows j0..j1-1 from the int8 keys (per-row scale ks) and int8 queries q8 [rep][HD]:
// s[r][j] = qf[r] * ks[j] * dot. The exact score lies within cb[r] * ks[j] of it; lb/ub [rep] get the block's
// largest lower and upper bound.
TT_DOT static void attn_coarse(const int8_t *q8, const float *qf, const float *cb, const int8_t *K8, const float *ks, int rep,
                        int64_t j0, int64_t j1, float *s, int64_t stride, float *lb, float *ub) {
    int8x16_t qv[2][8];
    for (int r = 0; r < 2; r++)
        for (int i = 0; i < 8; i++) qv[r][i] = vld1q_s8(q8 + (r < rep ? r : 0) * HD + 16 * i);
    float lbm[2] = {-INFINITY, -INFINITY}, ubm[2] = {-INFINITY, -INFINITY};
    for (int64_t j = j0; j < j1; j++) {
        const int8_t *k = K8 + j * HD;
        int32x4_t a0 = vdupq_n_s32(0), a1 = vdupq_n_s32(0);
        for (int i = 0; i < 8; i++) {
            const int8x16_t kv = vld1q_s8(k + 16 * i);
            a0 = vdotq_s32(a0, kv, qv[0][i]);
            a1 = vdotq_s32(a1, kv, qv[1][i]);
        }
        const float kj = ks[j];
        const int32_t d[2] = {vaddvq_s32(a0), vaddvq_s32(a1)};
        for (int r = 0; r < rep; r++) {
            const float sc = (float)d[r] * (qf[r] * kj), mg = cb[r] * kj;
            s[r * stride + j] = sc;
            lbm[r] = std::max(lbm[r], sc - mg);
            ubm[r] = std::max(ubm[r], sc + mg);
        }
    }
    for (int r = 0; r < rep; r++) { lb[r] = lbm[r]; ub[r] = ubm[r]; }
}

// Exact pass over the rows of j0..j1-1 whose score bound reaches thr[r] for some query head: scores from the fp16
// keys, e = exp(score - m0[r]) (m0: an upper bound of the largest score), part[r] += e * V[j], sum[r] += e.
// Returns the rows read.
static int64_t attn_fine(const float *q, const kv_t *K, const kv_t *V, const float *ks, const float *cs,
                         const float *cb, const float *thr, const float *m0, int rep, int64_t j0, int64_t j1,
                         int64_t stride, float fs, float *part, float *sum) {
    int32_t idx[BR];
    float e[2][BR + 4];
    int n = 0;
    for (int64_t j = j0; j < j1; j++) {
        const float kj = ks[j];
        if (cs[j] + cb[0] * kj >= thr[0] || (rep > 1 && cs[stride + j] + cb[1] * kj >= thr[1])) idx[n++] = (int32_t)j;
    }
    for (int k = 0; k < n; k++) {
        const float16_t *kr = (const float16_t *)(K + (int64_t)idx[k] * HD);
        float32x4_t a[2][4];
        for (int r = 0; r < 2; r++)
            for (int i = 0; i < 4; i++) a[r][i] = vdupq_n_f32(0.f);
        for (int i = 0; i < HD / 16; i++) {
            const float16x8_t k0 = vld1q_f16(kr + 16 * i), k1 = vld1q_f16(kr + 16 * i + 8);
            const float32x4_t f0 = vcvt_f32_f16(vget_low_f16(k0)), f1 = vcvt_high_f32_f16(k0);
            const float32x4_t f2 = vcvt_f32_f16(vget_low_f16(k1)), f3 = vcvt_high_f32_f16(k1);
            for (int r = 0; r < rep; r++) {
                const float *qr = q + r * HD + 16 * i;
                a[r][0] = vfmaq_f32(a[r][0], f0, vld1q_f32(qr));
                a[r][1] = vfmaq_f32(a[r][1], f1, vld1q_f32(qr + 4));
                a[r][2] = vfmaq_f32(a[r][2], f2, vld1q_f32(qr + 8));
                a[r][3] = vfmaq_f32(a[r][3], f3, vld1q_f32(qr + 12));
            }
        }
        for (int r = 0; r < rep; r++)
            e[r][k] = vaddvq_f32(vaddq_f32(vaddq_f32(a[r][0], a[r][1]), vaddq_f32(a[r][2], a[r][3]))) * fs - m0[r];
    }
    for (int r = 0; r < rep; r++) {
        float32x4_t vs = vdupq_n_f32(0.f);
        for (int k = n; k < ((n + 3) & ~3); k++) e[r][k] = -200.f;  // padding: exp -> 0
        for (int k = 0; k < n; k += 4) {
            const float32x4_t x = vexp_neg(vminq_f32(vld1q_f32(e[r] + k), vdupq_n_f32(0.f)));
            vst1q_f32(e[r] + k, x);
            vs = vaddq_f32(vs, x);
        }
        sum[r] = vaddvq_f32(vs);
    }
    for (int pt = 0; pt < 4; pt++) {
        float32x4_t acc[2][8];
        for (int r = 0; r < 2; r++)
            for (int i = 0; i < 8; i++) acc[r][i] = vdupq_n_f32(0.f);
        for (int k = 0; k < n; k++) {
            const float16_t *v = (const float16_t *)(V + (int64_t)idx[k] * HD) + pt * 32;
            float32x4_t f[8];
            for (int i = 0; i < 4; i++) {
                const float16x8_t v8 = vld1q_f16(v + 8 * i);
                f[2 * i] = vcvt_f32_f16(vget_low_f16(v8));
                f[2 * i + 1] = vcvt_high_f32_f16(v8);
            }
            for (int r = 0; r < rep; r++) {
                const float32x4_t pw = vdupq_n_f32(e[r][k]);
                for (int i = 0; i < 8; i++) acc[r][i] = vfmaq_f32(acc[r][i], pw, f[i]);
            }
        }
        for (int r = 0; r < rep; r++)
            for (int i = 0; i < 8; i++) vst1q_f32(part + r * HD + pt * 32 + 4 * i, acc[r][i]);
    }
    return n;
}

// Transposed int8 value cache of one KV head: [column group 8][segment S/8][column pair 8][2 columns][8 rows],
// values v8 = round(v / vs[row]). One SMMLA takes 8 rows of two columns against the high and low bytes of the
// 16-bit row weights.

// part[r][HD] = sum over rows j0..j1-1 (j0 a multiple of 8) of e[r][j] * v[j], from the int8 values:
// weights w = e * vs are quantized to 16 bits per block (step = block maximum / 32639).
TT_I8MM static void attn_values8(const float *e, const int8_t *V8, const float *vs, int rep, int64_t j0, int64_t j1,
                         int64_t stride, int64_t S, float *part) {
    const int n = (int)(j1 - j0), nseg = (n + 7) / 8;
    alignas(16) int8_t wb[2][(BR / 8) * 16];
    float delta[2] = {0.f, 0.f};
    memset(wb, 0, sizeof(wb));
    for (int r = 0; r < rep; r++) {
        float w[BR + 8];
        float wmax = 0.f;
        int k = 0;
        float32x4_t vm = vdupq_n_f32(0.f);
        for (; k + 4 <= n; k += 4) {
            const float32x4_t wv = vmulq_f32(vld1q_f32(e + r * stride + j0 + k), vld1q_f32(vs + j0 + k));
            vst1q_f32(w + k, wv);
            vm = vmaxq_f32(vm, wv);
        }
        wmax = vmaxvq_f32(vm);
        for (; k < n; k++) { w[k] = e[r * stride + j0 + k] * vs[j0 + k]; wmax = std::max(wmax, w[k]); }
        for (k = n; k < nseg * 8; k++) w[k] = 0.f;
        const float inv = wmax > 0.f ? 32639.f / wmax : 0.f;
        delta[r] = wmax / 32639.f;
        for (int k = 0; k < nseg * 8; k += 8) {
            const int32x4_t c0 = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(w + k), inv));
            const int32x4_t c1 = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(w + k + 4), inv));
            const int16x8_t c = vcombine_s16(vqmovn_s32(c0), vqmovn_s32(c1));
            vst1_s8(wb[r] + (k / 8) * 16, vrshrn_n_s16(c, 8));
            vst1_s8(wb[r] + (k / 8) * 16 + 8, vmovn_s16(c));
        }
    }
    for (int pg = 0; pg < 8; pg++) {
        int32x4_t acc[2][8];
        for (int r = 0; r < 2; r++)
            for (int i = 0; i < 8; i++) acc[r][i] = vdupq_n_s32(0);
        const int8_t *vp = V8 + ((size_t)pg * (S / 8) + j0 / 8) * 128;
        for (int sg = 0; sg < nseg; sg++, vp += 128) {
            const int8x16_t b0 = vld1q_s8(wb[0] + sg * 16), b1 = vld1q_s8(wb[1] + sg * 16);
            for (int pr = 0; pr < 8; pr++) {
                const int8x16_t a = vld1q_s8(vp + pr * 16);
                acc[0][pr] = vmmlaq_s32(acc[0][pr], a, b0);
                acc[1][pr] = vmmlaq_s32(acc[1][pr], a, b1);
            }
        }
        for (int r = 0; r < rep; r++)
            for (int pr = 0; pr < 8; pr += 2)
                vst1q_f32(part + r * HD + pg * 16 + pr * 2,
                          vmulq_n_f32(vcvtq_f32_s32(mm_reduce(acc[r][pr], acc[r][pr + 1])), delta[r]));
    }
}

// In place: s -> exp(s - max); returns the sum.
static float softmax_num(float *s, int64_t L) {
    float mx = -INFINITY;
    int64_t i = 0;
    float32x4_t vm = vdupq_n_f32(-INFINITY);
    for (; i + 4 <= L; i += 4) vm = vmaxq_f32(vm, vld1q_f32(s + i));
    mx = vmaxvq_f32(vm);
    for (; i < L; i++) mx = std::max(mx, s[i]);
    float32x4_t vs = vdupq_n_f32(0.f);
    int64_t j = 0;
    for (; j + 4 <= L; j += 4) {
        const float32x4_t e = vexp_neg(vsubq_f32(vld1q_f32(s + j), vdupq_n_f32(mx)));
        vst1q_f32(s + j, e);
        vs = vaddq_f32(vs, e);
    }
    float sum = vaddvq_f32(vs);
    for (; j < L; j++) { s[j] = std::exp(s[j] - mx); sum += s[j]; }
    return sum;
}
#endif

#ifdef __EMSCRIPTEN__
#include "web_linear.h"
#endif

#ifdef TT_KERN
// kern::Par over the pool
static void pool_run(void *p, int64_t n, int64_t grain, kern::RangeFn fn, void *ctx) {
    ((Pool *)p)->run(n, grain, [&](int64_t a, int64_t b) { fn(ctx, a, b); });
}
static kern::Par pool_par(Pool *p) {
    kern::Par r;
    r.run = pool_run;
    r.self = p;
    r.threads = p->n;
    return r;
}
#endif
#ifdef TT_X86
static inline void kv_store_row(kv_t *d, const float *x, int n) {
    if (g_kf) kern::store_f16((uint16_t *)d, x, n);
    else for (int i = 0; i < n; i++) d[i] = kv_t(x[i]);
}
#endif

// ---------------------------------------------------------------- model
struct Options;
static void native_kernels(Options &opt);
struct Options {
    int act_bits = 16;
    bool neon = true, mm = false;
    bool usdot = true;    // aarch64: q8 layers through USDOT (i8mm); without it they run the reference kernel
    int64_t chunk = 16;
    int kv_sim8 = 0;
    float vskip = 0.f;  // V rows with softmax numerator below this are skipped
    bool v8 = false;      // attention values from the transposed int8 cache (SMMLA)
    bool fml = false;     // attention scores with FMLAL (query rounded to fp16)
    bool act_stats = false;
    int64_t head2 = 0;        // two-stage head: exact logits only for this many best rows of a coarse pass (0: off)
    bool head2_check = false, head2_build = false, head2_mask = false;
    bool keep_v16 = false;    // also write fp16 value rows when the int8 value cache is on (path sweeps)
    float mlp_drop = 0.f;
    bool coarse = false;  // attention: int8 key pass first, exact rows only where the bound allows weight >= vskip
    int bf16 = 0;  // x86: q4 / q8 linears in bf16 on AMX / AVX512-BF16 where present (changes the numbers): 1 all,
                   // 2 only steps of more than MB rows (text reading; frames stay int16)
};

// The kernels this CPU can run in a native build (aarch64: NEON with the dot product, q8 through USDOT with i8mm,
// the matrix-multiply kernels off); otherwise the portable ones.
static void native_kernels(Options &opt) {
    opt.neon = opt.mm = opt.usdot = false;
#ifdef TT_ARM
    opt.neon = kern::has_dotprod();
    opt.usdot = kern::isa().i8mm;
#endif
}

struct Model {
    int64_t H = 0, NL = 0, nq = 0, nkv = 0, I = 0, n_local = 0, S = 0, W = 0;
    float eps = 1e-6f, theta = 1e6f, invf[HD / 2];  // rotary frequencies
    struct Layer { const float *in_norm, *post_norm, *q_norm, *k_norm; Lin qkv, o, gate_up, down; };
    std::vector<Layer> layers;
    const float *norm = nullptr;
    Lin fc1, fc2, fc2c;  // fc2c: coarse 4-bit copy of fc2 for the two-stage head
    std::vector<int32_t> cand, qlist;
    std::vector<float> zc;
    int64_t h2_miss = 0, h2_steps = 0, h2_rows = 0;
    kv_t *kv = nullptr;  // [layer][k|v][kv head][S][HD]
    int8_t *k8 = nullptr;   // coarse attention: int8 keys [layer][kv head][S][HD] ...
    float *k8s = nullptr;   // ... and their row scales [layer][kv head][S]
    std::vector<float> cbuf;  // per attention task: lower/upper bounds and sums
    int8_t *v8 = nullptr;   // int8 values, transposed per head (v8_index) [layer][kv head][S * HD] ...
    float *v8s = nullptr;   // ... and their row scales [layer][kv head][S]
    std::vector<float> X, bh, bq, ba, by, sbuf, pbuf;  // hidden states [MB][H]; work rows [MB][W]
    const float *hid = nullptr;  // hidden state of the last token of the last step
    Act acts[MBP];
    Pool *pool = nullptr;
    Options opt;
#ifdef TT_KERN
    kern::QBuf qbuf;
#endif
#ifdef TT_X86
    kern::BF16Buf bbuf;
#endif
    double t_gemv = 0, t_attn = 0, t_head = 0, t_quant = 0, t_a_scores = 0, t_a_soft = 0, t_a_vals = 0;
    double st_gate[5] = {0, 0, 0, 0, 0}, st_prod[5] = {0, 0, 0, 0, 0}, st_n = 0;  // activation statistics
    double st_drop = 0, st_dn = 0;
    std::atomic<int64_t> kept{0};  // V rows read
    int64_t v_rows = 0;            // V rows in range

    void load(const File &f, int64_t max_len) {
        const int32_t *c = (const int32_t *)f.get("cfg").p;
        H = c[0]; NL = c[1]; nq = c[2]; nkv = c[3]; I = c[5]; n_local = c[6];
        if (c[4] != HD) { fprintf(stderr, "head_dim must be %d\n", HD); exit(2); }
        const float *cf = (const float *)f.get("cfg_f").p;
        eps = cf[0];
        theta = cf[1];
        for (int i = 0; i < HD / 2; i++) invf[i] = 1.f / std::pow(theta, (float)(2 * i) / (float)HD);
        layers.resize(NL);
        for (int64_t l = 0; l < NL; l++) {
            const std::string p = "L" + std::to_string(l) + ".";
            Layer &L = layers[l];
            L.in_norm = (const float *)f.get(p + "in_norm").p;
            L.post_norm = (const float *)f.get(p + "post_norm").p;
            L.q_norm = (const float *)f.get(p + "q_norm").p;
            L.k_norm = (const float *)f.get(p + "k_norm").p;
            L.qkv.load(f, p + "qkv");
            L.o.load(f, p + "o");
            L.gate_up.load(f, p + "gate_up");
            L.down.load(f, p + "down");
        }
        norm = (const float *)f.get("norm").p;
        fc1.load(f, "fc1");
        fc2.load(f, "fc2");
        if (opt.head2 > 0 || opt.head2_build) fc2c.coarse_of(fc2);
        S = (max_len + 7) / 8 * 8;
        kv = (kv_t *)big_alloc((size_t)NL * 2 * nkv * S * HD * sizeof(kv_t));
        if (opt.coarse) {
        k8 = (int8_t *)big_alloc((size_t)NL * nkv * S * HD);
        k8s = (float *)big_alloc((size_t)NL * nkv * S * sizeof(float));
        }
        cbuf.resize(nkv * ((S + 127) / 128) * 8);
        if (opt.v8) {
        v8 = (int8_t *)big_alloc((size_t)NL * nkv * S * HD);
        v8s = (float *)big_alloc((size_t)NL * nkv * S * sizeof(float));
        }
        W = std::max<int64_t>({H, (nq + 2 * nkv) * HD, 2 * I, fc1.N, (fc2.N + 3) / 4 * 4});
        X.resize(MBP * H);
        for (auto *v : {&bh, &bq, &ba, &by}) v->resize(MBP * W);
        sbuf.resize(nkv * (nq / nkv) * (S + 8));
        pbuf.resize(nkv * ((S + 127) / 128) * (nq / nkv) * HD);
        for (auto &a : acts) a.resize(std::max<int64_t>({H, I, nq * HD, fc1.N}));
    }
    size_t pack_all() {
        size_t b = 0;
        for (auto &L : layers)
            for (Lin *x : {&L.qkv, &L.o, &L.gate_up, &L.down}) {
                if (opt.mm) x->pack_mm(); else x->pack();
                b += x->packed_bytes();
            }
        if (opt.mm) { fc1.pack_mm(); fc2.pack_mm(); } else { fc1.pack(); fc2.pack(); }
        if (opt.head2 > 0 || opt.head2_build) { if (opt.mm) fc2c.pack_mm(); else fc2c.pack(); }
        return b + fc1.packed_bytes() + fc2.packed_bytes();
    }

    // y[m] = W x[m] for M rows (rows W floats apart).
    void linear(const Lin &L, const float *x, float *y, int M = 1) {
        double t0 = now_s();
        if (L.kind == 2) {
#ifdef TT_KERN
            if (g_kf) {
                pool->run(L.N, 32, [&](int64_t a, int64_t b) { kern::linear_rows(L.wf, L.bias, L.K, x, W, M, y, W, a, b); });
                t_gemv += now_s() - t0;
                return;
            }
#endif
            pool->run(L.N, 32, [&](int64_t a, int64_t b) {
                for (int64_t n = a; n < b; n++) {
                    const float *w = L.wf + n * L.K;
                    for (int m = 0; m < M; m++) {
                        const float *xm = x + m * W;
                        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                        int64_t k = 0;
                        for (; k + 8 <= L.K; k += 8)
                            for (int j = 0; j < 8; j++) acc[j] += w[k + j] * xm[k + j];
                        float t = ((acc[0] + acc[4]) + (acc[2] + acc[6])) + ((acc[1] + acc[5]) + (acc[3] + acc[7]));
                        for (; k < L.K; k++) t += w[k] * xm[k];
                        y[m * W + n] = t + (L.bias ? L.bias[n] : 0.f);
                    }
                }
            });
            t_gemv += now_s() - t0;
            return;
        }
#ifdef __EMSCRIPTEN__
        if (opt.act_bits == 16 && web_linear(L, x, W, y, W, M, acts, pool)) { t_gemv += now_s() - t0; return; }
#endif
#ifdef TT_X86
        const kern::LinBF16Impl *bi = (opt.bf16 == 1 || (opt.bf16 == 2 && M > MB)) && g_kf ? kern::lin_bf16_impl() : nullptr;
        if (bi) {
            kern::QW w;
            w.kind = L.kind; w.N = L.N; w.K = L.K; w.codes = L.codes; w.scales = L.scales; w.mins = L.mins; w.bias = L.bias;
            kern::lin_bf16_f32(pool_par(pool), w, x, W, M, y, W, bbuf, *bi);
            t_gemv += now_s() - t0;
            return;
        }
#endif
#ifdef TT_KERN
        if (g_kf && opt.act_bits == 16 && !opt.mm) {  // any M (prefill batches up to MBP)
            const kern::Par par = pool_par(pool);
            const int64_t K = L.K, G = K / AG;
            if ((int64_t)qbuf.q.size() < M * K) qbuf.q.resize((size_t)(M * K));
            if ((int64_t)qbuf.s.size() < M * G) { qbuf.s.resize((size_t)(M * G)); qbuf.xs.resize((size_t)(M * G)); }
            kern::quant16(par, x, W, M, K, qbuf.q.data(), qbuf.s.data(), qbuf.xs.data());
            const double t1 = now_s();
            t_quant += t1 - t0;
            kern::QW w;
            w.kind = L.kind; w.N = L.N; w.K = K; w.codes = L.codes; w.scales = L.scales; w.mins = L.mins; w.bias = L.bias;
            w.packed = L.packed;  // the quad layout of pack(): aarch64 reads it for single rows
            kern::lin_q(par, w, kern::QX{qbuf.q.data(), qbuf.s.data(), qbuf.xs.data(), M, K}, y, W);
            t_gemv += now_s() - t1;
            return;
        }
#endif
        if (M > MB) {  // the kernels below take up to MB rows
            for (int m0 = 0; m0 < M; m0 += MB) linear(L, x + (int64_t)m0 * W, y + (int64_t)m0 * W, std::min(MB, M - m0));
            return;
        }
        for (int m = 0; m < M; m++) quantize(x + m * W, L.K, acts[m], opt.act_bits, opt.mm);
        double t1 = now_s();
        t_quant += t1 - t0;
#ifdef __aarch64__
        if (opt.neon && (L.kind == 0 || opt.usdot || opt.mm)) {
            const int bits = opt.act_bits;
            if (opt.mm) {
                pool->run(L.NQ, std::max<int64_t>(1, opt.chunk / (M > 1 ? 4 : 1)), [&](int64_t a, int64_t b) {
                    if (L.kind == 0) gemv_q4_mm(L, acts, M, y, W, a, b);
                    else gemv_q8_mm(L, acts, M, y, W, a, b);
                }, true);
            } else if (M == 1) {
                pool->run(L.NQ, opt.chunk, [&](int64_t a, int64_t b) {
                    if (L.kind == 0) gemv_q4_neon(L, acts[0], y, a, b, bits);
                    else gemv_q8_neon(L, acts[0], y, a, b, bits);
                });
            } else {
                pool->run(L.NQ, std::max<int64_t>(1, opt.chunk / 4), [&](int64_t a, int64_t b) {
                    if (L.kind == 0) gemv_q4_neon_m(L, acts, M, y, W, a, b);
                    else gemv_q8_neon_m(L, acts, M, y, W, a, b);
                });
            }
            t_gemv += now_s() - t1;
            return;
        }
#endif
        for (int m = 0; m < M; m++)
            pool->run(L.N, opt.chunk * 4, [&](int64_t a, int64_t b) { gemv_scalar(L, acts[m], y + m * W, a, b); });
        t_gemv += now_s() - t1;
    }

#ifdef TT_KERN
    // kern::attn_rows (exact softmax over every row; not with the aarch64 research options)
    bool kern_attn() const { return g_kf && opt.vskip == 0.f && !opt.coarse && !opt.v8 && !opt.fml; }
#endif
    // One query token (q heads at q) over cache rows 0..t of one layer; out [nq * HD].
    void attention(const float *q, const kv_t *K, const kv_t *V, int64_t t, float *a, int64_t layer = 0,
                   int64_t r0 = 0) {
        const float scale = 1.f / std::sqrt((float)HD);
        const int rep = (int)(nq / nkv);
#ifdef TT_KERN
        if (kern_attn()) {
            const int64_t a0 = r0, a1 = t + 1;
            kern::attn_rows<kern::KV_F16>(pool_par(pool), 1, q, 0, (const uint16_t *)K, (const uint16_t *)V, S * HD, S * HD,
                                          nq, nkv, &a0, &a1, scale, 128, a, 0);
            return;
        }
#endif
#ifdef __aarch64__
        // rows r0..t; whole blocks from b0 * BR are scored, the rows before r0 get weight 0
        const int64_t b0 = r0 / BR, Ln = t + 1, nb = (Ln + BR - 1) / BR - b0, st = S + 8, nt = nkv * nb;
        float *sc = sbuf.data(), *part = pbuf.data();
        const double tp0 = now_s();
        if (r0 == 0 && opt.coarse && opt.vskip > 0.f && Ln >= 192) {
            // 1) coarse scores of all rows from the int8 keys; 2) exact scores and values of the rows that can matter
            int8_t q8[64 * HD];
            float qf[64], cb[64], thr[64], m0[64];
            for (int64_t hq = 0; hq < nq; hq++) {
                const float *qr = q + hq * HD;
                float mx = 0.f, l1 = 0.f;
                for (int d = 0; d < HD; d++) { mx = std::max(mx, std::fabs(qr[d])); l1 += std::fabs(qr[d]); }
                const float inv = mx > 0.f ? 127.f / mx : 0.f, qs = mx / 127.f;
                for (int d = 0; d < HD; d++) q8[hq * HD + d] = (int8_t)std::lrintf(qr[d] * inv);
                qf[hq] = scale * qs;
                // |exact - coarse| <= scale * (|dk| . |q| + |k8 ks| . |dq|) <= scale * ks * (l1 / 2 + 64 * mx)
                cb[hq] = scale * (0.5f * l1 + 64.f * mx);
            }
            const int8_t *K8 = k8 + layer * nkv * S * HD;
            const float *KS = k8s + layer * nkv * S;
            float *bnd = cbuf.data();  // per task: lb[2], ub[2], sum[2]
            pool->run(nt, 1, [&](int64_t a0, int64_t a1) {
                for (int64_t k = a0; k < a1; k++) {
                    const int64_t hh = k / nb, j0 = (k % nb) * BR;
                    attn_coarse(q8 + hh * rep * HD, qf + hh * rep, cb + hh * rep, K8 + hh * S * HD, KS + hh * S, rep, j0,
                                std::min(Ln, j0 + BR), sc + hh * rep * st, st, bnd + k * 8, bnd + k * 8 + 2);
                }
            });
            const float T = -std::log(opt.vskip);
            for (int64_t hq = 0; hq < nq; hq++) {
                const int64_t hh = hq / rep, r = hq % rep;
                float lb = -INFINITY, ub = -INFINITY;
                for (int64_t b = 0; b < nb; b++) {
                    lb = std::max(lb, bnd[(hh * nb + b) * 8 + r]);
                    ub = std::max(ub, bnd[(hh * nb + b) * 8 + 2 + r]);
                }
                thr[hq] = lb - T;  // rows whose upper bound is below this have weight < vskip
                m0[hq] = ub;
            }
            pool->run(nt, 1, [&](int64_t a0, int64_t a1) {
                for (int64_t k = a0; k < a1; k++) {
                    const int64_t hh = k / nb, j0 = (k % nb) * BR;
                    kept.fetch_add(attn_fine(q + hh * rep * HD, K + hh * S * HD, V + hh * S * HD, KS + hh * S,
                                             sc + hh * rep * st, cb + hh * rep, thr + hh * rep, m0 + hh * rep, rep, j0,
                                             std::min(Ln, j0 + BR), st, scale, part + k * rep * HD, bnd + k * 8 + 4),
                                   std::memory_order_relaxed);
                }
            });
            v_rows += nkv * Ln;
            for (int64_t hq = 0; hq < nq; hq++) {
                const int64_t hh = hq / rep, r = hq % rep;
                float32x4_t o[HD / 4];
                for (int i = 0; i < HD / 4; i++) o[i] = vdupq_n_f32(0.f);
                float sum = 0.f;
                for (int64_t b = 0; b < nb; b++) {
                    const float *src = part + ((hh * nb + b) * rep + r) * HD;
                    for (int i = 0; i < HD / 4; i++) o[i] = vaddq_f32(o[i], vld1q_f32(src + 4 * i));
                    sum += bnd[(hh * nb + b) * 8 + 4 + r];
                }
                for (int i = 0; i < HD / 4; i++) vst1q_f32(a + hq * HD + 4 * i, vmulq_n_f32(o[i], 1.f / sum));
            }
            return;
        }
        float16_t q16[64 * HD];
        if (opt.fml)
            for (int64_t d = 0; d < nq * HD; d++) q16[d] = (float16_t)q[d];
        pool->run(nt, Ln < 96 ? nt : 1, [&](int64_t a0, int64_t a1) {
            for (int64_t k = a0; k < a1; k++) {
                const int64_t hh = k / nb, j0 = (b0 + k % nb) * BR;
                if (opt.fml)
                    attn_scores_fml(q16 + hh * rep * HD, K + hh * S * HD, rep, j0, std::min(Ln, j0 + BR), scale,
                                    sc + hh * rep * st, st);
                else
                    attn_scores(q + hh * rep * HD, K + hh * S * HD, rep, j0, std::min(Ln, j0 + BR), scale,
                                sc + hh * rep * st, st);
            }
        });
        const double tp1 = now_s();
        float inv[64];
        pool->run(nq, Ln < 256 ? nq : 1, [&](int64_t h0, int64_t h1) {
            for (int64_t hq = h0; hq < h1; hq++) {
                inv[hq] = 1.f / softmax_num(sc + hq * st + r0, Ln - r0);
                for (int64_t j = b0 * BR; j < r0; j++) sc[hq * st + j] = 0.f;
            }
        });
        const double tp2 = now_s();
        t_a_scores += tp1 - tp0;
        t_a_soft += tp2 - tp1;
        if (opt.v8) {
            const int8_t *V8 = v8 + (size_t)layer * nkv * S * HD;
            const float *VS = v8s + layer * nkv * S;
            pool->run(nt, Ln < 96 ? nt : 1, [&](int64_t a0, int64_t a1) {
                for (int64_t k = a0; k < a1; k++) {
                    const int64_t hh = k / nb, j0 = (b0 + k % nb) * BR;
                    attn_values8(sc + hh * rep * st, V8 + (size_t)hh * S * HD, VS + hh * S, rep, j0,
                                 std::min(Ln, j0 + BR), st, S, part + k * rep * HD);
                }
            });
        } else
        pool->run(nt, Ln < 96 ? nt : 1, [&](int64_t a0, int64_t a1) {
            for (int64_t k = a0; k < a1; k++) {
                const int64_t hh = k / nb, j0 = (b0 + k % nb) * BR;
                kept.fetch_add(attn_values(sc + hh * rep * st, V + hh * S * HD, rep, j0, std::min(Ln, j0 + BR), st,
                                           part + k * rep * HD, opt.vskip), std::memory_order_relaxed);
            }
        });
        v_rows += nkv * Ln;
        t_a_vals += now_s() - tp2;
        for (int64_t hq = 0; hq < nq; hq++) {
            const int64_t hh = hq / rep, r = hq % rep;
            float32x4_t o[HD / 4];
            for (int i = 0; i < HD / 4; i++) o[i] = vdupq_n_f32(0.f);
            for (int64_t b = 0; b < nb; b++) {
                const float *src = part + ((hh * nb + b) * rep + r) * HD;
                for (int i = 0; i < HD / 4; i++) o[i] = vaddq_f32(o[i], vld1q_f32(src + 4 * i));
            }
            for (int i = 0; i < HD / 4; i++) vst1q_f32(a + hq * HD + 4 * i, vmulq_n_f32(o[i], inv[hq]));
        }
#else
#ifdef __EMSCRIPTEN__
        web_attention(q, K, V, S, nq, nkv, t + 1 - r0, r0, scale, a, sbuf.data(), pool);
        return;
#endif
        pool->run(nkv, t < 96 ? nkv : 1, [&](int64_t h0, int64_t h1) {
            for (int64_t hh = h0; hh < h1; hh++)
                attend_head(q + hh * rep * HD, K + (hh * S + r0) * HD, V + (hh * S + r0) * HD, rep, t + 1 - r0, scale,
                            a + hh * rep * HD, sbuf.data() + hh * rep * (S + 8), opt.vskip);
        });
#endif
    }

    // Rows ms..M-1 of a step (queries and outputs W floats apart), row m over cache rows ar0[m]..ar1[m]-1 (default
    // 0..t0+m): on x86 all rows in one parallel region.
    void attention_rows(const float *q, const kv_t *K, const kv_t *V, int ms, int M, int64_t t0, const int64_t *ar0,
                        const int64_t *ar1, float *a, int64_t layer) {
#ifdef TT_KERN
        if (kern_attn()) {
            int64_t r0[MBP], r1[MBP];
            for (int m = ms; m < M; m++) { r0[m - ms] = ar0 ? ar0[m] : 0; r1[m - ms] = ar1 ? ar1[m] : t0 + m + 1; }
            kern::attn_rows<kern::KV_F16>(pool_par(pool), M - ms, q + ms * W, W, (const uint16_t *)K, (const uint16_t *)V,
                                          S * HD, S * HD, nq, nkv, r0, r1, 1.f / std::sqrt((float)HD), 128, a + ms * W, W);
            return;
        }
#endif
        for (int m = ms; m < M; m++) attention(q + m * W, K, V, ar1 ? ar1[m] - 1 : t0 + m, a + m * W, layer, ar0 ? ar0[m] : 0);
    }

    // M consecutive tokens (embeddings x_in [M][H]) at cache rows t0.., rotated to pos[m]. Each token sees the
    // cache up to its own row. Leaves the last token's hidden state in hid.
    // wrow / ar0 / ar1 (optional, [M]): cache row each token is written to and the rows [ar0, ar1) it attends to.
    std::function<void(int64_t, const float *, int)> layer_hook;  // (layer, hidden states [M][H], M) after each layer
    int last_skip = 0;  // the first last_skip tokens of a step only write their keys / values in the last layer
    double t_lin[4] = {0, 0, 0, 0}, t_kvw = 0, t_other = 0;  // step(): qkv / o / gate_up / down passes, cache writes, rest
    void step(const float *x_in, const float *pos, int64_t t0, int M = 1, const int64_t *wrow = nullptr,
              const int64_t *ar0 = nullptr, const int64_t *ar1 = nullptr) {
        float cs[MBP][HD / 2], sn[MBP][HD / 2];
        for (int m = 0; m < M; m++)
            for (int i = 0; i < HD / 2; i++) {
                const float ang = pos[m] * invf[i];
                cs[m][i] = std::cos(ang);
                sn[m][i] = std::sin(ang);
            }
        float *h = bh.data(), *qkv = bq.data(), *a = ba.data(), *y = by.data(), *x = X.data();
        memcpy(x, x_in, (size_t)M * H * sizeof(float));
        for (int64_t l = 0; l < NL; l++) {
            Layer &L = layers[l];
            kv_t *K = kv + (l * 2) * nkv * S * HD, *V = kv + (l * 2 + 1) * nkv * S * HD;
            double tp = now_s();
            for (int m = 0; m < M; m++) rmsnorm(x + m * H, L.in_norm, H, eps, h + m * W);
            t_other += now_s() - tp; tp = now_s();
            linear(L.qkv, h, qkv, M);
            t_lin[0] += now_s() - tp; tp = now_s();
            for (int m = 0; m < M; m++) {
                float *qm = qkv + m * W;
                for (int64_t i = 0; i < nq + nkv; i++) {
                    float *v = qm + i * HD;
                    rmsnorm(v, i < nq ? L.q_norm : L.k_norm, HD, eps, v);
                    rope(v, cs[m], sn[m]);
                }
                for (int64_t i = 0; i < nkv; i++) {
                    const float *kr = qm + (nq + i) * HD, *vr = qm + (nq + nkv + i) * HD;
                    const int64_t row = wrow ? wrow[m] : t0 + m;
                    kv_t *kd = K + (i * S + row) * HD, *vd = V + (i * S + row) * HD;
                    if (opt.kv_sim8) {  // quality simulation of an int8 KV cache: rows rounded to int8, per-row scale
                        float km = 0.f, vm = 0.f;  // kv_sim8: 1 keys and values, 2 keys only, 3 values only
                        for (int d = 0; d < HD; d++) {
                            km = std::max(km, std::fabs(kr[d]));
                            vm = std::max(vm, std::fabs(vr[d]));
                        }
                        const float ks = km / 127.f, vs = vm / 127.f;
                        for (int d = 0; d < HD; d++) {
                            kd[d] = (kv_t)(opt.kv_sim8 == 3 ? kr[d] : ks > 0 ? std::nearbyint(kr[d] / ks) * ks : 0.f);
                            vd[d] = (kv_t)(opt.kv_sim8 == 2 ? vr[d] : vs > 0 ? std::nearbyint(vr[d] / vs) * vs : 0.f);
                        }
                    } else {
#ifdef __EMSCRIPTEN__
                        web_store_row(kd, kr, HD);
                        web_store_row(vd, vr, HD);
#elif defined(TT_X86)
                        kv_store_row(kd, kr, HD);
                        if (!opt.v8 || opt.keep_v16) kv_store_row(vd, vr, HD);
#else
                        for (int d = 0; d < HD; d++) kd[d] = (kv_t)kr[d];
                        if (!opt.v8 || opt.keep_v16)
                            for (int d = 0; d < HD; d++) vd[d] = (kv_t)vr[d];
#endif
                    }
                    if (opt.v8) {  // int8 values, transposed
                        float vm = 0.f;
#ifdef __aarch64__
                        float32x4_t vmx = vdupq_n_f32(0.f);
                        for (int d = 0; d < HD; d += 4) vmx = vmaxq_f32(vmx, vabsq_f32(vld1q_f32(vr + d)));
                        vm = vmaxvq_f32(vmx);
#else
                        for (int d = 0; d < HD; d++) vm = std::max(vm, std::fabs(vr[d]));
#endif
                        const float inv = vm > 0.f ? 127.f / vm : 0.f;
                        int8_t *vh = v8 + (size_t)(l * nkv + i) * S * HD;
                        int8_t tmp[HD];
#ifdef __aarch64__
                        for (int d = 0; d < HD; d += 8) {
                            const int32x4_t c0 = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(vr + d), inv));
                            const int32x4_t c1 = vcvtnq_s32_f32(vmulq_n_f32(vld1q_f32(vr + d + 4), inv));
                            vst1_s8(tmp + d, vqmovn_s16(vcombine_s16(vqmovn_s32(c0), vqmovn_s32(c1))));
                        }
#else
                        for (int d = 0; d < HD; d++) tmp[d] = (int8_t)std::lrintf(vr[d] * inv);
#endif
                        {  // column d of row j sits at base(d) + (j & 7): bases of the 128 columns are 8 bytes apart
                            for (int d = 0; d < HD; d++) vh[v8_index(S, d, row)] = tmp[d];
                        }
                        v8s[(l * nkv + i) * S + row] = vm / 127.f;
                    }
                    if (opt.coarse) {  // int8 copy of the key row for the coarse attention pass
                        float km = 0.f;
                        for (int d = 0; d < HD; d++) km = std::max(km, std::fabs(kr[d]));
                        const float inv = km > 0.f ? 127.f / km : 0.f;
                        int8_t *k8d = k8 + ((l * nkv + i) * S + row) * HD;
                        for (int d = 0; d < HD; d++) k8d[d] = (int8_t)std::lrintf(kr[d] * inv);
                        k8s[(l * nkv + i) * S + row] = km / 127.f;
                    }
                }
            }
            t_kvw += now_s() - tp;
            const int ms = l == NL - 1 ? std::min(last_skip, M) : 0;  // rows whose output nobody reads
            if (ms >= M) break;
            double ta = now_s();
            attention_rows(qkv, K, V, ms, M, t0, ar0, ar1, a, l);
            t_attn += now_s() - ta;
            tp = now_s();
            linear(L.o, a + ms * W, y + ms * W, M - ms);
            t_lin[1] += now_s() - tp; tp = now_s();
            for (int m = ms; m < M; m++) {
                float *xm = x + m * H;
                const float *ym = y + m * W;
                for (int64_t i = 0; i < H; i++) xm[i] += ym[i];
                rmsnorm(xm, L.post_norm, H, eps, h + m * W);
            }
            t_other += now_s() - tp; tp = now_s();
            linear(L.gate_up, h + ms * W, a + ms * W, M - ms);
            t_lin[2] += now_s() - tp; tp = now_s();
            if (opt.act_stats)
                for (int m = 0; m < M; m++) {  // how small are the gate values and the products entering `down`?
                    const float *am = a + m * W;
                    float gmax = 0.f, pmax = 0.f;
                    for (int64_t k = 0; k < I; k++) {
                        const float sg = am[k] / (1.f + std::exp(-am[k]));
                        gmax = std::max(gmax, std::fabs(sg));
                        pmax = std::max(pmax, std::fabs(sg * am[I + k]));
                    }
                    for (int64_t k = 0; k < I; k++) {
                        const float sg = std::fabs(am[k] / (1.f + std::exp(-am[k]))), pr = sg * std::fabs(am[I + k]);
                        for (int b = 0; b < 5; b++) {
                            const float th = std::pow(10.f, -(float)(b + 1));  // 1e-1 .. 1e-5 of the maximum
                            st_gate[b] += sg < th * gmax;
                            st_prod[b] += pr < th * pmax;
                        }
                    }
                    st_n += I;
                }
            {  // all rows in one region, in pieces of 512 values from each row's start
                const int64_t np = (I + 511) / 512;
                pool->run((M - ms) * np, 1, [&](int64_t t0, int64_t t1) {
                    for (int64_t t = t0; t < t1; t++) {
                        const int64_t i0 = t % np * 512;
                        silu_mul(a + (ms + t / np) * W, i0, std::min(I, i0 + 512), I);
                    }
                });
            }
            if (opt.mlp_drop > 0.f)  // simulation: entries of the `down` input below mlp_drop * row maximum are zeroed
                for (int m = 0; m < M; m++) {
                    float *am = a + m * W, amax = 0.f;
                    for (int64_t k = 0; k < I; k++) amax = std::max(amax, std::fabs(am[k]));
                    const float th = opt.mlp_drop * amax;
                    for (int64_t k = 0; k < I; k++)
                        if (std::fabs(am[k]) < th) { am[k] = 0.f; st_drop++; }
                    st_dn += I;
                }
            t_other += now_s() - tp; tp = now_s();
            linear(L.down, a + ms * W, y + ms * W, M - ms);
            t_lin[3] += now_s() - tp; tp = now_s();
            for (int m = ms; m < M; m++) {
                float *xm = x + m * H;
                const float *ym = y + m * W;
                for (int64_t i = 0; i < H; i++) xm[i] += ym[i];
            }
            t_other += now_s() - tp;
            if (layer_hook) layer_hook(l, x, M);
        }
        hid = x + (size_t)(M - 1) * H;
    }

    // ---- draft head for speculative decoding (linears as q8)
    bool has_draft = false;
    const float *d_nh[2], *d_ne[2], *d_bn[2], *d_no[2];  // LayerNorm weight, bias
    Lin d_ih, d_ie, d_ba, d_bb, d_bott, d_out;
    const uint16_t *embed16 = nullptr;  // model input embeddings of the local ids, fp16 [n_local][H]
    std::vector<float> dq;
    double t_draft = 0;

    void load_draft(const File &f, const File &fe) {
        auto ln = [&](const char *n, const float **dst) {
            dst[0] = (const float *)f.get(std::string(n) + ".w").p;
            dst[1] = (const float *)f.get(std::string(n) + ".b").p;
        };
        ln("nh", d_nh); ln("ne", d_ne); ln("bn", d_bn); ln("no", d_no);
        Lin *ls[6] = {&d_ih, &d_ie, &d_ba, &d_bb, &d_bott, &d_out};
        const char *names[6] = {"ih", "ie", "ba", "bb", "bott", "out"};
        for (int i = 0; i < 6; i++) {
            ls[i]->load(f, names[i]);
            if (opt.mm) ls[i]->pack_mm(); else ls[i]->pack();
        }
        embed16 = (const uint16_t *)fe.get("embed").p;
        dq.resize((d_out.N + 3) / 4 * 4 + 8);
        has_draft = true;
    }
    void embed_row(int tok, float *out) const {
        const uint16_t *e = embed16 + (size_t)tok * H;
        for (int64_t i = 0; i < H; i++) out[i] = h2f(e[i]);
    }
    static void layernorm(const float *x, const float *const *wb, int64_t n, float *out) {
        double mean = 0, var = 0;
        for (int64_t i = 0; i < n; i++) mean += x[i];
        mean /= (double)n;
        for (int64_t i = 0; i < n; i++) var += (x[i] - mean) * (x[i] - mean);
        const float r = 1.f / std::sqrt((float)(var / (double)n) + 1e-5f), mu = (float)mean;
        for (int64_t i = 0; i < n; i++) out[i] = (x[i] - mu) * r * wb[0][i] + wb[1][i];
    }
    static void gelu(float *x, int64_t n) {
        for (int64_t i = 0; i < n; i++) x[i] = 0.5f * x[i] * (1.f + std::erf(x[i] * 0.70710678f));
    }
    // Draft logits for the token after `tok`, from the hidden state hrow that tok's logits were computed from.
    const float *draft(const float *hrow, int tok) {
        const double t0 = now_s();
        float *r0 = by.data(), *r1 = r0 + W, *r2 = r1 + W, *r3 = r2 + W;
        const int64_t Dh = d_ih.N;
        rmsnorm(hrow, norm, H, eps, r2);  // the head reads the state after the model's final norm
        layernorm(r2, d_nh, H, r0);
        linear(d_ih, r0, r1);
        embed_row(tok, r0);
        layernorm(r0, d_ne, H, r2);
        linear(d_ie, r2, r3);
        for (int64_t i = 0; i < Dh; i++) r1[i] += r3[i];
        layernorm(r1, d_bn, Dh, r0);
        linear(d_ba, r0, r2);
        gelu(r2, d_ba.N);
        linear(d_bb, r2, r3);
        for (int64_t i = 0; i < Dh; i++) r1[i] += r3[i];
        gelu(r1, Dh);
        layernorm(r1, d_no, Dh, r2);
        linear(d_bott, r2, r3);
        linear(d_out, r3, dq.data());
        t_draft += now_s() - t0;
        return dq.data();
    }

    // Audio head on the current hidden state: logits [n_local].
    void head(float *logits, const float *hrow = nullptr) {
        double t0 = now_s();
        float *h = bh.data(), *y = bq.data(), *z = ba.data();
        rmsnorm(hrow ? hrow : hid, norm, H, eps, h);
        linear(fc1, h, y);
        int64_t i = 0;
#ifdef __aarch64__
        for (; i + 4 <= fc1.N; i += 4) {  // mish = x * tanh(softplus(x)) = x * (1 - u^2) / (1 + u^2), u = sigmoid(-x)
            const float32x4_t x = vld1q_f32(y + i);
            const float32x4_t e = vexp_neg(vnegq_f32(vabsq_f32(x)));  // exp(-|x|)
            const float32x4_t den = vaddq_f32(e, vdupq_n_f32(1.f));
            const float32x4_t u = vdivq_f32(vbslq_f32(vcgezq_f32(x), e, vdupq_n_f32(1.f)), den);
            const float32x4_t u2 = vmulq_f32(u, u);
            vst1q_f32(y + i, vmulq_f32(x, vdivq_f32(vsubq_f32(vdupq_n_f32(1.f), u2), vaddq_f32(vdupq_n_f32(1.f), u2))));
        }
#endif
        for (; i < fc1.N; i++) {  // mish; softplus is linear above 20 (exp overflow)
            const float v = y[i];
            const float sp = v > 20.f ? v : std::log1p(std::exp(v));
            y[i] = v * std::tanh(sp);
        }
        if (opt.head2 > 0) {
            // 1) coarse logits of all rows from the 4-bit copy; 2) exact logits of the best head2 rows (whole quads)
            linear(fc2c, y, z);
            zc.assign(z, z + fc2.N);
            cand.resize(fc2.N);
            for (int64_t i2 = 0; i2 < fc2.N; i2++) cand[i2] = (int32_t)i2;
            const int64_t nc = std::min<int64_t>(opt.head2, fc2.N);
            std::nth_element(cand.begin(), cand.begin() + nc, cand.end(),
                             [&](int32_t a, int32_t b) { return zc[a] > zc[b]; });
            std::vector<uint8_t> need((fc2.N + 3) / 4, 0);
            for (int64_t i2 = 0; i2 < nc; i2++) need[cand[i2] / 4] = 1;
            for (int64_t i2 = 16384; i2 < fc2.N; i2++) need[i2 / 4] = 1;  // EOS / marker rows: always exact
            float *ex = by.data();
            if (opt.head2_check) {  // exact logits of everything, to count misses of the true top 50
                linear(fc2, y, ex);
                std::vector<int32_t> top(fc2.N);
                for (int64_t i2 = 0; i2 < fc2.N; i2++) top[i2] = (int32_t)i2;
                std::nth_element(top.begin(), top.begin() + 50, top.end(), [&](int32_t a, int32_t b) { return ex[a] > ex[b]; });
                bool miss = false;
                for (int i2 = 0; i2 < 50; i2++) miss |= !need[top[i2] / 4];
                h2_miss += miss;
                for (int64_t q = 0; q < (int64_t)need.size(); q++)
                    if (need[q]) { h2_rows += 4; for (int r = 0; r < 4 && q * 4 + r < fc2.N; r++) z[q * 4 + r] = ex[q * 4 + r]; }
            } else {
                // quantize() already ran for y in linear(fc2c): acts[0] holds it (same activations for fc2)
                qlist.clear();
                for (int64_t q = 0; q < (int64_t)need.size(); q++)
                    if (need[q]) qlist.push_back((int32_t)q);
                h2_rows += 4 * (int64_t)qlist.size();
                const double tg = now_s();
                pool->run((int64_t)qlist.size(), 8, [&](int64_t a0, int64_t a1) {
                    for (int64_t k = a0; k < a1; k++) {
                        const int64_t q = qlist[k];
#ifdef __aarch64__
                        if (opt.mm) { gemv_q8_mm(fc2, acts, 1, z, W, q, q + 1); continue; }
#endif
                        gemv_scalar(fc2, acts[0], z, q * 4, std::min<int64_t>(q * 4 + 4, fc2.N));
                    }
                });
                t_gemv += now_s() - tg;
                if (opt.head2_mask)  // sampling: only rows with exact logits are eligible
                    for (int64_t q = 0; q < (int64_t)need.size(); q++)
                        if (!need[q])
                            for (int r = 0; r < 4 && q * 4 + r < fc2.N; r++) z[q * 4 + r] = -1e30f;
            }
            h2_steps++;
        } else {
            linear(fc2, y, z);
        }
        memcpy(logits, z, n_local * sizeof(float));
        t_head += now_s() - t0;
    }
};

// ---------------------------------------------------------------- sampling and speculation
struct Dist {  // a distribution on a small support
    std::vector<int> id;
    std::vector<float> p;
    float prob(int t) const {
        for (size_t i = 0; i < id.size(); i++)
            if (id[i] == t) return p[i];
        return 0.f;
    }
    int sample(std::mt19937_64 &rng) const {
        float u = std::uniform_real_distribution<float>(0.f, 1.f)(rng), c = 0.f;
        for (size_t i = 0; i < id.size(); i++) {
            c += p[i];
            if (u < c) return id[i];
        }
        return id.back();
    }
};

struct Sampler {
    static constexpr int EOS = 16384, MARKER = 16385;
    float temp = 0.7f, top_p = 0.95f, freq_pen = 0.1f;
    int top_k = 50, min_tokens = 13;
    std::vector<float> bias, counts, z;
    std::vector<int> order;
    std::mt19937_64 rng{1234};

    explicit Sampler(int n) : bias(n, 0.f), counts(n, 0.f), z(n), order(n) {
        static const int sil[] = {3716, 12725, 3974, 4299, 10309, 5995, 6144, 15170, 13601, 398,
                                  7362, 17, 3710, 1531, 1384, 12293, 5169, 4544};
        bias[EOS] = -0.8f;
        for (int t : sil) bias[t] = -0.2f;
    }
    // softmax(z / temp) on the k largest entries of z
    void top_softmax(const float *zz, int n, int k, float t, Dist &d) {
        for (int i = 0; i < n; i++) order[i] = i;
        k = std::min(k, n);
        std::nth_element(order.begin(), order.begin() + k, order.begin() + n, [&](int a, int b) { return zz[a] > zz[b]; });
        d.id.assign(order.begin(), order.begin() + k);
        d.p.resize(k);
        float mx = -INFINITY, sum = 0.f;
        for (int i = 0; i < k; i++) mx = std::max(mx, zz[d.id[i]]);
        for (int i = 0; i < k; i++) { d.p[i] = std::exp((zz[d.id[i]] - mx) / t); sum += d.p[i]; }
        for (int i = 0; i < k; i++) d.p[i] /= sum;
    }
    // The product sampling distribution: bias, frequency penalty, min_tokens, temperature, top-k, top-p (vLLM order).
    void target(const float *logits, int n_generated, Dist &d) {
        const int n = (int)bias.size();
        for (int i = 0; i < n; i++) z[i] = logits[i] + bias[i] - freq_pen * counts[i];
        if (n_generated < min_tokens) z[EOS] = z[MARKER] = -INFINITY;
        top_softmax(z.data(), n, top_k, temp, d);
        std::vector<int> ord(d.id.size());
        for (size_t i = 0; i < ord.size(); i++) ord[i] = (int)i;
        std::sort(ord.begin(), ord.end(), [&](int a, int b) { return d.p[a] < d.p[b]; });  // ascending
        float c = 0.f, keep = 0.f;
        std::vector<char> drop(ord.size(), 0);
        for (size_t i = 0; i + 1 < ord.size(); i++) {
            c += d.p[ord[i]];
            if (c <= 1.f - top_p) drop[ord[i]] = 1;
        }
        Dist o;
        for (size_t i = 0; i < d.id.size(); i++)
            if (!drop[i]) { o.id.push_back(d.id[i]); o.p.push_back(d.p[i]); keep += d.p[i]; }
        for (auto &v : o.p) v /= keep;
        d = o;
    }
    // Speculative sampling: guess g drawn from q; accept with min(1, p(g) / q(g)), else draw from max(0, p - q).
    // Returns the token (distributed as p) and whether the guess was accepted.
    int accept_or_resample(const Dist &p, const Dist &q, int g, bool &accepted) {
        const float pg = p.prob(g), qg = q.prob(g);
        const float u = std::uniform_real_distribution<float>(0.f, 1.f)(rng);
        accepted = u * qg < pg;
        if (accepted) return g;
        Dist r;
        float sum = 0.f;
        for (size_t i = 0; i < p.id.size(); i++) {
            const float v = std::max(0.f, p.p[i] - q.prob(p.id[i]));
            if (v > 0.f) { r.id.push_back(p.id[i]); r.p.push_back(v); sum += v; }
        }
        if (r.id.empty()) return p.sample(rng);
        for (auto &v : r.p) v /= sum;
        return r.sample(rng);
    }
    static float overlap(const Dist &p, const Dist &q) {
        float s = 0.f;
        for (size_t i = 0; i < p.id.size(); i++) s += std::min(p.p[i], q.prob(p.id[i]));
        return s;
    }
};

// ---------------------------------------------------------------- main
