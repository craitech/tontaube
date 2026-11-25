// The engine's calls into the operating system, all in this file: Linux, Windows, macOS (Apple silicon) and the
// browser build (Emscripten). The rest of the engine is the same everywhere.
//
//   now_s, sleep_ms        monotonic clock in seconds; sleep
//   wait, wake_all         sleep while an atomic holds a value; wake every thread asleep on it (a futex)
//   Thread                 start and join a thread
//   pin_thread, Affinity   the calling thread onto one CPU; the calling thread's CPUs, put back at the end of a scope
//   cpus, cpu_quota        the CPUs this thread may run on (core, NUMA node, last-level cache, capacity); the CPUs a
//                          container's quota allows (0: no quota)
//   open_file, map_file    a file by its UTF-8 path; a file read-only in memory, once per process
//   big_alloc, big_free    large zeroed buffers (null if there is no memory)
//
// CPU ids: Linux the kernel's; Windows 64 * processor group + number in the group; macOS the performance cores
// 0..n-1 (macOS has no pinning: a pinned thread asks for the performance cores instead). The browser: no CPU ids.
#pragma once
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#if defined(__EMSCRIPTEN__)
#include <emscripten/threading.h>
#elif defined(__APPLE__)
#include <pthread/qos.h>
#include <sys/sysctl.h>
#else
#include <dirent.h>
#include <fstream>
#include <sched.h>
#include <sys/syscall.h>
#include <linux/futex.h>
#endif
#endif

namespace plat {

inline double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
inline void sleep_ms(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

struct Cpu { int id = 0, core = 0, node = 0, llc = -1; double cap = 0; };  // core: the core's first allowed CPU

inline std::vector<int> parse_list(const std::string &s) {  // "0-3,8,10-11"
    std::vector<int> v;
    const char *p = s.c_str();
    while (*p) {
        char *e;
        const long a = strtol(p, &e, 10);
        if (e == p) break;
        long b = a;
        p = e;
        if (*p == '-') { b = strtol(p + 1, &e, 10); p = e; }
        for (long k = a; k <= b && k - a < 65536; k++) v.push_back((int)k);
        if (*p != ',') break;
        p++;
    }
    return v;
}

// One mapping per file and process, found by the file's identity (device, file number, size, time).
inline const uint8_t *shared_mapping(const std::vector<int64_t> &key, const uint8_t *(*map)(void *), void *arg) {
    static std::mutex mu;
    static std::map<std::vector<int64_t>, const uint8_t *> maps;
    std::lock_guard<std::mutex> g(mu);
    auto it = maps.find(key);
    if (it != maps.end()) return it->second;
    const uint8_t *p = map(arg);
    if (p) maps[key] = p;
    return p;
}

}  // namespace plat

#if defined(_WIN32)
// ---------------------------------------------------------------- Windows (x86-64; link with -lsynchronization)
namespace plat {

inline std::wstring widen(const char *s) {  // UTF-8 -> UTF-16
    const int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, nullptr, 0);
    std::wstring w(n > 0 ? n : 1, L'\0');
    if (n > 0) MultiByteToWideChar(CP_UTF8, 0, s, -1, &w[0], n);
    w.resize(wcslen(w.c_str()));
    return w;
}

inline void wait(std::atomic<uint32_t> &a, uint32_t seen) { WaitOnAddress((volatile void *)&a, &seen, sizeof seen, INFINITE); }
inline void wake_all(std::atomic<uint32_t> &a) { WakeByAddressAll((void *)&a); }

struct Thread {
    HANDLE h = nullptr;
    struct Call { void *(*fn)(void *); void *arg; };
    static DWORD WINAPI entry(LPVOID p) {
        const Call c = *(Call *)p;
        delete (Call *)p;
        c.fn(c.arg);
        return 0;
    }
    bool start(void *(*fn)(void *), void *arg) {
        Call *c = new Call{fn, arg};
        h = CreateThread(nullptr, (SIZE_T)8 << 20, entry, c, STACK_SIZE_PARAM_IS_A_RESERVATION, nullptr);
        if (!h) delete c;
        return h != nullptr;
    }
    void join() { if (h) { WaitForSingleObject(h, INFINITE); CloseHandle(h); h = nullptr; } }
};

inline void pin_thread(int cpu) {
    GROUP_AFFINITY g;
    memset(&g, 0, sizeof g);
    g.Group = (WORD)(cpu / 64);
    g.Mask = (KAFFINITY)1 << (cpu % 64);
    SetThreadGroupAffinity(GetCurrentThread(), &g, nullptr);
}

struct Affinity {  // the calling thread's CPUs, put back when this goes out of scope
    GROUP_AFFINITY g;
    bool ok;
    Affinity() { memset(&g, 0, sizeof g); ok = GetThreadGroupAffinity(GetCurrentThread(), &g) != 0; }
    ~Affinity() { if (ok) SetThreadGroupAffinity(GetCurrentThread(), &g, nullptr); }
};

// The CPUs of this thread's processor group it may run on. cap: the core's efficiency class + 1 (hybrid CPUs: the
// efficiency cores have the lower one).
inline std::vector<Cpu> cpus() {
    GROUP_AFFINITY self;
    if (!GetThreadGroupAffinity(GetCurrentThread(), &self)) return {};
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationAll, nullptr, &len);
    std::vector<uint8_t> buf(len);
    if (!len || !GetLogicalProcessorInformationEx(RelationAll, (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)buf.data(), &len)) return {};
    auto allowed = [&](const GROUP_AFFINITY &g) {
        std::vector<int> v;
        if (g.Group == self.Group)
            for (int b = 0; b < 64; b++)
                if ((g.Mask & self.Mask) >> b & 1) v.push_back(64 * g.Group + b);
        return v;
    };
    std::map<int, Cpu> out;
    std::map<int, int> node, llc, level;
    for (DWORD off = 0; off + sizeof(LOGICAL_PROCESSOR_RELATIONSHIP) < len;) {
        const auto *x = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)(buf.data() + off);
        if (!x->Size) break;
        off += x->Size;
        if (x->Relationship == RelationProcessorCore) {
            for (WORD k = 0; k < x->Processor.GroupCount; k++) {
                const std::vector<int> v = allowed(x->Processor.GroupMask[k]);
                for (int c : v) {
                    Cpu &p = out[c];
                    p.id = c;
                    p.core = v[0];
                    p.cap = 1 + x->Processor.EfficiencyClass;
                }
            }
        } else if (x->Relationship == RelationNumaNode) {
            for (int c : allowed(x->NumaNode.GroupMask)) node[c] = (int)x->NumaNode.NodeNumber;
        } else if (x->Relationship == RelationCache) {  // the highest level shared by a group: the last-level cache
            const std::vector<int> v = allowed(x->Cache.GroupMask);
            for (int c : v)
                if (x->Cache.Level > level[c]) { level[c] = x->Cache.Level; llc[c] = v[0]; }
        }
    }
    std::vector<Cpu> r;
    for (auto &kv : out) {
        Cpu c = kv.second;
        c.node = node.count(c.id) ? node[c.id] : 0;
        c.llc = llc.count(c.id) ? llc[c.id] : -1;
        r.push_back(c);
    }
    return r;
}

inline double cpu_quota() { return 0; }

inline FILE *open_file(const char *path, const char *mode) { return _wfopen(widen(path).c_str(), widen(mode).c_str()); }

inline const uint8_t *map_file(const char *path, size_t &size, bool) {
    HANDLE f = CreateFileW(widen(path).c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE) return nullptr;
    BY_HANDLE_FILE_INFORMATION fi;
    const uint8_t *p = nullptr;
    if (GetFileInformationByHandle(f, &fi)) {
        size = (size_t)fi.nFileSizeHigh << 32 | fi.nFileSizeLow;
        const std::vector<int64_t> key{fi.dwVolumeSerialNumber, fi.nFileIndexHigh, fi.nFileIndexLow, (int64_t)size,
                                       fi.ftLastWriteTime.dwHighDateTime, fi.ftLastWriteTime.dwLowDateTime};
        p = shared_mapping(key, [](void *h) -> const uint8_t * {
            HANDLE m = CreateFileMappingW((HANDLE)h, nullptr, PAGE_READONLY, 0, 0, nullptr);
            if (!m) return nullptr;
            const uint8_t *v = (const uint8_t *)MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
            CloseHandle(m);  // the view keeps the mapping
            return v;
        }, (void *)f);
    }
    CloseHandle(f);
    return p;
}

inline void *big_alloc(size_t bytes) { return VirtualAlloc(nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE); }
inline void big_free(void *a, size_t) { if (a) VirtualFree(a, 0, MEM_RELEASE); }

}  // namespace plat
#else
// ---------------------------------------------------------------- POSIX: Linux, macOS, the browser
namespace plat {

struct Thread {
    pthread_t t;
    bool start(void *(*fn)(void *), void *arg) {
#ifdef __APPLE__
        pthread_attr_t a;  // macOS gives new threads 512 KiB
        pthread_attr_init(&a);
        pthread_attr_setstacksize(&a, (size_t)8 << 20);
        const bool ok = pthread_create(&t, &a, fn, arg) == 0;
        pthread_attr_destroy(&a);
        return ok;
#else
        return pthread_create(&t, nullptr, fn, arg) == 0;
#endif
    }
    void join() { pthread_join(t, nullptr); }
};

inline FILE *open_file(const char *path, const char *mode) { return fopen(path, mode); }

#if defined(MADV_HUGEPAGE) && !defined(__EMSCRIPTEN__)
// The file copied into 2 MiB pages (MADV_HUGEPAGE): streaming ~0.6 GB of weights per frame through 4 KiB page-cache
// pages stops the hardware prefetcher at every page (measured: LM frame step 9.4-10.0 -> 7.5-8.1 ms, decoder 18.5-20.0
// -> 17.8 ms at 4 threads). Only with transparent huge pages on and enough free memory; null: keep the mapping.
inline const uint8_t *huge_copy(const uint8_t *p, size_t n) {
    static const bool thp = [] {
        FILE *f = fopen("/sys/kernel/mm/transparent_hugepage/enabled", "r");
        char b[128] = {0};
        if (f) { if (!fgets(b, sizeof b, f)) b[0] = 0; fclose(f); }
        return b[0] && !strstr(b, "[never]");
    }();
    if (!thp) return nullptr;
    long long avail_kb = 0;
    if (FILE *f = fopen("/proc/meminfo", "r")) {
        char line[256];
        while (fgets(line, sizeof line, f)) if (sscanf(line, "MemAvailable: %lld kB", &avail_kb) == 1) break;
        fclose(f);
    }
    if ((size_t)avail_kb * 1024 < n + ((size_t)2 << 30)) return nullptr;  // keep 2 GiB for everything else
    const size_t A = (size_t)2 << 20, sz = (n + A - 1) & ~(A - 1);
    uint8_t *m = (uint8_t *)mmap(nullptr, sz + A, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) return nullptr;
    uint8_t *a = (uint8_t *)(((uintptr_t)m + A - 1) & ~((uintptr_t)A - 1));
    if (a > m) munmap(m, a - m);
    if (m + A > a) munmap(a + sz, m + A - a);
    madvise(a, sz, MADV_HUGEPAGE);
    memcpy(a, p, n);
    mprotect(a, sz, PROT_READ);
    return a;
}
#else
inline const uint8_t *huge_copy(const uint8_t *, size_t) { return nullptr; }
#endif

// copy: on Linux into huge pages where possible (huge_copy); the browser: a plain mapping of its file system's file.
inline const uint8_t *map_file(const char *path, size_t &size, bool copy) {
    const int fd = ::open(path, O_RDONLY);
    if (fd < 0) return nullptr;
    struct stat st;
    fstat(fd, &st);
    size = st.st_size;
#ifdef __EMSCRIPTEN__
    (void)copy;
    const uint8_t *p = (const uint8_t *)mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (p == MAP_FAILED) p = nullptr;
#else
#ifdef __APPLE__
    const struct timespec &mt = st.st_mtimespec;
#else
    const struct timespec &mt = st.st_mtim;
#endif
    struct Arg { int fd; size_t n; bool copy; } arg{fd, size, copy};
    const uint8_t *p = shared_mapping({(int64_t)st.st_dev, (int64_t)st.st_ino, (int64_t)st.st_size, (int64_t)mt.tv_sec, (int64_t)mt.tv_nsec},
                                      [](void *v) -> const uint8_t * {
                                          const Arg &x = *(const Arg *)v;
                                          const uint8_t *m = (const uint8_t *)mmap(nullptr, x.n, PROT_READ, MAP_PRIVATE, x.fd, 0);
                                          if (m == MAP_FAILED) return nullptr;
                                          if (const uint8_t *c = x.copy ? huge_copy(m, x.n) : nullptr) { munmap((void *)m, x.n); m = c; }
                                          return m;
                                      }, &arg);
#endif
    close(fd);
    return p;
}

#ifdef __EMSCRIPTEN__
inline void *big_alloc(size_t bytes) { return aligned_alloc(64, (bytes + 63) / 64 * 64); }  // fresh heap pages are zero
inline void big_free(void *a, size_t) { free(a); }
#else
// 2 MiB aligned, in transparent huge pages where the system allows them (Linux).
inline void *big_alloc(size_t bytes) {
    const size_t A = 2u << 20, sz = (bytes + A - 1) & ~(A - 1);
    uint8_t *p = (uint8_t *)mmap(nullptr, sz + A, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return nullptr;
    uint8_t *a = (uint8_t *)(((uintptr_t)p + A - 1) & ~((uintptr_t)A - 1));
    if (a > p) munmap(p, a - p);                // keep exactly [a, a + sz)
    if (p + A > a) munmap(a + sz, p + A - a);
#ifdef MADV_HUGEPAGE
    madvise(a, sz, MADV_HUGEPAGE);
#endif
    return a;
}
inline void big_free(void *a, size_t bytes) {  // bytes: as given to big_alloc
    if (a) munmap(a, (bytes + (2u << 20) - 1) & ~((size_t)(2u << 20) - 1));
}
#endif

}  // namespace plat

#if defined(__EMSCRIPTEN__)
// ---------------------------------------------------------------- the browser
namespace plat {
inline void wait(std::atomic<uint32_t> &a, uint32_t seen) { emscripten_futex_wait((void *)&a, seen, INFINITY); }
inline void wake_all(std::atomic<uint32_t> &a) { emscripten_futex_wake((void *)&a, 1 << 30); }
inline void pin_thread(int) {}
}  // namespace plat

#elif defined(__APPLE__)
// ---------------------------------------------------------------- macOS
extern "C" int __ulock_wait(uint32_t op, void *addr, uint64_t value, uint32_t timeout_us);  // libSystem (as libc++'s
extern "C" int __ulock_wake(uint32_t op, void *addr, uint64_t value);                      // std::atomic wait)
namespace plat {
inline void wait(std::atomic<uint32_t> &a, uint32_t seen) { __ulock_wait(1 /* UL_COMPARE_AND_WAIT */, (void *)&a, seen, 0); }
inline void wake_all(std::atomic<uint32_t> &a) { __ulock_wake(1 | 0x100 /* ULF_WAKE_ALL */, (void *)&a, 0); }

// No pinning on macOS: the thread asks for the performance cores (quality of service class).
inline void pin_thread(int) { pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0); }
struct Affinity {  // the calling thread's quality of service class, put back when this goes out of scope
    qos_class_t q = QOS_CLASS_UNSPECIFIED;
    int rel = 0;
    Affinity() { pthread_get_qos_class_np(pthread_self(), &q, &rel); }
    ~Affinity() { if (q != QOS_CLASS_UNSPECIFIED) pthread_set_qos_class_self_np(q, rel); }
};

inline int sysctl_int(const char *name) {
    int v = 0;
    size_t n = sizeof v;
    return sysctlbyname(name, &v, &n, nullptr, 0) == 0 ? v : 0;
}
// The performance cores (Apple silicon: perflevel0), one CPU each.
inline std::vector<Cpu> cpus() {
    int n = sysctl_int("hw.perflevel0.physicalcpu");
    if (n <= 0) n = sysctl_int("hw.physicalcpu");
    std::vector<Cpu> r;
    for (int i = 0; i < n; i++) { Cpu c; c.id = c.core = i; c.llc = 0; r.push_back(c); }
    return r;
}
inline double cpu_quota() { return 0; }
}  // namespace plat

#else
// ---------------------------------------------------------------- Linux
namespace plat {
inline void wait(std::atomic<uint32_t> &a, uint32_t seen) { syscall(SYS_futex, (uint32_t *)&a, FUTEX_WAIT, seen, nullptr, nullptr, 0); }
inline void wake_all(std::atomic<uint32_t> &a) { syscall(SYS_futex, (uint32_t *)&a, FUTEX_WAKE, 1 << 30, nullptr, nullptr, 0); }

inline void pin_thread(int cpu) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    sched_setaffinity(0, sizeof(set), &set);
}
struct Affinity {  // the calling thread's CPU set, put back when this goes out of scope
    cpu_set_t set;
    bool ok;
    Affinity() { CPU_ZERO(&set); ok = sched_getaffinity(0, sizeof set, &set) == 0; }
    ~Affinity() { if (ok) sched_setaffinity(0, sizeof set, &set); }
};

inline std::string read_line(const std::string &path) {
    std::ifstream f(path);
    std::string s;
    std::getline(f, s);
    return s;
}

// From sysfs; cap: cpu_capacity (big.LITTLE; 0 where there is none).
inline std::vector<Cpu> cpus() {
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set)) return {};
    std::map<int, int> node_of;
    if (DIR *d = opendir("/sys/devices/system/node")) {
        while (dirent *e = readdir(d))
            if (!strncmp(e->d_name, "node", 4) && isdigit((unsigned char)e->d_name[4]))
                for (int c : parse_list(read_line(std::string("/sys/devices/system/node/") + e->d_name + "/cpulist")))
                    node_of[c] = atoi(e->d_name + 4);
        closedir(d);
    }
    std::vector<Cpu> out;
    for (int c = 0; c < CPU_SETSIZE; c++) {
        if (!CPU_ISSET(c, &set)) continue;
        const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(c);
        Cpu x;
        x.id = x.core = c;
        for (int s : parse_list(read_line(base + "/topology/thread_siblings_list")))
            if (s < x.core && CPU_ISSET(s, &set)) x.core = s;
        x.node = node_of.count(c) ? node_of[c] : 0;
        int best = -1;
        for (int i = 0; i < 16; i++) {  // the highest cache level shared by a group: the last-level cache
            const std::string ix = base + "/cache/index" + std::to_string(i);
            const std::string lv = read_line(ix + "/level");
            if (lv.empty()) break;
            const std::vector<int> sh = parse_list(read_line(ix + "/shared_cpu_list"));
            if (atoi(lv.c_str()) > best && !sh.empty()) { best = atoi(lv.c_str()); x.llc = *std::min_element(sh.begin(), sh.end()); }
        }
        x.cap = atof(read_line(base + "/cpu_capacity").c_str());
        out.push_back(x);
    }
    return out;
}

// cgroup v2 (this group and its parents) and v1.
inline double cpu_quota() {
    double q = 0;
    auto take = [&](double v) { if (v > 0 && (q == 0 || v < q)) q = v; };
    std::ifstream cg("/proc/self/cgroup");
    std::string line, path;
    while (std::getline(cg, line))
        if (line.rfind("0::", 0) == 0) path = line.substr(3);
    for (std::string p = path;; p = p.substr(0, p.rfind('/'))) {
        const std::string s = read_line("/sys/fs/cgroup" + p + "/cpu.max");
        if (!s.empty() && s.rfind("max", 0) != 0) {
            double a = 0, b = 0;
            if (sscanf(s.c_str(), "%lf %lf", &a, &b) == 2 && b > 0) take(a / b);
        }
        if (p.empty() || p == "/") break;
    }
    const double a = atof(read_line("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").c_str()),
                 b = atof(read_line("/sys/fs/cgroup/cpu/cpu.cfs_period_us").c_str());
    if (a > 0 && b > 0) take(a / b);
    return q;
}
}  // namespace plat
#endif
#endif
