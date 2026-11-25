// Runtime CPU feature detection for the kernels: x86 (cpuid + OS register state; AMX on Linux only), aarch64 (Linux
// hwcaps, macOS sysctl). Others: all false.
//
// Kernels for each instruction set live in one translation unit under KERN_TARGET_* regions (see below) and are
// chosen at run time from isa(), so the code builds without -m flags and runs on any x86-64 CPU.
#pragma once
#include <cstdint>
#include <string>

#if defined(__x86_64__) || defined(_M_X64)
#define KERN_X86 1
#include <cpuid.h>
#include <immintrin.h>
#ifdef __linux__
#include <sys/syscall.h>
#include <unistd.h>
#endif
#elif defined(__aarch64__) && !defined(__EMSCRIPTEN__)
#define KERN_ARM 1
#include <arm_neon.h>
#ifdef __linux__
#include <sys/auxv.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#endif

// Regions of functions compiled for one instruction set: KERN_TARGET_BEGIN_AVX2 ... KERN_TARGET_END.
// Helpers defined in a smaller set (AVX2) inline into functions of a larger one (AVX-512).
#if defined(__clang__)
#define KERN_PRAGMA(x) _Pragma(#x)
#define KERN_TARGET_BEGIN(t) KERN_PRAGMA(clang attribute push(__attribute__((target(t))), apply_to = function))
#define KERN_TARGET_END _Pragma("clang attribute pop")
#elif defined(__GNUC__)
#define KERN_PRAGMA(x) _Pragma(#x)
#define KERN_TARGET_BEGIN(t) _Pragma("GCC push_options") KERN_PRAGMA(GCC target(t))
#define KERN_TARGET_END _Pragma("GCC pop_options")
#endif
#define KERN_T_AVX2 "avx2,fma,f16c"
#define KERN_T_AVX512 "avx2,fma,f16c,avx512f,avx512bw,avx512vl,avx512vnni"
#define KERN_T_AVXVNNI "avx2,fma,f16c,avxvnni"
#define KERN_T_VNNI256 "avx2,fma,f16c,avx512f,avx512vl,avx512vnni"
// aarch64: NEON is the baseline; the dot product (ARMv8.2) and the int8 matrix multiply (ARMv8.6) are chosen at run time
#ifdef __clang__
#define KERN_T_DOT "dotprod"
#define KERN_T_I8MM "dotprod,i8mm"
#else
#define KERN_T_DOT "+dotprod"
#define KERN_T_I8MM "+dotprod+i8mm"
#endif

namespace kern {

struct Isa {
    bool avx2 = false, fma = false, f16c = false;
    bool avx512f = false, avx512bw = false, avx512vl = false, avx512vnni = false, avx512bf16 = false;
    bool avxvnni = false;
    bool amx_tile = false, amx_int8 = false, amx_bf16 = false;  // AMX: only when the OS granted the tile state
    bool dotprod = false, i8mm = false, sve2 = false;            // aarch64 (hwcaps)
};

#ifdef KERN_X86
inline Isa detect_isa() {
    Isa r;
    unsigned a, b, c, d;
    if (!__get_cpuid(0, &a, &b, &c, &d)) return r;
    const unsigned maxleaf = a;
    __get_cpuid(1, &a, &b, &c, &d);
    const bool osxsave = c >> 27 & 1, avx = c >> 28 & 1;
    if (!osxsave || !avx) return r;
    unsigned xlo, xhi;
    __asm__ volatile("xgetbv" : "=a"(xlo), "=d"(xhi) : "c"(0));
    const uint64_t xcr0 = (uint64_t)xhi << 32 | xlo;
    const bool ymm = (xcr0 & 0x6) == 0x6, zmm = ymm && (xcr0 & 0xe0) == 0xe0, tiles = (xcr0 & 0x60000) == 0x60000;
    if (!ymm) return r;
    r.fma = c >> 12 & 1;
    r.f16c = c >> 29 & 1;
    if (maxleaf < 7) return r;
    __cpuid_count(7, 0, a, b, c, d);
    const unsigned b7 = b, c7 = c, d7 = d;
    __cpuid_count(7, 1, a, b, c, d);
    const unsigned a71 = a;
    r.avx2 = b7 >> 5 & 1;
    if (zmm) {
        r.avx512f = b7 >> 16 & 1;
        r.avx512bw = b7 >> 30 & 1;
        r.avx512vl = b7 >> 31 & 1;
        r.avx512vnni = c7 >> 11 & 1;
    }
    r.avxvnni = a71 >> 4 & 1;
    r.avx512bf16 = r.avx512f && (a71 >> 5 & 1);
    if (tiles && (d7 >> 24 & 1)) {
        bool ok = false;
#ifdef __linux__
        ok = syscall(SYS_arch_prctl, 0x1023 /* ARCH_REQ_XCOMP_PERM */, 18 /* XTILEDATA */) == 0;
#endif
        r.amx_tile = ok;
        r.amx_bf16 = ok && (d7 >> 22 & 1);
        r.amx_int8 = ok && (d7 >> 25 & 1);
    }
    return r;
}
#elif defined(KERN_ARM) && defined(__linux__)
inline Isa detect_isa() {
    Isa r;
    const unsigned long hw = getauxval(AT_HWCAP), hw2 = getauxval(AT_HWCAP2);
    r.dotprod = hw >> 20 & 1;  // HWCAP_ASIMDDP
    r.i8mm = hw2 >> 13 & 1;    // HWCAP2_I8MM
    r.sve2 = hw2 >> 1 & 1;     // HWCAP2_SVE2
    return r;
}
#elif defined(KERN_ARM) && defined(__APPLE__)
inline Isa detect_isa() {
    auto has = [](const char *name) { int v = 0; size_t n = sizeof v; return sysctlbyname(name, &v, &n, nullptr, 0) == 0 && v; };
    Isa r;
    r.dotprod = has("hw.optional.arm.FEAT_DotProd");
    r.i8mm = has("hw.optional.arm.FEAT_I8MM");
    return r;
}
#else
inline Isa detect_isa() { return Isa(); }
#endif

inline Isa &isa_state() {
    static Isa r = detect_isa();
    return r;
}
inline const Isa &isa() { return isa_state(); }

// Use at most this instruction set (checks, comparisons): "scalar", "avx2", "avxvnni" (AVX2 + AVX-VNNI), "avx512"
// (AVX-512 with VNNI, no BF16 / AMX), "avx512bf16" (+ BF16, no AMX), "all". Call before the first kernel runs.
// Returns false for an unknown name.
inline bool isa_limit(const char *level) {
    Isa &i = isa_state();
    const std::string l = level;
    const int lv = l == "scalar" ? 0 : l == "avx2" ? 1 : l == "avxvnni" ? 2 : l == "avx512" ? 3 : l == "avx512bf16" ? 4
                 : l == "all" ? 5 : -1;
    if (lv < 0) return false;
    if (lv < 5) i.amx_tile = i.amx_int8 = i.amx_bf16 = false;
    if (lv < 4) i.avx512bf16 = false;
    if (lv < 3) i.avx512f = i.avx512bw = i.avx512vl = i.avx512vnni = false;
    if (lv != 2 && lv < 3) i.avxvnni = false;
    if (lv < 1) i.avx2 = i.fma = i.f16c = false;
    return true;
}

// The kernel variants this CPU runs (the best one is used): AVX2 needs FMA and F16C too; AVX-512 means F+BW+VL+VNNI.
inline bool has_avx2() { const Isa &i = isa(); return i.avx2 && i.fma && i.f16c; }
inline bool has_avx512() { const Isa &i = isa(); return has_avx2() && i.avx512f && i.avx512bw && i.avx512vl && i.avx512vnni; }
inline bool has_avxvnni() { return has_avx2() && isa().avxvnni; }
inline bool has_vnni256() { const Isa &i = isa(); return has_avx2() && i.avx512f && i.avx512vl && i.avx512vnni; }
inline bool has_dotprod() { return isa().dotprod; }
inline bool has_i8mm() { return isa().dotprod && isa().i8mm; }

}  // namespace kern
