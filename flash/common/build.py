"""The C++ parts of Flash (text front end, CPU engine) as shared libraries, cached under a name with the platform and
a hash of their sources and flags (a change gives a new name). A library missing from the cache is downloaded
prebuilt from the GitHub release of this version, checked against the release's SHA256SUMS and tried once; otherwise
(a modified checkout, no network, FLASH_BUILD_FROM_SOURCE=1) it is built with the system's C++ compiler (C++17):
Linux g++, macOS clang++ (Xcode command line tools), Windows clang++ from LLVM MinGW. The engine picks the CPU's
instruction set at run time, so a prebuilt library is as fast as one built here.

Cache folder: $TONTAUBE_CACHE_DIR/flash, else $XDG_CACHE_HOME/tontaube/flash (default ~/.cache/tontaube/flash). It
must allow loading libraries (not a noexec mount).

The release's libraries (.github/workflows/flash-native.yml; the same flags, from this file), and a check of them
as a user gets them (text front end, one sentence spoken with the CPU weights in WEIGHTS/web/):
    python -m flash.common.build OUT_DIR [--system Windows] [--check WEIGHTS]
"""
import ctypes
import hashlib
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FLASH = Path(__file__).resolve().parent.parent
TEXT = FLASH / "text"
ENGINE = FLASH / "engine"
KERN = FLASH.parent / "app" / "native"          # the x86 kernels shared with TontaubeV1.1 (kern/*.h)
TEXT_INCLUDE = ["-I", str(TEXT / "include"), "-I", str(TEXT / "src"), "-I", str(TEXT)]
RELEASES = "https://github.com/craitech/tontaube/releases/download"
_LOCK = threading.Lock()
log = logging.getLogger("flash.build")

# Per system: compilers (the first one found; target: what `cxx -dumpmachine` must contain), flags of every unit, link
# flags, the library's suffix, and what to install. The C++ runtime is linked in (Linux: any glibc from the one it was
# built with; Windows: the DLL needs nothing beyond Windows itself).
SYSTEMS = {
    "Linux": dict(tag="linux", cxx=["g++", "c++"], target="", cflags=["-fPIC", "-pthread"],
                  ldflags=["-shared", "-pthread", "-static-libstdc++", "-static-libgcc"], suffix=".so",
                  install="Install one, e.g. `sudo apt install g++`, and run again."),
    "Darwin": dict(tag="macos", cxx=["clang++"], target="apple", cflags=["-fPIC", "-pthread"], ldflags=["-dynamiclib", "-pthread"],
                   suffix=".dylib", install="Install Apple's command line tools with `xcode-select --install` and run again."),
    "Windows": dict(tag="windows", cxx=["x86_64-w64-mingw32-clang++", "clang++"], target="w64", cflags=["-D_USE_MATH_DEFINES"],
                    ldflags=["-shared", "-static", "-lsynchronization"], suffix=".dll",
                    install="Install LLVM MinGW (`winget install MartinStorsjo.LLVM-MinGW.UCRT`, or the ucrt-x86_64 zip from "
                            "https://github.com/mstorsjo/llvm-mingw/releases with its bin folder on the PATH) and run again."),
}
MACHINES = {"amd64": "x86_64", "x64": "x86_64", "arm64": "aarch64"}


def host() -> tuple:
    """(system, machine) of this computer, e.g. ("Linux", "x86_64"), ("Windows", "x86_64"), ("Darwin", "aarch64")."""
    m = platform.machine().lower()
    return platform.system(), MACHINES.get(m, m)


def cache_dir() -> Path:
    root = os.environ.get("TONTAUBE_CACHE_DIR")
    base = Path(root).expanduser() if root else Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "tontaube"
    return base / "flash"


def text_sources() -> list:
    """The rule verbalizer and the chunker with their C API (text_front.cpp)."""
    return sorted([*TEXT.glob("src/core/**/*.cpp"), *TEXT.glob("src/rules/**/*.cpp"), TEXT / "text_front.cpp"])


def engine_units():
    """(name, units, roots, what) of the CPU engine with the text front end (one library)."""
    flags = ["-std=c++17", "-O3", "-I", str(ENGINE), "-I", str(KERN), *TEXT_INCLUDE]
    text = ["-std=c++17", "-O2", *TEXT_INCLUDE]
    return "flash_cpu", [(ENGINE / "tts_native.cpp", flags), *[(s, text) for s in text_sources()]], [ENGINE, KERN / "kern", TEXT], "CPU engine"


def text_units():
    """(name, units, roots, what) of the text front end alone (the CPU engine's library contains it too)."""
    flags = ["-std=c++17", "-O2", *TEXT_INCLUDE]
    return "flash_text", [(s, flags) for s in text_sources()], [TEXT], "text front end"


def _rel(s: str) -> str:
    """A path or flag with the repository's folder taken out (the hash is the same wherever the repository lies)."""
    return s.replace(str(FLASH.parent), ".").replace("\\", "/")


def library_name(name: str, units: list, roots: list, what: str, system=None) -> str:
    """lib<name>-<system>-<machine>-<hash><suffix>: the hash covers every source file under `roots` (line ends as LF),
    the units with their flags, and the system and machine (host() unless given)."""
    system = system or host()
    files = sorted({p for r in roots for p in Path(r).rglob("*") if p.suffix in (".cpp", ".h", ".inc")})
    if not files:
        raise RuntimeError(f"the sources of the {what} are missing ({', '.join(map(str, roots))})")
    sysd = SYSTEMS.get(system[0], SYSTEMS["Linux"])
    h = hashlib.sha256("/".join(system).encode() + b"\n")
    for p in files:
        h.update(p.relative_to(FLASH.parent).as_posix().encode() + b"\0" + p.read_bytes().replace(b"\r\n", b"\n"))
    for src, flags in units:
        h.update("\0".join(_rel(f) for f in [str(src), *flags, *sysd["cflags"]]).encode() + b"\n")
    h.update("\0".join(sysd["ldflags"]).encode())
    return f"lib{name}-{sysd['tag']}-{system[1]}-{h.hexdigest()[:16]}{sysd['suffix']}"


def compiler(sysd: dict):
    """The first of the system's compilers on the PATH that builds for it, or None."""
    for c in sysd["cxx"]:
        path = shutil.which(c)
        if path and sysd["target"]:
            r = subprocess.run([path, "-dumpmachine"], capture_output=True, text=True)
            if sysd["target"] not in r.stdout:
                continue
        if path:
            return path
    return None


def build(lib: Path, units: list, what: str, sysd: dict, cxx: str):
    """Compiles the units in parallel and links them into lib (written atomically)."""
    lib.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=lib.parent) as tmp:
        def compile_one(k_unit):
            k, (src, flags) = k_unit
            obj = Path(tmp) / f"{k}.o"
            r = subprocess.run([cxx, *flags, *sysd["cflags"], "-c", str(src), "-o", str(obj)], capture_output=True, text=True)
            if r.returncode:
                raise RuntimeError(f"compiling {src} failed:\n{r.stderr[-4000:]}")
            return str(obj)
        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as ex:
            objs = list(ex.map(compile_one, enumerate(units)))
        part = Path(tmp) / lib.name
        r = subprocess.run([cxx, *sysd["ldflags"], "-o", str(part), *objs], capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"linking the {what} failed:\n{r.stderr[-4000:]}")
        _place(part, lib)


def _place(part: Path, lib: Path):
    try:
        os.replace(part, lib)                        # atomic: a second process sees the whole file or none
    except OSError:                                  # Windows: another process has loaded the same library
        if not lib.exists():
            raise


def version():
    """This checkout's version (pyproject.toml, else the installed package), e.g. "1.2.0"; None if unknown."""
    try:
        import tomllib
        return tomllib.loads((FLASH.parent / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    except Exception:                                # noqa: BLE001
        pass
    try:
        from importlib.metadata import version as package_version
        return package_version("tontaube-inference")
    except Exception:                                # noqa: BLE001
        return None


def prebuilt(lib: Path, what: str) -> bool:
    """The library from the GitHub release of this version into the cache, if the release lists it in SHA256SUMS,
    the download matches and it loads; False otherwise (other sources than the release's, no network)."""
    v = version()
    if not v:
        return False
    url = f"{RELEASES}/v{v}"
    try:
        with urllib.request.urlopen(f"{url}/SHA256SUMS", timeout=20) as r:
            sums = dict(reversed(line.split(None, 1)) for line in r.read().decode().splitlines() if line.strip())
        want = sums.get(lib.name) or sums.get("*" + lib.name)
        if not want:
            return False
        log.info("downloading the prebuilt %s from %s/%s", what, url, lib.name)
        with urllib.request.urlopen(f"{url}/{lib.name}", timeout=60) as r:
            data = r.read()
    except Exception:                                # noqa: BLE001  (no release, no network: build it instead)
        return False
    if hashlib.sha256(data).hexdigest() != want.strip():
        log.warning("the prebuilt %s from %s does not match its SHA256SUMS entry; building it from source", what, url)
        return False
    lib.parent.mkdir(parents=True, exist_ok=True)
    part = lib.with_name(f"{lib.name}.{os.getpid()}.part")
    part.write_bytes(data)
    _place(part, lib)
    try:
        ctypes.CDLL(str(lib))                        # the caller loads the same file (the same module)
    except OSError as e:                             # e.g. an older system than the release's libraries support
        lib.unlink(missing_ok=True)
        log.warning("the prebuilt %s does not load on this system (%s); building it from source", what, e)
        return False
    return True


def library(name: str, units: list, roots: list, what: str) -> Path:
    """The path of the library (library_name), from the cache, prebuilt from the release or built. units: (source,
    compiler flags) pairs, compiled in parallel and linked into one shared library. what: its name in messages."""
    lib = cache_dir() / library_name(name, units, roots, what)
    with _LOCK:                                      # one build per process; other processes may build in parallel
        if lib.exists() or (os.environ.get("FLASH_BUILD_FROM_SOURCE") != "1" and prebuilt(lib, what)):
            return lib
        sysd = SYSTEMS.get(host()[0], SYSTEMS["Linux"])
        cxx = compiler(sysd)
        if cxx is None:
            raise RuntimeError(f"the {what} has no prebuilt library that fits this checkout (the GitHub release v{version()} has "
                               f"none for these sources, or it could not be reached), and building it needs a C++17 compiler "
                               f"({' or '.join(sysd['cxx'])}) on the PATH; none was found. {sysd['install']}")
        log.info("building the %s with %s (once)", what, cxx)
        build(lib, units, what, sysd, cxx)
    return lib


def check(out: Path, weights: str):
    """The libraries in `out` from an empty cache, as a user gets them (nothing may be built or downloaded): the text
    front end reads a sentence, the CPU engine speaks it."""
    import numpy as np
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        os.environ["TONTAUBE_CACHE_DIR"] = tmp
        cache = cache_dir()
        cache.mkdir(parents=True)
        for f in out.iterdir():
            shutil.copy(f, cache / f.name)
        before = sorted(os.listdir(cache))
        from flash.common.frontend import FrontEnd, text_library
        from flash.cpu.engine import Engine
        text = "Dr. Weber's invoice from 5/9/2025 lists 3 items."
        print("text front end:", FrontEnd(text_library()).verbalize(text))
        eng = Engine(threads=2, weights=weights)
        x = eng.speak(text, seed=0)
        eng.close()
        print(f"CPU engine: {len(x) / eng.sample_rate:.1f} s of audio, kernels {eng.info['kernels']}")
        if not (len(x) > 24000 and np.isfinite(x).all() and np.abs(x).max() > 0.01):
            sys.exit("the CPU engine's audio is not right")
        if sorted(os.listdir(cache)) != before:
            sys.exit(f"a library was built or downloaded during the check: {sorted(set(os.listdir(cache)) - set(before))}")


if __name__ == "__main__":                           # the release's libraries
    import argparse
    ap = argparse.ArgumentParser(description="Build Flash's libraries (the CPU engine and the text front end) into a folder.")
    ap.add_argument("out", help="output folder")
    ap.add_argument("--system", default=host()[0], choices=sorted(SYSTEMS), help="target system (default: this one; another one cross-compiles)")
    ap.add_argument("--machine", default=host()[1], help="target machine (default: this one)")
    ap.add_argument("--check", metavar="WEIGHTS", help="then check the libraries (this system only): WEIGHTS/web/ holds the CPU weights")
    a = ap.parse_args()
    sysd = SYSTEMS[a.system]
    cxx = compiler(sysd)
    if cxx is None:
        sys.exit(f"no compiler for {a.system} on the PATH ({', '.join(sysd['cxx'])})")
    for name, units, roots, what in (engine_units(), text_units()):
        lib = Path(a.out) / library_name(name, units, roots, what, (a.system, a.machine))
        build(lib, units, what, sysd, cxx)
        print(lib)
    if a.check:
        check(Path(a.out), a.check)
