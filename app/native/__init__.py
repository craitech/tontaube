"""x86-64 kernels (AVX2; AVX-512 / AVX-VNNI where the CPU has them), compiled on first use and cached.

lm_kernels.cpp serves the lite codebook models on the CPU (app/lite/cpu) through kern/ (shared with the
Flash engine), vibevoice_kernels.cpp the VibeVoice tokenizer (app/codec/vibevoice_cpu.py); both are one
extension module. Building needs a C++ compiler; the result is cached in TONTAUBE_CACHE_DIR (default
~/.cache/tontaube).
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

_SOURCES = ("module.cpp", "lm_kernels.cpp", "vibevoice_kernels.cpp")
_EXT = None
_LOCK = threading.Lock()


def kernels():
    """The compiled extension (built once per process, cached across runs)."""
    global _EXT
    with _LOCK:
        if _EXT is None:
            from torch.utils.cpp_extension import load

            here = Path(__file__).resolve().parent
            root = Path(os.getenv("TONTAUBE_CACHE_DIR", Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"))
                                  / "tontaube")).expanduser()
            build = root / "native-kernels"
            build.mkdir(parents=True, exist_ok=True)
            _EXT = load(
                name="tontaube_native",
                sources=[str(here / s) for s in _SOURCES],
                extra_include_paths=[str(here)],
                # at::parallel_for is an inline OpenMP region: without -fopenmp it runs on one thread.
                extra_cflags=["-O3", "-mavx2", "-mfma", "-mf16c", "-fopenmp"],
                extra_ldflags=["-fopenmp"],
                build_directory=str(build),
                verbose=False,
            )
    return _EXT
