"""The float x86 kernels (app/native/kern: act.h, gemm_f32.h, vv.h, detect.h) against scalar references."""
import platform
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _has_avx2() -> bool:
    try:
        flags = Path("/proc/cpuinfo").read_text()
    except OSError:
        return False
    return platform.machine() == "x86_64" and all(f in flags for f in (" avx2", " fma", " f16c"))


@pytest.mark.skipif(not _has_avx2(), reason="needs an x86-64 CPU with AVX2/FMA/F16C")
@pytest.mark.parametrize("cxx", ["g++", "clang++"])
def test_float_kernels(cxx, tmp_path):
    if not shutil.which(cxx):
        pytest.skip(f"{cxx} not found")
    exe = tmp_path / "kern_float_test"
    subprocess.run([cxx, "-O2", "-std=c++17", "-I", str(ROOT / "app/native"), "-I", str(ROOT / "flash/engine"),
                    str(ROOT / "tests/native/kern_float_test.cpp"), "-o", str(exe)], check=True)
    r = subprocess.run([str(exe)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
