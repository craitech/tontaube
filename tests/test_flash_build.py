"""flash/common/build.py: library names, the prebuilt download (checked against SHA256SUMS, tried once) and the
fallback to building, without a network or a compiler."""
import hashlib
import io
import re

import pytest

from flash.common import build


def test_names_carry_platform_and_a_location_free_hash():
    name, units, roots, what = build.text_units()
    lin = build.library_name(name, units, roots, what, ("Linux", "x86_64"))
    win = build.library_name(name, units, roots, what, ("Windows", "x86_64"))
    assert re.fullmatch(r"libflash_text-linux-x86_64-[0-9a-f]{16}\.so", lin)
    assert re.fullmatch(r"libflash_text-windows-x86_64-[0-9a-f]{16}\.dll", win)
    assert build._rel(str(build.ENGINE)) == "./flash/engine"


class FakeRelease:
    """urlopen() for a release folder: {file name: bytes}; records the URLs asked for."""

    def __init__(self, files):
        self.files, self.asked = files, []

    def __call__(self, url, timeout=None):
        self.asked.append(url)
        name = url.rsplit("/", 1)[1]
        if name not in self.files:
            raise OSError("404")
        return io.BytesIO(self.files[name])


def _release(monkeypatch, lib_name, data, digest=None):
    sums = f"{digest or hashlib.sha256(data).hexdigest()}  {lib_name}\n".encode()
    fake = FakeRelease({"SHA256SUMS": sums, lib_name: data})
    monkeypatch.setattr(build.urllib.request, "urlopen", fake)
    monkeypatch.setattr(build, "version", lambda: "9.9.9")
    return fake


def test_prebuilt_mismatch_is_not_used(monkeypatch, tmp_path):
    lib = tmp_path / "libflash_text-linux-x86_64-0123456789abcdef.so"
    fake = _release(monkeypatch, lib.name, b"library bytes", digest="0" * 64)
    assert not build.prebuilt(lib, "text front end")
    assert not lib.exists() and fake.asked[0].endswith("/v9.9.9/SHA256SUMS")


def test_prebuilt_that_does_not_load_is_removed(monkeypatch, tmp_path):
    lib = tmp_path / "libflash_text-linux-x86_64-0123456789abcdef.so"
    _release(monkeypatch, lib.name, b"not a shared library")
    assert not build.prebuilt(lib, "text front end")
    assert not lib.exists()


def test_prebuilt_not_in_the_release(monkeypatch, tmp_path):
    lib = tmp_path / "libflash_text-linux-x86_64-0123456789abcdef.so"
    fake = _release(monkeypatch, "libflash_text-linux-x86_64-ffffffffffffffff.so", b"x")
    assert not build.prebuilt(lib, "text front end")
    assert len(fake.asked) == 1                     # only SHA256SUMS


def test_cache_then_prebuilt_then_compiler(monkeypatch, tmp_path):
    monkeypatch.setenv("TONTAUBE_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("FLASH_BUILD_FROM_SOURCE", raising=False)
    calls = []

    def fake_prebuilt(lib, what):
        calls.append("prebuilt")
        lib.parent.mkdir(parents=True, exist_ok=True)
        lib.write_bytes(b"x")
        return True
    monkeypatch.setattr(build, "prebuilt", fake_prebuilt)
    monkeypatch.setattr(build, "build", lambda *a: calls.append("build"))
    lib = build.library(*build.text_units())
    assert lib.exists() and calls == ["prebuilt"]
    assert build.library(*build.text_units()) == lib and calls == ["prebuilt"]   # cached


@pytest.mark.parametrize("system, needs", [("Linux", "apt install g++"), ("Windows", "LLVM MinGW"), ("Darwin", "xcode-select")])
def test_forced_source_build_without_compiler(monkeypatch, tmp_path, system, needs):
    monkeypatch.setenv("TONTAUBE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("FLASH_BUILD_FROM_SOURCE", "1")
    monkeypatch.setattr(build, "prebuilt", lambda *a: pytest.fail("no download when building from source is forced"))
    monkeypatch.setattr(build, "host", lambda: (system, "x86_64"))
    monkeypatch.setattr(build.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match=needs):
        build.library(*build.text_units())
