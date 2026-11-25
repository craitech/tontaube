"""Text front end: the rule verbalizer (written English -> the words a narrator says) and the chunker, C++ from
flash/text/, called through ctypes. Prebuilt or built on first use (build.py)."""
import ctypes

import numpy as np

from .build import library, text_units


def text_library():
    """The text front end's library: from the cache, prebuilt or built on first use."""
    return library(*text_units())


class FrontEnd:
    def __init__(self, lib=None, max_chars=1 << 18, max_chunks=4096):
        """lib: a library with the front end's C API (default: the text front end alone)."""
        self.lib = ctypes.CDLL(str(lib or text_library()))
        self.out = np.zeros(max_chars, np.int32)
        self.lens = np.zeros(max_chunks, np.int32)
        self.times = np.zeros(2, np.float64)
        P = ctypes.POINTER
        self.lib.flash_prepare.argtypes = [ctypes.c_char_p, ctypes.c_int64, ctypes.c_int, ctypes.c_int, P(ctypes.c_int32), ctypes.c_int64,
                                           P(ctypes.c_int32), ctypes.c_int, P(ctypes.c_double)]
        self.lib.flash_verbalize.argtypes = [ctypes.c_char_p, ctypes.c_int64, ctypes.c_char_p, ctypes.c_int64]
        self.lib.flash_verbalize.restype = ctypes.c_int64
        self._args = (self.out.ctypes.data_as(P(ctypes.c_int32)), len(self.out), self.lens.ctypes.data_as(P(ctypes.c_int32)), max_chunks,
                      self.times.ctypes.data_as(P(ctypes.c_double)))
        self.chunks("Warm up at 10:30 on 5/9, $5.")            # the verbalizer builds its tables on the first call

    def chunks(self, text: str, verbalize: bool = True, first_max: int = 0) -> list[np.ndarray]:
        """Chunks of at most 350 characters as arrays of code points. self.times: seconds in verbalizer and chunker.
        first_max > 0: a first chunk of at most that many characters (faster first audio for long texts)."""
        b = text.encode("utf-8")
        n = self.lib.flash_prepare(b, len(b), int(verbalize), first_max, *self._args)
        if n < 0:
            raise ValueError("text too long for the front end's buffers")
        ends = np.cumsum(self.lens[:n])
        return [self.out[e - k:e].copy() for e, k in zip(ends, self.lens[:n])]

    def verbalize(self, text: str) -> str:
        """The spoken form of a written text."""
        b = text.encode("utf-8")
        buf = ctypes.create_string_buffer(8 * len(b) + 4096)
        n = self.lib.flash_verbalize(b, len(b), buf, len(buf))
        return buf.raw[:n].decode("utf-8")
