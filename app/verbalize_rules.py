"""The rule verbalizer (written English -> the words a narrator says: numbers, dates, times, money, units,
abbreviations) for TontaubeV1.1 and Turbo. It is TontaubeV1.1 Flash's own (flash/common/frontend.py: C++ from
flash/text/, called through ctypes, prebuilt or built on first use), used as it is: one shared instance, a lock around
its calls. The model verbalizer (app/verbalizer.py) is the other mode; see app.config VERBALIZER_MODES."""
from __future__ import annotations

import threading

_lock = threading.Lock()
_front = None


def front_end():
    """Flash's text front end, loaded (and built, once) on first use; raises if it cannot be (no compiler, no library)."""
    global _front
    with _lock:
        if _front is None:
            from flash.common.frontend import FrontEnd

            _front = FrontEnd()
        return _front


def verbalize(text: str) -> str:
    """The spoken form of `text` by the rules."""
    front = front_end()
    with _lock:
        return front.verbalize(text)
