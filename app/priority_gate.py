"""Priority gate: pauses low-priority TTS requests between vLLM generation
calls while any high-priority request is active.

Each `_generate_with_engine` call in low-priority context awaits `wait_if_low`
before submitting to vLLM. In-flight vLLM calls are never aborted — they are
short (a few seconds) and allowed to finish. Only the *next* call blocks.
"""
import asyncio
from contextvars import ContextVar
from enum import IntEnum


class Priority(IntEnum):
    LOW = 0
    HIGH = 10


current_priority: ContextVar[Priority] = ContextVar("current_priority", default=Priority.LOW)


class PriorityGate:
    def __init__(self, enabled: bool = True) -> None:
        self._enabled = enabled
        self._high_count = 0
        self._idle = asyncio.Event()
        self._idle.set()
        self._lock = asyncio.Lock()

    @property
    def high_active(self) -> int:
        return self._high_count

    async def _enter_high(self) -> None:
        async with self._lock:
            self._high_count += 1
            self._idle.clear()

    async def _exit_high(self) -> None:
        async with self._lock:
            self._high_count -= 1
            if self._high_count <= 0:
                self._high_count = 0
                self._idle.set()

    async def wait_if_low(self) -> None:
        """Low-priority callers await this before each vLLM generation call."""
        if not self._enabled:
            return
        if self._idle.is_set():
            return
        await self._idle.wait()

    def high(self) -> "_HighScope":
        return _HighScope(self)


class _HighScope:
    def __init__(self, gate: PriorityGate) -> None:
        self._gate = gate
        self._token = None

    async def __aenter__(self) -> "_HighScope":
        self._token = current_priority.set(Priority.HIGH)
        await self._gate._enter_high()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            await self._gate._exit_high()
        finally:
            if self._token is not None:
                current_priority.reset(self._token)
                self._token = None


_gate: PriorityGate | None = None


def get_gate() -> PriorityGate:
    """Return the process-wide PriorityGate, creating it on first use."""
    global _gate
    if _gate is None:
        from app.config import ServerConfig
        _gate = PriorityGate(enabled=ServerConfig().priority_gate_enabled)
    return _gate
