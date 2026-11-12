"""Request and output types shared by the GPU and CPU engines (vLLM-compatible outputs)."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from app.lite.sampling import SamplingParams, SeqSampling


@dataclass
class Completion:
    token_ids: list[int]
    finish_reason: str


@dataclass
class Output:
    request_id: str
    outputs: list[Completion]
    finished: bool = True


@dataclass(eq=False)
class Request:
    request_id: str
    prompt: list[int]
    params: SamplingParams
    priority: int
    loop: asyncio.AbstractEventLoop
    future: asyncio.Future
    max_tokens: int = 0
    blocks: list[int] = field(default_factory=list)
    sampling: SeqSampling | None = None
    out: list[int] = field(default_factory=list)
    aborted: bool = False
    arrival: int = 0

    @property
    def n_prompt(self) -> int:
        return len(self.prompt)
