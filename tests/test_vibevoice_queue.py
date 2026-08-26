"""Tests for bounded VibeVoice streaming decode admission."""
import asyncio
from unittest.mock import MagicMock

import pytest
import torch

from app.vibevoice_utils import VVDecodeEngine


@pytest.mark.asyncio
async def test_decode_fails_fast_when_queue_is_full():
    engine = VVDecodeEngine(MagicMock(), max_queue_size=1)
    engine._queue = asyncio.Queue(maxsize=1)
    queued_future = asyncio.get_running_loop().create_future()
    engine._queue.put_nowait((torch.zeros(1, 1, 1), 0, queued_future))

    with pytest.raises(RuntimeError, match="decode queue is full"):
        await engine.decode(torch.zeros(1, 1, 1), slot=1)
