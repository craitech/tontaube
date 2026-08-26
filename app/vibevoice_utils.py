"""VibeVoice acoustic tokenizer utilities for audio postprocessing."""
import asyncio
import os

import torch
from VibeVoice.vibevoice.modular.modular_vibevoice_tokenizer import (
    VibeVoiceAcousticTokenizerModel,
    VibeVoiceTokenizerStreamingCache,
)
from app.logging_utils import get_logger

logger = get_logger("vibevoice")


def load_vibevoice_tokenizer(model_path: str, device: str | None = None):
    """Load VibeVoice acoustic tokenizer model onto the specified device."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    # Zen 3 / pre-Sapphire-Rapids CPUs have no native bf16 — fp32 wins on CPU.
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    tokenizer = VibeVoiceAcousticTokenizerModel.from_pretrained(
        model_path, torch_dtype=dtype,
    )
    tokenizer = tokenizer.to(device).eval()
    tokenizer._tontaube_dtype = dtype

    # Streaming and short-form requests naturally produce many different
    # sequence lengths. Fixed-shape torch.compile recompiles for each new
    # length, so eager is the safe default. Compilation remains available for
    # fixed-shape benchmarks.
    if os.getenv("COMPILE_VIBEVOICE", "0") == "1":
        logger.info("Compiling VibeVoice encode/decode")
        tokenizer.encode = torch.compile(tokenizer.encode, dynamic=False)
        tokenizer.decode = torch.compile(tokenizer.decode, dynamic=False)
    else:
        logger.debug("Using eager VibeVoice encode/decode")

    logger.info("Warming up VibeVoice encode/decode")
    _warmup_vibevoice(tokenizer, device)
    logger.info("VibeVoice warmup complete")

    return tokenizer


def _warmup_vibevoice(tokenizer, device: str):
    """Run a tiny dummy encode/decode to prime dynamo + Inductor caches.

    Uses a 1-second random input to initialize CUDA kernels and, when the
    opt-in compile mode is enabled, prime Dynamo/Inductor. Random input avoids
    tracing a silence-specific path.
    """
    sample_rate = 24000
    dtype = _vv_dtype(tokenizer)
    # 1s of random audio (small enough to compile fast, not zero so all branches fire).
    dummy_wav = torch.randn(1, 1, sample_rate, dtype=dtype, device=device)
    with torch.no_grad():
        frames = tokenizer.encode(dummy_wav).mean  # [1, ~7-8, D]
        cache = VibeVoiceTokenizerStreamingCache()
        sample_indices = torch.tensor([0], device=device)
        tokenizer.decode(frames, cache=cache, sample_indices=sample_indices, use_cache=True)


def _vv_dtype(vv_tokenizer) -> torch.dtype:
    return getattr(vv_tokenizer, "_tontaube_dtype", torch.bfloat16)


def vv_encode(vv_tokenizer, audio_tensor: torch.Tensor) -> torch.Tensor:
    """Non-causal (bidirectional) encode: audio → VV latent frames [1, T, D]."""
    wav = audio_tensor
    while wav.dim() < 3:
        wav = wav.unsqueeze(0)
    wav = wav.to(vv_tokenizer.device, dtype=_vv_dtype(vv_tokenizer))
    return vv_tokenizer.encode(wav).mean  # [1, T, D]


def vv_decode_streaming(
    vv_tokenizer,
    frames: torch.Tensor,
    cache=None,
) -> tuple[torch.Tensor, object]:
    """Causal streaming decode: VV latent frames → audio.

    Pass the same cache across successive calls to extend seamlessly.
    Returns (audio [1, 1, T_audio], updated cache).
    """
    if cache is None:
        cache = VibeVoiceTokenizerStreamingCache()
    if frames.shape[1] == 0:
        return torch.zeros(1, 1, 0), cache
    frames = frames.to(vv_tokenizer.device, dtype=_vv_dtype(vv_tokenizer))
    sample_indices = torch.tensor([0], device=vv_tokenizer.device)
    audio = vv_tokenizer.decode(frames, cache=cache, sample_indices=sample_indices, use_cache=True)
    return audio.cpu().to(torch.float32), cache


class VVDecodeEngine:
    """Batched causal streaming decode engine for VibeVoice.

    Encode is done by the caller via vv_encode() (fast bidirectional pass,
    no benefit to batching). Only the causal streaming decode is centralised.

    Decode uses a staircase strategy so no padding is ever needed:
      Given frame counts [2, 2, 4, 3, 7]:
        Step 1: decode 2 for all 5  → remaining [-, -, 2, 1, 5]
        Step 2: decode 1 for 3 left → remaining [-, -, 1, -, 4]
        Step 3: decode 1 for 2 left → remaining [-, -, -, -, 3]
        Step 4: decode 3 for 1 left → done
    No padding token ever touches the cache.

    A single shared VibeVoiceTokenizerStreamingCache stores state for all
    active streams, keyed by a per-stream slot index (see alloc_slot).
    """

    def __init__(
        self,
        vv_tokenizer,
        batch_size: int = 8,
        max_queue_size: int = 64,
    ):
        self.vv_tokenizer = vv_tokenizer
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self._shared_cache = VibeVoiceTokenizerStreamingCache()
        self._next_slot: int = 0
        self._free_slots: list[int] = []
        self._queue: asyncio.Queue | None = None
        self._worker_task: asyncio.Task | None = None

    def start(self):
        """Call once from an async context after the event loop is running."""
        self._queue = asyncio.Queue(maxsize=self.max_queue_size)
        self._worker_task = asyncio.create_task(self._worker())

    def stop(self):
        if self._worker_task:
            self._worker_task.cancel()
            self._worker_task = None

    def alloc_slot(self) -> int:
        """Allocate a cache slot, reusing freed slots to prevent index growth."""
        if self._free_slots:
            return self._free_slots.pop()
        slot = self._next_slot
        self._next_slot += 1
        return slot

    def free_slot(self, slot: int):
        """Release a slot's cache state when a stream ends."""
        self._shared_cache.clear(sample_indices=torch.tensor([slot]))
        self._free_slots.append(slot)

    async def decode(self, frames: torch.Tensor, slot: int) -> torch.Tensor:
        """Decode VV latent frames for the given slot.

        frames: [1, T, D] — only the new frames to decode (already-decoded
                            frames must be excluded by the caller).
        slot:   integer from alloc_slot() identifying this stream.
        Returns audio [1, 1, T_audio].
        """
        if frames.shape[1] == 0:
            return torch.zeros(1, 1, 0)
        if self._queue is None:
            raise RuntimeError("VibeVoice decode engine is not running")
        future = asyncio.get_running_loop().create_future()
        try:
            self._queue.put_nowait((frames, slot, future))
        except asyncio.QueueFull as exc:
            future.cancel()
            raise RuntimeError("VibeVoice decode queue is full") from exc
        try:
            return await future
        except asyncio.CancelledError:
            future.cancel()
            raise

    async def _worker(self):
        while True:
            first = await self._queue.get()
            batch = [first]
            while len(batch) < self.batch_size:
                try:
                    batch.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
            batch = [item for item in batch if not item[2].cancelled()]
            if not batch:
                continue
            try:
                self._process_batch(batch)
            except Exception as e:
                for *_, future in batch:
                    if not future.done():
                        future.set_exception(e)

    def _process_batch(self, batch: list):
        """Staircase batched decode — no padding, no cache corruption."""
        device = self.vv_tokenizer.device

        # active entries: (remaining_frames [1, T_i, D], slot, accumulated_chunks, future)
        active = [
            (frames.to(device, dtype=torch.bfloat16), slot, [], future)
            for frames, slot, future in batch
            if not future.cancelled()
        ]

        while active:
            min_len = min(f.shape[1] for f, _, _, _ in active)
            # All active items have >= min_len frames: stack without padding → [B, min_len, D]
            stacked = torch.cat([f[:, :min_len] for f, _, _, _ in active], dim=0)
            slots = torch.tensor([s for _, s, _, _ in active], device=device)
            with torch.no_grad():
                audio_chunk = self.vv_tokenizer.decode(
                    stacked, cache=self._shared_cache,
                    sample_indices=slots, use_cache=True,
                )  # [B, 1, T_audio_chunk]
            audio_chunk = audio_chunk.cpu().to(torch.float32)

            new_active = []
            for i, (frames, slot, acc, future) in enumerate(active):
                if future.cancelled():
                    continue
                acc.append(audio_chunk[i:i + 1])  # [1, 1, T_chunk]
                remaining = frames[:, min_len:]
                if remaining.shape[1] > 0:
                    new_active.append((remaining, slot, acc, future))
                elif not future.done():
                    future.set_result(torch.cat(acc, dim=-1))  # [1, 1, T_total]
            active = new_active
