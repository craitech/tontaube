"""Regressions for VibeVoice layout and comparison-free output-sanity retries."""
import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch

from app.config import ServerConfig
from app.inference import TTSEngine
from app.vibevoice_utils import VVDecodeEngine, _warmup_vibevoice, vv_decode_streaming


def upstream_decode():
    # conftest mocks heavy model imports. Exercise the real vendored layout
    # logic, with only its neural decoder replaced by a cheap value probe.
    path = Path(__file__).resolve().parents[1] / "VibeVoice/vibevoice/modular/modular_vibevoice_tokenizer.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "VibeVoiceAcousticTokenizerModel")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "decode")
    env = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), env)
    return env["decode"]


class ProbeTokenizer:
    device = torch.device("cpu")
    _tontaube_dtype = torch.float32
    config = SimpleNamespace(vae_dim=64)

    def __init__(self):
        self.frames = torch.randn(1, 64, 64, generator=torch.Generator().manual_seed(123))
        self.decode_impl = upstream_decode()
        self.calls = []

    def encode(self, audio):
        return SimpleNamespace(mean=self.frames)

    def decoder(self, latents, **kwargs):
        self.calls.append((latents.clone(), kwargs))
        # One sample per frame is sufficient to catch a time/channel swap.
        return latents[:, :1, :]

    def decode(self, latents, **kwargs):
        return self.decode_impl(self, latents, **kwargs)


@pytest.mark.parametrize("length", [1, 63, 64, 65, 128, 225])
def test_single_decode_passes_explicit_channel_first_values(length):
    model = ProbeTokenizer()
    frames = torch.randn(1, length, 64, generator=torch.Generator().manual_seed(length))
    cache = object()
    audio, returned_cache = vv_decode_streaming(model, frames, cache)
    torch.testing.assert_close(model.calls[-1][0], frames.transpose(1, 2))
    torch.testing.assert_close(audio, frames[:, :, 0].unsqueeze(1))
    assert returned_cache is cache
    assert model.calls[-1][1]["cache"] is cache
    assert model.calls[-1][1]["use_cache"] is True


def test_warmup_also_uses_explicit_layout():
    model = ProbeTokenizer()
    _warmup_vibevoice(model, "cpu")
    torch.testing.assert_close(model.calls[0][0], model.frames.transpose(1, 2))


@pytest.mark.asyncio
@pytest.mark.parametrize("lengths", [(64, 128), (65, 129), (63, 128)])
async def test_batched_staircase_keeps_layout_when_any_step_has_64_frames(lengths):
    model = ProbeTokenizer()
    engine = VVDecodeEngine(model)
    inputs = [torch.randn(1, n, 64, generator=torch.Generator().manual_seed(n))
              for n in lengths]
    futures = [asyncio.get_running_loop().create_future() for _ in lengths]
    engine._process_batch([(frames, slot, future)
                           for slot, (frames, future) in enumerate(zip(inputs, futures))])
    for frames, future in zip(inputs, futures):
        # Existing batched worker converts to bfloat16.
        expected = frames.to(torch.bfloat16).float()[:, :, 0].unsqueeze(1)
        torch.testing.assert_close(future.result(), expected)
    assert all(call[0].shape[1] == 64 for call in model.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("n_tokens", [500, 2500])
@pytest.mark.parametrize("vv_enabled", [False, True])
@pytest.mark.parametrize("level", [0.0, 0.1])
async def test_only_silent_short_audio_triggers_output_retry(n_tokens, vv_enabled, level):
    engine = TTSEngine(ServerConfig())
    engine.vv_tokenizer = object()
    engine.dc_inference = object()
    sampling = dict(temperature=0.6, top_p=0.95, top_k=50, seed=123)
    engine._resolve_params = MagicMock(return_value=(sampling, (None, None), "", 0.0, -1))
    engine._resolve_prefix = AsyncMock(return_value=(None, {}))
    engine._n_codebooks = MagicMock(return_value=4)
    engine._generate_semantic = AsyncMock(return_value=([1] * n_tokens, []))
    engine._generate_acoustic = AsyncMock(return_value=[2] * n_tokens)
    engine._append_trailing_silence = MagicMock()
    audio = torch.full((1, 1, 2400), level)
    with patch("app.inference.free_memory"), patch(
            "app.inference.decode_chunked_with_context", return_value=audio) as decode:
        result, codes = await engine.generate("Test text.", temperature=0.6,
                                               vibevoice_postprocess=vv_enabled)
    assert result is audio
    assert codes == [[1] * n_tokens] + [[2] * n_tokens] * 3
    attempts = 5 if level == 0.0 and n_tokens / 12.5 < 60 else 1
    assert engine._generate_semantic.await_count == attempts
    assert engine._generate_acoustic.await_count == 3 * attempts
    assert [c.kwargs["sampling"]["temperature"] for c in
            engine._generate_semantic.await_args_list] == pytest.approx(
        [0.6 + i * 0.1 for i in range(attempts)])
    assert sampling["temperature"] == 0.6
    assert decode.call_count == attempts
    assert decode.call_args.args[1] is (engine.vv_tokenizer if vv_enabled else None)
    assert not decode.call_args.kwargs.get("return_pre_vv", False)


@pytest.mark.parametrize("audio,reason", [
    (torch.empty(1, 1, 0), "empty_audio"),
    (torch.tensor([float("nan")]), "non_finite_audio"),
    (torch.tensor([float("inf")]), "non_finite_audio"),
    (torch.zeros(1, 1, 100), "silent"),
])
def test_invalid_and_silent_output_remains_retriable(audio, reason):
    assert TTSEngine._check_audio_quality(audio).startswith(reason)


def test_valid_output_needs_no_raw_waveform_comparison():
    assert TTSEngine._check_audio_quality(torch.full((1, 1, 100), 0.1)) is None
