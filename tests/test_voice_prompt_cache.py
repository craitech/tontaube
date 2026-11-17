"""Focused tests for request-independent voice-prompt caching."""
import math
from collections import OrderedDict
from unittest.mock import MagicMock

import pytest
import torch

from app import utils


def test_cache_stores_full_encoding_and_applies_each_request_cap(tmp_path, monkeypatch):
    voice_path = tmp_path / "voice.wav"
    voice_path.write_bytes(b"placeholder")
    full_semantic = list(range(20))
    full_acoustic = [list(range(100, 120)) for _ in range(3)]
    encode = MagicMock(return_value=(full_semantic, full_acoustic))
    monkeypatch.setattr(utils, "encode_voice_prompt", encode)
    cache = OrderedDict()
    dc_inference = object()

    first = utils.resolve_voice_prompt(
        None, str(voice_path), dc_inference, 60.0, 5, 4,
        cache, 8, (None, None),
    )
    second = utils.resolve_voice_prompt(
        None, str(voice_path), dc_inference, 60.0, 12, 4,
        cache, 8, (None, None),
    )
    third = utils.resolve_voice_prompt(
        None, str(voice_path), dc_inference, 60.0, 3, 4,
        cache, 8, (None, None),
    )

    encode.assert_called_once_with(dc_inference, str(voice_path), 60.0, None, 4)
    assert first == (full_semantic[:5], [codes[:5] for codes in full_acoustic])
    assert second == (full_semantic[:12], [codes[:12] for codes in full_acoustic])
    assert third == (full_semantic[:3], [codes[:3] for codes in full_acoustic])
    assert cache["path:" + str(voice_path) + "|4"] == (full_semantic, full_acoustic)


def test_default_prompt_is_capped_per_request():
    default = (list(range(20)), [list(range(100, 120)) for _ in range(3)])
    result = utils.resolve_voice_prompt(
        None, None, object(), 60.0, 6, 4, OrderedDict(), 8, default,
    )
    assert result == (default[0][:6], [codes[:6] for codes in default[1]])


def test_release_mapped_weights_is_lossless_and_skips_written_mappings(tmp_path):
    import sys

    import pytest
    import torch
    from safetensors.torch import save_file

    from app.lite.weights import LazyWeights

    if not sys.platform.startswith("linux"):
        pytest.skip("uses /proc/self/smaps and madvise")
    ref = {"a": torch.randn(512, 1024), "b": torch.randn(256, 1024)}
    save_file(ref, str(tmp_path / "w.safetensors"))
    w = LazyWeights(tmp_path / "w.safetensors", mmap=True)
    clean = torch.nn.Linear(1, 1)
    clean.weight = torch.nn.Parameter(w.pop("a"), requires_grad=False)
    assert torch.equal(clean.weight, ref["a"])  # pages resident now
    assert utils.release_mapped_weights(clean) >= ref["a"].numel() * 4
    assert torch.equal(clean.weight, ref["a"])  # re-read from the file

    written = torch.nn.Linear(1, 1)
    written.weight = torch.nn.Parameter(w.pop("b"), requires_grad=False)
    written.weight.data[0, 0] = 123.0  # copy-on-write page: must not be dropped
    assert utils.release_mapped_weights(written) == 0
    assert written.weight[0, 0].item() == 123.0


def _tone(amplitude: float, seconds: float = 3.0, silence: float = 0.0) -> torch.Tensor:
    t = torch.arange(int(24000 * seconds)) / 24000
    x = amplitude * torch.sin(2 * math.pi * 220 * t)
    pad = torch.zeros(int(24000 * silence))
    return torch.cat([pad, x, pad]).reshape(1, 1, -1)


def test_gated_level_ignores_pauses():
    sine = 20 * math.log10(0.1 / math.sqrt(2))
    assert utils.gated_level_db(_tone(0.1)) == pytest.approx(sine, abs=0.05)
    assert utils.gated_level_db(_tone(0.1, silence=4.0)) == pytest.approx(sine, abs=0.6)


@pytest.mark.parametrize("amplitude, level", [(0.005, -26.0), (0.1, None), (0.6, -14.0)])
def test_reference_level_window(amplitude, level):
    x = _tone(amplitude)
    y = utils.fit_reference_level(x, window=(-26.0, -14.0))
    if level is None:
        assert y is x
    else:
        assert utils.gated_level_db(y) == pytest.approx(level, abs=0.05)
    assert utils.fit_reference_level(x, window=None) is x


def test_reference_level_peak_cap():
    x = _tone(0.005, silence=0.5)
    x[..., 100] = 0.5
    y = utils.fit_reference_level(x, window=(-26.0, -14.0), peak_db=-1.0)
    assert 20 * math.log10(float(y.abs().max())) == pytest.approx(-1.0, abs=0.01)
