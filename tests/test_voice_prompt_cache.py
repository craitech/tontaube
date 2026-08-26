"""Focused tests for request-independent voice-prompt caching."""
from collections import OrderedDict
from unittest.mock import MagicMock

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
    assert cache["path:" + str(voice_path)] == (full_semantic, full_acoustic)


def test_default_prompt_is_capped_per_request():
    default = (list(range(20)), [list(range(100, 120)) for _ in range(3)])
    result = utils.resolve_voice_prompt(
        None, None, object(), 60.0, 6, 4, OrderedDict(), 8, default,
    )
    assert result == (default[0][:6], [codes[:6] for codes in default[1]])
