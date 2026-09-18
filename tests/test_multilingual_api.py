import io
import numpy as np

import pytest
from pydantic import ValidationError

from app.config import (
    ACOUSTIC_VOCAB_SIZE,
    MAX_PREFIX_TOKENS_PER_ROW,
    MAX_REQUEST_TEXT_CHARS,
    MAX_REQUEST_TEXT_SEGMENTS,
    MAX_TEXT_CHARS_PER_CHUNK,
    MAX_VOICE_PROMPT_TOKENS,
    MAX_VOICE_REFERENCE_B64_CHARS,
    DEFAULT_STREAMING_INITIAL_SECONDS,
    REQUIRE_API_KEY,
    SEMANTIC_VOCAB_SIZE,
    ServerConfig,
    build_system_prompt,
    normalize_language,
)
from app.inference import TTSEngine
from app.main import (
    TTSRequest,
    TTSRequestOptionError,
    _bitrate_number,
    _encode_audio,
    _prepare_raw_text,
    _require_vibevoice,
    _require_verbalizer,
    _streaming_option_error,
)


class _FakeAudio:
    def cpu(self):
        return self

    def squeeze(self, _axis):
        return self

    def numpy(self):
        return np.zeros((1, 24), dtype=np.float32)


def test_complete_opus_response_uses_ogg_opus_mime(monkeypatch):
    class EncodedAudio:
        def export(self, target: io.BytesIO, **_kwargs):
            target.write(b"ogg-opus")

    monkeypatch.setattr(
        "app.main.AudioSegment.from_file",
        lambda *_args, **_kwargs: EncodedAudio(),
    )
    encoded, mime = _encode_audio(_FakeAudio(), "opus", "64k")
    assert encoded == b"ogg-opus"
    assert mime == "audio/ogg; codecs=opus"


@pytest.mark.parametrize("punctuation", [".", "!", "?", ";", ":", ","])
def test_raw_text_does_not_duplicate_terminal_punctuation(punctuation):
    text = f"Hello{punctuation}"
    assert _prepare_raw_text(text) == text


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("en", "english"),
        ("de", "german"),
        ("es", "spanish"),
        ("fr", "french"),
        ("it", "italian"),
        ("nl", "dutch"),
        ("pt", "portuguese"),
        ("German", "german"),
    ],
)
def test_language_aliases(value, expected):
    assert normalize_language(value) == expected


def test_request_builds_exact_training_label():
    req = TTSRequest(text="Hallo", language="de", tag="conversational")
    assert req.language == "german"
    assert req.temperature == 0.8
    assert build_system_prompt(req.language, req.tag) == "german : conversational"


def test_request_accepts_backend_compatibility_payload():
    req = TTSRequest.model_validate(
        {
            "instances": [
                {
                    "text": "Hello",
                    "voice_opus_b64": "YXVkaW8=",
                    "language": "en",
                    "tag": "audiobook",
                }
            ]
        }
    )
    assert req.text == "Hello"
    assert req.voice_audio_b64 == "YXVkaW8="
    assert req.language == "english"


def test_request_rejects_ambiguous_instances_envelope():
    with pytest.raises(ValidationError, match="exactly one request"):
        TTSRequest.model_validate({"instances": []})
    with pytest.raises(ValidationError, match="exactly one request"):
        TTSRequest.model_validate(
            {"instances": [{"text": "one"}, {"text": "two"}]}
        )


def test_request_and_service_defaults():
    assert REQUIRE_API_KEY is False
    assert TTSRequest(text="Hello").use_verbalization is False
    assert TTSRequest(text="Hello").mossformer2_postprocess is True
    assert TTSRequest(text="Hello").trim_silence_padding_ms == 250
    opt_out = TTSRequest(
        text="Hello",
        mossformer2_postprocess=False,
        trim_silence_padding_ms=None,
    )
    assert opt_out.mossformer2_postprocess is False
    assert opt_out.trim_silence_padding_ms is None
    assert ServerConfig().enable_verbalization is False


def test_requested_verbalizer_must_be_loaded_and_english():
    class Engine:
        verbalizer = None

    with pytest.raises(TTSRequestOptionError, match="ENABLE_VERBALIZATION=1"):
        _require_verbalizer(
            TTSRequest(text="Hello", use_verbalization=True),
            Engine(),
        )
    with pytest.raises(TTSRequestOptionError, match="only for English"):
        _require_verbalizer(
            TTSRequest(text="Hallo", language="de", use_verbalization=True),
            Engine(),
        )


def test_explicit_vibevoice_requires_loaded_component():
    class Engine:
        vv_tokenizer = None

    _require_vibevoice(TTSRequest(text="Hello"), Engine())
    _require_vibevoice(
        TTSRequest(text="Hello", vibevoice_postprocess=False),
        Engine(),
    )
    with pytest.raises(TTSRequestOptionError, match="ENABLE_VIBEVOICE=1"):
        _require_vibevoice(
            TTSRequest(text="Hello", vibevoice_postprocess=True),
            Engine(),
        )


def test_voice_path_is_confined_to_configured_root(tmp_path, monkeypatch):
    voice_root = tmp_path / "voices"
    voice_root.mkdir()
    monkeypatch.setenv("VOICE_PATH", str(voice_root))

    request = TTSRequest(text="Hello", voice_path="en/example.wav")
    assert request.voice_path == str((voice_root / "en" / "example.wav").resolve())
    with pytest.raises(ValidationError, match="inside VOICE_PATH"):
        TTSRequest(text="Hello", voice_path="../secret.wav")
    with pytest.raises(ValidationError, match="supported audio"):
        TTSRequest(text="Hello", voice_path="notes.txt")


def test_request_caps_generation_at_32_seconds():
    assert TTSRequest(text="Hello").max_new_tokens is None
    assert TTSRequest(text="Hello", max_new_tokens=50).max_new_tokens == 50
    assert TTSRequest(text="Hello", max_new_tokens=400).max_new_tokens == 400
    with pytest.raises(ValidationError):
        TTSRequest(text="Hello", max_new_tokens=49)
    with pytest.raises(ValidationError):
        TTSRequest(text="Hello", max_new_tokens=401)


def test_request_rejects_zero_prompt_cap():
    assert TTSRequest(text="Hello", prompt_max_tokens=1).prompt_max_tokens == 1
    with pytest.raises(ValidationError):
        TTSRequest(text="Hello", prompt_max_tokens=0)


@pytest.mark.parametrize("voice_tokens", [[], [[]], [[], []]])
def test_request_rejects_empty_voice_tokens(voice_tokens):
    with pytest.raises(ValidationError):
        TTSRequest(text="Hello", voice_tokens=voice_tokens)


def test_default_voice_is_bundled_marcus(monkeypatch):
    monkeypatch.delenv("DEFAULT_VOICE", raising=False)
    assert ServerConfig().default_voice.endswith("/voices/samples/Marcus.mp3")


def test_voice_paths_accept_audio_but_not_token_json(tmp_path, monkeypatch):
    monkeypatch.setenv("VOICE_PATH", str(tmp_path))
    assert TTSRequest(text="Hello", voice_path="Marcus.mp3").voice_path.endswith("Marcus.mp3")
    with pytest.raises(ValidationError, match="supported audio"):
        TTSRequest(text="Hello", voice_path="Marcus.json")


def test_request_limits_text_and_reference_audio_size():
    TTSRequest(text="x" * MAX_REQUEST_TEXT_CHARS)
    TTSRequest(text=["x"] * MAX_REQUEST_TEXT_SEGMENTS)
    TTSRequest(text="x", voice_audio_b64="a" * MAX_VOICE_REFERENCE_B64_CHARS)
    with pytest.raises(ValidationError):
        TTSRequest(text="x" * (MAX_REQUEST_TEXT_CHARS + 1))
    with pytest.raises(ValidationError):
        TTSRequest(text=["x"] * (MAX_REQUEST_TEXT_SEGMENTS + 1))
    with pytest.raises(ValidationError):
        TTSRequest(text="x", voice_audio_b64="a" * (MAX_VOICE_REFERENCE_B64_CHARS + 1))


def test_streaming_options_are_explicit_and_transport_specific():
    assert _streaming_option_error(TTSRequest(text="x"), "mp3") is None
    assert _streaming_option_error(TTSRequest(text="x", format="mp3"), "mp3") is None
    assert "outputs mp3" in _streaming_option_error(
        TTSRequest(text="x", format="wav"), "mp3"
    )
    assert "outputs opus" in _streaming_option_error(
        TTSRequest(text="x", format="mp3"), "opus"
    )
    assert "requires vibevoice" in _streaming_option_error(
        TTSRequest(text="x", vibevoice_postprocess=False), "mp3"
    )
    assert "not supported for streaming" in _streaming_option_error(
        TTSRequest(text="x", vllm_priority=10), "mp3"
    )
    assert "non-streaming /predict" in _streaming_option_error(
        TTSRequest(text="x", mossformer2_postprocess=True), "mp3"
    )
    assert "at least 80" in _streaming_option_error(
        TTSRequest(
            text="x",
            max_new_tokens=50,
            streaming_initial_seconds=6.0,
        ),
        "mp3",
    )
    assert _streaming_option_error(
        TTSRequest(
            text="x",
            max_new_tokens=80,
            streaming_initial_seconds=6.0,
        ),
        "mp3",
    ) is None
    assert _bitrate_number("32k", bits_per_second=False) == 32
    assert _bitrate_number("96k", bits_per_second=True) == 96_000


def test_streaming_initial_seconds_default_and_alignment():
    assert (
        TTSRequest(text="x").streaming_initial_seconds
        == DEFAULT_STREAMING_INITIAL_SECONDS
    )
    for value in (0.4, 2.8, 6.0):
        assert (
            TTSRequest(text="x", streaming_initial_seconds=value).streaming_initial_seconds
            == value
        )
    for value in (0.5, 2.7, 0.0, 6.4):
        with pytest.raises(ValidationError):
            TTSRequest(text="x", streaming_initial_seconds=value)


def test_request_limits_prefix_text_and_token_rows():
    TTSRequest(
        text="x",
        prefix_text="p" * MAX_TEXT_CHARS_PER_CHUNK,
        voice_tokens=[[0] * MAX_VOICE_PROMPT_TOKENS for _ in range(4)],
        prefix_tokens=[[0] * MAX_PREFIX_TOKENS_PER_ROW],
    )
    with pytest.raises(ValidationError):
        TTSRequest(text="x", prefix_text="p" * (MAX_TEXT_CHARS_PER_CHUNK + 1))
    with pytest.raises(ValidationError):
        TTSRequest(text="x", voice_tokens=[[0]] * 5)
    with pytest.raises(ValidationError):
        TTSRequest(text="x", voice_tokens=[[0] * (MAX_VOICE_PROMPT_TOKENS + 1)])
    with pytest.raises(ValidationError):
        TTSRequest(text="x", voice_tokens=[[0, 1], [0]])
    with pytest.raises(ValidationError):
        TTSRequest(text="x", prefix_tokens=[[0] * (MAX_PREFIX_TOKENS_PER_ROW + 1)])


def test_request_accepts_only_one_voice_source():
    with pytest.raises(ValidationError):
        TTSRequest(
            text="x",
            voice_path="en/example.wav",
            voice_audio_b64="YXVkaW8=",
        )
    with pytest.raises(ValidationError):
        TTSRequest(
            text="x",
            voice_audio_b64="YXVkaW8=",
            voice_tokens=[[0]],
        )


def test_request_rejects_out_of_range_token_ids():
    with pytest.raises(ValidationError):
        TTSRequest(text="x", voice_tokens=[[SEMANTIC_VOCAB_SIZE]])
    with pytest.raises(ValidationError):
        TTSRequest(text="x", voice_tokens=[[0], [ACOUSTIC_VOCAB_SIZE]])
    with pytest.raises(ValidationError):
        TTSRequest(text="x", prefix_tokens=[[-1]])


def test_invalid_language_and_tag_fail_at_request_boundary():
    with pytest.raises(ValidationError):
        TTSRequest(text="hello", language="xx")
    with pytest.raises(ValidationError):
        TTSRequest(text="hello", tag="conversation")
    with pytest.raises(ValidationError):
        TTSRequest(text="hello", system_prompt="english : audiobook")


def test_engine_uses_one_stack_for_every_language_tag():
    engine = TTSEngine(ServerConfig())
    _, _, german_tag, _, _ = engine._resolve_params(language="de", tag="agentic")
    _, _, french_tag, _, _ = engine._resolve_params(language="fr", tag="audiobook")
    assert german_tag == "<|im_start|>german : agentic<|im_end|>\n"
    assert french_tag == "<|im_start|>french : audiobook<|im_end|>\n"
    assert isinstance(engine.engines, list)


def test_verbalizer_capacity_must_be_positive():
    with pytest.raises(ValueError, match="verbalization_gpu_memory_gib"):
        ServerConfig(verbalization_gpu_memory_gib=0)
    with pytest.raises(ValueError, match="verbalization_max_num_seqs"):
        ServerConfig(verbalization_max_num_seqs=0)


def test_end_to_end_limit_defaults_to_twice_cb0_and_can_be_overridden():
    assert ServerConfig().max_inflight_requests == 2 * ServerConfig().max_num_seqs_per_cb[0]
    assert ServerConfig(max_inflight_requests=5).max_inflight_requests == 5
    with pytest.raises(ValueError, match="max_inflight_requests"):
        ServerConfig(max_inflight_requests=0)


def test_batched_token_defaults_follow_context_and_sequence_limits():
    config = ServerConfig(
        gpu_memory_gib_per_cb=(1.0, 1.0, 1.0, 1.0),
        max_model_len_per_cb=(100, 200, 300, 400),
        max_num_seqs_per_cb=(2, 3, 4, 5),
    )
    assert config.max_num_batched_tokens_per_cb == (200, 600, 1200, 2000)


def test_batched_token_limits_must_cover_all_active_contexts():
    with pytest.raises(ValueError, match="so configured prefills are not split"):
        ServerConfig(
            gpu_memory_gib_per_cb=(1.0, 1.0, 1.0, 1.0),
            max_model_len_per_cb=(100, 200, 300, 400),
            max_num_seqs_per_cb=(2, 3, 4, 5),
            max_num_batched_tokens_per_cb=(64, 64, 64, 64),
        )
