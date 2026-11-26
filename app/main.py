"""HTTP API for Tontaube text-to-speech inference."""
import asyncio
import math
import os
import multiprocessing
from importlib.metadata import PackageNotFoundError, version as package_version

if multiprocessing.get_start_method(allow_none=True) is None:
    multiprocessing.set_start_method('spawn')

import base64
import contextlib
from contextlib import asynccontextmanager
import io
import json
from pathlib import Path
import secrets
import time
from typing import Literal

import numpy as np
import soundfile as sf
from numpy.lib.stride_tricks import sliding_window_view
from pydub import AudioSegment
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from app.config import (
    ACOUSTIC_VOCAB_SIZE,
    MOSSFORMER2_POSTPROCESS_DEFAULT,
    DEFAULT_LANGUAGE,
    DEFAULT_TAG,
    DEFAULT_STREAMING_INITIAL_SECONDS,
    DUALCODEC_CODEC_ID,
    DUALCODEC_HZ,
    DUALCODEC_MAX_CODEBOOKS,
    DUALCODEC_SAMPLE_RATE,
    MAX_STREAMING_INITIAL_SECONDS,
    MAX_PREFIX_TOKENS_PER_ROW,
    MAX_REQUEST_TEXT_CHARS,
    MAX_REQUEST_TEXT_SEGMENTS,
    MAX_SEMANTIC_TOKENS_PER_CHUNK,
    MAX_TEXT_CHARS_PER_CHUNK,
    MAX_VOICE_PROMPT_TOKENS,
    MAX_VOICE_REFERENCE_B64_CHARS,
    MERGE_SHORT_CLIENT_SEGMENTS,
    MERGE_SHORT_PIECES,
    MODEL_VERSION,
    N_CODEBOOKS,
    NO_SHORT_FINAL_SENTENCE,
    VERBALIZER_CONTEXTUAL,
    VERBALIZER_MODES,
    VERBALIZER_OFF,
    VERBALIZER_RULES,
    SEMANTIC_VOCAB_SIZE,
    MAX_SEMANTIC_TOKENS_PER_CHUNK,
    STREAM_AUTO_MIN_INITIAL_SECONDS,
    STREAM_CALIBRATION,
    STREAM_CALIBRATION_MARGIN,
    STREAM_CALIBRATION_WARNING,
    STREAMING_DC_PAD_TOKENS,
    MIN_STREAMING_INITIAL_SECONDS,
    STREAMING_DC_PAD_TOKENS,
    STREAMING_SECONDS_QUANTUM,
    BUNDLED_VOICE_PATH,
    TTS_MODEL,
    VOICE_AUDIO_EXTENSIONS,
    ServerConfig,
    normalize_language,
)
from app.flash import EngineBusy, UnsupportedOption
from app.priority_gate import get_gate
from app.logging_utils import get_logger
from app.model.text_vocab import clean_text, strip_markup, strip_separators
from app.text import (
    MAX_MERGED_PIECE_CHARS,
    merge_short_pieces,
    sanitize_text_for_tts,
    split_long_text,
)
from app.voice_catalog import VoicePromptError

logger = get_logger("api")

try:
    API_VERSION = package_version("tontaube-inference")
except PackageNotFoundError:
    API_VERSION = "1.0.0"


def _device_label(engine) -> str:
    """Where the loaded model runs, for clients (GET /controls `device`): "GPU (<name>)" or "CPU (<n> threads)"."""
    from app.config import LITE_ON_CPU
    try:
        on_cpu = LITE_ON_CPU
        if _flash():
            from app import flash
            on_cpu = flash.device() == "cpu"
        if on_cpu:
            threads = sum(int((getattr(e, "info", None) or {}).get("threads", 0)) for e in getattr(engine, "engines", []))
            return f"CPU ({threads} threads)" if threads else "CPU"
        import torch
        return f"GPU ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "GPU"
    except Exception:                      # informational only
        return "unknown"


def _flash() -> bool:
    """TTS_MODEL=flash: the server runs TontaubeV1.1 Flash (app/flash.py) instead of TontaubeV1.1."""
    return TTS_MODEL == "flash"


async def _get_engine():
    if _flash():
        from app.flash import get_engine
    else:
        from app.inference import get_engine

    return await get_engine()


def _unload_engine() -> None:
    if _flash():
        from app.flash import unload_engine
    else:
        from app.inference import unload_engine

    unload_engine()


def _ensure_punctuation(text: str) -> str:
    """Append a period if text doesn't end with punctuation."""
    t = text.rstrip()
    if t and t[-1] not in ".!?;:,":
        return t + "."
    return text


def _prepare_raw_text(text: str | list[str]) -> str | list[str]:
    segments = text if isinstance(text, list) else [text]
    if not any(char.isalnum() for segment in segments for char in clean_text(strip_separators(strip_markup(segment)))):
        raise TTSRequestOptionError(
            "text contains no characters the model can read (Latin-script letters of the "
            "supported languages, digits and common punctuation)"
        )
    if isinstance(text, list):
        prepared = list(text)
        prepared[-1] = _ensure_punctuation(prepared[-1])
        return prepared
    return _ensure_punctuation(text)


def _trim_boundary_silence(
    audio,
    *,
    padding_ms: int,
    silence_threshold_dbfs: float = -50.0,
    headroom_db: float = 0.0,
    lead_padding_ms: int | None = None,
    sample_rate: int = DUALCODEC_SAMPLE_RATE,
):
    """Trim excess edge silence while preserving all internal pauses.

    silence_threshold_dbfs is at the decoder's own level. For audio already
    scaled by OUTPUT_HEADROOM_DB, pass that headroom: the threshold moves
    down by the same dB, so the trim points do not depend on the headroom.
    headroom_db=0 is the old threshold.
    """
    bounds = _trim_bounds(
        audio, padding_ms=padding_ms, silence_threshold_dbfs=silence_threshold_dbfs,
        headroom_db=headroom_db, lead_padding_ms=lead_padding_ms, sample_rate=sample_rate,
    )
    return audio if bounds is None else audio[..., bounds[0]:bounds[1]]


def _trim_bounds(
    audio,
    *,
    padding_ms: int,
    silence_threshold_dbfs: float = -50.0,
    headroom_db: float = 0.0,
    lead_padding_ms: int | None = None,
    sample_rate: int = DUALCODEC_SAMPLE_RATE,
) -> tuple[int, int] | None:
    """_trim_boundary_silence's kept samples [start, end), or None (audio unchanged). audio: a torch tensor or a
    numpy array (TontaubeV1.1 Flash)."""
    if padding_ms < 0 or audio.shape[-1] == 0:
        return None

    if isinstance(audio, np.ndarray):
        mono_power = np.square(audio.reshape(-1, audio.shape[-1]), dtype=np.float32).mean(axis=0)
    else:
        mono_power = audio.detach().float().reshape(-1, audio.shape[-1]).square().mean(dim=0)
    frame_samples = max(1, round(sample_rate * 0.02))
    hop_samples = max(1, round(sample_rate * 0.01))
    if len(mono_power) < frame_samples:
        return None

    threshold = 10.0 ** ((silence_threshold_dbfs - headroom_db) / 20.0)
    if isinstance(mono_power, np.ndarray):
        frame_rms = np.sqrt(sliding_window_view(mono_power, frame_samples)[::hop_samples].mean(axis=-1))
        active = frame_rms >= threshold
        loud_tail = lambda covered: np.sqrt(mono_power[covered:].mean()) >= threshold  # noqa: E731
    else:
        frame_rms = mono_power.unfold(0, frame_samples, hop_samples).mean(dim=-1).sqrt()
        active = (frame_rms >= threshold).cpu().numpy()
        loud_tail = lambda covered: bool(mono_power[covered:].mean().sqrt() >= threshold)  # noqa: E731
    active_frames = np.flatnonzero(active)
    if len(active_frames) == 0:
        return None

    padding_samples = round(sample_rate * padding_ms / 1000)
    # Retain the entire active frames, including at zero requested padding.
    # Frame centers would otherwise cut into the start and end of speech.
    first_sample = int(active_frames[0]) * hop_samples
    last_sample = int(active_frames[-1]) * hop_samples + frame_samples
    # unfold omits a partial final hop; retain it if it contains signal.
    covered = (len(frame_rms) - 1) * hop_samples + frame_samples
    if covered < len(mono_power) and loud_tail(covered):
        last_sample = len(mono_power)
    lead_samples = padding_samples if lead_padding_ms is None else round(sample_rate * max(0, lead_padding_ms) / 1000)
    start = max(0, first_sample - lead_samples)
    end = min(audio.shape[-1], last_sample + padding_samples)
    return start, end


def _encode_audio(
    audio,
    output_format: str,
    bitrate: str,
    sample_rate: int = DUALCODEC_SAMPLE_RATE,
) -> tuple[bytes, str]:
    wav_buffer = io.BytesIO()
    sf.write(
        wav_buffer,
        audio.T if isinstance(audio, np.ndarray) else audio.cpu().squeeze(0).numpy().T,
        sample_rate,
        format="WAV",
    )
    wav_bytes = wav_buffer.getvalue()
    if output_format == "wav":
        return wav_bytes, "audio/wav"

    encoded = io.BytesIO()
    AudioSegment.from_file(io.BytesIO(wav_bytes), format="wav").export(
        encoded,
        format=output_format,
        bitrate=bitrate,
    )
    mime_type = "audio/mpeg" if output_format == "mp3" else "audio/ogg; codecs=opus"
    return encoded.getvalue(), mime_type


def _apply_mossformer2_postprocessing(
    audio, input_sample_rate: int, *, headroom_db: float = 0.0, cuts_out: list | None = None,
):
    """Load and run the optional enhancer outside the async event loop.

    headroom_db: the OUTPUT_HEADROOM_DB already applied to `audio`; the pause
    cap lowers its fixed thresholds by it. cuts_out: receives the pause cap's
    applied (left, right, width) cuts in input samples (include_timestamps).
    """
    from app.mossformer2_postprocessor import (
        MOSSFORMER2_SAMPLE_RATE,
        get_mossformer2_postprocessor,
    )

    enhanced = get_mossformer2_postprocessor().process(
        audio, input_sample_rate, headroom_db=headroom_db,
        **({"cuts_out": cuts_out} if cuts_out is not None else {}),
    )
    return enhanced, MOSSFORMER2_SAMPLE_RATE


class _InstancesRequest(BaseModel):
    """Request body that also accepts the `{"instances": [request]}` envelope."""

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def unwrap_instances_envelope(cls, value):
        if isinstance(value, dict) and "instances" in value:
            instances = value.get("instances")
            if not isinstance(instances, list) or len(instances) != 1:
                raise ValueError("instances must contain exactly one request")
            return instances[0]
        return value


def _verbalizer_name(value: str | None) -> str | None:
    """A request's `verbalizer`, lower-cased; ValueError (a 422) for a name that is none of VERBALIZER_MODES."""
    if value is None:
        return None
    mode = value.strip().lower()
    if mode not in VERBALIZER_MODES:
        raise ValueError(f"unknown verbalizer {value!r}; expected one of {', '.join(VERBALIZER_MODES)}")
    return mode


class TTSRequest(_InstancesRequest):
    text: str | list[str]

    @field_validator('verbalizer')
    @classmethod
    def known_verbalizer(cls, value: str | None) -> str | None:
        return _verbalizer_name(value)

    @field_validator('text')
    @classmethod
    def text_not_empty(cls, v):
        if isinstance(v, str):
            if not v.strip():
                raise ValueError('text must not be empty')
            total_chars = len(v)
        else:
            if not v or any(not s.strip() for s in v):
                raise ValueError('text must not be empty')
            if len(v) > MAX_REQUEST_TEXT_SEGMENTS:
                raise ValueError(
                    f'text must contain at most {MAX_REQUEST_TEXT_SEGMENTS} segments'
                )
            total_chars = sum(len(segment) for segment in v)
        if total_chars > MAX_REQUEST_TEXT_CHARS:
            raise ValueError(
                f'text must contain at most {MAX_REQUEST_TEXT_CHARS} characters'
            )
        return v
    temperature: float | None = Field(
        default=None,
        ge=0,
        le=2,
        description=(
            "CB0 sampling temperature; omitted = server default 0.7. Lower favors "
            "correctness, higher expression."
        ),
    )
    top_p: float | None = Field(default=None, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=-1)
    frequency_penalty: float | None = Field(
        default=None, ge=-2, le=2, description="CB0 frequency penalty; omitted = server default 0.1.",
    )
    silence_logit_bias: float | None = Field(
        default=None,
        ge=-10,
        le=10,
        description=(
            "Base logit bias applied to the CB0 silence-collapse token group. "
            "Negative values suppress silence; omitted uses the server default."
        ),
    )
    seed: int | None = Field(default=None, ge=0)
    voice_path: str | None = Field(
        default=None,
        description="Audio path inside the server's configured VOICE_PATH root.",
    )
    voice_audio_b64: str | None = Field(
        default=None,
        max_length=MAX_VOICE_REFERENCE_B64_CHARS,
        validation_alias=AliasChoices("voice_audio_b64", "voice_opus_b64"),
        description="Base64-encoded voice-reference audio in any FFmpeg-supported format.",
    )
    voice_tokens: list[list[int]] | None = None

    prefix_text: str | None = Field(default=None, max_length=MAX_TEXT_CHARS_PER_CHUNK)
    prefix_tokens: list[list[int]] | None = None
    max_new_tokens: int | None = Field(
        default=None,
        ge=50,
        le=MAX_SEMANTIC_TOKENS_PER_CHUNK,
        description=(
            "Total CB0 semantic-token budget for each text chunk, including "
            "the forced initial-silence token and any streaming early prefix. "
            "The 400-token ceiling corresponds to 32 seconds at 12.5 Hz."
        ),
    )
    streaming_initial_seconds: float = Field(
        default=DEFAULT_STREAMING_INITIAL_SECONDS,
        ge=MIN_STREAMING_INITIAL_SECONDS,
        le=MAX_STREAMING_INITIAL_SECONDS,
        description=(
            "Initial audio buffered by streaming routes. Must be a multiple of "
            f"{STREAMING_SECONDS_QUANTUM} seconds. Larger values increase time "
            "to first audio but provide more playback headroom. Omitted: 2.8 s, or, on a server "
            "started with STREAM_CALIBRATION=1, chosen per request from its startup calibration so "
            "playback does not pause on that machine (an explicit value is then logged as a warning)."
        ),
    )
    acoustic_temperature: float | None = Field(default=None, ge=0, le=2)
    acoustic_top_k: int | None = Field(default=None, ge=-1)
    prompt_max_tokens: int | None = Field(
        default=None,
        ge=1,
        le=MAX_VOICE_PROMPT_TOKENS,
        description=(
            "Maximum reference-token count for CB0. Omitted uses the server "
            "default; later codebooks apply their own smaller caps."
        ),
    )
    vibevoice_postprocess: bool | None = Field(
        default=None,
        description=(
            "Enable VibeVoice postprocessing for non-streaming generation. "
            "Streaming always requires and uses VibeVoice."
        ),
    )
    language: str = DEFAULT_LANGUAGE
    tag: Literal["audiobook", "conversational", "agentic"] = DEFAULT_TAG
    use_system_tag: bool = True
    use_verbalization: bool = Field(
        default=False,
        description=(
            f"Legacy form of `verbalizer`: true is {VERBALIZER_CONTEXTUAL!r} on TontaubeV1.1 and "
            f"{VERBALIZER_RULES!r} on Flash, false is {VERBALIZER_OFF!r}. Sent together with `verbalizer` they "
            "must agree (422 otherwise)."
        ),
    )
    verbalizer: str | None = Field(
        default=None,
        description=(
            f"How the text is verbalized: {VERBALIZER_RULES!r} (Flash's rule verbalizer: numbers, dates, times, "
            f"money, units, abbreviations; English), {VERBALIZER_CONTEXTUAL!r} (the model verbalizer, English; the "
            f"server needs ENABLE_VERBALIZATION=1) or {VERBALIZER_OFF!r}. Omitted: {VERBALIZER_OFF!r} on "
            f"TontaubeV1.1, {VERBALIZER_RULES!r} on Flash. Unsupported combinations are refused with 422."
        ),
    )
    trim_silence_padding_ms: int | None = Field(
        default=250,
        ge=0,
        le=2000,
        description=(
            "Trim leading and trailing silence beyond this padding. Defaults "
            "to 250 ms; internal pauses are preserved and null disables trimming."
        ),
    )
    mossformer2_postprocess: bool = Field(
        default=MOSSFORMER2_POSTPROCESS_DEFAULT,
        description=(
            "Cap quiet pauses at 2.5 seconds, then apply hybrid MossFormer2_SR_48K "
            "super-resolution after optional boundary trimming. Enabled by default "
            "for non-streaming requests unless the server sets "
            "MOSSFORMER2_POSTPROCESS_DEFAULT=0 (the default for the lite backend on CPU); "
            "set false to return the original 24 kHz output."
        ),
    )
    include_timestamps: bool = Field(
        default=False,
        description=(
            "Non-streaming /predict, English only: also return `timestamps`, the start and end (seconds "
            "on the returned audio) of each spoken word and letter, from a forced alignment of the text "
            "the model read against its CB0 codes."
        ),
    )
    format: Literal["wav", "mp3", "opus", "pcm16"] = "wav"
    bitrate: Literal["32k", "64k", "96k", "128k"] = Field(
        default="64k",
        description=(
            "Audio bitrate for encoded MP3 or Opus output, including streaming. "
            "`format: \"pcm16\"` (`POST /stream` only: raw 16-bit little-endian mono PCM at 24 kHz, "
            "no codec and so no codec delay; the default of `/stream`) ignores it."
        ),
    )
    priority: Literal["low", "high"] = Field(
        default="low",
        description="Request-level priority gate used consistently by batch and streaming routes.",
    )
    # Lower values receive earlier scheduling within vLLM.
    vllm_priority: int | None = Field(
        default=None,
        description=(
            "Raw vLLM scheduler priority for non-streaming /predict calls only. "
            "Streaming callers should use priority instead."
        ),
    )
    # Slider controls of a control-enabled CB0 bundle (app/controls.py).
    controls: dict[str, int] | str | None = Field(
        default=None,
        description=(
            "Style sliders: {name: bin}, bin 0 (low end) to bins-1 (high end); a slider left out keeps the "
            "model's choice. GET /controls lists the sliders. A control-token string is accepted too."
        ),
        examples=[{"rate": 1, "pitch_range": 4}],
    )
    segment_controls: list[dict[str, int | None] | str | None] | None = Field(
        default=None,
        description=(
            "Per-segment sliders aligned with a list `text`, merged over `controls` "
            "(a null value drops that control for the segment)."
        ),
    )

    @model_validator(mode="after")
    def segment_controls_aligned(self):
        if self.segment_controls is not None and (
            not isinstance(self.text, list) or len(self.segment_controls) != len(self.text)
        ):
            raise ValueError("segment_controls needs a list text with one entry per segment")
        return self

    @field_validator("language")
    @classmethod
    def canonical_language(cls, value: str) -> str:
        return normalize_language(value)

    @field_validator("streaming_initial_seconds")
    @classmethod
    def aligned_streaming_initial_seconds(cls, value: float) -> float:
        steps = round(value / STREAMING_SECONDS_QUANTUM)
        aligned = steps * STREAMING_SECONDS_QUANTUM
        if not math.isclose(value, aligned, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                f"streaming_initial_seconds must be a multiple of "
                f"{STREAMING_SECONDS_QUANTUM}"
            )
        return round(aligned, 10)

    @field_validator("voice_path")
    @classmethod
    def safe_server_voice_path(cls, value: str | None) -> str | None:
        if not value:
            return None
        root = Path(
            os.getenv("VOICE_PATH", str(BUNDLED_VOICE_PATH))
        ).expanduser().resolve()
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError("voice_path must remain inside VOICE_PATH") from exc
        if candidate.suffix.lower() not in VOICE_AUDIO_EXTENSIONS:
            raise ValueError("voice_path must point to a supported audio file")
        return str(candidate)

    @staticmethod
    def _validate_token_rows(
        rows: list[list[int]] | None,
        max_tokens_per_row: int,
    ) -> list[list[int]] | None:
        if rows is None or not rows:
            return rows
        if len(rows) > 4:
            raise ValueError('token inputs must contain at most four codebook rows')
        for row_index, row in enumerate(rows):
            if len(row) > max_tokens_per_row:
                raise ValueError(
                    f'token row {row_index} must contain at most '
                    f'{max_tokens_per_row} tokens'
                )
            vocab_size = SEMANTIC_VOCAB_SIZE if row_index == 0 else ACOUSTIC_VOCAB_SIZE
            if any(token < 0 or token >= vocab_size for token in row):
                raise ValueError(
                    f'token row {row_index} contains an ID outside [0, {vocab_size})'
                )
        return rows

    @field_validator('voice_tokens', mode='before')
    @classmethod
    def validate_voice_tokens(cls, value):
        if value is None:
            return None
        if not isinstance(value, list) or any(
            not isinstance(row, list) or any(type(token) is not int for token in row)
            for row in value
        ):
            raise ValueError('voice tokens must be lists of integer IDs')
        rows = cls._validate_token_rows(value, MAX_VOICE_PROMPT_TOKENS)
        if not rows or not rows[0]:
            raise ValueError('voice-token rows must not be empty')
        if len({len(row) for row in rows}) > 1:
            raise ValueError('voice-token rows must have equal lengths')
        return rows

    @field_validator('prefix_tokens')
    @classmethod
    def validate_prefix_tokens(cls, value):
        return cls._validate_token_rows(value, MAX_PREFIX_TOKENS_PER_ROW)

    @model_validator(mode="after")
    def one_voice_source(self):
        sources = (
            bool(self.voice_path),
            bool(self.voice_audio_b64),
            bool(self.voice_tokens),
        )
        if sum(sources) > 1:
            raise ValueError(
                'provide only one of voice_path, voice_audio_b64, or voice_tokens'
            )
        return self

class TTSResponse(BaseModel):
    audio_b64: str
    mime_type: str
    sample_rate: int
    codebook_tokens: list[list[int]] | None = None
    timestamps: dict | None = Field(
        default=None,
        description=(
            "Only with include_timestamps: {text, words: [{text, start, end, char_start, char_end, "
            "confidence}], characters: [{char, index, start, end}]} (app/timestamps.py)."
        ),
    )


class EncodeVoiceRequest(_InstancesRequest):
    voice_audio_b64: str = Field(
        min_length=1,
        max_length=MAX_VOICE_REFERENCE_B64_CHARS,
        validation_alias=AliasChoices("voice_audio_b64", "voice_opus_b64"),
        description=(
            "Base64-encoded voice-reference audio in any FFmpeg-supported format; "
            "only its first 60 seconds are encoded."
        ),
    )
    prompt_max_tokens: int | None = Field(
        default=None,
        ge=1,
        le=MAX_VOICE_PROMPT_TOKENS,
        description="Maximum token count to return; omitted returns the full encoded prompt.",
    )
    n_codebooks: int | None = Field(
        default=None,
        ge=1,
        le=DUALCODEC_MAX_CODEBOOKS,
        description=(
            f"Codebook rows to return, up to all {DUALCODEC_MAX_CODEBOOKS} DualCodec levels (for "
            f"decoding back to audio); omitted = the {N_CODEBOOKS} the model uses. Generation "
            f"routes accept at most the first {N_CODEBOOKS} rows."
        ),
    )


class EncodeVoiceResponse(BaseModel):
    voice_tokens: list[list[int]] = Field(
        description=(
            "Row 0 semantic, rows 1.. acoustic; its first four rows are the `voice_tokens` "
            "of /predict, /stream and /ws/stream."
        ),
    )
    n_tokens: int
    n_codebooks: int
    duration_seconds: float
    token_rate_hz: float
    model_version: str
    codec: str


class TTSErrorResponse(BaseModel):
    error: str
    code: str | None = Field(
        default=None,
        description="Stable machine-readable reason, where one is defined (e.g. timestamps_unavailable).",
    )


class TTSRequestOptionError(ValueError):
    """A valid request whose selected feature is unavailable. ``code``: optional stable reason for clients."""

    def __init__(self, message: str, code: str | None = None):
        super().__init__(message)
        self.code = code


class _CodedMessage(str):
    """An error message (compares as the plain string) with a stable machine-readable ``code``."""

    code: str | None = None

    def __new__(cls, message: str, code: str):
        obj = super().__new__(cls, message)
        obj.code = code
        return obj


def _error_body(message: str, code: str | None = None) -> dict:
    """{"error": message} plus "code" when the message carries one."""
    code = code or getattr(message, "code", None)
    return TTSErrorResponse(error=str(message), code=code).model_dump(exclude_none=True)


def _verbalizer_mode(req: TTSRequest, engine) -> str:
    """The verbalization mode of a request (VERBALIZER_RULES, VERBALIZER_CONTEXTUAL or VERBALIZER_OFF) from
    `verbalizer` and the legacy `use_verbalization`; raises TTSRequestOptionError for a contradiction or for a mode
    this server or language cannot do. Defaults: off on TontaubeV1.1, rules on Flash."""
    flash = _flash()
    legacy = req.use_verbalization if "use_verbalization" in req.model_fields_set else None
    legacy_mode = (None if legacy is None else VERBALIZER_OFF if not legacy
                   else VERBALIZER_RULES if flash else VERBALIZER_CONTEXTUAL)
    named = req.verbalizer
    if named is not None and legacy_mode is not None and named != legacy_mode:
        raise TTSRequestOptionError(
            f"verbalizer={named!r} contradicts use_verbalization={str(legacy).lower()}, which means "
            f"{legacy_mode!r} on this server; send one of the two", code="verbalizer_conflict",
        )
    mode = named or legacy_mode or (VERBALIZER_RULES if flash else VERBALIZER_OFF)
    if mode == VERBALIZER_OFF:
        return mode
    asked = named is not None or legacy is True
    label = f"verbalizer={mode!r}" if named is not None else "use_verbalization=true"
    if req.language != "english":
        if not asked:
            return VERBALIZER_OFF     # Flash's default on a request it refuses for its language anyway
        raise TTSRequestOptionError(f"{label} is supported only for English requests")
    if mode == VERBALIZER_CONTEXTUAL and (engine is None or getattr(engine, "verbalizer", None) is None):
        raise TTSRequestOptionError(
            f"{label} requires a loaded verbalizer; start the server with ENABLE_VERBALIZATION=1",
            code="verbalizer_unavailable",
        )
    if mode == VERBALIZER_RULES and not flash:
        from app import verbalize_rules

        try:
            verbalize_rules.front_end()
        except Exception as exc:  # noqa: BLE001
            raise TTSRequestOptionError(
                f"{label} is not available on this server: {exc}", code="verbalizer_unavailable",
            ) from exc
    return mode


def _require_verbalizer(req: TTSRequest, engine) -> str:
    """The request's verbalization mode (_verbalizer_mode), checked."""
    return _verbalizer_mode(req, engine)


def _require_timestamps(req: TTSRequest, engine) -> None:
    if not req.include_timestamps:
        return
    if req.language != "english":
        raise TTSRequestOptionError(
            "include_timestamps is only available for English", code="timestamps_unsupported_language",
        )
    if engine is None or getattr(engine, "cb0_ctc", None) is None:
        raise TTSRequestOptionError("timestamps are not available on this server", code="timestamps_unavailable")


# The request fields TontaubeV1.1 Flash reads; it refuses the others (voices, styles, sliders, timestamps, prefixes,
# token and sampling controls of TontaubeV1.1).
FLASH_FIELDS = frozenset({
    "text", "temperature", "seed", "format", "bitrate", "priority", "trim_silence_padding_ms",
    "use_verbalization", "verbalizer", "language",
})


def _flash_option_error(req: TTSRequest) -> str | None:
    # Flash has no MossFormer2 step; an explicit false is accepted for clients that always send the field.
    unsupported = sorted(
        f for f in req.model_fields_set - FLASH_FIELDS
        if not (f == "mossformer2_postprocess" and not req.mossformer2_postprocess)
    )
    if unsupported:
        return f"TontaubeV1.1 Flash does not support {', '.join(unsupported)}"
    if req.language != "english":
        return "TontaubeV1.1 Flash speaks English only"
    return None


def _flash_kwargs(req: TTSRequest) -> dict:
    """FlashEngine.generate / generate_streaming arguments. Flash reads the raw text with its own rule verbalizer
    and chunker (on unless the request sets verbalizer to off or use_verbalization to false), not TontaubeV1.1's text
    preparation."""
    verbalize = _verbalizer_mode(req, None) == VERBALIZER_RULES
    return {"seed": req.seed, "temperature": req.temperature, "verbalize": verbalize, "priority": req.priority}


def _scaled_chars(limit: int, scale: float) -> int:
    return max(1, round(limit * scale))


def _split_with_controls(
    text, segment_controls, digit_weight: int = 1, chunk_scales: list[float] | None = None,
):
    """split_long_text keeping segment_controls aligned: pieces inherit their segment's entry.

    Then pieces under 50 chars join a neighbour (merge_short_pieces):
    pieces split from the same text with MERGE_SHORT_PIECES; across client segments whose
    segment_controls entries are identical with MERGE_SHORT_CLIENT_SEGMENTS (both opt-in).
    ``chunk_scales`` (one per segment, TTSEngine.chunk_char_scales) scale each segment's split
    length and merge limit. NO_SHORT_FINAL_SENTENCE (on by default) keeps the splitter and the
    merge from ending a piece on a 1-2-word sentence (app/text.py).
    """
    scales = chunk_scales or [1.0] * (1 if isinstance(text, str) else len(text))
    if isinstance(text, str):
        pieces = split_long_text(
            text, max_chars=_scaled_chars(MAX_TEXT_CHARS_PER_CHUNK, scales[0]),
            digit_weight=digit_weight, avoid_short_final=NO_SHORT_FINAL_SENTENCE,
        )
        if isinstance(pieces, str) or not (MERGE_SHORT_PIECES or MERGE_SHORT_CLIENT_SEGMENTS):
            return pieces, None
        pieces, _ = merge_short_pieces(
            pieces, max_chars=_scaled_chars(MAX_MERGED_PIECE_CHARS, scales[0]),
            digit_weight=digit_weight, avoid_short_final=NO_SHORT_FINAL_SENTENCE,
        )
        return (pieces[0] if len(pieces) == 1 else pieces), None
    entries = segment_controls if segment_controls is not None else [None] * len(text)
    pieces, controls, origins, limits = [], [], [], []
    for index, (segment, value, scale) in enumerate(zip(text, entries, scales)):
        split = split_long_text(
            [segment], max_chars=_scaled_chars(MAX_TEXT_CHARS_PER_CHUNK, scale),
            digit_weight=digit_weight, avoid_short_final=NO_SHORT_FINAL_SENTENCE,
        )
        pieces.extend(split)
        controls.extend([value] * len(split))
        origins.extend([index] * len(split))
        limits.extend([_scaled_chars(MAX_MERGED_PIECE_CHARS, scale)] * len(split))
    if MERGE_SHORT_CLIENT_SEGMENTS:
        # Neighbours with identical controls entries, within a segment and across segments.
        pieces, controls = merge_short_pieces(
            pieces, controls, max_chars=limits, digit_weight=digit_weight, avoid_short_final=NO_SHORT_FINAL_SENTENCE,
        )
    elif MERGE_SHORT_PIECES:
        pieces, origins = merge_short_pieces(
            pieces, origins, max_chars=limits, digit_weight=digit_weight, avoid_short_final=NO_SHORT_FINAL_SENTENCE,
        )
        controls = [entries[index] for index in origins]
    return pieces, (controls if segment_controls is not None else None)


def _chunk_scales(req: TTSRequest, engine, text, segment_controls) -> list[float] | None:
    """Per-segment chunk-length scales from the rate slider (CB0_RATE_AWARE_CHUNK_CHARS)."""
    if req.controls is None and segment_controls is None:
        return None
    return engine.chunk_char_scales(
        len(text) if isinstance(text, list) else 1, req.language, req.tag, req.use_system_tag,
        controls=req.controls, segment_controls=segment_controls,
    )


def _require_controls(req: TTSRequest, engine, text, segment_controls) -> None:
    if req.controls is None and segment_controls is None:
        return
    n_segments = len(text) if isinstance(text, list) else 1
    try:
        engine._cb0_system_tags(
            "", n_segments, req.language, req.tag, req.use_system_tag,
            controls=req.controls, segment_controls=segment_controls,
        )
    except ValueError as exc:
        raise TTSRequestOptionError(str(exc)) from exc


def _require_vibevoice(req: TTSRequest, engine) -> None:
    if req.vibevoice_postprocess is True and engine.vv_tokenizer is None:
        raise TTSRequestOptionError(
            "vibevoice_postprocess=true requires VibeVoice; start the server "
            "with ENABLE_VIBEVOICE=1"
        )


def _streaming_option_error(req: TTSRequest, transport: Literal["mp3", "opus"]) -> str | None:
    """Return a clear error for request fields unsupported by a stream transport."""
    if req.vibevoice_postprocess is False:
        return "streaming requires vibevoice_postprocess=true"
    if (
        req.mossformer2_postprocess
        and "mossformer2_postprocess" in req.model_fields_set
    ):
        return "mossformer2_postprocess is supported only by non-streaming /predict"
    if req.include_timestamps:
        return _CodedMessage(
            "include_timestamps is supported only by non-streaming /predict", "timestamps_not_streamable",
        )
    if "format" in req.model_fields_set and req.format != transport:
        return f"this streaming route outputs {transport}; set format={transport!r} or omit format"
    if req.vllm_priority is not None:
        return "vllm_priority is not supported for streaming; use priority='low' or 'high'"
    required_tokens = (
        round(req.streaming_initial_seconds * DUALCODEC_HZ)
        + STREAMING_DC_PAD_TOKENS
    )
    if req.max_new_tokens is not None and req.max_new_tokens < required_tokens:
        return (
            f"max_new_tokens must be at least {required_tokens} when "
            f"streaming_initial_seconds={req.streaming_initial_seconds:g}"
        )
    return None


def _bitrate_number(value: str, *, bits_per_second: bool) -> int:
    kbps = int(value.removesuffix("k"))
    return kbps * 1000 if bits_per_second else kbps


def _continuation_context(req: TTSRequest, engine) -> str | None:
    """prefix_text, when the first segment's verbalization reads it as context (VERBALIZE_PREFIX_CONTEXT)."""
    if req.prefix_text and req.prefix_text.strip() and engine.config.verbalize_prefix_context:
        return req.prefix_text
    return None


async def _verbalize_segment(engine, segment: str, context: str | None) -> str:
    if context:
        return await engine.verbalizer.verbalize_continuation(context, segment)
    return await engine.verbalizer.verbalize(segment)


async def synthesize(req: TTSRequest) -> dict:
    if req.priority == "high":
        async with get_gate().high():
            return await _synthesize(req)
    return await _synthesize(req)


async def _synthesize_flash(req: TTSRequest, engine) -> dict:
    """/predict with TontaubeV1.1 Flash: edge trim and encoding as for TontaubeV1.1, on the audio as the engine makes
    it."""
    error = _flash_option_error(req)
    if error:
        raise TTSRequestOptionError(error)
    audio = await engine.generate(text=req.text, **_flash_kwargs(req))
    if not len(audio):
        raise TTSRequestOptionError("text contains nothing to speak")
    sample_rate = engine.sample_rate
    if req.trim_silence_padding_ms is not None:
        audio = _trim_boundary_silence(audio, padding_ms=req.trim_silence_padding_ms, sample_rate=sample_rate)
    audio_bytes, mime_type = _encode_audio(audio, req.format, req.bitrate, sample_rate=sample_rate)
    return TTSResponse(
        audio_b64=base64.b64encode(audio_bytes).decode("ascii"),
        mime_type=mime_type,
        sample_rate=sample_rate,
    ).model_dump()


async def _synthesize(req: TTSRequest) -> dict:
    engine = await _get_engine()
    if _flash():
        return await _synthesize_flash(req, engine)
    mode = _require_verbalizer(req, engine)
    _require_vibevoice(req, engine)
    _require_timestamps(req, engine)
    raw = _prepare_raw_text(req.text)
    segment_controls = req.segment_controls
    if mode == VERBALIZER_RULES:
        from app import verbalize_rules

        raw = [verbalize_rules.verbalize(segment) for segment in raw] if isinstance(raw, list) \
            else verbalize_rules.verbalize(raw)
    elif mode == VERBALIZER_CONTEXTUAL:
        context = _continuation_context(req, engine)   # only the first segment continues the prefix
        if isinstance(raw, str) and context is None:
            # One text without a prefix: the verbalizer reads it whole and cuts it at paragraphs itself,
            # so tables, headings and lists keep their context. The spoken text is split below.
            raw, segment_controls = await engine.verbalizer.verbalize(raw), None
        else:
            raw, segment_controls = _split_with_controls(
                raw, segment_controls, digit_weight=4,
                chunk_scales=_chunk_scales(req, engine, raw, segment_controls),
            )
            if isinstance(raw, list):
                raw = [
                    await _verbalize_segment(engine, segment, context if index == 0 else None)
                    for index, segment in enumerate(raw)
                ]
            else:
                raw = await _verbalize_segment(engine, raw, context)
    text = (
        [sanitize_text_for_tts(segment) for segment in raw]
        if isinstance(raw, list)
        else sanitize_text_for_tts(raw)
    )
    text, segment_controls = _split_with_controls(
        text, segment_controls,
        chunk_scales=_chunk_scales(req, engine, text, segment_controls),
    )
    _require_controls(req, engine, text, segment_controls)

    generated = await engine.generate(
        text=text,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        frequency_penalty=req.frequency_penalty,
        silence_logit_bias=req.silence_logit_bias,
        seed=req.seed,
        voice_path=req.voice_path,
        voice_audio_b64=req.voice_audio_b64,
        voice_tokens=req.voice_tokens,
        prefix_text=req.prefix_text or None,
        prefix_tokens=req.prefix_tokens,
        max_new_tokens=req.max_new_tokens,
        vibevoice_postprocess=req.vibevoice_postprocess,
        acoustic_temperature=req.acoustic_temperature,
        acoustic_top_k=req.acoustic_top_k,
        prompt_max_tokens=req.prompt_max_tokens,
        language=req.language,
        tag=req.tag,
        use_system_tag=req.use_system_tag,
        vllm_priority=req.vllm_priority,
        controls=req.controls,
        segment_controls=segment_controls,
        **({"return_piece_frames": True} if req.include_timestamps else {}),
    )
    audio, codebook_tokens = generated[0], generated[1]
    alignments = timeline = None
    if req.include_timestamps:
        # include_timestamps: align each piece on its own CB0 codes now; the times are on the decoded audio and
        # `timeline` carries them through the edge trim and the pause cap below (app/timestamps.py).
        from app import timestamps as ts

        pieces = [text] if isinstance(text, str) else list(text)
        alignments = await asyncio.to_thread(
            ts.align_pieces_safe, engine.cb0_ctc, codebook_tokens[0], generated[2], pieces,
        )
        timeline = ts.Timeline()

    output_sample_rate = DUALCODEC_SAMPLE_RATE
    # engine.generate scaled the audio by
    # -OUTPUT_HEADROOM_DB dB. The edge trim and the pause cap lower their fixed
    # thresholds by the same dB, so their cut points match the unscaled audio
    # and the headroom is a pure level change (0 dB: the old thresholds).
    headroom_db = engine.config.output_headroom_db
    if req.trim_silence_padding_ms is not None:
        bounds = _trim_bounds(
            audio,
            padding_ms=req.trim_silence_padding_ms,
            headroom_db=headroom_db,
            # PREFIX_LEAD_TRIM_PADDING_MS: a shorter LEADING edge for a continuation (prefixed request); the tail keeps the request's
            lead_padding_ms=engine.config.prefix_lead_trim_padding_ms if req.prefix_text else None,
        )
        if bounds is not None:
            audio = audio[..., bounds[0]:bounds[1]]
            if timeline is not None:
                timeline.trim(bounds[0], bounds[1], output_sample_rate)
    if req.mossformer2_postprocess:
        pause_cuts = [] if timeline is not None else None
        cap_rate = output_sample_rate
        audio, output_sample_rate = await asyncio.to_thread(
            _apply_mossformer2_postprocessing,
            audio,
            output_sample_rate,
            headroom_db=headroom_db,
            **({"cuts_out": pause_cuts} if pause_cuts is not None else {}),
        )
        if timeline is not None:
            timeline.cut(pause_cuts, cap_rate)  # the cuts the cap applied, in samples at its input rate
    if req.prefix_text and engine.config.prefix_start_fade:
        # PREFIX_START_FADE: the client splices this continuation after its own audio; fade the
        # response's first prefix_start_fade_ms in, on the final audio (after trim / pause cap / MossFormer2).
        from app.dualcodec_utils import start_fade

        audio = start_fade(audio, engine.config.prefix_start_fade_ms, output_sample_rate)
    timestamps = None
    if timeline is not None:
        from app import timestamps as ts

        timeline.finish(audio.shape[-1] / output_sample_rate)
        timestamps = ts.assemble(alignments, timeline)
    audio_bytes, mime_type = _encode_audio(
        audio,
        req.format,
        req.bitrate,
        sample_rate=output_sample_rate,
    )
    return TTSResponse(
        audio_b64=base64.b64encode(audio_bytes).decode("ascii"),
        mime_type=mime_type,
        sample_rate=output_sample_rate,
        codebook_tokens=codebook_tokens,
        timestamps=timestamps,
    ).model_dump()


# --- OpenAI-compatible speech route (/v1/audio/speech) ---
OPENAI_MEDIA_TYPES = {
    "mp3": "audio/mpeg", "opus": "audio/ogg", "aac": "audio/aac", "flac": "audio/flac", "wav": "audio/wav",
    "pcm": "audio/pcm",
}
OPENAI_PCM_RATE = 24000


class OpenAISpeechRequest(BaseModel):
    """The body of OpenAI's POST /v1/audio/speech."""

    model: str
    input: str
    voice: str | None = None
    response_format: Literal["mp3", "opus", "aac", "flac", "wav", "pcm"] = "mp3"
    speed: float | None = None
    instructions: str | None = None
    stream_format: str | None = None
    verbalizer: str | None = Field(
        default=None,
        description=f"Tontaube extension: {', '.join(VERBALIZER_MODES)} (see POST /predict); the model's default if omitted.",
    )

    @field_validator("verbalizer")
    @classmethod
    def known_verbalizer(cls, value: str | None) -> str | None:
        return _verbalizer_name(value)

    @field_validator("input")
    @classmethod
    def input_like_predict(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("input must not be empty")
        if len(value) > MAX_REQUEST_TEXT_CHARS:
            raise ValueError(f"input must contain at most {MAX_REQUEST_TEXT_CHARS} characters")
        return value


def _openai_error_body(message: str, status_code: int = 400) -> dict:
    kind = "invalid_request_error" if status_code < 500 else "server_error"
    return {"error": {"message": str(message), "type": kind, "code": None}}


def _validation_message(exc: Exception) -> str:
    errors = getattr(exc, "errors", None)
    if errors is None:
        return "request body must be JSON"
    return "; ".join(
        f"{'.'.join(str(part) for part in error['loc']) or 'body'}: {error['msg'].removeprefix('Value error, ')}"
        for error in errors()
    )


def _openai_voice_path(voice: str | None) -> str | None:
    """The VOICE_PATH voice named `voice` (case-insensitive), or None for the default voice."""
    if not voice:
        return None
    from app.voice_catalog import available_voice_files

    root = Path(os.getenv("VOICE_PATH", str(BUNDLED_VOICE_PATH))).expanduser().resolve()
    wanted = voice.strip().casefold()
    return next((str(path) for path, _ in available_voice_files(root) if path.stem.casefold() == wanted), None)


def _ffmpeg_found() -> bool:
    import shutil

    return shutil.which(AudioSegment.converter) is not None


def _float_samples(audio) -> np.ndarray:
    if isinstance(audio, np.ndarray):
        return audio.reshape(-1).astype(np.float32, copy=False)
    return audio.squeeze().cpu().float().numpy().reshape(-1)


class _OpusFramedWriter:
    """/stream with format "opus": the framed Opus of /ws/stream (app/opus_stream.py OpusStreamEncoder: a frame count
    and each 20 ms frame with its length, little-endian uint16) in each event's `audio_b64`."""

    def __init__(self, sample_rate: int, bitrate: int):
        from app.opus_stream import OpusStreamEncoder

        self._encoder = OpusStreamEncoder(sample_rate, bitrate=bitrate)

    def write_chunk(self, audio) -> bytes:
        return self._encoder.encode(audio)

    def close(self) -> bytes:
        return self._encoder.flush()


STREAM_FORMATS = ("pcm16", "mp3", "opus")


def _stream_format(req: TTSRequest) -> str:
    """The output format of /stream: pcm16 unless the request names one (mp3, opus; `wav` is not a stream format)."""
    return req.format if "format" in req.model_fields_set else "pcm16"


class _PcmWriter:
    """16-bit little-endian mono PCM at `rate`: the engine's own (/stream, a plain copy) or OpenAI's 24 kHz, for
    which a 48 kHz engine's audio goes through a Kaiser-windowed sinc low-pass (11 kHz, about 86 dB stopband from
    12 kHz) and keeps every second sample."""

    def __init__(self, sample_rate: int, rate: int = OPENAI_PCM_RATE):
        from app.stream_writers import pcm16

        if sample_rate % rate:
            raise ValueError(f"cannot make {rate} Hz pcm from {sample_rate} Hz")
        self._pcm16, self._step, self._in, self._out = pcm16, sample_rate // rate, 0, 0
        n = 80 * self._step + 1
        t = np.arange(n) - n // 2
        cutoff = 11000 / sample_rate
        taps = 2 * cutoff * np.sinc(2 * cutoff * t) * np.kaiser(n, 8.6)
        self._taps = taps / taps.sum()
        self._buf = np.zeros(n // 2)        # output k is centred on input sample k * step

    def _decimate(self, samples: np.ndarray) -> np.ndarray:
        if self._step == 1:
            return samples
        self._buf = np.concatenate([self._buf, samples])
        n = len(self._taps)
        count = (len(self._buf) - n) // self._step + 1 if len(self._buf) >= n else 0
        if count == 0:
            return np.zeros(0)
        out = sliding_window_view(self._buf, n)[: count * self._step: self._step] @ self._taps
        self._buf = self._buf[count * self._step:]
        return out

    def write_chunk(self, audio) -> bytes:
        samples = _float_samples(audio)
        self._in += len(samples)
        out = self._decimate(samples)
        self._out += len(out)
        return self._pcm16(out).astype("<i2").tobytes()

    def close(self) -> bytes:
        if self._step == 1:
            return b""
        out = self._decimate(np.zeros(len(self._taps)))[: -(-self._in // self._step) - self._out]
        self._out += len(out)
        return self._pcm16(out).astype("<i2").tobytes()


def _ogg_crc_table() -> list[int]:
    table = []
    for byte in range(256):
        crc = byte << 24
        for _ in range(8):
            crc = ((crc << 1) ^ 0x04C11DB7 if crc & 0x80000000 else crc << 1) & 0xFFFFFFFF
        table.append(crc)
    return table


_OGG_CRC = _ogg_crc_table()


class _OggOpusWriter:
    """Ogg Opus (RFC 7845), streamable: 20 ms Opus packets, one or more pages per chunk. The last page's granule
    position ends the stream on the last real sample."""

    def __init__(self, sample_rate: int, bitrate: int):
        import struct

        import opuslib

        from app.stream_writers import pcm16

        self._struct, self._pcm16 = struct, pcm16
        self._encoder = opuslib.Encoder(sample_rate, 1, opuslib.APPLICATION_AUDIO)
        self._encoder.bitrate = bitrate
        self._frame, self._scale = sample_rate // 50, 48000 // sample_rate
        self._lookahead = self._encoder.lookahead
        self._rest = np.zeros(0, np.int16)
        self._serial, self._seq, self._granule, self._samples, self._encoded = secrets.randbits(32), 0, 0, 0, 0
        head = b"OpusHead" + struct.pack("<BBHIhB", 1, 1, self._lookahead * self._scale, sample_rate, 0, 0)
        vendor = b"Tontaube"
        tags = b"OpusTags" + struct.pack("<I", len(vendor)) + vendor + struct.pack("<I", 0)
        self._header = self._page([head], 0, 0x02) + self._page([tags], 0, 0)

    def _page(self, packets: list[bytes], granule: int, flags: int) -> bytes:
        lacing = b"".join(bytes([255] * (len(p) // 255) + [len(p) % 255]) for p in packets)
        page = bytearray(self._struct.pack("<4sBBqIIIB", b"OggS", 0, flags, granule, self._serial, self._seq, 0,
                                           len(lacing)) + lacing + b"".join(packets))
        self._seq += 1
        crc = 0
        for byte in page:
            crc = ((crc << 8) & 0xFFFFFFFF) ^ _OGG_CRC[(crc >> 24) ^ byte]
        page[22:26] = self._struct.pack("<I", crc)
        return bytes(page)

    def _pages(self, pcm: np.ndarray, last: bool = False) -> bytes:
        header, self._header = self._header, b""
        fs = self._frame
        packets = [self._encoder.encode(pcm[i:i + fs].tobytes(), fs) for i in range(0, len(pcm) - fs + 1, fs)]
        self._encoded += len(packets) * fs
        pages, group, segments = [header], [], 0
        for index, packet in enumerate(packets):
            group.append(packet)
            segments += len(packet) // 255 + 1
            self._granule += fs * self._scale
            final = last and index == len(packets) - 1
            if final or index == len(packets) - 1 or segments > 200:
                granule = (self._lookahead + self._samples) * self._scale if final else self._granule
                pages.append(self._page(group, granule, 0x04 if final else 0))
                group, segments = [], 0
        return b"".join(pages)

    def write_chunk(self, audio) -> bytes:
        pcm = self._pcm16(audio)
        self._samples += len(pcm)
        pcm = np.concatenate([self._rest, pcm])
        whole = len(pcm) // self._frame * self._frame
        self._rest = pcm[whole:]
        return self._pages(pcm[:whole])

    def close(self) -> bytes:
        """The rest, zero-padded past the encoder's lookahead so that every real sample is decoded."""
        needed = self._samples + self._lookahead - self._encoded
        pcm = np.zeros(-(-needed // self._frame) * self._frame, np.int16)
        pcm[:len(self._rest)] = self._rest
        return self._pages(pcm, last=True)


def create_app():
    import asyncio as _asyncio

    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse
    from fastapi import status
    from app.config import CORS_ALLOWED_ORIGINS, REQUIRE_API_KEY, TTS_API_KEY
    from app.stream_writers import Mp3StreamWriter

    _enable_streaming = os.getenv("ENABLE_STREAMING", "1") == "1"
    _engine_ready = _asyncio.Event()
    _server_config = ServerConfig()
    # TontaubeV1.1 Flash queues requests for its engines itself (app/flash.py).
    _request_slots = contextlib.nullcontext() if _flash() else _asyncio.BoundedSemaphore(_server_config.max_inflight_requests)
    _stream_calibration: dict = {"value": None, "warned": False}   # StreamCalibration once measured (lifespan)

    async def _calibrate_streaming(engine):
        """Stream CALIBRATION_TEXT once through the normal streaming path; None (fixed default buffer) on failure."""
        from app.stream_calibration import CALIBRATION_INITIAL_SECONDS, CALIBRATION_TEXT, StreamCalibration

        started = time.perf_counter()
        try:
            req = TTSRequest(
                text=list(CALIBRATION_TEXT),
                use_verbalization=getattr(engine, "verbalizer", None) is not None,
                streaming_initial_seconds=CALIBRATION_INITIAL_SECONDS,
            )
            text, preprocessor, segment_controls = _prepare_streaming_request(req, engine)
            timings: list = []
            async for _chunk in engine.generate_streaming(
                text=text, timings=timings, **_streaming_kwargs(req, preprocessor, segment_controls),
            ):
                pass
            calibration = StreamCalibration.from_timings(
                timings, [len(piece) for piece in CALIBRATION_TEXT], time.perf_counter() - started,
            )
        except ValueError as exc:
            logger.warning("Stream calibration failed (%s); streams without streaming_initial_seconds use %.1f s",
                           exc, DEFAULT_STREAMING_INITIAL_SECONDS)
            return None
        except Exception:
            logger.exception("Stream calibration failed; streams without streaming_initial_seconds use %.1f s",
                             DEFAULT_STREAMING_INITIAL_SECONDS)
            return None
        logger.info("Stream calibration: %s", calibration.summary(STREAM_AUTO_MIN_INITIAL_SECONDS,
                                                                  STREAM_CALIBRATION_MARGIN))
        if calibration.rate < 1.0:
            logger.warning(
                "Stream calibration: this machine generates slower than real time (%.2fx); long streams will "
                "pause between chunks whatever the start buffer. Use /predict for long texts.", calibration.rate,
            )
        return calibration

    def _initial_seconds(req: TTSRequest, engine, text) -> float:
        """streaming_initial_seconds for this stream: the request's own value (with a warning), else the
        calibrated choice for its segments (app/stream_calibration.py), else the fixed default."""
        from app.stream_calibration import quantize_down

        calibration = _stream_calibration["value"]
        pieces = [text] if isinstance(text, str) else list(text)
        config = getattr(engine, "config", None)
        max_tokens = req.max_new_tokens or getattr(config, "max_new_tokens", MAX_SEMANTIC_TOKENS_PER_CHUNK)
        by_tokens = quantize_down((max_tokens - STREAMING_DC_PAD_TOKENS) / DUALCODEC_HZ)
        ceiling = min(MAX_STREAMING_INITIAL_SECONDS, max(MIN_STREAMING_INITIAL_SECONDS, by_tokens))
        chars = [len(piece) for piece in pieces]
        explicit = "streaming_initial_seconds" in req.model_fields_set
        if calibration is None:
            if not STREAM_CALIBRATION and STREAM_CALIBRATION_WARNING and not _stream_calibration["warned"]:
                _stream_calibration["warned"] = True
                logger.warning(
                    "Stream calibration is off: streams without streaming_initial_seconds start with %.1f s of audio, "
                    "which can pause playback on slower machines. STREAM_CALIBRATION=1 measures this machine at "
                    "startup; STREAM_CALIBRATION_WARNING=0 hides this message.", DEFAULT_STREAMING_INITIAL_SECONDS,
                )
            return req.streaming_initial_seconds if explicit else min(DEFAULT_STREAMING_INITIAL_SECONDS, ceiling)
        if explicit:
            needed, pauses = calibration.plan(chars, floor=MIN_STREAMING_INITIAL_SECONDS, ceiling=ceiling,
                                              margin=STREAM_CALIBRATION_MARGIN)
            note = ("; playback may pause on this machine" if req.streaming_initial_seconds < needed or pauses
                    else "; the first audio comes later than needed" if req.streaming_initial_seconds > needed else "")
            logger.warning(
                "streaming_initial_seconds=%.1f set explicitly; the calibrated start buffer for this request "
                "is %.1f s%s", req.streaming_initial_seconds, needed, note,
            )
            return req.streaming_initial_seconds
        initial, pauses = calibration.plan(chars, floor=STREAM_AUTO_MIN_INITIAL_SECONDS, ceiling=ceiling,
                                           margin=STREAM_CALIBRATION_MARGIN)
        log = logger.warning if pauses else logger.info
        log("Stream start buffer %.1f s for %d segment(s), first %d characters%s", initial, len(chars), chars[0],
            "; playback is expected to pause even so" if pauses else "")
        return initial

    @asynccontextmanager
    async def lifespan(_app):
        if REQUIRE_API_KEY and not TTS_API_KEY:
            raise RuntimeError("TTS_API_KEY is required when REQUIRE_API_KEY=1")
        engine = await _get_engine()
        if not _flash():
            logger.info(
                "End-to-end request concurrency limited to %d",
                _server_config.max_inflight_requests,
            )
        if _enable_streaming and STREAM_CALIBRATION and getattr(engine, "vv_tokenizer", None) is not None:
            _stream_calibration["value"] = await _calibrate_streaming(engine)
        _engine_ready.set()
        try:
            yield
        finally:
            _unload_engine()

    app = FastAPI(title="Tontaube TTS API", version=API_VERSION, lifespan=lifespan)

    @app.exception_handler(TTSRequestOptionError)
    async def request_option_error(_request: Request, exc: TTSRequestOptionError):
        return JSONResponse(
            content=_error_body(str(exc), getattr(exc, "code", None)),
            status_code=422,
        )

    @app.exception_handler(EngineBusy)
    async def engine_busy(_request: Request, exc: EngineBusy):
        return JSONResponse(content=_error_body(str(exc), "engine_busy"), status_code=503)

    @app.exception_handler(UnsupportedOption)
    async def unsupported_option(_request: Request, exc: UnsupportedOption):
        return JSONResponse(content=_error_body(str(exc)), status_code=422)

    @app.exception_handler(VoicePromptError)
    async def voice_prompt_error(_request: Request, exc: VoicePromptError):
        return JSONResponse(
            content=_error_body(str(exc)),
            status_code=422,
        )

    if CORS_ALLOWED_ORIGINS:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(CORS_ALLOWED_ORIGINS),
            allow_credentials=False,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["content-type", "x-api-key"],
        )

    @app.get("/")
    async def root():
        return {"name": "Tontaube TTS API", "docs": "/docs"}

    @app.get("/api")
    async def api_info():
        return {"name": "Tontaube TTS API", "docs": "/docs"}

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/controls")
    async def controls_catalog():
        """Style sliders of the loaded model, for the `controls` / `segment_controls` request fields. A model
        that reads only some request fields (TontaubeV1.1 Flash) also lists those as `fields`, with its name."""
        if not _engine_ready.is_set():
            return JSONResponse(content={"error": "Model still loading"}, status_code=503)
        if _flash():
            engine = await _get_engine()
            return {"controls": [], "model": "TontaubeV1.1 Flash", "fields": sorted(FLASH_FIELDS - {"language"}),
                    "device": _device_label(engine)}
        engine = await _get_engine()
        catalog = engine.controls.catalog() if engine.controls is not None else []
        return {"controls": catalog, "device": _device_label(engine)}

    @app.get("/readyz")
    async def readyz():
        if _engine_ready.is_set():
            return {"status": "ready"}
        return JSONResponse(content={"status": "loading"}, status_code=503)

    @app.middleware("http")
    async def tts_api_key_guard(request: Request, call_next):
        if REQUIRE_API_KEY and request.method != "OPTIONS":
            open_paths = {"/", "/api", "/healthz", "/readyz"}
            if request.url.path.startswith("/v1/"):
                # OpenAI clients send the key as a bearer token.
                scheme, _, token = request.headers.get("authorization", "").partition(" ")
                keys = [request.headers.get("x-api-key"), token.strip() if scheme.lower() == "bearer" else None]
                if not any(key and secrets.compare_digest(key.encode(), TTS_API_KEY.encode()) for key in keys):
                    return JSONResponse(content=_openai_error_body("unauthorized", 401),
                                        status_code=status.HTTP_401_UNAUTHORIZED)
            elif request.url.path not in open_paths:
                provided_key = request.headers.get("x-api-key")
                if not provided_key or not secrets.compare_digest(provided_key, TTS_API_KEY):
                    return JSONResponse(
                        content=_error_body("unauthorized"),
                        status_code=status.HTTP_401_UNAUTHORIZED,
                    )
        return await call_next(request)

    def _make_text_preprocessor(req: TTSRequest, engine):
        """Build an async callback that verbalizes + sanitizes a single segment.

        Used by streaming paths so segments are processed just-in-time instead
        of all-upfront, overlapping verbalization with audio generation.
        Returns None if no verbalization is needed (segments already sanitized).
        """
        mode = _require_verbalizer(req, engine)
        if mode == VERBALIZER_CONTEXTUAL:
            # generate_streaming preprocesses segments[0] first; only it continues the prefix.
            pending_context = [_continuation_context(req, engine)]

            async def preprocess(text: str) -> str:
                context, pending_context[0] = pending_context[0], None
                spoken = await _verbalize_segment(engine, text, context)
                logger.debug("verbalizer %s: %r -> %r", mode, text, spoken)
                return sanitize_text_for_tts(spoken)
            return preprocess
        if mode == VERBALIZER_RULES:
            from app import verbalize_rules

            async def preprocess_rules(text: str) -> str:
                spoken = verbalize_rules.verbalize(text)
                logger.debug("verbalizer %s: %r -> %r", mode, text, spoken)
                return sanitize_text_for_tts(spoken)
            return preprocess_rules
        return None

    def _prepare_streaming_request(req: TTSRequest, engine):
        preprocessor = _make_text_preprocessor(req, engine)
        text = _prepare_raw_text(req.text)
        if preprocessor is None:
            text = (
                [sanitize_text_for_tts(segment) for segment in text]
                if isinstance(text, list)
                else sanitize_text_for_tts(text)
            )
        text, segment_controls = _split_with_controls(
            text, req.segment_controls,
            digit_weight=4 if preprocessor is not None else 1,
            chunk_scales=_chunk_scales(req, engine, text, req.segment_controls),
        )
        _require_controls(req, engine, text, segment_controls)
        return text, preprocessor, segment_controls

    def _stream_option_error(req: TTSRequest, transport: Literal["mp3", "opus"]) -> str | None:
        return (_flash_option_error(req) if _flash() else None) or _streaming_option_error(req, transport)

    def _stream_inputs(req: TTSRequest, engine) -> tuple:
        """(text, generate_streaming keyword arguments) of a stream request."""
        if _flash():
            return req.text, _flash_kwargs(req)
        text, preprocessor, segment_controls = _prepare_streaming_request(req, engine)
        initial_seconds = _initial_seconds(req, engine, text)
        return text, _streaming_kwargs(req, preprocessor, segment_controls, initial_seconds)

    def _streaming_kwargs(req: TTSRequest, preprocessor, segment_controls=None, initial_seconds=None) -> dict:
        if initial_seconds is None:
            initial_seconds = req.streaming_initial_seconds
        return {
            "controls": req.controls,
            "segment_controls": segment_controls,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "top_k": req.top_k,
            "frequency_penalty": req.frequency_penalty,
            "silence_logit_bias": req.silence_logit_bias,
            "seed": req.seed,
            "voice_path": req.voice_path,
            "voice_audio_b64": req.voice_audio_b64,
            "voice_tokens": req.voice_tokens,
            "prefix_text": req.prefix_text or None,
            "prefix_tokens": req.prefix_tokens,
            "max_new_tokens": req.max_new_tokens,
            "streaming_initial_seconds": initial_seconds,
            "acoustic_temperature": req.acoustic_temperature,
            "acoustic_top_k": req.acoustic_top_k,
            "prompt_max_tokens": req.prompt_max_tokens,
            "text_preprocessor": preprocessor,
            "language": req.language,
            "tag": req.tag,
            "use_system_tag": req.use_system_tag,
        }

    @app.post(
        "/predict",
        response_model=TTSResponse,
        description="Generate one encoded audio response.",
    )
    async def predict(req: TTSRequest):
        if not _engine_ready.is_set():
            return JSONResponse(content={"error": "Model still loading"}, status_code=503)
        if req.format == "pcm16":
            return JSONResponse(content=_error_body('format "pcm16" is for POST /stream only'), status_code=422)
        async with _request_slots:
            result = await synthesize(req)
        return JSONResponse(content=result)

    @app.post(
        "/encode_voice",
        response_model=EncodeVoiceResponse,
        description="Encode a voice reference once into `voice_tokens` for later requests.",
    )
    async def encode_voice(req: EncodeVoiceRequest):
        if not _engine_ready.is_set():
            return JSONResponse(content={"error": "Model still loading"}, status_code=503)
        if _flash():
            return JSONResponse(content=_error_body("TontaubeV1.1 Flash has one voice and takes no voice tokens"),
                                status_code=422)
        async with _request_slots:
            engine = await _get_engine()
            started = time.perf_counter()
            rows = await _asyncio.to_thread(
                engine.encode_voice, req.voice_audio_b64, req.prompt_max_tokens, req.n_codebooks,
            )
        n_tokens = len(rows[0])
        logger.info(
            "Encoded voice: %d tokens x %d codebooks in %.2fs",
            n_tokens, len(rows), time.perf_counter() - started,
        )
        return EncodeVoiceResponse(
            voice_tokens=rows,
            n_tokens=n_tokens,
            n_codebooks=len(rows),
            duration_seconds=n_tokens / DUALCODEC_HZ,
            token_rate_hz=DUALCODEC_HZ,
            model_version=MODEL_VERSION,
            codec=DUALCODEC_CODEC_ID,
        )

    # --- Streaming routes ---
    if _enable_streaming:
        async def _audio_sse_generator(pieces, writer, label="STREAM", priority_high: bool = False,
                                       sample_rate: int | None = None):
            """Shared SSE audio streaming generator over the engine's audio pieces.

            Yields SSE `data:` lines with base64-encoded audio chunks, each with
            `sample_rate` when one is given (raw PCM, which does not carry it). When
            `priority_high` is true, generation runs inside the priority gate's
            HIGH scope so low-priority requests park between vLLM calls.
            """
            gate_ctx = get_gate().high() if priority_high else None
            rate = {} if sample_rate is None else {"sample_rate": sample_rate}
            async with _request_slots:
                if gate_ctx is not None:
                    await gate_ctx.__aenter__()
                try:
                    chunk_idx = 0
                    t_stream_start = time.monotonic()
                    try:
                        async for audio_chunk in pieces:
                            t0 = time.monotonic()
                            encoded_bytes = writer.write_chunk(audio_chunk)
                            t1 = time.monotonic()
                            audio_b64 = base64.b64encode(encoded_bytes).decode('ascii')
                            event = json.dumps({"index": chunk_idx, "audio_b64": audio_b64, **rate, "done": False})
                            t2 = time.monotonic()
                            if chunk_idx == 0:
                                logger.debug(
                                    "%s first chunk: inference=%.3fs encode=%.3fs "
                                    "serialization=%.3fs total=%.3fs",
                                    label,
                                    t0 - t_stream_start,
                                    t1 - t0,
                                    t2 - t1,
                                    t2 - t_stream_start,
                                )
                            yield f"data: {event}\n\n"
                            chunk_idx += 1
                        final_bytes = writer.close()
                        if final_bytes:
                            audio_b64 = base64.b64encode(final_bytes).decode('ascii')
                            yield f"data: {json.dumps({'index': chunk_idx, 'audio_b64': audio_b64, **rate, 'done': False})}\n\n"
                        yield f"data: {json.dumps({'done': True})}\n\n"
                    except _asyncio.CancelledError:
                        logger.info(
                            "%s client disconnected after %d chunks (%.1fs)",
                            label,
                            chunk_idx,
                            time.monotonic() - t_stream_start,
                        )
                        writer.close()
                        return
                    except Exception:
                        logger.exception("%s streaming request failed", label)
                        writer.close()
                        yield f"data: {json.dumps({'error': 'streaming generation failed', 'done': True})}\n\n"
                finally:
                    if gate_ctx is not None:
                        await gate_ctx.__aexit__(None, None, None)

        async def _stream_response(req: TTSRequest):
            """Build an SSE response (raw PCM16 by default; format "mp3" or "opus"), honoring bitrate and priority."""
            transport = _stream_format(req)
            if transport not in STREAM_FORMATS:
                return JSONResponse(content=_error_body(
                    f"format {req.format!r} is not a stream format; POST /stream outputs pcm16 (default), mp3 or opus"),
                    status_code=422)
            option_error = _stream_option_error(req, transport)
            if option_error:
                return JSONResponse(content=_error_body(option_error), status_code=422)
            if transport == "opus":
                try:  # opuslib needs the Opus library (libopus), which Windows and macOS do not have by default
                    from app.opus_stream import OpusStreamEncoder  # noqa: F401
                except Exception as exc:  # noqa: BLE001
                    return JSONResponse(content=_error_body(
                        f"format 'opus' needs the Opus library (libopus), which this server does not have ({exc}); "
                        "use pcm16 or mp3", "opus_unavailable"), status_code=422)
            engine = await _get_engine()
            if not engine.can_stream:
                return JSONResponse(
                    content={"error": "streaming is unavailable because VibeVoice is not loaded"},
                    status_code=503,
                )
            text, gen_kwargs = _stream_inputs(req, engine)
            # TontaubeV1.1 starts generating on the first read; Flash takes the request here (or refuses: 503).
            pieces = engine.generate_streaming(text=text, **gen_kwargs)
            sample_rate = getattr(engine, "sample_rate", DUALCODEC_SAMPLE_RATE)
            if transport == "pcm16":
                # The engine's own rate (24 kHz from TontaubeV1.1, 48 kHz from Flash by default), named in every event.
                writer = _PcmWriter(sample_rate, sample_rate)
            elif transport == "opus":
                writer = _OpusFramedWriter(sample_rate, bitrate=_bitrate_number(req.bitrate, bits_per_second=True))
            else:
                writer = Mp3StreamWriter(sample_rate, bitrate=_bitrate_number(req.bitrate, bits_per_second=False))

            return StreamingResponse(
                _audio_sse_generator(pieces, writer, priority_high=req.priority == "high",
                                     sample_rate=sample_rate if transport == "pcm16" else None),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        @app.post(
            "/stream",
            description=(
                "Stream audio chunks as server-sent events. `format`: \"pcm16\" (default) raw 16-bit little-endian "
                "mono PCM at the engine's sample rate (in each event's sample_rate), no codec delay; \"mp3\" MP3 chunks; "
                "\"opus\" the framed Opus of /ws/stream (a uint16 frame count, then each 20 ms frame as a uint16 "
                "length and its bytes). Every event is {index, audio_b64, done}, the audio base64-encoded; the last "
                "is {done: true}. Streaming requires "
                "VibeVoice; bitrate (MP3 and Opus) and priority are honored."
            ),
        )
        async def stream_tts(req: TTSRequest):
            if not _engine_ready.is_set():
                return JSONResponse(content={"error": "Model still loading"}, status_code=503)
            return await _stream_response(req)

        @app.websocket("/ws/stream")
        async def ws_stream_tts(ws: WebSocket):
            origin = ws.headers.get("origin")
            if origin and origin not in CORS_ALLOWED_ORIGINS:
                await ws.close(code=4403)
                return
            if REQUIRE_API_KEY:
                provided_key = ws.headers.get("x-api-key")
                if not provided_key or not secrets.compare_digest(provided_key, TTS_API_KEY):
                    await ws.close(code=4401)
                    return
            await ws.accept()
            try:
                payload = await ws.receive_json()
                req = TTSRequest.model_validate(payload)
                option_error = _stream_option_error(req, "opus")
                if option_error:
                    await ws.send_json(_error_body(option_error))
                    await ws.close(code=1008)
                    return
                try:  # opuslib needs the Opus library (libopus), which Windows and macOS do not have by default
                    from app.opus_stream import OpusStreamEncoder as _OpusStreamEncoder
                except Exception as exc:  # noqa: BLE001
                    await ws.send_json(_error_body(f"Opus streaming is unavailable on this server: {exc}"))
                    await ws.close(code=1011)
                    return
                engine = await _get_engine()
                if not engine.can_stream:
                    await ws.send_json({
                        "error": "streaming is unavailable because VibeVoice is not loaded"
                    })
                    await ws.close(code=1013)
                    return
                try:
                    text, gen_kwargs = _stream_inputs(req, engine)
                except (TTSRequestOptionError, UnsupportedOption) as exc:
                    await ws.send_json(_error_body(str(exc), getattr(exc, "code", None)))
                    await ws.close(code=1008)
                    return
                try:
                    pieces = engine.generate_streaming(text=text, **gen_kwargs)
                except (EngineBusy, UnsupportedOption) as exc:
                    busy = isinstance(exc, EngineBusy)
                    await ws.send_json(_error_body(str(exc), "engine_busy" if busy else None))
                    await ws.close(code=1013 if busy else 1008)
                    return
                opus_enc = _OpusStreamEncoder(
                    getattr(engine, "sample_rate", DUALCODEC_SAMPLE_RATE),
                    bitrate=_bitrate_number(req.bitrate, bits_per_second=True),
                )
                async with _request_slots:
                    gate_ctx = get_gate().high() if req.priority == "high" else None
                    if gate_ctx is not None:
                        await gate_ctx.__aenter__()
                    try:
                        async for audio_chunk in pieces:
                            await ws.send_bytes(opus_enc.encode(audio_chunk))
                        tail = opus_enc.flush()
                        if tail:
                            await ws.send_bytes(tail)
                        await ws.send_json({"done": True})
                    finally:
                        if gate_ctx is not None:
                            await gate_ctx.__aexit__(None, None, None)
            except WebSocketDisconnect:
                pass
            except Exception:
                logger.exception("WebSocket streaming request failed")
                try:
                    await ws.send_json({"error": "streaming generation failed"})
                except Exception:
                    pass

        # --- OpenAI-compatible routes ---
        def _openai_error(message: str, status_code: int) -> JSONResponse:
            return JSONResponse(content=_openai_error_body(message, status_code), status_code=status_code)

        @app.get("/v1/models", description="OpenAI-compatible model list: the served model.")
        async def openai_models():
            name = "TontaubeV1.1 Flash" if _flash() else MODEL_VERSION
            return {"object": "list", "data": [{"id": name, "object": "model", "owned_by": "tontaube"}]}

        @app.post(
            "/v1/audio/speech",
            description=(
                "OpenAI-compatible speech (POST /v1/audio/speech). mp3, opus and pcm stream as they are made; "
                "wav, aac and flac come whole."
            ),
            openapi_extra={"requestBody": {"required": True, "content": {
                "application/json": {"schema": OpenAISpeechRequest.model_json_schema()}}}},
        )
        async def openai_speech(request: Request):
            from fastapi.responses import Response

            if not _engine_ready.is_set():
                return _openai_error("Model still loading", 503)
            try:
                body = OpenAISpeechRequest.model_validate(await request.json())
            except ValueError as exc:
                return _openai_error(_validation_message(exc), 400)
            if body.speed is not None and body.speed != 1.0:
                return _openai_error("speed is not supported", 400)
            if body.stream_format not in (None, "audio"):
                return _openai_error(f"stream_format {body.stream_format!r} is not supported; use 'audio'", 400)
            fmt = body.response_format
            if fmt in ("aac", "flac") and not _ffmpeg_found():
                return _openai_error(f"response_format {fmt!r} needs FFmpeg, which this server does not find; "
                                     "use mp3, opus, wav or pcm", 400)
            if fmt == "opus":
                try:
                    import opuslib  # noqa: F401
                except Exception:  # noqa: BLE001
                    return _openai_error("response_format 'opus' needs the Opus library (libopus), which this "
                                         "server does not have; use mp3, wav or pcm", 400)
            engine = await _get_engine()
            if not engine.can_stream:
                return _openai_error("speech is unavailable because VibeVoice is not loaded", 503)
            voice_path = None if _flash() else _openai_voice_path(body.voice)
            try:
                req = TTSRequest(
                    text=body.input,
                    **({"verbalizer": body.verbalizer} if body.verbalizer is not None else {}),
                    **({"voice_path": voice_path} if voice_path else {}),
                )
                text, gen_kwargs = _stream_inputs(req, engine)
                # The same generation as /stream: Flash's redraws are settled before audio is handed out.
                pieces = engine.generate_streaming(text=text, **gen_kwargs)
            except EngineBusy as exc:
                return _openai_error(str(exc), 503)
            except (TTSRequestOptionError, UnsupportedOption, VoicePromptError) as exc:
                return _openai_error(str(exc), 400)
            sample_rate = getattr(engine, "sample_rate", DUALCODEC_SAMPLE_RATE)

            if fmt in ("wav", "aac", "flac"):
                from app.stream_writers import pcm16

                async with _request_slots:
                    try:
                        audio = [pcm16(piece) async for piece in pieces]
                    except Exception:
                        logger.exception("OpenAI speech request failed")
                        return _openai_error("speech generation failed", 500)
                audio = np.concatenate(audio) if audio else np.zeros(0, np.int16)
                if not len(audio):
                    return _openai_error("input contains nothing to speak", 400)
                encoded, _ = await _asyncio.to_thread(
                    _encode_audio, audio, "adts" if fmt == "aac" else fmt, "64k", sample_rate,
                )
                return Response(content=encoded, media_type=OPENAI_MEDIA_TYPES[fmt])

            if fmt == "mp3":
                writer = Mp3StreamWriter(sample_rate, bitrate=64)
            elif fmt == "opus":
                writer = _OggOpusWriter(sample_rate, bitrate=64000)
            else:
                writer = _PcmWriter(sample_rate)

            async def audio_bytes():
                async with _request_slots:
                    try:
                        async for piece in pieces:
                            data = writer.write_chunk(piece)
                            if data:
                                yield bytes(data)
                        tail = writer.close()
                        if tail:
                            yield bytes(tail)
                    except Exception:
                        # Headers are out: end the response without its last chunk, so the client sees an error.
                        logger.exception("OpenAI speech stream failed")
                        raise

            return StreamingResponse(audio_bytes(), media_type=OPENAI_MEDIA_TYPES[fmt],
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


app = create_app()


def server_bind_address() -> tuple[str, int]:
    host = os.getenv("TTS_HOST", "127.0.0.1")
    port = int(os.getenv("TTS_PORT", os.getenv("PORT", "8080")))
    return host, port


def main(*, host: str | None = None, port: int | None = None) -> None:
    import uvicorn
    from app.preflight import main as run_preflight

    run_preflight()
    default_host, default_port = server_bind_address()
    uvicorn.run(
        app,
        host=default_host if host is None else host,
        port=default_port if port is None else port,
    )


if __name__ == "__main__":
    main()
