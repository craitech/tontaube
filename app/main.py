"""HTTP API for Tontaube text-to-speech inference."""
import asyncio
import math
import os
import multiprocessing
from importlib.metadata import PackageNotFoundError, version as package_version

if multiprocessing.get_start_method(allow_none=True) is None:
    multiprocessing.set_start_method('spawn')

import base64
from contextlib import asynccontextmanager
import io
import json
from pathlib import Path
import secrets
import time
from typing import Literal

import soundfile as sf
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
    VOICE_AUDIO_EXTENSIONS,
    ServerConfig,
    normalize_language,
)
from app.priority_gate import get_gate
from app.logging_utils import get_logger
from app.model.text_vocab import clean_text, strip_markup, strip_separators
from app.text import (
    MAX_MERGED_PIECE_CHARS,
    merge_short_pieces,
    sanitize_text_for_tts,
    split_long_text,
)
from app.utils import VoicePromptError

logger = get_logger("api")

try:
    API_VERSION = package_version("tontaube-inference")
except PackageNotFoundError:
    API_VERSION = "1.0.0"


async def _get_engine():
    from app.inference import get_engine

    return await get_engine()


def _unload_engine() -> None:
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
):
    """Trim excess edge silence while preserving all internal pauses.

    silence_threshold_dbfs is at the decoder's own level. For audio already
    scaled by OUTPUT_HEADROOM_DB, pass that headroom: the threshold moves
    down by the same dB, so the trim points do not depend on the headroom.
    headroom_db=0 is the old threshold.
    """
    bounds = _trim_bounds(
        audio, padding_ms=padding_ms, silence_threshold_dbfs=silence_threshold_dbfs,
        headroom_db=headroom_db, lead_padding_ms=lead_padding_ms,
    )
    return audio if bounds is None else audio[..., bounds[0]:bounds[1]]


def _trim_bounds(
    audio,
    *,
    padding_ms: int,
    silence_threshold_dbfs: float = -50.0,
    headroom_db: float = 0.0,
    lead_padding_ms: int | None = None,
) -> tuple[int, int] | None:
    """_trim_boundary_silence's kept samples [start, end), or None (audio unchanged)."""
    if padding_ms < 0 or audio.shape[-1] == 0:
        return None

    samples = audio.detach().float().reshape(-1, audio.shape[-1])
    mono_power = samples.square().mean(dim=0)
    frame_samples = max(1, round(DUALCODEC_SAMPLE_RATE * 0.02))
    hop_samples = max(1, round(DUALCODEC_SAMPLE_RATE * 0.01))
    if mono_power.numel() < frame_samples:
        return None

    frame_rms = mono_power.unfold(0, frame_samples, hop_samples).mean(dim=-1).sqrt()
    threshold = 10.0 ** ((silence_threshold_dbfs - headroom_db) / 20.0)
    active_frames = (frame_rms >= threshold).nonzero(as_tuple=False).flatten()
    if active_frames.numel() == 0:
        return None

    padding_samples = round(DUALCODEC_SAMPLE_RATE * padding_ms / 1000)
    # Retain the entire active frames, including at zero requested padding.
    # Frame centers would otherwise cut into the start and end of speech.
    first_sample = int(active_frames[0].item()) * hop_samples
    last_sample = int(active_frames[-1].item()) * hop_samples + frame_samples
    # unfold omits a partial final hop; retain it if it contains signal.
    covered = (len(frame_rms) - 1) * hop_samples + frame_samples
    if covered < mono_power.numel() and mono_power[covered:].mean().sqrt() >= threshold:
        last_sample = mono_power.numel()
    lead_samples = padding_samples if lead_padding_ms is None else round(DUALCODEC_SAMPLE_RATE * max(0, lead_padding_ms) / 1000)
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
        audio.cpu().squeeze(0).numpy().T,
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


class TTSRequest(_InstancesRequest):
    text: str | list[str]

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
    use_verbalization: bool = False
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
    format: Literal["wav", "mp3", "opus"] = "wav"
    bitrate: Literal["32k", "64k", "96k", "128k"] = Field(
        default="64k",
        description="Audio bitrate for encoded MP3 or Opus output, including streaming.",
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


def _require_verbalizer(req: TTSRequest, engine) -> None:
    if not req.use_verbalization:
        return
    if req.language != "english":
        raise TTSRequestOptionError(
            "use_verbalization=true is supported only for English requests"
        )
    if engine is None or engine.verbalizer is None:
        raise TTSRequestOptionError(
            "use_verbalization=true requires a loaded verbalizer; start the "
            "server with ENABLE_VERBALIZATION=1"
        )


def _require_timestamps(req: TTSRequest, engine) -> None:
    if not req.include_timestamps:
        return
    if req.language != "english":
        raise TTSRequestOptionError(
            "include_timestamps is only available for English", code="timestamps_unsupported_language",
        )
    if engine is None or getattr(engine, "cb0_ctc", None) is None:
        raise TTSRequestOptionError("timestamps are not available on this server", code="timestamps_unavailable")


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


async def _synthesize(req: TTSRequest) -> dict:
    engine = await _get_engine()
    _require_verbalizer(req, engine)
    _require_vibevoice(req, engine)
    _require_timestamps(req, engine)
    raw = _prepare_raw_text(req.text)
    segment_controls = req.segment_controls
    if req.use_verbalization:
        raw, segment_controls = _split_with_controls(
            raw, segment_controls, digit_weight=4,
            chunk_scales=_chunk_scales(req, engine, raw, segment_controls),
        )
        context = _continuation_context(req, engine)   # only the first segment continues the prefix
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


def create_app():
    import asyncio as _asyncio

    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse, StreamingResponse
    from fastapi import status
    from app.config import CORS_ALLOWED_ORIGINS, REQUIRE_API_KEY, TTS_API_KEY
    from app.opus_stream import OpusStreamEncoder as _OpusStreamEncoder
    from app.stream_writers import Mp3StreamWriter

    _enable_streaming = os.getenv("ENABLE_STREAMING", "1") == "1"
    _engine_ready = _asyncio.Event()
    _server_config = ServerConfig()
    _request_slots = _asyncio.BoundedSemaphore(_server_config.max_inflight_requests)
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
        """Style sliders of the loaded model, for the `controls` / `segment_controls` request fields."""
        if not _engine_ready.is_set():
            return JSONResponse(content={"error": "Model still loading"}, status_code=503)
        engine = await _get_engine()
        catalog = engine.controls.catalog() if engine.controls is not None else []
        return {"controls": catalog}

    @app.get("/readyz")
    async def readyz():
        if _engine_ready.is_set():
            return {"status": "ready"}
        return JSONResponse(content={"status": "loading"}, status_code=503)

    @app.middleware("http")
    async def tts_api_key_guard(request: Request, call_next):
        if REQUIRE_API_KEY and request.method != "OPTIONS":
            open_paths = {"/", "/api", "/healthz", "/readyz"}
            if request.url.path not in open_paths:
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
        _require_verbalizer(req, engine)
        if req.use_verbalization:
            # generate_streaming preprocesses segments[0] first; only it continues the prefix.
            pending_context = [_continuation_context(req, engine)]

            async def preprocess(text: str) -> str:
                context, pending_context[0] = pending_context[0], None
                text = await _verbalize_segment(engine, text, context)
                return sanitize_text_for_tts(text)
            return preprocess
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
        async def _audio_sse_generator(engine, text, writer, label="STREAM", priority_high: bool = False, **gen_kwargs):
            """Shared SSE audio streaming generator.

            Yields SSE `data:` lines with base64-encoded audio chunks. When
            `priority_high` is true, generation runs inside the priority gate's
            HIGH scope so low-priority requests park between vLLM calls.
            """
            gate_ctx = get_gate().high() if priority_high else None
            async with _request_slots:
                if gate_ctx is not None:
                    await gate_ctx.__aenter__()
                try:
                    chunk_idx = 0
                    t_stream_start = time.monotonic()
                    try:
                        async for audio_chunk in engine.generate_streaming(text=text, **gen_kwargs):
                            t0 = time.monotonic()
                            encoded_bytes = writer.write_chunk(audio_chunk)
                            t1 = time.monotonic()
                            audio_b64 = base64.b64encode(encoded_bytes).decode('ascii')
                            event = json.dumps({"index": chunk_idx, "audio_b64": audio_b64, "done": False})
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
                            yield f"data: {json.dumps({'index': chunk_idx, 'audio_b64': audio_b64, 'done': False})}\n\n"
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
            """Build an MP3 SSE response, honoring bitrate and request priority."""
            option_error = _streaming_option_error(req, "mp3")
            if option_error:
                return JSONResponse(content=_error_body(option_error), status_code=422)
            engine = await _get_engine()
            if engine.vv_tokenizer is None:
                return JSONResponse(
                    content={"error": "streaming is unavailable because VibeVoice is not loaded"},
                    status_code=503,
                )
            text, preprocessor, segment_controls = _prepare_streaming_request(req, engine)
            initial_seconds = _initial_seconds(req, engine, text)
            writer = Mp3StreamWriter(
                DUALCODEC_SAMPLE_RATE,
                bitrate=_bitrate_number(req.bitrate, bits_per_second=False),
            )

            return StreamingResponse(
                _audio_sse_generator(
                    engine, text, writer,
                    priority_high=req.priority == "high",
                    **_streaming_kwargs(req, preprocessor, segment_controls, initial_seconds),
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        @app.post(
            "/stream",
            description=(
                "Stream MP3 chunks as server-sent events. Streaming requires "
                "VibeVoice; bitrate and priority are honored."
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
                option_error = _streaming_option_error(req, "opus")
                if option_error:
                    await ws.send_json(_error_body(option_error))
                    await ws.close(code=1008)
                    return
                engine = await _get_engine()
                if engine.vv_tokenizer is None:
                    await ws.send_json({
                        "error": "streaming is unavailable because VibeVoice is not loaded"
                    })
                    await ws.close(code=1013)
                    return
                text, preprocessor, segment_controls = _prepare_streaming_request(req, engine)
                initial_seconds = _initial_seconds(req, engine, text)
                opus_enc = _OpusStreamEncoder(
                    DUALCODEC_SAMPLE_RATE,
                    bitrate=_bitrate_number(req.bitrate, bits_per_second=True),
                )
                async with _request_slots:
                    gate_ctx = get_gate().high() if req.priority == "high" else None
                    if gate_ctx is not None:
                        await gate_ctx.__aenter__()
                    try:
                        async for audio_chunk in engine.generate_streaming(
                            text=text,
                            **_streaming_kwargs(req, preprocessor, segment_controls, initial_seconds),
                        ):
                            await ws.send_bytes(opus_enc.encode(audio_chunk))
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
