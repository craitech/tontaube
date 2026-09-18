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
    DEFAULT_LANGUAGE,
    DEFAULT_TAG,
    DEFAULT_STREAMING_INITIAL_SECONDS,
    DUALCODEC_HZ,
    DUALCODEC_SAMPLE_RATE,
    MAX_STREAMING_INITIAL_SECONDS,
    MAX_PREFIX_TOKENS_PER_ROW,
    MAX_REQUEST_TEXT_CHARS,
    MAX_REQUEST_TEXT_SEGMENTS,
    MAX_SEMANTIC_TOKENS_PER_CHUNK,
    MAX_TEXT_CHARS_PER_CHUNK,
    MAX_VOICE_PROMPT_TOKENS,
    MAX_VOICE_REFERENCE_B64_CHARS,
    SEMANTIC_VOCAB_SIZE,
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
from app.text import sanitize_text_for_tts, split_long_text
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
):
    """Trim excess edge silence while preserving all internal pauses."""
    if padding_ms < 0 or audio.shape[-1] == 0:
        return audio

    samples = audio.detach().float().reshape(-1, audio.shape[-1])
    mono_power = samples.square().mean(dim=0)
    frame_samples = max(1, round(DUALCODEC_SAMPLE_RATE * 0.02))
    hop_samples = max(1, round(DUALCODEC_SAMPLE_RATE * 0.01))
    if mono_power.numel() < frame_samples:
        return audio

    frame_rms = mono_power.unfold(0, frame_samples, hop_samples).mean(dim=-1).sqrt()
    threshold = 10.0 ** (silence_threshold_dbfs / 20.0)
    active_frames = (frame_rms >= threshold).nonzero(as_tuple=False).flatten()
    if active_frames.numel() == 0:
        return audio

    padding_samples = round(DUALCODEC_SAMPLE_RATE * padding_ms / 1000)
    # Retain the entire active frames, including at zero requested padding.
    # Frame centers would otherwise cut into the start and end of speech.
    first_sample = int(active_frames[0].item()) * hop_samples
    last_sample = int(active_frames[-1].item()) * hop_samples + frame_samples
    # unfold omits a partial final hop; retain it if it contains signal.
    covered = (len(frame_rms) - 1) * hop_samples + frame_samples
    if covered < mono_power.numel() and mono_power[covered:].mean().sqrt() >= threshold:
        last_sample = mono_power.numel()
    start = max(0, first_sample - padding_samples)
    end = min(audio.shape[-1], last_sample + padding_samples)
    return audio[..., start:end]


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


def _apply_mossformer2_postprocessing(audio, input_sample_rate: int):
    """Load and run the optional enhancer outside the async event loop."""
    from app.mossformer2_postprocessor import (
        MOSSFORMER2_SAMPLE_RATE,
        get_mossformer2_postprocessor,
    )

    enhanced = get_mossformer2_postprocessor().process(audio, input_sample_rate)
    return enhanced, MOSSFORMER2_SAMPLE_RATE


class TTSRequest(BaseModel):
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
    temperature: float = Field(
        default=0.8,
        ge=0,
        le=2,
        description=(
            "CB0 sampling temperature. The 0.8 default favors expression; "
            "0.6 is a more conservative choice for correctness."
        ),
    )
    top_p: float | None = Field(default=None, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=-1)
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2)
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
            "to first audio but provide more playback headroom."
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
        default=True,
        description=(
            "Cap quiet pauses at 2.5 seconds, then apply hybrid MossFormer2_SR_48K "
            "super-resolution after optional boundary trimming. Enabled by default "
            "for non-streaming requests; "
            "set false to return the original 24 kHz output."
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

    @field_validator('voice_tokens')
    @classmethod
    def validate_voice_tokens(cls, value):
        rows = cls._validate_token_rows(value, MAX_VOICE_PROMPT_TOKENS)
        if rows is not None and (not rows or not rows[0]):
            raise ValueError('voice-token rows must not be empty')
        if rows and len({len(row) for row in rows}) > 1:
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


class TTSErrorResponse(BaseModel):
    error: str


class TTSRequestOptionError(ValueError):
    """A valid request whose selected feature is unavailable."""


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


async def synthesize(req: TTSRequest) -> dict:
    if req.priority == "high":
        async with get_gate().high():
            return await _synthesize(req)
    return await _synthesize(req)


async def _synthesize(req: TTSRequest) -> dict:
    engine = await _get_engine()
    _require_verbalizer(req, engine)
    _require_vibevoice(req, engine)
    raw = _prepare_raw_text(req.text)
    if req.use_verbalization:
        raw = split_long_text(raw, digit_weight=4)
        if isinstance(raw, list):
            raw = [await engine.verbalizer.verbalize(segment) for segment in raw]
        else:
            raw = await engine.verbalizer.verbalize(raw)
    text = (
        [sanitize_text_for_tts(segment) for segment in raw]
        if isinstance(raw, list)
        else sanitize_text_for_tts(raw)
    )
    text = split_long_text(text)

    audio, codebook_tokens = await engine.generate(
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
        vllm_priority=req.vllm_priority,
    )

    output_sample_rate = DUALCODEC_SAMPLE_RATE
    if req.trim_silence_padding_ms is not None:
        audio = _trim_boundary_silence(
            audio,
            padding_ms=req.trim_silence_padding_ms,
        )
    if req.mossformer2_postprocess:
        audio, output_sample_rate = await asyncio.to_thread(
            _apply_mossformer2_postprocessing,
            audio,
            output_sample_rate,
        )
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

    @asynccontextmanager
    async def lifespan(_app):
        if REQUIRE_API_KEY and not TTS_API_KEY:
            raise RuntimeError("TTS_API_KEY is required when REQUIRE_API_KEY=1")
        await _get_engine()
        logger.info(
            "End-to-end request concurrency limited to %d",
            _server_config.max_inflight_requests,
        )
        _engine_ready.set()
        try:
            yield
        finally:
            _unload_engine()

    app = FastAPI(title="Tontaube TTS API", version=API_VERSION, lifespan=lifespan)

    @app.exception_handler(TTSRequestOptionError)
    async def request_option_error(_request: Request, exc: TTSRequestOptionError):
        return JSONResponse(
            content=TTSErrorResponse(error=str(exc)).model_dump(),
            status_code=422,
        )

    @app.exception_handler(VoicePromptError)
    async def voice_prompt_error(_request: Request, exc: VoicePromptError):
        return JSONResponse(
            content=TTSErrorResponse(error=str(exc)).model_dump(),
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
                        content=TTSErrorResponse(error="unauthorized").model_dump(),
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
            async def preprocess(text: str) -> str:
                text = await engine.verbalizer.verbalize(text)
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
        return split_long_text(
            text,
            digit_weight=4 if preprocessor is not None else 1,
        ), preprocessor

    def _streaming_kwargs(req: TTSRequest, preprocessor) -> dict:
        return {
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
            "streaming_initial_seconds": req.streaming_initial_seconds,
            "acoustic_temperature": req.acoustic_temperature,
            "acoustic_top_k": req.acoustic_top_k,
            "prompt_max_tokens": req.prompt_max_tokens,
            "text_preprocessor": preprocessor,
            "language": req.language,
            "tag": req.tag,
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
                return JSONResponse(content={"error": option_error}, status_code=422)
            engine = await _get_engine()
            if engine.vv_tokenizer is None:
                return JSONResponse(
                    content={"error": "streaming is unavailable because VibeVoice is not loaded"},
                    status_code=503,
                )
            text, preprocessor = _prepare_streaming_request(req, engine)
            writer = Mp3StreamWriter(
                DUALCODEC_SAMPLE_RATE,
                bitrate=_bitrate_number(req.bitrate, bits_per_second=False),
            )

            return StreamingResponse(
                _audio_sse_generator(
                    engine, text, writer,
                    priority_high=req.priority == "high",
                    **_streaming_kwargs(req, preprocessor),
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
                    await ws.send_json({"error": option_error})
                    await ws.close(code=1008)
                    return
                engine = await _get_engine()
                if engine.vv_tokenizer is None:
                    await ws.send_json({
                        "error": "streaming is unavailable because VibeVoice is not loaded"
                    })
                    await ws.close(code=1013)
                    return
                text, preprocessor = _prepare_streaming_request(req, engine)
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
                            **_streaming_kwargs(req, preprocessor),
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
