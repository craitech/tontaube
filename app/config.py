import os
from dataclasses import dataclass, field
from pathlib import Path

from app.logging_utils import get_logger
from app.voice_catalog import (
    LANGUAGE_ALIASES,
    SUPPORTED_VOICE_STYLES,
    VOICE_AUDIO_EXTENSIONS,
    VOICE_FILES_DIRECTORY,
)

logger = get_logger("config")

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

# The package exposes the Tontaube architecture through vLLM's standard plugin
# entry point. Restrict discovery to that plugin by default, while preserving
# any additional plugins explicitly requested by an advanced deployment.
_configured_vllm_plugins = [
    name.strip()
    for name in os.environ.get("VLLM_PLUGINS", "").split(",")
    if name.strip()
]
if "tontaube" not in _configured_vllm_plugins:
    _configured_vllm_plugins.append("tontaube")
os.environ["VLLM_PLUGINS"] = ",".join(_configured_vllm_plugins)

DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "english")
DEFAULT_TAG = os.getenv("DEFAULT_TAG", "audiobook")

# The API accepts ISO aliases but always renders the canonical language label
# used by the TontaubeV1 input scheme.
SUPPORTED_TAGS: tuple[str, ...] = SUPPORTED_VOICE_STYLES
BUNDLED_VOICE_PATH = REPOSITORY_ROOT / "voices"
DEFAULT_BUNDLED_VOICE = "Miles.wav"


def normalize_language(language: str) -> str:
    canonical = LANGUAGE_ALIASES.get(language.strip().lower())
    if canonical is None:
        supported = ", ".join(sorted(set(LANGUAGE_ALIASES.values())))
        raise ValueError(f"unsupported language {language!r}; expected one of: {supported}")
    return canonical


def build_system_prompt(language: str, tag: str) -> str:
    language = normalize_language(language)
    tag = tag.strip().lower()
    if tag not in SUPPORTED_TAGS:
        raise ValueError(f"unsupported tag {tag!r}; expected one of: {', '.join(SUPPORTED_TAGS)}")
    return f"{language} : {tag}"


# Validate deployment defaults at import time rather than failing on the first
# request after a multi-minute model load.
DEFAULT_LANGUAGE = normalize_language(DEFAULT_LANGUAGE)
DEFAULT_TAG = DEFAULT_TAG.strip().lower()
build_system_prompt(DEFAULT_LANGUAGE, DEFAULT_TAG)


DUALCODEC_MODEL_TYPE = "12hz_v1"
DUALCODEC_SAMPLE_RATE = 24000
DUALCODEC_HZ = 12.5
DUALCODEC_SEMANTIC_NORMALIZATION_FILE = "w2vbert2_mean_var_stats_emilia.pt"

# Streaming durations advance in 0.4-second units, which align exactly to both
# five DualCodec tokens at 12.5 Hz and three VibeVoice frames at 7.5 Hz.
STREAMING_SECONDS_QUANTUM = 0.4
MIN_STREAMING_INITIAL_SECONDS = STREAMING_SECONDS_QUANTUM
MAX_STREAMING_INITIAL_SECONDS = 6.0
DEFAULT_STREAMING_INITIAL_SECONDS = 2.8
STREAMING_DC_PAD_TOKENS = 5

# API-key authentication is opt-in for self-hosted deployments.
REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "0") == "1"
TTS_API_KEY = os.getenv("TTS_API_KEY")

# The standalone local UI is useful out of the box. Deployments serving it from
# another origin can replace this narrow loopback allowlist.
_cors_origins = os.getenv("TTS_CORS_ORIGINS")
if _cors_origins is None:
    _cors_origins = (
        "http://127.0.0.1:3000,http://localhost:3000,"
        "http://0.0.0.0:3000,http://[::1]:3000"
    )
CORS_ALLOWED_ORIGINS: tuple[str, ...] = tuple(
    dict.fromkeys(
        origin.strip()
        for origin in _cors_origins.split(",")
        if origin.strip()
    )
)

SEMANTIC_VOCAB_SIZE = 16384
ACOUSTIC_VOCAB_SIZE = 4096
# Public request envelope: DualCodec runs at 12.5 Hz, so 400 semantic tokens
# correspond to 32 seconds. The 350-character split threshold keeps additional
# headroom below the context limits while preserving most complete sentences.
MAX_SEMANTIC_TOKENS_PER_CHUNK = 400
MAX_TEXT_CHARS_PER_CHUNK = 350
MAX_REQUEST_TEXT_CHARS = 40_000
MAX_REQUEST_TEXT_SEGMENTS = 1_000
MAX_VOICE_REFERENCE_B64_CHARS = 8 * 1024 * 1024
MAX_VOICE_PROMPT_TOKENS = 750
MAX_PREFIX_TOKENS_PER_ROW = MAX_SEMANTIC_TOKENS_PER_CHUNK
MAX_VV_DECODE_QUEUE_SIZE = 64
DC_SILENCE_TOKEN = 15478  # dominant cb0 token when encoding silence
SILENCE_TOKEN_ID = 3716   # cb0 token forced as first predicted token (silence onset)
# Original long-silence tokens plus repeatable collapse attractors.
COLLAPSE_SILENCE_TOKEN_IDS: tuple[int, ...] = (
    3716, 12725, 3974, 4299,
    10309, 5995, 6144, 15170, 13601, 398,
)
# High-precision tokens from ordinary internal pauses over 200 ms.
ORDINARY_SILENCE_TOKEN_IDS: tuple[int, ...] = (
    7362, 17, 3710, 1531, 1384, 12293, 5169, 4544,
)
# Complete set used for early-silence ratio checks.
SILENCE_TOKEN_IDS: tuple[int, ...] = (
    COLLAPSE_SILENCE_TOKEN_IDS + ORDINARY_SILENCE_TOKEN_IDS
)

END_OF_SPEECH = '<|end_of_speech|>'
TEXT_MARKER = '<|text_split|>'
AUDIO_MARKER = '<|audio_split|>'

def _get_gpu_memory_gib() -> float:
    """Detect and cache total GPU memory in GiB."""
    if not hasattr(_get_gpu_memory_gib, '_cached'):
        try:
            import torch
            props = torch.cuda.get_device_properties(0)
            _get_gpu_memory_gib._cached = props.total_memory / (1024 ** 3)
            logger.info(
                "Detected %s with %.3f GiB VRAM",
                props.name,
                _get_gpu_memory_gib._cached,
            )
        except Exception as e:
            _get_gpu_memory_gib._cached = float(os.getenv("VRAM", "24"))
            logger.warning(
                "VRAM detection failed (%s); using %.3f GiB fallback",
                e,
                _get_gpu_memory_gib._cached,
            )
    return _get_gpu_memory_gib._cached


CODEBOOK_MODEL_DIRS: tuple[str, ...] = ("cb0", "cb1", "cb2", "cb3")
N_CODEBOOKS = len(CODEBOOK_MODEL_DIRS)


@dataclass(frozen=True)
class CapacityProfile:
    """vLLM capacity targets ordered as cb0, cb1, cb2, cb3."""

    gpu_memory_gib_per_cb: tuple[float, ...]
    max_num_seqs_per_cb: tuple[int, ...]
    verbalization_gpu_memory_gib: float
    verbalization_max_num_seqs: int


# The GiB budgets leave full-context KV headroom during a cold compilation.
# Keep the sequence and batched-token limits linked: splitting a configured
# prefill can invalidate logical-position assignment during inference.
CAPACITY_PROFILES: dict[str, CapacityProfile] = {
    "low-vram": CapacityProfile(
        gpu_memory_gib_per_cb=(4.500, 1.860, 1.420, 1.220),
        max_num_seqs_per_cb=(1, 4, 4, 4),
        verbalization_gpu_memory_gib=4.8,
        verbalization_max_num_seqs=4,
    ),
    "balanced": CapacityProfile(
        gpu_memory_gib_per_cb=(4.900, 2.470, 1.920, 1.590),
        max_num_seqs_per_cb=(3, 8, 8, 8),
        verbalization_gpu_memory_gib=5.0,
        verbalization_max_num_seqs=10,
    ),
    "high-throughput": CapacityProfile(
        gpu_memory_gib_per_cb=(6.954, 5.204, 3.728, 3.250),
        max_num_seqs_per_cb=(8, 20, 20, 20),
        verbalization_gpu_memory_gib=5.5,
        verbalization_max_num_seqs=16,
    ),
}

CAPACITY_PROFILE_NAME = os.getenv("TTS_CAPACITY_PROFILE", "balanced").strip().lower()
if CAPACITY_PROFILE_NAME not in CAPACITY_PROFILES:
    raise ValueError(
        f"unsupported TTS_CAPACITY_PROFILE {CAPACITY_PROFILE_NAME!r}; expected one of: "
        f"{', '.join(CAPACITY_PROFILES)}"
    )
CAPACITY_PROFILE = CAPACITY_PROFILES[CAPACITY_PROFILE_NAME]


def _env_tuple(name: str, default: tuple, cast):
    raw = os.getenv(name)
    values = default if not raw else tuple(cast(v.strip()) for v in raw.split(","))
    if len(values) < N_CODEBOOKS:
        raise ValueError(f"{name} needs at least {N_CODEBOOKS} comma-separated values")
    return tuple(values[:N_CODEBOOKS])


_DEFAULT_MAX_MODEL_LEN = (2500, 2000, 2400, 3000)
MAX_MODEL_LEN_PER_CB = _env_tuple("MAX_MODEL_LEN_PER_CB", _DEFAULT_MAX_MODEL_LEN, int)


def _default_gpu_memory_gib_per_cb() -> tuple[float, ...]:
    return _env_tuple(
        "GPU_MEMORY_GIB_PER_CB",
        CAPACITY_PROFILE.gpu_memory_gib_per_cb,
        float,
    )


def _default_max_num_seqs_per_cb() -> tuple[int, ...]:
    return _env_tuple(
        "MAX_NUM_SEQS_PER_CB",
        CAPACITY_PROFILE.max_num_seqs_per_cb,
        int,
    )


def _configured_max_num_batched_tokens_per_cb() -> tuple[int, ...] | None:
    raw = os.getenv("MAX_NUM_BATCHED_TOKENS_PER_CB")
    if not raw:
        return None
    return _env_tuple("MAX_NUM_BATCHED_TOKENS_PER_CB", (), int)


def _configured_max_inflight_requests() -> int | None:
    raw = os.getenv("MAX_INFLIGHT_REQUESTS")
    return int(raw) if raw else None


@dataclass
class ServerConfig:
    codebook_model_dirs: tuple[str, ...] = CODEBOOK_MODEL_DIRS
    capacity_profile_name: str = CAPACITY_PROFILE_NAME

    max_new_tokens: int = MAX_SEMANTIC_TOKENS_PER_CHUNK
    # CB0 samples autoregressively; acoustic codebooks default to greedy decoding.
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50

    acoustic_temperature: float = 0.0
    acoustic_top_p: float = 0.95

    chunk_context_seconds: float = 6.0

    cb0_max_retries: int = 5
    cb0_min_tokens: int = 13
    cb0_min_cps: float = 6.0
    cb0_max_cps: float = 24.0
    cb0_max_silence_ratio: float = 0.40
    # Mild penalties reduce premature EOS and silence-collapse onsets.
    cb0_eos_logit_bias: float = -0.8
    cb0_silence_logit_bias: float = -0.2
    cb0_silence_logit_bias_step: float = -0.3  # added per inner cb0 retry attempt
    cb0_ordinary_silence_logit_bias: float = -0.2
    vv_retry_temperature_step: float = 0.1     # added to cb0 temp per outer VV retry

    # GPU memory per codebook engine in GiB (converted to fractions of total VRAM at runtime)
    gpu_memory_gib_per_cb: tuple[float, ...] = field(
        default_factory=_default_gpu_memory_gib_per_cb
    )
    max_model_len_per_cb: tuple[int, ...] = MAX_MODEL_LEN_PER_CB
    max_num_seqs_per_cb: tuple[int, ...] = field(
        default_factory=_default_max_num_seqs_per_cb
    )
    # Limit complete requests admitted to the multi-engine pipeline. When not
    # overridden, allow twice the CB0 active-sequence limit so later stages can
    # overlap without saturating every engine and VibeVoice simultaneously.
    max_inflight_requests: int | None = field(
        default_factory=_configured_max_inflight_requests
    )
    # None preserves the original max_model_len * max_num_seqs behavior.
    # Set this independently to tune prefill throughput versus latency.
    max_num_batched_tokens_per_cb: tuple[int, ...] | None = field(
        default_factory=_configured_max_num_batched_tokens_per_cb
    )

    prompt_seconds: float = 60.0
    prompt_max_tokens: int = MAX_VOICE_PROMPT_TOKENS
    prompt_max_tokens_per_cb: tuple[int, ...] = (750, 300, 150, 100)

    # VibeVoice is enabled by default for output re-encoding and streaming, but
    # can be omitted for a smaller DualCodec-only runtime.
    enable_vibevoice: bool = os.getenv("ENABLE_VIBEVOICE", "1") == "1"
    streaming_dc_pad_tokens: int = STREAMING_DC_PAD_TOKENS
    streaming_chunk_seconds: float = 2.0  # must be multiple of 0.4s (aligns DC@12.5Hz and VV@7.5Hz)
    max_vv_decode_queue_size: int = MAX_VV_DECODE_QUEUE_SIZE

    trailing_silence_tokens: int = 3  # silence tokens appended after final segment to avoid cut-off
    force_first_silence: bool = True  # force first predicted cb0 token to be SILENCE_TOKEN_ID
    # When True, low-priority requests (e.g. /predict without priority="high")
    # pause between vLLM generation calls while any high-priority request is
    # active. Disable to fall back to pre-gate FCFS behavior.
    priority_gate_enabled: bool = True

    # Loaded once and applied only to English requests whose use_verbalization
    # flag is true.
    enable_verbalization: bool = os.getenv("ENABLE_VERBALIZATION", "0") == "1"
    verbalization_model_path: str | None = None
    verbalization_gpu_memory_gib: float = float(
        os.getenv(
            "VERBALIZATION_GPU_MEMORY_GIB",
            str(CAPACITY_PROFILE.verbalization_gpu_memory_gib),
        )
    )
    verbalization_max_num_seqs: int = int(
        os.getenv(
            "VERBALIZATION_MAX_NUM_SEQS",
            str(CAPACITY_PROFILE.verbalization_max_num_seqs),
        )
    )

    def __post_init__(self) -> None:
        n = len(self.codebook_model_dirs)
        if n != N_CODEBOOKS:
            raise ValueError(f"TontaubeV1 requires {N_CODEBOOKS} codebook models, got {n}")
        if self.max_inflight_requests is None:
            self.max_inflight_requests = 2 * self.max_num_seqs_per_cb[0]
        if self.max_inflight_requests <= 0:
            raise ValueError("max_inflight_requests must be positive")
        # The logical-position adapter expects every configured active sequence
        # to fit into one prefill scheduler step. Keep this budget equal to
        # max_model_len * max_num_seqs: lowering it can split prefills and cause
        # incorrect position assignment during inference.
        if self.max_num_batched_tokens_per_cb is None:
            self.max_num_batched_tokens_per_cb = tuple(
                model_len * max_num_seqs
                for model_len, max_num_seqs in zip(
                    self.max_model_len_per_cb,
                    self.max_num_seqs_per_cb,
                )
            )
        for name, values in (
            ("gpu_memory_gib_per_cb", self.gpu_memory_gib_per_cb),
            ("max_model_len_per_cb", self.max_model_len_per_cb),
            ("max_num_seqs_per_cb", self.max_num_seqs_per_cb),
            ("max_num_batched_tokens_per_cb", self.max_num_batched_tokens_per_cb),
            ("prompt_max_tokens_per_cb", self.prompt_max_tokens_per_cb),
        ):
            if len(values) != n:
                raise ValueError(f"{name} must contain {n} values")
            if any(value <= 0 for value in values):
                raise ValueError(f"{name} values must all be positive")
        for cb, (batched_tokens, max_num_seqs, model_len) in enumerate(zip(
            self.max_num_batched_tokens_per_cb,
            self.max_num_seqs_per_cb,
            self.max_model_len_per_cb,
        )):
            expected_batched_tokens = max_num_seqs * model_len
            if batched_tokens != expected_batched_tokens:
                raise ValueError(
                    f"max_num_batched_tokens_per_cb[{cb}] must equal "
                    f"max_num_seqs_per_cb[{cb}] * max_model_len_per_cb[{cb}] "
                    f"({expected_batched_tokens}) so configured prefills are not split"
                )
        if self.verbalization_gpu_memory_gib <= 0:
            raise ValueError("verbalization_gpu_memory_gib must be positive")
        if self.verbalization_max_num_seqs <= 0:
            raise ValueError("verbalization_max_num_seqs must be positive")

    @property
    def generate_n_codebooks(self) -> int:
        return len(self.codebook_model_dirs)

    @property
    def voice_path(self) -> str:
        return os.environ.get('VOICE_PATH', str(BUNDLED_VOICE_PATH))

    @property
    def default_voice(self) -> str:
        configured = os.environ.get("DEFAULT_VOICE")
        if configured:
            return configured if os.path.isabs(configured) else os.path.join(self.voice_path, configured)
        return str(BUNDLED_VOICE_PATH / VOICE_FILES_DIRECTORY / DEFAULT_BUNDLED_VOICE)

    @property
    def gpu_memory_per_cb(self) -> tuple[float, ...]:
        """Convert GiB values to fractions of total GPU VRAM."""
        total_gib = _get_gpu_memory_gib()
        return tuple(gib / total_gib for gib in self.gpu_memory_gib_per_cb)

    @property
    def verbalization_gpu_util(self) -> float:
        """Convert verbalization GiB to fraction of total GPU VRAM."""
        return self.verbalization_gpu_memory_gib / _get_gpu_memory_gib()
