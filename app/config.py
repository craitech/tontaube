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

# "lite" (default): all codebook models in this process with quantized weights and a
#         paged KV pool sized in tokens (small GPUs); see app/lite.
# "vllm": one vLLM engine process per codebook (high-throughput serving).
TTS_BACKEND = os.getenv("TTS_BACKEND", "lite").strip().lower()
if TTS_BACKEND not in ("vllm", "lite"):
    raise ValueError(f"unsupported TTS_BACKEND {TTS_BACKEND!r}; expected 'vllm' or 'lite'")

# "v1.1" (default): TontaubeV1.1 on the backend above. "flash": TontaubeV1.1 Flash, one English voice, on the GPU or
# the CPU (LITE_DEVICE); see app/flash.py.
TTS_MODEL = os.getenv("TTS_MODEL", "v1.1").strip().lower()
if TTS_MODEL not in ("v1.1", "flash"):
    raise ValueError(f"unsupported TTS_MODEL {TTS_MODEL!r}; expected 'v1.1' or 'flash'")


def _lite_on_cpu() -> bool:
    if TTS_BACKEND != "lite" and TTS_MODEL != "flash":
        return False
    device = os.getenv("LITE_DEVICE", "").strip().lower()
    if device:
        return device == "cpu"
    try:
        import torch

        return not torch.cuda.is_available()
    except Exception:
        return True


LITE_ON_CPU = _lite_on_cpu()


def _backend_defaults(backend: str, lite_on_cpu: bool) -> dict[str, str]:
    """Defaults that depend on the backend and device; the variables still override them.

    The lite backend targets small GPUs and CPUs: the low-vram capacity profile and
    a bounded re-decode context for streaming; 12 s decode windows on the GPU
    (bounded VRAM), 30 s on the CPU (about 35% faster there); and MossFormer2 off
    by default on the CPU, where it costs about 3x real time.
    """
    lite = backend == "lite"
    return {
        "TTS_CAPACITY_PROFILE": "low-vram" if lite else "balanced",
        "STREAMING_DECODE_CONTEXT_SECONDS": "6" if lite else "",
        "DECODE_WINDOW_SECONDS": "12" if lite and not lite_on_cpu else "30",
        "MOSSFORMER2_POSTPROCESS_DEFAULT": "0" if lite_on_cpu else "1",
    }


BACKEND_DEFAULTS = _backend_defaults(TTS_BACKEND, LITE_ON_CPU)
MOSSFORMER2_POSTPROCESS_DEFAULT = os.getenv(
    "MOSSFORMER2_POSTPROCESS_DEFAULT", BACKEND_DEFAULTS["MOSSFORMER2_POSTPROCESS_DEFAULT"]
) == "1"

DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "english")
DEFAULT_TAG = os.getenv("DEFAULT_TAG", "audiobook")

# The API accepts ISO aliases but always renders the canonical language label
# used by the TontaubeV1 input scheme.
SUPPORTED_TAGS: tuple[str, ...] = SUPPORTED_VOICE_STYLES
BUNDLED_VOICE_PATH = REPOSITORY_ROOT / "voices"
DEFAULT_BUNDLED_VOICE = "Marcus.mp3"


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


# Release name of the model family; voice_tokens are tied to its DualCodec codebooks
# (the same for the 4-bit and full-precision weights).
MODEL_VERSION = "TontaubeV1.1"
# How a request's text is verbalized (request field `verbalizer`; the one place that names the modes): the rule
# verbalizer of Flash (numbers, dates, times, money, units, abbreviations; app/verbalize_rules.py), the model
# verbalizer (app/verbalizer.py, ENABLE_VERBALIZATION=1, English), or none.
VERBALIZER_RULES = "rules"
VERBALIZER_CONTEXTUAL = "contextual"
VERBALIZER_OFF = "off"
VERBALIZER_MODES = (VERBALIZER_RULES, VERBALIZER_CONTEXTUAL, VERBALIZER_OFF)

DUALCODEC_MODEL_TYPE = "12hz_v1"
# The codec that made a token sequence, recorded with stored voice tokens.
DUALCODEC_CODEC_ID = f"dualcodec_{DUALCODEC_MODEL_TYPE}"
# 12hz_v1 has 1 semantic + 7 acoustic RVQ codebooks (dualcodec conf/model/dualcodec_12hz_16384_4096_8vq.yaml);
# the Tontaube models use the first N_CODEBOOKS, /encode_voice can return all of them.
DUALCODEC_MAX_CODEBOOKS = 8
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
# Streaming start buffer (app/stream_calibration.py). Off by default: requests without streaming_initial_seconds use
# 2.8 s, and the first stream logs a warning that this can pause playback on slower machines
# (STREAM_CALIBRATION_WARNING=0 hides it). STREAM_CALIBRATION=1 measures the machine with one short stream at startup;
# a request without the field then gets the smallest start buffer that keeps its playback from pausing, at least
# STREAM_AUTO_MIN_INITIAL_SECONDS (default 2.8 s, so a fast machine keeps its behaviour; lower it to let streams start
# sooner where the machine allows), and an explicit value is logged as a warning next to the calibrated one.
# STREAM_CALIBRATION_MARGIN scales the measured times.
STREAM_CALIBRATION = os.getenv("STREAM_CALIBRATION", "0") == "1"
STREAM_CALIBRATION_WARNING = os.getenv("STREAM_CALIBRATION_WARNING", "1") == "1"
STREAM_AUTO_MIN_INITIAL_SECONDS = float(
    os.getenv("STREAM_AUTO_MIN_INITIAL_SECONDS") or DEFAULT_STREAMING_INITIAL_SECONDS
)
STREAM_CALIBRATION_MARGIN = float(os.getenv("STREAM_CALIBRATION_MARGIN") or 1.25)

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
# CB0 chunk-end tokens: only bundles that declare their rows in BOUNDARY_TOKENS_FILENAME carry them (ids right
# after the slider tokens).
UTT_END = '<|utt_end|>'
UTT_CONT = '<|utt_cont|>'
BOUNDARY_TOKENS = (UTT_END, UTT_CONT)
BOUNDARY_TOKENS_FILENAME = 'boundary_tokens.json'

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
    # One active sequence per codebook. With the lite backend this keeps the
    # KV pools at one full-context request each; long texts then run their
    # acoustic segments one after another instead of batched.
    "minimal": CapacityProfile(
        gpu_memory_gib_per_cb=(4.500, 1.860, 1.420, 1.220),
        max_num_seqs_per_cb=(1, 1, 1, 1),
        verbalization_gpu_memory_gib=4.8,
        verbalization_max_num_seqs=1,
    ),
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

CAPACITY_PROFILE_NAME = os.getenv(
    "TTS_CAPACITY_PROFILE", BACKEND_DEFAULTS["TTS_CAPACITY_PROFILE"]
).strip().lower()
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


def _configured_kv_cache_gib_per_cb() -> tuple[float, ...] | None:
    # Fixed KV cache size per engine (vLLM kv_cache_memory_bytes). Skips vLLM's memory profiling, whose global
    # free-memory measurement charges other processes' allocations on a shared GPU to the engine being started.
    raw = os.getenv("KV_CACHE_GIB_PER_CB")
    if not raw:
        return None
    return _env_tuple("KV_CACHE_GIB_PER_CB", (), float)


def _prefix_lead_trim_padding_ms() -> int | None:
    raw = (os.getenv("PREFIX_LEAD_TRIM_PADDING_MS") or "").strip().lower()
    if raw in ("off", "none"):
        return None
    return int(raw) if raw else 80


def _configured_max_inflight_requests() -> int | None:
    raw = os.getenv("MAX_INFLIGHT_REQUESTS")
    return int(raw) if raw else None


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None or not raw.strip() else raw.strip() == "1"


def _env_headroom_db(name: str, default: float) -> float:
    """A headroom in dB (>= 0, finite); unset or blank -> default, "0" -> 0.0 (no gain change)."""
    raw = os.getenv(name)
    value = default if raw is None or not raw.strip() else float(raw.strip())
    if not (0.0 <= value < float("inf")):
        raise ValueError(f"{name} must be a finite dB value >= 0, got {raw!r}")
    return value


# Join pieces under 50 chars to a neighbour after split_long_text (app/text.py merge_short_pieces).
# MERGE_SHORT_PIECES: pieces split from the same text. MERGE_SHORT_CLIENT_SEGMENTS: also across client-provided
# segments whose segment_controls entries are identical (clients may split on purpose). MERGE_SHORT_PIECES on by
# default, MERGE_SHORT_CLIENT_SEGMENTS opt-in.
MERGE_SHORT_PIECES = _env_flag("MERGE_SHORT_PIECES", True)
MERGE_SHORT_CLIENT_SEGMENTS = _env_flag("MERGE_SHORT_CLIENT_SEGMENTS", False)


def _env_level_window() -> tuple[float, float] | None:
    raw = os.getenv("VOICE_REF_LEVEL_WINDOW", "-26,-14").strip()
    if raw.lower() in ("", "0", "off", "none"):
        return None
    low, high = (float(v) for v in raw.split(","))
    if low > high:
        raise ValueError("VOICE_REF_LEVEL_WINDOW needs low,high (dB) with low <= high")
    return low, high


# Voice references whose gated speech level (dB) lies outside this window are scaled to the nearer edge before
# encoding (very quiet references make cloning fail). VOICE_REF_LEVEL_WINDOW=off disables it.
VOICE_REF_LEVEL_WINDOW = _env_level_window()
# Never end a chunk on a 1-2-word sentence ("No.", "Wow.", "I know."), which is clipped far more often than a longer
# final sentence. split_long_text moves it to the start of the next chunk (app/text.py). On by default;
# NO_SHORT_FINAL_SENTENCE=0 restores the previous splitter.
NO_SHORT_FINAL_SENTENCE = _env_flag("NO_SHORT_FINAL_SENTENCE", True)
# The sentence-first cut of split_long_text takes the last real sentence end in its window (_sentence_cuts: no cut
# after an abbreviation, an ordinal, an initial, inside a decimal or a domain; after a closing quote, not before it)
# instead of the last ". ! ?" character. On by default; SPLIT_SENTENCE_CUTS=0 restores the character search.
SPLIT_SENTENCE_CUTS = _env_flag("SPLIT_SENTENCE_CUTS", True)

# Expected chars/s factor per rate slider bin. Bins in between are interpolated log-linearly (app/cb0_guard.py).
DEFAULT_CB0_RATE_CPS_FACTORS: dict[int, float] = {0: 0.70, 4: 1.30}
DEFAULT_RATE_CPS_FACTORS_JSON = Path(__file__).resolve().parent / "cb0_rate_window_factors.json"


def _env_rate_cps_factors() -> dict[int, float]:
    """CB0_RATE_CPS_FACTORS="0:0.70,4:1.30" (bin:factor pairs) -> {0: 0.7, 4: 1.3}."""
    raw = os.getenv("CB0_RATE_CPS_FACTORS")
    if not raw:
        return dict(DEFAULT_CB0_RATE_CPS_FACTORS)
    factors = {}
    for item in raw.split(","):
        rate_bin, _, factor = item.partition(":")
        factors[int(rate_bin)] = float(factor)
    if any(f <= 0 for f in factors.values()):
        raise ValueError("CB0_RATE_CPS_FACTORS factors must all be positive")
    return factors
def _configured_lite_kv_tokens_per_cb() -> tuple[int, ...] | None:
    raw = os.getenv("LITE_KV_TOKENS_PER_CB")
    if not raw:
        return None
    return _env_tuple("LITE_KV_TOKENS_PER_CB", (), int)


@dataclass
class ServerConfig:
    codebook_model_dirs: tuple[str, ...] = CODEBOOK_MODEL_DIRS
    backend: str = TTS_BACKEND
    # Lite backend: weight quantization plan (one plan, or four separated by
    # ";" for cb0..cb3; see app/lite/plans.py), a bundle directory, or "auto" (the
    # model root's int4/ bundle if present, else bf16), and the paged KV pool size per
    # codebook in tokens (default: max_model_len * max_num_seqs, i.e. never
    # waits for KV blocks).
    lite_quant: str = os.getenv("LITE_QUANT", "auto")
    lite_kv_tokens_per_cb: tuple[int, ...] | None = field(
        default_factory=_configured_lite_kv_tokens_per_cb
    )
    # Lite backend codec stack: keep W2V-BERT (only used to encode voice
    # references) in host memory between encodes, and quantize VibeVoice's
    # FFN layers on the GPU, its wide FFNs and convolutions on the CPU
    # ("bf16" keeps them unquantized; the CPU then computes in fp32).
    lite_offload_w2v: bool = os.getenv("LITE_W2V_OFFLOAD", "1") == "1"
    # Streaming: seconds of already-emitted audio re-decoded as left context for
    # each new chunk (a multiple of 0.4 s). None re-decodes the whole stream on
    # every chunk, whose memory and time grow with the stream length.
    streaming_decode_context_seconds: float | None = (
        float(os.getenv("STREAMING_DECODE_CONTEXT_SECONDS") or BACKEND_DEFAULTS["STREAMING_DECODE_CONTEXT_SECONDS"])
        if os.getenv("STREAMING_DECODE_CONTEXT_SECONDS") or BACKEND_DEFAULTS["STREAMING_DECODE_CONTEXT_SECONDS"]
        else None
    )
    # Non-streaming decode window (DualCodec + VibeVoice), a multiple of 0.4 s.
    # Shorter windows bound activation memory (30 s: ~1.1 GiB, 12 s: ~0.45 GiB
    # peak) at the cost of re-decoding more context; with the fixed context
    # padding the output is unchanged (verified at 4-12 s against 30 s). The
    # lite backend uses 12 s on the GPU; on the CPU, 30 s decodes ~35% faster
    # for ~0.3 GiB more transient host memory.
    decode_window_seconds: float = float(
        os.getenv("DECODE_WINDOW_SECONDS", BACKEND_DEFAULTS["DECODE_WINDOW_SECONDS"])
    )
    lite_vibevoice_quant: str = os.getenv("LITE_VIBEVOICE_QUANT", "int8g128")
    # Lite backend verbalizer weights (a plan string, see app/lite/plans.py).
    lite_verbalizer_quant: str = os.getenv("LITE_VERBALIZER_QUANT", "int8g128")
    capacity_profile_name: str = CAPACITY_PROFILE_NAME

    max_new_tokens: int = MAX_SEMANTIC_TOKENS_PER_CHUNK
    # CB0 samples autoregressively; acoustic codebooks default to greedy decoding.
    temperature: float = 0.7
    frequency_penalty: float = 0.1
    top_p: float = 0.95
    top_k: int = 50

    acoustic_temperature: float = 0.0
    acoustic_top_p: float = 0.95

    chunk_context_seconds: float = 6.0

    cb0_max_retries: int = 5
    cb0_min_tokens: int = 13
    cb0_min_cps: float = 8.0
    cb0_max_cps: float = 20.0
    # Styles with a higher upper bound (conversational speech runs faster than audiobook reading).
    cb0_max_cps_by_style: dict[str, float] = field(
        default_factory=lambda: {"conversational": 24.0, "agentic": 24.0}
    )
    # Each digit counts as this many chars in the chars/s check (numbers are read as words).
    cb0_cps_digit_weight: int = 4
    cb0_retry_temperature_step: float = 0.05
    cb0_max_silence_ratio: float = 0.40
    # Whole-piece gap check. An attempt whose CB0 codes hold a run of silence codes (SILENCE_TOKEN_IDS,
    # + DC_SILENCE_TOKEN under CB0_SILENCE_COUNT_DIGITAL: the early-silence check's set) longer than this many
    # seconds anywhere in the piece fails serving's check like too_much_silence. 0 disables it.
    # Off unless CB0_GAP_CHECK=1 (the silence-code check misses real gaps).
    cb0_gap_check: bool = field(default_factory=lambda: _env_flag("CB0_GAP_CHECK", False))
    cb0_max_gap_seconds: float = 2.5
    # Mild penalties reduce premature EOS and silence-collapse onsets.
    cb0_eos_logit_bias: float = -0.8
    cb0_silence_logit_bias: float = -0.2
    cb0_silence_logit_bias_step: float = -0.3  # added per inner cb0 retry attempt
    cb0_ordinary_silence_logit_bias: float = -0.2
    vv_retry_temperature_step: float = 0.1     # added to cb0 temp per outer VV retry

    # CB0 serving guard (app/cb0_guard.py).
    # The rate-aware window, the CB0-CTC guard and the previous-audio hygiene below are on by default (v1.1 settings).
    # Rate-aware chars/s window: cb0_min_cps / cb0_max_cps are scaled by the factor of the rate slider
    # bin in the segment's CB0 header (no rate slider = x1). CB0_RATE_AWARE_CPS=0 to disable.
    cb0_rate_aware_cps: bool = field(default_factory=lambda: _env_flag("CB0_RATE_AWARE_CPS", True))
    cb0_rate_cps_factors: dict[int, float] = field(default_factory=_env_rate_cps_factors)
    # Optional per-model / per-style table (JSON: model -> style -> bin -> factor); missing styles / bins fall back
    # to its "all" entry, then to cb0_rate_cps_factors.
    cb0_rate_cps_factors_json: str | None = field(
        default_factory=lambda: os.getenv("CB0_RATE_CPS_FACTORS_JSON") or (
            str(DEFAULT_RATE_CPS_FACTORS_JSON) if DEFAULT_RATE_CPS_FACTORS_JSON.is_file() else None
        )
    )
    cb0_rate_cps_factors_model: str | None = field(
        default_factory=lambda: os.getenv("CB0_RATE_CPS_FACTORS_MODEL") or None
    )
    # Rate-aware chunk length: the text splitter's max chars (350) and the short-piece merge limit
    # (400) are scaled by the same factor per segment, capped at x1, so slow-rate pieces stay within
    # the CB0 token cap (max_new_tokens 400 = 32 s). CB0_RATE_AWARE_CHUNK_CHARS=1 to enable.
    cb0_rate_aware_chunk_chars: bool = field(
        default_factory=lambda: _env_flag("CB0_RATE_AWARE_CHUNK_CHARS", False)
    )
    # CB0-CTC guard: after an attempt passes the check above, retry when the CER of its CB0-CTC
    # transcript vs the segment text exceeds cb0_ctc_max_cer, at most cb0_ctc_max_retries times
    # within cb0_max_retries; then keep the serving-ok attempt with the lowest CER. CB0_CTC_GUARD=0 to disable.
    # Checkpoint: CB0_CTC_MODEL_PATH, else cb0_ctc.pt in the CB0 model dir; none found = guard off.
    cb0_ctc_guard: bool = field(default_factory=lambda: _env_flag("CB0_CTC_GUARD", True))
    cb0_ctc_model_path: str | None = field(default_factory=lambda: os.getenv("CB0_CTC_MODEL_PATH") or None)
    cb0_ctc_device: str | None = field(default_factory=lambda: os.getenv("CB0_CTC_DEVICE") or None)
    cb0_ctc_max_cer: float = 0.25
    cb0_ctc_max_retries: int = 2
    # Segments with digits: numbers are often read differently from the recogniser's spelled-out reference.
    cb0_ctc_digit_max_cer: float = 0.5
    # A kept segment above this CER counts as failed for the previous-audio chain (CB0_DROP_BAD_PREV_AUDIO).
    cb0_ctc_chain_max_cer: float = 0.5
    cb0_ctc_guard_languages: tuple[str, ...] = ("english",)  # the recogniser is English-only
    # Count DC_SILENCE_TOKEN (15478, digital silence) in the early-silence check (otherwise a segment of
    # mostly digital silence reads as barely silent). CB0_SILENCE_COUNT_DIGITAL=0 to disable.
    cb0_silence_count_digital: bool = field(
        default_factory=lambda: _env_flag("CB0_SILENCE_COUNT_DIGITAL", True)
    )
    # Previous-audio hygiene (needs the guard): a segment whose kept attempt failed the guard is not passed on as
    # previous audio (the next segment gets the last one that passed, or none), and a retry of a non-speech attempt
    # (CB0-CTC transcript under 3 chars) drops the previous audio. CB0_DROP_BAD_PREV_AUDIO=0 to disable.
    cb0_drop_bad_prev_audio: bool = field(
        default_factory=lambda: _env_flag("CB0_DROP_BAD_PREV_AUDIO", True)
    )
    # Boundary tokens: CB0 token windows. The current chunk ends in <|utt_end|> (instead of the trailing '\n') when
    # it is the request's last, else in <|utt_cont|> (before the lookahead's text marker); the previous text and the
    # lookahead carry neither a token nor '\n'. CB1-3 keep the legacy window. Needs a CB0 bundle that declares the
    # rows (boundary_tokens.json); refused otherwise. CB0_BOUNDARY_TOKENS=1 to enable. CB0_BOUNDARY_DROP_SPACE=1
    # (token windows only): no leading ' ' on any segment of the window.
    cb0_boundary_tokens: bool = field(default_factory=lambda: _env_flag("CB0_BOUNDARY_TOKENS", False))
    cb0_boundary_drop_space: bool = field(
        default_factory=lambda: _env_flag("CB0_BOUNDARY_DROP_SPACE", False)
    )
    # /predict quality gate (app/quality_gate.py). A serving mitigation, not a model fix. Per CB0 piece of /predict
    # (never /stream), after an attempt passes the check above, a CB0-only checker (one fp32 CB0-CTC forward: speech
    # span vs text length, CER, first / last word on the forced alignment, word coverage) decides: the first passing
    # attempt is kept; otherwise retry with a fresh seed, at most predict_gate_max_attempts scored attempts per piece
    # and predict_gate_budget_s added seconds per request, then keep the best attempt. With the CB0-CTC guard also on,
    # the gate makes the retry decision. Needs the CB0-CTC checkpoint (CB0_CTC_MODEL_PATH, as the guard); none found
    # = gate off. PREDICT_QUALITY_GATE=1 to enable.
    predict_quality_gate: bool = field(default_factory=lambda: _env_flag("PREDICT_QUALITY_GATE", False))
    predict_gate_max_attempts: int = field(
        default_factory=lambda: int(os.getenv("PREDICT_GATE_MAX_ATTEMPTS") or 4)
    )
    predict_gate_budget_s: float = field(
        default_factory=lambda: float(os.getenv("PREDICT_GATE_BUDGET_S") or 2.0)
    )
    # Operating point of the frozen thresholds: A strict, B medium (default), C loose.
    predict_gate_point: str = field(
        default_factory=lambda: (os.getenv("PREDICT_GATE_POINT") or "B").strip().upper()
    )
    # Optional JSONL sink: one line per piece and per request (text, voice prompt hash, seeds, CB0 codes, rule flags).
    predict_gate_log: str | None = field(default_factory=lambda: os.getenv("PREDICT_GATE_LOG") or None)

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
    # None = vLLM sizes the KV cache from gpu_memory_gib_per_cb (default behaviour).
    kv_cache_gib_per_cb: tuple[float, ...] | None = field(
        default_factory=_configured_kv_cache_gib_per_cb
    )

    prompt_seconds: float = 60.0
    prompt_max_tokens: int = MAX_VOICE_PROMPT_TOKENS
    prompt_max_tokens_per_cb: tuple[int, ...] = (750, 300, 150, 100)
    # Classifier-free guidance on the voice prompt per codebook (CB0 entry ignored), e.g.
    # CFG_WEIGHT_PER_CB=0,1,1,1. All zeros (default) = off, serving unchanged. See app/cfg_acoustic.py.
    cfg_weight_per_cb: tuple[float, ...] = field(
        default_factory=lambda: _env_tuple("CFG_WEIGHT_PER_CB", (0.0,) * N_CODEBOOKS, float)
    )

    # VibeVoice is enabled by default for output re-encoding and streaming, but
    # can be omitted for a smaller DualCodec-only runtime.
    enable_vibevoice: bool = os.getenv("ENABLE_VIBEVOICE", "1") == "1"
    streaming_dc_pad_tokens: int = STREAMING_DC_PAD_TOKENS
    streaming_chunk_seconds: float = 2.0  # must be multiple of 0.4s (aligns DC@12.5Hz and VV@7.5Hz)
    max_vv_decode_queue_size: int = MAX_VV_DECODE_QUEUE_SIZE

    trailing_silence_tokens: int = 0
    # Streaming lead trim. STREAM_LEAD_TRIM=1 cuts the first streamed chunk's lead silence to
    # STREAM_LEAD_TRIM_PADDING_MS before the first sound, with /predict's edge-trim frames and threshold, on the 20 ms
    # Opus grid (app/stream_lead_trim.py). Both streaming routes; later chunks and generation are unchanged. Default
    # on; STREAM_LEAD_TRIM=0 makes the stream byte-identical to the code before.
    stream_lead_trim: bool = field(default_factory=lambda: _env_flag("STREAM_LEAD_TRIM", True))
    stream_lead_trim_padding_ms: int = field(
        default_factory=lambda: int(os.getenv("STREAM_LEAD_TRIM_PADDING_MS") or 250)
    )
    # Shorter lead after a prefix. The forced first-silence token stays (generation unchanged: without it the
    # continuation's first word degrades), but for a request WITH a prefix the audio keeps only
    # PREFIX_LEAD_TRIM_PADDING_MS before the first sound: /stream's first-chunk lead trim (when STREAM_LEAD_TRIM is on)
    # and /predict's leading edge (when the request trims at all). Requests without a prefix are never affected.
    # Default 80 ms. PREFIX_LEAD_TRIM_PADDING_MS=off (or none) restores the old behaviour (STREAM_LEAD_TRIM_PADDING_MS /
    # the request's trim_silence_padding_ms); an integer sets another value; unset or empty = 80.
    prefix_lead_trim_padding_ms: int | None = field(default_factory=lambda: _prefix_lead_trim_padding_ms())
    # Streaming early-chunk check. STREAM_EARLY_CHECK=1 checks the first streamed chunk before it is played: its CB0
    # codes with 12794 counted as silence in the early silence ratio, and its decoded audio with /predict's
    # silent-audio check plus an active-speech floor (STREAM_EARLY_MIN_ACTIVE: the share of 20 ms frames at or above
    # -50 dBFS). A failed early chunk is re-sampled from scratch (STREAM_EARLY_CHECK_ATTEMPTS in all); if every
    # attempt fails, the stream starts at segment 0 without an early chunk, so a failed early prefix is never forced
    # into segment 0 (app/stream_early_check.py). Independent of STREAM_LEAD_TRIM. Default on; STREAM_EARLY_CHECK=0
    # leaves the stream unchanged.
    stream_early_check: bool = field(default_factory=lambda: _env_flag("STREAM_EARLY_CHECK", True))
    stream_early_check_attempts: int = field(
        default_factory=lambda: int(os.getenv("STREAM_EARLY_CHECK_ATTEMPTS") or 3)
    )
    stream_early_min_active: float = field(
        default_factory=lambda: float(os.getenv("STREAM_EARLY_MIN_ACTIVE") or 0.10)
    )
    # Streaming CB0 retries. Streaming keeps the played early codes as the prefix of every segment-0 attempt, and with
    # the request's seed attempt 2 repeats attempt 1 (later ones stay correlated), so its retries rarely change the
    # chars/s. STREAM_RETRY_FIX=1, streaming calls only: every CB0 retry draws a fresh sample (seed + attempt - 1), and
    # when every segment attempt fails (and no CB0-CTC best exists) the attempt closest to the chars/s window is kept
    # instead of the last. Default on; STREAM_RETRY_FIX=0 turns both parts off.
    stream_retry_fix: bool = field(default_factory=lambda: _env_flag("STREAM_RETRY_FIX", True))
    # Streaming early pace floor: with STREAM_EARLY_CHECK=1, STREAM_EARLY_PACE_MIN_ACTIVE > 0 also re-samples an
    # early chunk whose share of active 20 ms frames (at or above -50 dBFS) is below it: a slow, pause-heavy start
    # that segment 0 would keep as its prefix and that fails the chars/s window in every retry. 0 (default) = off.
    stream_early_pace_min_active: float = field(
        default_factory=lambda: float(os.getenv("STREAM_EARLY_PACE_MIN_ACTIVE") or 0.0)
    )
    # Request-end decode fix: /predict floors the last decode window's VibeVoice frame count, so the audio stops
    # 0-107 ms before the codes end, with no fade; /stream keeps the partial frame but runs up to 107 ms past the
    # codes, also with a hard stop. With DECODE_END_FIX=1 both decode the last partial frame, end exactly at the
    # codes' length and fade the last DECODE_END_FADE_MS ms (raised cosine). On by default; DECODE_END_FIX=0
    # restores the old output.
    decode_end_fix: bool = field(default_factory=lambda: _env_flag("DECODE_END_FIX", True))
    decode_end_fade_ms: float = field(
        default_factory=lambda: float(os.getenv("DECODE_END_FADE_MS") or 15.0)
    )
    # Prefix-continuation join: /predict with a prefix stripped the prefix codes BEFORE the decode, so the
    # continuation was decoded cold and its level / timbre jumped at the join. With PREFIX_JOINT_DECODE=1 the last
    # chunk_context_seconds of the prefix codes + the continuation codes are decoded in one pass (the prefix as the
    # decoder's acoustic left context) and the prefix's samples are cut off after the decode; no level matching or
    # crossfade. Only when prefix_tokens carries every codebook row (a client splice); prefix_text alone keeps the
    # old decode. PREFIX_JOINT_DECODE=0 restores the old output. Requests without a prefix: unchanged.
    prefix_joint_decode: bool = field(default_factory=lambda: _env_flag("PREFIX_JOINT_DECODE", True))
    # Prefix-continuation fade-in: serving returns only the continuation and the client concatenates it after its
    # own audio. When the client cuts its audio while it still sounds, the hard splice clicks (with or without
    # PREFIX_JOINT_DECODE). With PREFIX_START_FADE=1 the first PREFIX_START_FADE_MS ms (default 10) of every response to
    # a request with a prefix (non-empty prefix_text; prefix_tokens alone are ignored by serving) fade in with a raised
    # cosine, the mirror image of DECODE_END_FIX's end fade (the first sample becomes 0). /predict: on the final
    # audio (after trim, pause cap and MossFormer2, at the output sample rate), right before the encode. /stream and
    # /ws/stream: on the first emitted samples; a first chunk shorter than the fade is faded in full and the ramp
    # carries on into the next chunk. The client's half of the fix: fade the last 10 ms of the audio it keeps out
    # (app/splice.py; README "Continuing from your own audio"). Requests without a prefix: unchanged (bit-identical).
    # On by default; PREFIX_START_FADE=0 restores the old output.
    prefix_start_fade: bool = field(default_factory=lambda: _env_flag("PREFIX_START_FADE", True))
    prefix_start_fade_ms: float = field(
        default_factory=lambda: float(os.getenv("PREFIX_START_FADE_MS") or 10.0)
    )
    # Output headroom. The VibeVoice decoder overshoots: outputs can clip at the int16 export and peak above
    # -1 dBTP. /predict and /stream scale the decoded float waveform by 10^(-OUTPUT_HEADROOM_DB/20) right after the
    # decode (after DECODE_END_FIX), before trim, pause cap, MossFormer2 and the int16 conversion; with VibeVoice off
    # the DualCodec output is scaled the same way. Default 3.6 dB (x0.6607, loudness -3.6 LU); OUTPUT_HEADROOM_DB=0
    # restores the old output bit-exactly.
    output_headroom_db: float = field(
        default_factory=lambda: _env_headroom_db("OUTPUT_HEADROOM_DB", 3.6)
    )
    # VibeVoice residual adapter. VV_ADAPTER_CKPT=<checkpoint .pt> corrects /predict's VibeVoice latent from the
    # codes' DualCodec latent before the VibeVoice decode (app/vv_adapter.py). Unset or empty (default) = off: the
    # decode is byte-identical to the code before the flag. Needs ENABLE_VIBEVOICE=1; /stream uses it too with
    # STREAM_VV_ADAPTER (below).
    # Checkpoint identity: when VV_ADAPTER_CKPT is set, VV_ADAPTER_SHA256 is required and the file must hash to it, and
    # the hash must be in app/vv_adapter.APPROVED_SHA256 unless VV_ADAPTER_ALLOW_UNLISTED=1. TTSEngine.load checks
    # this before any model loads: a missing file, a mismatch or an unlisted hash stops the server at once.
    vv_adapter_ckpt: str = field(default_factory=lambda: os.getenv("VV_ADAPTER_CKPT", "").strip())
    vv_adapter_sha256: str = field(default_factory=lambda: os.getenv("VV_ADAPTER_SHA256", "").strip().lower())
    vv_adapter_allow_unlisted: bool = field(
        default_factory=lambda: os.getenv("VV_ADAPTER_ALLOW_UNLISTED", "0").strip() == "1"
    )
    # The VibeVoice adapter on /stream. With VV_ADAPTER_CKPT set, STREAM_VV_ADAPTER=1 corrects the streamed VibeVoice
    # frames as /predict's (the early chunk, every segment chunk and the flush with /predict's tail fix;
    # app/vv_adapter.correct_stream_frames): each frame gets what one adapter pass over the request gives on the
    # stream's own inputs. The adapter's 1-code lookahead fits in streaming_dc_pad_tokens, so nothing is held back
    # longer. Both streaming routes; generation, STREAM_LEAD_TRIM, STREAM_EARLY_CHECK and STREAM_RETRY_FIX are
    # unchanged. Default on; STREAM_VV_ADAPTER=0, or VV_ADAPTER_CKPT unset, leaves the stream byte-identical to the
    # code before.
    stream_vv_adapter: bool = field(default_factory=lambda: _env_flag("STREAM_VV_ADAPTER", True))
    force_first_silence: bool = True  # force first predicted cb0 token to be SILENCE_TOKEN_ID
    # Prefix continuations: forcing SILENCE_TOKEN_ID as the continuation's first CB0 token right after
    # [prefix][marker] (a format CB0 never sees in training) makes every mid-sentence continuation pause ~0.3 s.
    # PREFIX_FORCE_FIRST_SILENCE=0 lets the continuation start from the natural next code. Requests without a prefix
    # always force the silence. Default on: without the forced silence the continuation's first word is lost far more
    # often (mostly on responses that start speaking at frame 0); the shorter join is not worth it.
    prefix_force_first_silence: bool = field(default_factory=lambda: _env_flag("PREFIX_FORCE_FIRST_SILENCE", True))
    # When True, low-priority requests (e.g. /predict without priority="high")
    # pause between vLLM generation calls while any high-priority request is
    # active. Disable to fall back to pre-gate FCFS behavior.
    priority_gate_enabled: bool = True

    # Loaded once and applied only to English requests whose use_verbalization
    # flag is true.
    enable_verbalization: bool = os.getenv("ENABLE_VERBALIZATION", "0") == "1"
    # VERBALIZER_GUARD=1 turns on the pre-pass and word checks (app/verbalizer_guard.py) that were built for the
    # first verbalizer model; the current model drops citation markers, link parameters and markup on purpose,
    # which those checks would reject. Off, each output only has to be closed, of plausible length, and copy its
    # marked numbers from the input.
    verbalizer_guard: bool = field(default_factory=lambda: _env_flag("VERBALIZER_GUARD", False))
    # VERBALIZER_DRAFT=0: decode one token at a time with the lite engine instead of draft and verify.
    verbalizer_draft: bool = field(default_factory=lambda: _env_flag("VERBALIZER_DRAFT", True))
    # With VERBALIZER_GUARD=1 only.
    # Verbalizer growth guard: on top of the misread guard (app/verbalizer_guard.py check_output), reject a
    # verbalizer output that adds words the input does not explain (numbers, symbols and abbreviations may expand;
    # plain words may not grow) or runs far longer than the input (e.g. it returns its own system instructions) ->
    # the pre-pass text. VERBALIZER_GROWTH_GUARD=0 restores check_output alone. It runs on EVERY verbalizer call, so
    # with the verbalizer on it can change outputs of requests without a prefix too (with it at 0, the path equals
    # the misread guard alone).
    verbalizer_growth_guard: bool = field(default_factory=lambda: _env_flag("VERBALIZER_GROWTH_GUARD", True))
    # Prefix continuations with use_verbalization: the verbalizer saw only the rest of the sentence and read it as a
    # new one (capitalised first word, an inserted "A", dropped words, its own instructions). With
    # VERBALIZE_PREFIX_CONTEXT=1 the first segment is verbalized
    # after the end of prefix_text and only its own part is kept, in the rest's own first-letter case
    # (Verbalizer.verbalize_continuation). VERBALIZE_PREFIX_CONTEXT=0 restores the old path. Requests without a
    # prefix: unchanged.
    verbalize_prefix_context: bool = field(default_factory=lambda: _env_flag("VERBALIZE_PREFIX_CONTEXT", True))
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
        if self.cb0_boundary_drop_space and not self.cb0_boundary_tokens:
            raise ValueError(
                "CB0_BOUNDARY_DROP_SPACE applies to token windows only (CB0_BOUNDARY_TOKENS=1); "
                "legacy windows never change"
            )
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
        if self.lite_kv_tokens_per_cb is None:
            self.lite_kv_tokens_per_cb = tuple(
                model_len * max_num_seqs
                for model_len, max_num_seqs in zip(
                    self.max_model_len_per_cb,
                    self.max_num_seqs_per_cb,
                )
            )
        if len(self.lite_kv_tokens_per_cb) != n:
            raise ValueError(f"lite_kv_tokens_per_cb must contain {n} values")
        quanta = self.decode_window_seconds / STREAMING_SECONDS_QUANTUM
        if self.decode_window_seconds <= 0 or abs(quanta - round(quanta)) > 1e-6:
            raise ValueError("decode_window_seconds must be a positive multiple of 0.4")
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
