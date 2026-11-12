"""CB0 retry-check additions for the control sliders and the CB0-CTC recogniser.

The rate window and the guard key off the CB0 system header a segment is generated with,
``<|im_start|>english : audiobook<|ctrl_rate_03|><|im_end|>\\n`` (app/controls.py).

* ``RateWindow``: rate-aware chars/s window. Serving rejects a CB0 attempt whose chars/s fall
  outside ``cb0_min_cps``..``cb0_max_cps`` (8-20). The rate slider moves the expected rate, so
  both bounds are scaled by the factor of the header's rate bin (median chars/s of the bin over the
  no-slider median, measured on the serving model). No rate token = x1.
* ``SegmentGuard``: CB0-CTC guard. After an attempt passes serving's check, the CER of its
  CB0-CTC transcript against the segment text decides: CER > ``max_cer`` -> retry, at most
  ``max_retries`` guard retries inside serving's own attempt budget. The segment then keeps the
  attempt with the lowest CER among those that passed serving's check (none passed = the last
  attempt, serving's rule).
* ``PrevAudioChain``: previous-audio hygiene. A non-speech segment passed on as previous audio
  tends to make the next segment run away too. After a segment whose kept attempt failed the guard,
  the next segment starts like a chunk start: no previous audio and no previous text (with segment
  i-1's text left in the window the model re-reads it). A retry of a non-speech attempt does the same.
* ``retry_seed`` / ``longest_run``: one fresh, reproducible seed per retry attempt, and the run length
  behind the whole-piece gap check (``cb0_max_gap_seconds``).
"""

import asyncio
import json
import math
import os
import re

from app.logging_utils import get_logger

logger = get_logger("cb0_guard")

CB0_CTC_FILENAME = "cb0_ctc.pt"
RUNAWAY_HYP_CHARS = 3  # a CB0-CTC transcript shorter than this = non-speech attempt
_HEADER_RE = re.compile(r"<\|im_start\|>\s*([^:<]+?)\s*:\s*([^<\s]+)")
_RATE_TOKEN_RE = re.compile(r"<\|ctrl_rate_(\d+)\|>")
_RATE_BIN_KEY_RE = re.compile(r"(?:rate)?(\d+)")


def parse_cb0_header(system_tag: str) -> tuple[str | None, str | None, int | None]:
    """(language, style, rate bin) named in a CB0 system header; None where absent."""
    header = _HEADER_RE.match(system_tag or "")
    rate = _RATE_TOKEN_RE.search(system_tag or "")
    return (
        header.group(1).lower() if header else None,
        header.group(2).lower() if header else None,
        int(rate.group(1)) if rate else None,
    )


def interpolate_factor(table: dict[int, float], rate_bin: int) -> float | None:
    """Factor of ``rate_bin``: measured value, else log-linear between the nearest measured bins,
    else (outside the measured range) the nearest end. None for an empty table."""
    if not table:
        return None
    if rate_bin in table:
        return table[rate_bin]
    bins = sorted(table)
    if rate_bin < bins[0]:
        return table[bins[0]]
    if rate_bin > bins[-1]:
        return table[bins[-1]]
    lo = max(b for b in bins if b < rate_bin)
    hi = min(b for b in bins if b > rate_bin)
    w = (rate_bin - lo) / (hi - lo)
    return math.exp((1 - w) * math.log(table[lo]) + w * math.log(table[hi]))


def load_factor_table(path: str, model: str | None = None) -> dict[str, dict[int, float]]:
    """Read a ``rate_window_factors.json`` (model -> style -> bin -> factor) for one
    model: style -> {bin: factor}. Bins are keyed ``"rate<k>"`` or ``"<k>"``; a factor is a number or
    ``{"factor": x, ...}``. Keys starting with ``_`` are metadata; ``"none"`` (the x1 reference) is
    skipped. ``model`` may be omitted when the file holds a single model."""
    with open(path) as f:
        data = json.load(f)
    models = [k for k in data if not k.startswith("_")]
    if model is None:
        if len(models) != 1:
            raise ValueError(f"{path} holds models {models}; set CB0_RATE_CPS_FACTORS_MODEL")
        model = models[0]
    if model not in models:
        raise ValueError(f"model {model!r} not in {path} (models: {models})")
    styles: dict[str, dict[int, float]] = {}
    for style, bins in data[model].items():
        table = {}
        for key, value in bins.items():
            match = _RATE_BIN_KEY_RE.fullmatch(str(key))
            if match is None:
                continue
            factor = float(value["factor"] if isinstance(value, dict) else value)
            if factor <= 0:
                raise ValueError(f"{path}: {model}/{style}/{key} factor must be positive")
            table[int(match.group(1))] = factor
        styles[style.lower()] = table
    return styles


class RateWindow:
    """Chars/s window scale per CB0 header: style table -> pooled ("all") table -> default table."""

    def __init__(
        self,
        default: dict[int, float],
        styles: dict[str, dict[int, float]] | None = None,
    ):
        if any(f <= 0 for f in default.values()):
            raise ValueError("rate factors must all be positive")
        self.default = dict(default)
        self.styles = styles or {}

    @classmethod
    def from_config(cls, config) -> "RateWindow":
        styles = None
        if config.cb0_rate_cps_factors_json:
            styles = load_factor_table(
                config.cb0_rate_cps_factors_json, config.cb0_rate_cps_factors_model,
            )
        return cls(config.cb0_rate_cps_factors, styles)

    def factor(self, style: str | None, rate_bin: int | None) -> float:
        if rate_bin is None:
            return 1.0
        for table in (self.styles.get(style or ""), self.styles.get("all"), self.default):
            factor = interpolate_factor(table or {}, rate_bin)
            if factor is not None:
                return factor
        return 1.0


def load_recognizer(config, cb0_model_path: str):
    """CB0-CTC recogniser for the guard and the /predict quality gate, or None (both off): flags off,
    or no checkpoint at ``cb0_ctc_model_path`` / ``<CB0 model dir>/cb0_ctc.pt``."""
    gate = getattr(config, "predict_quality_gate", False)
    if not (config.cb0_ctc_guard or gate):
        return None
    path = config.cb0_ctc_model_path
    if path is None:
        path = os.path.join(cb0_model_path, CB0_CTC_FILENAME)
        if not os.path.isfile(path):
            if config.cb0_ctc_guard:
                logger.warning(
                    "CB0-CTC guard is on but no checkpoint was found (set CB0_CTC_MODEL_PATH or ship "
                    "%s in the CB0 model dir); guard off",
                    CB0_CTC_FILENAME,
                )
            if gate:
                logger.warning(
                    "PREDICT_QUALITY_GATE is on but no CB0-CTC checkpoint was found (set CB0_CTC_MODEL_PATH "
                    "or ship %s in the CB0 model dir); quality gate off",
                    CB0_CTC_FILENAME,
                )
            return None
    import torch
    from app import cb0_ctc

    device = config.cb0_ctc_device or ("cuda" if torch.cuda.is_available() else "cpu")
    recognizer = cb0_ctc.load(path, device)
    if config.cb0_ctc_guard:
        logger.info(
            "CB0-CTC guard: %s on %s (CER > %.2f -> retry, at most %d guard retries)",
            path, device, config.cb0_ctc_max_cer, config.cb0_ctc_max_retries,
        )
    else:
        logger.info("CB0-CTC recogniser for the /predict quality gate: %s on %s (guard off)", path, device)
    return recognizer


class SegmentGuard:
    """CB0-CTC guard state for one segment's attempts (see the module docstring)."""

    def __init__(self, recognizer, text: str, max_cer: float, max_retries: int):
        self.recognizer = recognizer
        self.text = text
        self.max_cer = max_cer
        self.max_retries = max_retries
        self.retries = 0
        self.passed: list[tuple[float, int, object]] = []  # (CER, order, attempt) of serving-ok attempts
        self._last: tuple[list[int], tuple[float, bool]] | None = None

    def transcribe(self, codes: list[int]) -> str:
        return self.recognizer.recognize([codes])[0]

    async def score(self, codes: list[int]) -> tuple[float, bool]:
        """(CER vs the segment text, non-speech: transcript under RUNAWAY_HYP_CHARS) of one attempt's
        local codes; the last result is reused for the same codes. Fails open: (0.0, False)."""
        if self._last is not None and self._last[0] is codes:
            return self._last[1]
        if not codes:
            result = (1.0, True)
        else:
            from app.cb0_ctc import cer

            try:
                hyp = await asyncio.to_thread(self.transcribe, codes)
                result = (cer(hyp, self.text), len(hyp) < RUNAWAY_HYP_CHARS)
            except Exception:
                logger.exception("CB0-CTC guard failed; accepting the attempt")
                result = (0.0, False)
        self._last = (codes, result)
        return result

    async def check(self, codes: list[int], attempt) -> str | None:
        """Call for an attempt that passed serving's check. Returns a retry reason, or None to stop
        retrying (the segment then keeps ``best()``)."""
        cer, _ = await self.score(codes)
        self.passed.append((cer, len(self.passed), attempt))
        if cer <= self.max_cer or self.retries >= self.max_retries:
            return None
        self.retries += 1
        return f"ctc_cer ({cer:.2f} > {self.max_cer:g})"

    def best(self):
        """The serving-ok attempt with the lowest CER (earliest on ties), or None."""
        return min(self.passed, key=lambda p: p[:2])[2] if self.passed else None

    def best_cer(self) -> float | None:
        return min(p[0] for p in self.passed) if self.passed else None


class PrevAudioChain:
    """Previous audio for one request's CB0 segments (CB0_DROP_BAD_PREV_AUDIO).

    ``previous`` is the preceding segment's kept output if that attempt passed the guard
    (CER <= max_cer) or was not scored, else None. There is no fallback to an earlier segment:
    pairing an earlier segment's audio with segment i-1's text is worse than a clean start.
    Callers generate a segment with ``previous`` None (after the first) without context.
    """

    def __init__(self, max_cer: float):
        self.max_cer = max_cer
        self.previous: list[int] | None = None

    def add(self, seg_global: list[int], kept_cer: float | None) -> None:
        self.previous = seg_global if kept_cer is None or kept_cer <= self.max_cer else None

    def drops_context(self, seg_idx: int) -> bool:
        """Whether segment ``seg_idx`` starts without previous audio and text."""
        return seg_idx > 0 and self.previous is None


# Retry seeds: with the request seed on every CB0 retry attempt (the first retry also keeps the base temperature),
# attempt 2 would regenerate attempt 1 exactly: one wasted CB0 call per failing piece and a retry budget one less
# than configured.
RETRY_SEED_STRIDE = 1_000_003  # a prime; attempt k's seed is seed + (k - 1) * stride, mod 2 ** 31


def retry_seed(seed: int | None, attempt: int) -> int | None:
    """The sampling seed of CB0 attempt ``attempt`` (1-based) of a request with seed ``seed``.

    Attempt 1 keeps the request seed (unchanged first attempts); each retry gets its own seed, derived only from the
    request seed and the attempt index, so a request stays reproducible. No request seed (None) stays None: the
    engine then draws a fresh random seed per call already.
    """
    if seed is None or attempt <= 1:
        return seed
    return (int(seed) + (attempt - 1) * RETRY_SEED_STRIDE) % (2 ** 31)


def longest_run(codes: list[int], members: set[int]) -> int:
    """Length of the longest run of consecutive ``codes`` that are all in ``members`` (0 for none)."""
    best = run = 0
    for c in codes:
        run = run + 1 if c in members else 0
        if run > best:
            best = run
    return best
