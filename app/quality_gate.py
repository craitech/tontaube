"""/predict quality gate.

A serving MITIGATION, not a model fix: it hides failures by resampling, it does not make the model fail less, and
/stream gets nothing from it. Behind ``PREDICT_QUALITY_GATE`` (default off).

Where it runs: per piece, inside /predict's CB0 attempt loop (``TTSEngine._generate_semantic_segment``), after an attempt
passed serving's own ``_check_cb0_generation`` and before CB1-3 and the decode. Only /predict
(``TTSEngine.generate``) builds a ``RequestGate``; /stream never does. English only (the CB0-CTC recogniser's language).

The checker (``QualityGate.score``) reads only the attempt's CB0 codes and the piece text: one fp32 forward of the
CB0-CTC recogniser (the guard's checkpoint, with its VAD head), greedy transcript, CTC Viterbi forced alignment of the
text, and code-level counts. Rules (a rule fires when ``direction * (value - threshold) > 0``):
  R_cap    CB0 frames >= 400 (serving's max_tokens; kept so the ranking sees it)
  R_len    speech span vs text: ((frames - leading - trailing silence codes) / 12.5 + 0.5) / E, E = 1 + 3 L / 14 s
           (L = letters of the normalised text), E divided by the rate-slider factor of the piece
           header. It covers every piece length, including replies of <= 30 chars that serving's chars/s window skips.
  R_cer    CER of the greedy transcript vs the normalised text
  R_last   mean log-prob of the last word on the alignment path, only when the greedy transcript also misses that word
  R_first  same for the first word
  R_cov    share of text words whose aligned mean log-prob is under T_CB0 = -1.4697
  R_unread distinct-code ratio (silence / hold; off at point B)
A logistic failure probability ``p`` (19 features) only RANKS attempts; it never rejects.

The policy (``RequestGate`` / ``PieceGate``): accept the first scored attempt with no rule flag; otherwise retry, at most
``max_attempts`` scored attempts per piece (attempts that fail serving's own check are retried as before and do not
count; at most ``cb0_max_retries`` of those, and ``cb0_max_retries + max_attempts - 1`` CB0 attempts per piece in all);
no new gate retry once the request's added time (gate-caused CB0 attempts + checker) reached ``budget_s``; then keep the
best scored attempt by (no R_cap, fewest flags, lowest p, earliest). When no attempt passed serving's check, the
serving-failed attempts are ranked the same way (instead of keeping the last). Seeds: attempt k (k >= 1) of piece i uses
``(seed + 7919 k + 104729 i) mod 2^31`` (also for serving's own retries), request-level pass r uses
``seed + 1299709 (r - 1)`` as its base; without a request seed vLLM draws a fresh one. A gate retry keeps the base
temperature and silence bias; serving's own retries keep serving's schedule. Any exception in the checker accepts the
attempt (fail open).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from app import cb0_ctc as C
from app.config import DC_SILENCE_TOKEN, SILENCE_TOKEN_IDS
from app.logging_utils import get_logger

logger = get_logger("quality_gate")

CAP_FRAMES = 400            # serving's max_new_tokens (32 s of CB0 at 12.5 Hz); R_cap
HZ = 12.5                   # CB0 frames per second
U = 4                       # CB0-CTC output frames per CB0 frame (50 Hz)
T_CB0 = -1.4697             # calibrated word-presence threshold (mean log-prob of a word's emission frames)
FA_FLOOR = -12.0            # a word / path without alignment counts as this log-prob
FILLERS = {"um", "uh", "hmm", "hm", "er", "ah", "erm", "mm", "mhm", "uhm", "umm", "uhh"}
SIL = set(SILENCE_TOKEN_IDS) | {DC_SILENCE_TOKEN}   # serving's silence codes + digital silence (19 codes)
SEED_STRIDE_ATTEMPT = 7919
SEED_STRIDE_PIECE = 104729
SEED_STRIDE_PASS = 1299709
SEED_MOD = 2 ** 31

# ---------------------------------------------------------------------------------------------------------------------
# FROZEN calibration: rule thresholds at three operating points ("frr02" = A, "frr05" = B, "frr10" = C) and the
# ranking logistic. Thresholds of +-1e9 = rule off. Do not edit: tests/test_quality_gate.py pins these values.
# ---------------------------------------------------------------------------------------------------------------------
RULES = {   # name -> (derived feature, direction): fires when direction * (value - threshold) > 0
    "R_cap": ("cap", +1),
    "R_len": ("speech_codes_ratio", +1),
    "R_unread": ("distinct", -1),
    "R_cer": ("cer", +1),
    "R_last": ("last_lp_g", -1),
    "R_first": ("first_lp_g", -1),
    "R_cov": ("frac_below_T", +1),
}
POINTS = {
    "A": {"R_cap": 0.0, "R_len": 2.4330991847826056, "R_unread": -1000000000.0, "R_cer": 1.0982115384615414,
          "R_last": -5.218758149147034, "R_first": -8.855300064086913, "R_cov": 0.9284495192307685},
    "B": {"R_cap": 0.0, "R_len": 2.2521739130434786, "R_unread": -1000000000.0, "R_cer": 0.6984343434343433,
          "R_last": -2.4672347784042357, "R_first": -8.855300064086913, "R_cov": 0.9284495192307685},
    "C": {"R_cap": 0.0, "R_len": 1.4128434782608676, "R_unread": 0.39655172413793105, "R_cer": 0.6984343434343433,
          "R_last": -2.25081247806549, "R_first": -4.143005695343017, "R_cov": 0.9284495192307685},
}
POINT_NAMES = {"A": "frr02", "B": "frr05", "C": "frr10"}
LOGIT_FEATS = ["codes_ratio_log", "speech_codes_ratio", "cap", "cer", "last_lp", "first_lp", "min_word_lp", "frac_below_T",
               "score_pf", "gap_to_eos", "after_end", "distinct", "top_share", "rep4", "ref_cov", "not_last", "not_first",
               "unread_g", "fa_ok"]
LOGISTIC_MEAN = [-0.5294074341276791, 0.7309581833520341, 0.043799212598425195, 0.19090846959022933, -0.8684900183021154,
                 -0.7883173557509752, -2.5410288996282024, 0.13320292937049835, -0.14954532290147787, 0.7691732283464576,
                 0.22427165354330775, 0.799737907518775, 0.1469244802402576, 0.08254408238869178, 0.7625224157950572,
                 0.25541338582677164, 0.2002952755905512, 0.04625984251968504, 1.0]
LOGISTIC_SCALE = [0.5822303177733977, 0.8259808982490586, 0.20464809203651024, 0.23656545759551878, 2.14220661905162,
                  2.012615632190093, 2.9668938540894176, 0.24113247061367826, 0.16683129344942377, 0.5522649369126379,
                  0.3097533007196445, 0.22039017621135712, 0.1859340569795272, 0.18291764553309253, 0.26146976052446746,
                  0.4360933250661807, 0.4002212864986849, 0.2100473029813457, 1.0]
LOGISTIC_COEF = [1.0288123717839106, 0.7055747131368527, 0.5319582356740796, 0.5709302992292612, -1.6506545964912085,
                 -0.8008241846600856, -0.20180877107468503, -0.42918964871136855, -0.2924797859364216, 0.09057819859930213,
                 0.20997543745849176, 0.7029052912181758, 0.5153336768994988, -0.5197056005775976, -0.06007475776262719,
                 0.29533526287962464, -0.01192990690517799, 0.20663655327328728, 0.0]
LOGISTIC_INTERCEPT = -3.9341579921820666


# ---------------------------------------------------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------------------------------------------------
def _lev(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[len(b)]


def _wsim(a, b):
    return _lev(a, b) / max(len(b), 1) <= 0.34


def _word_align(h, r):
    """Levenshtein word alignment of hypothesis h vs reference r (fuzzy word match) -> matched ref index per hyp word."""
    n, m = len(h), len(r)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    eq = [[_wsim(h[i], r[j]) for j in range(m)] for i in range(n)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (0 if eq[i - 1][j - 1] else 1))
    i, j, match = n, m, [None] * n
    while i > 0 and j > 0:
        if D[i, j] == D[i - 1, j - 1] + (0 if eq[i - 1][j - 1] else 1):
            if eq[i - 1][j - 1]:
                match[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif D[i, j] == D[i - 1, j] + 1:
            i -= 1
        else:
            j -= 1
    return match


def code_features(codes: list[int], text: str) -> dict:
    """The features that need no CB0-CTC forward (codes + text only)."""
    f = {}
    n = len(codes)
    f["frames"] = n
    f["dur_codes_s"] = n / HZ
    norm = C.normalize_text(text)
    f["norm"] = norm
    f["ref_letters"] = len(re.sub(r"[^a-z]", "", norm))
    f["exp_s_c1"] = 1.0 + 3 * f["ref_letters"] / 14.0
    c = np.asarray(codes)
    lead = 0
    for v in codes:
        if int(v) not in SIL:
            break
        lead += 1
    f["lead_sil_frames"] = lead
    if n:
        vals, cnt = np.unique(c, return_counts=True)
        f["top_code_share"] = float(cnt.max() / n)
        sil = np.isin(c, list(SIL))
        f["distinct_ratio"] = float(len(vals) / n)
        tail = 0
        for x in sil[::-1]:
            if not x:
                break
            tail += 1
        f["tail_sil_frames"] = tail
        nonsil = c[~sil]
        if len(nonsil) >= 8:
            grams = [tuple(nonsil[i:i + 4]) for i in range(len(nonsil) - 3)]
            gc = Counter(grams)
            f["rep4_share"] = float(sum(v for v in gc.values() if v > 1) / len(grams))
        else:
            f["rep4_share"] = 0.0
    else:
        f.update(top_code_share=0.0, distinct_ratio=0.0, tail_sil_frames=0, rep4_share=0.0)
    return f


def ctc_features(f: dict, lp: np.ndarray, ol: int, vad: np.ndarray | None) -> dict:
    """Adds the CB0-CTC features to ``code_features``' dict ``f``: lp [T50, 29] fp32 log-probs (numpy), ol the valid
    output length, vad [T50] speech probabilities or None."""
    import torch

    norm = f["norm"]
    rw = [w for w in norm.split() if w and w not in FILLERS]
    best = lp[:ol].argmax(-1)
    keep = np.r_[True, best[1:] != best[:-1]] & (best != 0) if ol else np.zeros(0, bool)
    g = re.sub(" +", " ", C.decode_ids(best[keep])).strip() if ol else ""
    hw = [w for w in g.split() if w and w not in FILLERS]
    f["greedy"] = g
    f["hyp_letters"] = len(re.sub(r"[^a-z]", "", g))
    f["cer"] = C.cer(g, norm) if norm else None
    run, mx = 1, 1 if hw else 0
    for i in range(1, len(hw)):
        run = run + 1 if hw[i] == hw[i - 1] else 1
        mx = max(mx, run)
    f["max_word_repeat"] = mx
    if rw:
        f["g_first_said"] = any(_wsim(w, rw[0]) for w in hw[:3])
        f["g_last_said"] = any(_wsim(w, rw[-1]) for w in hw[-3:])
        f["g_first_late"] = bool(hw) and not _wsim(hw[0], rw[0]) and any(_wsim(hw[0], t) for t in rw[1:3])
        m = _word_align(hw, rw) if hw else []
        matched = [j for j in m if j is not None]
        f["g_ref_cov"] = len(set(matched)) / len(rw)
        last_m = max([i for i, j in enumerate(m) if j is not None], default=-1)
        f["g_trailing_words"] = len(hw) - 1 - last_m if hw else 0
    tg = C.encode_text(norm)
    f["fa_ok"] = False
    if ol and tg:
        spans, score, ok = C.ctc_viterbi(torch.from_numpy(lp[None, :ol]), torch.tensor([ol]), [tg])[0]
        f["fa_ok"] = bool(ok)
        if ok:
            f["fa_score_pf"] = score / max(1, ol)
            wid, w = [], 0
            for ch in norm:
                if ch not in C.CHAR_TO_ID:
                    continue
                if ch == " ":
                    w += 1
                    wid.append(-1)
                else:
                    wid.append(w)
            words = norm.split(" ")
            wlp, wspan = {}, {}
            for k, (lab, wi) in enumerate(zip(tg, wid)):
                if wi < 0:
                    continue
                a0, a1 = int(spans[k][0]), int(spans[k][1])
                wlp.setdefault(wi, []).extend(lp[t, lab] for t in range(a0, a1))
                s = wspan.get(wi)
                wspan[wi] = (a0, a1) if s is None else (min(s[0], a0), max(s[1], a1))
            cw = [i for i, x in enumerate(words) if x and x not in FILLERS and i in wlp and wlp[i]]
            if cw:
                means = {i: float(np.mean(wlp[i])) for i in cw}
                f["fa_first_lp"] = means[cw[0]]
                f["fa_last_lp"] = means[cw[-1]]
                f["fa_min_word_lp"] = min(means.values())
                f["fa_words_below_T"] = sum(v < T_CB0 for v in means.values())
                f["fa_frac_below_T"] = f["fa_words_below_T"] / len(cw)
                en = wspan[cw[-1]][1]
                f["fa_gap_to_eos_s"] = (ol - en) / U / HZ
                if vad is not None:
                    v = vad[:ol] > 0.5
                    f["vad_after_end_s"] = float(v[en:].sum()) / U / HZ
    return f


def derive(f: dict, rate_factor: float = 1.0) -> dict:
    """Gate-side features derived from the fields above. ``rate_factor``: the expected length E is divided by the
    piece header's rate-slider factor (uncalibrated: the thresholds were calibrated without a slider; 1.0 = no
    slider)."""
    g = {}
    exp = f["exp_s_c1"] if rate_factor == 1.0 else f["exp_s_c1"] / rate_factor
    g["codes_ratio"] = f["dur_codes_s"] / exp
    speech_frames = max(0, f["frames"] - (f.get("lead_sil_frames") or 0) - f["tail_sil_frames"])
    g["speech_codes_ratio"] = (speech_frames / HZ + 0.5) / exp
    g["cap"] = float(f["frames"] >= CAP_FRAMES)
    g["cer"] = min(2.0, f["cer"] if f.get("cer") is not None else 2.0)
    ok = bool(f.get("fa_ok"))
    g["fa_ok"] = float(ok)
    g["last_lp"] = f.get("fa_last_lp", FA_FLOOR) if ok else FA_FLOOR
    g["first_lp"] = f.get("fa_first_lp", FA_FLOOR) if ok else FA_FLOOR
    g["min_word_lp"] = f.get("fa_min_word_lp", FA_FLOOR) if ok else FA_FLOOR
    g["frac_below_T"] = f.get("fa_frac_below_T", 1.0) if ok else 1.0
    g["score_pf"] = f.get("fa_score_pf", -5.0) if ok else -5.0
    g["gap_to_eos"] = min(3.0, f.get("fa_gap_to_eos_s", 0.0)) if ok else 0.0
    g["after_end"] = min(10.0, f.get("vad_after_end_s", 0.0)) if ok else 0.0
    g["distinct"] = f["distinct_ratio"]
    g["top_share"] = f["top_code_share"]
    g["rep4"] = f["rep4_share"]
    g["ref_cov"] = f.get("g_ref_cov", 0.0)
    g["not_last"] = float(not f.get("g_last_said", True))
    g["not_first"] = float(not f.get("g_first_said", True))
    g["unread_g"] = float(f.get("hyp_letters", 0) < min(3, math.ceil(0.5 * f["ref_letters"])))
    g["last_lp_g"] = g["last_lp"] if g["not_last"] else 0.0
    g["first_lp_g"] = g["first_lp"] if g["not_first"] else 0.0
    return g


def logistic_p(g: dict) -> float:
    v = dict(g)
    v["codes_ratio_log"] = math.log(max(1e-3, g["codes_ratio"]))
    x = np.array([float(v[k]) for k in LOGIT_FEATS])
    z = (x - np.array(LOGISTIC_MEAN)) / np.array(LOGISTIC_SCALE)
    return float(1 / (1 + np.exp(-(z @ np.array(LOGISTIC_COEF) + LOGISTIC_INTERCEPT))))


def rule_flags(g: dict, thresholds: dict) -> dict:
    return {r: bool(d * (g[feat] - thresholds[r]) > 0) for r, (feat, d) in RULES.items()}


# ---------------------------------------------------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------------------------------------------------
@dataclass
class Decision:
    passed: bool
    flags: dict = field(default_factory=dict)          # rule -> fired
    values: dict = field(default_factory=dict)         # rule -> the value its threshold is compared with
    p: float = 1.0                                     # logistic failure probability (ranking only)
    ctc: bool = True                                   # False: CB0-CTC forward skipped (cap) or failed
    ms: float = 0.0
    greedy: str | None = None
    error: str | None = None

    @property
    def fired(self) -> list[str]:
        return [r for r, v in self.flags.items() if v]

    def rank(self, order: int) -> tuple:
        """Best first: no R_cap, fewest flags, lowest p, earliest."""
        return (bool(self.flags.get("R_cap")), len(self.fired), self.p, order)

    def record(self) -> dict:
        return {"passed": self.passed, "fired": self.fired, "values": {r: round(v, 4) for r, v in self.values.items()},
                "p": round(self.p, 4), "ctc": self.ctc, "ms": round(self.ms, 1), "greedy": self.greedy, "error": self.error}


class QualityGate:
    """The checker and the per-server gate settings. ``recognizer``: the loaded CB0-CTC (app.cb0_ctc.CB0CTC); the gate
    always runs it in fp32 (``forward_fp32``), on its device (CB0_CTC_DEVICE)."""

    def __init__(self, recognizer, point: str = "B", max_attempts: int = 4, budget_s: float = 2.0,
                 log_path: str | None = None, rate_window=None, languages=("english",)):
        if point not in POINTS:
            raise ValueError(f"PREDICT_GATE_POINT must be one of {sorted(POINTS)}, got {point!r}")
        if max_attempts < 1:
            raise ValueError("PREDICT_GATE_MAX_ATTEMPTS must be at least 1")
        if not budget_s >= 0:
            raise ValueError("PREDICT_GATE_BUDGET_S must be >= 0")
        self.recognizer = recognizer
        self.point = point
        self.thresholds = dict(POINTS[point])
        self.max_attempts = int(max_attempts)
        self.budget_s = float(budget_s)
        self.log_path = log_path
        self.rate_window = rate_window
        self.languages = tuple(languages)
        self._log_lock = threading.Lock()

    # -- checker ---------------------------------------------------------------------------------------------------
    def features(self, codes: list[int], text: str, skip_ctc_on_cap: bool = True) -> dict:
        """code + CB0-CTC features of one attempt (local CB0 codes, piece text)."""
        f = code_features(codes, text)
        if skip_ctc_on_cap and len(codes) >= CAP_FRAMES:
            f["ctc_skipped"] = True
            return f
        lp, ol, vad = self.recognizer.forward_fp32([list(codes)])
        lp = lp[0].cpu().numpy()
        n = int(ol[0])
        vad = vad[0].cpu().numpy() if vad is not None else None
        return ctc_features(f, lp, n, vad)

    def decide(self, f: dict, rate_factor: float = 1.0) -> Decision:
        if f.get("ctc_skipped"):   # a cap: rejected on the frame count, no CB0-CTC forward
            g = derive({**f, "cer": None, "fa_ok": False}, rate_factor)
            flags = {r: False for r in RULES}
            for r in ("R_cap", "R_len", "R_unread"):
                flags[r] = rule_flags(g, self.thresholds)[r]
            values = {r: g[RULES[r][0]] for r in ("R_cap", "R_len", "R_unread")}
            return Decision(passed=False, flags=flags, values=values, p=1.0, ctc=False)
        g = derive(f, rate_factor)
        flags = rule_flags(g, self.thresholds)
        values = {r: g[feat] for r, (feat, _) in RULES.items()}
        return Decision(passed=not any(flags.values()), flags=flags, values=values, p=logistic_p(g), ctc=True,
                        greedy=f.get("greedy"))

    def score(self, codes: list[int], text: str, rate_factor: float = 1.0, skip_ctc_on_cap: bool = True) -> Decision:
        t0 = time.perf_counter()
        d = self.decide(self.features(codes, text, skip_ctc_on_cap), rate_factor)
        d.ms = 1000 * (time.perf_counter() - t0)
        return d

    def score_safe(self, codes: list[int], text: str, rate_factor: float = 1.0) -> Decision:
        """``score``, failing open: any exception accepts the attempt (logged), as the CB0-CTC guard does."""
        t0 = time.perf_counter()
        try:
            return self.score(codes, text, rate_factor)
        except Exception as e:  # noqa: BLE001
            logger.exception("Quality gate checker failed; accepting the attempt")
            return Decision(passed=True, p=0.0, ctc=False, ms=1000 * (time.perf_counter() - t0), error=repr(e)[:200])

    # -- policy ------------------------------------------------------------------------------------------------------
    def new_request(self, seed: int | None, voice_prompt=None, n_pieces: int = 1,
                    serving_max_retries: int = 5) -> "RequestGate":
        return RequestGate(self, seed, _voice_sha1(voice_prompt), n_pieces, serving_max_retries)

    def emit(self, record: dict) -> None:
        if not self.log_path:
            return
        try:
            line = json.dumps(record, default=_json_default)
            with self._log_lock:
                d = os.path.dirname(self.log_path)
                if d:
                    os.makedirs(d, exist_ok=True)
                with open(self.log_path, "a") as fo:
                    fo.write(line + "\n")
        except Exception:  # noqa: BLE001
            logger.exception("Quality gate: could not write PREDICT_GATE_LOG")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


def _voice_sha1(voice_prompt) -> str | None:
    try:
        if not voice_prompt or not voice_prompt[0]:
            return None
        return hashlib.sha1(json.dumps([int(x) for x in voice_prompt[0]]).encode()).hexdigest()[:16]
    except Exception:  # noqa: BLE001
        return None


def seed_for(seed: int | None, k: int, piece: int) -> int | None:
    """Seed of attempt k (0 = first) of piece ``piece``: the base seed for k = 0, else
    (seed + 7919 k + 104729 i) mod 2^31; None (no request seed) stays None, so vLLM draws a fresh seed."""
    if seed is None:
        return None
    if k == 0:
        return seed
    return (int(seed) + SEED_STRIDE_ATTEMPT * k + SEED_STRIDE_PIECE * piece) % SEED_MOD


class RequestGate:
    """One /predict request's gate state: seeds, the added-time budget, per-piece gates, logging."""

    def __init__(self, gate: QualityGate, seed: int | None, voice_sha1: str | None, n_pieces: int,
                 serving_max_retries: int):
        self.gate = gate
        self.id = uuid.uuid4().hex[:12]
        self.seed = seed
        self.voice_sha1 = voice_sha1
        self.n_pieces = n_pieces
        self.serving_max_retries = int(serving_max_retries)
        self.pass_idx = 1
        self.pass_seed = seed
        self.added_s = 0.0
        self.budget_hit = False
        self.pieces: list[dict] = []           # piece summaries of the current request-level pass
        self.passes: list[list[dict]] = []     # earlier passes
        self.t0 = time.perf_counter()

    def begin_pass(self, r: int) -> int | None:
        """Start request-level pass r (1 = first): -> the pass's base seed (the request seed on pass 1)."""
        if r > 1 and self.pieces:
            self.passes.append(self.pieces)
            self.pieces = []
        self.pass_idx = r
        self.pass_seed = None if self.seed is None else (
            self.seed if r == 1 else (int(self.seed) + SEED_STRIDE_PASS * (r - 1)) % SEED_MOD)
        return self.pass_seed

    def piece(self, seg_idx: int, text: str, system_tag: str, early_mode: bool = False):
        """PieceGate for one CB0 piece, or None (pass-through: early streaming chunk, unknown / non-English language,
        a text without letters)."""
        from app.cb0_guard import parse_cb0_header

        if early_mode:
            return None
        language, style, rate_bin = parse_cb0_header(system_tag)
        if language not in self.gate.languages:
            logger.info("Quality gate piece %d: pass-through (lang %s)", seg_idx, language)
            return None
        if not C.normalize_text(text):
            logger.info("Quality gate piece %d: pass-through (no letters)", seg_idx)
            return None
        factor = 1.0
        if rate_bin is not None and self.gate.rate_window is not None:
            factor = float(self.gate.rate_window.factor(style, rate_bin))
        return PieceGate(self, seg_idx, text, factor)

    def pass_rank(self) -> tuple:
        """Ranking key of the current pass's kept pieces (best first): R_cap count, flags, max p."""
        caps = sum(1 for p in self.pieces if "R_cap" in (p.get("kept_fired") or []))
        flags = sum(len(p.get("kept_fired") or []) for p in self.pieces)
        maxp = max([p.get("kept_p") or 0.0 for p in self.pieces], default=0.0)
        return caps, flags, maxp

    def finish(self, kept_pass: int, n_passes: int, pass_reasons: list) -> dict:
        all_passes = self.passes + [self.pieces]
        pieces = [p for ps in all_passes for p in ps]
        rec = {"type": "request", "request": self.id, "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "point": self.gate.point, "request_seed": self.seed, "voice_sha1": self.voice_sha1,
               "n_pieces": self.n_pieces, "passes": n_passes, "kept_pass": kept_pass, "pass_reasons": pass_reasons,
               "attempts_total": sum(p["n_attempts"] for p in pieces), "scored_total": sum(p["n_scored"] for p in pieces),
               "gate_retries": sum(p["gate_retries"] for p in pieces), "added_ms": round(1000 * self.added_s, 1),
               "budget_hit": self.budget_hit,
               "kept_fired": [p.get("kept_fired") for p in (all_passes[kept_pass - 1] if 0 < kept_pass <= len(all_passes) else [])],
               "wall_ms": round(1000 * (time.perf_counter() - self.t0), 1)}
        logger.info("Quality gate request %s: %d piece(s), %d pass(es) (kept %d), %d CB0 attempts, %d scored, %d gate "
                    "retries, added %.0f ms%s", self.id, self.n_pieces, n_passes, kept_pass, rec["attempts_total"],
                    rec["scored_total"], rec["gate_retries"], rec["added_ms"], ", BUDGET HIT" if self.budget_hit else "")
        self.gate.emit(rec)
        return rec


class PieceGate:
    """Accept-first / best-of-N policy for one CB0 piece (see the module docstring)."""

    def __init__(self, request: RequestGate, seg_idx: int, text: str, rate_factor: float):
        self.request = request
        self.gate = request.gate
        self.seg_idx = seg_idx
        self.text = text
        self.rate_factor = rate_factor
        self.serving_cap = request.serving_max_retries
        self.attempt_cap = self.serving_cap + self.gate.max_attempts - 1
        self.serving_failures = 0
        self.attempts: list[dict] = []
        self.scored: list[tuple] = []          # (Decision, order, payload, record)
        self.failed: list[tuple] = []          # (codes, payload, record) of serving-failed attempts
        self.accepted = None
        self.stop_reason = None
        self._since: float | None = None       # perf_counter of the last gate rejection (its retry's clock)

    # -- schedule ----------------------------------------------------------------------------------------------------
    def seed(self, attempt: int) -> int | None:
        return seed_for(self.request.pass_seed, attempt - 1, self.seg_idx)

    @property
    def retry_steps(self) -> int:
        """Steps of serving's temperature / silence-bias schedule for the next attempt: serving's own failures only
        (a gate retry keeps the base values)."""
        return self.serving_failures

    def _clock(self, t_check0: float, t_check1: float) -> None:
        if self._since is None:
            self.request.added_s += t_check1 - t_check0        # checker time of an attempt the gate did not cause
        else:
            self.request.added_s += t_check1 - self._since     # the gate retry's CB0 attempt(s) + this check
            self._since = None

    # -- per attempt -------------------------------------------------------------------------------------------------
    async def check(self, serving_failure: str | None, codes: list[int], payload, seed=None) -> str | None:
        """Call for every CB0 attempt with serving's own verdict. Returns a retry reason, or None to stop retrying (the
        piece then keeps ``finish()``'s choice)."""
        rec = {"attempt": len(self.attempts) + 1, "seed": seed, "frames": len(codes)}
        self.attempts.append(rec)
        if serving_failure is not None:
            self.serving_failures += 1
            rec.update(kind="serving", serving_failure=serving_failure, codes=list(codes))
            self.failed.append((list(codes), payload, rec))
            if self.serving_failures >= self.serving_cap:
                self.stop_reason = "serving retries spent"
                return None
            return serving_failure
        t0 = time.perf_counter()
        d = await asyncio.to_thread(self.gate.score_safe, codes, self.text, self.rate_factor)
        t1 = time.perf_counter()
        self._clock(t0, t1)
        rec.update(kind="scored", codes=list(codes), **d.record())
        self.scored.append((d, len(self.scored), payload, rec))
        if d.passed:
            rec["verdict"] = "accepted"
            self.accepted = payload
            self.stop_reason = "accepted"
            return None
        rec["verdict"] = "rejected"
        if len(self.scored) >= self.gate.max_attempts:
            self.stop_reason = "max scored attempts"
            return None
        if self.request.added_s >= self.gate.budget_s:
            self.stop_reason = "budget"
            self.request.budget_hit = True
            return None
        self._since = t1
        return "quality_gate (" + ", ".join(d.fired) + ")"

    async def finish(self, last_payload):
        """-> the attempt payload the piece keeps; logs the piece."""
        if self._since is not None:
            self.request.added_s += time.perf_counter() - self._since
            self._since = None
        if self.stop_reason is None:
            self.stop_reason = "attempt cap"
        kept, how, kept_rec = last_payload, "last", (self.attempts[-1] if self.attempts else None)
        if self.accepted is not None:
            d, _, payload, rec = self.scored[-1]
            kept, how, kept_rec = payload, "accepted", rec
        elif self.scored:
            d, _, payload, rec = min(self.scored, key=lambda s: s[0].rank(s[1]))
            kept, how, kept_rec = payload, "best_scored", rec
        elif self.failed:
            # no attempt passed serving's own check: rank those attempts like scored ones instead of keeping the last
            try:
                ranked = []
                for order, (codes, payload, rec) in enumerate(self.failed):
                    dd = await asyncio.to_thread(self.gate.score, codes, self.text, self.rate_factor)
                    rec.update(rank_only=dd.record())
                    ranked.append((dd.rank(order), payload, rec))
                _, kept, kept_rec = min(ranked, key=lambda x: x[0])
                how = "best_of_serving_failed"
            except Exception as e:  # noqa: BLE001
                logger.exception("Quality gate: ranking the serving-failed attempts failed; keeping the last")
                how = f"last (ranking failed: {e!r})"[:120]
        if kept_rec is not None:
            kept_rec["kept"] = True
        kept_fired = (kept_rec or {}).get("fired") or ((kept_rec or {}).get("rank_only") or {}).get("fired") or []
        kept_p = (kept_rec or {}).get("p", ((kept_rec or {}).get("rank_only") or {}).get("p"))
        gate_retries = sum(1 for r in self.attempts[:-1] if r.get("verdict") == "rejected")
        summary = {"piece": self.seg_idx, "n_attempts": len(self.attempts), "n_scored": len(self.scored),
                   "gate_retries": gate_retries, "kept_attempt": (kept_rec or {}).get("attempt"), "kept_how": how,
                   "kept_fired": kept_fired, "kept_p": kept_p, "stop": self.stop_reason}
        self.request.pieces.append(summary)
        text_sha1 = hashlib.sha1(self.text.encode()).hexdigest()[:12]
        parts = []
        for r in self.attempts:
            if r.get("kind") == "serving":
                parts.append(f"{r['attempt']}: seed {r['seed']} {r['frames']} fr serving {r['serving_failure']}")
            else:
                vals = ", ".join(f"{k} {r['values'].get(k)}" for k in r.get("fired") or [])
                parts.append(f"{r['attempt']}: seed {r['seed']} {r['frames']} fr "
                             f"{'pass' if r.get('passed') else 'FLAG ' + vals} p {r.get('p')} {r.get('verdict')} "
                             f"{r.get('ms')} ms" + (f" error {r['error']}" if r.get("error") else ""))
        logger.info("Quality gate piece %d (pass %d, %d chars, text sha1 %s, voice %s, request seed %s, rate x%.3g): %s; "
                    "kept attempt %s (%s), request added %.0f ms%s", self.seg_idx, self.request.pass_idx, len(self.text),
                    text_sha1, self.request.voice_sha1, self.request.seed, self.rate_factor, " | ".join(parts),
                    summary["kept_attempt"], how, 1000 * self.request.added_s,
                    ", BUDGET HIT" if self.request.budget_hit else "")
        self.gate.emit({"type": "piece", "request": self.request.id,
                        "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "point": self.gate.point,
                        "pass": self.request.pass_idx, "piece": self.seg_idx, "text": self.text, "text_sha1": text_sha1,
                        "chars": len(self.text), "voice_sha1": self.request.voice_sha1,
                        "request_seed": self.request.seed, "rate_factor": self.rate_factor, "attempts": self.attempts,
                        **summary, "request_added_ms": round(1000 * self.request.added_s, 1),
                        "budget_hit": self.request.budget_hit})
        return kept


def build_quality_gate(config, recognizer, rate_window=None):
    """QualityGate for TTSEngine.load(), or None (flag off, or no CB0-CTC recogniser: gate off with a warning)."""
    if not getattr(config, "predict_quality_gate", False):
        return None
    if recognizer is None:
        logger.warning("PREDICT_QUALITY_GATE is on but no CB0-CTC checkpoint was loaded (set CB0_CTC_MODEL_PATH); "
                       "quality gate off")
        return None
    if rate_window is None:
        from app.cb0_guard import RateWindow

        rate_window = RateWindow.from_config(config)
    gate = QualityGate(recognizer, point=config.predict_gate_point, max_attempts=config.predict_gate_max_attempts,
                       budget_s=config.predict_gate_budget_s, log_path=config.predict_gate_log, rate_window=rate_window,
                       languages=config.cb0_ctc_guard_languages)
    logger.info("Quality gate (/predict): point %s (%s), at most %d scored attempts per piece, budget %.2f s per "
                "request, CB0-CTC fp32 on %s%s", gate.point, POINT_NAMES[gate.point], gate.max_attempts, gate.budget_s,
                getattr(recognizer, "device", "?"), f", log {gate.log_path}" if gate.log_path else "")
    return gate
