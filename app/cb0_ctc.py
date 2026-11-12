"""CB0 -> text recogniser (CTC) for the serving CB0-CTC guard and the /predict quality gate (app/quality_gate.py).

Reads top-1 DualCodec semantic codes (CB0, 12.5 Hz, 16384 codes) and emits characters at 50 Hz
(greedy CTC). Text normalisation (``normalize_text``): NFKD with accents stripped, lowercase,
numbers spelled out (a number followed by a comma is kept: "In 1984, he"), a few symbols as words,
every other character a word separator, apostrophes kept only inside words, single spaces.
Output alphabet: " abcdefghijklmnopqrstuvwxyz'". ``edit_distance`` / ``cer`` score a transcript
against the normalised reference. ``encode_text`` and the forced aligner ``ctc_viterbi`` serve the
quality gate, which also reads the log-probs and the VAD head in fp32 (``CB0CTC.forward_fp32``).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Character set and text normalisation
# ---------------------------------------------------------------------------

CHARSET = " abcdefghijklmnopqrstuvwxyz'"
BLANK = 0
CHAR_TO_ID = {c: i + 1 for i, c in enumerate(CHARSET)}
ID_TO_CHAR = {i + 1: c for i, c in enumerate(CHARSET)}
N_OUT = len(CHARSET) + 1
SPACE_ID = CHAR_TO_ID[" "]

N_CODES = 16384
MASK_CODE = N_CODES       # unknown / masked frame (also used for out-of-range codes)
PAD_CODE = N_CODES + 1

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
         "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
_SCALES = [(10 ** 12, "trillion"), (10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand")]
_ORD_IRREG = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth",
              "nine": "ninth", "twelve": "twelfth"}
_LETTER_MAP = {"æ": "ae", "œ": "oe", "ß": "ss", "ø": "o", "đ": "d", "ł": "l", "þ": "th", "ð": "d", "ı": "i",
               "’": "'", "‘": "'", "ʼ": "'", "`": "'", "´": "'"}
_SYMBOL_WORDS = {"&": "and", "%": "percent", "+": "plus", "=": "equals", "@": "at", "°": "degrees",
                 "$": "dollars", "£": "pounds", "€": "euros"}
_CURRENCY = {"$": "dollars", "£": "pounds", "€": "euros"}

_TOKEN_RE = re.compile(
    r"(?P<money>[$£€]\s?\d(?:[\d,]*\d)?(?:\.\d+)?)"
    r"|(?P<num>\d(?:[\d,]*\d)?(?:\.\d+)?(?:st|nd|rd|th|s)?%?)"
    r"|(?P<word>[^\W\d_]+(?:['’ʼ][^\W\d_]+)*)"
    r"|(?P<sym>[&%+=@°$£€])",
    re.IGNORECASE)


def _below_1000(n: int) -> List[str]:
    out = []
    if n >= 100:
        out += [_ONES[n // 100], "hundred"]
        n %= 100
    if n >= 20:
        out.append(_TENS[n // 10])
        if n % 10:
            out.append(_ONES[n % 10])
    elif n > 0:
        out.append(_ONES[n])
    return out


def number_words(n: int) -> List[str]:
    """Cardinal number as words ("one hundred twenty three"; no "and")."""
    if n == 0:
        return ["zero"]
    if n >= 10 ** 15:
        return [_ONES[int(d)] for d in str(n)]
    out = []
    for scale, name in _SCALES:
        if n >= scale:
            out += _below_1000(n // scale) + [name]
            n %= scale
    return out + _below_1000(n)


def _year_words(n: int) -> List[str]:
    hi, lo = divmod(n, 100)
    if lo == 0:
        return number_words(hi) + ["hundred"]
    if lo < 10:
        return number_words(hi) + ["oh", _ONES[lo]]
    return number_words(hi) + number_words(lo)


def _ordinal(words: List[str]) -> List[str]:
    last = words[-1]
    if last in _ORD_IRREG:
        last = _ORD_IRREG[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last = last + "th"
    return words[:-1] + [last]


def _plural(words: List[str]) -> List[str]:
    last = words[-1]
    last = last[:-1] + "ies" if last.endswith("y") else (last + "es" if last.endswith("x") else last + "s")
    return words[:-1] + [last]


def _int_words(s: str, allow_year: bool) -> List[str]:
    if len(s) > 1 and s.startswith("0"):
        return [_ONES[int(d)] for d in s]
    n = int(s)
    if allow_year and len(s) == 4 and 1100 <= n <= 2099 and (n % 1000 >= 10 or n < 2000):
        return _year_words(n)
    return number_words(n)


def _num_token_words(tok: str) -> List[str]:
    tok = tok.lower()
    percent = tok.endswith("%")
    if percent:
        tok = tok[:-1]
    suffix = ""
    m = re.match(r"^([\d,.]*\d)(st|nd|rd|th|s)?$", tok)
    if m is None:
        return []
    body, suffix = m.group(1), m.group(2) or ""
    intpart, _, frac = body.partition(".")
    if "," in intpart:
        if re.fullmatch(r"\d{1,3}(,\d{3})+", intpart):
            groups = [intpart.replace(",", "")]
        else:
            groups = [g for g in intpart.split(",") if g]
    else:
        groups = [intpart]
    words: List[str] = []
    for g in groups:
        words += _int_words(g, allow_year=(not frac and "," not in intpart))
    if frac:
        words += ["point"] + [_ONES[int(d)] for d in frac if d.isdigit()]
    if suffix in ("st", "nd", "rd", "th") and not frac:
        words = _ordinal(words)
    elif suffix == "s":
        words = _plural(words)
    if percent:
        words.append("percent")
    return words


def _clean_word(s: str) -> str:
    out = []
    for ch in unicodedata.normalize("NFKD", s.lower()):
        if unicodedata.combining(ch):
            continue
        ch = _LETTER_MAP.get(ch, ch)
        for c in ch:
            if "a" <= c <= "z" or c == "'":
                out.append(c)
    return "".join(out).strip("'")


def _token_pieces(text: str):
    """[(normalised word, src index of each of its chars)] for one text."""
    pieces = []
    for m in _TOKEN_RE.finditer(text):
        kind = m.lastgroup
        tok = m.group(0)
        start = m.start()
        if kind == "word":
            # keep a per-character source index (accents / ligatures map to their source char)
            chars, src = [], []
            for i, ch in enumerate(tok):
                for c in unicodedata.normalize("NFKD", ch.lower()):
                    if unicodedata.combining(c):
                        continue
                    c = _LETTER_MAP.get(c, c)
                    for cc in c:
                        if "a" <= cc <= "z" or cc == "'":
                            chars.append(cc)
                            src.append(start + i)
            # strip apostrophes at the edges, split on doubled ones
            word = "".join(chars)
            lo, hi = 0, len(word)
            while lo < hi and word[lo] == "'":
                lo += 1
            while hi > lo and word[hi - 1] == "'":
                hi -= 1
            if hi > lo:
                pieces.append((word[lo:hi], src[lo:hi]))
        elif kind == "num":
            for w in _num_token_words(tok):
                pieces.append((w, [start] * len(w)))
        elif kind == "money":
            cur = _CURRENCY[tok[0]]
            amount = tok[1:].strip()
            whole, _, cents = amount.partition(".")
            words = _num_token_words(whole) + [cur]
            if len(cents) == 2 and int(cents) > 0:
                words += number_words(int(cents)) + ["pence" if cur == "pounds" else "cents"]
            elif cents and len(cents) != 2:
                words = _num_token_words(amount) + [cur]
            for w in words:
                pieces.append((w, [start] * len(w)))
        else:  # symbol
            w = _SYMBOL_WORDS.get(tok)
            if w:
                pieces.append((w, [start] * len(w)))
    return pieces


def normalize_with_map(text: str) -> Tuple[str, np.ndarray]:
    """(normalised text, src) with src[i] = index in `text` of the character normalised char i came
    from (a space gets the index of the next word's first source character)."""
    out, src = [], []
    for w, s in _token_pieces(text):
        if out:
            out.append(" ")
            src.append(s[0])
        out.extend(w)
        src.extend(s)
    return "".join(out), np.asarray(src, dtype=np.int64)


def normalize_text(text: str) -> str:
    return normalize_with_map(text)[0]


def encode_text(norm: str) -> List[int]:
    """Normalised text -> label ids (characters outside the alphabet are dropped)."""
    return [CHAR_TO_ID[c] for c in norm if c in CHAR_TO_ID]


def decode_ids(ids: Sequence[int]) -> str:
    return "".join(ID_TO_CHAR.get(int(i), "") for i in ids)


# ---------------------------------------------------------------------------
# Edit distance / CER
# ---------------------------------------------------------------------------

def edit_distance(a: Sequence, b: Sequence) -> int:
    """Levenshtein distance (numpy, O(len(a)) vector steps)."""
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    if isinstance(a, str):
        a = np.frombuffer(a.encode("utf-32-le"), dtype=np.uint32)
    if isinstance(b, str):
        b = np.frombuffer(b.encode("utf-32-le"), dtype=np.uint32)
    a = np.asarray(a)
    b = np.asarray(b)
    m = len(b)
    ar = np.arange(m + 1, dtype=np.int64)
    prev = ar.copy()
    for i in range(1, len(a) + 1):
        sub = prev[:-1] + (b != a[i - 1])
        cand = np.empty(m + 1, dtype=np.int64)
        cand[0] = i
        cand[1:] = np.minimum(sub, prev[1:] + 1)
        prev = np.minimum.accumulate(cand - ar) + ar
    return int(prev[-1])


def cer(hyp: str, ref: str, normalize: bool = True) -> float:
    """Character error rate of `hyp` against `ref` after normalisation (edits / reference length).
    An empty reference gives 0.0 for an empty hypothesis and 1.0 otherwise."""
    if normalize:
        hyp, ref = normalize_text(hyp), normalize_text(ref)
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(hyp, ref) / len(ref)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class CB0CTCConfig:
    n_codes: int = N_CODES
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    conv_kernel: int = 7
    upsample: int = 4
    d_up: int = 256
    up_layers: int = 2
    up_kernel: int = 5
    dropout: float = 0.1
    n_out: int = N_OUT
    vad_head: bool = True
    rope_base: float = 10000.0
    boundary_head: bool = False  # word start / word end events at 50 Hz


def _rope(T: int, dim: int, base: float, device, dtype):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    t = torch.arange(T, device=device, dtype=torch.float32)
    f = torch.outer(t, inv)
    return f.cos().to(dtype), f.sin().to(dtype)


def _apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


class _Block(nn.Module):
    def __init__(self, cfg: CB0CTCConfig):
        super().__init__()
        d = cfg.d_model
        self.h = cfg.n_heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.dw = nn.Conv1d(d, d, cfg.conv_kernel, padding=cfg.conv_kernel // 2, groups=d)
        self.pw = nn.Linear(d, d)
        self.ln3 = nn.LayerNorm(d)
        self.ff1 = nn.Linear(d, cfg.d_ff)
        self.ff2 = nn.Linear(cfg.d_ff, d)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x, valid, attn_mask, cos, sin):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k = _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(B, T, D)))
        h = self.ln2(x) * valid
        h = self.pw(F.gelu(self.dw(h.transpose(1, 2)).transpose(1, 2)))
        x = x + self.drop(h)
        x = x + self.drop(self.ff2(self.drop(F.gelu(self.ff1(self.ln3(x))))))
        return x


class _UpBlock(nn.Module):
    def __init__(self, d: int, k: int, p: float):
        super().__init__()
        self.ln = nn.LayerNorm(d)
        self.conv = nn.Conv1d(d, d, k, padding=k // 2)
        self.drop = nn.Dropout(p)

    def forward(self, x, valid):
        h = self.ln(x) * valid
        return x + self.drop(F.gelu(self.conv(h.transpose(1, 2)).transpose(1, 2)))


class CB0CTCModel(nn.Module):
    """codes [B, T] (0..16383, MASK_CODE, PAD_CODE) -> (CTC logits [B, U*T, N_OUT], output lengths,
    VAD logits [B, U*T] or None, word start/end logits [B, U*T, 2] or None)."""

    def __init__(self, cfg: CB0CTCConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.emb = nn.Embedding(cfg.n_codes + 2, d)
        self.in_conv = nn.Conv1d(d, d, 5, padding=2)
        self.in_ln = nn.LayerNorm(d)
        self.blocks = nn.ModuleList([_Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.up = nn.Linear(d, cfg.upsample * cfg.d_up)
        self.up_blocks = nn.ModuleList([_UpBlock(cfg.d_up, cfg.up_kernel, cfg.dropout) for _ in range(cfg.up_layers)])
        self.ln_up = nn.LayerNorm(cfg.d_up)
        self.head = nn.Linear(cfg.d_up, cfg.n_out)
        self.vad = nn.Linear(cfg.d_up, 1) if cfg.vad_head else None
        self.bnd = nn.Linear(cfg.d_up, 2) if cfg.boundary_head else None
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, codes: torch.Tensor, lengths: torch.Tensor):
        B, T = codes.shape
        U = self.cfg.upsample
        ar = torch.arange(T, device=codes.device)
        valid_b = ar[None, :] < lengths[:, None]                       # [B, T]
        x = self.emb(codes)
        valid = valid_b[..., None].to(x.dtype)
        x = x + F.gelu(self.in_conv((x * valid).transpose(1, 2)).transpose(1, 2))
        x = self.in_ln(x)
        hd = self.cfg.d_model // self.cfg.n_heads
        cos, sin = _rope(T, hd, self.cfg.rope_base, codes.device, x.dtype)
        attn_mask = valid_b[:, None, None, :]
        for blk in self.blocks:
            x = blk(x, valid, attn_mask, cos, sin)
        x = self.up(self.ln_f(x)).view(B, T * U, self.cfg.d_up)
        valid_u = valid_b.repeat_interleave(U, dim=1)[..., None].to(x.dtype)
        for blk in self.up_blocks:
            x = blk(x, valid_u)
        x = self.ln_up(x)
        logits = self.head(x)
        vad = self.vad(x).squeeze(-1) if self.vad is not None else None
        bnd = self.bnd(x) if self.bnd is not None else None
        return logits, lengths * U, vad, bnd


# ---------------------------------------------------------------------------
# Batching helpers
# ---------------------------------------------------------------------------

def _pad_codes(codes_list: Sequence[Sequence[int]], device) -> Tuple[torch.Tensor, torch.Tensor]:
    lens = [len(c) for c in codes_list]
    T = max(1, max(lens) if lens else 1)
    arr = np.full((len(codes_list), T), PAD_CODE, dtype=np.int64)
    for i, c in enumerate(codes_list):
        if len(c):
            a = np.asarray(c, dtype=np.int64)
            a = np.where((a >= 0) & (a < N_CODES), a, MASK_CODE)
            arr[i, :len(a)] = a
    return torch.from_numpy(arr).to(device), torch.tensor(lens, dtype=torch.long, device=device)


def greedy_decode(log_probs: torch.Tensor, out_lens: torch.Tensor) -> List[str]:
    best = log_probs.argmax(-1).cpu().numpy()
    lens = out_lens.cpu().numpy()
    texts = []
    for b in range(best.shape[0]):
        p = best[b, :lens[b]]
        if len(p) == 0:
            texts.append("")
            continue
        keep = np.r_[True, p[1:] != p[:-1]] & (p != BLANK)
        t = decode_ids(p[keep])
        texts.append(re.sub(" +", " ", t).strip())
    return texts


@torch.no_grad()
def ctc_viterbi(log_probs: torch.Tensor, out_lens: torch.Tensor, targets: Sequence[Sequence[int]]):
    """Batched CTC forced alignment (Viterbi over the 2L+1 blank-extended states).

    log_probs [B, T, C] (log-softmax), out_lens [B], targets: label id lists.
    Returns per sample: (spans [L, 2] int (first, last+1 output frame of each label's emission),
    path_score float (sum of log-probs on the path), ok bool (False if T is too short for the labels)).
    """
    dev = log_probs.device
    B, T, C = log_probs.shape
    L = max(1, max(len(t) for t in targets))
    S = 2 * L + 1
    tgt = torch.zeros(B, L, dtype=torch.long, device=dev)
    tl = torch.tensor([len(t) for t in targets], dtype=torch.long, device=dev)
    for b, t in enumerate(targets):
        if len(t):
            tgt[b, :len(t)] = torch.as_tensor(list(t), dtype=torch.long, device=dev)
    ext = torch.zeros(B, S, dtype=torch.long, device=dev)
    ext[:, 1::2] = tgt
    s_idx = torch.arange(S, device=dev)
    state_ok = s_idx[None, :] < (2 * tl + 1)[:, None]                 # [B, S]
    # skip transition s-2 -> s allowed for labels that differ from the label two states back
    skip_ok = torch.zeros(B, S, dtype=torch.bool, device=dev)
    skip_ok[:, 3::2] = tgt[:, 1:] != tgt[:, :-1]
    em = log_probs.float().gather(2, ext[:, None, :].expand(B, T, S))   # [B, T, S]
    em = em.masked_fill(~state_ok[:, None, :], -1e30)
    NEG = torch.tensor(-1e30, device=dev)
    ap = torch.full((B, S + 2), -1e30, device=dev)  # alpha with two guard columns (states -2, -1)
    ap[:, 2] = em[:, 0, 0]
    ap[:, 3] = torch.where(tl > 0, em[:, 0, 1], NEG)
    skip_pen = torch.zeros(B, S, device=dev).masked_fill(~skip_ok, -1e30)
    bp = torch.zeros(B, T, S, dtype=torch.uint8, device=dev)
    live_all = torch.arange(T, device=dev)[:, None] < out_lens.to(dev)[None, :]   # [T, B]
    for t in range(1, T):
        a0, a1 = ap[:, 2:], ap[:, 1:-1]
        a2 = ap[:, :-2] + skip_pen
        b01 = torch.maximum(a0, a1)
        arg = torch.where(a2 > b01, 2, (a1 > a0).to(torch.uint8))
        new = torch.maximum(b01, a2) + em[:, t]
        live = live_all[t][:, None]
        ap[:, 2:] = torch.where(live, new, a0)
        bp[:, t] = arg * live
    alpha = ap[:, 2:]
    # final state: last label or trailing blank
    last = 2 * tl
    fin_blank = alpha.gather(1, last[:, None]).squeeze(1)
    fin_lab = alpha.gather(1, (last - 1).clamp(min=0)[:, None]).squeeze(1)
    fin_lab = torch.where(tl > 0, fin_lab, NEG)
    use_lab = fin_lab > fin_blank
    s_end = torch.where(use_lab, last - 1, last)
    score = torch.maximum(fin_blank, fin_lab)
    bp = bp.cpu().numpy()
    s_cur = s_end.cpu().numpy().astype(np.int64)
    Tb = out_lens.cpu().numpy().astype(np.int64)
    path = np.zeros((B, T), dtype=np.int64)
    rows = np.arange(B)
    for t in range(T - 1, -1, -1):
        on = t < Tb
        path[on, t] = s_cur[on]
        step = bp[rows, t, s_cur].astype(np.int64)
        s_cur = np.where(on & (t > 0), s_cur - step, s_cur)
    out = []
    score = score.cpu().numpy()
    tl_np = tl.cpu().numpy()
    for b in range(B):
        n = int(tl_np[b])
        p = path[b, :Tb[b]]
        lab_states = 2 * np.arange(n) + 1
        first = np.searchsorted(p, lab_states, side="left")
        endx = np.searchsorted(p, lab_states, side="right")
        ok = bool(score[b] > -1e29) and Tb[b] > 0
        out.append((np.stack([first, endx], 1) if n else np.zeros((0, 2), np.int64), float(score[b]), ok))
    return out


# ---------------------------------------------------------------------------
# Inference wrapper
# ---------------------------------------------------------------------------

class CB0CTC:
    """Loaded recogniser. Create with `load(path)`."""

    def __init__(self, model: CB0CTCModel, device: str = "cuda", dtype=torch.bfloat16):
        self.model = model.to(device).eval()
        self.device = device
        self.dtype = dtype
        self.U = model.cfg.upsample
        self.out_hz = 12.5 * self.U

    # -- forward -------------------------------------------------------
    @torch.no_grad()
    def _forward(self, codes_list):
        codes, lens = _pad_codes(codes_list, self.device)
        with torch.autocast(device_type="cuda" if "cuda" in str(self.device) else "cpu", dtype=self.dtype,
                            enabled=self.dtype != torch.float32):
            logits, out_lens, vad, bnd = self.model(codes, lens)
        lp = torch.log_softmax(logits.float(), -1)
        return (lp, out_lens, (torch.sigmoid(vad.float()) if vad is not None else None),
                (torch.sigmoid(bnd.float()) if bnd is not None else None))

    @torch.no_grad()
    def forward_fp32(self, codes_list):
        """(CTC log-probs [B, U*T, N_OUT], output lengths, VAD probabilities [B, U*T] or None) in fp32, whatever
        ``self.dtype`` is (no autocast; the weights stay fp32): the quality gate's thresholds were calibrated in
        fp32. Same as ``_forward`` with dtype float32."""
        codes, lens = _pad_codes(codes_list, self.device)
        logits, out_lens, vad, _ = self.model(codes, lens)
        lp = torch.log_softmax(logits.float(), -1)
        return lp, out_lens, (torch.sigmoid(vad.float()) if vad is not None else None)

    def _batches(self, n: int, lens: Sequence[int], batch_size: int):
        order = np.argsort(np.asarray(lens))[::-1]
        for i in range(0, n, batch_size):
            yield order[i:i + batch_size]

    # -- recognition ---------------------------------------------------
    @torch.no_grad()
    def recognize(self, codes_list: Sequence[Sequence[int]], batch_size: int = 128) -> List[str]:
        """Greedy CTC transcripts (normalised alphabet) for a batch of CB0 code lists."""
        out: List[Optional[str]] = [None] * len(codes_list)
        for idx in self._batches(len(codes_list), [len(c) for c in codes_list], batch_size):
            lp, ol, _, _ = self._forward([codes_list[i] for i in idx])
            for i, t in zip(idx, greedy_decode(lp, ol)):
                out[i] = t
        return out


def load(path: str, device: Optional[str] = None, dtype=None) -> CB0CTC:
    """Load a recogniser checkpoint: a dict with ``config``, ``charset`` and the model ``state_dict``."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("charset", CHARSET) != CHARSET:
        raise ValueError("checkpoint charset differs from this module's CHARSET")
    cfg = CB0CTCConfig(**{k: v for k, v in ck["config"].items() if k in CB0CTCConfig.__dataclass_fields__})
    model = CB0CTCModel(cfg)
    sd = ck["state_dict"]
    model.load_state_dict(sd)
    if dtype is None:
        dtype = torch.bfloat16 if "cuda" in str(device) else torch.float32
    return CB0CTC(model, device=device, dtype=dtype)

