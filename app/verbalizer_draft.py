"""A cheap guess of the verbalizer's output, used only as a draft for block verification.

The verbalizer model checks every drafted token (app/lite/gpu/block.py), so the guess does not have to
be right: where it is, many tokens are confirmed in one pass; where it is not, the model's own token is
taken and the draft is re-aligned. The output is the model's, never the draft's.

Normalizer writes the guess from rules for numbers, dates, units, citation markers, tables and reference
lines, and from two tables of fixed readings (short spans such as "e.g." and words respelled such as
"NVIDIA"). DraftFollower keeps the position in that guess while the model writes.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from app.verbalizer_markers import CLOSE, NUM, ONES, cardinal, ordinal

_TABLES = json.loads(Path(__file__).with_name("verbalizer_draft_tables.json").read_text(encoding="utf-8"))

UNITS = {"ms": "milliseconds", "s": "seconds", "Hz": "hertz", "kHz": "kilohertz", "GiB": "gibibytes", "MiB": "mebibytes",
         "GB": "gigabytes", "MB": "megabytes", "KB": "kilobytes", "km": "kilometers", "kg": "kilograms", "m": "meters",
         "cm": "centimeters", "mm": "millimeters", "h": "hours", "min": "minutes"}
MONTHS = "January February March April May June July August September October November December".split()
MONTH_ABBR = {m[:3]: m for m in MONTHS} | {"Sept": "September"}
SMALL = {"a", "an", "the", "and", "or", "of", "for", "in", "on", "to", "with"}


def pairs(n: int) -> str:
    """A year or model number read in pairs: 2026 -> twenty twenty-six, 5090 -> fifty ninety."""
    hi, lo = divmod(n, 100)
    if lo == 0:
        return cardinal(hi) + " hundred" if hi % 10 else cardinal(n)
    if 2000 <= n <= 2009:
        return "two thousand " + ONES[lo]
    return cardinal(hi) + " " + ("oh " + ONES[lo] if lo < 10 else cardinal(lo))


_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d+"
# A marked number is kept out of reach of the other rules until the draft is done.
_HIDE = str.maketrans("0123456789,.", "".join(map(chr, range(0xE000, 0xE00C))))
_SHOW = {value: chr(key) for key, value in _HIDE.items()} | {0xE00C: NUM, 0xE00D: CLOSE}


def number(text: str, before: str = "") -> str:
    whole = text.split(".", 1)[0].replace(",", "")
    if len(whole) >= 6 and whole[0] != "0":
        return "\ue00c" + text.translate(_HIDE) + "\ue00d"   # the model copies long numbers into a marker
    if "." in text:
        whole, frac = text.split(".", 1)
        return cardinal(int(whole.replace(",", ""))) + " point " + " ".join(ONES[int(d)] for d in frac)
    n = int(text.replace(",", ""))
    if "," not in text and len(text) == 4 and (1100 <= n <= 2099 or re.search(r"[A-Z]{2,} $", before)):
        return pairs(n)
    return cardinal(n) if n < 10 ** 15 else text


def _num(m: re.Match, group: int = 0) -> str:
    return number(m.group(group), m.string[max(0, m.start(group) - 12):m.start(group)])


def _iso(m: re.Match) -> str:
    y, mo, d, h, mi, zone = m.groups()
    if not 1 <= int(mo) <= 12:
        return m.group(0)
    date = f"{MONTHS[int(mo) - 1]} {ordinal(int(d))}, {pairs(int(y))}"
    if h is None:
        return date
    clock = cardinal(int(h)) + " " + ("hundred" if mi == "00" else "oh " + ONES[int(mi)] if int(mi) < 10 else cardinal(int(mi)))
    return f"{date}, at {clock}" + (" UTC" if zone else "")


def _table(m: re.Match) -> str:
    """Rows of a table with a header row: every cell is read with its header."""
    cells = [[c.strip() for c in re.split(r"\t|\|", line.strip().strip("|"))] for line in m.group(0).rstrip("\n").split("\n")]
    cells = [row for row in cells if not all(re.fullmatch(r"[-: ]*", c) for c in row)]
    head = [h.rstrip("*") for h in cells[0]] if cells else []
    if len(cells) < 2 or any(re.search(r"\d", h) for h in head) or any(len(row) != len(head) for row in cells):
        return m.group(0)
    out = []
    for row in cells[1:]:
        vals = [plain(c) for c in row]
        out.append(f"{head[0]} {vals[0][:1].upper() + vals[0][1:]}: " + ", ".join(f"{h} {v}" for h, v in zip(head[1:], vals[1:])) + ".")
    return "\n".join(out) + ("\n" if m.group(0).endswith("\n") else "")


def _reference(m: re.Match) -> str:
    body = re.sub(r" [—·|-] ", ", ", m.group(2))
    cut = re.search(r"\s*(?:\.\.\.|…)\s*$", body)
    return f"Reference {cardinal(int(m.group(1)))}, " + (body[:cut.start()] if cut else body.rstrip(".") + ".")


def _heading(m: re.Match) -> str:
    line = m.group(1)
    if not (line.isupper() or m.string[m.end():m.end() + 2] in ("\n\n", "")):
        return m.group(0)
    if line.isupper():
        words = line.split(" ")
        line = " ".join(w.lower() if i and w.lower().strip(":") in SMALL else w[:1] + w[1:].lower() for i, w in enumerate(words))
    return line + ("" if line[-1] in ".:!?" else ".")


_STRUCTURE = [
    (re.compile(r"^(?:[^\n\t]*\t[^\n]*\n?){2,}|^(?:\|[^\n]*\|[ \t]*\n?){2,}", re.M), _table),
    (re.compile(r"^\[(\d+)\] (.+)$", re.M), _reference),
    (re.compile(r"^([A-Z][^\n.!?]{2,78}[^\n.!?,;])(?=\n\n|\n[A-Z]|$)", re.M), _heading),
    (re.compile(r"^(?=[^\n]* \| )[^\n]+$", re.M), lambda m: m.group(0).replace(" | ", ", ").rstrip(".") + "."),
]
_TLD = lambda m: " dot " + (m.group(1).upper() if len(m.group(1)) in (2, 3) and m.group(1) not in ("co", "me", "com", "org", "net", "gov", "dev", "app") else m.group(1))
_UNIT = "|".join(sorted(UNITS, key=len, reverse=True))
_RULES = [
    (re.compile(r" ?(?:\[\d+\])+(?!\w)"), ""),                                             # citation markers
    (re.compile(r"(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{2}):(\d{2})(?::\d{2}(?:\.\d+)?)?( ?UTC|Z|\+00:00)?)?"), _iso),
    (re.compile(rf"\b({'|'.join(MONTH_ABBR)})\.? (\d{{1,2}}), (\d{{4}})"),
     lambda m: f"{MONTH_ABBR[m.group(1)]} {ordinal(int(m.group(2)))}, {pairs(int(m.group(3)))}"),
    (re.compile("[\U0001F300-\U0001FAFF☀-➿ℹ]\ufe0f? ?"), ""),                               # decorative symbols
    (re.compile("\u2011"), "-"),                                                            # non-breaking hyphen
    (re.compile(r"\((e?s)\)"), r"\1"),                                                      # iteration(s)
    (re.compile(r"\s*—\s*"), ", "),
    (re.compile(r"\b(\d+)(?:st|nd|rd|th)\b"), lambda m: ordinal(int(m.group(1)))),
    (re.compile(r"(?<=[a-z])_(?=[a-z])"), " "),                                             # snake_case
    (re.compile(r"(?<=[A-Za-z])\.([a-z]{2,12})\b(?![a-z]*\.[A-Z])"), _TLD),                 # domains
    (re.compile(r"\.([A-Z]{2,5})\b"), r" \1"),                                              # .PDF, .TXT
    (re.compile(r"\bv(?=\d+\.\d)"), "version "),
    (re.compile(r"([a-z])([A-Z])(?=\d)"), r"\1 \2"),                                        # TontaubeV1
    (re.compile(r"(?<=[A-Za-z]\d)-(?=\d)"), ", "),                                          # Qwen3-1.7B
    (re.compile(r"(?<=\d)–(?=\d)|(?<=\d)–(?=[A-Za-z]+\d)"), " to "),                        # ranges
    (re.compile(r"(?<=\d) / (?=\d)"), " and "),
    (re.compile(rf"\$({_NUM})"), lambda m: _num(m, 1) + " dollars"),
    (re.compile(rf"({_NUM})\s?%"), lambda m: _num(m, 1) + " percent"),
    (re.compile(rf"({_NUM})\s?×"), lambda m: _num(m, 1) + " times"),
    (re.compile(rf"({_NUM})[ -]?({_UNIT})\b"), lambda m: _num(m, 1) + " " + UNITS[m.group(2)]),
    (re.compile(rf"(?<=[A-Za-z])-({_NUM})"), lambda m: " " + _num(m, 1)),                   # sub-200
    (re.compile(rf"(?<=[A-Za-z])({_NUM})"), lambda m: " " + _num(m, 1)),                    # CB0, S2, qwen3.8
    (re.compile(rf"({_NUM})(?=[A-Za-z])"), lambda m: _num(m, 1) + " "),                     # 1.7B
    (re.compile(r"(?<=[a-z])/(?=[A-Za-z])"), " slash "),
    (re.compile(r" & "), " and "),
    (re.compile(r" ?= ?"), " equals "),
    (re.compile(_NUM), _num),
]
_CAMEL = re.compile(r"[A-Z][a-z]+(?:[A-Z][a-z]+)+")
_WORDS = re.compile(r"(?<![A-Za-z0-9])[A-Za-z][A-Za-z0-9]*(?![A-Za-z0-9])")


def plain(text: str) -> str:
    for pattern, repl in _RULES:
        text = pattern.sub(repl, text)
    return text


def _key(text: str) -> str:
    """Letters and number words only: two spellings of the same name give the same key."""
    return re.sub(r"[^a-z]", "", plain(text).lower())


class Normalizer:
    """Rules, the two tables, and the respellings the model has made earlier in the same text."""

    def __init__(self):
        self.words: dict[str, str] = dict(_TABLES["spellings"])
        self.phrases: dict[str, str] = _TABLES["phrases"]

    def _phrases(self, x: str) -> str:
        """Replace short spans that always read the same way (Dr., e.g., vs.)."""
        out, last, words = [], 0, list(re.finditer(r"\S+", x))
        i = 0
        while i < len(words):
            for n in (2, 1):
                if i + n > len(words):
                    continue
                start, end = words[i].start(), words[i + n - 1].end()
                span = x[start:end]
                core = span.lstrip("\"'(")
                lead = len(span) - len(core)
                core = core.rstrip(".,;:!?\"')")
                hit = self.phrases.get(core) or (self.phrases.get(span[lead:]) and span[lead:])
                if hit and hit in self.phrases:
                    core, hit = hit, self.phrases[hit]
                if hit and (n == 1 or " " in core):
                    out.append(x[last:start + lead] + "\x00" + hit + "\x01")
                    last = start + lead + len(core)
                    i += n - 1
                    break
            i += 1
        return "".join(out) + x[last:]

    def _word(self, m: re.Match) -> str:
        w = m.group(0)
        known = self.words.get(w)
        if known is not None:
            return "\x00" + known + "\x01"
        return w.capitalize() if _CAMEL.fullmatch(w) else w

    def draft(self, x: str) -> str:
        for pattern, repl in _STRUCTURE:
            x = pattern.sub(repl, x)
        parts = re.split("(\x00[^\x01]*\x01)", self._phrases(x))   # replaced spans are not touched again
        parts = [p if p.startswith("\x00") else _WORDS.sub(self._word, p) for p in parts]
        parts = re.split("(\x00[^\x01]*\x01)", "".join(parts))
        return "".join(part[1:-1] if part.startswith("\x00") else plain(part) for part in parts).translate(_SHOW)

    def learn(self, x: str, y: str) -> None:
        """Remember how the model spelled the words of this input that it changed."""
        spoken = {}
        ys = y.split()
        for n in (1, 2, 3):
            for i in range(len(ys) - n + 1):
                phrase = " ".join(ys[i:i + n]).strip(".,;:!?()\"'")
                spoken.setdefault(_key(phrase), phrase)
        for w in set(_WORDS.findall(x)):
            if w.islower() or (w[0].isupper() and w[1:].islower()):
                continue
            form = spoken.get(_key(w))
            if form and form != w:
                self.words[w] = form


class DraftFollower:
    """Follow a draft of the output text. The position in the draft is kept in characters, so a rewrite
    that changes how the following text splits into tokens does not lose the place.

    Every token the model commits moves the position: if the draft continues with that text, past it;
    if the text occurs a little further on, the model skipped or respelled something and the position
    jumps there; otherwise the model is inserting text the draft does not have and the position stays."""

    def __init__(self, draft: str, encode, decode, chars_per_token: float = 5.0, look: int = 40):
        """draft: the guessed output text; encode: text -> token ids; decode: one token id -> its text."""
        self.dt, self.low = draft, draft.lower()
        self.encode, self.decode, self.cpt, self.look = encode, decode, chars_per_token, look
        self.c, self.n_out = 0, 0

    def propose(self, room: int) -> list[int]:
        rest = self.dt[self.c:self.c + int(room * self.cpt) + 24]
        if not rest:
            return []
        ids = self.encode(rest)
        # The last token of a cut window may be half a word: drop it unless the window reached the end.
        return ids[:room] if self.c + len(rest) >= len(self.dt) else ids[:-1][:room]

    def advance(self, out: list[int]) -> None:
        """Move the position past the tokens the model has committed since the last call."""
        dt, low = self.dt, self.low
        for token in out[self.n_out:]:
            t = self.decode(token)
            if dt.startswith(t, self.c):
                self.c += len(t)
                continue
            key = t.strip().lower()
            if not key or not (t[0].isspace() or not t[0].isalnum()):
                continue                       # a piece inside a word that the draft does not continue with: stay
            reach = self.look if len(key) > 1 else 3
            q = low.find(key, self.c, self.c + reach + len(key))
            while q > 0 and key[0].isalnum() and low[q - 1].isalnum():   # only at the start of a word
                q = low.find(key, q + 1, self.c + reach + len(key))
            if q >= 0:
                self.c = q + len(key)
        self.n_out = len(out)
