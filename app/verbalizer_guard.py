"""Deterministic guard around the verbalizer model (runs only when ENABLE_VERBALIZATION=1).

The verbalizer model misreads some patterns: "4:02 PM" -> "four PM two
minutes", "11:05" -> "eleven point zero five", ".xlsx" -> "dot X", MAX_RETRY_COUNT -> "Maximum Retry Count", a dropped
"A" in "1Z999AA...", and a dropped "OK," in prose. Two steps, no model change:

prepare()       a pre-pass that rewrites those patterns into unambiguous spoken words BEFORE the model sees them:
                  clock times H:MM [am / pm]         -> "four oh two PM", "eleven oh five", "eight AM", "six o'clock"
                  version pre-release suffixes       -> "2.7.0 R C one" (the hyphen is not a minus)
                  known file extensions              -> "dot X L S X", "dot P D F"
                  identifiers with "_" / camelCase   -> their words as written ("max retry count", "get user name")
                  alphanumeric codes                 -> letter by letter, digits as digit words ("one Z nine nine nine A A")
                Each rewrite is recorded as a protected span.
check_output()  a post-check that rejects a model output that drops or changes a protected span or a plain word of
                the input. The caller then uses the pre-pass text for that request (the spans are already spoken).
"""

import re
from dataclasses import dataclass, field

from app.text import _ABBREVIATIONS

_ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen"
).split()
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")


def _cardinal(n: int) -> str:
    """0..99 -> words ("forty-five")."""
    if n < 20:
        return _ONES[n]
    tens, ones = divmod(n, 10)
    return _TENS[tens] + (f"-{_ONES[ones]}" if ones else "")


def _spell(token: str) -> str:
    """Letter by letter, digits as digit words: "1Z9-9A" -> "one Z nine nine A"."""
    return " ".join(_ONES[int(c)] if c.isdigit() else c.upper() for c in token if c.isalnum())


# ── Pre-pass rules ───────────────────────────────────────────────────────────

# H:MM, optional am / pm; not H:MM:SS, not inside a longer token (IP:port, ratios with 1-digit minutes don't match).
_TIME_RE = re.compile(
    r"(?<![\w:./])([01]?\d|2[0-3]):([0-5]\d)(?![\d:]|\.\d)"
    r"(?:\s?([AaPp])(?:\.\s?[Mm]\.|\s?[Mm])(?![A-Za-z]))?(?![A-Za-z_])"
)
_VERSION_SUFFIX_RE = re.compile(
    r"(?<![\w.])(v?\d+(?:\.\d+)+)-(rc|alpha|beta|dev|pre|preview)(\d*)(?![\w-])", re.IGNORECASE
)
# Extensions spoken letter by letter (the model shortens these, ".xlsx" -> "dot X"). Word-like ones (json, log,
# zip) are read fine by the model and are left to it.
_EXTENSIONS = {
    "xlsx": "X L S X", "xls": "X L S", "xlsm": "X L S M", "docx": "D O C X", "pptx": "P P T X", "ppt": "P P T",
    "csv": "C S V", "tsv": "T S V", "pdf": "P D F", "txt": "T X T", "py": "P Y", "js": "J S", "ts": "T S",
    "sh": "S H", "md": "M D", "sql": "S Q L", "xml": "X M L", "html": "H T M L", "yml": "Y M L", "png": "P N G",
    "jpg": "J P G", "jpeg": "J P E G", "svg": "S V G", "mp3": "M P three", "mp4": "M P four", "exe": "E X E",
}
_EXTENSION_RE = re.compile(
    r"(?<=[A-Za-z0-9_)\]])\.(" + "|".join(sorted(_EXTENSIONS, key=len, reverse=True)) + r")(?![\w-]|\.\w)",
    re.IGNORECASE,
)
_SNAKE_RE = re.compile(r"(?<![\w.@/-])[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+(?![\w@/-])")
_CAMEL_RE = re.compile(r"(?<![\w.@/-])[a-z]{2,}(?:[A-Z]+[a-z]*)+(?![\w@/-])")
_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
# Codes: upper-case letters + digits (hyphen-joined chains allowed), or lower-case hex (commit hashes).
_CODE_RE = re.compile(
    r"(?<![\w.@/-])(?=[A-Z0-9-]*[A-Z])(?=[A-Z0-9-]*\d)[A-Z0-9]+(?:-[A-Z0-9]+)*(?![\w@/]|[.-]\w)"
)
_HEX_RE = re.compile(r"(?<![\w.@/-])(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{6,40}(?![\w@/]|[.-]\w)")
_UNIT_LIKE_RE = re.compile(r"\d+[A-Z]{1,3}")   # 100GB, 1080P, 4K: a number with a unit, not a code


@dataclass
class Prepared:
    """Pre-pass result: the text the model sees and the spoken words it must keep."""
    text: str
    spans: list[str] = field(default_factory=list)


def _say_time(m: re.Match) -> str | None:
    hour, minute, ampm = int(m.group(1)), int(m.group(2)), m.group(3)
    if ampm and not 1 <= hour <= 12 and hour != 0:
        return None   # "14:30 PM": contradictory, leave it to the model
    if not ampm and hour == 0 and minute == 0:
        return None
    suffix = f" {ampm.upper()}M" if ampm else ""
    h = _cardinal(12 if ampm and hour == 0 else hour)
    if minute == 0:
        return h + (suffix or (" o'clock" if 1 <= hour <= 12 else " hundred"))
    mm = f"oh {_ONES[minute]}" if minute < 10 else _cardinal(minute)
    return f"{h} {mm}{suffix}"


def _say_version_suffix(m: re.Match) -> tuple[str, str]:
    tag, num = m.group(2).lower(), m.group(3)
    words = "R C" if tag == "rc" else tag
    if num:
        words += " " + (_cardinal(int(num)) if len(num) <= 2 else _spell(num))
    return f"{m.group(1)} {words}", words


def _say_snake(m: re.Match) -> str:
    parts = [p for p in m.group(0).split("_") if p]
    return " ".join(_spell(p) if re.search(r"\d", p) else p.lower() for p in parts)


def _say_camel(m: re.Match) -> str:
    return " ".join(p if p.isupper() else p.lower() for p in _CAMEL_SPLIT_RE.split(m.group(0)))


def _is_code(token: str) -> bool:
    alnum = token.replace("-", "")
    if len(alnum) < 5 or _UNIT_LIKE_RE.fullmatch(alnum):
        return False
    if max(len(r) for r in re.findall(r"[A-Z]+", alnum)) > 3:
        return False   # a word with a number (COVID19), not a code
    if "-" in token:
        return all(re.search(r"[A-Z]", s) and re.search(r"\d", s) for s in token.split("-"))
    return True


def prepare(text: str) -> Prepared:
    """Rewrite the patterns the verbalizer misreads into spoken words; record each rewrite as a protected span.
    The rules' outputs contain no digits, dots or underscores, so no rule re-matches another rule's output."""
    spans: list[str] = []

    def keep(spoken: str | None, span: str | None = None, original: str = "") -> str:
        if spoken is None:
            return original
        spans.append(span if span is not None else spoken)
        return spoken

    def version(m: re.Match) -> str:
        spoken, words = _say_version_suffix(m)
        return keep(spoken, words)

    text = _VERSION_SUFFIX_RE.sub(version, text)
    text = _TIME_RE.sub(lambda m: keep(_say_time(m), original=m.group(0)), text)
    text = _EXTENSION_RE.sub(lambda m: " " + keep("dot " + _EXTENSIONS[m.group(1).lower()]), text)
    text = _SNAKE_RE.sub(lambda m: keep(_say_snake(m)), text)
    text = _CAMEL_RE.sub(lambda m: keep(_say_camel(m)), text)
    text = _CODE_RE.sub(lambda m: keep(_spell(m.group(0))) if _is_code(m.group(0)) else m.group(0), text)
    text = _HEX_RE.sub(lambda m: keep(_spell(m.group(0))), text)
    return Prepared(text, spans)


# ── Post-check ───────────────────────────────────────────────────────────────

_EDGE_OPEN = "\"'([{“‘«"
_EDGE_CLOSE = "\"')]}”’»,;:!?.…"
_PLAIN_WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")
# Undotted short forms the model may expand to a word that does not start with them.
_ALIASES = {"dr": "doctor", "mr": "mister", "mrs": "missus", "vs": "versus", "w": "with"}


def _fuse_letters(tokens: list[str]) -> list[str]:
    """Lower-case the tokens and fuse runs of single capital letters ("P Y" -> "py"), so a spelled run and the
    model's fused form of it ("PY") compare equal; a dropped or changed letter still differs."""
    out: list[str] = []
    run = False
    for tok in tokens:
        letter = len(tok) == 1 and tok.isupper()
        if letter and run:
            out[-1] += tok.lower()
        else:
            out.append(tok.lower())
        run = letter
    return out


def _norm_words(text: str) -> list[str]:
    """Words for comparison: a.m. / p.m. -> am / pm, hyphens split words, other punctuation dropped, spelled capital
    letters fused."""
    text = re.sub(r"(?i)(?<![a-z])([ap])\.\s?m\.?(?![a-z])", r"\1m", text)
    text = text.replace("\u2019", "'").replace("'", "").replace("-", " ")
    return _fuse_letters(re.sub(r"[^A-Za-z0-9 ]+", " ", text).split())


def _classify_tokens(text: str) -> list[tuple[int, int, str | None]]:
    """(start, end, plain word or None) for each whitespace token of ``text``: the plain word (apostrophes removed)
    when the model must keep the token as written, None when the token is the model's to expand."""
    tokens, prev_has_digit, letter_run_skipped = [], False, False
    for m in re.finditer(r"\S+", text):
        tok = m.group(0)
        core = tok.lstrip(_EDGE_OPEN).rstrip(_EDGE_CLOSE)
        dotted = tok.rstrip(_EDGE_CLOSE.replace(".", "")).endswith(".")
        letter = len(core) == 1 and core.isupper()
        skip = (
            prev_has_digit
            or (letter and letter_run_skipped)   # the rest of a skipped capital run ("2.7.0 R C one")
            or not _PLAIN_WORD_RE.fullmatch(core)
            or (dotted and core.lower() in _ABBREVIATIONS)
        )
        tokens.append((m.start(), m.end(), None if skip else core.replace("\u2019", "").replace("'", "")))
        letter_run_skipped = letter and skip
        prev_has_digit = any(c.isdigit() for c in tok)
    return tokens


def _plain_words(text: str) -> list[str]:
    """The words the model must keep: letters-only tokens, minus abbreviations (Dr., e.g.) and unit / am-pm words
    right after a number (those are the model's to expand). Single capitals fuse only across adjacent kept tokens: a
    skipped token between them ("R & D", read "R and D") ends the run."""
    words, run = [], []
    for _, _, word in _classify_tokens(text) + [(0, 0, None)]:
        if word is None:
            words += _fuse_letters(run)
            run = []
        else:
            run.append(word)
    return words


def _lev(a: str, b: str) -> int:
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[len(b)]


def _word_kept(have: str, need: str) -> bool:
    if have == need or _ALIASES.get(need) == have:
        return True
    if len(need) >= 2 and have.startswith(need):   # "OK" -> "okay", "Jan" -> "January"
        return True
    return len(need) >= 3 and _lev(have, need) / len(need) <= 0.34


def _contains_run(needle: list[str], hay: list[str]) -> int:
    n = len(needle)
    return sum(hay[i:i + n] == needle for i in range(len(hay) - n + 1)) if n else 0


def check_output(prepared: Prepared, output: str) -> tuple[bool, str]:
    """Accept the model's output only if every protected span is still there word for word and every plain word of
    the input is kept in order. -> (ok, reason)."""
    out_words = _norm_words(output)
    if not out_words:
        return False, "empty output"
    span_counts: dict[str, int] = {}
    for span in prepared.spans:
        span_counts[span] = span_counts.get(span, 0) + 1
    for span, count in span_counts.items():
        if _contains_run(_norm_words(span), out_words) < count:
            return False, f"protected span {span!r} dropped or changed"
    have = [w for w in out_words if not w.isdigit()]
    j = 0
    for need in _plain_words(prepared.text):
        while j < len(have) and not _word_kept(have[j], need):
            j += 1
        if j >= len(have):
            return False, f"word {need!r} dropped or changed"
        j += 1
    return True, "ok"


# ── Growth guard: added words and runaway length ────────────────────────────
#
# check_output() keeps every plain word in order but lets the model ADD words anywhere. On mid-sentence fragments
# (prefix continuations) the model completed the fragment ("smaller plan, then ..." -> "A smaller plan, then ...")
# and once returned its own system instructions.
# check_growth() aligns the output's words to the input: every plain word must map to one output word, in order, and
# every other output word must be explained by the expandable tokens (numbers, symbols, abbreviations, codes) that
# sit in the same gap between two plain words. Plain words never grow. Before the first and after the last plain
# word no unexplained word is allowed (that is where a fragment gets completed); INTERIOR_SLACK unexplained words are
# tolerated between plain words in the whole text. A char cap backs this up.

INTERIOR_SLACK = 1
_GROWTH_CHAR_FACTOR, _GROWTH_CHARS_PER_DIGIT, _GROWTH_CHAR_CONST = 3, 12, 40
_SYMBOL_CHARS = frozenset("$€£¥₹%&@#+=*/\\<>~^°±×÷§©®™|")
# A word as _norm_words sees it, with its span: a.m. / p.m., else a run of letters, digits and apostrophes.
_WORD_RE = re.compile(r"(?i:(?<![a-z])[ap]\.\s?m\.?(?![a-z]))|[A-Za-z0-9'’]*[A-Za-z0-9][A-Za-z0-9'’]*")


def _words(text: str) -> list[tuple[str, int, int]]:
    """(word, start, end) for each comparison word of ``text``, normalised like _norm_words: lower case, apostrophes
    dropped, hyphens and other punctuation split words, a.m. -> am, runs of single capital letters fused."""
    out: list[tuple[str, int, int]] = []
    run = False
    for m in _WORD_RE.finditer(text):
        tok = m.group(0)
        word = tok[0] + "m" if "." in tok else tok.replace("'", "").replace("’", "")
        letter = len(word) == 1 and word.isupper()
        if letter and run:
            prev, start, _ = out[-1]
            out[-1] = (prev + word.lower(), start, m.end())
        else:
            out.append((word.lower(), m.start(), m.end()))
        run = letter
    return out


@dataclass
class _Units:
    """The input as the alignment sees it: plain words (one output word each) and the gaps between them."""
    words: list[str]                        # plain words, normalised
    spans: list[tuple[int, int]]            # their character spans in the input
    gaps: list[list[tuple[int, int]]]       # gaps[k]: (token start, budget) of the expandable tokens before words[k];
                                            # gaps[len(words)]: after the last plain word

    def budget(self, k: int) -> int:
        return sum(b for _, b in self.gaps[k])


def _token_budget(tok: str, n_words: int) -> int:
    """How many output words an expandable token may become: its own words + 3 per digit / symbol + 3."""
    digits = sum(c.isdigit() for c in tok)
    symbols = sum(c in _SYMBOL_CHARS for c in tok)
    return max(1, n_words) + 3 * digits + 3 * symbols + 3


def _units(text: str) -> _Units:
    tokens = _classify_tokens(text)
    words = _words(text)
    starts = [s for s, _, _ in tokens]

    def token_index(pos: int) -> int:
        lo, hi = 0, len(starts) - 1
        while lo < hi:   # last token starting at or before pos
            mid = (lo + hi + 1) // 2
            if starts[mid] <= pos:
                lo = mid
            else:
                hi = mid - 1
        return lo

    # A word is plain when every token it touches is a plain token; it then stands for those tokens.
    plain_by_first: dict[int, tuple[str, int, int, int]] = {}
    covered: set[int] = set()
    words_per_token = [0] * len(tokens)
    for word, s, e in words:
        first, last = token_index(s), token_index(e - 1)
        words_per_token[first] += 1
        if all(tokens[i][2] is not None for i in range(first, last + 1)):
            plain_by_first[first] = (word, s, e, last)
            covered.update(range(first, last + 1))
    units = _Units([], [], [[]])
    for i, (s, e, plain) in enumerate(tokens):
        if i in plain_by_first:
            word, ws, we, _ = plain_by_first[i]
            units.words.append(word)
            units.spans.append((ws, we))
            units.gaps.append([])
        elif i not in covered:
            budget = 1 if plain is not None else _token_budget(text[s:e], words_per_token[i])
            units.gaps[-1].append((s, budget))
    return units


def _align(units: _Units, out: list[str], lead_free: bool = False, slack: int = INTERIOR_SLACK) -> list[set[int]] | None:
    """For each plain word, the output positions it takes in SOME valid alignment (None: no valid alignment).

    Valid: plain words map in order to output words that keep them (_word_kept); the output words between two plain
    words number at most that gap's budget, with ``slack`` extra words in total over the interior gaps; before the
    first plain word at most gaps[0]'s budget (any number when ``lead_free``) and after the last at most the final
    gap's budget."""
    n, m = len(units.words), len(out)
    budgets = [units.budget(k) for k in range(n + 1)]
    if n == 0:
        return [] if lead_free or m <= budgets[0] else None
    kept_cache: dict[tuple[int, int], bool] = {}

    def kept(j: int, i: int) -> bool:
        key = (j, i)
        if key not in kept_cache:
            kept_cache[key] = _word_kept(out[j], units.words[i])
        return kept_cache[key]

    # Forward: fwd[i][j] = least slack used to place words 0..i with word i at output position j.
    fwd: list[dict[int, int]] = []
    for i in range(n):
        cur: dict[int, int] = {}
        if i == 0:
            for j in range(m if lead_free else min(m, budgets[0] + 1)):
                if kept(j, 0):
                    cur[j] = 0
        else:
            for j, used in fwd[-1].items():
                for jj in range(j + 1, min(m, j + 2 + budgets[i] + slack - used)):
                    u = used + max(0, jj - j - 1 - budgets[i])
                    if kept(jj, i) and u < cur.get(jj, slack + 1):
                        cur[jj] = u
        if not cur:
            return None
        fwd.append(cur)
    # Backward: bwd[j] = least slack the words after i need with word i at j (only over forward-reachable j).
    bwd = {j: 0 for j in fwd[-1] if m - 1 - j <= budgets[n]}
    feasible: list[set[int]] = [set() for _ in range(n)]
    feasible[-1] = {j for j, used in fwd[-1].items() if j in bwd and used <= slack}
    for i in range(n - 2, -1, -1):
        nxt, bwd = bwd, {}
        for j in fwd[i]:
            best = slack + 1
            for jj, rest in nxt.items():
                if jj > j:
                    best = min(best, rest + max(0, jj - j - 1 - budgets[i + 1]))
            if best <= slack:
                bwd[j] = best
        feasible[i] = {j for j, used in fwd[i].items() if j in bwd and used + bwd[j] <= slack}
    if not feasible[0]:
        return None
    return feasible


def growth_cap(prepared: Prepared) -> int:
    """The longest output (chars) accepted for this input."""
    text = prepared.text
    return (_GROWTH_CHAR_FACTOR * len(text) + _GROWTH_CHARS_PER_DIGIT * sum(c.isdigit() for c in text)
            + _GROWTH_CHAR_CONST)


def check_growth(prepared: Prepared, output: str) -> tuple[bool, str]:
    """Reject an output that is far longer than its input or holds words the input does not explain (an added word,
    a completed fragment, the model's own instructions). -> (ok, reason)."""
    cap = growth_cap(prepared)
    if len(output) > cap:
        return False, f"output too long ({len(output)} > {cap} chars)"
    if _align(_units(prepared.text), [w for w, _, _ in _words(output)]) is None:
        return False, "output words not explained by the input (added, dropped or reordered words)"
    return True, "ok"


# ── Continuation split (prefix context) ──────────────────────────────────────

def split_continuation(context: str, rest: str, output: str) -> str | None:
    """The part of ``output`` (the model's reading of ``context + " " + rest``) that reads ``rest``, or None when the
    split is not certain: the alignment fails, the boundary word's output position is not unique, or expandable
    tokens on both sides of the boundary share one gap. ``context`` and ``rest`` are pre-pass texts. Only the part's
    start is fixed here; the caller checks the part against ``rest`` (check_output / check_growth)."""
    boundary = len(context) + 1
    units = _units(f"{context} {rest}")
    out = _words(output)
    feasible = _align(units, [w for w, _, _ in out], lead_free=True)
    if feasible is None:
        return None
    if any(s < boundary - 1 < e for s, e in units.spans):
        return None   # a fused word straddles the boundary
    b = sum(1 for s, _ in units.spans if s < boundary)   # plain words in the context
    ctx_side = sum(budget for start, budget in units.gaps[b] if start < boundary)
    rest_side = sum(budget for start, budget in units.gaps[b] if start >= boundary)
    lead = re.match(r"[^A-Za-z0-9\s]*", rest).group(0)   # the rest's own leading punctuation ('"', '$', '(')
    if ctx_side == 0:
        if b == 0 or len(feasible[b - 1]) != 1:
            return None
        part = output[out[next(iter(feasible[b - 1]))][2]:]   # after the context's last plain word
        part = part.lstrip(" \t\n,;:.!?…—–-") if lead else re.sub(r"^[^A-Za-z0-9]+", "", part)
    elif rest_side == 0:
        if b == len(units.words) or len(feasible[b]) != 1:
            return None
        start = out[next(iter(feasible[b]))][1]   # at the rest's first plain word
        while lead and start > 0 and output[start - 1] in lead:
            start -= 1
        part = output[start:]
    else:
        return None
    return part.strip() or None
