"""Markers in the verbalizer's output. The model copies a written number or Roman numeral between an opening marker
token and the closing one; expand() writes the words.

  <|fim_prefix|>27,040,500<|fim_pad|>   number          -> twenty-seven million forty thousand five hundred
  <|fim_middle|>XCIII<|fim_pad|>        Roman numeral   -> Ninety-three
  <|fim_suffix|>VIII<|fim_pad|>         Roman ordinal   -> Eighth

The four tokens are reserved, non-special tokens of the Qwen tokenizer, so decoding with skip_special_tokens=True
keeps them. A number reads in full, decimals digit by digit, with a capital at the start of a sentence. Roman
numerals are canonical (I to MMMCMXCIX, all upper or all lower case) and always get a capital ("Chapter Twelve",
"Henry the Eighth"). Copied text that is neither ("1,23,456", "IIII") is passed on as written.
"""
import re

NUM, ROMAN, ORDINAL, CLOSE = "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>"
MARKER_RE = re.compile(r"(<\|fim_(?:prefix|middle|suffix)\|>)(.*?)<\|fim_pad\|>", re.S)
ANY_MARKER_RE = re.compile(r"<\|fim_(?:prefix|middle|suffix|pad)\|>")

ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
        "seventeen eighteen nineteen").split()
TENS = "zero ten twenty thirty forty fifty sixty seventy eighty ninety".split()
SCALES = ((10 ** 18, "quintillion"), (10 ** 15, "quadrillion"), (10 ** 12, "trillion"), (10 ** 9, "billion"),
          (10 ** 6, "million"), (1000, "thousand"))
ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth",
       "twelve": "twelfth"}
ROMAN_STEPS = ((1000, "M"), (900, "CM"), (500, "D"), (400, "CD"), (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
               (10, "X"), (9, "IX"), (5, "V"), (4, "IV"), (1, "I"))
NUMBER_RE = re.compile(r"(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?")


def cardinal(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("-" + ONES[n % 10] if n % 10 else "")
    if n < 1000:
        return ONES[n // 100] + " hundred" + (" " + cardinal(n % 100) if n % 100 else "")
    for size, name in SCALES:
        if n >= size:
            return cardinal(n // size) + " " + name + (" " + cardinal(n % size) if n % size else "")


def ordinal(n: int) -> str:
    words = cardinal(n)
    head, sep, last = words.rpartition("-") if "-" in words.split(" ")[-1] else words.rpartition(" ")
    return head + sep + (ORD.get(last) or (last[:-1] + "ieth" if last.endswith("y") else last + "th"))


def to_roman(n: int) -> str:
    out = ""
    for value, letters in ROMAN_STEPS:
        while n >= value:
            out, n = out + letters, n - value
    return out


def from_roman(text: str) -> int | None:
    """The value of a canonical Roman numeral in one case (XIV, xiv), else None."""
    if not text or not (text.isupper() or text.islower()) or set(text.upper()) - set("IVXLCDM"):
        return None
    values = dict(I=1, V=5, X=10, L=50, C=100, D=500, M=1000)
    upper, total = text.upper(), 0
    for a, b in zip(upper, upper[1:] + " "):
        total += -values[a] if b in values and values[b] > values[a] else values[a]
    return total if 0 < total < 4000 and to_roman(total) == upper else None


def read_number(text: str) -> str | None:
    """'27,040,500' -> words; decimals digit by digit; None for anything else (leading zeros, bad grouping)."""
    m = NUMBER_RE.fullmatch(text)
    if not m:
        return None
    whole = m.group(1).replace(",", "")
    if (len(whole) > 1 and whole[0] == "0") or len(whole) > 21:
        return None
    return cardinal(int(whole)) + (" point " + " ".join(ONES[int(d)] for d in m.group(2)) if m.group(2) else "")


def _sentence_start(before: str) -> bool:
    return re.search(r"(?:^|[.!?…]|\n)[\"'”’“‘)\]\s(\[]*$", before) is not None


def expand(text: str) -> str | None:
    """The text with every marker replaced by its words. Marked text that cannot be read (another number format,
    a non-canonical numeral) stays as written, without its marker. None if a marker is empty or left open."""
    out, pos = "", 0
    for m in MARKER_RE.finditer(text):
        kind, body = m.group(1), m.group(2)
        if not body.strip():
            return None
        if kind == NUM:
            words = read_number(body)
        else:
            value = from_roman(body)
            words = None if value is None else (cardinal(value) if kind == ROMAN else ordinal(value))
        out += text[pos:m.start()]
        if words is None:
            words = body
        elif kind != NUM or _sentence_start(out):
            words = words[0].upper() + words[1:]
        out, pos = out + words, m.end()
    out += text[pos:]
    return None if ANY_MARKER_RE.search(out) else out


def expand_copied(source: str, output: str) -> str | None:
    """expand(output), or None when a marker holds text that does not stand whole in ``source`` (a digit lost or
    changed in the copy). Cents or an ordinal ending may follow a marked number ("$4,150,355.09", "1,000,000th")."""
    for m in MARKER_RE.finditer(output):
        body = re.escape(m.group(2))
        whole = rf"(?<![\d,]){body}(?!\d|,\d)" if m.group(1) == NUM else rf"(?<![\w,]){body}(?!\w)"
        if not re.search(whole, source):
            return None
    return expand(output)
