"""Characters the TontaubeV1 codebook models read, and the embedding rows they need.

Text reaches the codebook models one character at a time: each character is
the first token of its tokenizer encoding (the models' input scheme). Only the
structural tokens and the BPE-encoded system prompt "<language> : <style>"
use other rows. So out of ~151k text rows a few hundred are ever read, and
input characters the models never saw in training would only hit untrained
(or shared byte-fallback) rows.

ALLOWED_CHARS is the character set for the seven supported languages: the
characters common in the models' text input, all letters of the seven languages
in both cases, digits, the Spanish/Portuguese ordinal indicators, common
fractions and symbols.

clean_text() keeps those characters and rewrites the rest, first match wins:
typographic variants with a trained equivalent (CHAR_MAP), the Unicode
compatibility form (ligatures, fullwidth forms, superscript letters as in
"2ᵉ"), the letter without its accents (foreign names: ř -> r, å -> a), a few
letters without a decomposition (ø -> o, ł -> l, þ -> th); everything else
(other scripts, emoji, superscript digits, which in books are mostly
footnote markers) is dropped.
text_token_rows() lists the embedding rows the models then need.
"""
from __future__ import annotations

import functools
import re
import unicodedata

_LETTERS = (
    "abcdefghijklmnopqrstuvwxyz"
    "äöüß"  # German
    "àâæçéèêëîïôœùûÿ"  # French
    "áíñóú"  # Spanish (with é, ü)
    "ìò"  # Italian (with à, è, é, ù, í, î, ó)
    "ãõ"  # Portuguese (with á, â, à, ç, é, ê, í, ó, ô, ú)
)
ALLOWED_CHARS: frozenset[str] = frozenset(
    set(_LETTERS)
    | {c.upper() for c in _LETTERS if c != "ß"}
    | set("0123456789")
    | set(" \n")
    | set("!\"#$%&'()*+,-./:;=?@[]_{}")
    | set("¡£§«°»¿–—‘’‚“”„…‹›€")
    | set("ºª½¼¾")  # ordinal indicators (1.º, 2.ª), fractions
)

# Typographic variants with a trained equivalent ("" drops invisible characters).
CHAR_MAP: dict[str, str] = {
    **{c: " " for c in (
        "\t", "\N{NO-BREAK SPACE}", "\N{EN SPACE}", "\N{EM SPACE}", "\N{THREE-PER-EM SPACE}",
        "\N{FOUR-PER-EM SPACE}", "\N{SIX-PER-EM SPACE}", "\N{FIGURE SPACE}", "\N{PUNCTUATION SPACE}",
        "\N{THIN SPACE}", "\N{HAIR SPACE}", "\N{NARROW NO-BREAK SPACE}", "\N{MEDIUM MATHEMATICAL SPACE}",
        "\N{IDEOGRAPHIC SPACE}",
    )},
    **{c: "\n" for c in (
        "\r", "\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\N{LINE SEPARATOR}", "\N{PARAGRAPH SEPARATOR}",
    )},
    **{c: "" for c in (
        "\N{SOFT HYPHEN}", "\N{ZERO WIDTH SPACE}", "\N{ZERO WIDTH NON-JOINER}", "\N{ZERO WIDTH JOINER}",
        "\N{WORD JOINER}", "\N{ZERO WIDTH NO-BREAK SPACE}",
    )},
    **{c: "-" for c in ("\N{HYPHEN}", "\N{NON-BREAKING HYPHEN}", "\N{FIGURE DASH}", "\N{MINUS SIGN}")},
    "\N{HORIZONTAL BAR}": "\N{EM DASH}",
    "\N{PRIME}": "'", "\N{ACUTE ACCENT}": "'", "`": "'",
    "\N{SINGLE HIGH-REVERSED-9 QUOTATION MARK}": "\N{LEFT SINGLE QUOTATION MARK}",
    "\N{DOUBLE PRIME}": '"',
    "\N{DOUBLE HIGH-REVERSED-9 QUOTATION MARK}": "\N{LEFT DOUBLE QUOTATION MARK}",
    "\N{LATIN SMALL LIGATURE FF}": "ff", "\N{LATIN SMALL LIGATURE FI}": "fi", "\N{LATIN SMALL LIGATURE FL}": "fl",
    "\N{LATIN SMALL LIGATURE FFI}": "ffi", "\N{LATIN SMALL LIGATURE FFL}": "ffl",
    "\N{LATIN CAPITAL LETTER SHARP S}": "SS",
    "\N{FRACTION SLASH}": "/",
}

# Letters without a Unicode decomposition, as their usual Latin transcription.
_TRANSLITERATION: dict[str, str] = {
    "\N{LATIN SMALL LETTER O WITH STROKE}": "o", "\N{LATIN CAPITAL LETTER O WITH STROKE}": "O",
    "\N{LATIN SMALL LETTER L WITH STROKE}": "l", "\N{LATIN CAPITAL LETTER L WITH STROKE}": "L",
    "\N{LATIN SMALL LETTER D WITH STROKE}": "d", "\N{LATIN CAPITAL LETTER D WITH STROKE}": "D",
    "\N{LATIN SMALL LETTER ETH}": "d", "\N{LATIN CAPITAL LETTER ETH}": "D",
    "\N{LATIN SMALL LETTER THORN}": "th", "\N{LATIN CAPITAL LETTER THORN}": "Th",
    "\N{LATIN SMALL LETTER DOTLESS I}": "i",
    "\N{LATIN SMALL LETTER H WITH STROKE}": "h", "\N{LATIN CAPITAL LETTER H WITH STROKE}": "H",
    "\N{LATIN SMALL LETTER T WITH STROKE}": "t", "\N{LATIN CAPITAL LETTER T WITH STROKE}": "T",
    "\N{LATIN SMALL LETTER ENG}": "ng", "\N{LATIN CAPITAL LETTER ENG}": "Ng",
    "\N{LATIN SMALL LETTER LONG S}": "s",
}

# Markup the models were not trained on in user text: control tokens, <<tags>>, "||".
_MARKUP_RE = re.compile(r"<\|[^|]*\|>|<<[^>]*>>|\|\|")

STRUCTURAL_TOKENS: tuple[str, ...] = (
    "<|im_start|>", "<|im_end|>", "<|start_of_speech|>", "<|end_of_speech|>",
    "<|text_split|>", "<|audio_split|>", "<|endoftext|>",
)
# Tokens the server renders into CB0 windows of v1.1 bundles: slider controls after the
# style label (app/controls.py) and the end-token windows <|utt_end|> / <|utt_cont|>
# (app/config.py UTT_END). User text never carries them: the sanitizer strips every <|...|>.
CONTROL_TOKEN_RE = re.compile(r"<\|(?:ctrl_[a-z_]+_\d{2}|utt_end|utt_cont)\|>")


def is_model_token(token: str) -> bool:
    """Whether tokenize_text maps a <|...|> string to its id (any other is dropped)."""
    return token in STRUCTURAL_TOKENS or CONTROL_TOKEN_RE.fullmatch(token) is not None


def _readable(candidate: str) -> str | None:
    """candidate with CHAR_MAP applied if that leaves only ALLOWED_CHARS, else None."""
    mapped = "".join(CHAR_MAP.get(c, c) for c in candidate)
    return mapped if mapped and all(c in ALLOWED_CHARS for c in mapped) else None


@functools.lru_cache(maxsize=8192)
def clean_char(ch: str) -> str:
    """What a single (NFC) character becomes in model input; "" when it is dropped."""
    if ch in ALLOWED_CHARS:
        return ch
    if ch in CHAR_MAP:
        return CHAR_MAP[ch]
    decomposition = unicodedata.decomposition(ch)
    if decomposition.startswith("<super>") and unicodedata.normalize("NFKC", ch).isdigit():
        return ""  # superscript digits are mostly footnote markers in books (also m²)
    stripped = "".join(c for c in unicodedata.normalize("NFD", ch) if not unicodedata.combining(c))
    for candidate in (unicodedata.normalize("NFKC", ch), stripped, _TRANSLITERATION.get(ch, "")):
        if candidate != ch and (readable := _readable(candidate)) is not None:
            return readable
    return ""


def clean_text(text: str) -> str:
    """Text as the models read it: NFC, then clean_char() for every character."""
    return "".join(map(clean_char, unicodedata.normalize("NFC", text)))


def strip_markup(text: str) -> str:
    """Remove control tokens, <<tags>> and "||" from user text."""
    return _MARKUP_RE.sub("", text)


# Scene breaks and rules ("-----", "* * *", "=====", "___") carry no words. Read one character at a time they are
# out-of-distribution input, and the CB0 chars/s window counts them as text to be spoken, so a long rule makes it
# reject good attempts as too fast.
_SEPARATOR_LINE_RE = re.compile(r"^[ \t]*(?:[-_*=~#][ \t]*){3,}$", re.M)
_DASH_RUN_RE = re.compile(r"-{3,}")
_SYMBOL_RUN_RE = re.compile(r"(?:[_*=~#][ \t]*){3,}")


def strip_separators(text: str) -> str:
    """Separator lines become paragraph breaks; inside a line, 3+ hyphens become an em dash ("well---no") and other
    runs of 3+ symbols a space. "--" and "..." are unchanged."""
    text = _SEPARATOR_LINE_RE.sub("", text)
    text = _DASH_RUN_RE.sub("\N{EM DASH}", text)
    return _SYMBOL_RUN_RE.sub(" ", text)


def char_token_id(tokenizer, ch: str) -> int:
    """The row a character is read through: the first token of its encoding."""
    return tokenizer(ch)["input_ids"][0]


def text_token_rows(tokenizer) -> list[int]:
    """Sorted token ids the codebook models can read for text input.

    ALLOWED_CHARS, the structural tokens and every system prompt. Callers keep
    the ids below their audio range.
    """
    from app.config import build_system_prompt
    from app.utils import tokenize_text
    from app.voice_catalog import LANGUAGE_ALIASES, SUPPORTED_VOICE_STYLES

    rows = {char_token_id(tokenizer, ch) for ch in ALLOWED_CHARS}
    rows |= {tokenizer.convert_tokens_to_ids(t) for t in STRUCTURAL_TOKENS}
    for language in sorted(set(LANGUAGE_ALIASES.values())):
        for style in SUPPORTED_VOICE_STYLES:
            rows |= set(tokenize_text(f"<|im_start|>{build_system_prompt(language, style)}<|im_end|>\n",
                                      tokenizer))
    rows.discard(tokenizer.unk_token_id)
    rows.discard(None)
    return sorted(rows)
