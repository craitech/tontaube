"""Text normalization and chunking for Tontaube inference."""

import re

from app.config import MAX_TEXT_CHARS_PER_CHUNK


_WHITESPACE_COLLAPSE_RE = re.compile(r"[ \t\n]+")
_VERTICAL_WS = {
    "\v",
    "\f",
    "\r",
    "\x1c",
    "\x1d",
    "\x1e",
    "\x85",
    "\u2028",
    "\u2029",
}


def _collapse_whitespace(match: re.Match) -> str:
    """Collapse whitespace while preserving single and paragraph newlines."""
    newline_count = match.group().count("\n")
    if newline_count == 0:
        return " "
    if newline_count == 1:
        return "\n"
    return "\n\n"


def sanitize_text_for_tts(text: str) -> str:
    """Normalize whitespace without discarding multilingual characters."""
    for char in _VERTICAL_WS:
        text = text.replace(char, "\n")
    text = _WHITESPACE_COLLAPSE_RE.sub(_collapse_whitespace, text).strip()
    if text and text[-1] not in ".!?;:,":
        text += "."
    return text


def split_long_text(
    text: str | list[str],
    max_chars: int = MAX_TEXT_CHARS_PER_CHUNK,
    digit_weight: int = 1,
) -> str | list[str]:
    """Split overlong segments at punctuation, whitespace, then a hard limit.

    ``digit_weight`` can reserve space for text expansion performed after this
    split, such as verbalizing written numbers into spoken words.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if digit_weight < 1:
        raise ValueError("digit_weight must be positive")

    def weighted_length(segment: str) -> int:
        return len(segment) + sum(
            digit_weight - 1 for char in segment if char.isdigit()
        )

    def prefix_limit(segment: str) -> int:
        cost = 0
        for index, char in enumerate(segment):
            next_cost = cost + (digit_weight if char.isdigit() else 1)
            if next_cost > max_chars:
                return max(1, index)
            cost = next_cost
        return len(segment)

    def split_segment(segment: str) -> list[str]:
        remainder = segment.strip()
        pieces: list[str] = []

        while weighted_length(remainder) > max_chars:
            limit = prefix_limit(remainder)
            boundary = max(
                remainder.rfind(char, 0, limit)
                for char in ".!?;:,"
            )
            if boundary >= 0:
                split_at = boundary + 1
            else:
                whitespace = max(
                    remainder.rfind(char, 0, limit + 1)
                    for char in (" ", "\t", "\n")
                )
                split_at = whitespace if whitespace > 0 else limit

            pieces.append(remainder[:split_at].strip())
            remainder = remainder[split_at:].strip()

        if remainder:
            pieces.append(remainder)
        return pieces

    if isinstance(text, str):
        pieces = split_segment(text)
        return pieces[0] if len(pieces) == 1 else pieces

    return [piece for segment in text for piece in split_segment(segment)]
