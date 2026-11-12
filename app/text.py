"""Text normalization and chunking for Tontaube inference."""

import re

from app.config import MAX_TEXT_CHARS_PER_CHUNK, NO_SHORT_FINAL_SENTENCE, SPLIT_SENTENCE_CUTS
from app.model.text_vocab import clean_text, strip_markup, strip_separators


_WHITESPACE_COLLAPSE_RE = re.compile(r"[ \t\n]+")
# Client text never carries special-token strings.
# app.utils.tokenize_text maps a known <|...|> (the text marker, sliders, <|utt_end|> / <|utt_cont|> on a
# bundle that declares them) to its id and drops any other. sanitize_text_for_tts removes them with
# strip_markup; prefix_text uses strip_special_token_strings.
_SPECIAL_TOKEN_STRING_RE = re.compile(r"<\|[^|]+\|>")


def strip_special_token_strings(text: str) -> str:
    """Remove literal <|...|> strings; nothing else changes (used for prefix_text)."""
    return _SPECIAL_TOKEN_STRING_RE.sub("", text)


def _collapse_whitespace(match: re.Match) -> str:
    """Collapse whitespace while preserving single and paragraph newlines."""
    newline_count = match.group().count("\n")
    if newline_count == 0:
        return " "
    if newline_count == 1:
        return "\n"
    return "\n\n"


def sanitize_text_for_tts(text: str) -> str:
    """Keep the characters the models read (see app/model/text_vocab.py), normalize whitespace."""
    text = clean_text(strip_separators(strip_markup(text)))  # clean_text also maps vertical whitespace to "\n"
    text = _WHITESPACE_COLLAPSE_RE.sub(_collapse_whitespace, text).strip()
    if text and text[-1] not in ".!?;:,":
        text += "."
    return text


# A chunk whose final sentence has only 1-2 words ("No.", "Wow.", "I know.") clips its end far more often than other
# chunk ends. With NO_SHORT_FINAL_SENTENCE the splitter moves such a sentence to the start of the next chunk. A
# "word" is a whitespace token with a letter or digit; a 1-2-token "sentence" longer than
# SHORT_FINAL_SENTENCE_MAX_CHARS (a URL, a code token) does not count as short.
SHORT_FINAL_SENTENCE_WORDS = 2
SHORT_FINAL_SENTENCE_MAX_CHARS = 40
# Words before a single "." that do not end a sentence: titles and common abbreviations (en, de, es, fr, it, nl, pt).
# Dotted abbreviations (U.S., e.g., z.B., Ph.D.), single letters (initials) and 1-2-digit numbers (list items,
# ordinals) are handled by rule in _sentence_cuts.
_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "mx", "dr", "prof", "sr", "sra", "srta", "jr", "st", "ste", "mt", "rev", "hon", "gen", "col",
    "capt", "lt", "sgt", "gov", "sen", "rep", "pres", "vs", "approx", "dept", "est", "fig", "figs", "vol", "vols",
    "cf", "al", "ca", "fr", "jan", "feb", "aug", "sep", "sept", "oct", "nov", "dec",
    "nr", "bzw", "vgl", "ggf", "evtl", "inkl", "zzgl", "sog", "hr", "frl", "str",
    "mme", "mlle", "mm", "av", "bd", "dra", "ud", "uds", "sig", "dott", "ing", "avv", "dhr", "mevr", "bijv", "blz",
})
_SENTENCE_END_RE = re.compile(r"[.!?]+[\"'”’»)\]]*(?=\s|$)")
_DOTTED_ABBREVIATION_RE = re.compile(r"(?:[^\W\d_]{1,3}\.)+[^\W\d_]{1,3}")
_CLOSERS = "\"'”’»)]"
_OPENERS = "\"'“‘«(["
# A "." with no space before a capitalised word ("the Pope.Hope. Wow.") also ends a sentence when the rule DETECTS
# a short final sentence, but it is never a cut position: in transcripts such a "." is often a caption join inside
# a spoken sentence ("because I look.So youthful."), and in code a member access ("DateTime.Now"), so the rule then
# moves the whole sentence to the previous spaced sentence end. Only a plain-word form counts, so abbreviations,
# initials, decimals, URLs / domains and code stay whole:
# the left part is the whole token before the "." (letters, apostrophes inside), >= 3 letters, ending in a lowercase
# letter, >= 4 letters if capitalised, and no known abbreviation; the right part is one capitalised word (an upper-case
# letter, then >= 1 lower-case letters, nothing else but trailing punctuation) and no domain / file suffix.
_NO_SPACE_END_RE = re.compile(r"(?<=[^\W\d_])\.(?=[^\W\d_])")
_NO_SPACE_LEFT_EXCLUDE = frozenset({
    "inc", "ltd", "llc", "corp", "etc", "bros", "ave", "blvd", "univ", "assn", "intl", "natl", "govt", "sec", "abs",
    "art", "ibid", "viz", "pts", "nos",
})
_NO_SPACE_RIGHT_EXCLUDE = frozenset({
    "com", "org", "net", "edu", "gov", "mil", "int", "info", "biz", "io", "ai", "app", "dev", "co", "uk", "de", "fr",
    "es", "it", "nl", "eu", "us", "ch", "at", "be", "tv", "me", "ly", "js", "ts", "py", "txt", "pdf", "html", "htm",
    "php", "exe", "jpg", "png", "gif", "csv", "json", "xml", "doc", "docx", "md",
})
_TRAILING_PUNCTUATION = ".,;:!?" + _CLOSERS
# '... said Henry. Lucy. "So do I; ...' -- a one-word sentence moved in front of a quotation or a dialogue dash
# becomes a chunk opener the model tends to skip. Such a move is not made.
_DIALOGUE_OPENERS = "\"'“”„‟«»‘’‚‛‹›「『—―"


def _word_count(text: str) -> int:
    """Whitespace tokens with a letter or digit."""
    return sum(1 for token in text.split() if any(char.isalnum() for char in token))


def _no_space_sentence_end(text: str, stop: int) -> bool:
    """True if the "." at ``stop`` (directly between two letters) ends a sentence: see _NO_SPACE_END_RE above."""
    start = max(text.rfind(char, 0, stop) for char in " \t\n") + 1
    end = min((index for index in (text.find(char, stop) for char in " \t\n") if index >= 0), default=len(text))
    left = text[start:stop].lstrip(_OPENERS)
    right = text[stop + 1:end].rstrip(_TRAILING_PUNCTUATION)
    letters = left.replace("'", "").replace("’", "")
    return (
        letters.isalpha()
        and len(letters) >= 3
        and left[-1].islower()
        and (len(letters) >= 4 or not left[0].isupper())
        and left.lower() not in _ABBREVIATIONS
        and left.lower() not in _NO_SPACE_LEFT_EXCLUDE
        and len(right) >= 2
        and right.isalpha()
        and right[0].isupper()
        and right[1:].islower()
        and right.lower() not in _NO_SPACE_RIGHT_EXCLUDE
    )


def _sentence_cuts(text: str, no_space: bool = True) -> list[tuple[int, int]]:
    """(index of the terminator, cut offset after it) for each sentence end in ``text``.

    A sentence end is a run of . ! ? (plus closing quotes or brackets) followed by whitespace or the end of the text,
    so a decimal ("3.5") or a dot inside a word ("example.com") never matches. A lone "." does not end a sentence
    after an abbreviation (_ABBREVIATIONS), a single letter (an initial), a dotted abbreviation of 1-3-letter parts
    (U.S., e.g., z.B.), a 1-2-digit number (a list item or an ordinal) or "No" / "Nr" before a number. An ellipsis is
    no sentence end, nor is a closing quote, "!" or "?" followed by a lowercase word (a dialogue tag: '"Wait!" said
    Derba.', "Oh! how cold"), nor a "." after a short or capitalised word followed by a lowercase word (an
    abbreviation missing from the list: "Acme Inc. in 1999", "etc. and more"). A "." with no space between a plain
    word and a capitalised word ("Pope.Hope") is a sentence end too (_no_space_sentence_end), unless ``no_space`` is
    False (the cut positions the rule may move to). The
    rule only needs these to be conservative: a missed sentence end leaves the splitter's own cut in place.
    """
    cuts = []
    for match in _SENTENCE_END_RE.finditer(text):
        group = match.group()
        run = group.rstrip(_CLOSERS)
        after = text[match.end():match.end() + 32].lstrip()
        lower_next = after[:1].islower()
        if lower_next and (run != group or run[-1] in "!?"):
            continue
        if run == ".":
            stop = match.start()
            word = text[max(text.rfind(char, 0, stop) for char in " \t\n") + 1:stop].lstrip(_OPENERS)
            if _abbreviation_word(word, after) or (lower_next and (len(word) <= 4 or word[:1].isupper())):
                continue
        elif set(run) == {"."}:
            continue
        cuts.append((match.start(), match.end()))
    if not no_space:
        return cuts
    joins = [(m.start(), m.end()) for m in _NO_SPACE_END_RE.finditer(text) if _no_space_sentence_end(text, m.start())]
    return sorted(cuts + joins) if joins else cuts


def ends_on_short_sentence(chunk: str) -> bool:
    """True if the final sentence of ``chunk`` (the text after its last inner sentence end, or the whole chunk) has
    1 to SHORT_FINAL_SENTENCE_WORDS words and at most SHORT_FINAL_SENTENCE_MAX_CHARS characters."""
    chunk = chunk.strip()
    inner = [cut for _, cut in _sentence_cuts(chunk) if cut < len(chunk)]
    final = chunk[inner[-1]:].strip() if inner else chunk
    words = _word_count(final)
    return 0 < words <= SHORT_FINAL_SENTENCE_WORDS and len(final) <= SHORT_FINAL_SENTENCE_MAX_CHARS


def _one_word_before_dialogue(moved: str, following: str, spaced: bool = True) -> bool:
    """True if ``moved`` (the text a move would put at a chunk start) is a single word and ``following`` (the text
    right after it) opens with an OPENING quotation mark or dialogue dash: the one-word sentence would open the chunk
    in front of a quotation ('Lucy. "So do I; ...'), so the rule does not move it. Opening = the mark
    stands right before a non-space character and, with ``spaced``, after whitespace; a closing mark ('Hank! "
    exclaimed ...', '"Oh!" The ...') or a spaced dash (' - ', ' — ') does not count. The merge passes the next piece
    (``spaced=False``: pieces are stripped)."""
    head = following.lstrip()
    return (
        _word_count(moved) == 1
        and head[:1] in _DIALOGUE_OPENERS
        and head[1:2].strip() != ""
        and (following[:1].isspace() or not spaced)
    )


def _cut_before_short_final_sentence(text: str, split_at: int, window_start: int) -> int:
    """If ``text[:split_at]`` would end on a short final sentence, return the latest earlier sentence end whose
    terminator lies at or after ``window_start`` (the start of split_long_text's sentence window) and whose chunk does
    not end on a short sentence. The short sentence(s) then start the next chunk. Otherwise keep ``split_at``.

    The other direction (extending the chunk to the next sentence end) never fits: the splitter already cut at the last
    sentence end within the length limit. A single moved word in front of a quotation or a dialogue dash
    (_one_word_before_dialogue) keeps ``split_at``. Candidate cuts are spaced sentence ends only (no_space=False): a
    no-space "." helps detect a short final sentence but is never cut at.
    """
    head = text[:split_at]
    if not ends_on_short_sentence(head):
        return split_at
    end = len(head.rstrip())
    for stop, cut in reversed(_sentence_cuts(head, no_space=False)):
        if stop < window_start:
            break
        if cut < end and not ends_on_short_sentence(head[:cut]):
            if _one_word_before_dialogue(head[cut:], text[split_at:]):
                return split_at
            return cut
    return split_at


def _abbreviation_word(word: str, after: str) -> bool:
    """True if a lone "." after ``word`` (followed by ``after``) never ends a sentence in _sentence_cuts: an
    abbreviation, an initial, a dotted abbreviation, a 1-2-digit number, or "No" / "Nr" before a number."""
    return bool(
        word.lower() in _ABBREVIATIONS
        or (len(word) == 1 and word.isalpha())
        or _DOTTED_ABBREVIATION_RE.fullmatch(word)
        or (word.isdigit() and len(word) <= 2)
        or (word.lower() in ("no", "nr") and after[:1].isdigit())
    )


def _known_non_end(text: str, stop: int) -> bool:
    """SPLIT_SENTENCE_CUTS: True if the ". ! ?" at ``stop`` is known not to end a sentence: a "." after an
    abbreviation-like word (_abbreviation_word: "Dr. Brown", "am 3. Oktober", "z. B. die", "Dr.Smith"), or a "." inside
    a token that a digit or a lowercase letter follows ("3.5", "example.com"). An ellipsis, "!" / "?" and a caption join
    before a capital ("room.And", "day.It's") still count, as in the previous rule."""
    if text[stop] != ".":
        return False
    end = stop + 1
    while end < len(text) and text[end] in _CLOSERS:
        end += 1
    if (stop > 0 and text[stop - 1] == ".") or (end < len(text) and text[end] == "."):
        return False                                                      # an ellipsis: as before
    word = text[max(text.rfind(char, 0, stop) for char in " \t\n") + 1:stop].lstrip(_OPENERS)
    if end == stop + 1 and end < len(text) and not text[end].isspace():
        word = word[word.rfind(".") + 1:] if not _DOTTED_ABBREVIATION_RE.fullmatch(word) else word
        return not text[end].isupper() or _abbreviation_word(word, text[end:])
    return _abbreviation_word(word, text[end:end + 32].lstrip())


def _inside_number(text: str, index: int) -> bool:
    """SPLIT_SENTENCE_CUTS: True if the separator at ``index`` sits between two digits ("2,000", "3:15")."""
    return 0 < index < len(text) - 1 and text[index - 1].isdigit() and text[index + 1].isdigit()


def _last_sentence_cut(text: str, start: int, limit: int) -> int:
    """SPLIT_SENTENCE_CUTS: the sentence-first cut of split_long_text, or -1 for none in the window.

    First the last real sentence end (_sentence_cuts, spaced ends only) whose terminator lies at or after ``start``
    and whose cut (closing quotes / brackets included) fits ``limit``. Without one, the previous rule (the last ". ! ?"
    in the window) minus the known non-ends (_known_non_end), so "am 3. Oktober", "z. B. die", "Dr. Brown", "3.5" and
    "example.com" never end a chunk, while an ellipsis or a "." before a lowercase word still can, as before."""
    best = -1
    for stop, cut in _sentence_cuts(text, no_space=False):
        if stop >= limit:
            break
        if stop >= start and cut <= limit:
            best = cut
    if best >= 0:
        return best
    for stop in range(limit - 1, start - 1, -1):
        if text[stop] in ".!?" and not _known_non_end(text, stop):
            cut = stop + 1
            while cut < limit and text[cut] in _CLOSERS:
                cut += 1
            return cut
    return -1


def split_long_text(
    text: str | list[str],
    max_chars: int = MAX_TEXT_CHARS_PER_CHUNK,
    digit_weight: int = 1,
    avoid_short_final: bool = NO_SHORT_FINAL_SENTENCE,
    sentence_cuts: bool = SPLIT_SENTENCE_CUTS,
) -> str | list[str]:
    """Split overlong segments sentence-first (. ! ?), then at ; : and commas, whitespace, then a hard limit.

    ``digit_weight`` can reserve space for text expansion performed after this
    split, such as verbalizing written numbers into spoken words.
    ``avoid_short_final`` (NO_SHORT_FINAL_SENTENCE): a cut that would end a chunk on a 1-2-word
    sentence moves back to an earlier sentence end in the same window (_cut_before_short_final_sentence); the
    remainder is split as before, so every piece keeps the length limit. The last piece of a segment is never changed.
    ``sentence_cuts`` (SPLIT_SENTENCE_CUTS): the sentence-first cut is the last real sentence end in the window,
    else the last ". ! ?" that is not a known non-end (_last_sentence_cut); without one the ; : and comma tiers follow.
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

    def split_segment(segment: str, avoid: bool = avoid_short_final) -> list[str]:
        remainder = segment.strip()
        pieces: list[str] = []

        while weighted_length(remainder) > max_chars:
            limit = prefix_limit(remainder)
            # Sentence-first: a sentence end in the window's final 70 %, else ; or :,
            # else a comma. Comma seams make each chunk restart like a new utterance.
            # SPLIT_SENTENCE_CUTS: the sentence tier is the last real sentence end (_last_sentence_cut).
            split_at = _last_sentence_cut(remainder, int(limit * 0.3), limit) if sentence_cuts else -1
            if split_at < 0:
                boundary = -1
                tiers = ((";:", 0), (",", 0)) if sentence_cuts else ((".!?", int(limit * 0.3)), (";:", 0), (",", 0))
                for chars, start in tiers:
                    boundary = max(remainder.rfind(char, start, limit) for char in chars)
                    while sentence_cuts and boundary > 0 and _inside_number(remainder, boundary):
                        boundary = max(remainder.rfind(char, start, boundary) for char in chars)
                    if boundary >= 0:
                        break
                if boundary >= 0:
                    split_at = boundary + 1
                else:
                    whitespace = max(
                        remainder.rfind(char, 0, limit + 1)
                        for char in (" ", "\t", "\n")
                    )
                    split_at = whitespace if whitespace > 0 else limit
            if avoid:
                split_at = _cut_before_short_final_sentence(remainder, split_at, int(limit * 0.3))

            pieces.append(remainder[:split_at].strip())
            remainder = remainder[split_at:].strip()

        if remainder:
            pieces.append(remainder)
        # A piece without any word character (e.g. a stray closing quote) would be a
        # wasted, out-of-distribution CB0 call: attach it to its neighbour instead.
        merged: list[str] = []
        for piece in pieces:
            if merged and not any(char.isalnum() for char in piece):
                merged[-1] = f"{merged[-1]}{piece}"
            elif merged and not any(char.isalnum() for char in merged[-1]):
                merged[-1] = f"{merged[-1]}{piece}"
            else:
                merged.append(piece)
        # The glue above has no length check, and a moved cut can shift a stray emoji or quote onto a
        # full chunk. The rule must not add an over-limit chunk that the plain split does not make.
        if avoid and any(weighted_length(piece) > max_chars for piece in merged):
            plain = split_segment(segment, avoid=False)
            if all(weighted_length(piece) <= max_chars for piece in plain):
                return plain
        return merged

    if isinstance(text, str):
        pieces = split_segment(text)
        return pieces[0] if len(pieces) == 1 else pieces

    return [piece for segment in text for piece in split_segment(segment)]


# Pieces under 50 characters fail far more often (the model reads on into the next piece's
# lookahead text), so they join a neighbour.
SHORT_PIECE_CHARS = 50
MAX_MERGED_PIECE_CHARS = MAX_TEXT_CHARS_PER_CHUNK + SHORT_PIECE_CHARS


def _join_pieces(left: str, right: str) -> str:
    # A plain space: the splitter keeps each cut's punctuation in the left piece, so a sentence end
    # stays one, and a clause cut at a space must not gain a false full stop. (Client segments are
    # sanitised before the split and already end in punctuation.)
    return f"{left} {right}"


def merge_short_pieces(
    pieces: list[str],
    keys: list | None = None,
    min_chars: int = SHORT_PIECE_CHARS,
    max_chars: int | list[int] = MAX_MERGED_PIECE_CHARS,
    digit_weight: int = 1,
    avoid_short_final: bool = NO_SHORT_FINAL_SENTENCE,
) -> tuple[list[str], list | None]:
    """Join each piece shorter than ``min_chars`` to its previous piece if the result fits in
    ``max_chars`` (digits weighted as in split_long_text), else to its next piece; repeat until
    no piece changes. Pieces join with a space (no punctuation is added). Only neighbours with equal
    ``keys`` entries join. ``max_chars`` may be one limit per piece (a join takes the smaller).
    With ``avoid_short_final``, a short piece that itself ends on a 1-2-word sentence tries its
    next piece first, so the joined chunk does not end on that sentence.
    Returns (pieces, keys), the keys aligned with the merged pieces.
    """
    pieces = list(pieces)
    merged_keys = list(keys) if keys is not None else [None] * len(pieces)
    limits = list(max_chars) if isinstance(max_chars, list) else [max_chars] * len(pieces)

    def fits(segment: str, limit: int) -> bool:
        return len(segment) + sum(digit_weight - 1 for char in segment if char.isdigit()) <= limit

    changed = True
    while changed and len(pieces) > 1:
        changed = False
        for index, piece in enumerate(pieces):
            if len(piece) >= min_chars:
                continue
            order = (index - 1, index)  # previous piece first, then next
            if (
                avoid_short_final
                and ends_on_short_sentence(piece)
                and not (index + 1 < len(pieces) and _one_word_before_dialogue(piece, pieces[index + 1], spaced=False))
            ):
                order = (index, index - 1)
            for left in order:
                if left < 0 or left + 1 >= len(pieces) or merged_keys[left] != merged_keys[left + 1]:
                    continue
                joined = _join_pieces(pieces[left], pieces[left + 1])
                limit = min(limits[left], limits[left + 1])
                if fits(joined, limit):
                    pieces[left:left + 2] = [joined]
                    limits[left:left + 2] = [limit]
                    del merged_keys[left + 1]
                    changed = True
                    break
            if changed:
                break
    return pieces, (merged_keys if keys is not None else None)
