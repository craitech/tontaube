"""Prefix context for verbalizing a continuation (VERBALIZE_PREFIX_CONTEXT).

A prefix continuation sends the start of a sentence as prefix_text (already spoken) and the rest as text. The
verbalizer used to see only the rest, a mid-sentence fragment, and read it as a new sentence: it capitalised the
first word, completed the fragment ("A smaller plan, then ..."), dropped words,
and once returned its own system instructions (the prompt has no separator, so a fragment like "or dairy, but ..."
reads as more instructions). Verbalizer.verbalize_continuation now reads context_tail(prefix_text) + " " + rest and
keeps only the rest's part (verbalizer_guard.split_continuation); match_first_case keeps the rest's own first-letter
case.
"""

import re

from app.text import _sentence_cuts

# The context is the prefix's last sentence, at most this long (the verbalizer's whole sequence is 480 tokens).
CONTEXT_MAX_WORDS = 12
CONTEXT_MAX_CHARS = 100
_TAG_SPLIT_RE = re.compile(r"<<[^>]+>>")
_CLAUSE_ENDS = (",", ";", ":", "—", "–")
_ENDS_SENTENCE_RE = re.compile(r"[.!?…][\"'”’»)\]]*\s*$")
_WORD_AT_RE = re.compile(r"[^\W\d_](?:[^\W\d_]|['’])*")          # a word of letters (any script) and apostrophes
_FIRST_WORD_RE = re.compile(r"^([\W_]*)([^\W\d_](?:[^\W\d_]|['’])*)")


def context_tail(prefix_text: str) -> str:
    """The end of ``prefix_text`` the verbalizer sees before the rest: the text after its last <<tag>>, last sentence
    only, at most CONTEXT_MAX_WORDS words / CONTEXT_MAX_CHARS chars. When that cuts the sentence, the context starts
    after the window's first clause boundary (if 3+ words remain). The first letter is capitalised, so the model sees
    a sentence start; the context's own reading is discarded."""
    text = " ".join(_TAG_SPLIT_RE.split(prefix_text)[-1].split())
    if not text:
        return ""
    inner = [cut for _, cut in _sentence_cuts(text) if cut < len(text)]
    words = (text[inner[-1]:] if inner else text).split()
    keep: list[str] = []
    for word in reversed(words):
        if len(keep) >= CONTEXT_MAX_WORDS or len(" ".join([word, *keep])) > CONTEXT_MAX_CHARS:
            break
        keep.insert(0, word)
    if not keep:
        return ""
    if len(keep) < len(words):
        for i, word in enumerate(keep[:-3]):
            if word.endswith(_CLAUSE_ENDS):
                keep = keep[i + 1:]
                break
    return re.sub(r"^([^A-Za-z0-9]*)([a-z])", lambda m: m.group(1) + m.group(2).upper(), " ".join(keep))


def ends_sentence(context: str) -> bool:
    """True if ``context`` ends with sentence-final punctuation: the rest then starts a sentence of its own."""
    return bool(_ENDS_SENTENCE_RE.search(context))


def match_first_case(raw_rest: str, part: str, context: str) -> str:
    """Give ``part`` (the verbalized rest) the raw rest's own first-letter case.

    Raw rest starts with an upper-case letter: unchanged. Starts lower-case: the first word gets the raw word's case
    if it is the same word, else a capitalised first word ("Twenty-five") is lower-cased; all-caps words (FBI) stay.
    Starts with a digit ("25 minutes", "$20"): the same, unless the context ends a sentence."""
    first = re.search(r"[^\W_]", raw_rest)
    if first is None or first.group(0).isupper():
        return part
    if first.group(0).isdigit() and ends_sentence(context):
        return part
    m = _FIRST_WORD_RE.match(part)
    if m is None:
        return part
    word = m.group(2)
    raw_word = _WORD_AT_RE.match(raw_rest, first.start())
    if raw_word and raw_word.group(0).lower() == word.lower():
        new = raw_word.group(0)
    elif word[0].isupper() and not any(c.isupper() for c in word[1:]) and not re.fullmatch(r"I(?:['’]\w+)?", word):
        new = word[0].lower() + word[1:]
    else:
        return part
    return part[:m.start(2)] + new + part[m.end(2):]
