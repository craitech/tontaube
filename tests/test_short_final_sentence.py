"""NO_SHORT_FINAL_SENTENCE: never end a chunk on a 1-2-word sentence.

A chunk whose final sentence has 1-2 words ("No.", "Wow.", "I know.") clips its end more often than other chunk ends.
split_long_text now moves such a sentence to the start of the next chunk, within the splitter's own sentence window
and length limit; the merge of short pieces prefers the next piece for a short piece that ends on one.
NO_SHORT_FINAL_SENTENCE=0 restores the previous code, kept verbatim below as the reference.
"""
import asyncio
import inspect
import os
import random

import pytest

import app.config as config_module
from app.config import MAX_TEXT_CHARS_PER_CHUNK
from app.text import (
    MAX_MERGED_PIECE_CHARS,
    SHORT_PIECE_CHARS,
    _cut_before_short_final_sentence,
    ends_on_short_sentence,
    merge_short_pieces,
    split_long_text,
)


# --- the earlier splitter and merge, verbatim ---------------------------------------------------------------------
def _old_split_long_text(
    text: str | list[str],
    max_chars: int = MAX_TEXT_CHARS_PER_CHUNK,
    digit_weight: int = 1,
) -> str | list[str]:
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
            boundary = -1
            for chars, start in ((".!?", int(limit * 0.3)), (";:", 0), (",", 0)):
                boundary = max(remainder.rfind(char, start, limit) for char in chars)
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

            pieces.append(remainder[:split_at].strip())
            remainder = remainder[split_at:].strip()

        if remainder:
            pieces.append(remainder)
        merged: list[str] = []
        for piece in pieces:
            if merged and not any(char.isalnum() for char in piece):
                merged[-1] = f"{merged[-1]}{piece}"
            elif merged and not any(char.isalnum() for char in merged[-1]):
                merged[-1] = f"{merged[-1]}{piece}"
            else:
                merged.append(piece)
        return merged

    if isinstance(text, str):
        pieces = split_segment(text)
        return pieces[0] if len(pieces) == 1 else pieces

    return [piece for segment in text for piece in split_segment(segment)]


def _old_merge_short_pieces(
    pieces: list[str],
    keys: list | None = None,
    min_chars: int = SHORT_PIECE_CHARS,
    max_chars: int | list[int] = MAX_MERGED_PIECE_CHARS,
    digit_weight: int = 1,
) -> tuple[list[str], list | None]:
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
            for left in (index - 1, index):  # previous piece first, then next
                if left < 0 or left + 1 >= len(pieces) or merged_keys[left] != merged_keys[left + 1]:
                    continue
                joined = f"{pieces[left]} {pieces[left + 1]}"
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


# The rule's tests pass avoid_short_final=True explicitly, so they also hold with NO_SHORT_FINAL_SENTENCE=0 in the
# environment (the rollback suite run); test_flag_default_on_and_rollback checks the default itself.
def _split(text, **kwargs):
    kwargs.setdefault("avoid_short_final", True)
    # this suite pins the short-final-sentence rule against the character-search splitter (_old_split_long_text);
    # the sentence-first cut (SPLIT_SENTENCE_CUTS, default on) has its own suite, tests/test_split_sentence_cuts.py.
    kwargs.setdefault("sentence_cuts", False)
    return split_long_text(text, **kwargs)


def _merge(pieces, **kwargs):
    kwargs.setdefault("avoid_short_final", True)
    return merge_short_pieces(pieces, **kwargs)


# --- texts --------------------------------------------------------------------------------------------------------
FILL = "The quick brown fox jumps over the lazy dog near the old river bank."      # 68 chars
LEAD = " ".join([FILL] * 4)                                                          # 275 chars
NEXT = "Then the dog woke up, looked around the quiet field and slowly walked back home for dinner."  # 91 chars
SHERLOCK = (
    "My own complete happiness, and the home-centred interests which rise up around the man who first "
    "finds himself master of his own establishment, were sufficient to absorb all my attention, while "
    "Holmes, who loathed every form of society with his whole Bohemian soul, remained in our lodgings in "
    "Baker Street, buried among his old books, and alternating from week to week between cocaine and "
    "ambition, the drowsiness of the drug, and the fierce energy of his own keen nature. "
)


def _pieces(result) -> list[str]:
    return [result] if isinstance(result, str) else list(result)


# --- the sentence test --------------------------------------------------------------------------------------------
@pytest.mark.parametrize("chunk, short", [
    ("No.", True),
    ("Wow.", True),
    ("I know.", True),
    ("Was it good? No.", True),
    ("Really? Absolutely!", True),
    (FILL + " Wow.", True),
    ('He left the door. "No."', True),
    (FILL, False),
    ("Thanks a lot.", False),                         # three words
    # abbreviations, initials, decimals and numbers do not end a sentence
    ("Dr. Smith arrived.", False),
    ("Dr. Smith arrived. 3.5 percent.", True),        # "3.5 percent." is the 2-word final sentence
    ("The rate rose to 3.5 percent.", False),
    ("Growth was 3.5. Okay.", True),
    ("It costs $5.", False),
    ("It costs $5. Okay.", True),
    ("Ask J. R. Smith.", False),
    ("See e.g. the docs.", False),
    ("We moved to the U.S. last year.", False),
    ("We picked item No. 5.", False),
    ("Er kam am 3. Mai.", False),                     # a German ordinal
    ('"Wait!" said Derba.', False),                   # a dialogue tag continues the sentence
    ('He said "No."', False),
    ("Well... I know.", False),                       # an ellipsis is no sentence end
    ("a" * 200 + ".", False),                         # one 200-char token is not a short sentence
    ("See https://example.com/a/very/long/path/to/some/page.", False),   # 2 tokens, but over 40 chars
])
def test_ends_on_short_sentence(chunk, short):
    assert ends_on_short_sentence(chunk) is short


# --- the splitter -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("short", ["No.", "Wow.", "I know.", "Really?", "Okay. Sure. Fine."])
def test_short_final_sentence_moves_to_the_next_chunk(short):
    text = f"{LEAD} {short} {NEXT}"
    assert _old_split_long_text(text) == [f"{LEAD} {short}", NEXT]           # the old chunk ended on it
    assert _split(text) == [LEAD, f"{short} {NEXT}"]
    assert _split(text, avoid_short_final=False) == _old_split_long_text(text)


@pytest.mark.parametrize("tail, first, second", [
    ("Dr. Smith arrived. 3.5 percent.", f"{LEAD} Dr. Smith arrived.", f"3.5 percent. {NEXT}"),
    ("It costs $5. Okay.", f"{LEAD} It costs $5.", f"Okay. {NEXT}"),
    # "p.m." reads as an abbreviation, so no short final sentence is seen: a conservative miss, the old cut
    ("Mr. J. R. Smith left at 5 p.m. Wow.", f"{LEAD} Mr. J. R. Smith left at 5 p.m. Wow.", NEXT),
])
def test_numbers_and_abbreviations_are_never_cut(tail, first, second):
    text = f"{LEAD} {tail} {NEXT}"
    assert _split(text) == [first, second]


@pytest.mark.parametrize("tail", [
    "I met Dr. Who.",                     # final sentence of four words: "Dr." is no sentence end
    "Ask J. R. Smith.",
    "The fee rose to 3.5 percent.",
    "Thanks a lot.",
])
def test_no_change_without_a_short_final_sentence(tail):
    text = f"{LEAD} {tail} {NEXT}"
    assert _split(text) == _old_split_long_text(text) == [f"{LEAD} {tail}", NEXT]


@pytest.mark.parametrize("text", [
    SHERLOCK * 3,
    "word " * 500,
    "a" * 1000,
    ("clause, " * 100) + "end",
    "a" * 169 + ". " + "b" * 168 + ". " + "c" * 169 + ". " + "d" * 168 + ". Thanks a lot.",
    " ".join([FILL] * 20),
    f"{LEAD} {NEXT} Okay.",                # the last piece of a text is never changed
])
def test_no_change_on_texts_without_short_chunk_ends(text):
    assert _split(text) == _old_split_long_text(text)
    assert _split(text, digit_weight=4) == _old_split_long_text(text, digit_weight=4)


def test_kept_when_no_other_sentence_end_lies_in_the_window():
    # the only earlier sentence end ("friend.") lies before the window's first 30 %, so the cut stays
    text = "Hello there, friend. " + "word " * 62 + "No. " + NEXT
    assert _split(text) == _old_split_long_text(text)
    assert _pieces(_split(text))[0].endswith("No.")


def test_short_piece_from_a_clause_cut_is_kept():
    # the comma fallback cuts after the only comma; no sentence end to move to
    text = "Yeah, no " + "uh so yeah I think " * 18 + "and that is all I wanted to say really. " + NEXT
    assert _pieces(_split(text))[0] == "Yeah,"
    assert _split(text) == _old_split_long_text(text)


def _random_text(rng: random.Random) -> str:
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet",
             "kilo", "lima", "mike", "november", "oscar", "papa", "quebec", "romeo", "sierra", "tango"]
    shorts = ["No.", "Wow.", "I know.", "Okay.", "Right!", "Really?", "Got it."]
    sentences = []
    for _ in range(rng.randint(3, 40)):
        if rng.random() < 0.3:
            sentences.extend(rng.choice(shorts) for _ in range(rng.randint(1, 2)))
        long = " ".join(rng.choice(words) for _ in range(rng.randint(4, 12)))
        if rng.random() < 0.3:
            long = long.replace(" ", ", ", 1)
        sentences.append(long.capitalize()[:99] + rng.choice(".!?"))
    return " ".join(sentences)


@pytest.mark.parametrize("seed", range(200))
def test_random_texts_keep_limits_and_words_and_never_end_short(seed):
    rng = random.Random(seed)
    text = _random_text(rng)
    pieces = _pieces(_split(text))
    assert all(0 < len(piece) <= MAX_TEXT_CHARS_PER_CHUNK for piece in pieces)
    assert " ".join(pieces).split() == text.split()
    # sentences of <= 100 chars with <= 2 short ones in a row: a safe cut always exists in the window
    assert not any(ends_on_short_sentence(piece) for piece in pieces[:-1])
    for max_chars, digit_weight in ((245, 1), (350, 4), (120, 1)):
        scaled = _pieces(_split(text, max_chars=max_chars, digit_weight=digit_weight))
        assert all(len(piece) <= max_chars for piece in scaled)
        assert " ".join(scaled).split() == text.split()
    # rollback flag: the old splitter exactly
    assert _split(text, avoid_short_final=False) == _old_split_long_text(text)
    assert _split(text, max_chars=120, avoid_short_final=False) == _old_split_long_text(text, max_chars=120)


def test_list_input_applies_the_rule_inside_each_segment():
    text = ["Hi there.", f"{LEAD} No. {NEXT}"]
    assert _split(text) == ["Hi there.", LEAD, f"No. {NEXT}"]
    # a client segment boundary is the client's choice: never crossed by the splitter
    assert _split([f"{LEAD} No.", NEXT]) == [f"{LEAD} No.", NEXT]


# --- no false sentence end before a lowercase tail ----------------------------------------------------------------
# An earlier version of the rule read "Inc." / "etc." / "usw." / "Fr." / "fail!" + a 1-2-word lowercase tail as a
# short final sentence and cut INSIDE the sentence, where the old splitter had cut cleanly at its end.
@pytest.mark.parametrize("chunk, short", [
    ("Built by Acme Inc. in 1999.", False),
    ("We checked the alerts, etc. and more.", False),
    ("Wir prüfen die Metriken usw. und mehr.", False),
    ("Answered by Fr. Brown.", False),                    # "Fr." (Father / Frau) is a title
    ("The plan would fail! no way.", False),              # "!" / "?" + a lowercase word: no sentence end
    ("Is it done? not yet.", False),
    ("Oh! how cold.", False),
    ("We checked the alerts, etc. Done.", True),          # a capitalised word after "etc.": a real sentence end
    ("He spoke of my father. two centuries.", True),      # "." after a long lowercase word stays an end
])
def test_no_sentence_end_before_a_lowercase_tail(chunk, short):
    assert ends_on_short_sentence(chunk) is short


@pytest.mark.parametrize("tail", [
    "The platform was built by Acme Inc. in 1999.",
    "Parts are supplied by Bolt Ltd. in Leeds.",
    "The contract went to Smith & Co. last year.",
    "We checked the logs, the alerts, etc. and more.",
    "Wir prüfen die Logs, die Metriken usw. und mehr.",
    "The letter was answered by Fr. Brown.",
    "He kept shouting that the plan would fail! no way.",
])
def test_no_cut_inside_a_sentence_before_a_lowercase_tail(tail):
    text = f"{LEAD} {tail} {NEXT}"
    assert _split(text) == _old_split_long_text(text) == [f"{LEAD} {tail}", NEXT]


def test_real_sentence_end_after_etc_still_moves():
    text = f"{LEAD} We checked the logs, the alerts, etc. Done. {NEXT}"
    assert _split(text) == [f"{LEAD} We checked the logs, the alerts, etc.", f"Done. {NEXT}"]


# --- the rule never adds an over-limit chunk that the plain split does not make -----------------------------------
def test_rule_falls_back_to_the_plain_split_instead_of_an_over_limit_chunk():
    # The old no-alnum glue (no length check) appends a trailing "⚠️." to its neighbour. Moving "Of course." makes
    # that neighbour a full 349-char chunk, so the glued chunk would be 352 chars (as an earlier version of the rule
    # did); the plain split fits, so the segment keeps the plain split.
    body = f"{NEXT} {' '.join([FILL] * 3)} It means the word of God and of us all."
    text = f"{LEAD} Of course. {body} ⚠️."
    plain = _old_split_long_text(text)
    assert plain == [f"{LEAD} Of course.", f"{body} ⚠️."]
    assert all(len(piece) <= MAX_TEXT_CHARS_PER_CHUNK for piece in plain)
    assert len(f"Of course. {body}⚠️.") > MAX_TEXT_CHARS_PER_CHUNK >= len(f"Of course. {body}")
    assert _split(text) == plain
    # without the emoji the rule moves the short sentence as usual
    assert _split(f"{LEAD} Of course. {body}") == [LEAD, f"Of course. {body}"]


# --- "X.Y." without a space is two sentences ("...for the Pope.Hope. Wow.") ---------------------------------------
@pytest.mark.parametrize("chunk, short", [
    ("That was a protection for the Pope.Hope.", True),       # "Hope." is a 1-word final sentence
    ("We talked about him.She left.", True),
    ("He wrote to Henry's.The end.", True),
    ("Pope.Hope.", True),
    # abbreviations, initials, acronyms, decimals, domains, files and code stay one sentence
    ("We were seen at the clinic by Dr.Smith.", False),
    ("The new office opened in St.Louis.", False),
    ("The book was written by J.Tolkien.", False),
    ("Like, I.If I was going to be there.", False),           # "I." may be an initial
    ("They served in the U.S.Army.", False),
    ("She wrote a thesis for the Ph.D.Program.", False),
    ("The phones are made by Samsung Co.Ltd.", False),
    ("Parts come from Acme Inc.Today.", False),               # a capitalised 3-letter word: an abbreviation
    ("The rate went up to 3.5.Okay then, fine.", False),     # a digit before the dot
    ("We booked the whole trip on Booking.Com.", False),
    ("Everything is hosted on example.com.", False),
    ("They registered the site Example.Co.Uk.", False),
    ("Our backend runs on ASP.NET.", False),
    ("Print it with Console.WriteLine.", False),
    ("The script is called Setup.Py.", False),
    ("Please visit our site www.Example.Org.", False),
    ("It was cold, and so was it.The end.", False),          # a 2-letter left word: a conservative miss
])
def test_no_space_sentence_end(chunk, short):
    assert ends_on_short_sentence(chunk) is short


def test_no_space_short_final_sentence_moves_the_whole_sentence():
    # The earlier rule kept "...for the Pope.Hope." (it saw one 7-word sentence). Now "Hope." is seen as a 1-word
    # final sentence, but the no-space "." is never a cut position, so the whole sentence moves to the next chunk
    tail = "That was a protection for the Pope.Hope."
    text = f"{LEAD} {tail} {NEXT}"
    assert _old_split_long_text(text) == [f"{LEAD} {tail}", NEXT]
    assert _split(text) == [LEAD, f"{tail} {NEXT}"]
    # the "...Pope.Hope. Wow." form: "Wow." and the undetected "Hope." move together with their sentence
    text = f"{LEAD} {tail} Wow. {NEXT}"
    assert _old_split_long_text(text) == [f"{LEAD} {tail} Wow.", NEXT]
    assert _split(text) == [LEAD, f"{tail} Wow. {NEXT}"]
    assert _split(text, avoid_short_final=False) == _old_split_long_text(text)


@pytest.mark.parametrize("tail", [
    "Even though it is so hard to believe, because I look.So youthful.",      # a caption join inside a sentence
    "That is greatness, and that's accessible.To everybody.",
    "To get the time, call DateTime.Now. Done.",                              # code: "DateTime.Now" stays whole
])
def test_no_space_dot_is_never_a_cut_position(tail):
    text = f"{LEAD} {tail} {NEXT}"
    pieces = _pieces(_split(text))
    assert pieces == [LEAD, f"{tail} {NEXT}"]
    assert not any(piece.endswith(("look.", "accessible.", "DateTime.")) for piece in pieces)


@pytest.mark.parametrize("tail", [
    "We were seen at the clinic by Dr.Smith.",
    "The new office opened in St.Louis.",
    "The book was written by J.Tolkien.",
    "They served in the U.S.Army.",
    "The phones are made by Samsung Co.Ltd.",
    "We booked the whole trip on Booking.Com.",
    "Everything is hosted on example.com.",
    "Our backend runs on ASP.NET.",
    "The rate went up to 3.5.Okay then, fine.",
])
def test_no_space_forms_that_are_no_sentence_end_keep_the_old_cut(tail):
    text = f"{LEAD} {tail} {NEXT}"
    assert _split(text) == _old_split_long_text(text) == [f"{LEAD} {tail}", NEXT]


# --- a one-word sentence is not moved in front of a quotation ('Lucy. "So do I; ...') -----------------------------
QUOTE = '"So do I; don\'t you remember, Henry, that we were sitting here once, thinking of it?"'


@pytest.mark.parametrize("opener", ['"', "“", "„", "«", "»", "‘", "'", "—", "―"])
def test_one_word_sentence_before_a_quotation_is_not_moved(opener):
    head = f'{LEAD} "I love this arbour," said Henry. Lucy.'
    rest = opener + QUOTE[1:]
    text = f"{head} {rest} {NEXT}"
    assert _old_split_long_text(text)[0] == head                    # the old chunk ends on "Lucy."
    assert _split(text) == _old_split_long_text(text)               # and keeps doing so


def test_moves_that_do_not_open_a_chunk_with_one_word_before_a_quotation_still_happen():
    head = f'{LEAD} "I love this arbour," said Henry.'
    # no quotation after the one-word sentence: moved as before
    text = f"{head} Lucy. So do I, and I remember that we were sitting here once, thinking of it. {NEXT}"
    assert _pieces(_split(text))[:2] == [head, "Lucy. So do I, and I remember that we were sitting here once, "
                                         f"thinking of it. {NEXT}"]
    # two words in front of the quotation: moved as before (the quotation check blocks one-word openers only)
    text = f"{head} I know. {QUOTE} {NEXT}"
    assert _pieces(_split(text))[:2] == [head, f"I know. {QUOTE} {NEXT}"]
    # two one-word sentences: the chunk opens with "Okay." in front of "Lucy.", not in front of the quotation
    text = f"{head} Okay. Lucy. {QUOTE} {NEXT}"
    assert _pieces(_split(text))[:2] == [head, f"Okay. Lucy. {QUOTE} {NEXT}"]


@pytest.mark.parametrize("tail, after", [
    ('He poured it down the throat. Hank!', '" exclaimed Tilda rebukingly, and, becoming sober, he sat down again by the fire.'),
    ('He would keep the appointment. "Oh!', '" The further I inquire, the more am I convinced that nothing will come of it.'),
    ('He stood up and left the room. Fine.', ' — and that was the last time anyone saw him in the village that summer.'),
])
def test_a_closing_quotation_mark_or_a_spaced_dash_does_not_block_the_move(tail, after):
    # real texts: the mark after the one-word sentence CLOSES a quotation (or is a spaced dash),
    # so no chunk opens in front of a quotation: the rule moves the word as usual
    text = f"{LEAD} {tail}{after} {NEXT}"
    old = _old_split_long_text(text)
    assert old[0] == f"{LEAD} {tail}"                                # the old chunk ends on the one-word sentence
    moved = _pieces(_split(text))
    assert moved[0] == f"{LEAD} {tail.rsplit(' ', 1)[0]}" and moved[1].startswith(tail.split()[-1] + after[:1])


def test_merge_does_not_put_a_one_word_sentence_in_front_of_a_quotation():
    pieces = [LEAD, "Lucy.", QUOTE]
    assert _merge(pieces)[0] == _old_merge_short_pieces(pieces)[0] == [f"{LEAD} Lucy.", QUOTE]
    # without the quotation the short piece still joins its next piece
    plain = [LEAD, "Lucy.", QUOTE.strip('"')]
    assert _merge(plain)[0] == [LEAD, f"Lucy. {QUOTE.strip(chr(34))}"]
    # a next piece that starts with a CLOSING mark (followed by a space) does not block the next-first join
    closing = [LEAD, "Hank!", '" exclaimed Tilda rebukingly, and, becoming sober, he sat down again.']
    assert _merge(closing)[0] == [LEAD, f"Hank! {closing[2]}"]


def _random_text_with_joins_and_quotes(rng: random.Random) -> str:
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]
    shorts = ["No.", "Wow.", "Lucy.", "I know.", "Okay.", "Pope.Hope.", "Dr.Smith.", "St.Louis.", "it.The"]
    sentences = []
    for _ in range(rng.randint(3, 40)):
        if rng.random() < 0.35:
            sentences.extend(rng.choice(shorts) for _ in range(rng.randint(1, 2)))
        long = " ".join(rng.choice(words) for _ in range(rng.randint(4, 12))).capitalize()[:99] + rng.choice(".!?")
        if rng.random() < 0.3:
            long = rng.choice(['"', "„", "«", "—"]) + long
        if rng.random() < 0.2:
            long = long[:-1] + "." + rng.choice(["Hope", "Never", "Smith", "Com", "NET"]) + "."
        sentences.append(long)
    return " ".join(sentences)


@pytest.mark.parametrize("seed", range(150))
def test_random_texts_with_joins_and_quotes(seed):
    text = _random_text_with_joins_and_quotes(random.Random(seed))
    for max_chars, digit_weight in ((350, 1), (350, 4), (120, 1)):
        pieces = _pieces(_split(text, max_chars=max_chars, digit_weight=digit_weight))
        assert all(0 < len(piece) <= max_chars for piece in pieces)
        assert "".join("".join(pieces).split()) == "".join(text.split())     # no character lost or added
        assert _split(text, max_chars=max_chars, digit_weight=digit_weight, avoid_short_final=False) == \
            _old_split_long_text(text, max_chars=max_chars, digit_weight=digit_weight)
    # the move itself, at every "." / "!" / "?" + space of the text taken as the plain cut: a moved cut lands right
    # after a terminator and before whitespace (never inside "Pope.Hope"), the chunk it leaves does not end short, and
    # it never moves exactly one word in front of a quotation or a dialogue dash (a plain cut can still open a chunk
    # there, e.g. after a cascade: that is the old splitter's own cut)
    for split_at in (index + 1 for index, char in enumerate(text) if char in ".!?" and text[index + 1:index + 2] == " "):
        cut = _cut_before_short_final_sentence(text, split_at, 0)
        if cut != split_at:
            moved = text[cut:split_at].split()
            assert text[cut - 1] in ".!?" and text[cut:cut + 1].isspace() and not ends_on_short_sentence(text[:cut])
            assert not (len(moved) == 1 and text[split_at:].lstrip()[:1] in "\"„«—"), (text[cut - 30:split_at + 10], seed)


# --- the short-piece merge (MERGE_SHORT_PIECES / MERGE_SHORT_CLIENT_SEGMENTS, opt-in) -----------------------------
def test_merge_joins_a_short_final_sentence_piece_to_the_next_piece():
    pieces = [LEAD, "No.", NEXT]
    assert _old_merge_short_pieces(pieces)[0] == [f"{LEAD} No.", NEXT]
    assert _merge(pieces)[0] == [LEAD, f"No. {NEXT}"]
    assert _merge(pieces, avoid_short_final=False)[0] == _old_merge_short_pieces(pieces)[0]
    # keys still decide: the next piece has other controls, so the short piece joins the previous one
    assert _merge(pieces, keys=[0, 0, 1]) == ([f"{LEAD} No.", NEXT], [0, 1])
    # the last piece has no next piece
    assert _merge([LEAD, "Okay."])[0] == [f"{LEAD} Okay."]


def test_merge_unchanged_for_short_pieces_without_a_short_final_sentence():
    for pieces in (["Short one here", "b" * 100 + "."], ["a" * 300 + ".", "Thanks a lot."], ["ab", "cd", "ef"]):
        assert _merge(pieces) == _old_merge_short_pieces(pieces)


# --- the serving paths: app.main (needs fastapi) ------------------------------------------------------------------
@pytest.fixture
def main(monkeypatch):
    pytest.importorskip("fastapi")
    import app.main as main
    monkeypatch.setattr(main, "MERGE_SHORT_PIECES", False)
    monkeypatch.setattr(main, "MERGE_SHORT_CLIENT_SEGMENTS", False)
    monkeypatch.setattr(main, "NO_SHORT_FINAL_SENTENCE", True)
    return main


def test_split_with_controls_applies_the_rule_and_the_flag_rolls_back(main, monkeypatch):
    text = f"{LEAD} I know. {NEXT}"
    assert main._split_with_controls(text, None) == ([LEAD, f"I know. {NEXT}"], None)
    assert main._split_with_controls(text, None, digit_weight=4) == ([LEAD, f"I know. {NEXT}"], None)
    controls = [{"rate": 1}, {"rate": 2}]
    assert main._split_with_controls(["Hi there.", text], controls) == (
        ["Hi there.", LEAD, f"I know. {NEXT}"], [{"rate": 1}, {"rate": 2}, {"rate": 2}])
    monkeypatch.setattr(main, "NO_SHORT_FINAL_SENTENCE", False)
    assert main._split_with_controls(text, None) == (_old_split_long_text(text), None)
    assert main._split_with_controls(text, None, digit_weight=4) == (
        _old_split_long_text(text, digit_weight=4), None)


def test_split_with_controls_with_the_merge_flags(main, monkeypatch):
    monkeypatch.setattr(main, "MERGE_SHORT_PIECES", True)
    text = f"{LEAD} Wow. {NEXT}"
    assert main._split_with_controls(text, None) == ([LEAD, f"Wow. {NEXT}"], None)
    monkeypatch.setattr(main, "MERGE_SHORT_CLIENT_SEGMENTS", True)
    same = [{"rate": 1}] * 3
    assert main._split_with_controls([LEAD, "No.", NEXT], same) == (
        [LEAD, f"No. {NEXT}"], [{"rate": 1}, {"rate": 1}])
    monkeypatch.setattr(main, "NO_SHORT_FINAL_SENTENCE", False)
    assert main._split_with_controls([LEAD, "No.", NEXT], same) == (
        [f"{LEAD} No.", NEXT], [{"rate": 1}, {"rate": 1}])


class _Captured(Exception):
    pass


class _FakeEngine:
    vv_tokenizer = object()
    verbalizer = None

    def __init__(self):
        self.texts = []

    async def generate(self, text, **kwargs):
        self.texts.append(text)
        raise _Captured

    async def generate_streaming(self, text, **kwargs):
        self.texts.append(text)
        return
        yield  # an async generator that yields nothing


def test_predict_path_sends_the_moved_chunks_to_the_engine(main, monkeypatch):
    engine = _FakeEngine()

    async def get_engine():
        return engine

    monkeypatch.setattr(main, "_get_engine", get_engine)
    with pytest.raises(_Captured):
        asyncio.run(main._synthesize(main.TTSRequest(text=f"{LEAD} No. {NEXT}")))
    assert engine.texts == [[LEAD, f"No. {NEXT}"]]


@pytest.mark.asyncio
async def test_stream_path_sends_the_moved_chunks_to_the_engine(main, monkeypatch):
    import httpx
    monkeypatch.setenv("ENABLE_STREAMING", "1")
    engine = _FakeEngine()

    async def get_engine():
        return engine

    monkeypatch.setattr(main, "_get_engine", get_engine)
    monkeypatch.setattr(main, "_unload_engine", lambda: None)
    application = main.create_app()
    async with application.router.lifespan_context(application):
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="http://testserver")
        response = await client.post("/stream", json={"text": f"{LEAD} No. {NEXT}"})
    assert response.status_code == 200 and '"done": true' in response.text
    assert engine.texts == [[LEAD, f"No. {NEXT}"]]


# --- the flag -----------------------------------------------------------------------------------------------------
def test_flag_default_on_and_rollback(monkeypatch):
    if os.environ.get("NO_SHORT_FINAL_SENTENCE"):
        pytest.skip("NO_SHORT_FINAL_SENTENCE set in the environment")
    assert config_module.NO_SHORT_FINAL_SENTENCE is True
    for function in (split_long_text, merge_short_pieces):
        assert inspect.signature(function).parameters["avoid_short_final"].default is True
    monkeypatch.setenv("NO_SHORT_FINAL_SENTENCE", "0")
    assert config_module._env_flag("NO_SHORT_FINAL_SENTENCE", True) is False
    monkeypatch.setenv("NO_SHORT_FINAL_SENTENCE", "1")
    assert config_module._env_flag("NO_SHORT_FINAL_SENTENCE", True) is True
