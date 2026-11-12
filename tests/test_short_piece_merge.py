"""Short-piece merge after split_long_text."""
import re

import pytest

from app.text import merge_short_pieces, sanitize_text_for_tts, split_long_text

# client lists, as serving sees them after sanitize_text_for_tts (which ends each in punctuation)
CITATION = [sanitize_text_for_tts(s) for s in
            ["Notes on imaginary rivers. 12-4. 163-169", "Entry 48213907", "Shelf 5903172"]]
CARGO = [sanitize_text_for_tts(s) for s in
         ["Camping supplies, 585 kg – 1290 lbs", "Repair tools, 5 kg – 11 lbs"]]
CITATION_MERGED = "Notes on imaginary rivers. 12-4. 163-169. Entry 48213907. Shelf 5903172."
TUTORIAL = ["When you want to turn the highlight off, you click it again.", "Okay."]
# 695 chars that split_long_text cuts into 340 + 340 + a 13-char remainder
SENTENCE_A = "a" * 169 + ". " + "b" * 168 + "."
SENTENCE_B = "c" * 169 + ". " + "d" * 168 + "."
LONG_TEXT = SENTENCE_A + " " + SENTENCE_B + " Thanks a lot."


def test_client_list_examples():
    assert merge_short_pieces(CITATION)[0] == [CITATION_MERGED]
    assert merge_short_pieces(CARGO)[0] == [
        "Camping supplies, 585 kg – 1290 lbs. Repair tools, 5 kg – 11 lbs."]
    assert merge_short_pieces(TUTORIAL)[0] == [
        "When you want to turn the highlight off, you click it again. Okay."]
    assert merge_short_pieces(["Oh, the old mill had two doors and a bell."])[0] == [
        "Oh, the old mill had two doors and a bell."]


def test_short_final_remainder_joins_previous_piece():
    pieces = split_long_text(LONG_TEXT)
    assert pieces == [SENTENCE_A, SENTENCE_B, "Thanks a lot."]
    merged, _ = merge_short_pieces(pieces)
    assert merged == [SENTENCE_A, SENTENCE_B + " Thanks a lot."]


def test_joins_next_when_previous_would_exceed_400_chars():
    long_piece = "a" * 390 + "."
    merged, _ = merge_short_pieces([long_piece, "Short one", "b" * 100 + "."])
    assert merged == [long_piece, "Short one " + "b" * 100 + "."]
    # neither neighbour fits: kept
    assert merge_short_pieces([long_piece, "Short one", long_piece])[0] == [
        long_piece, "Short one", long_piece]


def test_repeats_until_stable_and_keeps_keys_aligned():
    merged, keys = merge_short_pieces(["ab", "cd", "ef", "x" * 60], keys=[0, 0, 0, 1])
    assert merged == ["ab cd ef", "x" * 60] and keys == [0, 1]
    # neighbours with different keys never join
    assert merge_short_pieces(["ab", "cd"], keys=[0, 1]) == (["ab", "cd"], [0, 1])


@pytest.mark.parametrize("text", [
    "lorem ipsum " * 30 + "dolor sit amet",                    # one clause, cut at a space
    "the 12 cats and 7 dogs " * 15 + "ran home 42 times",        # ends in digits
    "Short intro. " + "word " * 70 + "tail end",                 # a real sentence end stays one
])
def test_merge_never_puts_a_full_stop_inside_a_clause(text):
    pieces = split_long_text(text)
    assert isinstance(pieces, list) and len(pieces[-1]) < 50 and pieces[-2][-1].isalnum()
    merged, _ = merge_short_pieces(pieces)
    assert merged == [" ".join(pieces)] and merged[0].count(". ") == text.count(". ")
    assert merged[0].split() == text.split()                     # the original text, nothing added


def test_digit_weight_counts_toward_the_limit():
    pieces = ["a" * 330 + ".", "12345678901234567890"]
    assert len(merge_short_pieces(pieces)[0]) == 1
    assert len(merge_short_pieces(pieces, digit_weight=4)[0]) == 2   # 331 + 1 + 4 x 20 > 400


# ---------------------------------------------------------------------------
# Serving path: app.main._split_with_controls (needs fastapi)
# ---------------------------------------------------------------------------

@pytest.fixture
def main(monkeypatch):
    pytest.importorskip("fastapi")
    import app.main as main
    monkeypatch.setattr(main, "MERGE_SHORT_PIECES", True)
    monkeypatch.setattr(main, "MERGE_SHORT_CLIENT_SEGMENTS", False)
    return main


def test_default_merges_pieces_of_one_text_only(main):
    assert main._split_with_controls(LONG_TEXT, None) == (
        [SENTENCE_A, SENTENCE_B + " Thanks a lot."], None)
    # a client list stays as sent, but a long client segment's own pieces merge
    assert main._split_with_controls(CITATION, None) == (CITATION, None)
    controls = [{"rate": 1}, {"rate": 2}]
    assert main._split_with_controls(["Hi there.", LONG_TEXT], controls) == (
        ["Hi there.", SENTENCE_A, SENTENCE_B + " Thanks a lot."],
        [{"rate": 1}, {"rate": 2}, {"rate": 2}],
    )
    assert main._split_with_controls("Short.", None) == ("Short.", None)


def test_client_segment_merge_needs_identical_controls(main, monkeypatch):
    monkeypatch.setattr(main, "MERGE_SHORT_CLIENT_SEGMENTS", True)
    assert main._split_with_controls(CITATION, None) == ([CITATION_MERGED], None)
    same = [{"rate": 1}] * 3
    assert main._split_with_controls(CITATION, same) == ([CITATION_MERGED], [{"rate": 1}])
    mixed = [{"rate": 1}, {"rate": 1}, {"rate": 3}]
    assert main._split_with_controls(CITATION, mixed) == (
        ["Notes on imaginary rivers. 12-4. 163-169. Entry 48213907.", "Shelf 5903172."],
        [{"rate": 1}, {"rate": 3}],
    )


def test_split_clause_round_trips_through_the_serving_merge(main):
    text = "lorem ipsum " * 30 + "dolor sit amet."
    assert main._split_with_controls(text, None) == (text, None)    # merged back to the original


@pytest.mark.parametrize("text, controls", [
    (LONG_TEXT, None),
    ("Short.", None),
    (CITATION, None),
    (CITATION + [LONG_TEXT], None),
    (CITATION + [LONG_TEXT], [{"rate": 0}, None, "<|ctrl_rate_04|>", {"pause": 1}]),
])
def test_flags_off_parity(main, monkeypatch, text, controls):
    monkeypatch.setattr(main, "MERGE_SHORT_PIECES", False)
    if controls is None:
        expected = (split_long_text(text), None)
    else:
        pieces, entries = [], []
        for segment, value in zip(text, controls):
            split = split_long_text([segment])
            pieces.extend(split)
            entries.extend([value] * len(split))
        expected = (pieces, entries)
    assert main._split_with_controls(text, controls) == expected
    assert main._split_with_controls(text, controls, digit_weight=4)[0] == split_long_text(
        text, digit_weight=4)


# ---------------------------------------------------------------------------
# Rate-aware chunk length: rate 0 scales the split length and merge limit by 0.70
# ---------------------------------------------------------------------------

SLOWEST_MEASURED_CPS = 9.7   # rate 0: first-attempt median chars/s, the slowest bin
SHERLOCK = (
    "My own complete happiness, and the home-centred interests which rise up around the man who first "
    "finds himself master of his own establishment, were sufficient to absorb all my attention, while "
    "Holmes, who loathed every form of society with his whole Bohemian soul, remained in our lodgings in "
    "Baker Street, buried among his old books, and alternating from week to week between cocaine and "
    "ambition, the drowsiness of the drug, and the fierce energy of his own keen nature. "
)


def test_per_piece_merge_limits():
    pieces = ["a" * 250 + ".", "Short one."]
    assert merge_short_pieces(pieces, max_chars=[280, 280])[0] == [pieces[0] + " Short one."]
    assert merge_short_pieces(pieces, max_chars=[250, 280])[0] == pieces


@pytest.mark.parametrize("text", [LONG_TEXT, SHERLOCK * 4, "word " * 400, ("clause, " * 120) + "end."])
def test_rate0_pieces_fit_the_token_cap(main, text):
    from app.config import MAX_SEMANTIC_TOKENS_PER_CHUNK
    pieces, _ = main._split_with_controls(text, None, chunk_scales=[0.70])
    pieces = [pieces] if isinstance(pieces, str) else pieces
    assert max(len(p) for p in pieces) <= 280
    assert all(len(p) / SLOWEST_MEASURED_CPS * 12.5 <= MAX_SEMANTIC_TOKENS_PER_CHUNK for p in pieces)
    assert re.findall(r"[\w'-]+", " ".join(pieces)) == re.findall(r"[\w'-]+", text)   # no words lost
    # without scaling the same text gives pieces over the rate-0 budget
    unscaled, _ = main._split_with_controls(text, None)
    unscaled = [unscaled] if isinstance(unscaled, str) else unscaled
    assert max(len(p) for p in unscaled) > 280 or len(unscaled) == 1


def test_client_segments_split_by_their_own_rate(main):
    text = [SHERLOCK * 2, SHERLOCK * 2]
    pieces, controls = main._split_with_controls(text, [{"rate": 0}, {"rate": 4}], chunk_scales=[0.70, 1.0])
    slow = [p for p, c in zip(pieces, controls) if c == {"rate": 0}]
    fast = [p for p, c in zip(pieces, controls) if c == {"rate": 4}]
    assert max(map(len, slow)) <= 280 and max(map(len, fast)) > 280
    assert fast == main._split_with_controls([SHERLOCK * 2], None)[0]


@pytest.mark.parametrize("text, controls", [
    (SHERLOCK * 3, None),
    ([SHERLOCK * 2, "Okay.", LONG_TEXT], None),
    ([SHERLOCK * 2, "Okay.", LONG_TEXT], [{"rate": 0}, None, {"rate": 4}]),
])
def test_unit_scales_are_parity(main, text, controls):
    n = len(text) if isinstance(text, list) else 1
    assert main._split_with_controls(text, controls, chunk_scales=[1.0] * n) == \
        main._split_with_controls(text, controls)
    assert main._split_with_controls(text, controls, digit_weight=4, chunk_scales=[1.0] * n) == \
        main._split_with_controls(text, controls, digit_weight=4)


def test_chunk_scales_from_request(main):
    from types import SimpleNamespace
    from app.config import ServerConfig
    from app.controls import ControlCatalog, ControlSpec
    from app.inference import TTSEngine
    config = ServerConfig()
    config.cb0_rate_aware_chunk_chars = True
    config.cb0_rate_cps_factors_json = None
    engine = TTSEngine(config)
    engine.controls = ControlCatalog([ControlSpec("rate", 5)])
    req = SimpleNamespace(language="english", tag="audiobook", use_system_tag=True, controls={"rate": 0})
    assert main._chunk_scales(req, engine, SHERLOCK, None) == pytest.approx([0.70])
    assert main._chunk_scales(req, engine, ["a", "b"], [None, {"rate": 4}]) == pytest.approx([0.70, 1.0])
    config.cb0_rate_aware_chunk_chars = False
    assert main._chunk_scales(req, TTSEngine(config), SHERLOCK, None) is None
