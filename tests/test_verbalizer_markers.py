"""Markers in the verbalizer's output: numbers and Roman numerals written out as words."""
import pytest

from app.verbalizer_markers import CLOSE, NUM, ORDINAL, ROMAN, expand, expand_copied, from_roman, to_roman


@pytest.mark.parametrize("text, spoken", [
    (f"Bookings were {NUM}27,040,500{CLOSE} dollars.", "Bookings were twenty-seven million forty thousand five hundred dollars."),
    (f"{NUM}918834{CLOSE} new cases.", "Nine hundred eighteen thousand eight hundred thirty-four new cases."),
    (f"Volume {ROMAN}XCIII{CLOSE}, Henry the {ORDINAL}VIII{CLOSE}, Scene {ROMAN}ii{CLOSE}.", "Volume Ninety-three, Henry the Eighth, Scene Two."),
    (f"It was {NUM}1,234,567.5678{CLOSE}. {NUM}1,000,000,000,000{CLOSE}!", "It was one million two hundred thirty-four thousand five hundred sixty-seven point five six seven eight. One trillion!"),
    (f"Year {ROMAN}MCMXCIV{CLOSE}, Louis the {ORDINAL}XXI{CLOSE}", "Year One thousand nine hundred ninety-four, Louis the Twenty-first"),
    ("No markers here.", "No markers here."),
])
def test_expand(text, spoken):
    assert expand(text) == spoken


@pytest.mark.parametrize("kind, copied", [(ROMAN, "IIII"), (ROMAN, "Xi"), (NUM, "012345"), (NUM, "1,23,456"),
                                          (NUM, "27.530,20")])
def test_marked_text_that_cannot_be_read_stays_as_written(kind, copied):
    assert expand(f"It was {kind}{copied}{CLOSE} then.") == f"It was {copied} then."


@pytest.mark.parametrize("text", [f"{NUM}12", f"a{CLOSE}", f"{NUM}{CLOSE}", f"{NUM}12{ROMAN}XI{CLOSE}"])
def test_broken_marker_gives_no_output(text):
    assert expand(text) is None


def test_roman_numerals_round_trip():
    assert all(from_roman(to_roman(n)) == n for n in range(1, 4000))


def test_marked_text_must_stand_whole_in_the_input():
    source = "It cost $1,500,000.50, as in Macbeth III.iv.12 of the series."
    assert expand_copied(source, f"Act {ROMAN}III{CLOSE}, scene {ROMAN}iv{CLOSE}") == "Act Three, scene Four"
    for wrong in ("1,500,001", "500,000", "1,500,00", "1,500"):   # changed or cut
        assert expand_copied(source, f"{NUM}{wrong}{CLOSE}") is None, wrong
    assert expand_copied(source, f"{NUM}1,500,000{CLOSE} dollars and fifty cents") == "One million five hundred thousand dollars and fifty cents"
    assert expand_copied(source, f"{NUM}1,500,000.50{CLOSE}") == "One million five hundred thousand point five zero"
    assert expand_copied("the 1,000,000th visitor", f"the {NUM}1,000,000{CLOSE}th visitor") == "the one millionth visitor"
    # Copied whole but not readable: only the number stays as written.
    assert expand_copied("It paid 1,23,456 rupees in 2025.", f"It paid {NUM}1,23,456{CLOSE} rupees in twenty twenty-five.") \
        == "It paid 1,23,456 rupees in twenty twenty-five."
    assert expand_copied("It paid 27.530,20 euros.", f"It paid {NUM}27.530{CLOSE} euros.") is None   # cut
