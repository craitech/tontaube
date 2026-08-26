"""Tests for whitespace / newline handling in sanitize_text_for_tts."""
import pytest

from app.config import MAX_TEXT_CHARS_PER_CHUNK
from app.text import sanitize_text_for_tts, split_long_text


# ---------------------------------------------------------------------------
# Newline preservation
# ---------------------------------------------------------------------------

class TestNewlinePreservation:
    def test_single_newline_preserved(self):
        assert sanitize_text_for_tts("hello\nworld") == "hello\nworld."

    def test_double_newline_preserved(self):
        assert sanitize_text_for_tts("hello\n\nworld") == "hello\n\nworld."

    def test_triple_newline_collapsed_to_double(self):
        assert sanitize_text_for_tts("hello\n\n\nworld") == "hello\n\nworld."

    def test_many_newlines_collapsed_to_double(self):
        assert sanitize_text_for_tts("hello\n\n\n\n\nworld") == "hello\n\nworld."


# ---------------------------------------------------------------------------
# Newlines mixed with spaces / tabs
# ---------------------------------------------------------------------------

class TestNewlineWithSpaces:
    def test_newline_surrounded_by_spaces(self):
        assert sanitize_text_for_tts("hello \n world") == "hello\nworld."

    def test_double_newline_with_spaces(self):
        assert sanitize_text_for_tts("hello \n \n world") == "hello\n\nworld."

    def test_tabs_around_newline(self):
        assert sanitize_text_for_tts("hello\t\n\tworld") == "hello\nworld."

    def test_tabs_around_double_newline(self):
        assert sanitize_text_for_tts("hello\t\n\t\n world") == "hello\n\nworld."


# ---------------------------------------------------------------------------
# Space-only collapsing (no newlines)
# ---------------------------------------------------------------------------

class TestSpaceCollapsing:
    def test_multiple_spaces(self):
        assert sanitize_text_for_tts("hello   world") == "hello world."

    def test_tabs_collapse_to_space(self):
        assert sanitize_text_for_tts("hello\t\tworld") == "hello world."

    def test_mixed_spaces_and_tabs(self):
        assert sanitize_text_for_tts("hello \t world") == "hello world."


# ---------------------------------------------------------------------------
# Exotic / vertical whitespace mapped to newline
# ---------------------------------------------------------------------------

class TestExoticWhitespace:
    def test_carriage_return(self):
        assert sanitize_text_for_tts("hello\rworld") == "hello\nworld."

    def test_crlf(self):
        # \r -> \n + \n -> two newlines -> \n\n
        assert sanitize_text_for_tts("hello\r\nworld") == "hello\n\nworld."

    def test_form_feed(self):
        assert sanitize_text_for_tts("hello\fworld") == "hello\nworld."

    def test_vertical_tab(self):
        assert sanitize_text_for_tts("hello\vworld") == "hello\nworld."

    def test_unicode_line_separator(self):
        assert sanitize_text_for_tts("hello\u2028world") == "hello\nworld."

    def test_unicode_paragraph_separator(self):
        assert sanitize_text_for_tts("hello\u2029world") == "hello\nworld."

    def test_next_line_nel(self):
        assert sanitize_text_for_tts("hello\x85world") == "hello\nworld."


# ---------------------------------------------------------------------------
# Stripping and trailing punctuation
# ---------------------------------------------------------------------------

class TestStrippingAndPunctuation:
    def test_leading_newline_stripped(self):
        assert sanitize_text_for_tts("\nhello") == "hello."

    def test_trailing_newline_stripped(self):
        assert sanitize_text_for_tts("hello\n") == "hello."

    def test_leading_and_trailing_newlines_stripped(self):
        assert sanitize_text_for_tts("\n\nhello\n\n") == "hello."

    def test_no_extra_period_if_punctuated(self):
        assert sanitize_text_for_tts("hello\nworld!") == "hello\nworld!"

    def test_no_extra_period_for_question_mark(self):
        assert sanitize_text_for_tts("line one\nline two?") == "line one\nline two?"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_string(self):
        assert sanitize_text_for_tts("") == ""

    def test_only_newlines(self):
        assert sanitize_text_for_tts("\n\n\n") == ""

    def test_only_spaces(self):
        assert sanitize_text_for_tts("   ") == ""

    def test_single_char(self):
        assert sanitize_text_for_tts("a") == "a."

    def test_paragraph_break(self):
        assert sanitize_text_for_tts("First.\n\nSecond.") == "First.\n\nSecond."

    def test_multiline_document(self):
        text = "Line one.\nLine two.\n\nNew paragraph."
        assert sanitize_text_for_tts(text) == "Line one.\nLine two.\n\nNew paragraph."


class TestSplitLongText:
    def test_short_string_is_unchanged(self):
        text = "A short sentence, with a clause."
        assert split_long_text(text) == text

    def test_uses_last_comma_before_limit(self):
        text = "a" * 200 + ", " + "b" * 180 + ", " + "c" * 80
        assert split_long_text(text) == [
            "a" * 200 + ",",
            "b" * 180 + ", " + "c" * 80,
        ]

    def test_exact_limit_is_unchanged(self):
        text = "a" * MAX_TEXT_CHARS_PER_CHUNK
        assert split_long_text(text) == text

    def test_does_not_search_past_limit_for_punctuation(self):
        text = "a" * (MAX_TEXT_CHARS_PER_CHUNK + 20) + ", " + "b" * 20
        assert split_long_text(text) == [
            "a" * MAX_TEXT_CHARS_PER_CHUNK,
            "a" * 20 + ", " + "b" * 20,
        ]

    def test_uses_whitespace_when_no_punctuation_precedes_limit(self):
        text = "a" * (MAX_TEXT_CHARS_PER_CHUNK - 10) + " " + "b" * 30
        assert split_long_text(text) == [
            "a" * (MAX_TEXT_CHARS_PER_CHUNK - 10),
            "b" * 30,
        ]

    def test_hard_splits_when_no_boundary_exists(self):
        text = "a" * (MAX_TEXT_CHARS_PER_CHUNK + 25)
        assert split_long_text(text) == [
            "a" * MAX_TEXT_CHARS_PER_CHUNK,
            "a" * 25,
        ]

    def test_can_reserve_four_units_per_digit(self):
        assert split_long_text(
            "abc1defgh",
            max_chars=8,
            digit_weight=4,
        ) == ["abc1d", "efgh"]

    def test_weighted_split_still_prefers_punctuation(self):
        assert split_long_text(
            "abc1, def",
            max_chars=8,
            digit_weight=4,
        ) == ["abc1,", "def"]

    def test_digit_weight_must_be_positive(self):
        with pytest.raises(ValueError, match="digit_weight must be positive"):
            split_long_text("123", digit_weight=0)

    def test_prefers_punctuation_over_nearer_whitespace(self):
        text = (
            "a" * (MAX_TEXT_CHARS_PER_CHUNK - 30)
            + ","
            + " b" * 25
        )
        parts = split_long_text(text)
        assert parts[0].endswith(",")

    def test_flattens_split_segments_in_existing_list(self):
        text = ["short", "a" * MAX_TEXT_CHARS_PER_CHUNK + ", trailing"]
        assert split_long_text(text) == [
            "short",
            "a" * MAX_TEXT_CHARS_PER_CHUNK,
            ", trailing",
        ]

    @pytest.mark.parametrize(
        "text",
        [
            "word " * 500,
            "a" * 1000,
            ("clause, " * 100) + "end",
        ],
    )
    def test_all_chunks_respect_limit(self, text):
        result = split_long_text(text)
        parts = [result] if isinstance(result, str) else result
        assert all(0 < len(part) <= MAX_TEXT_CHARS_PER_CHUNK for part in parts)

    def test_splits_sherlock_regression_sentence(self):
        text = (
            "My own complete happiness, and the home-centred interests which "
            "rise up around the man who first finds himself master of his own "
            "establishment, were sufficient to absorb all my attention, while "
            "Holmes, who loathed every form of society with his whole Bohemian "
            "soul, remained in our lodgings in Baker Street, buried among his "
            "old books, and alternating from week to week between cocaine and "
            "ambition, the drowsiness of the drug, and the fierce energy of his "
            "own keen nature."
        )
        parts = split_long_text(text)
        assert isinstance(parts, list)
        assert all(len(part) <= MAX_TEXT_CHARS_PER_CHUNK for part in parts)
        assert " ".join(parts) == text
