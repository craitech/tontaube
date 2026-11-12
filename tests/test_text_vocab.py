"""The character set the codebook models read and the text rows they need."""
import unicodedata

import pytest

from app.model.text_vocab import ALLOWED_CHARS, STRUCTURAL_TOKENS, clean_text, strip_markup, text_token_rows
from app.text import sanitize_text_for_tts
from app.utils import _char_token_cache, tokenize_text


class FakeTokenizer:
    """Characters get ids 1000+codepoint; BPE splits on spaces; structural tokens fixed ids."""

    unk_token_id = 0
    special = {t: 151640 + i for i, t in enumerate(STRUCTURAL_TOKENS)}

    def __call__(self, text):
        return {"input_ids": [1000 + ord(c) for c in text]}

    def convert_tokens_to_ids(self, token):
        return self.special.get(token, self.unk_token_id)

    def encode(self, text, add_special_tokens=False):
        return [500000 + hash(w) % 1000 for w in text.split(" ") if w]


@pytest.fixture(autouse=True)
def _fresh_char_cache():
    _char_token_cache.clear()
    yield
    _char_token_cache.clear()


LANGUAGE_SAMPLES = {
    "english": "\"Don't,\" she said — it's naïve; the café's 1st floor costs $20 (50%).",
    "german": "„Ärger über Öl?“ – »Größe!« Straße, ÄÖÜ, § 3, 20 °C, 5 € …",
    "french": "« Où êtes-vous ? » Le cœur, l'œuvre, Noël, garçon, Ève, À bientôt ; 1er, 2e.",
    "spanish": "¿Qué tal? ¡Olé! El niño, el pingüino, el 1.º y la 2.ª; Ángel, ½ kilo.",
    "italian": "Perché così? Città, più, però, È già l'età dell'oro.",
    "dutch": "Één coördinatie, zo'n ruïne; 's morgens ‘vlug’ naar „het” café.",
    "portuguese": "São Paulo: corações, ações, à noite; Ângela, Inês, pôr, avô, nº 1.",
}


@pytest.mark.parametrize("language", sorted(LANGUAGE_SAMPLES))
def test_texts_of_the_supported_languages_pass_unchanged(language):
    text = LANGUAGE_SAMPLES[language]
    assert clean_text(text) == unicodedata.normalize("NFC", text)


def test_allowed_chars_cover_the_supported_languages():
    for word in ("Straße", "Ölgemälde", "garçon", "œuvre", "niño", "¿Qué?", "città", "perché", "coração",
                 "Ação", "één", "naïve", "ÉCOLE", "Ÿ", "€ 5,00", "20 °C", "§ 3", "1.º", "2.ª", "¾", "{x}"):
        assert all(c in ALLOWED_CHARS for c in word), word
    for ch in ("<", ">", "|", "\\", "^", "~", "\N{CJK UNIFIED IDEOGRAPH-4F60}",
               "\N{CYRILLIC SMALL LETTER YA}", "\N{GREEK SMALL LETTER ALPHA}", "\N{COMBINING ACUTE ACCENT}"):
        assert ch not in ALLOWED_CHARS
    assert all(unicodedata.normalize("NFC", c) == c for c in ALLOWED_CHARS)


def test_foreign_letters_fall_back_to_readable_forms():
    assert clean_text("Dvo\N{LATIN SMALL LETTER R WITH CARON}\N{LATIN SMALL LETTER A WITH ACUTE}k") == "Dvorák"
    assert clean_text("\N{LATIN CAPITAL LETTER L WITH STROKE}\N{LATIN SMALL LETTER O WITH ACUTE}d\N{LATIN SMALL LETTER Z WITH ACUTE}") == "Lódz"
    assert clean_text("S\N{LATIN SMALL LETTER O WITH STROKE}ren \N{LATIN CAPITAL LETTER A WITH RING ABOVE}ngstr\N{LATIN SMALL LETTER O WITH DIAERESIS}m") == "Soren Angström"
    assert clean_text("Erdo\N{LATIN SMALL LETTER G WITH BREVE}an \N{LATIN CAPITAL LETTER I WITH DOT ABOVE}stanbul \N{LATIN CAPITAL LETTER THORN}\N{LATIN SMALL LETTER O WITH ACUTE}r") == "Erdogan Istanbul Thór"
    # compatibility forms: superscript letters, ligatures, fullwidth, Roman numerals; footnote digits dropped
    assert clean_text("2\N{MODIFIER LETTER SMALL E} \N{LATIN SMALL LIGATURE IJ}s \N{FULLWIDTH LATIN CAPITAL LETTER A}\N{ROMAN NUMERAL FOUR}") == "2e ijs AIV"
    assert clean_text("H\N{SUBSCRIPT TWO}O, voir\N{SUPERSCRIPT ONE}, 120 m\N{SUPERSCRIPT TWO}") == "H2O, voir, 120 m"
    assert clean_text("\N{VULGAR FRACTION ONE THIRD} \N{TRADE MARK SIGN}") == "1/3 TM"
    assert clean_text("\N{GRINNING FACE}\N{CJK UNIFIED IDEOGRAPH-4F60}\N{CYRILLIC SMALL LETTER PE}") == ""


def test_strip_markup_and_sanitize():
    assert strip_markup("a <<laugh>> b || c <|im_start|>x<|im_end|>") == "a  b  c x"
    assert sanitize_text_for_tts("Hi <<sigh>> there, \N{CJK UNIFIED IDEOGRAPH-4F60}\N{NO-BREAK SPACE}friend") == "Hi there, friend."
    assert sanitize_text_for_tts("你好") == ""


def test_tokenize_text_only_reads_trained_rows():
    tok = FakeTokenizer()
    ids = tokenize_text("<|im_start|>german : audiobook<|im_end|>\nAb<|text_split|>c", tok)
    assert ids[0] == tok.special["<|im_start|>"] and tok.special["<|im_end|>"] in ids
    assert tok.special["<|text_split|>"] in ids
    chars = [i for i in ids if i not in tok.special.values() and i < 500000]
    assert chars == [1000 + ord(c) for c in "\nAbc"]
    # untrained markup, unknown control tokens and unseen characters are dropped
    assert tokenize_text("a<<laugh>>b||c<|vision_start|>d你", tok) == [1000 + ord(c) for c in "abcd"]


def test_text_token_rows_cover_everything_tokenize_text_produces():
    tok = FakeTokenizer()
    rows = set(text_token_rows(tok))
    assert {1000 + ord(c) for c in ALLOWED_CHARS} <= rows
    assert set(tok.special.values()) <= rows
    sample = "<|im_start|>dutch : agentic<|im_end|>\nZo'n €5,- (ca.) — „gut“ <<x>> 你"
    assert set(tokenize_text(sample, tok)) <= rows


def test_api_rejects_text_without_readable_characters():
    from app.main import TTSRequestOptionError, _prepare_raw_text

    with pytest.raises(TTSRequestOptionError):
        _prepare_raw_text("你好 <<laugh>>")
    with pytest.raises(TTSRequestOptionError):
        _prepare_raw_text(["…", "—"])
    assert _prepare_raw_text("Hallo 你好") == "Hallo 你好."


def test_tokenize_text_maps_server_rendered_model_tokens():
    """v1.1 bundles: sliders in CB0's header and the end-token windows reach the model as their ids; the same
    strings in user text are stripped by the sanitizer."""
    tok = FakeTokenizer()
    tok.special = {**FakeTokenizer.special, "<|ctrl_rate_03|>": 180348, "<|utt_end|>": 180379}
    ids = tokenize_text("<|im_start|>english : audiobook<|ctrl_rate_03|><|im_end|>\nHi.<|utt_end|>", tok)
    assert 180348 in ids and 180379 in ids
    assert ids.index(180348) < ids.index(tok.special["<|im_end|>"])
    assert tokenize_text("a<|ctrl_rate_3|>b<|ctrl_RATE_03|>c", tok) == [1000 + ord(c) for c in "abc"]
    assert sanitize_text_for_tts("Hi <|ctrl_rate_03|>there<|utt_end|>") == "Hi there."


def test_separators_are_not_read_as_text():
    from app.model.text_vocab import strip_separators

    rule = "-" * 72
    assert sanitize_text_for_tts(f"The end of a part.\n\n{rule}\n\nThe next part begins.") == \
        "The end of a part.\n\nThe next part begins."
    for line in ("* * *", "=====", "___", "~~~~", "#####", "- - -"):
        assert sanitize_text_for_tts(f"One.\n{line}\nTwo.") == "One.\n\nTwo.", line  # a section break
    assert strip_separators("I was---no, never.") == "I was\N{EM DASH}no, never."
    assert strip_separators("He paused -- then went on... slowly.") == "He paused -- then went on... slowly."
    assert strip_separators("Name: ________ please") == "Name:  please"
    assert strip_separators("a - b - c, x_y, 2*3") == "a - b - c, x_y, 2*3"


def test_api_rejects_text_of_separators_only():
    from app.main import TTSRequestOptionError, _prepare_raw_text

    with pytest.raises(TTSRequestOptionError):
        _prepare_raw_text("-" * 40)
