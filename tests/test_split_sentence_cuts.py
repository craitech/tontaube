"""SPLIT_SENTENCE_CUTS: the sentence-first cut of split_long_text no longer takes the last ". ! ?" character in its
window, which cut after abbreviations and ordinals ("am 3. | Oktober", "z. B. | die", "Dr. | Brown") and inside
decimals ("3. | 5"). It takes the last real sentence end (_sentence_cuts), else the old character search minus the
known non-ends; the comma tier never cuts inside a number.
"""
import importlib
import random

import pytest

import app.config as config_module
from app.text import _known_non_end, _last_sentence_cut, split_long_text


def _new(text, **kwargs):
    kwargs.setdefault("avoid_short_final", True)
    return split_long_text(text, sentence_cuts=True, **kwargs)


def _old(text, **kwargs):
    kwargs.setdefault("avoid_short_final", True)
    return split_long_text(text, sentence_cuts=False, **kwargs)


def _no_loss(text, pieces):
    return "".join(pieces).replace(" ", "") == text.replace(" ", "")


# (lead sentence, text up to and including the abbreviation / ordinal, continuation): 7 languages
CASES = {
    "de_ordinal": ("Wir haben uns lange über den Umbau unterhalten.",
                   "Dann haben wir uns entschieden, am 3.",
                   "Oktober mit dem Umbau zu beginnen, weil die Handwerker vorher keine Zeit hatten und wir ohnehin "
                   "noch auf die Genehmigung der Stadtverwaltung warten mussten, die sich wie immer Zeit ließ."),
    "de_zb": ("Wir haben uns lange über den Umbau unterhalten.",
              "Wir brauchen für das neue Büro noch einige Dinge, z. B.",
              "die neuen Rechner, zwei Bildschirme, einen großen Tisch und genug Stühle für alle Kolleginnen und "
              "Kollegen, die ab dem nächsten Monat bei uns anfangen werden und sich schon darauf freuen."),
    "en_title": ("We talked about the results for a long time.",
                 "After the meeting we decided that we should call Dr.",
                 "Brown about the results, because he knows the patients well and has been working with the whole "
                 "team for several years now, which makes his opinion particularly valuable to all of us."),
    "es_title": ("Hablamos mucho tiempo sobre los resultados.",
                 "Ayer por la tarde hablamos por teléfono con la Sra.",
                 "López sobre los resultados del estudio, porque conoce muy bien a los pacientes y lleva varios años "
                 "trabajando con todo el equipo, lo que hace que su opinión sea muy valiosa para nosotros."),
    "fr_initial": ("Nous avons longuement parlé des résultats.",
                   "Hier après-midi, nous avons rencontré M.",
                   "Dupont pour discuter des résultats de l'étude, parce qu'il connaît très bien les patients et "
                   "travaille avec toute l'équipe depuis plusieurs années, ce qui rend son avis très précieux."),
    "it_title": ("Abbiamo parlato a lungo dei risultati.",
                 "Ieri pomeriggio abbiamo parlato a lungo con il Dott.",
                 "Rossi dei risultati dello studio, perché conosce molto bene i pazienti e lavora con tutta la "
                 "squadra da diversi anni, il che rende la sua opinione particolarmente preziosa per tutti noi."),
    "nl_title": ("We hebben lang over de resultaten gesproken.",
                 "Gisteren hebben we uitgebreid gesproken met Dhr.",
                 "Jansen over de resultaten van het onderzoek, omdat hij de patiënten goed kent en al jaren met het "
                 "hele team samenwerkt, wat zijn mening bijzonder waardevol maakt voor ons allemaal."),
    "pt_title": ("Falamos muito tempo sobre os resultados.",
                 "Ontem à tarde falamos por telefone com o Sr.",
                 "Silva sobre os resultados do estudo, porque ele conhece muito bem os pacientes e trabalha com toda "
                 "a equipe há vários anos, o que torna a sua opinião especialmente valiosa para todos nós."),
}


def _build(lead, before, after, target=300):
    """Leads, then ``before`` so that its final "." lands near ``target`` (inside the 350-char window), then ``after``."""
    head = lead
    while len(head) + 1 + len(before) < target:
        head += " " + lead
    text = f"{head} {before} {after}"
    assert 250 <= len(head) + 1 + len(before) <= 345 and len(text) > 350
    return text, head


@pytest.mark.parametrize("name", sorted(CASES))
def test_no_cut_after_an_abbreviation_or_ordinal_in_7_languages(name):
    lead, before, after = CASES[name]
    text, head = _build(lead, before, after)
    old, new = _old(text), _new(text)
    assert old[0].endswith(before)                                       # the bug: a cut right after "3." / "Dr."
    assert new[0] == head                                                 # the fix: the last real sentence end
    assert all(len(piece) <= 350 for piece in new) and _no_loss(text, new)


@pytest.mark.parametrize("tail, bad_end", [
    ("Revenue grew by 3.5 percent in the last quarter, which was much more than anyone on the board had expected "
     "after the weak spring and the long strike at the main plant", "3."),
    ("All the documents are hosted on example.com and can be downloaded by every member of the team at any time of "
     "the day, also from home or while travelling for work abroad", "example."),
])
def test_no_cut_inside_a_decimal_or_a_domain(tail, bad_end):
    lead = "We talked about the results for a long time."
    head = " ".join([lead] * 6)
    text = f"{head} {tail}."
    assert _old(text)[0].endswith(bad_end)
    new = _new(text)
    assert new[0] == head and _no_loss(text, new)


def test_closing_quote_stays_with_its_sentence():
    lead = "We talked about the results for a long time."
    text = (" ".join([lead] * 6) + ' He said, "I am delighted with it." Then we went home together and talked about '
            'everything that had happened during the long and tiring day at the office')
    old, new = _old(text), _new(text)
    assert old[0].endswith("with it.") and old[1].startswith('"')       # the old cut left a stray closing quote
    assert new[0].endswith('with it."') and not new[1].startswith('"')


def test_dialogue_tag_is_not_split_off():
    lead = "We talked about the results for a long time."
    text = (" ".join([lead] * 6) + ' "What is the adventure?" demanded Conny, and everyone laughed about it for a long '
            'while before they finally sat down at the table and started to eat their dinner together')
    new = _new(text)
    assert new[0] == " ".join([lead] * 6) and new[1].startswith('"What is the adventure?" demanded')


def test_ellipsis_and_caption_join_still_cut_without_a_real_sentence_end():
    """No real sentence end in the window: the old character search, minus known non-ends, still applies."""
    words = "and then we kept on talking about all the things that happened during the day "
    body = (words * 4)[:290]
    for joint, after in (("... ", "Then we went home"), (".", "It's late now and we went home"),
                         ('..." ', "and then we went home")):                # the cut keeps the closing quote
        text = body + "things" + joint + after + " " + words * 2
        new = _new(text)
        assert new[0].endswith("things" + joint.strip()) and _no_loss(text, new)


def test_comma_tier_never_cuts_inside_a_number():
    words = "and then we kept on talking about all the things that happened during the day "
    text = (words * 4)[:300] + " for $5,000 a month and more " + words * 2
    old, new = _old(text), _new(text)
    assert old[0].endswith("$5,")                                          # the old comma tier: "$5," | "000"
    assert not new[0].endswith("$5,") and "$5,000" in " ".join(new) and _no_loss(text, new)


@pytest.mark.parametrize("text, index, expected", [
    ("am 3. Oktober", 4, True), ("z. B. die", 1, True), ("z. B. die", 4, True), ("call Dr. Brown", 7, True),
    ("grew 3.5 percent", 6, True), ("see example.com now", 11, True), ("the U.S. Army", 7, True),
    ("Dr.Smith came", 2, True), ("at the end. Then", 10, False), ("wait... Then", 6, False),
    ("operating room.And now", 14, False), ("this day.It's late", 8, False), ("Really? yes", 6, False),
])
def test_known_non_end(text, index, expected):
    assert _known_non_end(text, index) is expected


def test_last_sentence_cut_window():
    text = "One sentence here. Another one. Dr. Brown came"
    assert _last_sentence_cut(text, 0, len(text)) == len("One sentence here. Another one.")
    assert _last_sentence_cut(text, 25, len(text)) == len("One sentence here. Another one.")
    assert _last_sentence_cut(text, 31, len(text)) == -1                  # "Dr." is no cut, nothing else after 31


def test_flag_default_on_and_env_off(monkeypatch):
    monkeypatch.delenv("SPLIT_SENTENCE_CUTS", raising=False)
    try:
        assert importlib.reload(config_module).SPLIT_SENTENCE_CUTS is True
        monkeypatch.setenv("SPLIT_SENTENCE_CUTS", "0")
        assert importlib.reload(config_module).SPLIT_SENTENCE_CUTS is False
    finally:
        monkeypatch.delenv("SPLIT_SENTENCE_CUTS", raising=False)
        importlib.reload(config_module)


def test_random_texts_keep_limits_and_never_cut_at_a_known_non_end():
    rng = random.Random(8)
    parts = ["The meeting ended.", "Dr. Brown called.", "Sales grew 3.5 percent,", "am 3. Oktober", "z. B. die Rechner",
             '"I agree," she said.', "We waited...", "Then we left!", "on example.com today", "about $5,000 a month",
             "really?", "St. Louis is far;", "and so on", "It's late.Okay then.", "M. Dupont arrived.", "Nr. 5 fehlt."]
    for _ in range(300):
        text = " ".join(rng.choice(parts) for _ in range(rng.randint(20, 90)))
        for digit_weight in (1, 4):
            for avoid in (True, False):
                pieces = _new(text, digit_weight=digit_weight, avoid_short_final=avoid)
                pieces = [pieces] if isinstance(pieces, str) else pieces
                assert _no_loss(text, pieces)
                assert all(len(p) + (digit_weight - 1) * sum(c.isdigit() for c in p) <= 350 for p in pieces[:-1])
                for piece in pieces[:-1]:
                    assert not piece.endswith(("Dr.", " 3.", "z. B.", "St.", " M.", "Nr.", "$5,", " 3.")), piece
