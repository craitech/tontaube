"""Tests for the verbalizer guard: the misread pre-pass and the output post-check."""
import asyncio
from unittest.mock import MagicMock

import pytest

from app.verbalizer_guard import Prepared, check_output, prepare


# ── Pre-pass ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text, spoken", [
    ("at 4:02 PM.", "at four oh two PM."),
    ("at 11:05 AM.", "at eleven oh five AM."),
    ("at 11:05.", "at eleven oh five."),
    ("at 8:00 AM.", "at eight AM."),
    ("at 3:45 PM.", "at three forty-five PM."),
    ("at 17:20.", "at seventeen twenty."),
    ("for 6:30.", "for six thirty."),
    ("at 07:45 from", "at seven forty-five from"),
    ("to 10:00.", "to ten o'clock."),
    ("at 16:00.", "at sixteen hundred."),
    ("at 12:30 p.m. tomorrow", "at twelve thirty PM tomorrow"),
    ("at 9:07pm.", "at nine oh seven PM."),
])
def test_clock_times(text, spoken):
    assert prepare(text).text == spoken


@pytest.mark.parametrize("text", [
    "The clip is 1:23:45 long.",
    "Use a 16:9 screen.",
    "Connect to 10.0.0.1:80 now.",
    "Version 4:02x is odd.",
    "It ends at 14:30 PM.",
    "The dashboard is at grafana.internal:3000.",
])
def test_non_times_untouched(text):
    assert prepare(text).text == text


def test_time_is_a_protected_span():
    assert prepare("Order #48213 was delivered yesterday at 4:02 PM.").spans == ["four oh two PM"]


@pytest.mark.parametrize("text, spoken", [
    ("I attached the file Q3_report_final.xlsx.", "I attached the file Q three report final dot X L S X."),
    ("I saved the report as summary.pdf.", "I saved the report as summary dot P D F."),
    ("I renamed main.py to app.py.", "I renamed main dot P Y to app dot P Y."),
    ("The script lives in ~/bin/sync.sh.", "The script lives in ~/bin/sync dot S H."),
])
def test_file_extensions(text, spoken):
    assert prepare(text).text == spoken


@pytest.mark.parametrize("text", [
    "Send the details to billing@acme.io.",
    "The URL is docs.example.org/v2/auth.",
    "Set the timeout to 30s in settings.json.",
    "The logs are in /var/log/app.log.",
])
def test_domains_and_word_extensions_untouched(text):
    assert prepare(text).text == text


@pytest.mark.parametrize("text, spoken", [
    ("The variable is named MAX_RETRY_COUNT.", "The variable is named max retry count."),
    ("a field called created_at.", "a field called created at."),
    ("Call getUserName on the client.", "Call get user name on the client."),
    ("Parse it with parseHTMLString.", "Parse it with parse HTML string."),
])
def test_identifiers(text, spoken):
    assert prepare(text).text == spoken


@pytest.mark.parametrize("text", [
    "I bought an iPhone and a MacBook.",
    "We met at McDonald's.",
    "Sold on eBay.",
    "Upload it to YouTube.",
])
def test_brand_names_untouched(text):
    assert prepare(text).text == text


@pytest.mark.parametrize("text, spoken", [
    ("Your tracking number is 1Z999AA10123456784.",
     "Your tracking number is one Z nine nine nine A A one zero one two three four five six seven eight four."),
    ("Your return label number is RMA58812.", "Your return label number is R M A five eight eight one two."),
    ("Your gift card code is 4KJ9-77TQ.", "Your gift card code is four K J nine seven seven T Q."),
    ("The error code is E1047.", "The error code is E one zero four seven."),
    ("in commit a3f9c2.", "in commit A three F nine C two."),
])
def test_codes_letter_by_letter(text, spoken):
    assert prepare(text).text == spoken


@pytest.mark.parametrize("text", [
    "Your flight leaves from gate B12.",
    "COVID-19 rules still apply.",
    "COVID19 rules still apply.",
    "The drive holds 100GB.",
    "The ticket ID is INC-20931.",
    "We last spoke on Friday, the 22nd.",
    "It was founded in 1998.",
    "The S3 bucket is called prod-assets-eu.",
    "The decade was great.",
])
def test_non_codes_untouched(text):
    assert prepare(text).text == text


def test_version_suffix():
    assert prepare("The container image is tagged 2.7.0-rc1.").text == "The container image is tagged 2.7.0 R C one."
    assert prepare("Roll back to v1.9.0-beta.").text == "Roll back to v1.9.0 beta."
    assert prepare("You're on version 2.4.1.").text == "You're on version 2.4.1."


@pytest.mark.parametrize("text", [
    "Dr. Patel will call you back this afternoon.",
    "OK, I've updated your profile.",
    "Your total comes to $42.50.",
    "The conference runs from June 3 to June 5, 2027.",
])
def test_prose_and_plain_numbers_untouched(text):
    p = prepare(text)
    assert p.text == text and p.spans == []


# ── Post-check ───────────────────────────────────────────────────────────────

def _check(text, output):
    return check_output(prepare(text), output)[0]


@pytest.mark.parametrize("text, bad", [
    ("Order #48213 was delivered yesterday at 4:02 PM.",
     "Order number four eight two one three was delivered yesterday at four PM two minutes."),
    ("Order #48213 was delivered yesterday at 4:02 PM.",
     "Order number four eight two one three was delivered yesterday at four PM twenty-two."),
    ("I attached the file Q3_report_final.xlsx.", "I attached the file Q three report final dot X."),
    ("The variable is named MAX_RETRY_COUNT.", "The variable is named Maximum Retry Count."),
    ("Your tracking number is 1Z999AA10123456784.",
     "Your tracking number is one Z nine nine nine A one zero one two three four five six seven eight four."),
    ("OK, I've updated your profile.", "I've updated your profile."),
])
def test_known_misreadings_rejected(text, bad):
    assert not _check(text, bad)


@pytest.mark.parametrize("text, good", [
    ("Order #48213 was delivered yesterday at 4:02 PM.",
     "Order number four eight two one three was delivered yesterday at four oh two PM."),
    ("Meet me at 4:02 PM.", "Meet me at four oh two p.m."),
    ("I attached the file Q3_report_final.xlsx.", "I attached the file Q three report final dot X L S X."),
    ("OK, I've updated your profile.", "Okay, I've updated your profile."),
    ("Dr. Patel will call you back this afternoon.", "Doctor Patel will call you back this afternoon."),
    ("Mr. and Mrs. Chen confirmed for dinner.", "Mister and Missus Chen confirmed for dinner."),
    ("The speed limit there is 65 mph.", "The speed limit there is sixty-five miles per hour."),
    ("The Q&A session is right after the keynote.", "The Q and A session is right after the keynote."),
    ("I moved the meeting to the office on Main St.", "I moved the meeting to the office on Main Street."),
    ("Please update to iOS 18.1.", "Please update to iOS eighteen point one."),
])
def test_good_readings_accepted(text, good):
    assert _check(text, good)


def test_empty_output_rejected():
    assert check_output(Prepared("Hello there."), "") == (False, "empty output")


def test_repeated_span_needs_every_occurrence():
    text = "I renamed main.py to app.py."
    assert _check(text, "I renamed main dot P Y to app dot P Y.")
    assert not _check(text, "I renamed main dot P Y to main dot P Y.")


@pytest.mark.parametrize("text, output, ok", [
    ("I renamed main.py to app.py.", "I renamed main dot PY to app dot PY.", True),
    ("The confirmation code is QX4B9.", "The confirmation code is QX four B nine.", True),
    ("I attached the file Q3_report_final.xlsx.", "I attached the file Q three report final dot XLSX.", True),
    ("I attached the file Q3_report_final.xlsx.", "I attached the file Q three report final dot XLS.", False),
    ("Your code is 1Z999AA1.", "Your code is one Z nine nine nine AA one.", True),
    ("Your code is 1Z999AA1.", "Your code is one Z nine nine nine A one.", False),
    ("The container image is tagged 2.7.0-rc1.",
     "The container image is tagged two point seven point zero R C one.", True),
    ("The container image is tagged 2.7.0-rc1.",
     "The container image is tagged two point seven point zero RC one.", True),
    ("The container image is tagged 2.7.0-rc1.",
     "The container image is tagged two point seven point zero minus RC one.", True),
    ("The container image is tagged 2.7.0-rc1.",
     "The container image is tagged two point seven point zero R one.", False),
])
def test_fused_letters(text, output, ok):
    assert _check(text, output) is ok


# ── Verbalizer integration: a rejected output falls back to the pre-pass text ─

def _verbalizer_returning(model_text):
    from app.verbalizer import Verbalizer

    v = MagicMock(spec=Verbalizer)
    v._verbalize_plain = Verbalizer._verbalize_plain.__get__(v, Verbalizer)
    v._extract_output = Verbalizer._extract_output
    v._generate = Verbalizer._generate.__get__(v, Verbalizer)   # _verbalize_plain's model call
    v._extract_tagged = Verbalizer._extract_tagged
    v.growth_guard = True
    v.guard = True
    v.block = None
    v.tokenizer = MagicMock()
    v.tokenizer.encode.return_value = [1]
    v.tokenizer.return_value = {"input_ids": [2]}
    v.tokenizer.decode.return_value = f"<tool_call>{model_text}</tool_call>"
    v._prefix_ids, v._think_ids = [0], [9]
    v.sampling_params = MagicMock()
    v._tokens_prompt = dict

    def generate(prompt, sampling_params, request_id):
        async def gen():
            yield MagicMock(outputs=[MagicMock(token_ids=[5, 6])])
        return gen()

    v.engine = MagicMock()
    v.engine.generate.side_effect = generate
    return v


def test_verbalize_plain_keeps_accepted_output():
    v = _verbalizer_returning("The webinar starts at eleven oh five AM.")
    assert asyncio.run(v._verbalize_plain("The webinar starts at 11:05 AM.")) == "The webinar starts at eleven oh five AM."


def test_verbalize_plain_rejects_misreading():
    v = _verbalizer_returning("The webinar starts at eleven point zero five AM.")
    assert asyncio.run(v._verbalize_plain("The webinar starts at 11:05 AM.")) == "The webinar starts at eleven oh five AM."


def test_verbalize_plain_rejects_dropped_word():
    v = _verbalizer_returning("I've updated your profile.")
    assert asyncio.run(v._verbalize_plain("OK, I've updated your profile.")) == "OK, I've updated your profile."
