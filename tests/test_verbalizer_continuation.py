"""Verbalizer on prefix continuations: the growth guard, the prefix context (VERBALIZE_PREFIX_CONTEXT) and the
serving wiring. The model is mocked."""
import asyncio
import logging
from unittest.mock import AsyncMock

import pytest

from app.verbalizer_context import context_tail, match_first_case
from app.verbalizer_guard import check_growth, check_output, prepare, split_continuation

# The serving verbalizer's reading of the raw rest "or dairy, but ...": its own system instructions, twice,
# 1,166 chars.
_INSTRUCTIONS = (
    "For example, expand all numbers and special characters into words. Convert spelled out terms or initialisms to "
    "full uppercase versions (FBI, NASA, US, etc.). Convert acronyms to normal capitalized words (NASA → Nasa, FBI → "
    "FBI, etc.). Convert symbols to words ($ → dollars, % → percent, @ → at, ± → plus or minus). Convert dates and "
    "times to spoken form. Never expand initialisms or acronyms into full words. Keep them as-is: VP stays VP, CEO "
    "stays CEO, US stays US, etc. Do not paraphrase or rephrase. Do not think. Output only the normalized spoken "
    "text, nothing else."
)
LEAK = f"{_INSTRUCTIONS}<think></think><tool_call>{_INSTRUCTIONS}"

DAIRY_PREFIX = "Please let us know about any allergies in advance, and the chef will check with the kitchen, and they can prepare the dish without nuts"
DAIRY = "or dairy, but they would like at least a day of notice to make sure everything is handled safely."
PLAN_PREFIX = "If you want to keep the costs down, I would start with the"
PLAN = "smaller plan, then upgrade later once you know how much storage your team actually needs from month to month."
AUTOPAY_PREFIX = "Your total comes to eighty-six dollars including tax, and since autopay is"
AUTOPAY = "turned on, the amount will be taken from your card automatically on the first of next month."
TRAFFIC_PREFIX = "I will be a little earlier tomorrow, because the roadworks near the bridge are causing long delays,"
TRAFFIC = "and the traffic report says the queue was over 25 minutes this morning."
TRAFFIC_SPOKEN = "and the traffic report says the queue was over twenty-five minutes this morning."


def _accepted(text, output):
    prepared = prepare(text)
    return check_output(prepared, output)[0] and check_growth(prepared, output)[0]


# ── Growth guard ─────────────────────────────────────────────────────────────

def test_instruction_leak_rejected():
    assert len(LEAK) == 1166
    ok, reason = check_growth(prepare(DAIRY), LEAK)
    assert not ok and "too long" in reason
    # Without the char cap the words alone give it away.
    ok, reason = check_growth(prepare(DAIRY), _INSTRUCTIONS[:200])
    assert not ok and "not explained" in reason


def test_completed_fragment_rejected():
    assert check_output(prepare(PLAN), "A " + PLAN)[0]   # the misread guard lets an added word through
    assert not check_growth(prepare(PLAN), "A " + PLAN)[0]
    assert not check_growth(prepare("a few lessons"), "a few lessons, please")[0]   # trailing word


def test_dropped_words_rejected():
    output = "The amount will be taken from your card automatically on the first of next month."
    assert not check_output(prepare(AUTOPAY), output)[0]
    assert not check_growth(prepare(AUTOPAY), output)[0]


def test_capitalised_first_word_is_not_a_change():
    assert _accepted("so they should each receive an email.", "So they should each receive an email.")


@pytest.mark.parametrize("text, output", [
    (TRAFFIC, "And the traffic report says the queue was over twenty-five minutes this morning."),
    ("we have added a credit of $20 to your bill.", "we have added a credit of twenty dollars to your bill."),
    ("the clinic is at 120 Oak Avenue.", "the clinic is at one hundred twenty Oak Avenue."),
    ("It costs $1,234,567.89 today.",
     "It costs one million two hundred thirty-four thousand five hundred sixty-seven dollars and eighty-nine cents today."),
    ("The speed limit there is 65 mph.", "The speed limit there is sixty-five miles per hour."),
    ("The Q&A session is right after the keynote.", "The Q and A session is right after the keynote."),
    ("Mr. and Mrs. Chen confirmed for dinner.", "Mister and Missus Chen confirmed for dinner."),
    ("See you e.g. tomorrow at 4:02 PM.", "See you for example tomorrow at four oh two p.m."),
    ("I attached the file Q3_report_final.xlsx.", "I attached the file Q three report final dot XLSX."),
    ("OK, I've updated your profile.", "Okay, I've updated your profile."),
    ("Please update to iOS 18.1.", "Please update to iOS eighteen point one."),
    ("Your tracking number is 1Z999AA10123456784.",
     "Your tracking number is one Z nine nine nine AA one zero one two three four five six seven eight four."),
])
def test_expansions_accepted(text, output):
    assert check_growth(prepare(text), output) == (True, "ok")


@pytest.mark.parametrize("text", [
    DAIRY, PLAN, AUTOPAY, TRAFFIC, "R & D is here.", "The container image is tagged 2.7.0-rc1.",
    "at 12:30 p.m. tomorrow", "Send the details to billing@acme.io, or call +1 (555) 123-4567!", "A B C D", "—",
])
def test_unchanged_text_always_accepted(text):
    prepared = prepare(text)
    assert check_growth(prepared, prepared.text) == (True, "ok")


def test_one_interior_extra_word_tolerated_not_two():
    text = "we will meet at the old station tomorrow."
    assert check_growth(prepare(text), "we will meet up at the old station tomorrow.")[0]
    assert not check_growth(prepare(text), "we will meet up at the old train station tomorrow.")[0]


# ── Prefix context helpers ───────────────────────────────────────────────────

def test_context_tail_takes_the_last_sentence_and_a_clause_start():
    assert context_tail("We checked it twice. so since autopay is") == "So since autopay is"
    assert context_tail(PLAN_PREFIX) == "I would start with the"   # 13 words: window, then after "down,"
    assert context_tail(DAIRY_PREFIX) == "And they can prepare the dish without nuts"
    assert context_tail("Hello <<laugh>> so then the") == "So then the"
    assert context_tail("<<laugh>>") == ""
    assert len(context_tail("word " * 40).split()) == 12


@pytest.mark.parametrize("raw, part, context, expected", [
    ("so they should", "So they should", "", "so they should"),
    ("turned on, the", "Turned on, the", "since autopay is", "turned on, the"),
    ("25 minutes", "Twenty-five minutes", "over", "twenty-five minutes"),
    ("25 minutes", "Twenty-five minutes", "It was over.", "Twenty-five minutes"),
    ("$20 credit", "Twenty dollars credit", "a", "twenty dollars credit"),
    ("fbi agents", "FBI agents", "the", "fbi agents"),
    ("nasa said", "NASA said", "the", "nasa said"),
    ("2 FBI agents", "FBI agents", "the", "FBI agents"),
    ("i think", "I think", "so", "i think"),
    ("so I think", "I think", "and", "I think"),
    ("Oak Avenue", "Oak Avenue", "at", "Oak Avenue"),
    ('"yes," she said', '"Yes," she said', "and", '"yes," she said'),
])
def test_match_first_case(raw, part, context, expected):
    assert match_first_case(raw, part, context) == expected


@pytest.mark.parametrize("context, rest, output, expected", [
    ("And since autopay is", AUTOPAY, f"And since autopay is {AUTOPAY}", AUTOPAY),
    ("The roadworks are causing long delays,", TRAFFIC,
     f"The roadworks are causing long delays, {TRAFFIC_SPOKEN}", TRAFFIC_SPOKEN),
    ("The total is", "$20 for now.", "The total is, twenty dollars for now.", "twenty dollars for now."),
    ('He said "stop"', "and left.", 'He said "stop" and left.', "and left."),
    ("He said", '"Yes, of course."', 'He said, "Yes, of course."', '"Yes, of course."'),
    ("I would start with the", PLAN, f"So I would start with the {PLAN}", PLAN),   # the context's reading is dropped
    ("It costs $5", "20% more.", "It costs five dollars twenty percent more.", None),   # numbers on both sides
    ("Without nuts", DAIRY, LEAK, None),
    ("I would start with the", PLAN, f"I would start with the {PLAN[:40]}", None),   # rest truncated
])
def test_split_continuation(context, rest, output, expected):
    assert split_continuation(prepare(context).text, prepare(rest).text, output) == expected


# ── Verbalizer (model mocked) ────────────────────────────────────────────────

def _verbalizer(readings, growth_guard=True):
    """A Verbalizer whose model call returns readings[text] (the text itself when not listed); records the calls."""
    from app.verbalizer import Verbalizer

    v = Verbalizer.__new__(Verbalizer)
    v.growth_guard = growth_guard
    v.guard = True
    v._generate = AsyncMock(side_effect=lambda text: readings.get(text, text.strip()))
    return v


def _calls(v):
    return [call.args[0] for call in v._generate.await_args_list]


def test_rest_alone_instruction_leak_falls_back_to_raw():
    v = _verbalizer({DAIRY: LEAK})
    assert asyncio.run(v.verbalize(DAIRY)) == DAIRY


def test_continuation_reads_prefix_context_and_keeps_the_rest_lowercase():
    joint = f"And they can prepare the dish without nuts {DAIRY}"
    v = _verbalizer({joint: "And they can prepare the dish without nuts " + DAIRY, DAIRY: LEAK})
    assert asyncio.run(v.verbalize_continuation(DAIRY_PREFIX, DAIRY)) == DAIRY
    assert _calls(v) == [joint]


def test_continuation_leak_in_both_readings_falls_back_to_raw():
    v = _verbalizer({f"And they can prepare the dish without nuts {DAIRY}": LEAK, DAIRY: LEAK})
    assert asyncio.run(v.verbalize_continuation(DAIRY_PREFIX, DAIRY)) == DAIRY
    assert len(_calls(v)) == 2


def test_completed_fragment_rejected_on_both_paths():
    joint = f"I would start with the {PLAN}"
    v = _verbalizer({joint: f"I would start with the {PLAN}", PLAN: "A " + PLAN})
    assert asyncio.run(v.verbalize_continuation(PLAN_PREFIX, PLAN)) == PLAN
    assert asyncio.run(v.verbalize(PLAN)) == PLAN   # alone: "A smaller plan" rejected -> raw


def test_dropped_words_rejected_on_both_paths():
    dropped = "The amount will be taken from your card automatically on the first of next month."
    joint = f"And since autopay is {AUTOPAY}"
    v = _verbalizer({joint: f"And since autopay is {dropped}", AUTOPAY: dropped})
    assert asyncio.run(v.verbalize_continuation(AUTOPAY_PREFIX, AUTOPAY)) == AUTOPAY
    assert asyncio.run(v.verbalize(AUTOPAY)) == AUTOPAY


def test_number_expansion_in_the_rest_accepted():
    context = context_tail(TRAFFIC_PREFIX)
    assert context == "Because the roadworks near the bridge are causing long delays,"
    joint = f"{context} {TRAFFIC}"
    v = _verbalizer({joint: f"{context} And the traffic report says the queue was over twenty-five minutes this morning."})
    assert asyncio.run(v.verbalize_continuation(TRAFFIC_PREFIX, TRAFFIC)) == TRAFFIC_SPOKEN


def test_uncertain_split_falls_back_to_the_rest_alone_recased():
    rest = "so they should each receive an email with a link."
    v = _verbalizer({"I have added the two team members you mentioned, " + rest: "garbled",
                     rest: "So they should each receive an email with a link."})
    assert asyncio.run(v.verbalize_continuation("I have added the two team members you mentioned,", rest)) == rest
    assert len(_calls(v)) == 2


def test_prefix_that_ends_a_sentence_needs_no_context():
    v = _verbalizer({"25 minutes later it stopped.": "Twenty-five minutes later it stopped."})
    assert asyncio.run(v.verbalize_continuation("We waited.", "25 minutes later it stopped.")) == \
        "Twenty-five minutes later it stopped."
    assert _calls(v) == ["25 minutes later it stopped."]
    v = _verbalizer({"so it stopped.": "So it stopped."})
    assert asyncio.run(v.verbalize_continuation("We waited.", "so it stopped.")) == "so it stopped."


def test_continuation_keeps_tags_and_only_the_first_part_gets_context():
    v = _verbalizer({})
    result = asyncio.run(v.verbalize_continuation("without nuts", "or dairy, <<laugh>> and 5 more."))
    assert result == "or dairy,<<laugh>>and 5 more."   # the model output is stripped, as in verbalize()
    assert _calls(v) == ["Without nuts or dairy, ", " and 5 more."]


def test_growth_guard_flag_off_restores_the_misread_guard_alone():
    v = _verbalizer({PLAN: "A " + PLAN}, growth_guard=False)
    assert asyncio.run(v.verbalize(PLAN)) == "A " + PLAN


def test_joint_reading_error_falls_back_to_the_rest_alone(caplog, monkeypatch):
    from app.verbalizer import COUNTS

    monkeypatch.setattr(logging.getLogger("tontaube"), "propagate", True)   # caplog listens on the root logger

    joint = f"And they can prepare the dish without nuts {DAIRY}"

    def model(text):
        if text == joint:
            raise RuntimeError("engine died")
        return text.strip()

    v = _verbalizer({})
    v._generate = AsyncMock(side_effect=model)
    before = COUNTS.copy()
    with caplog.at_level("WARNING", logger="tontaube.verbalizer"):
        assert asyncio.run(v.verbalize_continuation(DAIRY_PREFIX, DAIRY)) == DAIRY
    assert _calls(v) == [joint, DAIRY]
    assert COUNTS["context_error"] - before["context_error"] == 1
    assert COUNTS["context_not_used"] - before["context_not_used"] == 1
    assert "engine died" in caplog.text


def test_fallbacks_are_counted_and_logged_at_info(caplog, monkeypatch):
    from app.verbalizer import COUNTS

    monkeypatch.setattr(logging.getLogger("tontaube"), "propagate", True)

    before = COUNTS.copy()
    v = _verbalizer({DAIRY: LEAK, PLAN: None})
    with caplog.at_level("INFO", logger="tontaube.verbalizer"):
        assert asyncio.run(v.verbalize(DAIRY)) == DAIRY
        assert asyncio.run(v.verbalize(PLAN)) == PLAN
    assert COUNTS["rejected"] - before["rejected"] == 1
    assert COUNTS["no_output"] - before["no_output"] == 1
    assert "output rejected" in caplog.text and "no output" in caplog.text


@pytest.mark.parametrize("text, output", [
    ("We met the R & D team.", "We met the R and D team."),
    ("Any Q & A after the talk.", "Any Q and A after the talk."),
    ("Install P Y torch first.", "Install PY torch first."),
])
def test_spelled_capitals_fuse_only_when_adjacent(text, output):
    assert _accepted(text, output)


def test_non_ascii_capital_rest_keeps_its_case():
    assert match_first_case("Éric and Zoë arrived.", "Eric and Zoe arrived.", "and then") == "Eric and Zoe arrived."
    assert match_first_case("élan, said the critic.", "Élan, said the critic.", "with") == "élan, said the critic."


# ── Serving wiring ───────────────────────────────────────────────────────────

class _Captured(Exception):
    pass


class _FakeVerbalizer:
    def __init__(self):
        self.calls = []

    async def verbalize(self, text):
        self.calls.append(("plain", text))
        return f"V {text}"

    async def verbalize_continuation(self, prefix_text, text):
        self.calls.append(("continuation", prefix_text, text))
        return f"C {text}"


class _FakeConfig:
    def __init__(self, verbalize_prefix_context):
        self.verbalize_prefix_context = verbalize_prefix_context


class _FakeEngine:
    vv_tokenizer = object()
    can_stream = True

    def __init__(self, verbalize_prefix_context=True):
        self.verbalizer = _FakeVerbalizer()
        self.config = _FakeConfig(verbalize_prefix_context)
        self.texts = []
        self.preprocessed = []

    async def generate(self, text, **kwargs):
        self.texts.append(text)
        raise _Captured

    async def generate_streaming(self, text, text_preprocessor=None, **kwargs):
        for segment in [text] if isinstance(text, str) else text:
            self.preprocessed.append(await text_preprocessor(segment))
        return
        yield


@pytest.fixture
def main(monkeypatch):
    pytest.importorskip("fastapi")
    import app.main as main
    return main


def _run(main, monkeypatch, engine, **request):
    async def get_engine():
        return engine

    monkeypatch.setattr(main, "_get_engine", get_engine)
    with pytest.raises(_Captured):
        asyncio.run(main._synthesize(main.TTSRequest(use_verbalization=True, **request)))
    return engine.texts[0]


def test_no_prefix_request_uses_the_old_path(main, monkeypatch):
    engine = _FakeEngine()
    assert _run(main, monkeypatch, engine, text="so they should each receive an email.") == \
        "V so they should each receive an email."
    assert engine.verbalizer.calls == [("plain", "so they should each receive an email.")]


def test_prefix_request_verbalizes_the_first_segment_with_context(main, monkeypatch):
    engine = _FakeEngine()
    _run(main, monkeypatch, engine, text=["or dairy, but.", "Then more."], prefix_text=DAIRY_PREFIX)
    assert engine.verbalizer.calls == [("continuation", DAIRY_PREFIX, "or dairy, but."), ("plain", "Then more.")]


def test_flag_off_restores_the_old_path(main, monkeypatch):
    engine = _FakeEngine(verbalize_prefix_context=False)
    _run(main, monkeypatch, engine, text=DAIRY, prefix_text=DAIRY_PREFIX)
    assert engine.verbalizer.calls == [("plain", DAIRY)]


def test_blank_prefix_uses_the_old_path(main, monkeypatch):
    engine = _FakeEngine()
    _run(main, monkeypatch, engine, text=DAIRY, prefix_text="  ")
    assert engine.verbalizer.calls == [("plain", DAIRY)]


@pytest.mark.asyncio
async def test_stream_only_the_first_segment_gets_the_context(main, monkeypatch):
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
        response = await client.post("/stream", json={
            "text": ["or dairy, but.", "Then more."], "prefix_text": DAIRY_PREFIX, "use_verbalization": True,
        })
    assert response.status_code == 200
    assert engine.verbalizer.calls == [("continuation", DAIRY_PREFIX, "or dairy, but."), ("plain", "Then more.")]


def test_config_flags_default_on(monkeypatch):
    from app.config import ServerConfig
    for name in ("VERBALIZE_PREFIX_CONTEXT", "VERBALIZER_GROWTH_GUARD"):
        monkeypatch.delenv(name, raising=False)
    config = ServerConfig()
    assert config.verbalize_prefix_context is True and config.verbalizer_growth_guard is True
    monkeypatch.setenv("VERBALIZE_PREFIX_CONTEXT", "0")
    monkeypatch.setenv("VERBALIZER_GROWTH_GUARD", "0")
    config = ServerConfig()
    assert config.verbalize_prefix_context is False and config.verbalizer_growth_guard is False
