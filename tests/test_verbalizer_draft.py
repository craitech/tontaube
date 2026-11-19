"""Draft-and-verify verbalization: the text draft, its follower, chunking, and how pieces share model calls."""
import asyncio

import pytest

from app.verbalizer import CHUNK_MAX_CHARS, Verbalizer, _plausible, _weight, split_for_verbalizer
from app.verbalizer_draft import DraftFollower, Normalizer


# ── The draft ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("written, spoken", [
    ("It runs in 200 ms on 2 GPUs [2][5].", "It runs in two hundred milliseconds on two GPUs."),
    ("Dr. Lee paid $5 on the 1st, e.g. for NVIDIA.", "Doctor Lee paid five dollars on the first, for example for Nvidia."),
    ("Accuracy rose to 94.7% in 2026.", "Accuracy rose to ninety-four point seven percent in twenty twenty-six."),
    ("The RTX 5090 is 12.5× faster.", "The RTX fifty ninety is twelve point five times faster."),
    ("Generated on 2026-10-28 10:50 UTC by the tool.", "Generated on October twenty-eighth, twenty twenty-six, at ten fifty UTC by the tool."),
    ("[3] example.io — Model Card", "Reference three, example dot IO, Model Card."),
    ("Revenue was $27,040,500 in 2025.", "Revenue was <|fim_prefix|>27,040,500<|fim_pad|> dollars in twenty twenty-five."),
])
def test_draft_guesses_common_readings(written, spoken):
    assert Normalizer().draft(written) == spoken


def test_draft_reads_table_rows_with_their_headers():
    table = "Iter\tCoverage\tSources\n1\t0.90\t10\n2\t1.00\t7"
    assert Normalizer().draft(table) == (
        "Iter One: Coverage zero point nine zero, Sources ten.\n"
        "Iter Two: Coverage one point zero zero, Sources seven."
    )
    assert Normalizer().draft("\t\t") == "\t\t"   # separators only: not a table


def test_draft_learns_respellings_within_a_text():
    normalizer = Normalizer()
    assert "ZetaCodec" not in normalizer.words
    normalizer.learn("We use ZetaCodec here.", "We use Zeta Codec here.")
    assert normalizer.draft("ZetaCodec is fast.") == "Zeta Codec is fast."


# ── Following the draft (one "token" per character) ──────────────────────────

def _follower(draft):
    return DraftFollower(draft, lambda text: [ord(c) for c in text], chr)


def test_follower_advances_through_accepted_text():
    follower = _follower("It costs five dollars.")
    out = [ord(c) for c in "It costs"]
    follower.advance(out)
    assert "".join(map(chr, follower.propose(40))) == " five dollars."


def test_follower_rejoins_after_the_model_skips_or_respells():
    # The model drops the "$" the draft still has and writes " five": the draft continues after it.
    follower = DraftFollower("the price is $five today", lambda text: text.split(" "), lambda token: " " + token)
    follower.advance(["the", "price", "is", "five"])
    assert follower.dt[follower.c:] == " today"


def test_follower_stays_when_the_model_inserts_text():
    follower = DraftFollower("five per month", lambda text: text.split(" "), lambda token: " " + token)
    follower.c = len("five")
    follower.n_out = 1
    follower.advance(["five", "dollars"])
    assert follower.dt[follower.c:] == " per month"


# ── Chunking ─────────────────────────────────────────────────────────────────

def _rejoin(chunks):
    return "".join(piece + sep for piece, sep in chunks)


def test_split_short_text_is_one_input():
    assert split_for_verbalizer(" Hello $5 world ") == [(" Hello $5 world ", "")]


def test_split_cuts_at_paragraphs_and_keeps_the_text():
    paragraphs = [(f"Paragraph {i} " + "has several plain words in it. " * 9).strip() for i in range(6)]
    text = "\n\n".join(paragraphs)
    chunks = split_for_verbalizer(text)
    assert _rejoin(chunks) == text
    assert len(chunks) > 1
    assert all(piece.startswith("Paragraph") and piece.endswith(".") for piece, _ in chunks)
    assert all(_weight(piece) <= CHUNK_MAX_CHARS for piece, _ in chunks)


def test_split_long_paragraph_at_sentence_ends():
    chunks = split_for_verbalizer(("This sentence is here to fill the paragraph up. " * 40).strip())
    assert len(chunks) > 1
    assert all(piece.endswith(".") and _weight(piece) <= CHUNK_MAX_CHARS for piece, _ in chunks)


def test_split_keeps_heading_with_following_text():
    body = ("Some findings are listed in this paragraph for the reader. " * 9).strip()
    text = body + "\n\nKey Findings\n\n" + body
    chunks = split_for_verbalizer(text)
    assert _rejoin(chunks) == text
    assert chunks[1][0].startswith("Key Findings\n\nSome findings")


def test_split_table_repeats_header():
    rows = [f"| {i} | 0.{i:02d} | {i * 3} |" for i in range(1, 60)]
    chunks = split_for_verbalizer("\n".join(["| Iter | Coverage | Sources |", "|---|---|---|"] + rows))
    assert len(chunks) > 1
    assert all(piece.startswith("| Iter | Coverage | Sources |\n|---|---|---|\n") for piece, _ in chunks)
    assert sum(piece.count("\n") + 1 - 2 for piece, _ in chunks) == len(rows)


def test_plausible_output():
    assert _plausible("It costs $5 today.", "It costs five dollars today.")
    assert _plausible("Er kommt um 7 Uhr.", "Er kommt um 7 Uhr.")   # copied unchanged, digits and all
    assert not _plausible("A long sentence with many words that the model dropped entirely.", "A long")
    assert not _plausible("Short input.", "Short input. " * 40)


# ── Verbalizer: pieces of one text, and concurrent callers, share model calls ─

def _verbalizer(run):
    v = Verbalizer.__new__(Verbalizer)
    v.guard, v.block, v._queue, v._draining, v.calls = False, object(), [], False, []

    def _run(texts):
        v.calls.append(list(texts))
        return [run(text) for text in texts]
    v._run = _run
    return v


def test_pieces_of_a_long_text_are_read_in_one_call():
    paragraphs = [(f"Part {name} " + "has several plain words in it. " * 9).strip() for name in "ABCDE"]
    text = "\n\n".join(paragraphs)
    v = _verbalizer(lambda piece: piece.replace("Part", "Section"))
    assert asyncio.run(v.verbalize(text)) == text.replace("Part", "Section")
    assert len(v.calls) == 1 and len(v.calls[0]) > 1


def test_concurrent_requests_share_a_call():
    v = _verbalizer(str.upper)

    async def both():
        return await asyncio.gather(v.verbalize("first text."), v.verbalize("second text."))
    assert asyncio.run(both()) == ["FIRST TEXT.", "SECOND TEXT."]
    assert v.calls == [["first text.", "second text."]]


def test_rejected_piece_is_retried_by_sentence_then_kept_as_written():
    def run(text):
        if text == "It costs $5. It is 3 km away.":
            return None                      # the model did not close its output
        return "It costs five dollars." if text.startswith("It costs") else "It"   # most of the sentence lost
    v = _verbalizer(run)
    assert asyncio.run(v.verbalize("It costs $5. It is 3 km away.")) == "It costs five dollars. It is 3 km away."
    assert v.calls == [["It costs $5. It is 3 km away."], ["It costs $5.", "It is 3 km away."]]


def test_invisible_characters_are_removed_before_the_model():
    v = _verbalizer(lambda text: text)
    assert asyncio.run(v.verbalize("Hel\u200blo\U000e0101 wor\xadld\ud83d.")) == "Hello world."
    assert v.calls == [["Hello world."]]


def test_markers_are_read_out_and_checked_against_the_input():
    outputs = {
        "It cost $27,040,500.": "It cost <|fim_prefix|>27,040,500<|fim_pad|> dollars.",
        "Chapter XII. It cost $1,500,000.": "Chapter <|fim_middle|>XII<|fim_pad|>. It cost <|fim_prefix|>1,500,00<|fim_pad|> dollars.",
        "Chapter XII.": "Chapter <|fim_middle|>XII<|fim_pad|>.",
        "It cost $1,500,000.": "It cost <|fim_prefix|>500,000<|fim_pad|> dollars.",
    }
    v = _verbalizer(outputs.get)
    assert asyncio.run(v.verbalize("It cost $27,040,500.")) == "It cost twenty-seven million forty thousand five hundred dollars."
    # A digit lost in the copy: read sentence by sentence, the sentence that fails again stays as written.
    assert asyncio.run(v.verbalize("Chapter XII. It cost $1,500,000.")) == "Chapter Twelve. It cost $1,500,000."


def test_marker_strings_are_removed_before_the_model():
    v = _verbalizer(lambda text: text)
    assert asyncio.run(v.verbalize("A <|fim_prefix|>5<|fim_pad|> B.")) == "A 5 B."
    assert v.calls == [["A 5 B."]]


def test_broken_output_and_unreadable_text_give_no_output():
    import threading

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            if text == "bad":
                raise ValueError(text)
            return [ord(c) for c in text]

        def decode(self, ids, skip_special_tokens=False):
            return "".join(chr(i) for i in ids)

    class Model:   # the next token after output position i is script[i]; 900 is a prompt tag, 901 closes the output
        lanes, prefix_len, max_len, max_rows, max_heads = 1, 0, 64, 64, 64

        def __init__(self, script, prompt):
            self.script, self.prompt = script, prompt

        def reserve(self, lane, length):
            return True

        def release(self, lane):
            pass

        def forward(self, parts):
            return [[(self.script + [901] * 64)[start + i - self.prompt + 1] for i in range(need, len(tokens))]
                    for _, tokens, start, need in parts]

    def run(script, texts=("hi",)):
        v = Verbalizer.__new__(Verbalizer)
        v.tokenizer, v._suffix_ids, v._template, v._stop = Tokenizer(), [900], {900}, {901}
        v._pieces, v._lock, v.block = {}, threading.Lock(), Model(script, len("hi") + 1)
        return v._run(list(texts))

    assert run([ord("o"), ord("k"), 901]) == ["ok"]
    assert run([ord("o"), 900, ord("k"), 901]) == [None]
    assert run([ord("o"), ord("k"), 901], ("bad", "hi")) == [None, "ok"]   # a text that cannot be read fails alone


# ── Standalone API ───────────────────────────────────────────────────────────

def test_verbalize_endpoint(monkeypatch):
    from fastapi.testclient import TestClient

    from app import verbalizer_server

    class Fake:
        async def verbalize(self, text):
            return text.replace("$5", "five dollars")
    monkeypatch.setattr(verbalizer_server, "load_verbalizer", lambda: Fake())
    with TestClient(verbalizer_server.create_app()) as client:
        assert client.post("/v1/verbalize", json={"text": "It costs $5."}).json() == {"text": "It costs five dollars."}
        assert client.post("/v1/verbalize", json={"text": ["$5", "ok"]}).json() == {"text": ["five dollars", "ok"]}
        assert client.post("/v1/verbalize", json={"text": "  "}).status_code == 422
        assert client.get("/healthz").json() == {"status": "ok"}
