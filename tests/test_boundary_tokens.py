"""CB0 end-token windows: the boundary tokens in CB0's text window.

The token-free path stays byte-identical to the earlier implementation.
"""
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import (
    BOUNDARY_TOKENS_FILENAME,
    SEMANTIC_VOCAB_SIZE,
    TEXT_MARKER,
    UTT_CONT,
    UTT_END,
    ServerConfig,
)
from app.text import sanitize_text_for_tts, strip_special_token_strings
from app.utils import check_boundary_tokens, check_cb0_window_ids, tokenize_text
from tests.test_inference import make_engine

SLIDERS = [f"<|s{i}|>" for i in range(4)]
SLIDER_BASE = 1000
END_ID, CONT_ID = SLIDER_BASE + len(SLIDERS), SLIDER_BASE + len(SLIDERS) + 1
SEGS = ["First chunk.", "Middle chunk.", "Third chunk.", "Last chunk."]


def token_engine(drop_space: bool = False):
    engine = make_engine()
    engine.config.cb0_boundary_tokens = True
    engine.config.cb0_boundary_drop_space = drop_space
    return engine


def window(engine, seg_idx, segments, **kwargs):
    flags = dict(
        boundary_tokens=engine.config.cb0_boundary_tokens,
        drop_space=engine.config.cb0_boundary_drop_space,
    )
    return engine._text_window(seg_idx, segments, use_context=True, lookahead_chars=None, **flags, **kwargs)[0]


class FakeTokenizer:
    """Character ids from ord(), the special tokens of a vocab dict; len() = the vocab's end."""

    unk_token_id = None

    def __init__(self, boundary: bool):
        self.vocab = {TEXT_MARKER: 900, **{t: SLIDER_BASE + i for i, t in enumerate(SLIDERS)}}
        if boundary:
            self.vocab.update({UTT_END: END_ID, UTT_CONT: CONT_ID})

    def get_vocab(self):
        return dict(self.vocab)

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token)

    def __call__(self, text):
        return {"input_ids": [ord(c) for c in text]}

    def __len__(self):
        return max(self.vocab.values()) + 1


def write_bundle(path, *, declare=True, ids=(END_ID, CONT_ID), codebooks=("cb0",), vocab_size=CONT_ID + 1,
                 table_rows=CONT_ID + 1):
    path.mkdir(parents=True, exist_ok=True)
    if declare:
        (path / BOUNDARY_TOKENS_FILENAME).write_text(json.dumps(
            {"tokens": [UTT_END, UTT_CONT], "ids": list(ids), "codebooks": list(codebooks)}))
    (path / "config.json").write_text(json.dumps({"vocab_size": vocab_size}))
    header = json.dumps({"model.embed_tokens.weight": {
        "dtype": "F32", "shape": [table_rows, 1], "data_offsets": [0, 4 * table_rows]}}).encode()
    (path / "model.safetensors").write_bytes(len(header).to_bytes(8, "little") + header + bytes(4 * table_rows))
    return str(path)


# ---------------------------------------------------------------------------
# The window layouts (with and without the leading-space drop)
# ---------------------------------------------------------------------------

class TestTokenWindowLayout:
    def test_single_chunk_ends_in_utt_end(self):
        assert window(token_engine(), 0, ["Only chunk."], system_tag="<h>\n") == "<h>\nOnly chunk." + UTT_END

    def test_first_of_several_has_utt_cont_and_no_lookahead_quirk(self):
        assert window(token_engine(), 0, ["First.", "Last."]) == "First." + UTT_CONT + TEXT_MARKER + " Last."
        assert make_engine()._text_window(0, ["First.", "Last."], use_context=True)[0] == (
            "First." + TEXT_MARKER + " Last.\n")  # legacy: the lookahead quirk

    def test_middle_chunk(self):
        assert window(token_engine(), 2, SEGS) == (
            " Middle chunk." + TEXT_MARKER + " Third chunk." + UTT_CONT + TEXT_MARKER + " Last chunk.")

    def test_last_chunk(self):
        assert window(token_engine(), 3, SEGS) == " Third chunk." + TEXT_MARKER + " Last chunk." + UTT_END

    @pytest.mark.parametrize("seg_idx, expected", [
        (0, "First chunk." + UTT_CONT + TEXT_MARKER + "Middle chunk."),
        (2, "Middle chunk." + TEXT_MARKER + "Third chunk." + UTT_CONT + TEXT_MARKER + "Last chunk."),
        (3, "Third chunk." + TEXT_MARKER + "Last chunk." + UTT_END),
    ])
    def test_drop_space_removes_every_leading_space(self, seg_idx, expected):
        assert window(token_engine(drop_space=True), seg_idx, SEGS) == expected

    def test_streaming_early_slice_ends_in_utt_cont(self):
        engine = token_engine()
        text, _, _ = engine._text_window(
            0, ["First chunk."], use_context=False, boundary_total_segments=2, boundary_tokens=True)
        assert text == "First chunk." + UTT_CONT

    def test_restart_window_keeps_the_real_boundary_cue(self):
        engine = token_engine()
        assert engine._text_window(3, SEGS, use_context=True, restart=True, boundary_tokens=True)[0] == (
            " Last chunk." + UTT_END)
        assert engine._text_window(1, SEGS, use_context=True, restart=True, boundary_tokens=True,
                                   lookahead_chars=None)[0] == (
            " Middle chunk." + UTT_CONT + TEXT_MARKER + " Third chunk.")

    def test_prefix_separator(self):
        prefix = {"text": " Earlier words. "}
        keep = token_engine()._text_window(0, ["Only."], use_context=True, prefix=prefix, boundary_tokens=True)[0]
        drop = token_engine()._text_window(
            0, ["Only."], use_context=True, prefix=prefix, boundary_tokens=True, drop_space=True)[0]
        assert keep == "Earlier words." + TEXT_MARKER + " Only." + UTT_END
        assert drop == "Earlier words." + TEXT_MARKER + "Only." + UTT_END

    def test_every_token_window_is_the_legacy_window_with_one_token(self):
        legacy_engine, engine = make_engine(), token_engine()
        for n in range(1, len(SEGS) + 1):
            for seg_idx in range(n):
                for use_context in (True, False):
                    for total in (None, n + 1):
                        kw = dict(use_context=use_context, boundary_total_segments=total, lookahead_chars=8)
                        legacy = legacy_engine._text_window(seg_idx, SEGS[:n], **kw)
                        token = engine._text_window(seg_idx, SEGS[:n], boundary_tokens=True, **kw)
                        assert token[1:] == legacy[1:]
                        assert token[0].count(UTT_END) + token[0].count(UTT_CONT) == 1
                        assert "\n" not in token[0]
                        stripped = token[0].replace(UTT_END, "").replace(UTT_CONT, "")
                        assert stripped == legacy[0].replace("\n", "")

    def test_flags_off_is_the_legacy_call(self):
        engine = make_engine()
        for seg_idx in range(len(SEGS)):
            assert engine._text_window(seg_idx, SEGS, use_context=True) == engine._text_window(
                seg_idx, SEGS, use_context=True, boundary_tokens=False, drop_space=False)

    def test_drop_space_without_tokens_is_refused(self):
        with pytest.raises(ValueError):
            make_engine()._text_window(0, SEGS, use_context=True, drop_space=True)
        with pytest.raises(ValueError):
            ServerConfig(cb0_boundary_drop_space=True)
        assert ServerConfig(cb0_boundary_tokens=True, cb0_boundary_drop_space=True).cb0_boundary_drop_space


# ---------------------------------------------------------------------------
# Wiring: CB0 gets the token window, CB1-3 keep the legacy one
# ---------------------------------------------------------------------------

def _tokenizer_mock():
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.convert_tokens_to_ids = lambda t: {
        '<|start_of_speech|>': 1, '<|dc_0_0|>': 100, '<|end_of_speech|>': 2, '<|audio_split|>': 3,
        '<|text_split|>': 4,
    }.get(t, 99)
    return tokenizer


class TestWiring:
    @pytest.mark.asyncio
    async def test_cb0_window_carries_utt_end_and_is_checked(self):
        engine = token_engine()
        engine.config.force_first_silence = False
        engine.tokenizers = [_tokenizer_mock()] * 3
        engine.engines = [MagicMock()] * 3
        engine.boundary_token_ids = (END_ID, CONT_ID)
        engine._generate_with_engine = AsyncMock(return_value=[100 + i for i in range(50)] + [2])
        seen = []

        def fake_tokenize(text, tokenizer):
            seen.append(text)
            return [10] + ([END_ID] if UTT_END in text else []) + ([CONT_ID] if UTT_CONT in text else [])
        with patch("app.inference.tokenize_text", side_effect=fake_tokenize):
            await engine._generate_semantic(
                ["a]" * 25], sampling={"temperature": 0.9, "top_p": 0.95, "top_k": 10,
                                       "frequency_penalty": 0.0, "seed": None},
                voice_prompt=(None, None))
        assert seen and all(t.endswith(UTT_END) for t in seen)

    @pytest.mark.asyncio
    async def test_cb0_window_whose_token_is_dropped_fails_closed(self):
        engine = token_engine()
        engine.config.force_first_silence = False
        engine.tokenizers = [_tokenizer_mock()] * 3
        engine.engines = [MagicMock()] * 3
        engine.boundary_token_ids = (END_ID, CONT_ID)
        engine._generate_with_engine = AsyncMock(return_value=[100 + i for i in range(50)] + [2])
        with patch("app.inference.tokenize_text", return_value=[10, 11]), \
                pytest.raises(ValueError, match="boundary token"):
            await engine._generate_semantic(
                ["a]" * 25], sampling={"temperature": 0.9, "top_p": 0.95, "top_k": 10,
                                       "frequency_penalty": 0.0, "seed": None},
                voice_prompt=(None, None))

    @pytest.mark.asyncio
    async def test_cb1_to_3_window_is_the_legacy_window(self):
        engine = token_engine()
        engine.tokenizers = [_tokenizer_mock()] * 3
        engine.engines = [MagicMock()] * 3
        engine._generate_with_engine = AsyncMock(return_value=[100 + SEMANTIC_VOCAB_SIZE + i for i in range(20)])
        seen = []
        segments = ["seg one.", "seg two."]
        with patch("app.inference.tokenize_text", side_effect=lambda text, tok: seen.append(text) or [10, 11]):
            await engine._generate_acoustic(
                segments=segments, semantic_codes=list(range(40)), marker_indices=[20], prev_acoustic=[],
                codebook=1, voice_prompt=(None, None), use_context=False)
        legacy = make_engine()
        assert seen == [legacy._text_window(i, segments, use_context=False)[0] for i in range(2)]
        assert seen == ["seg one.", " seg two.\n"]


# ---------------------------------------------------------------------------
# Tokenizer / bundle checks (utils)
# ---------------------------------------------------------------------------

class TestCheckCb0WindowIds:
    def test_token_window_needs_exactly_one_boundary_id(self):
        tok = FakeTokenizer(boundary=True)
        text = "Hi." + UTT_END
        check_cb0_window_ids(text, tokenize_text(text, tok), (END_ID, CONT_ID), True)
        with pytest.raises(ValueError):
            check_cb0_window_ids(text, tokenize_text(text, FakeTokenizer(boundary=False)), (END_ID, CONT_ID), True)
        with pytest.raises(ValueError):
            check_cb0_window_ids(text + UTT_END, tokenize_text(text + UTT_END, tok), (END_ID, CONT_ID), True)
        with pytest.raises(ValueError):
            check_cb0_window_ids(text, tokenize_text(text, tok), None, True)

    def test_legacy_window_on_a_declaring_bundle_carries_none(self):
        tok = FakeTokenizer(boundary=True)
        check_cb0_window_ids("Hi.\n", tokenize_text("Hi.\n", tok), (END_ID, CONT_ID), False)
        with pytest.raises(ValueError):
            check_cb0_window_ids("Hi." + UTT_END, tokenize_text("Hi." + UTT_END, tok), (END_ID, CONT_ID), False)

    def test_legacy_window_on_a_legacy_bundle_is_unchecked(self):
        check_cb0_window_ids("Hi." + UTT_END, [1, 2, 3], None, False)


class TestCheckBoundaryTokens:
    def test_legacy_bundle(self, tmp_path):
        path = write_bundle(tmp_path / "cb0", declare=False, vocab_size=END_ID, table_rows=END_ID)
        assert check_boundary_tokens(path, FakeTokenizer(boundary=False), 0, SLIDERS) is None

    def test_declaring_bundle(self, tmp_path):
        path = write_bundle(tmp_path / "cb0")
        assert check_boundary_tokens(path, FakeTokenizer(boundary=True), 0, SLIDERS) == (END_ID, CONT_ID)

    @pytest.mark.parametrize("bundle, codebook, boundary_tokenizer", [
        (dict(declare=False), 0, True),                       # tokens without a declaration
        (dict(), 0, False),                                   # a declaration without the tokens
        (dict(), 1, True),                                    # a CB1-3 bundle declaring them
        (dict(codebooks=("cb0", "cb1")), 0, True),
        (dict(ids=(CONT_ID, END_ID)), 0, True),               # ids that disagree with the tokenizer
        (dict(vocab_size=END_ID), 0, True),                   # config.json without the rows
        (dict(table_rows=END_ID), 0, True),                   # a token table without the rows
    ])
    def test_inconsistent_bundles_fail_closed(self, tmp_path, bundle, codebook, boundary_tokenizer):
        path = write_bundle(tmp_path / "cb", **bundle)
        with pytest.raises(ValueError):
            check_boundary_tokens(path, FakeTokenizer(boundary=boundary_tokenizer), codebook, SLIDERS)

    def test_boundary_ids_must_follow_the_sliders(self, tmp_path):
        path = write_bundle(tmp_path / "cb0")
        with pytest.raises(ValueError):
            check_boundary_tokens(path, FakeTokenizer(boundary=True), 0, SLIDERS[:-1])

    @pytest.mark.asyncio
    async def test_engine_load_refuses_the_flag_for_a_legacy_bundle(self):
        from tests.test_vv_adapter import _engine, no_models
        engine = _engine()
        engine.config.cb0_boundary_tokens = True
        with no_models(), pytest.raises(ValueError, match="CB0_BOUNDARY_TOKENS"):
            await engine.load()

    @pytest.mark.asyncio
    async def test_engine_load_keeps_a_declaring_bundles_ids(self):
        from tests.test_vv_adapter import _engine, no_models
        engine = _engine()
        engine.config.cb0_boundary_tokens = True
        codebooks = []

        def fake_check(path, tokenizer, codebook, slider_tokens=()):
            codebooks.append(codebook)
            return (END_ID, CONT_ID) if codebook == 0 else None
        with no_models(), patch("app.inference.check_boundary_tokens", side_effect=fake_check):
            await engine.load()
        assert engine.boundary_token_ids == (END_ID, CONT_ID) and codebooks == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Client text (text.py) and window length
# ---------------------------------------------------------------------------

class TestSanitizeStripsSpecialTokens:
    @pytest.mark.parametrize("raw, expected", [
        ("Hello<|utt_end|> world", "Hello world."),
        ("Stop.<|text_split|> Go.", "Stop. Go."),
        ("<|utt_cont|>", ""),
        ("a <|dc_0_17|> b", "a b."),
    ])
    def test_strips(self, raw, expected):
        assert sanitize_text_for_tts(raw) == expected

    def test_text_without_special_tokens_is_unchanged(self):
        for raw in ("hello\n\n\nworld", "  spaced\t out  ", "Price: 3.50 <tag> | pipe", "Ünïcödé text", "a < | b | >"):
            assert sanitize_text_for_tts(raw) == _legacy_sanitize(raw)


def _legacy_sanitize(text):
    """app.text.sanitize_text_for_tts before the boundary tokens, with the character cleaning of app/model/text_vocab.py (which also
    maps vertical whitespace to newlines)."""
    import app.text as T
    from app.model.text_vocab import clean_text
    text = clean_text(text)
    text = T._WHITESPACE_COLLAPSE_RE.sub(T._collapse_whitespace, text).strip()
    if text and text[-1] not in ".!?;:,":
        text += "."
    return text


class TestWindowLength:
    def test_token_window_adds_at_most_one_text_slot(self):
        tok = FakeTokenizer(boundary=True)
        legacy_engine = make_engine()
        for n in range(1, len(SEGS) + 1):
            for seg_idx in range(n):
                for drop_space in (False, True):
                    kw = dict(use_context=True, system_tag="<h>")
                    legacy = tokenize_text(legacy_engine._text_window(seg_idx, SEGS[:n], **kw)[0], tok)
                    token = tokenize_text(legacy_engine._text_window(
                        seg_idx, SEGS[:n], boundary_tokens=True, drop_space=drop_space, **kw)[0], tok)
                    assert len(token) - len(legacy) <= 1


class TestPrefixTextStrip:
    def test_strip_only_removes_the_strings(self):
        assert strip_special_token_strings("Earlier<|utt_end|> words<|text_split|>") == "Earlier words"
        assert strip_special_token_strings("no tokens, no period") == "no tokens, no period"

    @pytest.mark.asyncio
    async def test_resolve_prefix_strips_and_skips_an_empty_prefix(self):
        engine = make_engine()
        engine.tokenizers = [_tokenizer_mock()] * 3
        engine.engines = [MagicMock()] * 3
        assert await engine._resolve_prefix("<|utt_end|>", None, {}, (None, None), None, 0.0, 0) == (None, {})
        seen = []

        async def fake_semantic(segments, **kw):
            seen.append(segments)
            return [1, 2, 3], []
        engine._generate_semantic = fake_semantic
        engine._generate_acoustic = AsyncMock(return_value=[4, 5, 6])
        prefix, _ = await engine._resolve_prefix("Earlier<|utt_cont|> words", None, {}, (None, None), None, 0.0, 0)
        assert seen == [["Earlier words"]] and prefix["text"] == "Earlier words"
