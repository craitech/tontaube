"""Tests for chunked decoding, acoustic context, and CB0 retry logic."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch, call
import pytest
import torch

from app.config import (
    ServerConfig,
    DUALCODEC_SAMPLE_RATE,
    SEMANTIC_VOCAB_SIZE,
    COLLAPSE_SILENCE_TOKEN_IDS,
    ORDINARY_SILENCE_TOKEN_IDS,
    SILENCE_TOKEN_IDS,
)
from app.inference import TTSEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_engine(
    chunk_context_seconds: float = 2.0,
    cb0_max_retries: int = 3,
    cb0_min_tokens: int = 13,
    cb0_min_cps: float = 6.0,
    cb0_max_cps: float = 18.0,
    max_new_tokens: int = 625,
) -> TTSEngine:
    """Build a TTSEngine with mocked internals for unit testing."""
    config = ServerConfig(
        chunk_context_seconds=chunk_context_seconds,
        cb0_max_retries=cb0_max_retries,
        cb0_min_tokens=cb0_min_tokens,
        cb0_min_cps=cb0_min_cps,
        cb0_max_cps=cb0_max_cps,
        max_new_tokens=max_new_tokens,
    )
    engine = TTSEngine(config)

    # Mock dc_inference — codes_to_audio_tensor will be patched per-test
    engine.dc_inference = MagicMock()
    return engine


SAMPLES_PER_TOKEN = int(DUALCODEC_SAMPLE_RATE / 12.5)  # 1920


def test_silence_token_sets_keep_collapse_and_ordinary_ids_separate():
    assert SILENCE_TOKEN_IDS == COLLAPSE_SILENCE_TOKEN_IDS + ORDINARY_SILENCE_TOKEN_IDS
    assert set(COLLAPSE_SILENCE_TOKEN_IDS).isdisjoint(ORDINARY_SILENCE_TOKEN_IDS)


def fake_decode(dc_inference, sem_codes, ac_codes):
    """Produce a deterministic audio tensor: 1920 samples per semantic token, value = token."""
    n_tokens = len(sem_codes)
    n_samples = n_tokens * SAMPLES_PER_TOKEN
    audio = torch.zeros(1, 1, n_samples)
    for i, code in enumerate(sem_codes):
        audio[..., i * SAMPLES_PER_TOKEN:(i + 1) * SAMPLES_PER_TOKEN] = float(code)
    return audio


# ---------------------------------------------------------------------------
# Text boundary layout
# ---------------------------------------------------------------------------

class TestTextBoundaryLayout:
    def setup_method(self):
        self.engine = make_engine()

    def test_first_chunk_has_no_leading_space_and_final_chunk_has_newline(self):
        text, start, end = self.engine._text_window(
            0, ["  First chunk.  ", "  Last chunk.  "], use_context=True,
            lookahead_chars=None,
        )

        assert (start, end) == (0, 2)
        assert text == "First chunk.<|text_split|> Last chunk.\n"

    def test_non_initial_single_chunk_keeps_leading_space(self):
        text, start, end = self.engine._text_window(
            1, ["First chunk.", "Middle chunk.", "Last chunk."],
            use_context=False, lookahead_chars=None,
        )

        assert (start, end) == (1, 2)
        assert text == " Middle chunk."

    def test_single_chunk_is_both_start_and_end(self):
        text, _, _ = self.engine._text_window(
            0, ["  Only chunk.  "], use_context=False,
        )

        assert text == "Only chunk.\n"

    def test_streaming_early_slice_is_not_mistaken_for_final_chunk(self):
        text, _, _ = self.engine._text_window(
            0, ["First chunk."], use_context=False,
            boundary_total_segments=2,
        )

        assert text == "First chunk."

    def test_final_lookahead_remains_end_marked_after_truncation(self):
        text, _, _ = self.engine._text_window(
            0, ["First.", "This final lookahead is deliberately rather long."],
            use_context=True, lookahead_chars=20,
        )

        assert text == "First.<|text_split|> This final\n"

    def test_system_tag_does_not_add_space_to_first_chunk(self):
        text, _, _ = self.engine._text_window(
            0, ["First."], use_context=False, system_tag="<system>\n",
        )

        assert text == "<system>\nFirst.\n"


def test_empty_voice_prompt_row_matches_missing_row():
    sem = [1, 2]
    acoustic = [[3, 4], [5, 6]]
    missing = TTSEngine._front_prompt_block(3, sem, acoustic, 100)
    empty = TTSEngine._front_prompt_block(3, sem, acoustic + [[]], 100)
    assert missing == empty


@pytest.mark.asyncio
async def test_missing_prefix_codebook_is_generated():
    engine = make_engine()
    engine.engines = [MagicMock() for _ in range(4)]
    engine._cb_ctx = MagicMock(return_value=SimpleNamespace(dc_base_id=100))
    engine._generate_acoustic = AsyncMock(return_value=[30, 31])

    _, acoustic = await engine._resolve_prefix(
        "Prefix.", [[1, 2], [10, 11], [20, 21]], {}, (None, None),
        None, 0.8, -1,
    )

    assert acoustic == {1: [10, 11], 2: [20, 21], 3: [30, 31]}
    assert engine._generate_acoustic.await_count == 1
    assert engine._generate_acoustic.await_args.kwargs["codebook"] == 3


# ---------------------------------------------------------------------------
# _check_cb0_generation
# ---------------------------------------------------------------------------

class TestCheckCb0Generation:
    def setup_method(self):
        self.engine = make_engine()

    def test_hit_max_tokens(self):
        result = self.engine._check_cb0_generation(
            n_tokens=100, n_chars=50, hit_max_tokens=True, hz=12.5,
        )
        assert result == "max_tokens_reached"

    def test_too_few_tokens(self):
        result = self.engine._check_cb0_generation(
            n_tokens=5, n_chars=50, hit_max_tokens=False, hz=12.5,
        )
        assert "too_few_tokens" in result

    def test_exactly_min_tokens_passes(self):
        # 13 tokens, 50 chars → duration=1.04s → cps=48.08 → too high
        # Use chars that give a valid CPS: 13 tokens / 12.5 = 1.04s, need 6 ≤ cps ≤ 18
        # chars = 10 → cps = 10/1.04 ≈ 9.6 → OK
        result = self.engine._check_cb0_generation(
            n_tokens=13, n_chars=10, hit_max_tokens=False, hz=12.5,
        )
        assert result is None

    def test_cps_too_low(self):
        # 100 tokens / 12.5 = 8s, 40 chars → cps = 5.0 (below min 6.0)
        result = self.engine._check_cb0_generation(
            n_tokens=100, n_chars=40, hit_max_tokens=False, hz=12.5,
        )
        assert "cps_out_of_range" in result

    def test_cps_too_high(self):
        # 13 tokens / 12.5 = 1.04s, 100 chars → cps = 96.2
        result = self.engine._check_cb0_generation(
            n_tokens=13, n_chars=100, hit_max_tokens=False, hz=12.5,
        )
        assert "cps_out_of_range" in result

    def test_cps_valid(self):
        # 100 tokens / 12.5 = 8s, 80 chars → cps = 10 → valid
        result = self.engine._check_cb0_generation(
            n_tokens=100, n_chars=80, hit_max_tokens=False, hz=12.5,
        )
        assert result is None

    def test_zero_tokens_zero_chars(self):
        result = self.engine._check_cb0_generation(
            n_tokens=0, n_chars=0, hit_max_tokens=False, hz=12.5,
        )
        assert "too_few_tokens" in result

    def test_cps_boundaries(self):
        # Exactly 6.0 cps should pass: 100 tokens / 12.5 = 8s, 48 chars → 6.0
        result = self.engine._check_cb0_generation(
            n_tokens=100, n_chars=48, hit_max_tokens=False, hz=12.5,
        )
        assert result is None

        # Exactly 18.0 cps should pass: 100 tokens / 12.5 = 8s, 144 chars → 18.0
        result = self.engine._check_cb0_generation(
            n_tokens=100, n_chars=144, hit_max_tokens=False, hz=12.5,
        )
        assert result is None




# ---------------------------------------------------------------------------
# 4. CB0 retry in _generate_semantic
# ---------------------------------------------------------------------------

class TestSemanticRetry:
    """Test retry logic in _generate_semantic."""

    def setup_method(self):
        self.engine = make_engine(cb0_max_retries=3, max_new_tokens=625)
        self.engine.config.force_first_silence = False  # disable to simplify token count assertions
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.convert_tokens_to_ids = lambda t: {
            '<|start_of_speech|>': 1,
            '<|dc_0_0|>': 100,
            '<|end_of_speech|>': 2,
            '<|audio_split|>': 3,
        }.get(t, 99)
        self.engine.tokenizers = [tokenizer, tokenizer, tokenizer]
        self.engine.engines = [MagicMock(), MagicMock(), MagicMock()]

        self.dc_base = 100
        self.eos_id = 2

    def _make_good_output(self, n_tokens=50, text_len=80):
        """Generate output that passes all checks. CPS = text_len / (n_tokens/12.5)"""
        return [self.dc_base + i for i in range(n_tokens)] + [self.eos_id]

    def _make_bad_short_output(self):
        """< 13 tokens → too_few_tokens."""
        return [self.dc_base + i for i in range(5)] + [self.eos_id]

    @pytest.mark.asyncio
    @patch("app.inference.tokenize_text", return_value=[10, 11, 12])
    async def test_succeeds_first_try(self, mock_tok):
        # 50 tokens → 4s, need CPS 6-18 → need 24-72 chars
        text = "a]" * 25  # 50 chars → CPS=12.5 ✓
        good = self._make_good_output(50)
        self.engine._generate_with_engine = AsyncMock(return_value=good)

        codes, markers = await self.engine._generate_semantic(
            [text],
            sampling={"temperature": 0.9, "top_p": 0.95, "top_k": 10,
                       "frequency_penalty": 0.0, "seed": None},
            voice_prompt=(None, None),
        )

        assert len(codes) == 50
        assert markers == []
        assert self.engine._generate_with_engine.call_count == 1

    @pytest.mark.asyncio
    @patch("app.inference.tokenize_text", return_value=[10, 11, 12])
    async def test_token_budget_includes_forced_initial_silence(self, mock_tok):
        self.engine.config.force_first_silence = True
        output = self._make_good_output(12)
        self.engine._generate_with_engine = AsyncMock(return_value=output)

        with patch("app.inference.SamplingParams",
                   side_effect=lambda **kwargs: SimpleNamespace(**kwargs)):
            codes, _ = await self.engine._generate_semantic(
                ["short text"],
                sampling={"temperature": 0.9, "top_p": 0.95, "top_k": 10,
                          "frequency_penalty": 0.0, "seed": None},
                voice_prompt=(None, None),
                max_new_tokens=50,
            )

        params = self.engine._generate_with_engine.await_args.args[2]
        assert params.max_tokens == 49
        assert params.min_tokens == 12
        assert len(codes) == 13

    @pytest.mark.asyncio
    @patch("app.inference.tokenize_text", return_value=[10, 11, 12])
    async def test_streaming_prefix_counts_toward_total_token_budget(self, mock_tok):
        early_prefix = list(range(30))
        self.engine._generate_with_engine = AsyncMock(return_value=[self.eos_id])

        with patch("app.inference.SamplingParams",
                   side_effect=lambda **kwargs: SimpleNamespace(**kwargs)):
            codes, _ = await self.engine._generate_semantic(
                ["short text"],
                sampling={"temperature": 0.9, "top_p": 0.95, "top_k": 10,
                          "frequency_penalty": 0.0, "seed": None},
                voice_prompt=(None, None),
                max_new_tokens=50,
                first_seg_forced_prefix=early_prefix,
            )

        params = self.engine._generate_with_engine.await_args.args[2]
        assert params.max_tokens == 20
        assert params.min_tokens == 0
        assert len(codes) == 30

    @pytest.mark.asyncio
    @patch("app.inference.tokenize_text", return_value=[10, 11, 12])
    async def test_retries_on_too_few_tokens(self, mock_tok):
        text = "a" * 50
        bad = self._make_bad_short_output()
        good = self._make_good_output(50)
        self.engine._generate_with_engine = AsyncMock(side_effect=[bad, good])

        codes, markers = await self.engine._generate_semantic(
            [text],
            sampling={"temperature": 0.9, "top_p": 0.95, "top_k": 10,
                       "frequency_penalty": 0.0, "seed": None},
            voice_prompt=(None, None),
        )

        assert len(codes) == 50
        assert self.engine._generate_with_engine.call_count == 2

    @pytest.mark.asyncio
    @patch("app.inference.tokenize_text", return_value=[10, 11, 12])
    async def test_collapse_bias_scales_on_retry_while_ordinary_bias_stays_fixed(self, mock_tok):
        text = "a" * 50
        bad = self._make_bad_short_output()
        good = self._make_good_output(50)
        self.engine._generate_with_engine = AsyncMock(side_effect=[bad, good])

        with patch("app.inference.SamplingParams",
                   side_effect=lambda **kwargs: SimpleNamespace(**kwargs)):
            await self.engine._generate_semantic(
                [text],
                sampling={"temperature": 0.9, "top_p": 0.95, "top_k": 10,
                          "frequency_penalty": 0.0, "seed": None},
                voice_prompt=(None, None),
                silence_logit_bias=-0.7,
            )

        first_params = self.engine._generate_with_engine.call_args_list[0].args[2]
        retry_params = self.engine._generate_with_engine.call_args_list[1].args[2]
        for token_id in COLLAPSE_SILENCE_TOKEN_IDS:
            global_id = self.dc_base + token_id
            assert first_params.logit_bias[global_id] == pytest.approx(-0.7)
            assert retry_params.logit_bias[global_id] == pytest.approx(-1.0)
        for token_id in ORDINARY_SILENCE_TOKEN_IDS:
            global_id = self.dc_base + token_id
            assert first_params.logit_bias[global_id] == pytest.approx(-0.2)
            assert retry_params.logit_bias[global_id] == pytest.approx(-0.2)

    @pytest.mark.asyncio
    @patch("app.inference.tokenize_text", return_value=[10, 11, 12])
    async def test_retries_on_max_tokens(self, mock_tok):
        text = "a" * 15  # 15 chars / (15/12.5)s = 12.5 CPS ✓
        self.engine.config.max_new_tokens = 20
        # Output exactly max_tokens without EOS → hit_max
        no_eos = [self.dc_base + i for i in range(20)]
        good = self._make_good_output(15)
        self.engine._generate_with_engine = AsyncMock(side_effect=[no_eos, good])

        codes, markers = await self.engine._generate_semantic(
            [text],
            sampling={"temperature": 0.9, "top_p": 0.95, "top_k": 10,
                       "frequency_penalty": 0.0, "seed": None},
            voice_prompt=(None, None),
        )

        assert len(codes) == 15
        assert self.engine._generate_with_engine.call_count == 2

    @pytest.mark.asyncio
    @patch("app.inference.tokenize_text", return_value=[10, 11, 12])
    async def test_exhausts_retries(self, mock_tok):
        bad = self._make_bad_short_output()
        self.engine._generate_with_engine = AsyncMock(return_value=bad)

        codes, markers = await self.engine._generate_semantic(
            ["hello world test text"],
            sampling={"temperature": 0.9, "top_p": 0.95, "top_k": 10,
                       "frequency_penalty": 0.0, "seed": None},
            voice_prompt=(None, None),
        )

        # Returns last attempt's result after exhausting retries
        assert len(codes) == 5
        assert self.engine._generate_with_engine.call_count == 3


# ---------------------------------------------------------------------------
# 5. CB0 retry in _generate_semantic (multi-segment)
# ---------------------------------------------------------------------------

class TestSemanticChunkedRetry:
    """Test per-chunk retry logic in _generate_semantic (multi-segment)."""

    def setup_method(self):
        self.engine = make_engine(cb0_max_retries=3, max_new_tokens=625)
        self.engine.config.force_first_silence = False
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.convert_tokens_to_ids = lambda t: {
            '<|start_of_speech|>': 1,
            '<|dc_0_0|>': 100,
            '<|end_of_speech|>': 2,
            '<|audio_split|>': 3,
            '<|text_split|>': 4,
        }.get(t, 99)
        self.engine.tokenizers = [tokenizer, tokenizer, tokenizer]
        self.engine.engines = [MagicMock(), MagicMock(), MagicMock()]

        self.dc_base = 100
        self.marker_id = 3

    def _good_chunk(self, n=30):
        return [self.dc_base + i for i in range(n)] + [self.marker_id]

    def _bad_chunk(self):
        return [self.dc_base + i for i in range(3)] + [self.marker_id]

    @pytest.mark.asyncio
    @patch("app.inference.tokenize_text", return_value=[10, 11, 12])
    async def test_chunk_retries_on_failure(self, mock_tok):
        bad = self._bad_chunk()
        good1 = self._good_chunk(30)
        good2 = self._good_chunk(25)
        # chunk 0: fails then succeeds. chunk 1: succeeds first try.
        self.engine._generate_with_engine = AsyncMock(
            side_effect=[bad, good1, good2]
        )

        codes, markers = await self.engine._generate_semantic(
            segments=["hello world test.", "another sentence here."],
            sampling={"temperature": 0.9, "top_p": 0.95, "top_k": 10,
                       "frequency_penalty": 0.0, "seed": None},
            voice_prompt=(None, None),
        )

        assert len(codes) == 55  # 30 + 25
        assert markers == [30]
        assert self.engine._generate_with_engine.call_count == 3


# ---------------------------------------------------------------------------
# 6. Semantic context
# ---------------------------------------------------------------------------

class TestUseContextForCb:
    """Test that use_context parameter correctly controls windowing."""

    def setup_method(self):
        # Disable CPS check — these tests don't test retry logic
        self.engine = make_engine(max_new_tokens=625, cb0_min_cps=0.0, cb0_max_cps=9999.0)
        self.engine.config.force_first_silence = False
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.convert_tokens_to_ids = lambda t: {
            '<|start_of_speech|>': 1,
            '<|dc_0_0|>': 100,
            '<|end_of_speech|>': 2,
            '<|audio_split|>': 3,
            '<|text_split|>': 4,
        }.get(t, 99)
        self.engine.tokenizers = [tokenizer, tokenizer, tokenizer]
        self.engine.engines = [MagicMock(), MagicMock(), MagicMock()]
        self.dc_base = 100
        self.marker_id = 3

    @pytest.mark.asyncio
    @patch("app.inference.tokenize_text")
    async def test_semantic_chunked_no_context(self, mock_tok):
        """use_context=False: no text windowing, no prev audio context."""
        mock_tok.return_value = [10, 11]
        good = [self.dc_base + i for i in range(20)] + [self.marker_id]
        self.engine._generate_with_engine = AsyncMock(return_value=good)

        segments = ["seg one.", "seg two.", "seg three."]

        await self.engine._generate_semantic(
            segments=segments,
            sampling={"temperature": 0.9, "top_p": 0.95, "top_k": 10,
                       "frequency_penalty": 0.0, "seed": None},
            voice_prompt=(None, None),
            use_context=False,
        )

        # With use_context=False, tokenize_text should be called with each
        # individual segment (no windowing), never with TEXT_MARKER joined text
        for c in mock_tok.call_args_list:
            text_arg = c[0][0]
            assert '<|text_split|>' not in text_arg

    @pytest.mark.asyncio
    @patch("app.inference.tokenize_text")
    async def test_semantic_chunked_with_context(self, mock_tok):
        """use_context=True: text windowing includes neighbors."""
        mock_tok.return_value = [10, 11]
        good = [self.dc_base + i for i in range(20)] + [self.marker_id]
        self.engine._generate_with_engine = AsyncMock(return_value=good)

        segments = ["seg one.", "seg two.", "seg three."]

        await self.engine._generate_semantic(
            segments=segments,
            sampling={"temperature": 0.9, "top_p": 0.95, "top_k": 10,
                       "frequency_penalty": 0.0, "seed": None},
            voice_prompt=(None, None),
            use_context=True,
        )

        # With context, at least some tokenize_text calls should have windowed text
        texts = [c[0][0] for c in mock_tok.call_args_list]
        # chunk 0: window includes seg 0 + seg 1 → has text_split
        assert '<|text_split|>' in texts[0]

    @pytest.mark.asyncio
    @patch("app.inference.tokenize_text")
    async def test_semantic_no_context_skips_prev_audio(self, mock_tok):
        """use_context=False should not include previous chunk audio tokens."""
        mock_tok.return_value = [10, 11]
        good = [self.dc_base + i for i in range(20)] + [self.marker_id]
        self.engine._generate_with_engine = AsyncMock(return_value=good)

        await self.engine._generate_semantic(
            segments=["seg one.", "seg two."],
            sampling={"temperature": 0.9, "top_p": 0.95, "top_k": 10,
                       "frequency_penalty": 0.0, "seed": None},
            voice_prompt=(None, None),
            use_context=False,
        )

        # Both calls should have the same input length (no prev audio appended)
        call1_input = self.engine._generate_with_engine.call_args_list[0][0][1]
        call2_input = self.engine._generate_with_engine.call_args_list[1][0][1]
        assert len(call1_input) == len(call2_input)

    @pytest.mark.asyncio
    @patch("app.inference.tokenize_text")
    async def test_semantic_with_context_includes_prev_audio(self, mock_tok):
        """use_context=True should include previous chunk audio tokens for chunk 1+."""
        mock_tok.return_value = [10, 11]
        good = [self.dc_base + i for i in range(20)] + [self.marker_id]
        self.engine._generate_with_engine = AsyncMock(return_value=good)

        await self.engine._generate_semantic(
            segments=["seg one.", "seg two."],
            sampling={"temperature": 0.9, "top_p": 0.95, "top_k": 10,
                       "frequency_penalty": 0.0, "seed": None},
            voice_prompt=(None, None),
            use_context=True,
        )

        call1_input = self.engine._generate_with_engine.call_args_list[0][0][1]
        call2_input = self.engine._generate_with_engine.call_args_list[1][0][1]
        # Chunk 1 should have prev audio (20 tokens + marker) → longer input
        assert len(call2_input) > len(call1_input)


# ---------------------------------------------------------------------------
# 7. Acoustic use_context (multi-segment)
# ---------------------------------------------------------------------------

class TestAcousticChunkedUseContext:
    """Test use_context in _generate_acoustic (multi-segment)."""

    def setup_method(self):
        self.engine = make_engine(max_new_tokens=625)
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.convert_tokens_to_ids = lambda t: {
            '<|start_of_speech|>': 1,
            '<|dc_0_0|>': 100,
            '<|end_of_speech|>': 2,
            '<|audio_split|>': 3,
            '<|text_split|>': 4,
        }.get(t, 99)
        self.engine.tokenizers = [tokenizer, tokenizer, tokenizer]
        self.engine.engines = [MagicMock(), MagicMock(), MagicMock()]

    @pytest.mark.asyncio
    @patch("app.inference.tokenize_text")
    async def test_no_context_no_prev_segment(self, mock_tok):
        """use_context=False: second segment should not include prev segment's codes."""
        mock_tok.return_value = [10, 11]
        dc_base = 100
        offset = SEMANTIC_VOCAB_SIZE  # cb1 offset

        # Generate cb1 codes for 2 segments (20 tokens each)
        cb1_output = [dc_base + offset + i for i in range(20)]
        self.engine._generate_with_engine = AsyncMock(return_value=cb1_output)

        sem_codes = list(range(40))
        markers = [20]

        result = await self.engine._generate_acoustic(
            segments=["seg one.", "seg two."],
            semantic_codes=sem_codes,
            marker_indices=markers,
            prev_acoustic=[],
            codebook=1,
            voice_prompt=(None, None),
            use_context=False,
        )

        # Both segment inputs should be same length (no prev segment appended)
        call1_input = self.engine._generate_with_engine.call_args_list[0][0][1]
        call2_input = self.engine._generate_with_engine.call_args_list[1][0][1]
        assert len(call1_input) == len(call2_input)

    @pytest.mark.asyncio
    @patch("app.inference.tokenize_text")
    async def test_with_context_includes_prev_segment(self, mock_tok):
        """use_context=True: second segment should include prev segment's codes."""
        mock_tok.return_value = [10, 11]
        dc_base = 100
        offset = SEMANTIC_VOCAB_SIZE

        cb1_output = [dc_base + offset + i for i in range(20)]
        self.engine._generate_with_engine = AsyncMock(return_value=cb1_output)

        sem_codes = list(range(40))
        markers = [20]

        result = await self.engine._generate_acoustic(
            segments=["seg one.", "seg two."],
            semantic_codes=sem_codes,
            marker_indices=markers,
            prev_acoustic=[],
            codebook=1,
            voice_prompt=(None, None),
            use_context=True,
        )

        call1_input = self.engine._generate_with_engine.call_args_list[0][0][1]
        call2_input = self.engine._generate_with_engine.call_args_list[1][0][1]
        # Segment 1 should be longer (includes prev segment + windowed text)
        assert len(call2_input) > len(call1_input)


# ---------------------------------------------------------------------------
# 8. generate() integration: fixed context routing
# ---------------------------------------------------------------------------

class TestGenerateContextRouting:
    """Test the released context strategy used by generate()."""

    def setup_method(self):
        self.engine = make_engine()
        tokenizer = MagicMock()
        tokenizer.pad_token_id = 0
        tokenizer.convert_tokens_to_ids = lambda t: {
            '<|start_of_speech|>': 1,
            '<|dc_0_0|>': 100,
            '<|end_of_speech|>': 2,
            '<|audio_split|>': 3,
            '<|text_split|>': 4,
        }.get(t, 99)
        self.engine.tokenizers = [tokenizer, tokenizer, tokenizer]
        self.engine.engines = [MagicMock(), MagicMock(), MagicMock()]
        self.engine.vv_tokenizer = None

    @pytest.mark.asyncio
    @patch("app.inference.decode_chunked_with_context",
           return_value=(torch.ones(1, 1, 1000), torch.ones(1, 1, 1000)))
    async def test_semantic_context_and_independent_acoustics(self, mock_decode):
        """CB0 uses chunk context while acoustic chunks remain independent."""
        sem_mock = AsyncMock(return_value=(list(range(60)), [30]))
        ac_mock = AsyncMock(return_value=list(range(60)))
        self.engine._generate_semantic = sem_mock
        self.engine._generate_acoustic = ac_mock

        await self.engine.generate(
            text=["seg one.", "seg two."],
        )

        # Semantic should be called with use_context=True (0 is in [0])
        sem_mock.assert_called_once()
        assert sem_mock.call_args[1]["use_context"] is True

        # Acoustic calls: cb1 and cb2, both should have use_context=False
        assert ac_mock.call_count == 2
        for c in ac_mock.call_args_list:
            assert c[1]["use_context"] is False

    @pytest.mark.asyncio
    @patch("app.inference.decode_chunked_with_context",
           return_value=(torch.ones(1, 1, 1000), torch.ones(1, 1, 1000)))
    async def test_default_context_only_cb0(self, mock_decode):
        """The context strategy is stable when callers use default arguments."""
        sem_mock = AsyncMock(return_value=(list(range(60)), [30]))
        ac_mock = AsyncMock(return_value=list(range(60)))
        self.engine._generate_semantic = sem_mock
        self.engine._generate_acoustic = ac_mock

        await self.engine.generate(
            text=["seg one.", "seg two."],
        )

        assert sem_mock.call_args[1]["use_context"] is True
        # Default is [0] — acoustic codebooks don't use context
        for c in ac_mock.call_args_list:
            assert c[1]["use_context"] is False
