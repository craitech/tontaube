"""Tests asserting assumptions from InputScheme_formal.md.

Each test references the relevant section/equation from the formal spec.
"""
import pytest

from app.config import SEMANTIC_VOCAB_SIZE, ACOUSTIC_VOCAB_SIZE
from app.model.vllm_model import get_codebook_offset
from app.dualcodec_utils import split_at_markers
from app.inference import TTSEngine, _extract_acoustic_codes


# ---------------------------------------------------------------------------
# Fixtures — concrete token IDs matching the spec notation
# ---------------------------------------------------------------------------

DC_BASE_ID = 151665  # realistic base id from Qwen tokenizer
PAD_ID = 151643
SOS_ID = 151661
EOS_ID = 151662
AUDIO_MARKER_ID = 151664


# ---------------------------------------------------------------------------
# §Properties — codebook offset layout
# ---------------------------------------------------------------------------

class TestCodebookOffset:
    """get_codebook_offset encodes the token-id layout assumed everywhere."""

    def test_cb0_offset_is_zero(self):
        """C^0 tokens start at dc_base_id + 0."""
        assert get_codebook_offset(0) == 0

    def test_general_formula(self):
        """offset(i) = SEMANTIC_VOCAB_SIZE + (i-1) * ACOUSTIC_VOCAB_SIZE for i >= 1."""
        for i in range(1, 5):
            expected = SEMANTIC_VOCAB_SIZE + (i - 1) * ACOUSTIC_VOCAB_SIZE
            assert get_codebook_offset(i) == expected


# ---------------------------------------------------------------------------
# Front voice-prompt block
# ---------------------------------------------------------------------------

class TestFrontPromptBlock:
    """Voice-prompt codebooks are serialized before text without row padding."""

    def test_cb0_no_leading_pad(self):
        prompt = [10, 20, 30]
        ids = TTSEngine._front_prompt_block(0, prompt, None, DC_BASE_ID)
        assert ids == [DC_BASE_ID + code for code in prompt]

    def test_cb1_concatenates_prompt_streams(self):
        semantic = [1, 2]
        acoustic = [[3, 4]]
        offset = get_codebook_offset(1)
        ids = TTSEngine._front_prompt_block(1, semantic, acoustic, DC_BASE_ID)
        assert ids == [
            DC_BASE_ID + 1,
            DC_BASE_ID + 2,
            DC_BASE_ID + offset + 3,
            DC_BASE_ID + offset + 4,
        ]

    def test_cb2_uses_cb2_offset(self):
        acoustic = [[6], [7]]
        offset = get_codebook_offset(2)
        ids = TTSEngine._front_prompt_block(2, [5], acoustic, DC_BASE_ID)
        assert ids[-1] == DC_BASE_ID + offset + 7

    def test_empty_prompt(self):
        assert TTSEngine._front_prompt_block(0, None, None, DC_BASE_ID) == []
        assert TTSEngine._front_prompt_block(1, None, None, DC_BASE_ID) == []


# ---------------------------------------------------------------------------
# Audio-row construction
# ---------------------------------------------------------------------------

class TestAudioRow:
    """Audio rows begin with PAD and use codebook-specific token offsets."""

    def _make_segments(self, lengths):
        """Create code segments with distinguishable local codes."""
        segs = []
        code = 0
        for n in lengths:
            segs.append(list(range(code, code + n)))
            code += n
        return segs

    def test_cb0_has_markers_between_segments(self):
        """δ_0 = ⟨am⟩ — cb0 inserts audio markers between windowed segments."""
        segs = self._make_segments([3, 4, 5])
        ids = TTSEngine._audio_row(
            0, segs, DC_BASE_ID, PAD_ID,
            prefix_segment=None, win_start=0, win_end=3, seg_offset=0,
            marker_id=AUDIO_MARKER_ID,
        )
        assert ids.count(AUDIO_MARKER_ID) == 2  # between seg0-seg1, seg1-seg2

    def test_cb1_no_markers(self):
        """δ_x = ε for x > 0 — acoustic codebooks have NO markers."""
        segs = self._make_segments([3, 4, 5])
        offset = get_codebook_offset(1)
        ids = TTSEngine._audio_row(
            1, segs, DC_BASE_ID, PAD_ID,
            prefix_segment=None, win_start=0, win_end=3, seg_offset=0,
            marker_id=None,  # no markers for cb1+
        )
        assert AUDIO_MARKER_ID not in ids

    def test_cb0_eos_appended_for_acoustic_generation(self):
        """[x=0 ∧ i>0] ⟨eos⟩ — cb0 block ends with EOS when building input for cb1+."""
        segs = self._make_segments([5])
        ids = TTSEngine._audio_row(
            0, segs, DC_BASE_ID, PAD_ID,
            prefix_segment=None, win_start=0, win_end=1, seg_offset=0,
            marker_id=AUDIO_MARKER_ID, eos_id=EOS_ID,
        )
        assert ids[-1] == EOS_ID

    def test_cb0_no_eos_for_semantic_generation(self):
        """cb0 block has no EOS when generating semantic tokens (eos_id=None)."""
        segs = self._make_segments([5])
        ids = TTSEngine._audio_row(
            0, segs, DC_BASE_ID, PAD_ID,
            prefix_segment=None, win_start=0, win_end=1, seg_offset=0,
            marker_id=AUDIO_MARKER_ID, eos_id=None,
        )
        assert EOS_ID not in ids

    def test_prefix_segment_followed_by_marker(self):
        """C̃^0 includes trailing ⟨am⟩ — prefix codes + marker for cb0."""
        segs = self._make_segments([3, 4])
        prefix_seg = [100, 101]
        ids = TTSEngine._audio_row(
            0, segs, DC_BASE_ID, PAD_ID,
            prefix_segment=prefix_seg, win_start=0, win_end=2, seg_offset=0,
            marker_id=AUDIO_MARKER_ID,
        )
        # After prompt prefix (just PAD for cb0 with no prompt), we should see:
        # prefix_codes, marker, seg0_codes, marker, seg1_codes
        prefix_end = ids.index(DC_BASE_ID + 100)
        marker_after_prefix = ids[prefix_end + len(prefix_seg)]
        assert marker_after_prefix == AUDIO_MARKER_ID

    def test_prefix_segment_no_marker_for_cb1(self):
        """C̃^i (i>0) has no trailing marker."""
        segs = self._make_segments([3])
        offset = get_codebook_offset(1)
        prefix_seg = [50, 51]
        ids = TTSEngine._audio_row(
            1, segs, DC_BASE_ID, PAD_ID,
            prefix_segment=prefix_seg, win_start=0, win_end=1, seg_offset=0,
            marker_id=None,
        )
        assert AUDIO_MARKER_ID not in ids

    def test_seg_offset_skips_prefix_segment(self):
        """seg_offset=1 means code_segments[0] is prefix, real segments start at [1]."""
        # segments: [prefix_codes, real_seg0, real_seg1]
        segs = [[99], [10, 11, 12], [20, 21]]
        ids = TTSEngine._audio_row(
            0, segs, DC_BASE_ID, PAD_ID,
            prefix_segment=segs[0], win_start=0, win_end=2, seg_offset=1,
            marker_id=AUDIO_MARKER_ID,
        )
        # Should contain prefix_seg codes, marker, real_seg0 codes, marker, real_seg1 codes
        # real_seg0 is segs[0+1]=segs[1], real_seg1 is segs[1+1]=segs[2]
        audio_codes = [t for t in ids if DC_BASE_ID <= t < DC_BASE_ID + SEMANTIC_VOCAB_SIZE]
        local = [t - DC_BASE_ID for t in audio_codes]
        assert local == [99, 10, 11, 12, 20, 21]

    def test_front_prompt_precedes_audio_rows(self):
        prompt0 = [1, 2]
        prompt1 = [3, 4]
        segs = self._make_segments([5])

        front = TTSEngine._front_prompt_block(
            1, prompt0, [prompt1], DC_BASE_ID,
        )
        a0 = TTSEngine._audio_row(
            0, segs, DC_BASE_ID, PAD_ID,
            prefix_segment=None, win_start=0, win_end=1, seg_offset=0,
            marker_id=AUDIO_MARKER_ID, eos_id=EOS_ID,
        )
        a1 = TTSEngine._audio_row(
            1, segs, DC_BASE_ID, PAD_ID,
            prefix_segment=None, win_start=0, win_end=1, seg_offset=0,
        )

        assert front[0] == DC_BASE_ID + 1
        assert front[-1] == DC_BASE_ID + get_codebook_offset(1) + 4
        assert a0[0] == PAD_ID
        assert a0[-1] == EOS_ID
        assert a1[0] == PAD_ID


# ---------------------------------------------------------------------------
# split_at_markers — boundary splitting
# ---------------------------------------------------------------------------

class TestSplitAtMarkers:
    def test_no_markers(self):
        """No markers → single segment containing all codes."""
        codes = [1, 2, 3, 4, 5]
        result = split_at_markers(codes, [])
        assert result == [[1, 2, 3, 4, 5]]

    def test_single_marker(self):
        """One marker splits into two segments."""
        codes = [10, 11, 12, 13, 14]
        result = split_at_markers(codes, [2])
        assert result == [[10, 11], [12, 13, 14]]

    def test_multiple_markers(self):
        """Multiple markers produce n+1 segments."""
        codes = list(range(10))
        result = split_at_markers(codes, [3, 7])
        assert result == [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9]]

    def test_marker_at_start(self):
        """Marker at position 0 → empty first segment."""
        codes = [1, 2, 3]
        result = split_at_markers(codes, [0])
        assert result == [[], [1, 2, 3]]

    def test_marker_at_end(self):
        """Marker at last position → empty last segment."""
        codes = [1, 2, 3]
        result = split_at_markers(codes, [3])
        assert result == [[1, 2, 3], []]


# ---------------------------------------------------------------------------
# _extract_acoustic_codes — Property 2: |C^i_k| = |C^0_k|
# ---------------------------------------------------------------------------

class TestExtractAcousticCodes:
    """_extract_acoustic_codes ensures acoustic output matches semantic length."""

    def test_pads_to_n_tokens(self):
        """Output length always equals n_tokens (Property 2: |C^i_k| = |C^0_k|)."""
        start_id = DC_BASE_ID + SEMANTIC_VOCAB_SIZE
        output_ids = [start_id + 5, start_id + 10]
        result = _extract_acoustic_codes(output_ids, n_tokens=5, start_id=start_id, pad_id=PAD_ID)
        assert len(result) == 5
        assert result[:2] == [5, 10]
        assert result[2:] == [0, 0, 0]

    def test_filters_pad_tokens(self):
        """PAD tokens in output are skipped."""
        start_id = DC_BASE_ID + SEMANTIC_VOCAB_SIZE
        output_ids = [PAD_ID, start_id + 1, PAD_ID, start_id + 2]
        result = _extract_acoustic_codes(output_ids, n_tokens=4, start_id=start_id, pad_id=PAD_ID)
        assert result == [1, 2, 0, 0]

    def test_out_of_range_mapped_to_zero(self):
        """Tokens outside [start_id, start_id + ACOUSTIC_VOCAB_SIZE) → 0."""
        start_id = DC_BASE_ID + SEMANTIC_VOCAB_SIZE
        out_of_range = start_id + ACOUSTIC_VOCAB_SIZE + 100
        output_ids = [out_of_range, start_id + 3]
        result = _extract_acoustic_codes(output_ids, n_tokens=3, start_id=start_id, pad_id=PAD_ID)
        assert result == [0, 3, 0]

    def test_exact_length_no_padding(self):
        """When output has exactly n_tokens valid codes, no zero-padding needed."""
        start_id = DC_BASE_ID + SEMANTIC_VOCAB_SIZE
        output_ids = [start_id + i for i in range(4)]
        result = _extract_acoustic_codes(output_ids, n_tokens=4, start_id=start_id, pad_id=PAD_ID)
        assert result == [0, 1, 2, 3]
        assert len(result) == 4


# ---------------------------------------------------------------------------
# Integration: audio rows match the released assembly
# ---------------------------------------------------------------------------

class TestSpecAssembly:
    """End-to-end tests for rows after the separate front-prompt block."""

    def test_single_segment_cb0_for_acoustic(self):
        segment_codes = [[10, 11, 12, 13]]
        ids = TTSEngine._audio_row(
            0, segment_codes, DC_BASE_ID, PAD_ID,
            prefix_segment=None, win_start=0, win_end=1, seg_offset=0,
            marker_id=AUDIO_MARKER_ID, eos_id=EOS_ID,
        )
        expected = [
            PAD_ID,
            DC_BASE_ID + 10, DC_BASE_ID + 11, DC_BASE_ID + 12, DC_BASE_ID + 13,  # segment
            EOS_ID,
        ]
        assert ids == expected

    def test_two_segment_cb0_with_context(self):
        segments = [[10, 11], [20, 21, 22]]
        ids = TTSEngine._audio_row(
            0, segments, DC_BASE_ID, PAD_ID,
            prefix_segment=None, win_start=0, win_end=2, seg_offset=0,
            marker_id=AUDIO_MARKER_ID, eos_id=EOS_ID,
        )
        expected = [
            PAD_ID,
            DC_BASE_ID + 10, DC_BASE_ID + 11,  # seg 0
            AUDIO_MARKER_ID,   # δ_0
            DC_BASE_ID + 20, DC_BASE_ID + 21, DC_BASE_ID + 22,  # seg 1
            EOS_ID,
        ]
        assert ids == expected

    def test_cb1_block_no_markers_no_eos(self):
        offset = get_codebook_offset(1)
        segments = [[30, 31], [40, 41, 42]]
        ids = TTSEngine._audio_row(
            1, segments, DC_BASE_ID, PAD_ID,
            prefix_segment=None, win_start=0, win_end=2, seg_offset=0,
        )
        expected = [
            PAD_ID,
            DC_BASE_ID + offset + 30, DC_BASE_ID + offset + 31,  # seg 0
            DC_BASE_ID + offset + 40, DC_BASE_ID + offset + 41, DC_BASE_ID + offset + 42,  # seg 1
        ]
        assert ids == expected

    def test_prefix_with_two_real_segments(self):
        prefix = [90, 91]
        segs = [prefix, [10, 11], [20, 21, 22]]
        ids = TTSEngine._audio_row(
            0, segs, DC_BASE_ID, PAD_ID,
            prefix_segment=prefix, win_start=0, win_end=2, seg_offset=1,
            marker_id=AUDIO_MARKER_ID, eos_id=EOS_ID,
        )
        expected = [
            PAD_ID,
            DC_BASE_ID + 90, DC_BASE_ID + 91,  # prefix
            AUDIO_MARKER_ID,   # marker after prefix
            DC_BASE_ID + 10, DC_BASE_ID + 11,  # real seg 0 (segs[1])
            AUDIO_MARKER_ID,   # marker between real segs
            DC_BASE_ID + 20, DC_BASE_ID + 21, DC_BASE_ID + 22,  # real seg 1 (segs[2])
            EOS_ID,
        ]
        assert ids == expected
