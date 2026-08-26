from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from app.model.vllm_model import Qwen3DualCodecForCausalLM


DC_BASE_ID = 151665
PAD_ID = 151643
EOS_ID = 151662
TEXT_MARKER_ID = 151663
AUDIO_MARKER_ID = 151664


class _PositionRecorder(torch.nn.Module):
    def forward(
        self,
        input_ids,
        positions,
        intermediate_tensors=None,
        inputs_embeds=None,
    ):
        return positions.clone()


@pytest.fixture
def position_adapter():
    adapter = Qwen3DualCodecForCausalLM.__new__(Qwen3DualCodecForCausalLM)
    torch.nn.Module.__init__(adapter)
    adapter.config = SimpleNamespace(max_position_embeddings=40960)
    adapter.dc_base_id = DC_BASE_ID
    adapter.pad_token_id = PAD_ID
    adapter.eos_global_id = EOS_ID
    adapter.text_marker_global_id = TEXT_MARKER_ID
    adapter.audio_marker_global_id = AUDIO_MARKER_ID
    adapter.chunk_realign_offset = 25
    adapter.target_codebook = 0
    adapter.model = _PositionRecorder()
    return adapter


def test_decode_translation_preserves_pairwise_differences(position_adapter):
    raw = torch.tensor([7, 10, 18], dtype=torch.long)
    translated = position_adapter(torch.tensor([1, 2, 3]), raw)

    assert torch.equal(translated, raw + 25)
    assert torch.equal(
        translated[:, None] - translated[None, :],
        raw[:, None] - raw[None, :],
    )


def test_short_continuation_prefill_no_longer_collapses(position_adapter):
    # A long prompt followed by one-character prefix/current text segments and
    # a 13-token semantic prefix previously translated below zero.
    ids = (
        [DC_BASE_ID + 5] * 750
        + [100, TEXT_MARKER_ID, 101, PAD_ID]
        + [DC_BASE_ID + 6] * 13
        + [AUDIO_MARKER_ID]
    )
    input_ids = torch.tensor(ids, dtype=torch.long)
    raw = position_adapter._compute_prefill_positions(input_ids.unsqueeze(0))
    translated = position_adapter(
        input_ids,
        torch.arange(len(ids), dtype=torch.long),
    )

    assert raw.min().item() < 0
    assert torch.equal(translated, raw + 25)
    assert translated.min().item() >= 0
    assert translated[:10].unique().numel() == 10


def test_ordinary_prefill_aligns_with_next_decode_position(position_adapter):
    input_ids = torch.tensor([100, 101, 102, PAD_ID], dtype=torch.long)
    prefill = position_adapter(
        input_ids,
        torch.arange(input_ids.numel(), dtype=torch.long),
    )
    decode = position_adapter(
        torch.tensor([DC_BASE_ID + 1]),
        torch.tensor([input_ids.numel()], dtype=torch.long),
    )

    assert prefill[-1].item() == input_ids.numel() - 1 + 25
    assert decode.item() == input_ids.numel() + 25
    assert decode.item() - prefill[-1].item() == 1


def test_warmup_and_cuda_graph_positions_receive_translation(position_adapter):
    raw = torch.arange(4, dtype=torch.long)
    dummy_ids = torch.tensor([1, 2, 3, 4], dtype=torch.long)

    warmup = position_adapter(dummy_ids, raw)
    assert torch.equal(warmup, raw + 25)

    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.is_current_stream_capturing", return_value=True),
    ):
        captured = position_adapter(dummy_ids, raw)
    assert torch.equal(captured, raw + 25)
