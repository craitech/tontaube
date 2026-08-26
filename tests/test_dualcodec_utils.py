from unittest.mock import patch

import pytest
import torch

from app.dualcodec_utils import decode_chunked_with_context


@pytest.mark.parametrize("n_tokens", [750, 751])
def test_raw_decode_is_returned_without_vibevoice_at_and_above_60_seconds(n_tokens):
    def fake_decode(_dc_inference, semantic_codes, _acoustic_codes):
        return torch.ones(1, 1, len(semantic_codes))

    semantic_codes = list(range(n_tokens))
    acoustic_codes = [list(range(n_tokens)) for _ in range(3)]

    with patch("app.dualcodec_utils.codes_to_audio_tensor", side_effect=fake_decode):
        audio = decode_chunked_with_context(
            object(),
            None,
            semantic_codes,
            acoustic_codes,
            hz=12.5,
            context_seconds=0,
            return_pre_vv=False,
        )

    assert isinstance(audio, torch.Tensor)
    assert audio.shape == (1, 1, n_tokens)
