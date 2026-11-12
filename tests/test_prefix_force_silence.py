"""PREFIX_FORCE_FIRST_SILENCE (default on): with a prefix and the flag off, the continuation's first CB0 token is not
forced to SILENCE_TOKEN_ID; with the flag on (default), or without a prefix, nothing changes."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from app.config import ServerConfig
from tests.test_inference import TestSemanticRetry


def test_config_flag(monkeypatch):
    monkeypatch.delenv("PREFIX_FORCE_FIRST_SILENCE", raising=False)
    assert ServerConfig().prefix_force_first_silence is True
    monkeypatch.setenv("PREFIX_FORCE_FIRST_SILENCE", "0")
    assert ServerConfig().prefix_force_first_silence is False


class TestPrefixForceSilence:
    setup_method = TestSemanticRetry.setup_method   # its engine fixture only (not its tests)
    _make_good_output = TestSemanticRetry._make_good_output

    async def _codes(self, flag, with_prefix):
        self.engine.config.force_first_silence = True
        self.engine.config.prefix_force_first_silence = flag
        self.engine._generate_with_engine = AsyncMock(return_value=self._make_good_output(12))
        prefix = None
        if with_prefix:
            prefix = {"text": "Before,", "semantic_codes": [5] * 7,
                      "semantic_global": [self.dc_base + 5] * 7}
        with patch("app.inference.SamplingParams", side_effect=lambda **kwargs: SimpleNamespace(**kwargs)):
            codes, _ = await self.engine._generate_semantic(
                ["short text"], sampling={"temperature": 0.9, "top_p": 0.95, "top_k": 10, "frequency_penalty": 0.0, "seed": None},
                voice_prompt=(None, None), max_new_tokens=50, prefix=prefix)
        return codes

    @pytest.mark.asyncio
    @pytest.mark.parametrize("flag, with_prefix, n", [(True, True, 7 + 13), (False, True, 7 + 12), (True, False, 13), (False, False, 13)])
    @patch("app.inference.tokenize_text", return_value=[10, 11, 12])
    async def test_forced_silence(self, mock_tok, flag, with_prefix, n):
        codes = await self._codes(flag, with_prefix)
        assert len(codes) == n
