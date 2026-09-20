from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import ServerConfig, COLLAPSE_SILENCE_TOKEN_IDS
from app.inference import TTSEngine


@pytest.mark.parametrize('characters, passes', [(63, False), (64, True), (160, True), (161, False)])
def test_new_cps_boundaries(characters, passes):
    engine = TTSEngine(ServerConfig())
    reason = engine._check_cb0_generation(100, characters, False, 12.5)
    assert (reason is None) == passes
    if not passes:
        assert reason.startswith('cps_out_of_range')


def test_early_stream_prefix_still_skips_cps_check():
    engine = TTSEngine(ServerConfig())
    assert engine._check_cb0_generation(10, 300, True, 12.5, codes=[20] * 10, early_mode=True) is None


@pytest.mark.asyncio
@pytest.mark.parametrize('base', [0.0, 0.65])
@pytest.mark.parametrize('early', [False, True])
async def test_retry_temperature_steps_and_resets_without_mutating_sampling(base, early):
    engine = TTSEngine(ServerConfig())
    engine.config.force_first_silence = False
    context = SimpleNamespace(tokenizer=MagicMock(), dc_base_id=100, pad_id=0,
                              eos_id=2, audio_marker_id=3, hz=12.5, engine=MagicMock())
    engine._cb_ctx = MagicMock(return_value=context)
    bad = ([100 + COLLAPSE_SILENCE_TOKEN_IDS[0]] * 40 if early else [120] * 5) + [2]
    good = [120] * 50 + [2]
    engine._generate_with_engine = AsyncMock(side_effect=[bad] * 4 + [good, good])
    sampling = dict(temperature=base, top_p=0.95, top_k=50, seed=123)
    original = sampling.copy()
    with patch('app.inference.tokenize_text', return_value=[10]), patch(
            'app.inference.SamplingParams', side_effect=lambda **kwargs: SimpleNamespace(**kwargs)):
        for _ in range(2):
            await engine._generate_semantic_segment(
                seg_idx=0, segments=['a' * 80], prev_chunk_global=None,
                sampling=sampling, voice_prompt=(None, None), prefix=None,
                max_new_tokens=400, early_mode=early)
    params = [call.args[2] for call in engine._generate_with_engine.await_args_list]
    assert [p.temperature for p in params] == pytest.approx(
        [base, base, base + 0.05, base + 0.10, base + 0.15, base])
    assert sampling == original
    assert all(p.top_p == 0.95 and p.top_k == 50 and p.seed == 123 for p in params)
