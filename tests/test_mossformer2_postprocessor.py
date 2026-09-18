import numpy as np
import pytest
import torch

import app.main as app_main
import app.mossformer2_postprocessor as mossformer_module
from app.main import TTSRequest
from app.mossformer2_postprocessor import _replace_high_band


@pytest.mark.parametrize("length", [1, 127, 192000, 192001, 500003])
def test_chunk_overlap_preserves_samples_without_seams(length):
    processor = mossformer_module.MossFormer2Postprocessor.__new__(
        mossformer_module.MossFormer2Postprocessor
    )
    processor.chunk_samples = 192000
    processor.overlap_samples = 48000
    processor.stride_samples = 144000
    processor._enhance_chunk = lambda chunk: chunk.copy()
    source = np.random.default_rng(7).normal(0, 0.1, length).astype(np.float32)
    np.testing.assert_allclose(processor._enhance_in_chunks(source), source, atol=1e-7)


def test_streaming_does_not_silently_apply_requested_mossformer():
    assert app_main._streaming_option_error(TTSRequest(text="Hello"), "mp3") is None
    assert app_main._streaming_option_error(
        TTSRequest(text="Hello", mossformer2_postprocess=False), "mp3"
    ) is None
    assert "non-streaming" in app_main._streaming_option_error(
        TTSRequest(text="Hello", mossformer2_postprocess=True), "mp3"
    )


def test_band_replacement_preserves_silent_input():
    source = np.zeros(2048, dtype=np.float32)
    predicted = np.ones_like(source)

    output = _replace_high_band(source, predicted)

    np.testing.assert_array_equal(output, source)


def test_band_replacement_keeps_shape_and_finite_output():
    sample_rate = 48_000
    time = np.arange(sample_rate, dtype=np.float32) / sample_rate
    source = 0.2 * np.sin(2 * np.pi * 440 * time)
    predicted = source + 0.01 * np.sin(2 * np.pi * 12_000 * time)

    output = _replace_high_band(source, predicted)

    assert output.shape == source.shape
    assert output.dtype == np.float32
    assert np.isfinite(output).all()


@pytest.mark.asyncio
async def test_predict_trims_before_mossformer_and_returns_48khz(monkeypatch):
    sample_rate = 24_000
    silence = torch.zeros(1, 1, sample_rate)
    speech = torch.full((1, 1, sample_rate // 2), 0.2)
    generated = torch.cat([silence, speech, silence], dim=-1)
    observed = {}

    class Engine:
        verbalizer = None
        vv_tokenizer = None

        async def generate(self, **_kwargs):
            return generated, [[1]]

    class Postprocessor:
        def process(self, audio, input_sample_rate):
            observed["postprocess_input_samples"] = audio.shape[-1]
            observed["postprocess_input_rate"] = input_sample_rate
            return audio.repeat_interleave(2, dim=-1)

    async def get_engine():
        return Engine()

    def encode_audio(audio, _format, _bitrate, sample_rate):
        observed["encoded_samples"] = audio.shape[-1]
        observed["encoded_rate"] = sample_rate
        return b"audio", "audio/wav"

    monkeypatch.setattr(app_main, "_get_engine", get_engine)
    monkeypatch.setattr(app_main, "_encode_audio", encode_audio)
    monkeypatch.setattr(
        mossformer_module,
        "get_mossformer2_postprocessor",
        lambda: Postprocessor(),
    )

    result = await app_main._synthesize(TTSRequest(
        text="Test",
        trim_silence_padding_ms=400,
    ))

    assert observed["postprocess_input_samples"] < generated.shape[-1]
    assert observed["postprocess_input_rate"] == 24_000
    assert observed["encoded_samples"] == 2 * observed["postprocess_input_samples"]
    assert observed["encoded_rate"] == 48_000
    assert result["sample_rate"] == 48_000
