import numpy as np
import pytest

from app.audio_pauses import clip_long_pauses


RATE = 24000
TONE = (0.1 * np.sin(2 * np.pi * 440 * np.arange(RATE) / RATE)).astype(np.float32)


def sample(gap, level=1e-5):
    return np.concatenate([TONE, np.full(round(gap * RATE), level, np.float32), TONE])


@pytest.mark.parametrize("gap", [0.2, 0.8, 2.0, 2.5])
def test_short_pauses_unchanged(gap):
    audio = sample(gap)
    result, removed, cuts = clip_long_pauses(audio, RATE)
    np.testing.assert_array_equal(result, audio)
    assert (removed, cuts) == (0, 0)


@pytest.mark.parametrize("gap", [2.52, 3, 6])
def test_cap_preserves_speech_and_does_not_mute(gap):
    audio = sample(gap)
    result, removed, cuts = clip_long_pauses(audio, RATE)
    assert len(result) == round(4.5 * RATE)
    assert removed == len(audio) - len(result)
    assert cuts == 1
    np.testing.assert_array_equal(result[:RATE], TONE)
    np.testing.assert_array_equal(result[-RATE:], TONE)
    np.testing.assert_allclose(result[RATE:-RATE], 1e-5, rtol=1e-6)


def test_multiple_pauses_and_transient_guard():
    audio = np.concatenate([sample(4), sample(6)])
    result, _, cuts = clip_long_pauses(audio, RATE)
    assert (len(result), cuts) == (9 * RATE, 2)
    audio = sample(6)
    audio[round(2.24 * RATE)] = 0.012
    result, removed, cuts = clip_long_pauses(audio, RATE)
    np.testing.assert_array_equal(result, audio)
    assert (removed, cuts) == (0, 0)
    noisy = sample(5, 0.003)
    np.testing.assert_array_equal(clip_long_pauses(noisy, RATE)[0], noisy)


def test_crossfade_matches_retained_samples_at_both_ends():
    audio = sample(6)
    audio[RATE:-RATE] = np.random.default_rng(4).normal(0, 1e-5, 6 * RATE)
    result, removed, _ = clip_long_pauses(audio, RATE)
    width = round(0.02 * RATE)
    left = RATE + (round(2.5 * RATE) - width) // 2
    right = left + removed
    np.testing.assert_array_equal(result[:left], audio[:left])
    assert result[left] == audio[left]
    assert result[left + width - 1] == audio[right + width - 1]
    np.testing.assert_array_equal(result[left + width:], audio[right + width:])
