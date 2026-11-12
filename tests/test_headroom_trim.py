"""OUTPUT_HEADROOM_DB as a pure level change for /predict's edge trim and pause cap.

engine.generate scales the decode by g = 10^(-h/20) (h = OUTPUT_HEADROOM_DB) before main.py's edge trim
(-50 dBFS frame RMS) and the MossFormer2 pause cap (-55 dBFS frame RMS for quiet runs, 0.01 sample peak at a join).
All three thresholds now move down by h dB, so the trim bounds and the pause-cap cuts of g*x equal those of x
(up to frames within ~1e-7 of a threshold: float32 rounding of g*x), and h = 0 is bit-identical to the code before
this change. Those earlier functions are kept verbatim below as the reference.
"""
import functools
import math
from threading import Lock
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import torch

import app.audio_pauses as ap
import app.main as app_main
import app.mossformer2_postprocessor as mossformer_module
from app.config import DUALCODEC_SAMPLE_RATE, ServerConfig
from app.dualcodec_utils import apply_output_headroom, headroom_gain
from app.inference import TTSEngine
from app.main import TTSRequest, _trim_boundary_silence

SR = DUALCODEC_SAMPLE_RATE
SEEDS = range(40)


# --- the earlier code, verbatim -----------------------------------------------------------------------------------
def _old_trim_boundary_silence(
    audio,
    *,
    padding_ms: int,
    silence_threshold_dbfs: float = -50.0,
):
    """Trim excess edge silence while preserving all internal pauses."""
    if padding_ms < 0 or audio.shape[-1] == 0:
        return audio

    samples = audio.detach().float().reshape(-1, audio.shape[-1])
    mono_power = samples.square().mean(dim=0)
    frame_samples = max(1, round(DUALCODEC_SAMPLE_RATE * 0.02))
    hop_samples = max(1, round(DUALCODEC_SAMPLE_RATE * 0.01))
    if mono_power.numel() < frame_samples:
        return audio

    frame_rms = mono_power.unfold(0, frame_samples, hop_samples).mean(dim=-1).sqrt()
    threshold = 10.0 ** (silence_threshold_dbfs / 20.0)
    active_frames = (frame_rms >= threshold).nonzero(as_tuple=False).flatten()
    if active_frames.numel() == 0:
        return audio

    padding_samples = round(DUALCODEC_SAMPLE_RATE * padding_ms / 1000)
    # Retain the entire active frames, including at zero requested padding.
    # Frame centers would otherwise cut into the start and end of speech.
    first_sample = int(active_frames[0].item()) * hop_samples
    last_sample = int(active_frames[-1].item()) * hop_samples + frame_samples
    # unfold omits a partial final hop; retain it if it contains signal.
    covered = (len(frame_rms) - 1) * hop_samples + frame_samples
    if covered < mono_power.numel() and mono_power[covered:].mean().sqrt() >= threshold:
        last_sample = mono_power.numel()
    start = max(0, first_sample - padding_samples)
    end = min(audio.shape[-1], last_sample + padding_samples)
    return audio[..., start:end]


def _old_clip_long_pauses(
    audio: np.ndarray,
    sample_rate: int,
    *,
    cap_seconds: float = 2.5,
) -> tuple[np.ndarray, int, int]:
    if not math.isfinite(cap_seconds) or cap_seconds < 0.04:
        raise ValueError("pause cap must be finite and at least 0.04 seconds")
    source = np.asarray(audio, dtype=np.float32)
    if source.ndim != 1 or sample_rate < 1000 or not np.isfinite(source).all():
        raise ValueError("expected finite mono audio and a valid sample rate")
    if not len(source):
        return source.copy(), 0, 0

    frame = round(sample_rate * 0.02)
    starts = np.arange(0, len(source), frame)
    energy = np.add.reduceat(source.astype(np.float64) ** 2, starts)
    rms = np.sqrt(energy / np.minimum(frame, len(source) - starts))
    quiet = rms < 10 ** (-55.0 / 20)
    changes = np.diff(np.r_[False, quiet, False].astype(np.int8))
    cap = round(sample_rate * cap_seconds)
    pieces = []
    cursor = removed = cuts = 0
    for first, last in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
        start = int(first * frame)
        end = min(int(last * frame), len(source))
        excess = end - start - cap
        if excess <= 0:
            continue
        width = min(round(sample_rate * 0.02), excess)
        if width < 2:
            continue
        left = start + (cap - width) // 2
        right = left + excess
        lhs = source[left:left + width]
        rhs = source[right:right + width]
        # RMS can hide a narrow transient. Skip an unsafe proposed join.
        if max(np.max(np.abs(lhs)), np.max(np.abs(rhs))) > 0.01:
            continue
        ramp = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, width))).astype(np.float32)
        blend = (1 - ramp) * lhs + ramp * rhs
        pieces.extend([source[cursor:left], blend])
        cursor = right + width
        removed += excess
        cuts += 1
    if not cuts:
        return source, 0, 0
    pieces.append(source[cursor:])
    return np.concatenate(pieces), removed, cuts


# --- synthetic decoder-like audio ---------------------------------------------------------------------------------
def _amp(level_db):
    return 10.0 ** (level_db / 20.0)


def _noise(rng, seconds, level_db):
    return (rng.standard_normal(round(seconds * SR)) * _amp(level_db)).astype(np.float32)


def _word(rng, seconds, peak):
    n = round(seconds * SR)
    t = np.arange(n) / SR
    f0 = rng.uniform(90.0, 260.0)
    wave = sum(np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 2 * np.pi)) / k for k in (1, 2, 3, 5))
    env = np.sin(np.pi * (np.arange(n) + 0.5) / n) ** 0.5
    return (peak * env * wave / np.abs(wave).max() + rng.standard_normal(n) * 1e-3).astype(np.float32)


def _decoder_like(seed):
    """Words (some overshooting full scale, like VibeVoice) between noise floors that straddle every threshold
    before and after a 3.6 dB gain: edge noise around -50 dBFS (trim), gap floors around -55 dBFS (pause-cap
    quiet RMS), long gaps over the 2.5 s cap."""
    rng = np.random.default_rng(seed)
    parts = [np.zeros(round(rng.uniform(0.0, 0.3) * SR), np.float32),
             _noise(rng, rng.uniform(0.1, 1.5), rng.uniform(-60.0, -44.0))]          # lead edge
    for _ in range(int(rng.integers(2, 6))):
        parts.append(_word(rng, rng.uniform(0.15, 0.7), rng.uniform(0.05, 1.3)))
        seconds = rng.uniform(2.6, 5.0) if rng.random() < 0.6 else rng.uniform(0.05, 1.0)
        parts.append(_noise(rng, seconds, rng.uniform(-75.0, -45.0)))
    parts.append(_word(rng, rng.uniform(0.15, 0.7), rng.uniform(0.05, 1.3)))
    parts += [_noise(rng, rng.uniform(0.1, 1.5), rng.uniform(-60.0, -44.0)),             # tail edge
              np.zeros(round(rng.uniform(0.0, 0.3) * SR), np.float32)]
    return np.concatenate(parts)


def _with_join_clicks(seed, x):
    """Put a click of 0.005-0.016 (around the 0.01 join guard and 0.01 * g) inside a planned crossfade window of
    every other cut that the pause cap plans on the /predict-trimmed audio at 0 dB."""
    rng = np.random.default_rng(10_000 + seed)
    t = torch.from_numpy(x).reshape(1, 1, -1)
    start, end = _bounds(t, _old_trim_boundary_silence(t, padding_ms=250))
    x = x.copy()
    for k, (left, right, width) in enumerate(ap.find_pause_cuts(x[start:end], SR)):
        if k % 2 == 0:
            at = start + (left if rng.random() < 0.5 else right) + int(rng.integers(0, width))
            x[at] = rng.choice([-1.0, 1.0]) * rng.uniform(0.005, 0.016)
    return x


def _battery(seed):
    return _with_join_clicks(seed, _decoder_like(seed))


def _bounds(full, trimmed):
    """(start, end) of the trim, from the view the trim returns."""
    assert trimmed.untyped_storage().data_ptr() == full.untyped_storage().data_ptr()
    start = trimmed.storage_offset() - full.storage_offset()
    return start, start + trimmed.shape[-1]


# --- threshold arithmetic -----------------------------------------------------------------------------------------
@pytest.mark.parametrize("headroom_db", [0.0, 1.0, 3.6, 6.0])
def test_trim_threshold_is_minus_50_minus_headroom(headroom_db):
    """The trim's frame-RMS threshold is -50 - h dBFS: a block 1e-4 above it is kept, 1e-4 below it is silence."""
    threshold = 10.0 ** ((-50.0 - headroom_db) / 20.0)
    assert 20 * math.log10(threshold) == pytest.approx(-50.0 - headroom_db, abs=1e-12)
    assert threshold == pytest.approx(10.0 ** (-50.0 / 20.0) * headroom_gain(headroom_db), rel=1e-12)
    for factor, active in [(1.0001, True), (0.9999, False)]:
        block = torch.full((SR // 2,), threshold * factor)
        audio = torch.cat([torch.zeros(SR // 2), block, torch.zeros(SR // 2)]).reshape(1, 1, -1)
        trimmed = _trim_boundary_silence(audio, padding_ms=0, headroom_db=headroom_db)
        if active:
            assert _bounds(audio, trimmed) == (SR // 2, SR)                    # the block's full frames
        else:
            assert trimmed is audio                                            # no active frame: unchanged


def test_pause_thresholds_at_zero_are_the_fixed_values():
    quiet_rms, join_peak = ap.pause_thresholds(0.0)
    assert quiet_rms == 10 ** (-55.0 / 20) and join_peak == 0.01              # exact, as before the change
    assert ap.pause_thresholds() == (quiet_rms, join_peak) == ap.pause_thresholds(-0.0)
    assert (ap.QUIET_DBFS, ap.JOIN_PEAK) == (-55.0, 0.01)


@pytest.mark.parametrize("headroom_db", [0.5, 1.0, 3.6, 6.0, 20.0])
def test_pause_thresholds_move_down_by_headroom_db(headroom_db):
    quiet_rms, join_peak = ap.pause_thresholds(headroom_db)
    quiet_0, join_0 = ap.pause_thresholds(0.0)
    assert 20 * math.log10(quiet_rms) == pytest.approx(-55.0 - headroom_db, abs=1e-12)    # dBFS (RMS)
    assert 20 * math.log10(join_peak) == pytest.approx(-40.0 - headroom_db, abs=1e-12)    # 0.01 = -40 dBFS peak
    assert quiet_rms / quiet_0 == pytest.approx(headroom_gain(headroom_db), rel=1e-12)
    assert join_peak / join_0 == pytest.approx(headroom_gain(headroom_db), rel=1e-12)


@pytest.mark.parametrize("bad", [-1.0, -1e-9, float("nan"), float("inf"), -float("inf")])
def test_bad_headroom_is_rejected(bad):
    with pytest.raises(ValueError):
        ap.pause_thresholds(bad)
    with pytest.raises(ValueError):
        ap.clip_long_pauses(np.zeros(SR, np.float32), SR, headroom_db=bad)
    with pytest.raises(ValueError):
        ap.find_pause_cuts(np.zeros(SR, np.float32), SR, headroom_db=bad)


def _gap_audio(level, gap_seconds=4.0):
    tone = (0.1 * np.sin(2 * np.pi * 440 * np.arange(SR) / SR)).astype(np.float32)
    return np.concatenate([tone, np.full(round(gap_seconds * SR), level, np.float32), tone])


@pytest.mark.parametrize("headroom_db", [0.0, 1.0, 3.6, 6.0])
def test_pause_cap_quiet_threshold_behaviour(headroom_db):
    quiet_rms, _ = ap.pause_thresholds(headroom_db)
    below = ap.clip_long_pauses(_gap_audio(quiet_rms * 0.9999), SR, headroom_db=headroom_db)
    above = ap.clip_long_pauses(_gap_audio(quiet_rms * 1.0001), SR, headroom_db=headroom_db)
    assert below[2] == 1 and below[1] == round(1.5 * SR)                       # 4 s quiet run capped to 2.5 s
    assert above[1:] == (0, 0)


@pytest.mark.parametrize("headroom_db", [0.0, 1.0, 3.6, 6.0])
def test_pause_cap_join_guard_behaviour(headroom_db):
    _, join_peak = ap.pause_thresholds(headroom_db)
    left = SR + (round(2.5 * SR) - round(0.02 * SR)) // 2                      # first crossfade window
    for factor, cuts in [(0.999, 1), (1.001, 0)]:
        audio = _gap_audio(1e-5)
        audio[left + 7] = join_peak * factor
        assert ap.clip_long_pauses(audio, SR, headroom_db=headroom_db)[2] == cuts
        assert len(ap.find_pause_cuts(audio, SR, headroom_db=headroom_db)) == cuts


# --- OUTPUT_HEADROOM_DB = 0: bit-identical to the earlier code ----------------------------------------------------
@functools.lru_cache(maxsize=None)
def _zero_cases():
    rng = np.random.default_rng(7)
    cases = [_battery(seed) for seed in range(12)]
    cases += [np.zeros(0, np.float32), np.zeros(100, np.float32), np.zeros(479, np.float32),
              np.zeros(3 * SR, np.float32), _gap_audio(1e-5, 6.0), _gap_audio(0.003, 5.0),
              (0.3 * rng.standard_normal(2 * SR)).astype(np.float32),
              np.concatenate([_gap_audio(1e-5, 4.0), _gap_audio(1e-5, 6.0)]),
              np.concatenate([np.zeros(SR // 3, np.float32), _gap_audio(_amp(-50.0) * 1.00001, 3.0)[: SR + 7]])]
    return tuple(cases)


@pytest.mark.parametrize("case", range(len(_zero_cases())))
@pytest.mark.parametrize("padding_ms", [0, 250, 400])
def test_zero_headroom_trim_is_bit_identical(case, padding_ms):
    x = torch.from_numpy(_zero_cases()[case]).reshape(1, 1, -1)
    old = _old_trim_boundary_silence(x, padding_ms=padding_ms)
    for new in (_trim_boundary_silence(x, padding_ms=padding_ms),
                _trim_boundary_silence(x, padding_ms=padding_ms, headroom_db=0.0)):
        assert (new is x) == (old is x)
        assert new.dtype == old.dtype and torch.equal(new, old)
        if old is not x:
            assert _bounds(x, new) == _bounds(x, old)


@pytest.mark.parametrize("case", range(len(_zero_cases())))
def test_zero_headroom_pause_cap_is_bit_identical(case):
    x = _zero_cases()[case]
    old, old_removed, old_cuts = _old_clip_long_pauses(x, SR)
    for new, removed, cuts in (ap.clip_long_pauses(x, SR), ap.clip_long_pauses(x, SR, headroom_db=0.0)):
        assert (removed, cuts) == (old_removed, old_cuts)
        assert (new is x) == (old is x) and new.dtype == old.dtype and np.array_equal(new, old)


def test_zero_headroom_battery_exercises_the_cap():
    """The bit-identity battery is not trivial: it has cuts, skipped joins and trims."""
    total_cuts = sum(_old_clip_long_pauses(x, SR)[2] for x in _zero_cases())
    assert total_cuts >= 10


# --- 3.6 dB (and others): the same trim and pause-cap points as at 0 dB -------------------------------------------
def _points(x, headroom_db, old=False):
    """Trim bounds (TTSRequest default 250 ms padding), then the pause cap's cuts on the trimmed audio."""
    if old:
        trimmed = _old_trim_boundary_silence(x, padding_ms=250)
    else:
        trimmed = _trim_boundary_silence(x, padding_ms=250, headroom_db=headroom_db)
    bounds = (0, x.shape[-1]) if trimmed is x else _bounds(x, trimmed)
    source = trimmed.reshape(-1).numpy()
    if old:
        cuts = ap.find_pause_cuts(source, SR)          # the fixed thresholds (headroom 0) on the scaled audio
    else:
        cuts = ap.find_pause_cuts(source, SR, headroom_db=headroom_db)
    return bounds, cuts, trimmed


@pytest.mark.parametrize("headroom_db", [1.0, 3.6, 6.0])
def test_trim_and_pause_points_identical_to_zero_db(headroom_db):
    gain = headroom_gain(headroom_db)
    trim_teeth = cut_teeth = 0
    for seed in SEEDS:
        off = torch.from_numpy(_battery(seed)).reshape(1, 1, -1)
        on = apply_output_headroom(off, headroom_db)                         # the serving gain, out of place
        bounds_off, cuts_off, trimmed_off = _points(off, 0.0)
        bounds_on, cuts_on, trimmed_on = _points(on, headroom_db)
        assert bounds_on == bounds_off, seed
        assert cuts_on == cuts_off, seed
        # the capped waveform is the 0 dB one, scaled
        capped_off, removed_off, n_off = ap.clip_long_pauses(trimmed_off.reshape(-1).numpy(), SR)
        capped_on, removed_on, n_on = ap.clip_long_pauses(trimmed_on.reshape(-1).numpy(), SR,
                                                          headroom_db=headroom_db)
        assert (removed_on, n_on) == (removed_off, n_off) and len(capped_on) == len(capped_off)
        np.testing.assert_allclose(capped_on, capped_off * gain, rtol=1e-6, atol=1e-8)
        # the earlier fixed thresholds would have moved the points on the scaled audio
        old_bounds, old_cuts, _ = _points(on, headroom_db, old=True)
        trim_teeth += old_bounds != bounds_off
        cut_teeth += old_cuts != cuts_off
    assert trim_teeth >= 5 and cut_teeth >= 3, (trim_teeth, cut_teeth)


def test_pause_points_identical_with_join_guard_clicks():
    """Clicks between 0.0066 and 0.01 (0 dB) are skipped joins at 0 dB and at 3.6 dB; with the fixed 0.01 guard
    the scaled click would pass and the join would be cut."""
    gain = headroom_gain(3.6)
    passed_at_fixed = 0
    for click in [0.0101, 0.0120, 0.0152]:
        off = _gap_audio(1e-5, 6.0)
        off[SR + (round(2.5 * SR) - 480) // 2 + 100] = click
        on = (torch.from_numpy(off) * gain).numpy()
        assert ap.find_pause_cuts(off, SR) == ap.find_pause_cuts(on, SR, headroom_db=3.6) == []
        passed_at_fixed += len(ap.find_pause_cuts(on, SR)) == 1
    assert passed_at_fixed == 2                                     # 0.0101 and 0.012 (0.0152 x g = 0.01004 > 0.01)


# --- end to end through /predict (real TTSEngine.generate gain, real MossFormer2 pause-cap call) ------------------
def _engine(headroom_db):
    config = ServerConfig()
    config.output_headroom_db = headroom_db
    engine = TTSEngine(config)
    engine.vv_tokenizer = object()
    engine.dc_inference = object()
    engine._resolve_params = MagicMock(return_value=(dict(temperature=0.8, top_p=0.95, top_k=50, seed=1),
                                                     (None, None), "", 0.0, -1))
    engine._resolve_prefix = AsyncMock(return_value=(None, {}))
    engine._n_codebooks = MagicMock(return_value=4)
    engine._generate_semantic = AsyncMock(return_value=([1] * 100, []))
    engine._generate_acoustic = AsyncMock(return_value=[2] * 100)
    return engine


def _postprocessor():
    """The real MossFormer2Postprocessor.process/_process without weights: the enhancer is the identity."""
    processor = mossformer_module.MossFormer2Postprocessor.__new__(mossformer_module.MossFormer2Postprocessor)
    processor._process_lock = Lock()
    processor._resident = None  # weights stay where they are (no host offload)
    processor._enhance_in_chunks = lambda source: source.copy()
    return processor


async def _predict(monkeypatch, decoded, headroom_db, mossformer=True):
    engine = _engine(headroom_db)
    observed = {"cap_calls": []}
    real_clip = ap.clip_long_pauses

    def spy_clip(source, sample_rate, **kwargs):
        result = real_clip(source, sample_rate, **kwargs)
        observed["cap_calls"].append(dict(source=source.copy(), kwargs=kwargs, result=result,
                                          cuts=ap.find_pause_cuts(source, sample_rate, **kwargs)))
        return result

    def encode_audio(audio, _format, _bitrate, sample_rate):
        observed["encoded"] = audio.clone()
        observed["sample_rate"] = sample_rate
        return b"audio", "audio/wav"

    async def get_engine():
        return engine

    processor = _postprocessor()
    monkeypatch.setattr(app_main, "_get_engine", get_engine)
    monkeypatch.setattr(app_main, "_encode_audio", encode_audio)
    monkeypatch.setattr(mossformer_module, "get_mossformer2_postprocessor", lambda: processor)
    monkeypatch.setattr(mossformer_module, "clip_long_pauses", spy_clip)
    with patch("app.inference.free_memory"), patch(
            "app.inference.decode_chunked_with_context", return_value=decoded.clone()):
        await app_main._synthesize(TTSRequest(text="Test", mossformer2_postprocess=mossformer))
    return observed


@pytest.mark.asyncio
@pytest.mark.parametrize("seed", [0, 3, 5, 11])
async def test_predict_trim_and_pause_cap_identical_at_0_and_3_6(monkeypatch, seed):
    x = _battery(seed).copy()
    x *= 0.95 / max(1.0, float(np.abs(x).max()))           # below full scale, so MF2's final clip is not involved
    decoded = torch.from_numpy(x).reshape(1, 1, -1)
    at_0 = await _predict(monkeypatch, decoded, 0.0)
    at_36 = await _predict(monkeypatch, decoded, 3.6)
    (cap_0,), (cap_36,) = at_0["cap_calls"], at_36["cap_calls"]
    assert cap_0["kwargs"] == {"headroom_db": 0.0} and cap_36["kwargs"] == {"headroom_db": 3.6}
    # =0: the earlier chain, bit for bit (old trim, then old pause cap)
    old_trimmed = _old_trim_boundary_silence(decoded, padding_ms=250).reshape(-1).numpy()
    assert np.array_equal(cap_0["source"], old_trimmed)
    old_capped = _old_clip_long_pauses(old_trimmed, SR)
    assert cap_0["result"][1:] == old_capped[1:] and np.array_equal(cap_0["result"][0], old_capped[0])
    # 3.6: the same trim bounds and the same cuts, at -3.6 dB
    assert len(cap_36["source"]) == len(cap_0["source"])
    np.testing.assert_allclose(cap_36["source"], cap_0["source"] * headroom_gain(3.6), rtol=1e-6, atol=0)
    assert cap_36["cuts"] == cap_0["cuts"] and cap_36["result"][1:] == cap_0["result"][1:]
    assert at_36["encoded"].shape == at_0["encoded"].shape and at_0["sample_rate"] == 48_000


@pytest.mark.asyncio
async def test_predict_without_mossformer_trim_identical_at_0_and_3_6(monkeypatch):
    decoded = torch.from_numpy(_battery(2)).reshape(1, 1, -1)
    at_0 = await _predict(monkeypatch, decoded, 0.0, mossformer=False)
    at_36 = await _predict(monkeypatch, decoded, 3.6, mossformer=False)
    assert at_0["cap_calls"] == at_36["cap_calls"] == []
    assert torch.equal(at_0["encoded"], _old_trim_boundary_silence(decoded, padding_ms=250))
    assert torch.equal(at_36["encoded"], _old_trim_boundary_silence(decoded, padding_ms=250) * headroom_gain(3.6))
