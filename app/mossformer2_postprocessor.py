"""MossFormer2 speech super-resolution for non-streaming output audio."""

from __future__ import annotations

import math
import os
from pathlib import Path
from threading import Lock

import librosa
import numpy as np
from huggingface_hub import hf_hub_download
from scipy import signal
import torch

from app.logging_utils import get_logger
from app.audio_pauses import clip_long_pauses
from app.third_party.clearvoice_mossformer2_sr import Generator, Mossformer


logger = get_logger("mossformer2")

MOSSFORMER2_SAMPLE_RATE = 48_000
_MODEL_REPO_ID = os.getenv(
    "MOSSFORMER2_SR_REPO_ID",
    "alibabasglab/MossFormer2_SR_48K",
)
_MODEL_REVISION = os.getenv(
    "MOSSFORMER2_SR_REVISION",
    "39eb1f25ea84f5e0315ade9ac0070fff216fc690",
)
_LOCAL_MODEL_PATH = os.getenv("MOSSFORMER2_SR_MODEL_PATH")
_MOSSFORMER_WEIGHTS = "last_best_checkpoint_m.pt"
_GENERATOR_WEIGHTS = "last_best_checkpoint_g.pt"


class _Config(dict):
    def __init__(self, values):
        super().__init__(values)
        self.__dict__ = self


_MODEL_CONFIG = _Config({
    "resblock": "1",
    "upsample_rates": [8, 8, 2, 2],
    "upsample_kernel_sizes": [16, 16, 4, 4],
    "upsample_initial_channel": 1024,
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "num_mels": 80,
    "n_fft": 1024,
    "hop_size": 256,
    "win_size": 1024,
    "sampling_rate": MOSSFORMER2_SAMPLE_RATE,
    "fmin": 0,
    "fmax": 8000,
})


def _resolve_weight(filename: str) -> Path:
    if _LOCAL_MODEL_PATH:
        path = Path(_LOCAL_MODEL_PATH).expanduser().resolve() / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing MossFormer2 weight: {path}")
        return path
    return Path(hf_hub_download(
        repo_id=_MODEL_REPO_ID,
        filename=filename,
        revision=_MODEL_REVISION,
    ))


def _load_state(model, path: Path, key: str) -> None:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint.get(key, checkpoint)
    model.load_state_dict(state, strict=True)


def _mel_spectrogram(audio: torch.Tensor) -> torch.Tensor:
    config = _MODEL_CONFIG
    mel = librosa.filters.mel(
        sr=config.sampling_rate,
        n_fft=config.n_fft,
        n_mels=config.num_mels,
        fmin=config.fmin,
        fmax=config.fmax,
    )
    mel_basis = torch.from_numpy(mel).float()
    window = torch.hann_window(config.win_size)
    padding = (config.n_fft - config.hop_size) // 2
    audio = torch.nn.functional.pad(
        audio.unsqueeze(1),
        (padding, padding),
        mode="reflect",
    ).squeeze(1)
    spectrum = torch.stft(
        audio,
        config.n_fft,
        hop_length=config.hop_size,
        win_length=config.win_size,
        window=window,
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    ).abs()
    return torch.log(torch.clamp(torch.matmul(mel_basis, spectrum), min=1e-5))


def _replace_high_band(original: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """Retain the source band and take only reconstructed high frequencies."""
    length = min(len(original), len(predicted))
    original = original[:length].astype(np.float64, copy=False)
    predicted = predicted[:length].astype(np.float64, copy=False)
    if length < 32 or not np.any(original):
        return original.astype(np.float32)

    frequencies, _, spectrum = signal.stft(original, fs=MOSSFORMER2_SAMPLE_RATE)
    band_energy = np.abs(spectrum) ** 2
    total_energy = float(band_energy.sum())
    if not math.isfinite(total_energy) or total_energy <= np.finfo(np.float64).eps:
        return original.astype(np.float32)
    cumulative = np.cumsum(band_energy.sum(axis=1)) / total_energy
    cutoff_index = int(np.searchsorted(cumulative, 0.9996, side="left"))
    cutoff_index = min(cutoff_index, len(frequencies) - 1)
    cutoff_hz = float(frequencies[cutoff_index])
    nyquist = MOSSFORMER2_SAMPLE_RATE / 2
    if cutoff_hz <= 0 or cutoff_hz >= nyquist * 0.99:
        return original.astype(np.float32)

    low_b, low_a = signal.butter(4, cutoff_hz / nyquist, btype="low")
    high_b, high_a = signal.butter(4, cutoff_hz / nyquist, btype="high")
    combined = (
        signal.filtfilt(low_b, low_a, original)
        + signal.filtfilt(high_b, high_a, predicted)
    )

    transition_samples = min(length, round(0.1 * MOSSFORMER2_SAMPLE_RATE))
    if transition_samples:
        fade = np.linspace(0.0, 1.0, transition_samples, endpoint=True)
        combined[:transition_samples] = (
            (1.0 - fade) * original[:transition_samples]
            + fade * combined[:transition_samples]
        )
    return combined.astype(np.float32)


class MossFormer2Postprocessor:
    """Lazily loaded, serialized MossFormer2_SR_48K inference runtime."""

    def __init__(self, *, chunk_seconds: float = 4.0) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("MossFormer2 postprocessing requires CUDA")
        self.device = torch.device("cuda", torch.cuda.current_device())
        logger.info("Loading MossFormer2_SR_48K on %s", self.device)

        self.model_m = Mossformer()
        self.model_g = Generator(_MODEL_CONFIG)
        _load_state(self.model_m, _resolve_weight(_MOSSFORMER_WEIGHTS), "mossformer")
        _load_state(self.model_g, _resolve_weight(_GENERATOR_WEIGHTS), "generator")
        self.model_m.to(self.device).eval()
        self.model_g.to(self.device).eval()
        self.model_g.remove_weight_norm()

        self.chunk_samples = round(chunk_seconds * MOSSFORMER2_SAMPLE_RATE)
        self.overlap_samples = self.chunk_samples // 4
        self.stride_samples = self.chunk_samples - self.overlap_samples
        self._process_lock = Lock()
        logger.info("MossFormer2_SR_48K ready: %.1fs windows, %.1fs overlap; 2.5s pause cap, no muting",
                    chunk_seconds, self.overlap_samples / MOSSFORMER2_SAMPLE_RATE)

    def _enhance_chunk(self, chunk: np.ndarray) -> np.ndarray:
        chunk_tensor = torch.from_numpy(chunk.astype(np.float32, copy=False)).unsqueeze(0)
        mel = _mel_spectrogram(chunk_tensor).to(self.device)
        with torch.inference_mode():
            enhanced = self.model_g(self.model_m(mel)).squeeze(0).squeeze(0)
        enhanced = enhanced.detach().float().cpu().numpy()
        if len(enhanced) < self.chunk_samples:
            enhanced = np.pad(enhanced, (0, self.chunk_samples - len(enhanced)))
        return enhanced[:self.chunk_samples]

    def _enhance_in_chunks(self, audio: np.ndarray) -> np.ndarray:
        original_length = len(audio)
        if original_length <= self.chunk_samples:
            padded_length = self.chunk_samples
        else:
            steps = math.ceil(
                (original_length - self.chunk_samples) / self.stride_samples
            )
            padded_length = self.chunk_samples + steps * self.stride_samples
        padded = np.pad(audio, (0, padded_length - original_length))
        starts = list(range(
            0,
            padded_length - self.chunk_samples + 1,
            self.stride_samples,
        ))
        output = np.zeros(padded_length, dtype=np.float64)
        weights = np.zeros(padded_length, dtype=np.float64)

        for index, start in enumerate(starts):
            chunk = self._enhance_chunk(padded[start:start + self.chunk_samples])
            window = np.ones(self.chunk_samples, dtype=np.float64)
            if index > 0:
                window[:self.overlap_samples] = np.linspace(
                    0.0,
                    1.0,
                    self.overlap_samples,
                    endpoint=False,
                )
            if index < len(starts) - 1:
                window[-self.overlap_samples:] = np.linspace(
                    1.0,
                    0.0,
                    self.overlap_samples,
                    endpoint=False,
                )
            end = start + self.chunk_samples
            output[start:end] += chunk * window
            weights[start:end] += window

        output /= np.maximum(weights, np.finfo(np.float64).eps)
        return output[:original_length].astype(np.float32)

    def process(self, audio, input_sample_rate: int) -> torch.Tensor:
        # A single model instance is shared by request workers. Serializing this
        # bounded-memory stage avoids concurrent activation spikes on the GPU.
        with self._process_lock:
            return self._process(audio, input_sample_rate)

    def _process(self, audio, input_sample_rate: int) -> torch.Tensor:
        source = (
            audio.detach()
            .float()
            .reshape(-1, audio.shape[-1])
            .mean(dim=0)
            .cpu()
            .numpy()
        )
        # Match the tested order: cap on the original waveform, then resample
        # and enhance. Existing boundary trimming happens in /predict first.
        source, removed, cuts = clip_long_pauses(source, input_sample_rate)
        if cuts:
            logger.info("Pause cap: shortened %d quiet runs to 2.5s; removed %.3fs; no muting",
                        cuts, removed / input_sample_rate)
        if input_sample_rate != MOSSFORMER2_SAMPLE_RATE:
            source = librosa.resample(
                source,
                orig_sr=input_sample_rate,
                target_sr=MOSSFORMER2_SAMPLE_RATE,
            )
        source = np.asarray(source, dtype=np.float32)
        predicted = self._enhance_in_chunks(source)
        enhanced = _replace_high_band(source, predicted)
        if not np.isfinite(enhanced).all():
            raise RuntimeError("MossFormer2 produced non-finite audio")
        enhanced = np.clip(enhanced, -1.0, 1.0)
        return torch.from_numpy(enhanced).reshape(1, 1, -1)


_instance: MossFormer2Postprocessor | None = None
_instance_lock = Lock()


def get_mossformer2_postprocessor() -> MossFormer2Postprocessor:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = MossFormer2Postprocessor()
    return _instance
