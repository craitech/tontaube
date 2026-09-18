import torch

from app.main import _trim_boundary_silence


SAMPLE_RATE = 24_000


def _silence(milliseconds: int) -> torch.Tensor:
    return torch.zeros(1, 1, SAMPLE_RATE * milliseconds // 1000)


def _speech(milliseconds: int) -> torch.Tensor:
    samples = SAMPLE_RATE * milliseconds // 1000
    time = torch.arange(samples) / SAMPLE_RATE
    return (0.2 * torch.sin(2 * torch.pi * 440 * time)).reshape(1, 1, -1)


def test_trims_only_edges_and_keeps_padding():
    audio = torch.cat(
        [_silence(1200), _speech(500), _silence(700), _speech(500), _silence(1800)],
        dim=-1,
    )

    trimmed = _trim_boundary_silence(audio, padding_ms=400)

    assert 2.48 <= trimmed.shape[-1] / SAMPLE_RATE <= 2.54


def test_zero_padding_keeps_active_frame_edges():
    audio = torch.cat([_silence(500), _speech(500), _silence(500)], dim=-1)
    trimmed = _trim_boundary_silence(audio, padding_ms=0)
    assert trimmed.shape[-1] >= SAMPLE_RATE // 2
    assert torch.count_nonzero(trimmed) == torch.count_nonzero(audio)


def test_leaves_short_edges_and_fully_silent_audio_unchanged():
    short_edges = torch.cat([_silence(250), _speech(500), _silence(300)], dim=-1)
    fully_silent = _silence(1200)

    assert _trim_boundary_silence(short_edges, padding_ms=400).shape == short_edges.shape
    assert _trim_boundary_silence(fully_silent, padding_ms=400).shape == fully_silent.shape
