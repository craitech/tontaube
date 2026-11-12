"""Which decode kernels a GPU runs: the tensor-core kernels, or FMA kernels before Volta.

The decode GEMV and attention kernels in app.lite.gpu.kernels pad the one to
few decode rows to 16 so that tl.dot runs on tensor cores. GPUs below compute
capability 7.0 (Pascal, Maxwell) have none: Triton lowers tl.dot to fp32
multiply-add loops there, and the padding multiplies the work by up to 16.
Those GPUs use app.lite.gpu.kernels_fma instead; every other GPU runs the
tensor-core kernels unchanged.

LITE_FMA_KERNELS: ``auto`` (default: FMA kernels below compute capability 7.0),
``on`` (FMA kernels on every GPU) or ``off`` (tensor-core kernels on every GPU).
It is read once per device, at the first decode kernel launched there.
"""
from __future__ import annotations

import os
from collections.abc import Mapping

import torch

from app.logging_utils import get_logger

FMA_KERNELS_ENV = "LITE_FMA_KERNELS"
_SETTINGS = {"auto": "auto", "": "auto", "on": "on", "1": "on", "off": "off", "0": "off"}
TENSOR_CORE_CAPABILITY = (7, 0)

logger = get_logger("lite")
_USE_FMA: dict[int, bool] = {}


def fma_kernels_setting(environ: Mapping[str, str] | None = None) -> str:
    """LITE_FMA_KERNELS as ``auto``, ``on`` or ``off`` (``1``/``0`` are accepted too)."""
    environ = os.environ if environ is None else environ
    raw = environ.get(FMA_KERNELS_ENV, "auto").strip().lower()
    if raw not in _SETTINGS:
        raise ValueError(f"unsupported {FMA_KERNELS_ENV} {raw!r}; expected auto, on or off")
    return _SETTINGS[raw]


def use_fma_kernels(device: torch.device) -> bool:
    """Whether decode kernels on this device use the FMA variants; decided once per CUDA device."""
    if device.type != "cuda":
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    use = _USE_FMA.get(index)
    if use is None:
        use = _USE_FMA[index] = _decide(index)
    return use


def _decide(index: int) -> bool:
    setting = fma_kernels_setting()
    capability = tuple(torch.cuda.get_device_capability(index))
    use = setting == "on" or (setting == "auto" and capability < TENSOR_CORE_CAPABILITY)
    if use:
        logger.info("GPU %d (compute capability %d.%d): decode kernels without tensor cores (%s=%s)",
                    index, capability[0], capability[1], FMA_KERNELS_ENV, setting)
    return use


def reset() -> None:
    """Forget the per-device decisions (tests)."""
    _USE_FMA.clear()
