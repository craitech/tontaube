"""VibeVoice on the GPU with weight-only quantized FFN layers (~88% of its weights)."""
from __future__ import annotations

import torch
import torch.nn as nn

from app.lite.quant import QLinear, QuantSpec


class _QLinearModule(nn.Module):
    def __init__(self, q: QLinear):
        super().__init__()
        self.q = q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q(x)


def quantize_vibevoice(vv_tokenizer, spec: QuantSpec) -> tuple[int, int]:
    """Replace VibeVoice FFN linears with weight-only quantized ones.

    Returns (#layers replaced, bytes saved).
    """
    replaced, saved = 0, 0
    targets = [(name, mod) for name, mod in vv_tokenizer.named_modules()
               if isinstance(mod, nn.Linear) and ".ffn." in f".{name}."]
    for name, lin in targets:
        parent = vv_tokenizer.get_submodule(name.rsplit(".", 1)[0])
        before = lin.weight.numel() * lin.weight.element_size()
        q = QLinear.from_weight(lin.weight.data, lin.bias.data if lin.bias is not None else None,
                                spec, lin.weight.device, lin.weight.dtype)
        setattr(parent, name.rsplit(".", 1)[1], _QLinearModule(q))
        saved += before - (q.nbytes() - (q.bias.numel() * q.bias.element_size() if q.bias is not None else 0))
        replaced += 1
    torch.cuda.empty_cache()
    return replaced, saved
