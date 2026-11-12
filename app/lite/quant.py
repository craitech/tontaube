"""Weight-only quantized linear layers for the lite backend.

Formats (see app.lite.gpu.kernels for the packed layout):
  bits=16  bf16 weights, plain cuBLAS
  bits=8   uint8 codes, asymmetric (scale, min) per group of `group` inputs
  bits=4   packed uint8 nibbles, asymmetric (scale, min) per group

Decode steps (few rows) use a fused Triton GEMV that reads the packed codes
directly; prefill dequantizes the layer to a temporary bf16 matrix and uses
cuBLAS, which is much faster than any weight-only kernel at large M.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

# Up to this many rows the fused GEMV beats dequantize + cuBLAS (weights are
# read once instead of being expanded to bf16 for every call).
GEMV_MAX_ROWS = 64


@dataclass(frozen=True)
class QuantSpec:
    bits: int = 16
    group: int = 128

    def __str__(self) -> str:
        return "bf16" if self.bits == 16 else f"int{self.bits}g{self.group}"


def quantize_rtn(w: torch.Tensor, bits: int, group: int):
    """Round-to-nearest asymmetric group quantization.

    Returns (codes uint8 [N, K or K/2], scales fp16 [N, K/group], mins fp16).
    Scales/mins are rounded to fp16 *before* computing the codes so the
    stored parameters reproduce exactly what was optimized.
    """
    N, K = w.shape
    if K % group:
        raise ValueError(f"K={K} is not divisible by group={group}")
    qmax = (1 << bits) - 1
    wg = w.float().reshape(N, K // group, group)
    mn = wg.amin(-1)
    mx = wg.amax(-1)
    scale = ((mx - mn) / qmax).clamp(min=1e-8).half()
    mn = mn.half()
    q = torch.round((wg - mn.float()[..., None]) / scale.float()[..., None]).clamp_(0, qmax)
    return pack_codes(q.reshape(N, K).to(torch.uint8), bits), scale, mn


def pack_codes(q: torch.Tensor, bits: int) -> torch.Tensor:
    if bits == 8:
        return q.contiguous()
    if bits == 4:
        return (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()
    raise ValueError(f"unsupported bits={bits}")


def unpack_codes(codes: torch.Tensor, bits: int, k: int) -> torch.Tensor:
    if bits == 8:
        return codes
    out = torch.empty((codes.shape[0], k), dtype=torch.uint8, device=codes.device)
    out[:, 0::2] = codes & 0xF
    out[:, 1::2] = codes >> 4
    return out


def dequantize_ref(codes, scales, mins, bits: int, group: int, k: int) -> torch.Tensor:
    """Reference (non-Triton) dequantization, fp32."""
    q = unpack_codes(codes, bits, k).float()
    n = q.shape[0]
    q = q.reshape(n, k // group, group)
    return (q * scales.float()[..., None] + mins.float()[..., None]).reshape(n, k)


class QLinear:
    """y = x @ W^T + b with bf16 or weight-only int8/int4 storage."""

    def __init__(self, n_out: int, k_in: int, spec: QuantSpec, bias: torch.Tensor | None,
                 weight: torch.Tensor | None = None, codes=None, scales=None, mins=None):
        self.n_out, self.k_in, self.spec = n_out, k_in, spec
        self.bias = bias
        self.weight = weight
        self.codes, self.scales, self.mins = codes, scales, mins

    @classmethod
    def from_weight(cls, w: torch.Tensor, bias: torch.Tensor | None, spec: QuantSpec,
                    device, dtype=torch.bfloat16) -> "QLinear":
        n, k = w.shape
        b = bias.to(device=device, dtype=dtype) if bias is not None else None
        if spec.bits == 16:
            return cls(n, k, spec, b, weight=w.to(device=device, dtype=dtype).contiguous())
        group = k if spec.group <= 0 or spec.group > k or k % spec.group else spec.group
        spec = QuantSpec(spec.bits, group)
        codes, scales, mins = quantize_rtn(w.to(device), spec.bits, group)
        return cls(n, k, spec, b, codes=codes, scales=scales, mins=mins)

    @classmethod
    def from_quantized(cls, n: int, k: int, spec: QuantSpec, bias, codes, scales, mins) -> "QLinear":
        return cls(n, k, spec, bias, codes=codes, scales=scales, mins=mins)

    def nbytes(self) -> int:
        tensors = [self.weight, self.codes, self.scales, self.mins, self.bias]
        return sum(t.numel() * t.element_size() for t in tensors if t is not None)

    def dense(self, dtype=torch.bfloat16) -> torch.Tensor:
        if self.weight is not None:
            return self.weight
        from app.lite.gpu.kernels import dequantize

        return dequantize(self.codes, self.scales, self.mins, self.spec.bits,
                          self.spec.group, self.n_out, self.k_in, dtype=dtype)

    def rows(self, idx: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
        """Dequantized rows W[idx] (embedding lookup through a tied matrix)."""
        if self.weight is not None:
            return self.weight[idx].to(dtype)
        codes = unpack_codes(self.codes[idx], self.spec.bits, self.k_in).float()
        g = self.spec.group
        codes = codes.view(len(idx), self.k_in // g, g)
        w = codes * self.scales[idx].float()[..., None] + self.mins[idx].float()[..., None]
        return w.view(len(idx), self.k_in).to(dtype)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight is not None:
            return F.linear(x, self.weight, self.bias)
        shape = x.shape
        x2 = x.reshape(-1, shape[-1])
        if x2.shape[0] <= GEMV_MAX_ROWS:
            from app.lite.gpu.kernels import qgemv

            y = qgemv(x2.contiguous(), self.codes, self.scales, self.mins,
                      self.spec.bits, self.spec.group, self.n_out, self.bias)
        else:
            y = F.linear(x2, self.dense(x.dtype), self.bias)
        return y.reshape(*shape[:-1], self.n_out)
