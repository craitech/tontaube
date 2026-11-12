"""Linear layers on the CPU: int4/int8 GEMV for a few rows, blocked dequant + MKL for prompts."""
from __future__ import annotations

import threading

import torch
import torch.nn.functional as F

from app.lite.quant import QLinear, unpack_codes
from app.native import kernels


class CpuLinear:
    """y = x @ W^T + b with AVX2 int4/int8 decode kernels or dense fp32."""

    GEMV_MAX_ROWS = 8

    def __init__(self, q: QLinear):
        self.n_out, self.k_in = q.n_out, q.k_in
        self.bias = q.bias.float().contiguous() if q.bias is not None else None
        self.kind = "f32"
        bits, group = q.spec.bits, q.spec.group
        if q.weight is not None:
            self.weight = q.weight.float().contiguous()
            return
        codes = unpack_codes(q.codes.cpu(), bits, q.k_in)  # [N, K] uint8
        scales, mins = q.scales.cpu().float(), q.mins.cpu().float()
        if bits == 4 and group % 64 == 0:
            rep = group // 64
            c = codes.view(q.n_out, q.k_in // 64, 64)
            self.codes = (c[:, :, :32] | (c[:, :, 32:] << 4)).contiguous()
            self.scales = scales.repeat_interleave(rep, dim=1).half().contiguous()
            self.mins = mins.repeat_interleave(rep, dim=1).half().contiguous()
            self.kind = "q4"
        elif bits == 8 and (group % 128 == 0 or group == q.k_in) and q.k_in % 128 == 0:
            rep = (group if group <= q.k_in else q.k_in) // 128
            self.codes = (codes.to(torch.int16) - 128).to(torch.int8).view(q.n_out, q.k_in // 128, 128).contiguous()
            s = scales.repeat_interleave(rep, dim=1)
            self.scales = s.half().contiguous()
            self.mins = (mins.repeat_interleave(rep, dim=1) + 128.0 * s).half().contiguous()
            self.kind = "q8"
        else:  # unsupported packing: dense fp32
            from app.lite.quant import dequantize_ref

            self.weight = dequantize_ref(q.codes.cpu(), q.scales.cpu(), q.mins.cpu(), bits, group, q.k_in).contiguous()

    def nbytes(self) -> int:
        tensors = [getattr(self, a, None) for a in ("weight", "codes", "scales", "mins", "bias")]
        return sum(t.numel() * t.element_size() for t in tensors if t is not None)

    # Prompt processing dequantizes this many fp32 bytes of rows at a time, so
    # the dense block stays in L3 between dequantization and the MKL matmul.
    BLOCK_BYTES = 8 << 20

    def __call__(self, x: torch.Tensor, qx=None) -> torch.Tensor:
        """x [M, K] fp32; qx: optional pre-quantized activations for x."""
        if self.kind == "f32":
            return F.linear(x, self.weight, self.bias)
        ext = kernels()
        if x.shape[0] <= self.GEMV_MAX_ROWS:
            q, s, xs = qx if qx is not None else ext.quantize_act(x)
            fn = ext.gemv_q4 if self.kind == "q4" else ext.gemv_q8
            return fn(q, s, xs, self.codes, self.scales, self.mins, self.bias)
        fn = ext.dequant_q4 if self.kind == "q4" else ext.dequant_q8
        rows = max(64, self.BLOCK_BYTES // (4 * self.k_in) // 64 * 64)
        w = _workspace(min(rows, self.n_out) * self.k_in).view(-1, self.k_in)
        y = torch.empty((x.shape[0], self.n_out), dtype=torch.float32)
        for r0 in range(0, self.n_out, rows):
            r1 = min(r0 + rows, self.n_out)
            fn(self.codes, self.scales, self.mins, w, r0, r1)
            if self.bias is not None:
                torch.addmm(self.bias[r0:r1], x, w[: r1 - r0].t(), out=y[:, r0:r1])
            else:
                torch.mm(x, w[: r1 - r0].t(), out=y[:, r0:r1])
        return y


_WS = threading.local()


def _workspace(numel: int) -> torch.Tensor:
    buf = getattr(_WS, "buf", None)
    if buf is None or buf.numel() < numel:
        buf = _WS.buf = torch.empty(numel, dtype=torch.float32)
    return buf[:numel]
