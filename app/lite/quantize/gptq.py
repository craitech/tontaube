"""GPTQ for the lite backend's packed int4/int8 format.

`gptq_quantize` implements the GPTQ column-wise error compensation (Frantar
et al., 2022) with asymmetric per-group parameters. With `act_order`, columns
are processed by decreasing Hessian diagonal while group parameters stay tied
to contiguous column ranges ("static groups"), so the packed layout used by
app.lite.gpu.kernels does not change.
"""
from __future__ import annotations

import torch

from app.lite.quant import pack_codes


def _group_params(w: torch.Tensor, bits: int, clip_search: bool, hdiag: torch.Tensor | None):
    """Asymmetric (scale, min) per row of w [N, G]; optional MSE clip search."""
    qmax = (1 << bits) - 1
    mn, mx = w.amin(-1), w.amax(-1)
    if not clip_search:
        scale = ((mx - mn) / qmax).clamp(min=1e-8).half().float()
        return scale, mn.half().float()
    best_err = None
    best = None
    weight = hdiag[None, :] if hdiag is not None else 1.0
    for shrink in (1.0, 0.97, 0.94, 0.91, 0.88, 0.85, 0.8, 0.75):
        lo, hi = mn * shrink, mx * shrink
        scale = ((hi - lo) / qmax).clamp(min=1e-8).half().float()
        lo = lo.half().float()
        q = torch.round((w - lo[:, None]) / scale[:, None]).clamp(0, qmax)
        err = (((q * scale[:, None] + lo[:, None]) - w) ** 2 * weight).sum(-1)
        if best_err is None:
            best_err, best = err, (scale, lo)
        else:
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            best = (torch.where(better, scale, best[0]), torch.where(better, lo, best[1]))
    return best


@torch.no_grad()
def gptq_quantize(W: torch.Tensor, H: torch.Tensor, bits: int, group: int,
                  blocksize: int = 128, percdamp: float = 0.01, act_order: bool = True,
                  clip_search: bool = True):
    """Quantize W [N, K] (fp32, cuda) given H = E[x x^T] [K, K].

    Returns (codes, scales fp16 [N, K/group], mins fp16 [N, K/group]) in the
    packed layout of app.lite.quant.
    """
    W = W.clone().float()
    N, K = W.shape
    if group <= 0 or group > K:
        group = K
    assert K % group == 0
    qmax = (1 << bits) - 1
    H = H.clone().float()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    hdiag = torch.diag(H).clone()

    # Static group parameters from the original weights.
    n_groups = K // group
    scales = torch.empty((N, n_groups), device=W.device)
    mins = torch.empty((N, n_groups), device=W.device)
    for g in range(n_groups):
        sl = slice(g * group, (g + 1) * group)
        s, m = _group_params(W[:, sl], bits, clip_search, hdiag[sl] / hdiag[sl].mean())
        scales[:, g], mins[:, g] = s, m

    perm = torch.argsort(hdiag, descending=True) if act_order else torch.arange(K, device=W.device)
    invperm = torch.argsort(perm)
    W = W[:, perm]
    H = H[perm][:, perm]
    col_group = (perm // group)  # group of each permuted column

    damp = percdamp * torch.mean(torch.diag(H))
    H += damp * torch.eye(K, device=W.device)
    L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    Q = torch.zeros((N, K), device=W.device, dtype=torch.uint8)
    for i1 in range(0, K, blocksize):
        i2 = min(i1 + blocksize, K)
        W1 = W[:, i1:i2].clone()
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w = W1[:, i]
            d = Hinv1[i, i]
            g = col_group[i1 + i]
            s, m = scales[:, g], mins[:, g]
            q = torch.round((w - m) / s).clamp(0, qmax)
            Q[:, i1 + i] = q.to(torch.uint8)
            err = (w - (q * s + m)) / d
            W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
            Err1[:, i] = err
        W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]

    Q = Q[:, invperm]
    return pack_codes(Q.contiguous(), bits), scales.half(), mins.half()
