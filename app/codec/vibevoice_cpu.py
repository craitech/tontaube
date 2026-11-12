"""VibeVoice acoustic tokenizer on the CPU: channels-last, fused kernels, int8 wide layers.

Drop-in for VibeVoiceAcousticTokenizerModel on the CPU (encode(audio).mean,
decode(latents, cache, sample_indices, use_cache)), computing the same fp32
function with a different memory layout:

- activations stay [batch, time, channels]: RMSNorm, the causal depthwise
  convolution and the FFN read contiguous rows and need no transposes;
- every convolution is one GEMM whose input rows are overlapping windows of
  the activation buffer (the k = 7 convolutions read rows t-6..t, the k = 2s
  strided convolutions two frames, the transposed convolutions the previous
  and the current frame), so there is no im2col copy;
- the streaming state of a causal layer is its last input rows, stored per
  stream in the caller's VibeVoiceTokenizerStreamingCache. A zero state is the
  non-streaming zero padding, so streaming and full decoding agree;
- blocks with C <= 256 (most of the FLOPs, few weights) run fused per tile of
  rows with fp32 weights; wider blocks (C >= 512, ~90% of the weights) and the
  large convolutions use int8 weights (asymmetric, group 128 along the input),
  dequantized one 16-column panel at a time. Against the fp32 reference this
  keeps 42-46 dB SI-SNR (the bf16 reference model the GPU runs: 37-41 dB) at
  713 MiB instead of 2.6 GiB.

Weights are read layer by layer from the bf16 checkpoint; the fp32 model of
the reference implementation (2.7 GiB) is never built. reference=True keeps
fp32 matrices and runs plain torch ops (the kernels' test oracle).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from app.native import kernels
from app.lite.quant import QuantSpec, quantize_rtn

FUSED_MAX_DIM = 256  # blocks up to this width run fused per row tile with fp32 weights
DW_K = 7  # depthwise kernel size of every block
_EMPTY = torch.empty(0)
_EMPTY_I8 = torch.empty(0, dtype=torch.int8)


@dataclass
class EncoderOutput:
    mean: torch.Tensor  # [B, T, vae_dim]


def _rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


class _Weight:
    """W [K, N] (y = x @ W) as kernel panels (fp32 or int8 g128), or fp32 for the reference path."""

    def __init__(self, w_kn: torch.Tensor, spec: QuantSpec | None, reference: bool):
        K, N = w_kn.shape
        self.K, self.N = K, N
        quant = spec is not None and spec.bits == 8 and K % 128 == 0 and N % 16 == 0
        if quant:
            codes, scales, mins = quantize_rtn(w_kn.t().contiguous(), 8, 128)  # per output column, groups along K
            sc = scales.float()
            if reference:
                self.dense = (codes.float().view(N, K // 128, 128) * sc[..., None]
                              + mins.float()[..., None]).view(N, K).t().contiguous()
                return
            mn = mins.float() + 128.0 * sc  # codes stored as code - 128
            q = (codes.to(torch.int16) - 128).to(torch.int8)
            self.parts = (_EMPTY,
                          q.view(N // 16, 16, K).permute(0, 2, 1).contiguous(),
                          sc.view(N // 16, 16, K // 128).permute(0, 2, 1).contiguous(),
                          mn.view(N // 16, 16, K // 128).permute(0, 2, 1).contiguous())
        elif reference or N % 16:
            self.dense = w_kn.float().contiguous()
            return
        else:
            self.parts = (kernels().vv_pack(w_kn.float()), _EMPTY_I8, _EMPTY, _EMPTY)
        self.dense = None

    def nbytes(self) -> int:
        if self.dense is not None:
            return self.dense.numel() * 4
        return sum(t.numel() * t.element_size() for t in self.parts)


class _State:
    """Per-layer streaming state in a VibeVoiceTokenizerStreamingCache (or none)."""

    def __init__(self, cache, sample_indices: torch.Tensor | None):
        self.cache = cache
        self.idx = sample_indices

    def get(self, key: str, rows: int, dim: int) -> torch.Tensor | None:
        if self.cache is None:
            return None
        st = self.cache.get(key, self.idx)
        return st.contiguous() if st is not None and st.shape[1:] == (rows, dim) else None

    def set(self, key: str, value: torch.Tensor) -> None:
        if self.cache is not None:
            self.cache.set(key, self.idx, value.contiguous())


def _tail(prev: torch.Tensor | None, x: torch.Tensor, rows: int) -> torch.Tensor:
    """Last `rows` rows of [prev (zeros if None); x]."""
    if x.shape[1] >= rows:
        return x[:, -rows:]
    if prev is None:
        prev = x.new_zeros(x.shape[0], rows, x.shape[2])
    return torch.cat([prev, x], dim=1)[:, -rows:]


class _Conv:
    """Causal stride-1 convolution: y[t] = b + [x[t-k+1] .. x[t]] @ W (k = kernel size)."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, key: str, spec: QuantSpec | None, reference: bool):
        cout, cin, k = weight.shape
        self.k, self.cin, self.cout, self.key, self.reference = k, cin, cout, key, reference
        w_kn = weight.float().permute(2, 1, 0).reshape(k * cin, cout)  # rows: (tap, cin), oldest tap first
        self.b = bias.float().contiguous()
        self.small = cin * cout <= 4096 and (cout % 8 == 0 or (cout == 1 and cin % 8 == 0))
        if self.small and not reference:
            self.w_small = w_kn.view(k, cin, cout).contiguous()
        else:
            self.w = _Weight(w_kn, spec, reference)

    def __call__(self, x: torch.Tensor, state: _State) -> torch.Tensor:
        B, T, C = x.shape
        prev = state.get(self.key, self.k - 1, C)
        if state.cache is not None:
            state.set(self.key, _tail(prev, x, self.k - 1))
        if self.reference:
            xp = torch.cat([prev if prev is not None else x.new_zeros(B, self.k - 1, C), x], dim=1)
            cols = xp.unfold(1, self.k, 1).transpose(2, 3).reshape(B * T, self.k * C)
            return torch.addmm(self.b, cols, self.w.dense).view(B, T, self.cout)
        prev = prev if prev is not None else _EMPTY
        if self.small:
            return kernels().vv_conv_small(x, prev, self.w_small, self.b)
        return kernels().vv_conv(x, prev, 1, self.k - 1, *self.w.parts, self.b)


class _Down:
    """Encoder downsampling: causal conv, kernel 2s, stride s (non-streaming)."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, spec: QuantSpec | None, reference: bool):
        cout, cin, k = weight.shape
        s = k // 2
        assert k == 2 * s
        self.s, self.cin, self.cout, self.reference = s, cin, cout, reference
        self.w = _Weight(weight.float().permute(2, 1, 0).reshape(k * cin, cout), spec, reference)
        self.b = bias.float().contiguous()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        n = -(-T // self.s)
        if n * self.s != T:  # the reference pads the end to whole frames
            x = F.pad(x, (0, 0, 0, n * self.s - T))
        if not self.reference:
            return kernels().vv_conv(x.contiguous(), _EMPTY, self.s, self.s, *self.w.parts, self.b)
        frames = F.pad(x, (0, 0, self.s, 0)).reshape(B, n + 1, self.s * C)
        cols = torch.cat([frames[:, :-1], frames[:, 1:]], dim=2)
        return torch.matmul(cols, self.w.dense).add_(self.b)


class _Up:
    """Decoder upsampling: causal transposed conv, kernel 2s, stride s, right trim s."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, key: str, spec: QuantSpec | None, reference: bool):
        cin, cout, k = weight.shape
        s = k // 2
        assert k == 2 * s
        self.s, self.cin, self.cout, self.key, self.reference = s, cin, cout, key, reference
        w = weight.float().permute(0, 2, 1)  # [cin, tap, cout]
        # Output block t = x[t-1] @ (taps s..2s-1) + x[t] @ (taps 0..s-1): window [x[t-1]; x[t]].
        w_kn = torch.cat([w[:, s:].reshape(cin, s * cout), w[:, :s].reshape(cin, s * cout)], dim=0)
        self.w = _Weight(w_kn, spec, reference)
        self.b = bias.float().repeat(s).contiguous()

    def __call__(self, x: torch.Tensor, state: _State) -> torch.Tensor:
        B, T, C = x.shape
        prev = state.get(self.key, 1, C)
        state.set(self.key, x[:, -1:])
        if self.reference:
            p = prev if prev is not None else x.new_zeros(B, 1, C)
            cols = torch.cat([torch.cat([p, x[:, :-1]], dim=1), x], dim=2)
            y = torch.matmul(cols, self.w.dense).add_(self.b)
        else:
            y = kernels().vv_conv(x, prev if prev is not None else _EMPTY, 1, 1, *self.w.parts, self.b)
        return y.view(B, T * self.s, self.cout)


class _Block:
    """Block1D: x += gamma * dwconv(rms(x)); x += ffn_gamma * FFN(rms(x))."""

    def __init__(self, w, prefix: str, dim: int, eps: float, key: str, spec: QuantSpec | None, reference: bool):
        g = lambda n: w.pop(prefix + n).float().contiguous()  # noqa: E731
        self.dim, self.eps, self.key, self.reference = dim, eps, key, reference
        self.norm_w = g("norm.weight")
        self.dw_w = g("mixer.conv.conv.conv.weight").view(dim, DW_K).t().contiguous()  # [tap, C]
        self.dw_b = g("mixer.conv.conv.conv.bias")
        self.gamma = g("gamma")
        self.ffn_norm_w = g("ffn_norm.weight")
        self.ffn_gamma = g("ffn_gamma")
        self.b1, self.b2 = g("ffn.linear1.bias"), g("ffn.linear2.bias")
        w1 = w.pop(prefix + "ffn.linear1.weight").float().t()  # [C, 4C]
        w2 = w.pop(prefix + "ffn.linear2.weight").float().t()  # [4C, C]
        self.fused = dim <= FUSED_MAX_DIM
        if self.fused and not reference:
            ext = kernels()
            self.w1p, self.w2p = ext.vv_pack(w1), ext.vv_pack(w2)
        else:
            spec = None if self.fused else spec
            self.w1, self.w2 = _Weight(w1, spec, reference), _Weight(w2, spec, reference)

    def nbytes(self) -> int:
        n = sum(t.numel() * t.element_size() for t in (self.norm_w, self.dw_w, self.dw_b, self.gamma,
                                                         self.ffn_norm_w, self.ffn_gamma, self.b1, self.b2))
        if hasattr(self, "w1p"):
            return n + (self.w1p.numel() + self.w2p.numel()) * 4
        return n + self.w1.nbytes() + self.w2.nbytes()

    def __call__(self, x: torch.Tensor, state: _State) -> torch.Tensor:
        B, T, C = x.shape
        prev = state.get(self.key, DW_K - 1, C)
        if self.reference:
            return self._reference(x, prev, state)
        ext = kernels()
        prev = prev if prev is not None else _EMPTY
        if self.fused:
            out, nxt = ext.vv_block(x, prev, self.norm_w, self.dw_w, self.dw_b, self.gamma, self.ffn_norm_w,
                                    self.w1p, self.b1, self.w2p, self.b2, self.ffn_gamma, self.eps)
        else:
            out, nxt = ext.vv_block_wide(x, prev, self.norm_w, self.dw_w, self.dw_b, self.gamma, self.ffn_norm_w,
                                         *self.w1.parts, self.b1, *self.w2.parts, self.b2, self.ffn_gamma, self.eps)
        state.set(self.key, nxt)
        return out

    def _reference(self, x: torch.Tensor, prev: torch.Tensor | None, state: _State) -> torch.Tensor:
        B, T, C = x.shape
        if prev is None:
            prev = x.new_zeros(B, DW_K - 1, C)
        h = torch.cat([prev, _rms(x, self.norm_w, self.eps)], dim=1)
        state.set(self.key, h[:, -(DW_K - 1):])
        y = self.dw_b + sum(h[:, j:j + T] * self.dw_w[j] for j in range(DW_K))
        x = x + self.gamma * y
        h2 = _rms(x, self.ffn_norm_w, self.eps).view(B * T, C)
        u = F.gelu(torch.addmm(self.b1, h2, self.w1.dense))
        v = torch.addmm(self.b2, u, self.w2.dense)
        return x + self.ffn_gamma * v.view(B, T, C)


class _Coder:
    """Encoder or decoder: per stage [resampling conv] + blocks, then head."""

    def __init__(self, w, prefix: str, depths: list[int], n_filters: int, eps: float, decoder: bool,
                 ffn_spec: QuantSpec | None, conv_spec: QuantSpec | None, reference: bool):
        self.stages = []
        for i, depth in enumerate(depths):
            lp = f"{prefix}.{'upsample' if decoder else 'downsample'}_layers.{i}.0."
            if i == 0:  # stem: k=7 stride-1 conv
                res = _Conv(w.pop(lp + "conv.conv.weight"), w.pop(lp + "conv.conv.bias"), f"{prefix}.stem",
                            conv_spec, reference)
            elif decoder:
                res = _Up(w.pop(lp + "convtr.convtr.weight"), w.pop(lp + "convtr.convtr.bias"), f"{prefix}.up{i}",
                          conv_spec, reference)
            else:
                res = _Down(w.pop(lp + "conv.conv.weight"), w.pop(lp + "conv.conv.bias"), conv_spec, reference)
            dim = n_filters * 2 ** (len(depths) - 1 - i) if decoder else n_filters * 2 ** i
            blocks = [_Block(w, f"{prefix}.stages.{i}.{j}.", dim, eps, f"{prefix}.s{i}b{j}", ffn_spec, reference)
                      for j in range(depth)]
            self.stages.append((res, blocks))
        self.head = _Conv(w.pop(f"{prefix}.head.conv.conv.weight"), w.pop(f"{prefix}.head.conv.conv.bias"),
                          f"{prefix}.head", conv_spec, reference)

    def nbytes(self) -> int:
        def size(layer):
            n = sum(t.numel() * t.element_size() for t in vars(layer).values() if isinstance(t, torch.Tensor))
            return n + (layer.w.nbytes() if isinstance(getattr(layer, "w", None), _Weight) else 0)

        return size(self.head) + sum(size(res) + sum(b.nbytes() for b in blocks) for res, blocks in self.stages)

    def __call__(self, x: torch.Tensor, state: _State) -> torch.Tensor:
        for res, blocks in self.stages:
            x = res(x) if isinstance(res, _Down) else res(x, state)
            for block in blocks:
                x = block(x, state)
        return self.head(x, state)


class CpuVibeVoice:
    """VibeVoice acoustic tokenizer (encoder + decoder) for the CPU."""

    def __init__(self, model_path: str | Path, ffn_quant: str | None = "int8g128",
                 conv_quant: str | None = "int8g128", reference: bool = False):
        from app.lite.plans import parse_spec
        from app.lite.weights import LazyWeights

        path = Path(model_path)
        cfg = json.loads((path / "config.json").read_text())
        spec = lambda name: parse_spec(name) if name and name not in ("fp32", "bf16") else None  # noqa: E731
        w = LazyWeights(path / "model.safetensors", mmap=True)
        enc_depths = [int(d) for d in str(cfg["encoder_depths"]).split("-")]
        dec_depths = cfg.get("decoder_depths") or list(reversed(enc_depths))
        if isinstance(dec_depths, str):
            dec_depths = [int(d) for d in dec_depths.split("-")]
        eps = cfg["layernorm_eps"]
        args = dict(ffn_spec=spec(ffn_quant), conv_spec=spec(conv_quant), reference=reference)
        self.encoder = _Coder(w, "encoder", enc_depths, cfg["encoder_n_filters"], eps, False, **args)
        self.decoder = _Coder(w, "decoder", dec_depths, cfg["decoder_n_filters"], eps, True, **args)
        self.vae_dim = cfg["vae_dim"]
        self.fix_std = cfg["fix_std"]
        self.device = torch.device("cpu")
        self._tontaube_dtype = torch.float32

    def nbytes(self) -> int:
        return self.encoder.nbytes() + self.decoder.nbytes()

    @torch.inference_mode()
    def encode(self, audio: torch.Tensor, cache=None, sample_indices=None, use_cache: bool = False) -> EncoderOutput:
        """audio [B, 1, samples] -> EncoderOutput(mean=[B, frames, vae_dim]) (non-streaming)."""
        x = audio.float().reshape(audio.shape[0], -1, 1).contiguous()
        return EncoderOutput(mean=self.encoder(x, _State(None, None)))

    @torch.inference_mode()
    def decode(self, latents: torch.Tensor, cache=None, sample_indices=None, use_cache: bool = False) -> torch.Tensor:
        """latents [B, vae_dim, T] or [B, T, vae_dim] -> audio [B, 1, T * hop]."""
        x = latents.float()
        if x.shape[1] == self.vae_dim:  # same layout rule as the reference model
            x = x.transpose(1, 2)
        state = _State(cache if use_cache else None, sample_indices)
        return self.decoder(x.contiguous(), state).transpose(1, 2).contiguous()
