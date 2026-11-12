"""One codebook model on the CPU: fp32 activations, dense bf16 KV cache, prefix reuse."""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from app.lite.config import ModelConfig, QuantPlan
from app.lite.cpu.linear import CpuLinear
from app.lite.quant import QLinear
from app.lite.weights import LazyWeights, load_codebook_weights, text_embedding_rows
from app.model.positions import get_codebook_vocab_size, prefill_positions
from app.native import kernels


def _rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def _rot(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


class CpuCodebookModel:
    """One codebook model on the CPU: fp32 activations, dense KV cache (one sequence)."""

    def __init__(self, path, target_codebook: int, plan: QuantPlan, max_len: int,
                 weights=None, prequant: dict[str, QLinear] | None = None,
                 kv_dtype: torch.dtype = torch.bfloat16, text_rows: list[int] | None = None):
        self.cfg = ModelConfig.from_dir(path, target_codebook)
        cfg = self.cfg
        self.max_len = max_len
        self.max_batch = 1
        self.local_ids = torch.tensor(cfg.local_to_global(), dtype=torch.long)
        self.n_local = get_codebook_vocab_size(target_codebook)
        self.eos_id = cfg.layout.eos_id
        if weights is None:
            weights = LazyWeights(Path(path) / "model.safetensors", mmap=True)
        cw = load_codebook_weights(weights, cfg, plan, torch.device("cpu"), torch.bfloat16, prequant, text_rows)
        self.embed_cpu, self.text_index = cw.text_embed, cw.text_index
        if cw.audio_embed is not None:
            self.embed_audio, self.embed_audio_scale = cw.audio_embed, None
        else:
            self.embed_audio, self.embed_audio_scale = cw.audio_codes, cw.audio_scale
        self.layers = [{
            "in_norm": L.in_norm.float(), "post_norm": L.post_norm.float(),
            "q_norm": L.q_norm.float(), "k_norm": L.k_norm.float(),
            "qkv": CpuLinear(L.qkv), "o": CpuLinear(L.o), "gate_up": CpuLinear(L.gate_up), "down": CpuLinear(L.down),
        } for L in cw.layers]
        self.norm = cw.norm.float()
        self.fc1, self.fc2 = CpuLinear(cw.fc1), CpuLinear(cw.fc2)
        self.inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, dtype=torch.float32) / cfg.head_dim))
        self.scale = cfg.head_dim ** -0.5
        # KV cache [layer][k|v][kv_head][position][head_dim]: per-head rows contiguous.
        # Uninitialized: rows are only read after being written, so pages commit lazily.
        self.kv = torch.empty((cfg.n_layers, 2, cfg.n_kv_heads, max_len + 1, cfg.head_dim), dtype=kv_dtype)
        # Tokens and positions whose KV the cache currently holds (prefix reuse).
        self.seq_ids: list[int] = []
        self.seq_pos: list[int] = []
        self.decoder = self._make_decoder() if kv_dtype == torch.bfloat16 else None

    def _make_decoder(self):
        """Fused one-token step (app/native/lm_kernels.cpp Decoder) when every layer linear is q4/q8."""
        cfg = self.cfg
        lins = [L[n] for L in self.layers for n in ("qkv", "o", "gate_up", "down")]
        if cfg.head_dim != 128 or any(lin.kind == "f32" for lin in lins):
            return None
        norms = [L[n] for L in self.layers for n in ("in_norm", "post_norm", "q_norm", "k_norm")]
        tensors = []
        for lin in lins:
            bias = lin.bias if lin.bias is not None else torch.empty(0)
            tensors += [lin.codes, lin.scales, lin.mins, bias]
        return kernels().Decoder(norms, tensors, [0 if lin.kind == "q4" else 1 for lin in lins],
                                 self.inv_freq, cfg.n_heads, cfg.n_kv_heads, cfg.rms_eps)

    def weight_bytes(self) -> int:
        lin = sum(L[n].nbytes() for L in self.layers for n in ("qkv", "o", "gate_up", "down"))
        return lin + self.fc1.nbytes() + self.fc2.nbytes()

    def logical_positions(self, ids: list[int]) -> list[int]:
        return prefill_positions(ids, self.cfg.layout)

    def reusable_prefix(self, ids: list[int], pos: list[int]) -> tuple[int, int]:
        """Longest cached prefix of (ids, pos) whose KV can be reused, and the position offset.

        Attention only sees relative positions, so a cached prefix whose positions differ
        from the requested ones by one constant offset is exact; the rest of the sequence
        then runs at pos + offset. At least one token is left to produce logits.
        """
        old_ids, old_pos = self.seq_ids, self.seq_pos
        n = min(len(old_ids), len(ids) - 1)
        if n <= 0 or old_ids[0] != ids[0]:
            return 0, 0
        offset = old_pos[0] - pos[0]
        k = 0
        while k < n and old_ids[k] == ids[k] and old_pos[k] - pos[k] == offset:
            k += 1
        return k, offset

    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
        start = self.cfg.gpu_vocab_start
        out = torch.empty((len(ids), self.cfg.hidden), dtype=torch.float32)
        text = ids < start
        if bool(text.any()):
            out[text] = text_embedding_rows(self.embed_cpu, self.text_index, ids[text]).float()
        if not bool(text.all()):
            a = ids[~text] - start
            if self.embed_audio_scale is None:
                out[~text] = self.embed_audio[a].float()
            else:
                out[~text] = self.embed_audio[a].float() * self.embed_audio_scale[a]
        return out

    @torch.inference_mode()
    def forward(self, ids: list[int], pos: list[int], start: int, all_hidden: bool = False):
        """Run tokens at cache indices start..start+T-1; returns last-token logits."""
        cfg = self.cfg
        T = len(ids)
        if T == 1 and self.decoder is not None and not all_hidden:
            return self._step(ids[0], pos[0], start)
        nq, nkv, hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        p = (torch.tensor(pos, dtype=torch.float32) + cfg.layout.chunk_realign_offset).clamp(
            0, cfg.max_position_embeddings - 1)
        ang = p[:, None] * self.inv_freq[None, :]
        emb = torch.cat((ang, ang), dim=-1)
        cos, sin = emb.cos()[:, None, :], emb.sin()[:, None, :]
        x = self._embed(torch.tensor(ids, dtype=torch.long))
        end = start + T
        del self.seq_ids[start:], self.seq_pos[start:]
        ext = kernels()
        for li, L in enumerate(self.layers):
            h = _rms(x, L["in_norm"], cfg.rms_eps)
            qkv = L["qkv"](h)
            q, k, v = qkv.split([nq * hd, nkv * hd, nkv * hd], dim=-1)
            q = _rms(q.view(T, nq, hd), L["q_norm"], cfg.rms_eps)
            k = _rms(k.view(T, nkv, hd), L["k_norm"], cfg.rms_eps)
            q = q * cos + _rot(q) * sin
            k = k * cos + _rot(k) * sin
            kc, vc = self.kv[li, 0], self.kv[li, 1]
            kc[:, start:end] = k.transpose(0, 1).to(kc.dtype)
            vc[:, start:end] = v.view(T, nkv, hd).transpose(0, 1).to(vc.dtype)
            if T == 1 and hd == 128 and kc.dtype == torch.bfloat16:
                a = ext.decode_attn(q[0], kc, vc, end, self.scale)[None]
            else:
                a = self._attend(q, kc, vc, start, end).reshape(T, nq * hd)
            x = x + L["o"](a)
            h = _rms(x, L["post_norm"], cfg.rms_eps)
            qh_act = ext.quantize_act(h) if T <= CpuLinear.GEMV_MAX_ROWS else None
            g, u = L["gate_up"](h, qh_act).chunk(2, dim=-1)
            x = x + L["down"](F.silu(g) * u)
        self.seq_ids.extend(ids)
        self.seq_pos.extend(pos)
        hs = _rms(x if all_hidden else x[-1:], self.norm, cfg.rms_eps)
        logits = self.fc2(F.mish(self.fc1(hs[-1:])))[0]
        return (logits, hs) if all_hidden else logits

    def _step(self, tok: int, pos: int, t: int) -> torch.Tensor:
        cfg = self.cfg
        p = float(min(max(pos + cfg.layout.chunk_realign_offset, 0), cfg.max_position_embeddings - 1))
        x = self._embed(torch.tensor([tok], dtype=torch.long))[0]
        del self.seq_ids[t:], self.seq_pos[t:]
        x = self.decoder.step(x, self.kv, p, t)
        self.seq_ids.append(tok)
        self.seq_pos.append(pos)
        h = _rms(x[None], self.norm, cfg.rms_eps)
        return self.fc2(F.mish(self.fc1(h)))[0]

    @staticmethod
    def _attend(q, kc, vc, start, end):
        """New tokens start..end-1 over the cache (fp32 from the bf16 cache, as in decode)."""
        T = end - start
        qh, kh, vh = q.transpose(0, 1)[None], kc[None, :, :end].float(), vc[None, :, :end].float()
        if start == 0:
            a = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, enable_gqa=True)
        else:  # causal within the new block, full view of the cached prefix
            mask = torch.ones((T, end), dtype=torch.bool)
            mask[:, start:] = torch.tril(torch.ones((T, T), dtype=torch.bool))
            a = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask, enable_gqa=True)
        return a[0].transpose(0, 1)
