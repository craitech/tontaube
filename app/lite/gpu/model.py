"""Single-process Qwen3 decoder for one TontaubeV1 codebook model.

Differences from the vLLM path that matter for memory:
  * one CUDA context for all models in the process,
  * a paged KV pool sized in tokens instead of a GPU-memory fraction,
  * weight-only int8/int4 linear layers (app.lite.quant),
  * the text part of the input embedding (ids below the end-of-speech token,
    i.e. the inherited Qwen vocabulary) stays in CPU memory: it is only read
    during prefill, while decode steps only ever feed audio and control
    tokens, whose rows stay on the GPU.

Numerics follow the HF/vLLM Qwen3 implementation in bf16: RMSNorm in fp32,
neox-style RoPE with bf16 cos/sin, q/k head norms, SiLU-gated MLP, and the
Tontaube audio head (Linear -> Mish -> Linear) evaluated on the last hidden
state only.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from app.lite.gpu.kernels import DecodeAttention, add_rmsnorm, qk_norm_rope_store, silu_mul
from app.lite.config import ModelConfig, QuantPlan
from app.lite.quant import QLinear
from app.lite.weights import load_codebook_weights, load_layers, text_embedding_rows
from app.model.positions import prefill_positions

BLOCK_SIZE = 64


class LiteCodebookModel:
    """Weights, paged KV pool, prefill and (graph-capturable) decode."""

    def __init__(
        self,
        path: str | Path,
        target_codebook: int,
        plan: QuantPlan,
        max_len: int,
        max_batch: int,
        kv_tokens: int,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        weights: dict | None = None,
        prequant: dict[str, QLinear] | None = None,
        text_rows: list[int] | None = None,
    ):
        self.path = str(path)
        self.cfg = self._config(path, target_codebook)
        self.text_rows = text_rows
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.dtype = dtype
        self.plan = plan
        self.prequant = prequant
        self.max_len = max_len
        self.max_batch = max_batch
        cfg = self.cfg
        self.local_ids = torch.tensor(cfg.local_to_global(), dtype=torch.long)
        self.n_local = len(self.local_ids)
        self.eos_id = cfg.eos_id

        if weights is None:
            from safetensors.torch import load_file

            weights = load_file(str(Path(path) / "model.safetensors"), device="cpu")
        self._load(weights, plan)
        del weights
        self.prequant = None

        inv_freq = 1.0 / (cfg.rope_theta ** (
            torch.arange(0, cfg.head_dim, 2, dtype=torch.float32, device=self.device) / cfg.head_dim))
        self.inv_freq = inv_freq

        # Paged KV pool (+1 scratch block for padded batch rows).
        self.blocks_per_seq = math.ceil((max_len + 1) / BLOCK_SIZE)
        n_blocks = max(math.ceil(kv_tokens / BLOCK_SIZE), self.blocks_per_seq)
        self.n_blocks = n_blocks
        self.scratch_block = n_blocks
        shape = (cfg.n_layers, 2, n_blocks + 1, BLOCK_SIZE, cfg.n_kv_heads, cfg.head_dim)
        self.kv = torch.zeros(shape, device=self.device, dtype=dtype)
        self.attn = DecodeAttention(max_batch, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim,
                                    self.blocks_per_seq * BLOCK_SIZE, BLOCK_SIZE, self.device)
        self._graphs: dict[int, tuple] = {}
        self._alloc_state()

    def _config(self, path, target_codebook) -> ModelConfig:
        return ModelConfig.from_dir(path, target_codebook)

    # ------------------------------------------------------------------ load
    def _load(self, w: dict, plan: QuantPlan):
        cw = load_codebook_weights(w, self.cfg, plan, self.device, self.dtype, self.prequant,
                                   getattr(self, "text_rows", None))
        self.embed_cpu, self.text_index = cw.text_embed, cw.text_index
        if torch.cuda.is_available() and self.device.type == "cuda":
            self.embed_cpu = self.embed_cpu.pin_memory()
        self.embed_gpu, self.embed_codes, self.embed_scale = cw.audio_embed, cw.audio_codes, cw.audio_scale
        self.layers, self.norm, self.fc1, self.fc2 = cw.layers, cw.norm, cw.fc1, cw.fc2
        torch.cuda.empty_cache()

    def linears(self) -> dict[str, QLinear]:
        out = {}
        for i, L in enumerate(self.layers):
            for name in ("qkv", "o", "gate_up", "down"):
                out[f"layers.{i}.{name}"] = getattr(L, name)
        out["head.fc1"], out["head.fc2"] = self.fc1, self.fc2
        return out

    def weight_bytes(self) -> dict[str, int]:
        lin = sum(q.nbytes() for L in self.layers for q in (L.qkv, L.o, L.gate_up, L.down))
        head = self.fc1.nbytes() + self.fc2.nbytes()
        if self.embed_gpu is not None:
            emb = self.embed_gpu.numel() * self.embed_gpu.element_size()
        else:
            emb = self.embed_codes.numel() + self.embed_scale.numel() * 4
        norms = sum(t.numel() * 2 for L in self.layers for t in (L.in_norm, L.post_norm, L.q_norm, L.k_norm))
        return {
            "linear": lin, "head": head, "embed_gpu": emb, "norms": norms,
            "kv_pool": self.kv.numel() * self.kv.element_size(),
            "embed_cpu": self.embed_cpu.numel() * self.embed_cpu.element_size(),
        }

    # ------------------------------------------------------------- helpers
    def text_embedding_rows(self, ids: torch.Tensor) -> torch.Tensor:
        return text_embedding_rows(self.embed_cpu, self.text_index, ids)

    def _embed_gpu_rows(self, ids_gpu: torch.Tensor) -> torch.Tensor:
        idx = ids_gpu - self.cfg.gpu_vocab_start
        if self.embed_gpu is not None:
            return self.embed_gpu[idx]
        return (self.embed_codes[idx].float() * self.embed_scale[idx]).to(self.dtype)

    def embed(self, ids: list[int]) -> torch.Tensor:
        """Prefill embedding: text rows from CPU memory, audio rows from GPU."""
        ids_cpu = torch.tensor(ids, dtype=torch.long)
        start = self.cfg.gpu_vocab_start
        is_text = ids_cpu < start
        out = torch.empty((len(ids), self.cfg.hidden), device=self.device, dtype=self.dtype)
        if bool(is_text.any()):
            rows = self.text_embedding_rows(ids_cpu[is_text])
            out[is_text.to(self.device)] = rows.to(self.device, non_blocking=True)
        if not bool(is_text.all()):
            gpu_ids = ids_cpu[~is_text].to(self.device)
            out[(~is_text).to(self.device)] = self._embed_gpu_rows(gpu_ids)
        return out

    def logical_positions(self, ids: list[int]) -> list[int]:
        return prefill_positions(ids, self.cfg.layout)

    def head_logits(self, h: torch.Tensor) -> torch.Tensor:
        """Audio head on final-normed hidden states [B, H] -> fp32 logits [B, n_local]."""
        return self.fc2(F.mish(self.fc1(h))).float()

    def _layers(self, x: torch.Tensor, pos: torch.Tensor, rows: torch.Tensor, attend):
        """Run all decoder layers; returns (final-normed hidden, last residual, last mlp out).

        `attend(q, layer_index)` computes attention for the new tokens after
        their k/v have been written to the pool rows.
        """
        cfg = self.cfg
        T = x.shape[0]
        nq, nkv, hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        off = cfg.pos_offset
        h, residual = add_rmsnorm(x, None, self.layers[0].in_norm, cfg.rms_eps)
        m = None
        for li, L in enumerate(self.layers):
            q = qk_norm_rope_store(
                L.qkv(h), L.q_norm, L.k_norm, pos, self.inv_freq,
                self.kv[li, 0], self.kv[li, 1], rows, nq, nkv, hd, cfg.rms_eps,
                off, cfg.max_position_embeddings,
            )
            a = attend(q, li).reshape(T, nq * hd)
            h, residual = add_rmsnorm(L.o(a), residual, L.post_norm, cfg.rms_eps)
            m = L.down(silu_mul(L.gate_up(h)))
            nxt = self.layers[li + 1].in_norm if li + 1 < len(self.layers) else self.norm
            h, residual = add_rmsnorm(m, residual, nxt, cfg.rms_eps)
        return h, residual

    # ------------------------------------------------------------- prefill
    @torch.inference_mode()
    def prefill(self, ids: list[int], block_ids: list[int], return_hidden: bool = False):
        """Run the prompt, write its KV into `block_ids`, return last-token logits.

        With return_hidden=True also returns the final-normed hidden states of
        every prompt token (used by the quantization tools).
        """
        cfg = self.cfg
        n = len(ids)
        nkv, hd = cfg.n_kv_heads, cfg.head_dim
        pos = torch.tensor(self.logical_positions(ids), device=self.device)
        blocks = torch.tensor(block_ids, device=self.device, dtype=torch.long)
        t = torch.arange(n, device=self.device)
        rows = blocks[t // BLOCK_SIZE] * BLOCK_SIZE + t % BLOCK_SIZE

        def attend(q, li):
            k = self.kv[li, 0].view(-1, nkv, hd).index_select(0, rows)
            v = self.kv[li, 1].view(-1, nkv, hd).index_select(0, rows)
            return F.scaled_dot_product_attention(
                q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
                is_causal=True, enable_gqa=True,
            )[0].transpose(0, 1)

        h, _ = self._layers(self.embed(ids), pos, rows, attend)
        logits = self.head_logits(h[-1:])[0]
        if return_hidden:
            return logits, h
        return logits

    # -------------------------------------------------------------- decode
    def _decode(self, ids, pos, table, seqlens, write_rows):
        def attend(q, li):
            return self.attn(q, self.kv[li, 0], self.kv[li, 1], table, seqlens)

        h, _ = self._layers(self._embed_gpu_rows(ids), pos, write_rows, attend)
        return self.head_logits(h)

    # Decode batches keep their state (input token, position, KV row, length,
    # block table) in static device buffers. The host writes them only when
    # the batch composition changes; between steps they are advanced on the
    # device, either eagerly (sampled rows) or inside the greedy CUDA graph,
    # which also takes the argmax so that greedy rows run many steps per
    # host round trip.
    GREEDY_CHUNK = 32

    def _alloc_state(self):
        mb, dev = self.max_batch, self.device
        self._in_ids = torch.full((mb,), self.cfg.gpu_vocab_start, device=dev, dtype=torch.long)
        self._in_pos = torch.zeros((mb,), device=dev, dtype=torch.long)
        self._in_table = torch.full((mb, self.blocks_per_seq), self.scratch_block, device=dev, dtype=torch.int32)
        self._in_lens = torch.ones((mb,), device=dev, dtype=torch.int32)
        self._in_rows = torch.full((mb,), self.scratch_block * BLOCK_SIZE, device=dev, dtype=torch.long)
        self._hist = torch.zeros((mb, self.GREEDY_CHUNK), device=dev, dtype=torch.long)
        self._step = torch.zeros((1,), device=dev, dtype=torch.long)
        self._local_ids_dev = self.local_ids.to(dev)
        self._b = self._g = 0

    def _state(self, g: int):
        return (self._in_ids[:g], self._in_pos[:g], self._in_table[:g], self._in_lens[:g], self._in_rows[:g])

    def _advance(self, g: int, next_ids: torch.Tensor):
        ids, pos, table, lens, rows = self._state(g)
        ids.copy_(next_ids)
        pos.add_(1)
        lens.add_(1)
        blk = table.gather(1, (pos // BLOCK_SIZE).unsqueeze(1)).squeeze(1)
        rows.copy_(blk.long() * BLOCK_SIZE + pos % BLOCK_SIZE)

    def _greedy_step(self, g: int):
        logits = self._decode(*self._state(g))
        nxt = self._local_ids_dev[logits.argmax(-1)]
        self._advance(g, nxt)
        self._hist[:g].index_copy_(1, self._step, nxt.unsqueeze(1))
        self._step.add_(1)

    def capture_graphs(self, batch_sizes: list[int]):
        """Capture logits and greedy-step graphs per batch size (one memory pool)."""
        pool = torch.cuda.graph_pool_handle()
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.inference_mode(), torch.cuda.stream(stream):
            for b in sorted(batch_sizes, reverse=True):
                self._reset_padding(0, b)
                for _ in range(2):  # warm up Triton JIT outside capture
                    self._decode(*self._state(b))
                    self._greedy_step(b)
                self._reset_padding(0, b)
                logits_graph = torch.cuda.CUDAGraph()
                # thread_local: CUDA calls from other threads (engine workers
                # starting up) must not invalidate this capture.
                with torch.cuda.graph(logits_graph, pool=pool, stream=stream,
                                      capture_error_mode="thread_local"):
                    out = self._decode(*self._state(b))
                greedy_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(greedy_graph, pool=pool, stream=stream,
                                      capture_error_mode="thread_local"):
                    self._greedy_step(b)
                self._graphs[b] = (logits_graph, out, greedy_graph)
        torch.cuda.current_stream().wait_stream(stream)
        self._reset_padding(0, self.max_batch)

    def graph_batch_size(self, b: int) -> int:
        if not self._graphs:
            return b
        for size in sorted(self._graphs):
            if size >= b:
                return size
        raise ValueError(f"no captured graph for batch {b}")

    def _reset_padding(self, start: int, end: int):
        """Rows [start, end) read and write only the scratch block."""
        self._in_ids[start:end].fill_(self.cfg.gpu_vocab_start)
        self._in_pos[start:end].zero_()
        self._in_table[start:end].fill_(self.scratch_block)
        self._in_lens[start:end].fill_(1)
        self._in_rows[start:end].fill_(self.scratch_block * BLOCK_SIZE)

    @torch.inference_mode()
    def set_batch(self, ids, pos, tables, seqlens, rows) -> None:
        """Load the state of a new batch composition (host lists/tensors)."""
        B = len(ids)
        g = self.graph_batch_size(B)
        dev = self.device
        self._in_ids[:B].copy_(torch.as_tensor(ids, dtype=torch.long).to(dev, non_blocking=True))
        self._in_pos[:B].copy_(torch.as_tensor(pos, dtype=torch.long).to(dev, non_blocking=True))
        self._in_table[:B].copy_(torch.as_tensor(tables, dtype=torch.int32).to(dev, non_blocking=True))
        self._in_lens[:B].copy_(torch.as_tensor(seqlens, dtype=torch.int32).to(dev, non_blocking=True))
        self._in_rows[:B].copy_(torch.as_tensor(rows, dtype=torch.long).to(dev, non_blocking=True))
        if g > B:
            self._reset_padding(B, g)
        self._b, self._g = B, g

    @torch.inference_mode()
    def step_logits(self) -> torch.Tensor:
        """Logits [B, n_local] for the current batch state (state not advanced)."""
        if self._g in self._graphs:
            graph, out, _ = self._graphs[self._g]
            graph.replay()
            return out[: self._b]
        return self._decode(*self._state(self._b))

    @torch.inference_mode()
    def advance(self, next_global_ids: torch.Tensor) -> None:
        """Feed the sampled tokens (device, global ids) as the next inputs."""
        nxt = torch.full((self._g,), self.cfg.gpu_vocab_start, device=self.device, dtype=torch.long)
        nxt[: self._b] = next_global_ids
        self._advance(self._g, nxt)

    @torch.inference_mode()
    def greedy_steps(self, k: int) -> torch.Tensor:
        """Run k greedy steps on the device; returns the global ids [B, k]."""
        assert 0 < k <= self.GREEDY_CHUNK
        self._step.zero_()
        if self._g in self._graphs:
            graph = self._graphs[self._g][2]
            for _ in range(k):
                graph.replay()
        else:
            for _ in range(k):
                self._greedy_step(self._g)
        return self._hist[: self._b, :k]

    @torch.inference_mode()
    def decode(self, ids: torch.Tensor, pos: torch.Tensor, tables: torch.Tensor,
               seqlens: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """One decode step for explicit inputs (analysis tools and tests)."""
        self.set_batch(ids, pos, tables, seqlens, rows)
        return self.step_logits()


class LiteCausalLM(LiteCodebookModel):
    """A plain Qwen3 causal LM with a tied LM head (the English verbalizer).

    Positions are 0..n-1; decode steps feed text tokens, so the embedding
    lives on the GPU. With a tied head the embedding rows are read from the
    (possibly quantized) LM-head matrix, so they are stored once.
    """

    def __init__(self, path, plan: QuantPlan, max_len: int, max_batch: int, kv_tokens: int, **kw):
        super().__init__(path, 0, plan, max_len, max_batch, kv_tokens, **kw)

    def _config(self, path, target_codebook) -> ModelConfig:
        return ModelConfig.causal_lm(path)

    def _load(self, w, plan: QuantPlan):
        cfg, dev, dt = self.cfg, self.device, self.dtype
        emb = w.pop("model.embed_tokens.weight")
        keys = w.keys() if hasattr(w, "keys") else set()
        head = w.pop("lm_head.weight") if "lm_head.weight" in keys else emb
        self.lm_head = QLinear.from_weight(head.to(dev), None, plan.spec("head.lm"), dev, dt)
        self._tied = head is emb
        self.embed_table = None if self._tied else emb.to(dev, dt)
        self.embed_cpu = torch.empty(0, cfg.hidden, dtype=dt)
        self.embed_gpu = self.embed_codes = None
        del emb, head
        self.fc1 = self.fc2 = None
        self.layers, self.norm = load_layers(w, cfg, plan, dev, dt)

    def linears(self):
        out = {f"layers.{i}.{n}": getattr(L, n) for i, L in enumerate(self.layers)
               for n in ("qkv", "o", "gate_up", "down")}
        out["head.lm"] = self.lm_head
        return out

    def weight_bytes(self) -> dict[str, int]:
        lin = sum(q.nbytes() for L in self.layers for q in (L.qkv, L.o, L.gate_up, L.down))
        emb = 0 if self.embed_table is None else self.embed_table.numel() * self.embed_table.element_size()
        return {"linear": lin, "head": self.lm_head.nbytes(), "embed_gpu": emb, "norms": 0,
                "kv_pool": self.kv.numel() * self.kv.element_size(), "embed_cpu": 0}

    def _embed_gpu_rows(self, ids_gpu: torch.Tensor) -> torch.Tensor:
        if self.embed_table is not None:
            return self.embed_table[ids_gpu]
        return self.lm_head.rows(ids_gpu, self.dtype)

    def embed(self, ids: list[int]) -> torch.Tensor:
        return self._embed_gpu_rows(torch.tensor(ids, dtype=torch.long, device=self.device))

    def logical_positions(self, ids: list[int]) -> list[int]:
        return list(range(len(ids)))

    def head_logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.lm_head(h).float()
