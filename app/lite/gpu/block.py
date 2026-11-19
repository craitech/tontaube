"""Block verification for a plain causal LM (the verbalizer): many drafted tokens checked in one pass.

A pass takes, for each active input, its not-yet-cached committed tokens plus a drafted continuation,
and returns the model's greedy next token after every drafted position. The caller accepts the longest
draft prefix that agrees with those predictions; the prediction at the first disagreement comes for
free. The result is what greedy decoding produces one step at a time.

The pass is the batched decode step of LiteCodebookModel with the batch rows used as consecutive
tokens: every row writes its key and value into the paged pool before attention runs, and a row at
position p attends to the first p + 1 rows of its sequence, which is causal attention over cache plus
block. Inputs run side by side in lanes. All lanes share the pool blocks that the fixed prompt prefix
fills completely; beyond those, a lane takes blocks from a common free list as its sequence grows and
gives them back when it finishes. Body and head are captured as CUDA graphs per row count.
"""
from __future__ import annotations

import bisect
from pathlib import Path

import torch

from app.lite.gpu.model import BLOCK_SIZE, LiteCausalLM
from app.lite.plans import parse_plan
from app.lite.weights import LazyWeights

ROWS = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1024)
# Predictions per head pass. A pass that needs more runs the head several times: above 64 rows the quantized
# output layer would be expanded to full size (app/lite/quant.py), which costs more than 1 GiB of GPU memory.
HEADS = (1, 4, 8, 16, 24, 32, 48, 64)
MAX_ROWS, MAX_HEADS = ROWS[-1], 512
RESCORE = 4   # top candidates whose scores are recomputed in fp32 (the bf16 head rounds logits to about 0.1)


class BlockLM:
    def __init__(self, path: str, quant: str, max_len: int, lanes: int, prefix: list[int], blocks_per_lane: int = 6):
        self.max_len, self.lanes, self.prefix_len = max_len, lanes, len(prefix)
        self.max_rows, self.max_heads = MAX_ROWS, MAX_HEADS   # rows and predictions one pass can hold
        per_seq = -(-(max_len + 1) // BLOCK_SIZE)
        self.shared = shared = len(prefix) // BLOCK_SIZE        # pool blocks filled completely by the prefix
        # One lane alone must always fit; beyond that the pool is sized for typical inputs, not for the longest.
        free_blocks = max(per_seq - shared, blocks_per_lane * lanes)
        self.template = shared                                  # holds the prefix's last, partly filled block
        self.free = list(range(shared + 1, shared + 1 + free_blocks))[::-1]
        self.owned: list[list[int]] = [[] for _ in range(lanes)]
        self.model = m = LiteCausalLM(path, parse_plan(quant), max_len=max_len, max_batch=MAX_ROWS,
                                      kv_tokens=(shared + 1 + free_blocks) * BLOCK_SIZE,
                                      weights=LazyWeights(Path(path) / "model.safetensors"))
        self.device = m.device
        self.tables = torch.full((lanes + 1, per_seq), m.scratch_block, dtype=torch.int32)   # last row: the template
        self.tables[:, :shared] = torch.arange(shared)
        self.tables[lanes, shared] = self.template
        self.scratch_row = m.scratch_block * BLOCK_SIZE
        self.hidden = torch.zeros((MAX_ROWS, m.cfg.hidden), device=self.device, dtype=m.dtype)   # last pass, per row
        self.stream = torch.cuda.Stream(device=self.device)
        self.bodies: dict[int, tuple] = {}
        self.heads: dict[int, tuple] = {}
        pool = torch.cuda.graph_pool_handle()
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.inference_mode(), torch.cuda.stream(self.stream):
            for sizes, store, make, fn in ((ROWS, self.bodies, self._body_buffers, self._body),
                                           (HEADS, self.heads, self._head_buffers, self._head)):
                for n in sizes:
                    buffers = make(n)
                    for _ in range(2):   # warm up the kernels outside the capture
                        fn(*buffers)
                    graph = torch.cuda.CUDAGraph()
                    # thread_local: CUDA calls from other threads (engines starting up) must not invalidate the capture.
                    with torch.cuda.graph(graph, pool=pool, stream=self.stream, capture_error_mode="thread_local"):
                        out = fn(*buffers)
                    store[n] = (graph, buffers, out)
        torch.cuda.current_stream().wait_stream(self.stream)
        # The prefix stays in the pool: its full blocks are shared, its last block is copied into each lane's first.
        self.forward([(lanes, list(prefix), 0, len(prefix) - 1)])

    def reserve(self, lane: int, length: int) -> bool:
        """Give the lane pool blocks for a sequence of `length` tokens; False (nothing taken) if the pool is short."""
        owned = self.owned[lane]
        missing = -(-length // BLOCK_SIZE) - self.shared - len(owned)
        if missing <= 0:
            return True
        if missing > len(self.free):
            return False
        first = not owned
        for _ in range(missing):
            block = self.free.pop()
            self.tables[lane, self.shared + len(owned)] = block
            owned.append(block)
        rest = self.prefix_len - self.shared * BLOCK_SIZE
        if first and rest:
            with torch.inference_mode(), torch.cuda.stream(self.stream):
                kv = self.model.kv
                kv[:, :, owned[0], :rest] = kv[:, :, self.template, :rest]
        return True

    def release(self, lane: int) -> None:
        self.free.extend(self.owned[lane])
        self.owned[lane] = []
        self.tables[lane, self.shared:] = self.model.scratch_block

    def _body_buffers(self, n: int):
        m, dev = self.model, self.device
        return (torch.zeros(n, dtype=torch.long, device=dev),                                       # token ids
                torch.zeros(n, dtype=torch.long, device=dev),                                       # positions
                torch.full((n, m.blocks_per_seq), m.scratch_block, dtype=torch.int32, device=dev),  # block table per row
                torch.ones(n, dtype=torch.int32, device=dev),                                       # rows visible to the token
                torch.full((n,), self.scratch_row, dtype=torch.long, device=dev))                   # pool row it writes

    def _head_buffers(self, n: int):
        return (torch.zeros(n, dtype=torch.long, device=self.device),)                              # rows that need a prediction

    def _body(self, ids, pos, table, lens, rows):
        m = self.model
        h, _ = m._layers(m._embed_gpu_rows(ids), pos, rows, lambda q, li: m.attn(q, m.kv[li, 0], m.kv[li, 1], table, lens))
        self.hidden[:ids.shape[0]] = h
        return h

    def _head(self, need):
        m = self.model
        h = self.hidden[need]
        top = m.head_logits(h).topk(RESCORE, dim=-1).indices
        exact = torch.einsum("nch,nh->nc", m.lm_head.rows(top.reshape(-1), torch.float32).view(*top.shape, -1), h.float())
        return top.gather(1, exact.argmax(1, keepdim=True))[:, 0]

    @torch.inference_mode()
    def forward(self, parts: list[tuple[int, list[int], int, int]]) -> list[list[int]]:
        """parts: (lane, tokens, first position, index of the first token whose prediction is needed).

        Returns, per part, the model's next token after each needed position."""
        n = sum(len(tokens) for _, tokens, _, _ in parts)
        n_need = sum(len(tokens) - need_from for _, tokens, _, need_from in parts)
        body, (ids, pos, table, lens, rows), _ = self.bodies[ROWS[bisect.bisect_left(ROWS, n)]]
        size = ids.shape[0]
        h_ids = torch.zeros(size, dtype=torch.long)
        h_pos = torch.zeros(size, dtype=torch.long)
        h_lens = torch.ones(size, dtype=torch.int32)
        h_rows = torch.full((size,), self.scratch_row, dtype=torch.long)
        h_table = torch.full((size, self.tables.shape[1]), self.model.scratch_block, dtype=torch.int32)
        h_need = torch.zeros(n_need, dtype=torch.long)
        r = k = 0
        spans = []
        for lane, tokens, start, need_from in parts:
            t = len(tokens)
            if start + t > self.max_len:
                raise ValueError("input does not fit the verbalizer's sequence length")
            p = torch.arange(start, start + t)
            h_ids[r:r + t] = torch.tensor(tokens)
            h_pos[r:r + t] = p
            h_lens[r:r + t] = (p + 1).int()
            h_rows[r:r + t] = self.tables[lane][p // BLOCK_SIZE].long() * BLOCK_SIZE + p % BLOCK_SIZE
            h_table[r:r + t] = self.tables[lane]
            c = t - need_from
            h_need[k:k + c] = torch.arange(r + need_from, r + t)
            spans.append((k, c))
            r += t
            k += c
        with torch.cuda.stream(self.stream):
            ids.copy_(h_ids); pos.copy_(h_pos); lens.copy_(h_lens); rows.copy_(h_rows); table.copy_(h_table)
            body.replay()
            tokens_out = []
            for first in range(0, n_need, HEADS[-1]):
                chunk = h_need[first:first + HEADS[-1]]
                head, (need,), out = self.heads[HEADS[bisect.bisect_left(HEADS, len(chunk))]]
                need.zero_()
                need[:len(chunk)] = chunk
                head.replay()
                tokens_out += out[:len(chunk)].tolist()
        return [tokens_out[k:k + c] for k, c in spans]
