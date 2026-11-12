"""Classifier-free guidance on the voice prompt for the acoustic codebooks (CB1-3).

logits = cond + w * (cond - uncond), where cond is the normal acoustic input and uncond is the same input without the
front voice-prompt block (the codebook models' trained no-prompt layout). Enabled per codebook with
CFG_WEIGHT_PER_CB (e.g. "0,1,1,1"; the CB0 entry is ignored); all zeros (the default) leaves serving unchanged.

vLLM has no way to run a cond and an uncond sequence in lockstep (they are scheduled independently), so the guided
codebooks run in a batched PyTorch loop over the same weights (bf16, like the engines): transformers Qwen3Model + the
bundle's audio head, positions from the vLLM model's own front-prompt scheme, decode positions as vLLM (index +
chunk_realign_offset). Inputs are built by the unchanged TTSEngine code; only `_generate_with_engine` is routed here
for the guided codebooks, so prompt caps, text windows, rows and forced prefixes behave as before. Sampling follows
the request's SamplingParams (temperature 0 = greedy; top-p; top-k).
"""
from __future__ import annotations

import asyncio
import logging
import os
import types

import torch

logger = logging.getLogger(__name__)


class GuidedCodebookLM:
    """One CB_k bundle as a plain PyTorch model with batched cond / uncond decoding."""

    def __init__(self, path: str, codebook: int, weight: float, device: str = "cuda"):
        from safetensors.torch import load_file
        from transformers import Qwen3Config, Qwen3Model

        from app.model.vllm_model import (AudioTokenHead, Qwen3DualCodecForCausalLM, get_codebook_offset,
                                          get_codebook_vocab_size, ACOUSTIC_VOCAB_SIZE)

        cfg = Qwen3Config.from_pretrained(path)
        sd = load_file(os.path.join(path, "model.safetensors"))
        self.model = Qwen3Model(cfg)
        res = self.model.load_state_dict(
            {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}, strict=False)
        if res.unexpected_keys or [k for k in res.missing_keys if "rotary" not in k]:
            raise RuntimeError(f"cb{codebook}: missing {res.missing_keys[:5]} unexpected {res.unexpected_keys[:5]}")
        self.head = AudioTokenHead(cfg.hidden_size, cfg.audio_head_hidden, get_codebook_vocab_size(codebook))
        self.head.load_state_dict({k[len("audio_head."):]: v for k, v in sd.items() if k.startswith("audio_head.")})
        self.model.to(device, torch.bfloat16).eval()
        self.head.to(device, torch.bfloat16).eval()
        self.codebook, self.weight, self.device = codebook, float(weight), device
        self.n_vocab = ACOUSTIC_VOCAB_SIZE
        self.dc = cfg.audio_token_start_id
        self.off = cfg.chunk_realign_offset
        self.maxpos = cfg.max_position_embeddings
        self.start_id = self.dc + get_codebook_offset(codebook)
        self._scheme = Qwen3DualCodecForCausalLM._front_prompt_position_scheme
        self._ns = types.SimpleNamespace(
            dc_base_id=self.dc, pad_token_id=cfg.pad_token_id, text_marker_global_id=cfg.text_marker_token_id,
            audio_marker_global_id=cfg.audio_marker_token_id, eos_global_id=cfg.eos_token_id,
            chunk_realign_offset=self.off, target_codebook=codebook)

    def _positions(self, ids):
        """As Qwen3DualCodecForCausalLM._compute_prefill_positions + forward(): shifted, + offset, clamped."""
        raw, gap = self._scheme(self._ns, ids, len(ids))
        shift = (len(ids) - 1 - gap) - raw[-1]
        return [min(max(p + shift + self.off, 0), self.maxpos - 1) for p in raw]

    def _forward(self, seqs=None, past=None, new_tok=None, new_pos=None):
        if past is None:
            L = max(len(s[0]) for s in seqs)
            ids = torch.zeros((len(seqs), L), dtype=torch.long)
            pos = torch.zeros((len(seqs), L), dtype=torch.long)
            mask = torch.zeros((len(seqs), L), dtype=torch.long)
            for b, (i, p) in enumerate(seqs):
                ids[b, L - len(i):] = torch.tensor(i)
                pos[b, L - len(p):] = torch.tensor(p)
                mask[b, L - len(i):] = 1
            self._mask = mask.to(self.device)
            out = self.model(input_ids=ids.to(self.device), position_ids=pos.to(self.device),
                             attention_mask=self._mask, use_cache=True)
        else:
            self._mask = torch.cat([self._mask, torch.ones((len(new_tok), 1), dtype=torch.long,
                                                           device=self.device)], 1)
            out = self.model(input_ids=torch.tensor(new_tok, device=self.device)[:, None],
                             position_ids=torch.tensor(new_pos, device=self.device)[:, None],
                             attention_mask=self._mask, past_key_values=past, use_cache=True)
        return self.head(out.last_hidden_state[:, -1]).float()[:, :self.n_vocab], out.past_key_values

    @torch.no_grad()
    def generate(self, reqs, seed=0):
        """reqs: list of (input_ids, max_tokens, temperature, top_p, top_k). Returns global token ids per request."""
        w = self.weight
        seqs = []
        for ids, *_ in reqs:
            seqs.append([list(ids), self._positions(ids)])
            k = 0
            while k < len(ids) and ids[k] >= self.dc:   # uncond: drop the front prompt block
                k += 1
            u = list(ids[k:])
            seqs.append([u, self._positions(u)])
        n = len(reqs)
        ci = torch.arange(0, 2 * n, 2, device=self.device)
        ui = ci + 1
        T = torch.tensor([r[2] for r in reqs], device=self.device, dtype=torch.float32)
        tp = torch.tensor([r[3] for r in reqs], device=self.device, dtype=torch.float32)
        tk = [int(r[4]) if r[4] and r[4] > 0 else 0 for r in reqs]
        samp = T > 0
        nxt = [len(s[0]) + self.off for s in seqs]   # vLLM decode: raw index + offset
        n_max = max(r[1] for r in reqs)
        out = [[] for _ in reqs]
        g = torch.Generator(device=self.device).manual_seed(seed)
        logits, past = self._forward(seqs=[(s[0], s[1]) for s in seqs])
        for step in range(n_max):
            lc = logits[ci]
            lg = lc + w * (lc - logits[ui])
            choice = lg.argmax(-1)
            if samp.any():
                rows = torch.nonzero(samp).flatten().tolist()
                x = lg[samp] / T[samp, None]
                for j, r in enumerate(rows):   # top-k (vLLM: -1 / 0 = off)
                    if tk[r] > 0:
                        kth = torch.topk(x[j], min(tk[r], x.shape[-1])).values[-1]
                        x[j] = torch.where(x[j] < kth, torch.full_like(x[j], -float("inf")), x[j])
                pr = torch.softmax(x, -1)
                sp, si = torch.sort(pr, descending=True, dim=-1)
                sp = sp * ((torch.cumsum(sp, -1) - sp) < tp[samp, None])
                pick = torch.multinomial(sp / sp.sum(-1, keepdim=True), 1, generator=g)
                choice[samp] = si.gather(1, pick)[:, 0]
            toks = [self.start_id + c for c in choice.tolist()]
            for r in range(n):
                if step < reqs[r][1]:
                    out[r].append(toks[r])
            if step + 1 == n_max:
                break
            new_tok = [t for t in toks for _ in (0, 1)]   # cond and uncond get the same token
            new_pos = [min(nxt[j], self.maxpos - 1) for j in range(2 * n)]
            nxt = [x + 1 for x in nxt]
            logits, past = self._forward(past=past, new_tok=new_tok, new_pos=new_pos)
        return out


class _Batcher:
    """Collects concurrent calls for one codebook and runs them as one batch (bounded by a token budget)."""

    def __init__(self, lm: GuidedCodebookLM, max_tokens: int):
        self.lm, self.max_tokens, self.pending, self.wake, self.n = lm, max_tokens, [], asyncio.Event(), 0
        self.task = None

    async def submit(self, ids, sp):
        if self.task is None:
            self.task = asyncio.get_running_loop().create_task(self._run())
        fut = asyncio.get_running_loop().create_future()
        self.pending.append((list(ids), sp, fut))
        self.wake.set()
        return await fut

    async def _run(self):
        while True:
            await self.wake.wait()
            await asyncio.sleep(0.02)   # let concurrent segments join the batch
            self.wake.clear()
            while self.pending:
                group, tok = [], 0
                while self.pending and (not group or tok + 2 * len(self.pending[0][0]) <= self.max_tokens):
                    x = self.pending.pop(0)
                    group.append(x)
                    tok += 2 * len(x[0])
                reqs = [(ids, sp.max_tokens, sp.temperature or 0.0, sp.top_p if sp.top_p is not None else 1.0,
                         sp.top_k if sp.top_k is not None else -1) for ids, sp, _ in group]
                self.n += 1
                try:
                    res = await asyncio.to_thread(self.lm.generate, reqs, self.n)
                    for (_, _, fut), r in zip(group, res):
                        if not fut.done():
                            fut.set_result(r)
                except Exception as exc:  # fail this batch, keep serving
                    for _, _, fut in group:
                        if not fut.done():
                            fut.set_exception(exc)


def maybe_attach(engine, model_paths, weights, max_batch_tokens: int = 150_000):
    """Route the acoustic codebooks with weight > 0 to guided generation. No-op when all weights are 0."""
    ws = list(weights)
    if not any(w > 0 for w in ws[1:]):
        return False
    batchers = {}
    for cb in range(1, len(engine.engines)):
        w = ws[cb] if cb < len(ws) else 0.0
        if w > 0:
            logger.info("CB%d: classifier-free guidance on the voice prompt, w = %.2f (PyTorch path)", cb, w)
            batchers[id(engine.engines[cb])] = _Batcher(GuidedCodebookLM(model_paths[cb], cb, w), max_batch_tokens)
    orig = engine._generate_with_engine

    async def generate_with_engine(eng, prompt_ids, sampling_params, priority: int = 0):
        b = batchers.get(id(eng))
        if b is None:
            return await orig(eng, prompt_ids, sampling_params, priority=priority)
        from app.priority_gate import current_priority, get_gate, Priority
        if current_priority.get() == Priority.LOW:
            await get_gate().wait_if_low()
        return await b.submit(prompt_ids, sampling_params)

    engine._generate_with_engine = generate_with_engine
    return True
