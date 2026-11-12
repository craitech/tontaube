"""vLLM-compatible sampling for the lite backend.

`SamplingParams` / `TokensPrompt` cover the subset of the vLLM API that
Tontaube uses, so app.inference can build requests without importing vLLM.
`Sampler` reproduces vLLM V1's order of operations:

  logit_bias + min_tokens stop masking -> frequency penalty -> greedy, or
  temperature -> top-k -> top-p (sorted cumulative mass) -> exponential-race
  sampling, with a per-request torch.Generator when a seed is given.

The model emits logits only over its audio-head vocabulary ("local" ids);
vLLM's full-vocabulary logits are -1e9 elsewhere, so restricting sampling to
the local ids is equivalent.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

_SAMPLING_EPS = 1e-5


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    frequency_penalty: float = 0.0
    seed: int | None = None
    max_tokens: int | None = 16
    min_tokens: int = 0
    stop_token_ids: list[int] | None = None
    logit_bias: dict[int, float] | None = None
    ignore_eos: bool = False


def TokensPrompt(prompt_token_ids: list[int]) -> dict:  # noqa: N802 (mirrors vLLM)
    return {"prompt_token_ids": prompt_token_ids}


@dataclass
class SeqSampling:
    """Per-sequence sampling state, resolved against the local vocabulary."""

    params: SamplingParams
    n_local: int
    global_to_local: dict[int, int]
    eos_id: int
    device: torch.device
    bias: torch.Tensor | None = None
    stop_local: torch.Tensor | None = None
    stop_global: frozenset = field(default_factory=frozenset)
    generator: torch.Generator | None = None
    counts: torch.Tensor | None = None

    def __post_init__(self):
        p = self.params
        if p.logit_bias:
            bias = torch.zeros(self.n_local, dtype=torch.float32)
            for gid, value in p.logit_bias.items():
                lid = self.global_to_local.get(int(gid))
                if lid is not None:
                    bias[lid] += float(value)
            self.bias = bias.to(self.device)
        stops = set(p.stop_token_ids or [])
        if not p.ignore_eos:
            stops.add(self.eos_id)
        self.stop_global = frozenset(stops)
        local = [self.global_to_local[g] for g in stops if g in self.global_to_local]
        if local:
            self.stop_local = torch.tensor(sorted(local), device=self.device)
        if p.seed is not None:
            self.generator = torch.Generator(device=self.device)
            self.generator.manual_seed(int(p.seed))
        if p.frequency_penalty:
            self.counts = torch.zeros(self.n_local, dtype=torch.float32, device=self.device)

    @property
    def greedy(self) -> bool:
        return self.params.temperature < _SAMPLING_EPS

    def top_k(self) -> int:
        k = self.params.top_k
        return self.n_local if k is None or k <= 0 or k >= self.n_local else int(k)


def _apply_top_k_top_p(logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None):
    """vLLM's apply_top_k_top_p (sort-based)."""
    if p is None:
        if k is None:
            return logits
        # top-k only: keep logits >= k-th largest value.
        max_k = int(k.max())
        kth = logits.topk(max_k, dim=-1).values.gather(1, (k - 1).unsqueeze(1).long())
        return logits.masked_fill(logits < kth, float("-inf"))
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)
    if k is not None:
        top_k_mask = logits_sort.size(1) - k.to(torch.long)
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        logits_sort.masked_fill_(logits_sort < top_k_mask, float("-inf"))
    probs_sort = logits_sort.softmax(dim=-1)
    probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
    top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
    top_p_mask[:, -1] = False
    logits_sort.masked_fill_(top_p_mask, float("-inf"))
    return logits_sort.scatter(dim=-1, index=logits_idx, src=logits_sort)


def sample(logits: torch.Tensor, seqs: list[SeqSampling], n_generated: list[int]) -> torch.Tensor:
    """Sample one local token id per row of fp32 logits [B, V]. Returns int64 [B]."""
    B = logits.shape[0]
    device = logits.device
    for i, s in enumerate(seqs):
        if s.bias is not None:
            logits[i] += s.bias
        if s.stop_local is not None and n_generated[i] < s.params.min_tokens:
            logits[i, s.stop_local] = float("-inf")
        if s.counts is not None:
            logits[i] -= s.params.frequency_penalty * s.counts
    greedy = [s.greedy for s in seqs]
    if all(greedy):
        return logits.argmax(dim=-1)

    temps = torch.tensor([1.0 if g else s.params.temperature for g, s in zip(greedy, seqs)],
                         device=device, dtype=torch.float32)
    greedy_tokens = logits.argmax(dim=-1) if any(greedy) else None
    logits = logits / temps.unsqueeze(1)
    ks = [s.top_k() for s in seqs]
    ps = [float(s.params.top_p) if s.params.top_p is not None else 1.0 for s in seqs]
    k = None if all(x == s.n_local for x, s in zip(ks, seqs)) else torch.tensor(ks, device=device)
    p = None if all(x >= 1.0 for x in ps) else torch.tensor(ps, device=device, dtype=torch.float32)
    logits = _apply_top_k_top_p(logits, k, p)
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    q = torch.empty_like(probs)
    if any(s.generator is None for s in seqs):
        q.exponential_()
    for i, s in enumerate(seqs):
        if s.generator is not None:
            q[i].exponential_(generator=s.generator)
    tokens = probs.div_(q).argmax(dim=-1)
    if greedy_tokens is not None:
        mask = torch.tensor(greedy, device=device)
        tokens = torch.where(mask, greedy_tokens, tokens)
    return tokens


def update_counts(seqs: list[SeqSampling], tokens: torch.Tensor) -> None:
    for i, s in enumerate(seqs):
        if s.counts is not None:
            s.counts[tokens[i]] += 1
