import pytest
import torch

from app.lite.plans import parse_codebook_plans, parse_plan, parse_spec
from app.lite.quant import QuantSpec
from app.lite.sampling import SamplingParams, SeqSampling, _apply_top_k_top_p, sample

V = 64
EOS = 1000
G2L = {2000 + i: i for i in range(V - 1)} | {EOS: V - 1}


def seq(**kw) -> SeqSampling:
    return SeqSampling(SamplingParams(**kw), V, G2L, EOS, torch.device("cpu"))


def test_plan_parsing():
    assert parse_spec("bf16") == QuantSpec(16)
    assert parse_spec("int8") == QuantSpec(8, 0)
    assert parse_spec("INT4G64") == QuantSpec(4, 64)
    with pytest.raises(ValueError):
        parse_spec("int4")
    plan = parse_plan("int4g64,layers.27.down=int8g128,head=bf16")
    assert plan.spec("layers.3.qkv") == QuantSpec(4, 64)
    assert plan.spec("layers.27.down") == QuantSpec(8, 128)
    assert plan.spec("layers.2.down") == QuantSpec(4, 64)
    assert plan.spec("head.fc2") == QuantSpec(16)
    # The embedding is only quantized when a rule names it.
    assert plan.spec("embed") == QuantSpec(16)
    assert parse_plan("int8,embed=int8").spec("embed") == QuantSpec(8, 0)
    assert len(parse_codebook_plans("bf16")) == 4
    plans = parse_codebook_plans("int8;int4g64;int4g64;bf16")
    assert [p.default.bits for p in plans] == [8, 4, 4, 16]
    with pytest.raises(ValueError):
        parse_codebook_plans("int8;int8")


# Verbatim reference copies of vLLM 0.16.0 v1/sample/ops/topk_topp_sampler.py
# (the test suite mocks the vllm package, so the originals cannot be imported).
def _vllm_apply_top_k_only(logits, k):
    no_top_k_mask = k == logits.shape[1]
    k = k.masked_fill(no_top_k_mask, 1)
    max_top_k = k.max()
    k_index = k.sub_(1).unsqueeze(1)
    top_k_mask = logits.topk(max_top_k, dim=1).values.gather(1, k_index.long())
    top_k_mask.masked_fill_(no_top_k_mask.unsqueeze(1), -float("inf"))
    return logits.masked_fill_(logits < top_k_mask, -float("inf"))


def _vllm_apply_top_k_top_p(logits, k, p):
    if p is None:
        if k is None:
            return logits
        return _vllm_apply_top_k_only(logits, k)
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)
    if k is not None:
        top_k_mask = logits_sort.size(1) - k.to(torch.long)
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float("inf"))
    if p is not None:
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float("inf"))
    return logits_sort.scatter(dim=-1, index=logits_idx, src=logits_sort)


def _vllm_random_sample(probs, generators):
    q = torch.empty_like(probs)
    if len(generators) != probs.shape[0]:
        q.exponential_()
    if generators:
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)
    return probs.div_(q).argmax(dim=-1).view(-1)


class _VllmOps:
    apply_top_k_top_p = staticmethod(_vllm_apply_top_k_top_p)
    random_sample = staticmethod(_vllm_random_sample)


def test_top_k_top_p_matches_vllm():
    vllm_ops = _VllmOps
    torch.manual_seed(0)
    logits = torch.randn(5, V) * 3
    k = torch.tensor([1, 5, V, 20, 7])
    p = torch.tensor([0.9, 1.0, 0.5, 0.95, 0.99])
    ours = _apply_top_k_top_p(logits.clone(), k, p)
    ref = vllm_ops.apply_top_k_top_p(logits.clone(), k, p)
    assert torch.equal(torch.isinf(ours), torch.isinf(ref))
    ours_k = _apply_top_k_top_p(logits.clone(), k, None)
    ref_k = vllm_ops.apply_top_k_top_p(logits.clone(), k, None)
    assert torch.equal(torch.isinf(ours_k), torch.isinf(ref_k))


def test_seeded_sampling_matches_vllm_random_sample():
    vllm_ops = _VllmOps
    torch.manual_seed(1)
    logits = torch.randn(1, V)
    s = seq(temperature=0.8, top_k=10, top_p=0.9, seed=42)
    ours = [int(sample(logits.clone(), [s], [5])[0]) for _ in range(20)]
    gen = torch.Generator().manual_seed(42)
    ref = []
    for _ in range(20):
        x = vllm_ops.apply_top_k_top_p(logits.clone() / 0.8, torch.tensor([10]), torch.tensor([0.9]))
        probs = x.softmax(dim=-1, dtype=torch.float32)
        ref.append(int(vllm_ops.random_sample(probs, {0: gen})[0]))
    assert ours == ref


def test_greedy_bias_and_min_tokens():
    logits = torch.zeros(2, V)
    logits[:, V - 1] = 5.0  # EOS would win
    logits[:, 3] = 4.0
    a = seq(temperature=0.0, min_tokens=2, stop_token_ids=[EOS])
    b = seq(temperature=0.0, logit_bias={EOS: -10.0, 2000 + 7: 6.0})
    out = sample(logits.clone(), [a, b], [0, 0])
    assert out.tolist() == [3, 7]
    # Once min_tokens is reached the stop token can be produced again.
    assert sample(logits.clone()[:1], [a], [2]).tolist() == [V - 1]
    assert EOS in a.stop_global
    assert EOS not in seq(ignore_eos=True).stop_global


def test_mixed_greedy_and_random_rows():
    torch.manual_seed(0)
    logits = torch.randn(3, V)
    rows = [seq(temperature=0.0), seq(temperature=1.0, seed=3), seq(temperature=0.0)]
    out = sample(logits.clone(), rows, [0, 0, 0])
    assert out[0] == logits[0].argmax() and out[2] == logits[2].argmax()
