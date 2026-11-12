"""Produce a pre-quantized lite bundle with sequential GPTQ.

Calibration sequences are recorded engine calls (prompt + generated tokens,
as `tontaube quantize` records them), i.e. exactly the token streams the
models see in production. Layers are processed in order; each linear layer's
Hessian is accumulated from inputs produced by the already-quantized
preceding layers. Linear layers whose plan spec is bf16 are kept as is.

Usage:
  python -m app.lite.quantize.calibrate --model-dir TontaubeV1 --calls calls.pkl \
      --plan "int4g64,layers.27.down=int8g128,head=int8g128" --out BUNDLE_DIR --cbs 0
"""
from __future__ import annotations

import argparse
import pickle
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from app.lite.weights import save_linears
from app.lite.quantize.gptq import gptq_quantize
from app.lite.gpu.model import LiteCodebookModel
from app.lite.plans import parse_codebook_plans
from app.lite.quant import QLinear, QuantSpec, quantize_rtn
from app.lite.weights import LazyWeights
from app.model.positions import prefill_positions


def _rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def _rot(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


class _Seq:
    def __init__(self, model: LiteCodebookModel, prompt: list[int], forced: list[int]):
        cfg = model.cfg
        ids = prompt + forced
        n = len(prompt)
        pos = [p for p in prefill_positions(prompt, cfg.layout)] + [n + i for i in range(len(forced))]
        pos_t = (torch.tensor(pos, device=model.device) + cfg.layout.chunk_realign_offset).clamp(
            0, cfg.max_position_embeddings - 1)
        freqs = pos_t.float()[:, None] * model.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos = emb.cos().to(model.dtype)[:, None, :]
        self.sin = emb.sin().to(model.dtype)[:, None, :]
        self.x = model.embed(ids)
        self.n_prompt = n


class _Hessian:
    def __init__(self, k: int, device):
        self.H = torch.zeros((k, k), device=device, dtype=torch.float32)
        self.n = 0

    def add(self, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1]).float()
        self.H.addmm_(x.t(), x)
        self.n += x.shape[0]

    def get(self) -> torch.Tensor:
        return self.H / max(1, self.n)


def _attn_out(model, L, h, seq):
    cfg = model.cfg
    T = h.shape[0]
    nq, nkv, hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    q, k, v = L.qkv(h).split([nq * hd, nkv * hd, nkv * hd], dim=-1)
    q = _rms(q.view(T, nq, hd), L.q_norm, cfg.rms_eps)
    k = _rms(k.view(T, nkv, hd), L.k_norm, cfg.rms_eps)
    v = v.view(T, nkv, hd)
    q = q * seq.cos + _rot(q) * seq.sin
    k = k * seq.cos + _rot(k) * seq.sin
    return F.scaled_dot_product_attention(
        q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None],
        is_causal=True, enable_gqa=True,
    )[0].transpose(0, 1).reshape(T, nq * hd)


def _quantize(q: QLinear, spec: QuantSpec, H: torch.Tensor | None, method: str, device) -> QLinear:
    if spec.bits == 16:
        return q
    W = q.dense(torch.float32).float() if q.weight is None else q.weight.float()
    group = spec.group if 0 < spec.group <= q.k_in else q.k_in
    if method == "gptq" and H is not None:
        codes, scales, mins = gptq_quantize(W, H, spec.bits, group)
    else:
        codes, scales, mins = quantize_rtn(W, spec.bits, group)
    return QLinear.from_quantized(q.n_out, q.k_in, QuantSpec(spec.bits, group), q.bias, codes, scales, mins)


@torch.inference_mode()
def calibrate_codebook(model: LiteCodebookModel, plan, sequences: list[tuple[list[int], list[int]]],
                       method: str = "gptq", log=print) -> dict[str, QLinear]:
    cfg, dev = model.cfg, model.device
    seqs = [_Seq(model, p, f) for p, f in sequences]
    log(f"  {len(seqs)} sequences, {sum(s.x.shape[0] for s in seqs)} tokens")
    for li, L in enumerate(model.layers):
        t0 = time.time()
        # 1) qkv
        H = _Hessian(L.qkv.k_in, dev)
        for s in seqs:
            H.add(_rms(s.x, L.in_norm, cfg.rms_eps))
        L.qkv = _quantize(L.qkv, plan.spec(f"layers.{li}.qkv"), H.get(), method, dev)
        # 2) o
        H = _Hessian(L.o.k_in, dev)
        attn = []
        for s in seqs:
            a = _attn_out(model, L, _rms(s.x, L.in_norm, cfg.rms_eps), s)
            H.add(a)
            attn.append(a)
        L.o = _quantize(L.o, plan.spec(f"layers.{li}.o"), H.get(), method, dev)
        # 3) gate_up
        H = _Hessian(L.gate_up.k_in, dev)
        mids = []
        for s, a in zip(seqs, attn):
            mid = s.x + L.o(a)
            mids.append(mid)
            H.add(_rms(mid, L.post_norm, cfg.rms_eps))
        del attn
        L.gate_up = _quantize(L.gate_up, plan.spec(f"layers.{li}.gate_up"), H.get(), method, dev)
        # 4) down
        H = _Hessian(L.down.k_in, dev)
        acts = []
        for mid in mids:
            g, u = L.gate_up(_rms(mid, L.post_norm, cfg.rms_eps)).chunk(2, dim=-1)
            act = F.silu(g) * u
            H.add(act)
            acts.append(act)
        L.down = _quantize(L.down, plan.spec(f"layers.{li}.down"), H.get(), method, dev)
        for s, mid, act in zip(seqs, mids, acts):
            s.x = mid + L.down(act)
        del mids, acts, H
        torch.cuda.empty_cache()
        log(f"  layer {li}: {time.time() - t0:.1f}s")
    # Audio head: only positions whose logits are used (last prompt token onwards).
    H1 = _Hessian(model.fc1.k_in, dev)
    finals = []
    for s in seqs:
        h = _rms(s.x[s.n_prompt - 1:], model.norm, cfg.rms_eps)
        finals.append(h)
        H1.add(h)
    model.fc1 = _quantize(model.fc1, plan.spec("head.fc1"), H1.get(), method, dev)
    H2 = _Hessian(model.fc2.k_in, dev)
    for h in finals:
        H2.add(F.mish(model.fc1(h)))
    model.fc2 = _quantize(model.fc2, plan.spec("head.fc2"), H2.get(), method, dev)
    return model.linears()


def select_sequences(calls: list[dict], cb: int, n: int, seed: int = 0) -> list[tuple[list[int], list[int]]]:
    # cb0 calls must end in EOS or an audio marker: rollouts cut off at
    # max_tokens are out-of-distribution failures, not calibration material.
    stops = (151670, 151672)
    pool = [
        c for c in calls
        if c["cb"] == cb and c["item"] != "__warmup__" and len(c["output_ids"]) >= 8
        and (cb != 0 or c["output_ids"][-1] in stops)
    ]
    random.Random(seed).shuffle(pool)
    return [(c["prompt_ids"], c["output_ids"][:-1]) for c in pool[:n]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--calls", required=True, nargs="+")
    ap.add_argument("--plan", required=True, help="one plan, or four separated by ';'")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cbs", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--n-seqs", type=int, default=64)
    ap.add_argument("--method", choices=["gptq", "rtn"], default="gptq")
    args = ap.parse_args()

    calls = []
    for path in args.calls:
        calls += pickle.load(open(path, "rb"))
    plans = parse_codebook_plans(args.plan)
    for cb in args.cbs:
        t0 = time.time()
        path = Path(args.model_dir) / f"cb{cb}"
        model = LiteCodebookModel(path, cb, parse_codebook_plans("bf16")[cb], max_len=3300, max_batch=1,
                                  kv_tokens=64, weights=LazyWeights(path / "model.safetensors"))
        seqs = select_sequences(calls, cb, args.n_seqs)
        print(f"cb{cb}: {plans[cb].describe()} via {args.method}", flush=True)
        linears = calibrate_codebook(model, plans[cb], seqs, args.method,
                                     log=lambda m: print(m, flush=True))
        plan_text = args.plan.split(";")[cb] if ";" in args.plan else args.plan
        save_linears(Path(args.out) / f"cb{cb}.safetensors", linears,
                     {"plan": plan_text, "method": args.method,
                      "calibration": f"{len(seqs)} sequences"})
        print(f"cb{cb}: done in {time.time() - t0:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
