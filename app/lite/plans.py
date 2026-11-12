"""Quantization plan strings.

A plan is a comma-separated list. The first entry without "=" is the default
spec for linear layers; "pattern=spec" entries override layers whose name
contains `pattern` (first match wins). Layer names: layers.{i}.qkv/.o/
.gate_up/.down, head.fc1, head.fc2, and embed (the GPU-resident audio rows of
the input embedding; bf16 unless a rule sets embed=int8).

Specs: bf16 | int8 (per output channel) | int8g<G> | int4g<G>

Examples:
  bf16
  int8
  int4g64,head=int8,embed=int8
  int4g128,layers.0.=int8,layers.27.=int8

Per-codebook plans join four plans with ";" (cb0;cb1;cb2;cb3), or use one
plan for all codebooks.
"""
from __future__ import annotations

import re

from app.lite.config import QuantPlan
from app.lite.quant import QuantSpec

_SPEC_RE = re.compile(r"^(bf16|int(8|4)(?:g(\d+))?)$")


def parse_spec(text: str) -> QuantSpec:
    m = _SPEC_RE.match(text.strip().lower())
    if not m:
        raise ValueError(f"invalid quantization spec {text!r}")
    if m.group(1) == "bf16":
        return QuantSpec(16)
    bits = int(m.group(2))
    group = int(m.group(3)) if m.group(3) else 0  # 0 = one group per row
    if bits == 4 and group == 0:
        raise ValueError("int4 needs a group size, e.g. int4g64")
    return QuantSpec(bits, group)


def parse_plan(text: str) -> QuantPlan:
    default = QuantSpec(16)
    rules: list[tuple[str, QuantSpec]] = []
    for part in [p.strip() for p in text.split(",") if p.strip()]:
        if "=" in part:
            pattern, spec = part.split("=", 1)
            rules.append((pattern.strip(), parse_spec(spec)))
        else:
            default = parse_spec(part)
    return QuantPlan(default, rules)


def parse_codebook_plans(text: str, n_codebooks: int = 4) -> list[QuantPlan]:
    parts = [p for p in text.split(";")]
    if len(parts) == 1:
        parts = parts * n_codebooks
    if len(parts) != n_codebooks:
        raise ValueError(f"expected 1 or {n_codebooks} ';'-separated plans, got {len(parts)}")
    return [parse_plan(p) for p in parts]
