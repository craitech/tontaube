"""Slider control tokens for control-enabled CB0 bundles.

A control-enabled CB0 bundle ships a
``controls.json`` next to its tokenizer. Each spec defines ``n_bins`` special tokens
``<|ctrl_{name}_{bin:02d}|>``. They are rendered into the CB0 system header right
after the style label:

    <|im_start|>english : audiobook<|ctrl_rate_03|><|ctrl_pause_02|><|im_end|>\\n

Tokens are emitted in spec order (training order). Only CB0
sees them; CB1-3 keep the plain header (their tokenizers have no control tokens).

Accepted request values:
  * ``{"rate": 3, "pause": 2}``: bin index per control (absent = not rendered)
  * a raw token string such as ``"<|ctrl_rate_03|>"`` (validated, used verbatim)
Per-segment values are merged over the request-level dict; a ``None`` value
removes a control for that segment, a raw string replaces the request value.

Registered versus public: ``ControlCatalog.specs`` / ``all_tokens`` are everything the bundle registers
(tokenizer check, boundary-id layout, lite vocab layout) and are never filtered. What clients can see
(``GET /controls``) and send (``render``) is the public view, ``PUBLIC_OVERRIDES`` applied on top.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, replace

CONTROLS_FILENAME = "controls.json"
_TOKEN_RE = re.compile(r"<\|[^|]+\|>")

ControlValue = dict | str | None

# Public view of the sliders of TontaubeV1.1, applied on top of the bundle's controls.json. The bundle declares
# seven sliders; this release offers five:
#   hesitation, varied_pacing: not offered. Hidden: None.
#   legato: offered with 3 steps (0 crispest, 1 crisp, 2 the model's default), one-sided: crisp articulation,
#     not legato. (The crisp steps also shorten pauses somewhat.)
# The override only narrows requests and the /controls listing. The tokens stay registered, so the tokenizer
# check, tokenize_text's boundary-id check and the lite vocab layout still see all seven sliders and all bins.
# Shrink or drop this table for a bundle that offers more.
# name -> None (hidden) or the fields to replace on the bundle's spec.
PUBLIC_OVERRIDES: dict[str, dict | None] = {
    "hesitation": None,
    "varied_pacing": None,
    "legato": {
        "n_bins": 3,
        "description": "crisp articulation",
        "low_description": "crisp staccato",
        "high_description": "normal articulation",
        "default_bin": 2,
    },
}


@dataclass(frozen=True)
class ControlSpec:
    name: str
    n_bins: int
    description: str = ""
    low_description: str = ""
    high_description: str = ""
    # The bin that sounds like the model without this slider, when known (None: not stated).
    default_bin: int | None = None

    def token(self, idx: int) -> str:
        return f"<|ctrl_{self.name}_{idx:02d}|>"


def _public_specs(specs: list[ControlSpec]) -> list[ControlSpec]:
    """``specs`` with PUBLIC_OVERRIDES applied (same order). Overrides only narrow: a bin count is never raised."""
    public = []
    for spec in specs:
        if spec.name not in PUBLIC_OVERRIDES:
            public.append(spec)
        elif (fields := PUBLIC_OVERRIDES[spec.name]) is not None:
            public.append(replace(spec, **{**fields, "n_bins": min(spec.n_bins, fields.get("n_bins", spec.n_bins))}))
    return public


class ControlCatalog:
    def __init__(self, specs: list[ControlSpec]):
        # Registered: everything in the bundle, in training order. Never filtered (see module docstring).
        self.specs = list(specs)
        self.by_name = {s.name: s for s in self.specs}
        self.all_tokens = {s.token(i) for s in self.specs for i in range(s.n_bins)}
        # Public: what GET /controls lists and requests may use.
        self.public_specs = _public_specs(self.specs)
        self.public_by_name = {s.name: s for s in self.public_specs}
        self.public_tokens = {s.token(i) for s in self.public_specs for i in range(s.n_bins)}

    @classmethod
    def load(cls, model_dir: str) -> "ControlCatalog | None":
        path = os.path.join(model_dir, CONTROLS_FILENAME)
        if not os.path.isfile(path):
            return None
        with open(path) as f:
            data = json.load(f)
        return cls([
            ControlSpec(
                str(s["name"]), int(s["n_bins"]), str(s.get("description", "")),
                str(s.get("low_description", "")), str(s.get("high_description", "")),
            )
            for s in data.get("specs", [])
        ])

    def check_tokenizer(self, tokenizer) -> None:
        """Every control token must be a real tokenizer entry: app.utils.tokenize_text
        silently drops unknown <|...|> tokens."""
        bad = [t for t in sorted(self.all_tokens)
               if tokenizer.convert_tokens_to_ids(t) in (None, tokenizer.unk_token_id)]
        if bad:
            raise ValueError(f"CB0 tokenizer lacks control tokens from {CONTROLS_FILENAME}: {bad[:5]}")

    def describe(self) -> dict[str, int]:
        """The public sliders and their bin counts (also the 'available' list of request errors)."""
        return {s.name: s.n_bins for s in self.public_specs}

    def catalog(self) -> list[dict]:
        """GET /controls: one entry per public slider, bins 0 (low) to bins-1 (high). ``default`` (only when
        known) is the bin that matches the model without the slider; a UI rests an unset slider there."""
        return [
            {"name": s.name, "bins": s.n_bins, "description": s.description,
             "low": s.low_description, "high": s.high_description,
             **({"default": s.default_bin} if s.default_bin is not None and 0 <= s.default_bin < s.n_bins else {})}
            for s in self.public_specs
        ]

    def merge(self, base: ControlValue, override: ControlValue) -> ControlValue:
        if override is None:
            return base
        if isinstance(override, str) or base is None or isinstance(base, str):
            return override
        merged = dict(base)
        merged.update(override)
        return merged

    def render(self, controls: ControlValue) -> str:
        if not controls:
            return ""
        if isinstance(controls, str):
            tokens = _TOKEN_RE.findall(controls)
            if "".join(tokens) != controls.strip():
                raise ValueError(f"control string must only contain control tokens: {controls!r}")
            unknown = [t for t in tokens if t not in self.public_tokens]
            if unknown:
                raise ValueError(f"unknown control tokens {unknown}; available: {self.describe()}")
            return "".join(tokens)
        if not isinstance(controls, dict):
            raise ValueError("controls must be a dict {name: bin} or a control-token string")
        unknown = [k for k in controls if k not in self.public_by_name]
        if unknown:
            raise ValueError(f"unknown controls {unknown}; available: {self.describe()}")
        parts = []
        for spec in self.public_specs:  # training order
            value = controls.get(spec.name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < spec.n_bins:
                raise ValueError(
                    f"control {spec.name!r} must be an integer bin in [0, {spec.n_bins}), got {value!r}"
                )
            parts.append(spec.token(value))
        return "".join(parts)


def render_system_tag(system_prompt: str, control_text: str) -> str:
    """The CB0 system header for a non-empty label."""
    return f"<|im_start|>{system_prompt}{control_text}<|im_end|>\n"
