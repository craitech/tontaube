"""What the lite engines need to know about a model: its shape and how to quantize it."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.lite.quant import QuantSpec
from app.model.positions import ACOUSTIC_VOCAB_SIZE, SEMANTIC_VOCAB_SIZE, LayoutIds, get_codebook_offset


def _rope_theta(cfg: dict) -> float:
    """rope_theta from either config layout (transformers 4: top level, 5: rope_parameters)."""
    params = cfg.get("rope_parameters") or {}
    if params.get("rope_type", "default") != "default" or cfg.get("rope_scaling"):
        raise ValueError("only unscaled (default) RoPE is supported")
    return float(cfg.get("rope_theta", params.get("rope_theta", 10000.0)))


@dataclass(frozen=True)
class ModelConfig:
    hidden: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    intermediate: int
    rms_eps: float
    rope_theta: float
    vocab_size: int
    max_position_embeddings: int
    audio_head_hidden: int | None
    layout: LayoutIds | None  # None: plain causal LM (standard positions)
    eos_id: int | None = None

    @classmethod
    def from_dir(cls, path: str | Path, target_codebook: int | None = None) -> "ModelConfig":
        cfg = json.loads((Path(path) / "config.json").read_text())
        target = cfg["target_codebook"] if target_codebook is None else target_codebook
        if cfg.get("chunk_realign_offset") != 25:
            raise ValueError("TontaubeV1 requires chunk_realign_offset=25")
        return cls(
            hidden=cfg["hidden_size"],
            n_layers=cfg["num_hidden_layers"],
            n_heads=cfg["num_attention_heads"],
            n_kv_heads=cfg["num_key_value_heads"],
            head_dim=cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"],
            intermediate=cfg["intermediate_size"],
            rms_eps=cfg["rms_norm_eps"],
            rope_theta=_rope_theta(cfg),
            vocab_size=cfg["vocab_size"],
            max_position_embeddings=cfg.get("max_position_embeddings", 32768),
            audio_head_hidden=cfg["audio_head_hidden"],
            layout=LayoutIds(
                dc_base_id=cfg["audio_token_start_id"],
                pad_token_id=cfg["pad_token_id"],
                text_marker_id=cfg["text_marker_token_id"],
                audio_marker_id=cfg["audio_marker_token_id"],
                eos_id=cfg["eos_token_id"],
                chunk_realign_offset=cfg["chunk_realign_offset"],
                target_codebook=target,
            ),
            eos_id=cfg["eos_token_id"],
        )

    @classmethod
    def causal_lm(cls, path: str | Path) -> "ModelConfig":
        """A standard Qwen3 causal LM (e.g. the verbalizer)."""
        cfg = json.loads((Path(path) / "config.json").read_text())
        eos = cfg.get("eos_token_id")
        gen = Path(path) / "generation_config.json"
        if gen.exists():
            eos = json.loads(gen.read_text()).get("eos_token_id", eos)
        if isinstance(eos, list):
            eos = eos[0]
        return cls(
            hidden=cfg["hidden_size"],
            n_layers=cfg["num_hidden_layers"],
            n_heads=cfg["num_attention_heads"],
            n_kv_heads=cfg["num_key_value_heads"],
            head_dim=cfg.get("head_dim") or cfg["hidden_size"] // cfg["num_attention_heads"],
            intermediate=cfg["intermediate_size"],
            rms_eps=cfg["rms_norm_eps"],
            rope_theta=_rope_theta(cfg),
            vocab_size=cfg["vocab_size"],
            max_position_embeddings=cfg.get("max_position_embeddings", 32768),
            audio_head_hidden=None,
            layout=None,
            eos_id=eos,
        )

    @property
    def pos_offset(self) -> int:
        return self.layout.chunk_realign_offset if self.layout else 0

    @property
    def target_codebook(self) -> int:
        return self.layout.target_codebook

    @property
    def gpu_vocab_start(self) -> int:
        """First id whose embedding row stays on the GPU (control + audio tokens)."""
        lay = self.layout
        if lay is None:
            return 0
        return min(lay.eos_id, lay.text_marker_id, lay.audio_marker_id, lay.dc_base_id)

    def local_to_global(self) -> list[int]:
        """Audio-head output index -> vocabulary id."""
        lay = self.layout
        if lay is None:
            return list(range(self.vocab_size))
        if self.target_codebook == 0:
            ids = [lay.dc_base_id + i for i in range(SEMANTIC_VOCAB_SIZE)]
            return ids + [lay.eos_id, lay.audio_marker_id]
        start = lay.dc_base_id + get_codebook_offset(self.target_codebook)
        return [start + i for i in range(ACOUSTIC_VOCAB_SIZE)]


class QuantPlan:
    """Maps parameter names to quantization specs.

    `rules` are (substring, spec) pairs checked in order; the first match
    wins, `default` applies otherwise. Names are the fused-layer names used
    here: layers.{i}.qkv, .o, .gate_up, .down, head.fc1, head.fc2, embed.
    """

    def __init__(self, default: QuantSpec = QuantSpec(16), rules: list[tuple[str, QuantSpec]] | None = None):
        self.default = default
        self.rules = rules or []

    def spec(self, name: str) -> QuantSpec:
        for pattern, spec in self.rules:
            if pattern in name:
                return spec
        # Embedding rows are only quantized on request (int8 per row).
        return QuantSpec(16) if name == "embed" else self.default

    def describe(self) -> str:
        parts = [f"default={self.default}"] + [f"{p}={s}" for p, s in self.rules]
        return ", ".join(parts)
