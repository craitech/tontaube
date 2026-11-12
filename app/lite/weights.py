"""Reading weights: checkpoints, pre-quantized bundles, and one loader for both engines.

A quantized bundle is a directory with cb0.safetensors ... cb3.safetensors.
Each file holds only the linear layers (fused names: layers.{i}.qkv, .o,
.gate_up, .down, head.fc1, head.fc2):

  <name>.codes / <name>.scales / <name>.mins   packed int4/int8 layer
  <name>.weight                                bf16 layer
  <name>.bias                                  optional bias (bf16)

and string metadata: format, plan, method, source (weights_fingerprint() of the
checkpoint it was quantized from) and "<name>" -> "bits,group,n,k".
Norms and embeddings are read from the codebook checkpoint (a standalone release ships a
reduced one, app/lite/quantize/standalone.py).

load_codebook_weights() turns a checkpoint (plus an optional bundle) into the
tensors of one codebook model; the GPU and the CPU model both build on it.
"""
from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path

import torch

from app.lite.config import ModelConfig, QuantPlan
from app.lite.quant import QLinear, QuantSpec
from app.logging_utils import get_logger

logger = get_logger("lite")


class LazyWeights:
    """dict.pop()-style access to a safetensors file, one tensor at a time.

    With mmap=True, bf16/fp16/fp32 tensors are copy-on-write views of the file:
    pages become resident only when read (the large text-embedding tables are
    mostly never touched) and stay reclaimable page cache.
    """

    _MMAP_DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}

    def __init__(self, path: str | Path, mmap: bool = False):
        from safetensors import safe_open

        self._path = str(path)
        self._file = safe_open(self._path, framework="pt", device="cpu")
        self._keys = set(self._file.keys())
        self._header = None
        if mmap:
            with open(self._path, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                self._header = json.loads(fh.read(n))
            self._data_start = 8 + n

    def pop(self, key: str) -> torch.Tensor:
        self._keys.remove(key)
        info = self._header.get(key) if self._header is not None else None
        if info is None or info["dtype"] not in self._MMAP_DTYPES:
            return self._file.get_tensor(key)
        import numpy as np

        dtype = self._MMAP_DTYPES[info["dtype"]]
        begin, end = info["data_offsets"]
        if end == begin:
            return torch.empty(info["shape"], dtype=dtype)
        raw = np.memmap(self._path, dtype=np.uint8, mode="c", offset=self._data_start + begin,
                        shape=(end - begin,))
        return torch.from_numpy(raw).view(dtype).view(info["shape"])

    def keys(self):
        return set(self._keys)


FORMAT = "tontaube-lite-linear-v1"
# Standalone bundle checkpoints (app/lite/quantize/standalone.py) store the embedding split and trimmed.
TRIMMED_TEXT_ROWS = "model.embed_tokens.text_rows"
TRIMMED_TEXT_WEIGHT = "model.embed_tokens.text_weight"
TRIMMED_AUDIO_WEIGHT = "model.embed_tokens.audio_weight"


def save_linears(path: str | Path, linears: dict[str, QLinear], meta: dict[str, str]) -> None:
    from safetensors.torch import save_file

    tensors: dict[str, torch.Tensor] = {}
    metadata = {"format": FORMAT, **{k: str(v) for k, v in meta.items()}}
    for name, q in linears.items():
        metadata[name] = f"{q.spec.bits},{q.spec.group},{q.n_out},{q.k_in}"
        if q.weight is not None:
            tensors[f"{name}.weight"] = q.weight.detach().to(torch.bfloat16).contiguous().cpu()
        else:
            tensors[f"{name}.codes"] = q.codes.contiguous().cpu()
            tensors[f"{name}.scales"] = q.scales.contiguous().cpu()
            tensors[f"{name}.mins"] = q.mins.contiguous().cpu()
        if q.bias is not None:
            tensors[f"{name}.bias"] = q.bias.detach().to(torch.bfloat16).contiguous().cpu()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata=metadata)


def load_linears(path: str | Path, device, dtype=torch.bfloat16) -> tuple[dict[str, QLinear], dict]:
    from safetensors import safe_open

    out: dict[str, QLinear] = {}
    with safe_open(str(path), framework="pt", device="cpu") as f:
        meta = f.metadata() or {}
        if meta.get("format") != FORMAT:
            raise ValueError(f"{path} is not a {FORMAT} file")
        keys = set(f.keys())
        for name, desc in meta.items():
            if name in ("format", "plan", "method", "source", "calibration"):
                continue
            bits, group, n, k = (int(v) for v in desc.split(","))
            spec = QuantSpec(bits, group)
            bias = f.get_tensor(f"{name}.bias").to(device, dtype) if f"{name}.bias" in keys else None
            if bits == 16:
                w = f.get_tensor(f"{name}.weight").to(device, dtype)
                out[name] = QLinear(n, k, spec, bias, weight=w)
            else:
                out[name] = QLinear.from_quantized(
                    n, k, spec, bias,
                    f.get_tensor(f"{name}.codes").to(device),
                    f.get_tensor(f"{name}.scales").to(device),
                    f.get_tensor(f"{name}.mins").to(device),
                )
    return out, meta


def weights_fingerprint(path: str | Path) -> str:
    """Identity of a safetensors checkpoint without reading all of it.

    sha256 over the header (names, dtypes, shapes) and the first, middle and
    last 4 KiB of every tensor: another release or fine-tune of the same
    architecture changes the sampled bytes even where the header is identical.
    A guard against loading a bundle onto the wrong weights, not a checksum.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        raw = fh.read(n)
        h.update(raw)
        tensors = [v for k, v in json.loads(raw).items() if k != "__metadata__"]
        for info in sorted(tensors, key=lambda v: v["data_offsets"][0]):
            begin, end = info["data_offsets"]
            for at in sorted({begin, max(begin, (begin + end) // 2 - 2048), max(begin, end - 4096)}):
                fh.seek(8 + n + at)
                h.update(fh.read(min(4096, end - at)))
    return h.hexdigest()[:32]


def load_bundle(bundle: str | Path, cb: int, model_path: str | Path, device) -> tuple[dict[str, QLinear], str]:
    """Codebook cb's pre-quantized linears and plan from a bundle directory.

    Refuses a bundle quantized from other weights than model_path (e.g. a v1.0
    bundle on v1.1 models); bundles without a recorded source only warn.
    """
    from safetensors import safe_open

    path = Path(bundle) / f"cb{cb}.safetensors"
    with safe_open(str(path), framework="pt", device="cpu") as f:
        source = (f.metadata() or {}).get("source")
    weights = Path(model_path) / "model.safetensors"
    if source is None:
        logger.warning("Lite bundle %s records no source fingerprint; make sure it was quantized from %s",
                       path, weights)
    elif source != weights_fingerprint(weights):
        raise ValueError(f"lite bundle {path} was quantized from other weights than {weights}; "
                         "rebuild it for these models with `tontaube quantize`")
    linears, meta = load_linears(path, device)
    return linears, meta.get("plan", "bf16")


@dataclass
class LayerWeights:
    """One decoder layer: norms and the fused linear layers."""

    in_norm: torch.Tensor
    post_norm: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    qkv: QLinear
    o: QLinear
    gate_up: QLinear
    down: QLinear


@dataclass
class CodebookWeights:
    """Everything a codebook model holds besides its KV cache."""

    text_embed: torch.Tensor  # rows below cfg.gpu_vocab_start (compact when text_index is set), host memory
    text_index: torch.Tensor | None  # token id -> row of text_embed, -1 where absent
    audio_embed: torch.Tensor | None  # rows from cfg.gpu_vocab_start on, on the device ...
    audio_codes: torch.Tensor | None  # ... or as per-row symmetric int8
    audio_scale: torch.Tensor | None
    layers: list[LayerWeights]
    norm: torch.Tensor
    fc1: QLinear
    fc2: QLinear


def load_linear(w, plan: QuantPlan, name: str, weight_keys: list[str], bias_key: str | None, device, dtype,
                prequant: dict[str, QLinear] | None = None) -> QLinear:
    """A fused linear layer from a bundle, or from checkpoint tensors quantized per the plan."""
    if prequant and name in prequant:
        return prequant[name]
    weight = torch.cat([w.pop(k) for k in weight_keys], 0)
    bias = w.pop(bias_key) if bias_key else None
    return QLinear.from_weight(weight.to(device), bias, plan.spec(name), device, dtype)


def load_layers(w, cfg: ModelConfig, plan: QuantPlan, device, dtype,
                prequant: dict[str, QLinear] | None = None) -> tuple[list[LayerWeights], torch.Tensor]:
    """The decoder layers and the final norm."""
    layers = []
    for i in range(cfg.n_layers):
        p = f"model.layers.{i}."
        lin = lambda name, keys, bias=None: load_linear(w, plan, f"layers.{i}.{name}", keys, bias,  # noqa: E731
                                                        device, dtype, prequant)
        layers.append(LayerWeights(
            in_norm=w.pop(p + "input_layernorm.weight").to(device, dtype),
            post_norm=w.pop(p + "post_attention_layernorm.weight").to(device, dtype),
            q_norm=w.pop(p + "self_attn.q_norm.weight").to(device, dtype),
            k_norm=w.pop(p + "self_attn.k_norm.weight").to(device, dtype),
            qkv=lin("qkv", [p + f"self_attn.{n}_proj.weight" for n in "qkv"]),
            o=lin("o", [p + "self_attn.o_proj.weight"]),
            gate_up=lin("gate_up", [p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"]),
            down=lin("down", [p + "mlp.down_proj.weight"]),
        ))
    return layers, w.pop("model.norm.weight").to(device, dtype)


def load_codebook_weights(w, cfg: ModelConfig, plan: QuantPlan, device, dtype,
                          prequant: dict[str, QLinear] | None = None,
                          text_rows: list[int] | None = None) -> CodebookWeights:
    """All weights of one codebook model.

    text_rows (app/model/text_vocab.py) keeps only the text-embedding rows the
    input scheme can produce; without it the whole text range is kept. A standalone
    bundle checkpoint stores those rows already trimmed (TRIMMED_*).
    """
    start = cfg.gpu_vocab_start
    text_index = None
    if TRIMMED_TEXT_ROWS in w.keys():
        if not prequant:
            raise ValueError("these are the reduced model files of a standalone int4 release; plan strings and bf16 "
                             "need the full-precision models (TontaubeAI/TontaubeV1.1)")
        if text_rows is None:
            raise ValueError("a checkpoint with a trimmed text embedding needs text_rows")
        stored = {r: i for i, r in enumerate(w.pop(TRIMMED_TEXT_ROWS).tolist())}
        rows = [r for r in text_rows if r < start]
        missing = [r for r in rows if r not in stored]
        if missing:
            raise ValueError(f"the checkpoint's trimmed text embedding lacks token ids {missing[:5]}")
        text_index = torch.full((start,), -1, dtype=torch.long)
        text_index[torch.tensor(rows, dtype=torch.long)] = torch.arange(len(rows))
        text_embed = w.pop(TRIMMED_TEXT_WEIGHT)[[stored[r] for r in rows]].to(dtype).contiguous()
        audio_rows = w.pop(TRIMMED_AUDIO_WEIGHT).to(dtype)
    else:
        emb = w.pop("model.embed_tokens.weight")
        if text_rows is not None:
            keep = torch.tensor([r for r in text_rows if r < start], dtype=torch.long)
            text_index = torch.full((start,), -1, dtype=torch.long)
            text_index[keep] = torch.arange(len(keep))
            text_embed = emb[keep].to(dtype).contiguous()
        else:
            text_embed = emb[:start].to(dtype).contiguous()
        audio_rows = emb[start:].to(dtype)
        del emb
    audio_embed = audio_codes = audio_scale = None
    espec = plan.spec("embed")
    if espec.bits == 16:
        audio_embed = audio_rows.to(device).contiguous()
    elif espec.bits == 8:  # per-row symmetric int8, dequantized on lookup
        g = audio_rows.float()
        scale = (g.abs().amax(-1, keepdim=True) / 127).clamp(min=1e-8)
        audio_codes = torch.round(g / scale).clamp(-127, 127).to(torch.int8).to(device)
        audio_scale = scale.to(device, torch.float32)
    else:
        raise ValueError("embedding quantization supports int8 only")
    layers, norm = load_layers(w, cfg, plan, device, dtype, prequant)
    fc1 = load_linear(w, plan, "head.fc1", ["audio_head.fc1.weight"], "audio_head.fc1.bias", device, dtype, prequant)
    fc2 = load_linear(w, plan, "head.fc2", ["audio_head.fc2.weight"], "audio_head.fc2.bias", device, dtype, prequant)
    return CodebookWeights(text_embed, text_index, audio_embed, audio_codes, audio_scale, layers, norm, fc1, fc2)


def text_embedding_rows(table: torch.Tensor, index: torch.Tensor | None, ids: torch.Tensor) -> torch.Tensor:
    """Rows of a text embedding table for token ids (CPU int64), dense or compact + index."""
    if index is None:
        return table[ids]
    idx = index[ids]
    if bool((idx < 0).any()):
        bad = sorted(set(ids[idx < 0].tolist()))[:8]
        raise ValueError(f"text token ids {bad} are outside the model's text vocabulary")
    return table[idx]
