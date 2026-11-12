"""A lite bundle only loads onto the weights it was quantized from."""
import logging
import shutil

import pytest
import torch
from safetensors.torch import save_file

from app.lite.quant import QLinear, QuantSpec
from app.lite.weights import load_bundle, save_linears, weights_fingerprint


def _checkpoint(path, scale=1.0, rows=64):
    torch.manual_seed(0)
    path.mkdir(parents=True, exist_ok=True)
    save_file({"model.embed_tokens.weight": torch.randn(rows, 32).to(torch.bfloat16) * scale,
               "model.norm.weight": torch.ones(32, dtype=torch.bfloat16)}, str(path / "model.safetensors"))
    return path


def _bundle(path, source):
    q = QLinear.from_weight(torch.randn(16, 128), None, QuantSpec(8, 128), "cpu")
    meta = {"plan": "int8g128", "method": "gptq"}
    if source is not None:
        meta["source"] = source
    save_linears(path / "cb0.safetensors", {"layers.0.o": q}, meta)
    return path


def test_fingerprint_tells_releases_apart(tmp_path):
    a = _checkpoint(tmp_path / "a")
    shutil.copytree(a, tmp_path / "copy")
    assert weights_fingerprint(a / "model.safetensors") == weights_fingerprint(tmp_path / "copy" / "model.safetensors")
    # same header, different values (a fine-tune or LoRA merge)
    assert weights_fingerprint(a / "model.safetensors") != weights_fingerprint(
        _checkpoint(tmp_path / "b", scale=1.01) / "model.safetensors")
    # a grown vocabulary
    assert weights_fingerprint(a / "model.safetensors") != weights_fingerprint(
        _checkpoint(tmp_path / "c", rows=65) / "model.safetensors")


def test_load_bundle_checks_the_source(tmp_path):
    v10, v11 = _checkpoint(tmp_path / "v10"), _checkpoint(tmp_path / "v11", scale=1.01)
    bundle = _bundle(tmp_path / "bundle", weights_fingerprint(v10 / "model.safetensors"))
    linears, plan = load_bundle(bundle, 0, v10, torch.device("cpu"))
    assert plan == "int8g128" and set(linears) == {"layers.0.o"}
    with pytest.raises(ValueError, match="quantized from other weights"):
        load_bundle(bundle, 0, v11, torch.device("cpu"))
    # bundles from before the fingerprint load with a warning
    old = _bundle(tmp_path / "old", None)
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    logger = logging.getLogger("tontaube.lite")
    logger.addHandler(handler)
    try:
        load_bundle(old, 0, v11, torch.device("cpu"))
    finally:
        logger.removeHandler(handler)
    assert any("records no source fingerprint" in r.getMessage() for r in records)


def _tiny_models(root):
    """Four tiny codebook checkpoints (cb0 ... cb3) shaped like TontaubeV1."""
    import json

    hid, inter, heads, kv, hd, head, dc = 128, 256, 2, 1, 64, 128, 200
    g = torch.Generator().manual_seed(0)
    r = lambda *shape: (torch.randn(*shape, generator=g) * 0.1).to(torch.bfloat16)  # noqa: E731
    for cb in range(4):
        d = root / f"cb{cb}"
        d.mkdir(parents=True)
        (d / "config.json").write_text(json.dumps({
            "hidden_size": hid, "num_hidden_layers": 1, "num_attention_heads": heads, "num_key_value_heads": kv,
            "head_dim": hd, "intermediate_size": inter, "rms_norm_eps": 1e-6, "rope_theta": 1e6,
            "vocab_size": dc + 16384, "max_position_embeddings": 4096, "audio_head_hidden": head,
            "target_codebook": cb, "n_codebooks": 4, "audio_token_start_id": dc, "pad_token_id": 196,
            "eos_token_id": 197, "text_marker_token_id": 198, "audio_marker_token_id": 199, "chunk_realign_offset": 25,
        }))
        (d / "tokenizer.json").write_text("{}")
        p = "model.layers.0."
        save_file({
            "model.embed_tokens.weight": r(dc + 16384, hid), "model.norm.weight": 1 + r(hid),
            p + "input_layernorm.weight": 1 + r(hid), p + "post_attention_layernorm.weight": 1 + r(hid),
            p + "self_attn.q_norm.weight": 1 + r(hd), p + "self_attn.k_norm.weight": 1 + r(hd),
            p + "self_attn.q_proj.weight": r(heads * hd, hid), p + "self_attn.k_proj.weight": r(kv * hd, hid),
            p + "self_attn.v_proj.weight": r(kv * hd, hid), p + "self_attn.o_proj.weight": r(hid, heads * hd),
            p + "mlp.gate_proj.weight": r(inter, hid), p + "mlp.up_proj.weight": r(inter, hid),
            p + "mlp.down_proj.weight": r(hid, inter),
            "audio_head.fc1.weight": r(head, hid), "audio_head.fc1.bias": r(head),
            "audio_head.fc2.weight": r(16386, head), "audio_head.fc2.bias": r(16386),
        }, str(d / "model.safetensors"))


def test_standalone_release_loads_like_full_models_plus_bundle(tmp_path, monkeypatch):
    import json

    from app.lite.config import ModelConfig
    from app.lite.plans import parse_plan
    from app.lite.quantize import standalone
    from app.lite.weights import LazyWeights, load_codebook_weights

    plan, rows = parse_plan("int8g128,embed=int8"), [5, 17, 42, 196, 197, 198, 199]
    _tiny_models(tmp_path / "full")
    bundle = tmp_path / "bundle"
    for cb in range(4):
        src = tmp_path / "full" / f"cb{cb}"
        w = LazyWeights(src / "model.safetensors")
        linears = {}
        for name, keys, bias in (
            ("layers.0.qkv", [f"model.layers.0.self_attn.{n}_proj.weight" for n in "qkv"], None),
            ("layers.0.o", ["model.layers.0.self_attn.o_proj.weight"], None),
            ("layers.0.gate_up", ["model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.up_proj.weight"], None),
            ("layers.0.down", ["model.layers.0.mlp.down_proj.weight"], None),
            ("head.fc1", ["audio_head.fc1.weight"], "audio_head.fc1.bias"),
            ("head.fc2", ["audio_head.fc2.weight"], "audio_head.fc2.bias"),
        ):
            weight = torch.cat([w.pop(k) for k in keys]).float()
            linears[name] = QLinear.from_weight(weight, w.pop(bias) if bias else None, QuantSpec(8, 128), "cpu")
        save_linears(bundle / f"cb{cb}.safetensors", linears,
                     {"plan": "int8g128", "method": "rtn", "source": weights_fingerprint(src / "model.safetensors")})
    (bundle / "bundle.json").write_text(json.dumps({"format": "tontaube-lite-bundle-v1", "source_models": ["/x"]}))
    monkeypatch.setattr(standalone, "load_tokenizer", lambda path: None)
    monkeypatch.setattr(standalone, "text_token_rows", lambda tokenizer: rows)
    standalone.export_standalone(tmp_path / "full", bundle, tmp_path / "out")

    out = tmp_path / "out"
    assert (out / "cb0" / "tokenizer.json").is_file() and json.loads((out / "int4" / "bundle.json").read_text())[
        "source_models"] == ["cb0", "cb1", "cb2", "cb3"]
    for cb in range(4):
        cfg = ModelConfig.from_dir(tmp_path / "full" / f"cb{cb}", cb)
        ref_pre, _ = load_bundle(bundle, cb, tmp_path / "full" / f"cb{cb}", torch.device("cpu"))
        new_pre, _ = load_bundle(out / "int4", cb, out / f"cb{cb}", torch.device("cpu"))  # fingerprint matches
        ref = load_codebook_weights(LazyWeights(tmp_path / "full" / f"cb{cb}" / "model.safetensors"), cfg, plan,
                                    torch.device("cpu"), torch.bfloat16, ref_pre, rows)
        new = load_codebook_weights(LazyWeights(out / f"cb{cb}" / "model.safetensors"), cfg, plan,
                                    torch.device("cpu"), torch.bfloat16, new_pre, rows)
        for field in ("text_embed", "text_index", "audio_codes", "audio_scale", "norm"):
            assert torch.equal(getattr(ref, field), getattr(new, field)), field
        assert torch.equal(ref.layers[0].in_norm, new.layers[0].in_norm)
        assert torch.equal(ref.layers[0].qkv.codes, new.layers[0].qkv.codes)
        assert torch.equal(ref.fc2.bias, new.fc2.bias)
        kept = LazyWeights(out / f"cb{cb}" / "model.safetensors").keys()
        assert not any(k.endswith("_proj.weight") or k.startswith("audio_head.") for k in kept)
    cfg0 = ModelConfig.from_dir(out / "cb0", 0)
    pre0, _ = load_bundle(out / "int4", 0, out / "cb0", torch.device("cpu"))
    with pytest.raises(ValueError, match="lacks token ids"):
        load_codebook_weights(LazyWeights(out / "cb0" / "model.safetensors"), cfg0,
                              plan, torch.device("cpu"), torch.bfloat16, pre0, rows + [7])
    with pytest.raises(ValueError, match="need the full-precision models"):  # a plan string without the bundle
        load_codebook_weights(LazyWeights(out / "cb0" / "model.safetensors"), cfg0,
                              plan, torch.device("cpu"), torch.bfloat16, None, rows)


def test_lite_quant_auto_finds_the_bundle(tmp_path):
    from app.lite import resolve_lite_quant

    models = [str(tmp_path / f"cb{cb}") for cb in range(4)]
    assert resolve_lite_quant("auto", models) == "bf16"
    (tmp_path / "int4").mkdir()
    (tmp_path / "int4" / "bundle.json").write_text("{}")
    assert resolve_lite_quant("auto", models) == str(tmp_path / "int4")
    assert resolve_lite_quant("int8g128", models) == "int8g128"
