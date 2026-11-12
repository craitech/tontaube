"""W2V-BERT 2.0 (DualCodec's semantic encoder): fewer layers and streamed weights.

W2V-BERT is only needed to encode voice references. DualCodec reads the hidden
state after layer 17 (`output_idx`), so later encoder layers are dropped
(exact), and on the GPU the remaining layers can stream from pinned host
memory one layer at a time while a reference is being encoded.
"""
from __future__ import annotations

import torch.nn as nn

from app.codec.offload import HostResident


def truncate_semantic_model(dc_inference) -> int:
    """Drop W2V-BERT encoder layers after DualCodec's output layer. Returns #dropped."""
    model = dc_inference.semantic_cfg["semantic_model"]
    keep = int(dc_inference.semantic_cfg["output_idx"])
    layers = model.encoder.layers
    dropped = len(layers) - keep
    if dropped > 0:
        model.encoder.layers = nn.ModuleList(list(layers)[:keep])
        model.config.num_hidden_layers = keep
    return max(0, dropped)


def offload_semantic_model(dc_inference) -> None:
    """Stream W2V-BERT's encoder layers to the GPU one at a time.

    Each layer's weights live in pinned host memory and are copied in by a
    forward pre-hook and dropped by a forward hook, so an encode holds one
    layer's weights instead of all of them. The small remaining modules
    (feature projection, position embeddings) stay on the GPU.
    """
    sm = dc_inference.semantic_cfg["semantic_model"]
    device = next(dc_inference.model.parameters()).device
    residents = []
    for layer in sm.encoder.layers:
        resident = HostResident(layer, device)

        # Hooks must return None, or their result replaces the inputs/outputs.
        def load(_module, _args, r=resident):
            r.__enter__()

        def unload(_module, _args, _output, r=resident):
            r.__exit__(None, None, None)

        layer.register_forward_pre_hook(load)
        layer.register_forward_hook(unload)
        residents.append(resident)
    dc_inference._tontaube_host_resident = residents
