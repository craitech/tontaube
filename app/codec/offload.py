"""Module weights kept in pinned host memory and moved to the device only while used."""
from __future__ import annotations

import torch
import torch.nn as nn


class HostResident:
    """Keeps a module's parameters in pinned host memory between uses."""

    def __init__(self, module: nn.Module, device: torch.device):
        self.module = module
        self.device = device
        self.host = {}
        for name, p in module.named_parameters():
            self.host[name] = p.data.detach().to("cpu").pin_memory()
            p.data = self.host[name]
        for name, b in module.named_buffers():
            # Buffers are small; keep them on the device.
            b.data = b.data.to(device)

    def __enter__(self):
        for name, p in self.module.named_parameters():
            p.data = self.host[name].to(self.device, non_blocking=True)
        return self.module

    def __exit__(self, *exc):
        for name, p in self.module.named_parameters():
            p.data = self.host[name]
        return False
