"""Inference-only MossFormer2 SR model components.

Adapted from ClearerVoice-Studio's MossFormer2_SR_48K implementation.
Copyright (c) 2024 Alibaba Inc. Licensed under Apache-2.0; see LICENSE.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d
from torch.nn.utils import remove_weight_norm, weight_norm

from .mossformer2 import MossFormer_MaskNet
from .snake import Snake1d


def _get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


def _init_weights(module: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    if "Conv" in module.__class__.__name__:
        module.weight.data.normal_(mean, std)


class _ResBlock1(nn.Module):
    def __init__(self, config, channels: int, kernel_size: int, dilation) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList([
            weight_norm(Conv1d(
                channels,
                channels,
                kernel_size,
                1,
                dilation=value,
                padding=_get_padding(kernel_size, value),
            ))
            for value in dilation
        ])
        self.convs2 = nn.ModuleList([
            weight_norm(Conv1d(
                channels,
                channels,
                kernel_size,
                1,
                dilation=1,
                padding=_get_padding(kernel_size, 1),
            ))
            for _ in dilation
        ])
        self.convs1.apply(_init_weights)
        self.convs2.apply(_init_weights)
        self.convs1_activates = nn.ModuleList([Snake1d(channels) for _ in dilation])
        self.convs2_activates = nn.ModuleList([Snake1d(channels) for _ in dilation])

    def forward(self, inputs):
        output = inputs
        for conv1, conv2, activate1, activate2 in zip(
            self.convs1,
            self.convs2,
            self.convs1_activates,
            self.convs2_activates,
        ):
            residual = conv2(activate2(conv1(activate1(output))))
            output = residual + output
        return output

    def remove_weight_norm(self) -> None:
        for layer in self.convs1:
            remove_weight_norm(layer)
        for layer in self.convs2:
            remove_weight_norm(layer)


class _ResBlock2(nn.Module):
    def __init__(self, config, channels: int, kernel_size: int, dilation) -> None:
        super().__init__()
        self.convs = nn.ModuleList([
            weight_norm(Conv1d(
                channels,
                channels,
                kernel_size,
                1,
                dilation=value,
                padding=_get_padding(kernel_size, value),
            ))
            for value in dilation
        ])
        self.convs.apply(_init_weights)
        self.convs_activates = nn.ModuleList([Snake1d(channels) for _ in dilation])

    def forward(self, inputs):
        output = inputs
        for conv, activate in zip(self.convs, self.convs_activates):
            output = conv(activate(output)) + output
        return output

    def remove_weight_norm(self) -> None:
        for layer in self.convs:
            remove_weight_norm(layer)


class Generator(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.num_kernels = len(config.resblock_kernel_sizes)
        self.num_upsamples = len(config.upsample_rates)
        self.conv_pre = weight_norm(Conv1d(
            80,
            config.upsample_initial_channel,
            7,
            1,
            padding=3,
        ))
        block_class = _ResBlock1 if config.resblock == "1" else _ResBlock2
        self.ups = nn.ModuleList()
        self.snakes = nn.ModuleList()
        for index, (rate, kernel) in enumerate(zip(
            config.upsample_rates,
            config.upsample_kernel_sizes,
        )):
            in_channels = config.upsample_initial_channel // (2**index)
            self.snakes.append(Snake1d(in_channels))
            self.ups.append(weight_norm(ConvTranspose1d(
                in_channels,
                in_channels // 2,
                kernel,
                rate,
                padding=(kernel - rate) // 2,
            )))

        self.resblocks = nn.ModuleList()
        for index in range(len(self.ups)):
            channels = config.upsample_initial_channel // (2 ** (index + 1))
            for kernel, dilation in zip(
                config.resblock_kernel_sizes,
                config.resblock_dilation_sizes,
            ):
                self.resblocks.append(block_class(config, channels, kernel, dilation))

        self.snake_post = Snake1d(channels)
        self.conv_post = weight_norm(Conv1d(channels, 1, 7, 1, padding=3))
        self.ups.apply(_init_weights)
        self.conv_post.apply(_init_weights)

    def forward(self, inputs):
        output = self.conv_pre(inputs)
        for index in range(self.num_upsamples):
            output = self.ups[index](self.snakes[index](output))
            residual_sum = None
            for kernel_index in range(self.num_kernels):
                residual = self.resblocks[
                    index * self.num_kernels + kernel_index
                ](output)
                residual_sum = residual if residual_sum is None else residual_sum + residual
            output = residual_sum / self.num_kernels
        output = self.conv_post(self.snake_post(output))
        return torch.tanh(output)

    def remove_weight_norm(self) -> None:
        for layer in self.ups:
            remove_weight_norm(layer)
        for layer in self.resblocks:
            layer.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class Mossformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mossformer = MossFormer_MaskNet(
            in_channels=80,
            out_channels=512,
            out_channels_final=80,
        )

    def forward(self, inputs):
        return self.mossformer(inputs)
