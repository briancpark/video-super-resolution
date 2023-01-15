from __future__ import print_function

from math import sqrt

import torch
from torch import nn as nn


class VDSR(nn.Module):
    """
    Accurate Image Super-Resolution Using Very Deep Convolutional Networks

    Jiwon Kim, Jung Kwon Lee, Kyoung Mu Lee

    https://arxiv.org/abs/1511.04587
    """

    def __init__(self, upscale_factor=4) -> None:
        super(VDSR, self).__init__()
        # Upscale factor is determined by test image shape

        # Input layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, (3, 3), (1, 1), (1, 1), bias=False),
            nn.ReLU(True),
        )

        # Features trunk blocks
        trunk = []
        for _ in range(18):
            trunk.append(ConvReLU(64))
        self.trunk = nn.Sequential(*trunk)

        # Output layer
        self.conv2 = nn.Conv2d(64, 1, (3, 3), (1, 1), (1, 1), bias=False)

        # Initialize model weights
        self._initialize_weights()

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(
            self.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4, nesterov=False
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=30, gamma=0.1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.trunk(out)
        out = self.conv2(out)

        out = torch.add(out, identity)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.data.normal_(
                    0.0,
                    sqrt(
                        2
                        / (
                            module.kernel_size[0]
                            * module.kernel_size[1]
                            * module.out_channels
                        )
                    ),
                )


class ConvReLU(nn.Module):
    def __init__(self, channels: int) -> None:
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False)
        self.relu = nn.ReLU(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.relu(out)

        return out
