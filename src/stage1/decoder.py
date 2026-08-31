"""Masked-reconstruction decoder (contract §8).

``tokens [N,192,128] -> reshape [N,128,12,16] -> ConvTranspose2d(128,64,k=4,s=4)
-> residual Conv block -> Conv2d(64,3,k=1) -> [N,3,48,64]``

The output is unconstrained; the loss applies channel-aware handling so the
temporal channel is not forced through a nonnegative activation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionDecoder(nn.Module):
    def __init__(self, dim: int = 128, out_channels: int = 3, grid_h: int = 12, grid_w: int = 16):
        super().__init__()
        self.grid_h, self.grid_w = grid_h, grid_w
        self.convt = nn.ConvTranspose2d(dim, 64, kernel_size=4, stride=4)
        self.res_block = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
        )
        self.head = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens [N, 192, dim] -> reconstruction [N, 3, 48, 64]."""
        n = tokens.shape[0]
        x = tokens.transpose(1, 2).reshape(n, tokens.shape[-1], self.grid_h, self.grid_w)
        x = self.convt(x)  # [N, 64, 48, 64]
        x = x + self.res_block(x)
        return self.head(x)  # [N, 3, 48, 64]
