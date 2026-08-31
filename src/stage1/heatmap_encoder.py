"""Trainable heatmap patch encoder (contract §5).

``Conv2d(3,128,k=4,s=4) -> GroupNorm+GELU -> flatten -> mask-token
replacement -> fixed 2D sine/cosine positions -> two lightweight residual
blocks`` producing ``[N, 192, 128]`` tokens on the shared 12x16 grid.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _fixed_2d_sincos_positions(n_tokens: int, dim: int, grid_h: int, grid_w: int) -> torch.Tensor:
    """Fixed 2-D sine/cosine positional encoding for a row-major grid."""
    assert dim % 4 == 0, "positional dim must be divisible by 4"
    rows = torch.arange(grid_h, dtype=torch.float32)
    cols = torch.arange(grid_w, dtype=torch.float32)
    quarter = dim // 4
    freqs = torch.arange(quarter, dtype=torch.float32)
    omega = 1.0 / (10000.0 ** (freqs / max(quarter - 1, 1)))
    # row encoding: sin/cos over rows for half the dims
    row_sin = torch.sin(rows[:, None] * omega[None, :])  # [H, q]
    row_cos = torch.cos(rows[:, None] * omega[None, :])
    col_sin = torch.sin(cols[:, None] * omega[None, :])  # [W, q]
    col_cos = torch.cos(cols[:, None] * omega[None, :])
    grid = []
    for i in range(grid_h):
        for j in range(grid_w):
            grid.append(
                torch.cat([row_sin[i], row_cos[i], col_sin[j], col_cos[j]], dim=0)
            )
    return torch.stack(grid, dim=0)  # [H*W, dim]


class ResidualBlock(nn.Module):
    """Pre-normalized MLP plus depthwise 3x3 convolution on the 12x16 grid."""

    def __init__(self, dim: int, grid_h: int = 12, grid_w: int = 16, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.drop = nn.Dropout(dropout)
        self.grid_h, self.grid_w = grid_h, grid_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, 192, dim]
        z = self.norm(x)
        token_branch = self.mlp(z)
        spatial_branch = self.dwconv(
            z.transpose(1, 2).reshape(-1, x.shape[-1], self.grid_h, self.grid_w)
        )
        spatial_branch = (
            spatial_branch.reshape(x.shape[0], x.shape[-1], -1).transpose(1, 2)
        )
        return x + self.drop(token_branch) + self.drop(spatial_branch)


class HeatmapPatchEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        d_model: int = 128,
        patch_size: int = 4,
        grid_h: int = 12,
        grid_w: int = 16,
        n_residual_blocks: int = 2,
        dropout: float = 0.0,
        positional_encoding: str = "fixed_2d_sincos",
    ):
        super().__init__()
        assert patch_size * grid_h == 48 and patch_size * grid_w == 64
        assert positional_encoding == "fixed_2d_sincos"
        self.grid_h, self.grid_w = grid_h, grid_w
        self.patch_embed = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.GroupNorm(num_groups=32, num_channels=d_model) if d_model % 32 == 0 else nn.LayerNorm(d_model)
        self.mask_token = nn.Parameter(torch.zeros(d_model))
        self.register_buffer(
            "pos_embed", _fixed_2d_sincos_positions(grid_h * grid_w, d_model, grid_h, grid_w)
        )
        self.residual_blocks = nn.ModuleList(
            ResidualBlock(d_model, grid_h, grid_w, dropout=dropout) for _ in range(n_residual_blocks)
        )

    def forward(self, heatmaps: torch.Tensor, token_mask: torch.Tensor | None) -> torch.Tensor:
        """heatmaps [N, 3, 48, 64]; token_mask [N, 192] bool (None = unmasked)."""
        x = self.patch_embed(heatmaps)  # [N, d, 12, 16]
        x = F.gelu(self.norm(x))
        x = x.flatten(2).transpose(1, 2)  # [N, 192, d]
        if token_mask is not None:
            mask_tokens = self.mask_token.expand(x.shape[0], 1, -1)
            x = torch.where(token_mask.unsqueeze(-1), mask_tokens, x)
        x = x + self.pos_embed
        for block in self.residual_blocks:
            x = block(x)
        return x
