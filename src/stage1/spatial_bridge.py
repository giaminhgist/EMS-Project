"""Residual spatial NN bridge (contract §7.2).

Single sequential path:

``LayerNorm -> Linear 128->256 + GELU -> reshape [N,256,12,16] ->
DepthwiseConv2d 3x3 pad 1 -> GroupNorm + GELU -> flatten -> Linear 256->128``

``token_mlp_ffn`` drops the depthwise conv + GroupNorm (matched-width MLP);
``identity`` is handled by the fusion module (named ablation).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialBridge(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        expansion_ratio: float = 2.0,
        kernel_size: int = 3,
        grid_h: int = 12,
        grid_w: int = 16,
        kind: str = "residual_dwconv_ffn",
    ):
        super().__init__()
        self.dim = dim
        self.grid_h, self.grid_w = grid_h, grid_w
        self.kind = kind
        if kind not in ("residual_dwconv_ffn", "token_mlp_ffn"):
            raise ValueError(f"unsupported bridge kind {kind!r}")
        self.norm = nn.LayerNorm(dim)
        hidden = int(round(dim * expansion_ratio))
        self.expand = nn.Sequential(nn.Linear(dim, hidden), nn.GELU())
        if kind == "residual_dwconv_ffn":
            self.dwconv = nn.Conv2d(
                hidden, hidden, kernel_size, padding=kernel_size // 2, groups=hidden
            )
            self.gn = nn.GroupNorm(32, hidden) if hidden % 32 == 0 else nn.LayerNorm(hidden)
        else:
            self.dwconv = None
        self.project = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [N, 192, dim] -> bridge output [N, 192, dim] (before gating)."""
        z = self.norm(x)
        h = self.expand(z)  # [N, 192, hidden]
        if self.dwconv is not None:
            n = x.shape[0]
            h = h.transpose(1, 2).reshape(-1, h.shape[-1], self.grid_h, self.grid_w)
            h = self.dwconv(h)
            h = F.gelu(self.gn(h))
            h = h.reshape(n, h.shape[1], -1).transpose(1, 2)  # [N, 192, hidden]
        return self.project(h)  # [N, 192, dim]
