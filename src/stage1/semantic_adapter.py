"""Trainable DINO semantic adapter (contract §6).

Frozen DINO tokens ``[S, 768, 384]`` are reshaped to the 24x32 patch grid,
aggregated 2x2 (depthwise or average pooling), projected to 128 dims, and
flattened to ``[S, 192, 128]`` on the shared 12x16 grid. Raw DINO tensors
never receive gradients.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedDepthwiseAdapter(nn.Module):
    """DepthwiseConv2d(384,384,k=2,s=2) -> Conv2d(384,128,1) -> GN+GELU."""

    def __init__(self, in_dim: int = 384, d_model: int = 128):
        super().__init__()
        self.depthwise = nn.Conv2d(in_dim, in_dim, kernel_size=2, stride=2, groups=in_dim)
        self.proj = nn.Conv2d(in_dim, d_model, kernel_size=1)
        self.norm = (
            nn.GroupNorm(32, d_model) if d_model % 32 == 0 else nn.LayerNorm(d_model)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [S, 768, 384] -> [S, 384, 24, 32]
        x = tokens.reshape(tokens.shape[0], 384, 24, 32)
        x = self.depthwise(x)  # [S, 384, 12, 16]
        x = self.proj(x)  # [S, d, 12, 16]
        x = F.gelu(self.norm(x))
        return x.flatten(2).transpose(1, 2)  # [S, 192, d]


class AvgPoolAdapter(nn.Module):
    """Fixed 2x2 average pooling plus 1x1 projection (ablation variant)."""

    def __init__(self, in_dim: int = 384, d_model: int = 128):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, d_model, kernel_size=1)
        self.norm = (
            nn.GroupNorm(32, d_model) if d_model % 32 == 0 else nn.LayerNorm(d_model)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens.reshape(tokens.shape[0], 384, 24, 32)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)  # [S, 384, 12, 16]
        x = self.proj(x)
        x = F.gelu(self.norm(x))
        return x.flatten(2).transpose(1, 2)


class SemanticAdapter(nn.Module):
    """Trainable, fold-specific adapter over frozen DINO patch tokens."""

    def __init__(self, kind: str = "learned_depthwise_2x2", in_dim: int = 384, d_model: int = 128):
        super().__init__()
        if kind == "learned_depthwise_2x2":
            self.adapter = LearnedDepthwiseAdapter(in_dim=in_dim, d_model=d_model)
        elif kind == "avgpool_2x2":
            self.adapter = AvgPoolAdapter(in_dim=in_dim, d_model=d_model)
        else:
            raise ValueError(f"unsupported semantic adapter {kind!r}")
        self.kind = kind

    def forward(self, dino_tokens: torch.Tensor) -> torch.Tensor:
        """dino_tokens [S, 768, 384] -> adapted [S, 192, 128]."""
        return self.adapter(dino_tokens)
