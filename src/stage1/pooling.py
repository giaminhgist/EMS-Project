"""Trial pooling (contract §9): attention pooling (default) and mean (ablation)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """a_j = softmax(w2^T tanh(W1 z_j)); z = sum_j a_j z_j."""

    def __init__(self, dim: int = 128):
        super().__init__()
        self.W1 = nn.Linear(dim, dim)
        self.w2 = nn.Linear(dim, 1)

    def forward(
        self, tokens: torch.Tensor, return_weights: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """tokens [N, 192, dim] -> (embedding [N, dim], weights [N, 192] | None)."""
        scores = self.w2(torch.tanh(self.W1(tokens)))  # [N, 192, 1]
        weights = scores.softmax(dim=1)
        embedding = (weights * tokens).sum(dim=1)  # [N, dim]
        return embedding, (weights.squeeze(-1) if return_weights else None)


class MeanPooling(nn.Module):
    def forward(
        self, tokens: torch.Tensor, return_weights: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return tokens.mean(dim=1), None


def build_pooling(kind: str, dim: int) -> nn.Module:
    if kind == "attention":
        return AttentionPooling(dim)
    if kind == "mean":
        return MeanPooling()
    raise ValueError(f"unsupported pooling {kind!r}")
