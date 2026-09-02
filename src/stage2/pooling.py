"""Stage-2 query attention pooler (guide 05 §5, contracts §14.2).

A new pooler trained by Stage 2; the Stage-1 trial pooling module is never
reused. Every one of the 192 spatial patches participates — a patch with zero
gaze density is still a real patch, so there is no missing-patch mask inside
an observed heatmap.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QueryAttentionPooler(nn.Module):
    """``LayerNorm(128) -> Linear(128,64) -> GELU -> Linear(64,1) ->
    masked spatial softmax over 192 tokens -> weighted sum``.

    Returns ``(patch attention [N,192], q0 [N,128])``.
    """

    def __init__(self, dim: int = 128, hidden: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, hidden)
        self.score = nn.Linear(hidden, 1)

    def forward(
        self, tokens: torch.Tensor, token_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``tokens [N,192,D] -> (patch_attention [N,192], q0 [N,D])``.

        ``token_mask`` exists only for defensive generality; Stage 2 always
        calls it with ``None`` because observed heatmaps have no missing
        patches.
        """
        if tokens.dim() != 3:
            raise ValueError(f"pooler expects [N,192,D], got {tuple(tokens.shape)}")
        logits = self.score(torch.nn.functional.gelu(self.proj(self.norm(tokens)))).squeeze(-1)
        if token_mask is not None:
            if token_mask.shape != logits.shape or token_mask.dtype != torch.bool:
                raise ValueError("token_mask must be a bool tensor of shape [N,192]")
            logits = logits.masked_fill(~token_mask, float("-inf"))
        weights = logits.softmax(dim=-1)  # [N,192]
        q0 = (weights.unsqueeze(-1) * tokens).sum(dim=1)  # [N,D]
        return weights, q0


class MeanQueryPooler(nn.Module):
    """`mean_query_pooling` ablation: uniform attention weights (1/192) and the
    masked mean over patches. No learned parameters — the declared factor."""

    def forward(
        self, tokens: torch.Tensor, token_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tokens.dim() != 3:
            raise ValueError(f"pooler expects [N,192,D], got {tuple(tokens.shape)}")
        n, t = tokens.shape[:2]
        weights = torch.full(
            (n, t), 1.0 / t, dtype=tokens.dtype, device=tokens.device
        )
        if token_mask is not None:
            weights = weights * token_mask
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        q0 = (weights.unsqueeze(-1) * tokens).sum(dim=1)
        return weights, q0


def build_query_pooler(kind: str, dim: int = 128) -> nn.Module:
    if kind == "attention":
        return QueryAttentionPooler(dim)
    if kind == "mean":
        return MeanQueryPooler()
    raise ValueError(f"unsupported query pooling {kind!r}")
