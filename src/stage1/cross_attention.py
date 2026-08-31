"""Pre-normalized multi-head cross-attention (contract §7.1/§7.3).

Each module has independent query, key, value, output-projection, and
normalization parameters. Attention weights are returned only on request.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, dim: int = 128, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        key_value: torch.Tensor,
        return_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """queries [N, 192, dim], key_value [N, 192, dim] -> ([N, 192, dim], weights)."""
        n = queries.shape[0]
        q = self.q_proj(self.norm_q(queries)).reshape(n, -1, self.heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(self.norm_kv(key_value)).reshape(n, -1, self.heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(self.norm_kv(key_value)).reshape(n, -1, self.heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [N, H, 192, 192]
        attn = attn.softmax(dim=-1)
        weights = attn if return_weights else None
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)  # [N, H, 192, hd]
        out = out.transpose(1, 2).reshape(n, -1, self.dim)
        out = self.proj_drop(self.out_proj(out))
        return out, weights
