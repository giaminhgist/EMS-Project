"""Serial Attention -> spatial NN bridge -> Attention fusion (contract §7).

Canonical order (immutable unless a named ablation is selected):

```text
heatmap tokens H0
-> cross-attention 1 with DINO key/value (+ residual, gate gamma_1)
-> residual spatial NN bridge (+ residual, gate eta)
-> cross-attention 2 with the same DINO tensor as key/value (+ residual, gate gamma_2)
-> final LayerNorm
```

Attention 2 receives the bridge output as its query. The two cross-attention
modules have independent parameters and are never parallel branches.

Named ablations are implemented as first-class variants: ``aligned_add`` and
``concat`` replace both attention injections (while retaining the bridge),
``single_cross_attention`` removes Attention 2, ``no_fusion_residual`` removes
the residual paths around both cross-attentions, and ``fixed_fusion_gates``
freezes gamma_1/gamma_2/eta at their initial values.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .cross_attention import CrossAttention
from .spatial_bridge import SpatialBridge


class AlignedAddInjection(nn.Module):
    """Position-aligned addition: independently projected semantic tokens."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, semantic: torch.Tensor, return_weights=False):
        return self.drop(self.proj(semantic)), None


class ConcatInjection(nn.Module):
    """Position-aligned concatenation + projection."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(2 * dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, semantic: torch.Tensor, return_weights=False):
        return self.drop(self.proj(torch.cat([query, semantic], dim=-1))), None


def _build_injection(fusion: str, dim: int, dropout: float) -> nn.Module:
    if fusion == "serial_attention_spatial_attention":
        return CrossAttention(dim=dim, heads=4, dropout=dropout)
    if fusion == "aligned_add":
        return AlignedAddInjection(dim, dropout)
    if fusion == "concat":
        return ConcatInjection(dim, dropout)
    raise ValueError(f"unsupported fusion {fusion!r}")


class SemanticFusion(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        heads: int = 4,
        attention_dropout: float = 0.1,
        gamma_total_init: float = 0.1,
        bridge_eta_init: float = 0.1,
        bridge_dropout: float = 0.1,
        bridge_kind: str = "residual_dwconv_ffn",
        bridge_expansion_ratio: float = 2.0,
        bridge_kernel_size: int = 3,
        semantic_source: str = "dino",
        fusion_kind: str = "serial_attention_spatial_attention",
        single_attention: bool = False,
        fusion_residual: bool = True,
        learn_gates: bool = True,
    ):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.semantic_source = semantic_source  # "dino" | "none"
        self.fusion_kind = fusion_kind
        self.single_attention = single_attention
        self.fusion_residual = fusion_residual

        # gamma_1_init = gamma_2_init = semantic_gamma_total_init / 2.
        init_gamma = torch.tensor(gamma_total_init / 2.0)
        init_eta = torch.tensor(bridge_eta_init)
        if learn_gates:
            self.gamma1 = nn.Parameter(init_gamma.clone())
            self.gamma2 = nn.Parameter(init_gamma.clone())
            self.eta = nn.Parameter(init_eta.clone())
        else:
            self.register_buffer("gamma1", init_gamma.clone())
            self.register_buffer("gamma2", init_gamma.clone())
            self.register_buffer("eta", init_eta.clone())

        self.injection1 = _build_injection(fusion_kind, dim, attention_dropout)
        if bridge_kind == "identity":
            self.bridge = nn.Identity()
        else:
            self.bridge = SpatialBridge(
                dim=dim,
                expansion_ratio=bridge_expansion_ratio,
                kernel_size=bridge_kernel_size,
                kind=bridge_kind,
            )
        self.bridge_drop = nn.Dropout(bridge_dropout)
        if not single_attention:
            self.injection2 = _build_injection(fusion_kind, dim, attention_dropout)
        else:
            self.injection2 = None
        self.norm_out = nn.LayerNorm(dim)

    @property
    def has_semantic(self) -> bool:
        return self.semantic_source == "dino"

    def forward(
        self,
        heatmap_tokens: torch.Tensor,
        semantic_tokens: torch.Tensor | None,
        return_attention_weights: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Fuse heatmap tokens H0 with adapted DINO tokens T.

        Returns a dict with ``fused`` (Z), ``attention1_tokens`` (H1),
        ``bridge_tokens`` (M), and optional debug attention weights.
        """
        h0 = heatmap_tokens
        a1_weights = a2_weights = None

        if self.has_semantic and semantic_tokens is not None:
            a1, a1_weights = self.injection1(
                h0, semantic_tokens, return_weights=return_attention_weights
            )
            h1 = h0 + self.gamma1 * a1 if self.fusion_residual else self.gamma1 * a1
        else:
            h1 = h0

        bridge_out = self.bridge(h1)
        m = h1 + self.eta * self.bridge_drop(bridge_out)

        if self.has_semantic and semantic_tokens is not None and not self.single_attention:
            a2, a2_weights = self.injection2(
                m, semantic_tokens, return_weights=return_attention_weights
            )
            h2 = m + self.gamma2 * a2 if self.fusion_residual else self.gamma2 * a2
        else:
            h2 = m

        z = self.norm_out(h2)
        return {
            "fused": z,
            "attention1_tokens": h1,
            "bridge_tokens": m,
            "attention1_weights": a1_weights,
            "attention2_weights": a2_weights,
        }
