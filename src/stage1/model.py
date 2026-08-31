"""Stage-1 HC-only normative model (contract §8).

Components: trainable heatmap patch encoder, trainable DINO semantic adapter,
serial residual fusion (attention 1 -> spatial NN bridge -> attention 2),
trainable masked-reconstruction decoder, attention/mean trial pooling.

No classifier and no VICReg projector exist anywhere in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from .decoder import ReconstructionDecoder
from .heatmap_encoder import HeatmapPatchEncoder
from .pooling import build_pooling
from .semantic_adapter import SemanticAdapter
from .semantic_fusion import SemanticFusion
from .types import Stage1Batch, Stage1ForwardOutput


class Stage1Model(nn.Module):
    def __init__(self, cfg: Any):
        super().__init__()
        m = cfg.model
        self.d_model = m.d_model
        self.input_channels = m.input_channels
        self.heatmap_encoder = HeatmapPatchEncoder(
            in_channels=m.input_channels,
            d_model=m.d_model,
            patch_size=m.heatmap_patch_size,
            grid_h=12,
            grid_w=16,
            n_residual_blocks=m.heatmap_residual_blocks,
        )
        self.semantic_adapter = SemanticAdapter(
            kind=m.semantic_adapter, in_dim=384, d_model=m.d_model
        )
        self.fusion = SemanticFusion(
            dim=m.d_model,
            heads=m.attention_heads,
            attention_dropout=m.attention_dropout,
            gamma_total_init=m.semantic_gamma_total_init,
            bridge_eta_init=m.spatial_bridge_eta_init,
            bridge_dropout=m.spatial_bridge_dropout,
            bridge_kind=m.spatial_bridge,
            bridge_expansion_ratio=m.spatial_bridge_expansion_ratio,
            bridge_kernel_size=m.spatial_bridge_kernel_size,
            semantic_source=m.semantic_source,
            fusion_kind=m.fusion,
            single_attention=m.single_cross_attention,
            fusion_residual=m.fusion_residual,
            learn_gates=m.learn_fusion_gates,
        )
        self.decoder = ReconstructionDecoder(dim=m.d_model, out_channels=m.input_channels)
        self.pooling = build_pooling(m.pooling, m.d_model)

    @property
    def gamma_init(self) -> float:
        return self.fusion.gamma1.item()

    def forward(
        self,
        batch: Stage1Batch,
        token_mask: torch.Tensor | None = None,
        *,
        return_fused: bool = False,
        return_attention1_tokens: bool = False,
        return_bridge_tokens: bool = False,
        return_pooling_weights: bool = False,
        return_semantic_tokens: bool = False,
        return_heatmap_tokens: bool = False,
        debug_attention: bool = False,
    ) -> Stage1ForwardOutput:
        """Forward pass; ``token_mask=None`` means unmasked (norm-bank mode).

        Raw DINO tensors never receive gradients (they are plain input
        tensors; the adapter is the trainable path).
        """
        if token_mask is None:
            token_mask = torch.zeros(
                batch.n_trials, 192, dtype=torch.bool, device=batch.heatmaps.device
            )
        heatmap_tokens = self.heatmap_encoder(batch.heatmaps, token_mask)

        if self.fusion.has_semantic:
            unique_tokens = self.semantic_adapter(batch.unique_dino_tokens)  # [S, 192, d]
            semantic_tokens = unique_tokens[batch.trial_to_stimulus_slot]  # [N, 192, d]
        else:
            semantic_tokens = None

        fusion_out = self.fusion(
            heatmap_tokens,
            semantic_tokens,
            return_attention_weights=debug_attention,
        )
        reconstruction = self.decoder(fusion_out["fused"])
        embedding, pooling_weights = self.pooling(
            fusion_out["fused"], return_weights=return_pooling_weights
        )

        return Stage1ForwardOutput(
            reconstruction=reconstruction,
            trial_embedding=embedding,
            mask=token_mask,
            fused_tokens=fusion_out["fused"] if return_fused else None,
            attention1_tokens=fusion_out["attention1_tokens"] if return_attention1_tokens else None,
            bridge_tokens=fusion_out["bridge_tokens"] if return_bridge_tokens else None,
            semantic_tokens=semantic_tokens if return_semantic_tokens else None,
            heatmap_tokens=heatmap_tokens if return_heatmap_tokens else None,
            pooling_weights=pooling_weights,
            attention1_weights=fusion_out["attention1_weights"],
            attention2_weights=fusion_out["attention2_weights"],
        )


@dataclass
class ModelSummary:
    n_parameters_total: int
    n_trainable: int
    n_frozen: int
    trainable_modules: list[str]
    frozen_modules: list[str]
    per_module: dict[str, int] = field(default_factory=dict)


def summarize_model(model: nn.Module) -> ModelSummary:
    trainable_mods: list[str] = []
    frozen_mods: list[str] = []
    per_module: dict[str, int] = {}
    n_trainable = 0
    n_frozen = 0
    for name, module in model.named_modules():
        params = sum(p.numel() for p in module.parameters(recurse=False))
        if params == 0:
            continue
        per_module[name] = params
        if any(p.requires_grad for p in module.parameters(recurse=False)):
            trainable_mods.append(name)
            n_trainable += params
        else:
            frozen_mods.append(name)
            n_frozen += params
    return ModelSummary(
        n_parameters_total=n_trainable + n_frozen,
        n_trainable=n_trainable,
        n_frozen=n_frozen,
        trainable_modules=trainable_mods,
        frozen_modules=frozen_mods,
        per_module=per_module,
    )
