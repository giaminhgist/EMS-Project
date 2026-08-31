"""Typed sample/batch/output structures for Stage 1 (contract §2, §8, §11)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class Stage1Sample:
    heatmap: torch.Tensor  # float32 [3, 48, 64] (fixed log1p/clip transform)
    subject_id: str
    stimulus_id: str
    stimulus_index: int
    trial_uid: str
    group: str  # always "HC" in Stage 1


@dataclass
class Stage1Batch:
    """Grouped training batch (contract §8)."""

    heatmaps: torch.Tensor  # [N, 3, 48, 64]
    unique_dino_tokens: torch.Tensor  # [S, 768, 384]
    trial_to_stimulus_slot: torch.Tensor  # [N] int64
    stimulus_indices: torch.Tensor  # [S] int64 (stimulus_index per unique slot)
    subject_ids: list[str]
    stimulus_ids: list[str]  # length N (per trial)
    trial_uids: list[str]
    groups: list[str]

    @property
    def n_trials(self) -> int:
        return self.heatmaps.shape[0]

    @property
    def n_unique_stimuli(self) -> int:
        return self.unique_dino_tokens.shape[0]

    def to_device(self, device: torch.device | str) -> "Stage1Batch":
        self.heatmaps = self.heatmaps.to(device)
        self.unique_dino_tokens = self.unique_dino_tokens.to(device)
        self.trial_to_stimulus_slot = self.trial_to_stimulus_slot.to(device)
        self.stimulus_indices = self.stimulus_indices.to(device)
        return self


@dataclass
class Stage1ForwardOutput:
    reconstruction: torch.Tensor  # [N, 3, 48, 64]
    trial_embedding: torch.Tensor  # [N, 128]
    mask: torch.Tensor  # [N, 192] bool
    fused_tokens: torch.Tensor | None = None
    attention1_tokens: torch.Tensor | None = None
    bridge_tokens: torch.Tensor | None = None
    semantic_tokens: torch.Tensor | None = None  # adapted, gathered [N, 192, 128]
    heatmap_tokens: torch.Tensor | None = None  # pre-fusion heatmap tokens [N, 192, 128]
    pooling_weights: torch.Tensor | None = None  # [N, 192]
    attention1_weights: torch.Tensor | None = None  # [N, 4, 192, 192] debug only
    attention2_weights: torch.Tensor | None = None  # [N, 4, 192, 192] debug only


@dataclass
class Stage1Losses:
    total: torch.Tensor
    reconstruction: torch.Tensor
    recon_fixation: torch.Tensor
    recon_transition: torch.Tensor
    recon_temporal: torch.Tensor
    normative: torch.Tensor
    within_stimulus_dispersion: float
    between_stimulus_dispersion: float
    n_skipped_norm_groups: int
    lambda_norm: float
    details: dict[str, Any] = field(default_factory=dict)
