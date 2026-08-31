"""Metric aggregation and diagnostic metrics for semantic usage (contract §11)."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


def attention_entropy(weights: torch.Tensor) -> float:
    """Mean per-head attention entropy over queries (weights [N, H, Tq, Tk])."""
    if weights is None or weights.numel() == 0:
        return float("nan")
    log_w = torch.log(weights.clamp_min(1e-12))
    entropy = -(weights * log_w).sum(dim=-1).mean()  # mean over N, H, queries
    return float(entropy.item())


def head_averaged_map(weights: torch.Tensor) -> torch.Tensor:
    """[N, H, Tq, Tk] -> [N, Tq, Tk] averaged over heads."""
    return weights.mean(dim=1)


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> float:
    """Jensen-Shannon divergence between two distributions over the last dim."""
    m = 0.5 * (p + q)
    kl_pm = (p * torch.log((p.clamp_min(1e-12)) / m.clamp_min(1e-12))).sum(dim=-1)
    kl_qm = (q * torch.log((q.clamp_min(1e-12)) / m.clamp_min(1e-12))).sum(dim=-1)
    js = 0.5 * (kl_pm + kl_qm)
    return float(js.mean().item())


def cosine_similarity_maps(p: torch.Tensor, q: torch.Tensor) -> float:
    """Mean cosine similarity between two sets of row-normalized maps."""
    pn = F.normalize(p.reshape(-1, p.shape[-1]), dim=-1)
    qn = F.normalize(q.reshape(-1, q.shape[-1]), dim=-1)
    return float((pn * qn).sum(dim=-1).mean().item())


def residual_norm_ratio(residual: torch.Tensor, base: torch.Tensor) -> float:
    """||residual|| / ||base|| averaged over the batch."""
    if base.numel() == 0 or float(base.norm()) == 0.0:
        return float("nan")
    return float((residual.norm(dim=(1, 2)) / base.norm(dim=(1, 2))).mean().item())


def embedding_dim_std(embeddings: torch.Tensor) -> torch.Tensor:
    """Per-dimension standard deviation of the trial embeddings."""
    if embeddings.shape[0] < 2:
        return torch.zeros(embeddings.shape[1], dtype=embeddings.dtype, device=embeddings.device)
    return embeddings.std(dim=0)


def shuffled_dino_reconstruction_delta(
    model: Any,
    batch: Any,
    token_mask: torch.Tensor,
) -> float:
    """Evaluation-only semantic-dependence test: reconstruct once normally and
    once with DINO stimulus features shuffled across trials. No weight updates;
    the batch's slot mapping is restored afterwards.
    """
    original_slot = batch.trial_to_stimulus_slot
    with torch.inference_mode():
        normal = model(batch, token_mask).reconstruction
        batch.trial_to_stimulus_slot = original_slot[
            torch.randperm(len(original_slot))
        ]
        shuffled = model(batch, token_mask).reconstruction
        delta = float((normal - shuffled).abs().mean().item())
    batch.trial_to_stimulus_slot = original_slot
    return delta


def realized_gates(model: Any) -> dict[str, float]:
    return {
        "semantic_gamma_attention1": float(model.fusion.gamma1.item()),
        "semantic_gamma_attention2": float(model.fusion.gamma2.item()),
        "spatial_bridge_eta": float(model.fusion.eta.item()),
    }
