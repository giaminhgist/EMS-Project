"""Stage-1 losses (contract §10): masked reconstruction + HC LOO norm.

No VICReg, InfoNCE, SupCon, diagnosis cross-entropy, or SZ label appears here.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .masking import token_mask_to_pixel_mask
from .types import Stage1Losses


def masked_reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    token_mask: torch.Tensor,
    *,
    scope: str = "masked",
    reduction: str = "smooth_l1",
    channel_weights: tuple[float, ...] = (1.0, 1.0, 1.0),
    channel_map: tuple[int, ...] = (0, 1, 2),
    return_per_channel: bool = True,
) -> dict[str, Any]:
    """Channel-aware reconstruction loss on masked (or full) pixels.

    ``target`` is the fixed ``log1p``/clip transform of the input heatmaps.
    ``channel_map`` maps each reconstruction channel index to its semantic
    channel (0=fixation, 1=transition, 2=temporal); inactive channels report
    exactly 0.0.
    """
    if reconstruction.shape != target.shape:
        raise ValueError("reconstruction and target shapes differ")
    if reconstruction.shape[1] != len(channel_weights) or reconstruction.shape[1] != len(channel_map):
        raise ValueError("channel_weights/channel_map must match the reconstruction channels")
    pixel_mask = token_mask_to_pixel_mask(token_mask)  # [N, 1, 48, 64]
    if scope == "masked":
        n_masked = int(pixel_mask.sum())
        if n_masked == 0:
            raise ValueError(
                "masked-only reconstruction configured with zero masked pixels"
            )
    elif scope == "full":
        pixel_mask = torch.ones_like(pixel_mask)
    else:
        raise ValueError(f"unsupported reconstruction scope {scope!r}")

    names = {0: "fixation", 1: "transition", 2: "temporal"}
    per_channel: dict[str, torch.Tensor] = {}
    for c in range(reconstruction.shape[1]):
        diff = reconstruction[:, c : c + 1] - target[:, c : c + 1]
        if reduction == "smooth_l1":
            elem = F.smooth_l1_loss(diff, torch.zeros_like(diff), reduction="none")
        elif reduction == "l1":
            elem = diff.abs()
        else:
            raise ValueError(f"unsupported reduction {reduction!r}")
        masked = elem * pixel_mask
        per_channel[names[channel_map[c]]] = masked.sum() / pixel_mask.sum()
    total = sum(
        per_channel[names[channel_map[c]]] * channel_weights[c]
        for c in range(reconstruction.shape[1])
    )
    out: dict[str, Any] = {
        "total": total,
        "fixation": per_channel.get("fixation", reconstruction.sum() * 0.0),
        "transition": per_channel.get("transition", reconstruction.sum() * 0.0),
        "temporal": per_channel.get("temporal", reconstruction.sum() * 0.0),
        "n_masked_pixels": int(pixel_mask.sum()),
    }
    return out


def loo_cosine_normative_loss(
    embeddings: torch.Tensor,
    stimulus_slot: torch.Tensor,
    *,
    min_hc_per_stimulus: int = 2,
    return_dispersion: bool = True,
    spread_floor: float = 0.1,
) -> dict[str, Any]:
    """Leave-one-HC-out cosine consistency within each stimulus group.

    ``mu_{-h,s}`` is the centroid of the other trials of the same stimulus,
    stop-gradiented. Groups with fewer than ``min_hc_per_stimulus`` trials are
    skipped and counted (never mixed across stimuli).

    Also returns ``spread_loss``: a hinge over between-centroid cosine
    dispersion (``max(0, spread_floor - (1 - cos(c_i, c_j)))^2`` over stimulus
    centroid pairs, 0 when fewer than 2 centroids). Centroid gradients are
    NOT detached here — this term is the anti-collapse repulsion between
    stimuli (approved contract amendment, 2026-08-31).
    """
    n = embeddings.shape[0]
    if n == 0:
        return {
            "loss": embeddings.sum() * 0.0,
            "within_dispersion": 0.0,
            "between_dispersion": 0.0,
            "n_skipped_groups": 0,
            "n_groups": 0,
        }
    z = F.normalize(embeddings, dim=-1, eps=1e-6)  # cosine operates on unit vectors
    total = embeddings.sum() * 0.0
    n_used = 0
    n_skipped = 0
    n_groups = 0
    within_parts: list[torch.Tensor] = []
    centroids: list[torch.Tensor] = []

    for s in torch.unique(stimulus_slot):
        idx = torch.nonzero(stimulus_slot == s, as_tuple=False).squeeze(-1)
        h = idx.numel()
        n_groups += 1
        if h < min_hc_per_stimulus:
            n_skipped += 1
            continue
        group = z[idx]  # [h, d]
        centroid = group.mean(dim=0)  # [d]
        for k in idx:
            z_k = z[k]
            mu_minus = (centroid - z_k / h) * (h / (h - 1.0))  # leave-one-out centroid
            total = total + (1.0 - torch.dot(z_k, mu_minus.detach()))
            n_used += 1
        # within-stimulus dispersion: mean distance of members to centroid
        within_parts.append(torch.mean(1.0 - group @ centroid))
        centroids.append(centroid)

    if n_used > 0:
        loss = total / n_used
    else:
        loss = embeddings.sum() * 0.0

    within = torch.mean(torch.stack(within_parts)) if within_parts else embeddings.sum() * 0.0
    if len(centroids) >= 2:
        c = torch.stack(centroids)
        c_norm = F.normalize(c, dim=-1, eps=1e-6)
        sim = c_norm @ c_norm.T
        off = ~torch.eye(len(c), dtype=torch.bool, device=c.device)
        between = (1.0 - sim[off]).mean()
        # Hinge on between-centroid dispersion: active only while centroids
        # are closer than the floor (cosine sim above 1 - spread_floor).
        violation = torch.clamp_min(spread_floor - (1.0 - sim[off]), 0.0)
        spread = (violation ** 2).mean()
    else:
        between = embeddings.sum() * 0.0
        spread = embeddings.sum() * 0.0

    return {
        "loss": loss,
        "spread_loss": spread,
        "within_dispersion": float(within.item()) if torch.is_tensor(within) else float(within),
        "between_dispersion": float(between.item()) if torch.is_tensor(between) else float(between),
        "n_skipped_groups": int(n_skipped),
        "n_groups": int(n_groups),
    }


def stage1_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    token_mask: torch.Tensor,
    embeddings: torch.Tensor,
    stimulus_slot: torch.Tensor,
    *,
    lambda_norm: float,
    lambda_spread: float = 0.0,
    spread_floor: float = 0.1,
    reconstruction_loss: str = "smooth_l1",
    channel_weights: tuple[float, ...] = (1.0, 1.0, 1.0),
    channel_map: tuple[int, ...] = (0, 1, 2),
    scope: str = "masked",
    min_hc_per_stimulus: int = 2,
) -> Stage1Losses:
    recon = masked_reconstruction_loss(
        reconstruction,
        target,
        token_mask,
        scope=scope,
        reduction=reconstruction_loss,
        channel_weights=channel_weights,
        channel_map=channel_map,
    )
    norm = loo_cosine_normative_loss(
        embeddings,
        stimulus_slot,
        min_hc_per_stimulus=min_hc_per_stimulus,
        spread_floor=spread_floor,
    )
    total = recon["total"] + lambda_norm * norm["loss"] + lambda_spread * norm["spread_loss"]
    return Stage1Losses(
        total=total,
        reconstruction=recon["total"],
        recon_fixation=recon["fixation"],
        recon_transition=recon["transition"],
        recon_temporal=recon["temporal"],
        normative=norm["loss"],
        spread_loss=norm["spread_loss"],
        within_stimulus_dispersion=norm["within_dispersion"],
        between_stimulus_dispersion=norm["between_dispersion"],
        n_skipped_norm_groups=norm["n_skipped_groups"],
        lambda_norm=lambda_norm,
        lambda_spread=lambda_spread,
        details={"n_masked_pixels": recon["n_masked_pixels"], "n_norm_groups": norm["n_groups"]},
    )
