"""Complete deterministic HC validation with diagnostic metrics (contract §5/§11)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from .losses import loo_cosine_normative_loss, stage1_loss
from .masking import realized_mask_ratio, validation_token_masks
from .metrics import (
    attention_entropy,
    cosine_similarity_maps,
    embedding_dim_std,
    head_averaged_map,
    js_divergence,
    residual_norm_ratio,
    shuffled_dino_reconstruction_delta,
)


@dataclass
class ValidationResult:
    metrics: dict[str, Any] = field(default_factory=dict)
    embeddings: np.ndarray | None = None
    trial_uids: list[str] = field(default_factory=list)
    stimulus_indices: list[int] = field(default_factory=list)
    subject_ids: list[str] = field(default_factory=list)


def run_validation(
    model: Any,
    dataset: Any,
    cfg: Any,
    epoch: int,
    device: str,
    *,
    max_batches: int | None = None,
    diagnostics_subset: int = 16,
) -> ValidationResult:
    """Deterministic validation over all HC validation trials.

    Fixed per-trial masks; embeddings grouped by stimulus for the LOO
    normative metric; diagnostic metrics on a fixed subset.
    """
    model.eval()
    val_batch_size = cfg.sampler.stimuli_per_batch * cfg.sampler.hc_per_stimulus
    ratio = cfg.masking.validation_mask_ratio
    lambda_norm = effective_lambda_norm(cfg, epoch)
    lambda_spread = effective_lambda_spread(cfg, epoch)

    all_uids = dataset.trial_uids()
    fixed_masks = validation_token_masks(all_uids, ratio, cfg.seed, cfg.fold)

    recon_total = 0.0
    recon_per_channel = np.zeros(3)
    embeddings_list: list[np.ndarray] = []
    uid_list: list[str] = []
    stimulus_list: list[int] = []
    subject_list: list[str] = []
    n_batches = 0
    n_groups = 0
    n_skipped = 0
    n_trials = 0

    indices = list(range(len(dataset)))
    with torch.inference_mode():
        for start in range(0, len(indices), val_batch_size):
            if max_batches is not None and n_batches >= max_batches:
                break
            idxs = indices[start : start + val_batch_size]
            batch = dataset.collate_from_indices(idxs)
            batch.to_device(device)
            mask = fixed_masks[start : start + val_batch_size].to(device)
            out = model(batch, mask)
            losses = stage1_loss(
                out.reconstruction, batch.heatmaps, mask,
                out.trial_embedding, batch.trial_to_stimulus_slot,
                lambda_norm=lambda_norm,
                lambda_spread=lambda_spread,
                spread_floor=cfg.loss.spread_floor,
                reconstruction_loss=cfg.loss.reconstruction,
                channel_weights=tuple(cfg.loss.channel_weights),
                channel_map=tuple(cfg.model.active_channels),
                scope=cfg.masking.reconstruction_scope,
                min_hc_per_stimulus=2,
            )
            n = len(idxs)
            recon_total += float(losses.reconstruction.item()) * n
            for c, name in enumerate(["fixation", "transition", "temporal"]):
                recon_per_channel[c] += float(getattr(losses, f"recon_{name}").item()) * n
            n_groups += losses.details.get("n_norm_groups", 0)
            n_skipped += losses.n_skipped_norm_groups
            embeddings_list.append(out.trial_embedding.cpu().numpy())
            uid_list.extend(batch.trial_uids)
            stimulus_list.extend(
                int(dataset.trial_rows.iloc[i].stimulus_index) for i in idxs
            )
            subject_list.extend(batch.subject_ids)
            n_trials += n
            n_batches += 1

    embeddings = (
        np.concatenate(embeddings_list, axis=0)
        if embeddings_list
        else np.zeros((0, cfg.d_model), dtype=np.float32)
    )
    recon_total = recon_total / max(n_trials, 1)
    recon_per_channel = recon_per_channel / max(n_trials, 1)

    emb_tensor = torch.from_numpy(embeddings)
    slot_tensor = torch.tensor(stimulus_list, dtype=torch.int64)
    norm = loo_cosine_normative_loss(
        emb_tensor, slot_tensor, min_hc_per_stimulus=2,
        spread_floor=cfg.loss.spread_floor,
    )
    val_loss = (
        recon_total
        + lambda_norm * float(norm["loss"].item())
        + lambda_spread * float(norm["spread_loss"].item())
    )

    # Diagnostics on the fixed subset (never updates weights).
    diagnostics: dict[str, Any] = {
        "semantic_gamma_attention1": float(model.fusion.gamma1.item()),
        "semantic_gamma_attention2": float(model.fusion.gamma2.item()),
        "spatial_bridge_eta": float(model.fusion.eta.item()),
    }
    diag_subset = min(diagnostics_subset, len(dataset))
    if diag_subset > 0 and n_batches > 0:
        diag_batch = dataset.collate_from_indices(list(range(diag_subset)))
        diag_batch.to_device(device)
        diag_mask = fixed_masks[:diag_subset].to(device)
        with torch.inference_mode():
            diag_out = model(
                diag_batch, diag_mask,
                return_attention1_tokens=True,
                return_bridge_tokens=True,
                return_heatmap_tokens=True,
                debug_attention=True,
            )
        a1_map = head_averaged_map(diag_out.attention1_weights)
        a2_map = (
            head_averaged_map(diag_out.attention2_weights)
            if diag_out.attention2_weights is not None
            else None
        )
        r1 = model.fusion.gamma1 * (diag_out.attention1_tokens - diag_out.heatmap_tokens)
        r_bridge = model.fusion.eta * (diag_out.bridge_tokens - diag_out.attention1_tokens)
        diagnostics.update(
            {
                "attention1_entropy": attention_entropy(diag_out.attention1_weights),
                "attention2_entropy": attention_entropy(diag_out.attention2_weights),
                "attn_map_js_divergence": (
                    js_divergence(a1_map, a2_map) if a2_map is not None else float("nan")
                ),
                "attn_map_cosine_similarity": (
                    cosine_similarity_maps(a1_map, a2_map)
                    if a2_map is not None
                    else float("nan")
                ),
                "residual_ratio_attention1": residual_norm_ratio(
                    r1, diag_out.heatmap_tokens
                ),
                "residual_ratio_bridge": residual_norm_ratio(
                    r_bridge, diag_out.attention1_tokens
                ),
                "per_dim_embedding_std_mean": float(
                    embedding_dim_std(emb_tensor).mean().item()
                ) if n_trials else float("nan"),
                "shuffled_dino_recon_delta": shuffled_dino_reconstruction_delta(
                    model,
                    dataset.collate_from_indices(list(range(diag_subset))).to_device(device),
                    diag_mask,
                ),
            }
        )

    metrics = {
        "val_loss": val_loss,
        "val_recon_loss": recon_total,
        "val_recon_fixation": float(recon_per_channel[0]),
        "val_recon_transition": float(recon_per_channel[1]),
        "val_recon_temporal": float(recon_per_channel[2]),
        "val_norm_loss": float(norm["loss"].item()),
        "val_spread_loss": float(norm["spread_loss"].item()),
        "val_within_stimulus_dispersion": norm["within_dispersion"],
        "val_between_stimulus_dispersion": norm["between_dispersion"],
        "n_val_batches": n_batches,
        "num_val_trials": n_trials,
        "n_val_stimulus_groups": n_groups,
        "n_skipped_norm_groups_val": n_skipped,
        "val_mask_ratio_realized": realized_mask_ratio(fixed_masks[:n_trials]),
        "diagnostics": diagnostics,
    }
    return ValidationResult(
        metrics=metrics,
        embeddings=embeddings,
        trial_uids=uid_list,
        stimulus_indices=stimulus_list,
        subject_ids=subject_list,
    )


def _ramp_progress(cfg: Any, epoch: int) -> tuple[float, bool]:
    """(progress, active) for the normative lambda ramp; inactive before start."""
    start = cfg.loss.norm_start_epoch
    ramp = cfg.loss.norm_ramp_epochs
    if epoch < start:
        return 0.0, False
    if ramp <= 0:
        return 1.0, True
    return min(1.0, (epoch - start + 1) / ramp), True


def effective_lambda_norm(cfg: Any, epoch: int) -> float:
    """Phase A (warm-up, lambda=0) -> Phase B (linear ramp) -> Phase C (full)."""
    progress, active = _ramp_progress(cfg, epoch)
    return cfg.loss.lambda_norm * progress if active else 0.0


def effective_lambda_spread(cfg: Any, epoch: int) -> float:
    """Spread-floor lambda shares the normative ramp schedule."""
    progress, active = _ramp_progress(cfg, epoch)
    return cfg.loss.lambda_spread * progress if active else 0.0


def best_checkpoint_eligible(cfg: Any, epoch: int) -> bool:
    """Epochs are eligible only after the normative ramp is complete."""
    return epoch >= cfg.loss.norm_start_epoch + cfg.loss.norm_ramp_epochs
