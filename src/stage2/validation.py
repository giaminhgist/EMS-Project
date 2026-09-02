"""Deterministic subject-level validation (guide 07 §9, contracts §24).

Runs in eval/inference mode over the fixed validation subject order, uses the
full fold bank (never a crossfit shard), includes each allowed subject exactly
once, aggregates losses as subject-weighted means and returns a typed result
plus the complete-set metrics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .config import Stage2Config
from .losses import compute_stage2_losses, generate_subset_masks
from .metrics import subject_metrics


class ValidationError(ValueError):
    pass


@dataclass
class ValidationResult:
    subject_ids: list[str]
    labels: torch.Tensor  # [N] int64
    raw_logits: torch.Tensor  # [N]
    probabilities: torch.Tensor  # [N]
    predictions: torch.Tensor  # [N] int64 (threshold 0.5)
    stimulus_attention: torch.Tensor  # [N,100]
    stimulus_evidence: torch.Tensor  # [N,100]
    stimulus_contribution: torch.Tensor  # [N,100]
    semantic_compatibility: torch.Tensor  # [N,100]
    normative_deviation: torch.Tensor  # [N,100]
    trial_mask: torch.Tensor  # [N,100] bool
    semantic_patch_maps: torch.Tensor | None  # [N,100,12,16] or None
    mean_losses: dict[str, float]
    metrics: dict[str, Any]
    # HC matching diagnostics (contract §22 history fields)
    attention_entropy_mean: float = 0.0
    matched_cosine_mean: float | None = None
    wrong_cosine_mean: float | None = None
    bank_rank_accuracy: float | None = None

    @property
    def n_subjects(self) -> int:
        return len(self.subject_ids)


def run_validation(
    model: Any,
    loader: Any,
    *,
    cfg: Stage2Config,
    epoch: int = 0,
    token_maps: bool = False,
) -> ValidationResult:
    """One deterministic validation pass over the full validation partition.

    ``loader`` must use the fixed-order subject sampler (shuffle=False,
    drop_last=False) so each allowed subject appears exactly once.
    """
    model.eval()
    expected_ids: list[str] = []
    for batch_indices in loader.batch_sampler:
        for idx in batch_indices:
            expected_ids.append(str(loader.dataset.subject_ids[idx]))
    if len(set(expected_ids)) != len(expected_ids):
        raise ValidationError("validation batch sampler repeats subjects")

    ids: list[str] = []
    labels_list: list[torch.Tensor] = []
    logits_list: list[torch.Tensor] = []
    attention_list: list[torch.Tensor] = []
    evidence_list: list[torch.Tensor] = []
    contribution_list: list[torch.Tensor] = []
    compatibility_list: list[torch.Tensor] = []
    deviation_list: list[torch.Tensor] = []
    mask_list: list[torch.Tensor] = []
    patch_maps_list: list[torch.Tensor] = []
    loss_sums: dict[str, float] = {}
    entropy_sum = 0.0
    matched_cos_sum = 0.0
    wrong_cos_sum = 0.0
    rank_acc_sum = 0.0
    n_hc_match = 0
    n_subjects_total = 0

    device = next(model.parameters()).device
    with torch.inference_mode():
        for batch in loader:
            batch = batch.to_device(device)
            n = batch.n_subjects
            n_subjects_total += n
            subset_masks = None
            if cfg.subsets.enabled:
                subset_masks = generate_subset_masks(
                    trial_mask=batch.trial_mask,
                    category_ids=batch.category_ids,
                    subject_ids=list(batch.subject_ids),
                    seed=cfg.seed,
                    fold=cfg.fold,
                    epoch=epoch,
                    min_fraction=cfg.subsets.min_fraction,
                    max_fraction=cfg.subsets.max_fraction,
                    train=False,  # validation masks are fixed by stable hash
                )
            result = model(
                batch, "val", subset_masks=subset_masks,
                debug_token_attention=token_maps,
            )
            enc = model.encode_trials(batch, "val", debug_token_attention=token_maps)
            match_inputs = model.matching_inputs(batch, enc=enc, epoch=epoch)
            anchor_current, anchor_stage1 = model.transferred_encoder.anchor_vectors()
            if cfg.loss.lambda_anchor == 0.0 or anchor_current.numel() == 0:
                anchor_current = anchor_stage1 = None
            losses = compute_stage2_losses(
                loss_cfg=cfg.loss,
                subsets_cfg=cfg.subsets,
                labels=batch.labels,
                full=result.full,
                subsets=result.subsets,
                match_inputs=match_inputs,
                anchor_current=anchor_current,
                anchor_stage1=anchor_stage1,
                epoch=epoch,
            )
            out = result.full
            for key, value in {
                "val_loss": float(losses.total),
                "val_cls_loss": float(losses.cls),
                "val_aux_loss": float(losses.aux),
                "val_match_loss": float(losses.match),
                "val_cons_loss": float(losses.cons),
            }.items():
                loss_sums[key] = loss_sums.get(key, 0.0) + value * n

            ids.extend(list(batch.subject_ids))
            labels_list.append(batch.labels.long().clone())
            logits_list.append(out.main_logit.clone())
            attention_list.append(out.stimulus_attention.clone())
            evidence_list.append(out.stimulus_evidence.clone())
            contribution_list.append(out.stimulus_contribution.clone())
            compatibility_list.append(out.semantic_compatibility.clone())
            deviation_list.append(out.normative_deviation.clone())
            mask_list.append(out.trial_mask.clone())
            if out.semantic_patch_map is not None and token_maps:
                patch_maps_list.append(out.semantic_patch_map.clone())

            entropy = attention_entropy(out.stimulus_attention, out.trial_mask)
            entropy_sum += float(entropy.mean()) * n
            if match_inputs is not None and int(match_inputs.hc_match_mask.sum()) > 0:
                hc_mask = match_inputs.hc_match_mask
                n_hc = int(hc_mask.sum())
                n_hc_match += n_hc
                matched_cos_sum += float(match_inputs.cos_pos[hc_mask].mean()) * n_hc
                wrong_cos_sum += float(match_inputs.cos_neg[hc_mask].mean()) * n_hc
                rank_acc_sum += float(
                    (match_inputs.comparator_pos[hc_mask] > match_inputs.comparator_neg[hc_mask])
                    .float().mean()
                ) * n_hc

    if len(ids) != len(expected_ids) or set(ids) != set(expected_ids):
        raise ValidationError(
            f"validation subject mismatch: collected {len(ids)}, expected {len(expected_ids)}"
        )
    if len(set(ids)) != len(ids):
        raise ValidationError("duplicate subject IDs in validation output")
    if n_subjects_total == 0:
        raise ValidationError("validation produced no subjects")

    labels = torch.cat(labels_list)
    logits = torch.cat(logits_list)
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= 0.5).long()
    mean_losses = {k: v / n_subjects_total for k, v in loss_sums.items()}
    metrics = subject_metrics(labels.numpy(), logits.numpy(), probabilities.numpy())

    return ValidationResult(
        subject_ids=ids,
        labels=labels,
        raw_logits=logits,
        probabilities=probabilities,
        predictions=predictions,
        stimulus_attention=torch.cat(attention_list),
        stimulus_evidence=torch.cat(evidence_list),
        stimulus_contribution=torch.cat(contribution_list),
        semantic_compatibility=torch.cat(compatibility_list),
        normative_deviation=torch.cat(deviation_list),
        trial_mask=torch.cat(mask_list),
        semantic_patch_maps=torch.cat(patch_maps_list) if patch_maps_list else None,
        mean_losses=mean_losses,
        metrics=metrics,
        attention_entropy_mean=entropy_sum / n_subjects_total,
        matched_cosine_mean=(matched_cos_sum / n_hc_match) if n_hc_match else None,
        wrong_cosine_mean=(wrong_cos_sum / n_hc_match) if n_hc_match else None,
        bank_rank_accuracy=(rank_acc_sum / n_hc_match) if n_hc_match else None,
    )


def attention_entropy(attention: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean per-subject entropy of the stimulus attention over valid slots."""
    a = attention.clamp_min(1e-7)
    entropy = -(a * torch.log(a) * mask).sum(dim=1)
    return entropy
