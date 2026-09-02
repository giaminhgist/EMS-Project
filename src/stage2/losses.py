"""Stage-2 loss components and the weighted total (guide 05 §15, contracts §18).

Every component is a pure function over explicit tensors and is independently
testable. The assembler ``compute_stage2_losses`` applies the contract §18.10
weighting and returns the full :class:`Stage2LossOutput`. Disabled components
return differentiable zeros built from the live graph.
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from .config import LossSectionConfig, SubsetsSectionConfig
from .contracts import N_STIMULI, Stage2ForwardOutput, Stage2LossOutput, Stage2MatchInputs


def differentiable_zero(template: torch.Tensor) -> torch.Tensor:
    """A zero scalar attached to the live graph (differentiable, grad 0)."""
    return template.sum() * 0.0


# ------------------------------------------------------------- subject-level BCE


def subject_bce(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """BCE on subject logits ``[B]`` and labels ``[B]``; trial-level inputs are rejected."""
    if logits.dim() != 1 or labels.dim() != 1:
        raise ValueError(
            f"subject BCE requires logits [B] and labels [B], got "
            f"{tuple(logits.shape)} and {tuple(labels.shape)}"
        )
    if logits.shape != labels.shape:
        raise ValueError(f"logits {tuple(logits.shape)} != labels {tuple(labels.shape)}")
    return F.binary_cross_entropy_with_logits(logits, labels.to(logits.dtype))


def auxiliary_bce(aux_logit: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Additive-evidence auxiliary logit BCE (contracts §18.2)."""
    return subject_bce(aux_logit, labels)


# ------------------------------------------------------------- matching losses


def trial_match_loss(
    cos_pos: torch.Tensor,
    cos_neg: torch.Tensor,
    hc_match_mask: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """``mean_HC[(1 - s+) + max(0, m + s- - s+)]`` (contracts §18.3)."""
    if not hc_match_mask.any():
        return differentiable_zero(cos_pos)
    s_pos = cos_pos[hc_match_mask]
    s_neg = cos_neg[hc_match_mask]
    return (1.0 - s_pos).mean() + F.relu(margin + s_neg - s_pos).mean()


def bank_rank_loss(
    comparator_pos: torch.Tensor,
    comparator_neg: torch.Tensor,
    hc_match_mask: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """``mean_HC max(0, m_b - r+ + r-)`` (contracts §18.4)."""
    if not hc_match_mask.any():
        return differentiable_zero(comparator_pos)
    r_pos = comparator_pos[hc_match_mask]
    r_neg = comparator_neg[hc_match_mask]
    return F.relu(margin - r_pos + r_neg).mean()


def token_match_loss(
    Q: torch.Tensor,
    N_mu_pos: torch.Tensor,
    N_mu_neg: torch.Tensor,
    token_rho: torch.Tensor,
    token_omega: torch.Tensor,
    hc_match_mask: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """``mean_{HC,t} rho * omega * [1 - cos(Q,N+) + max(0, m + cos(Q,N-) - cos(Q,N+))]``
    (contracts §18.5)."""
    if not hc_match_mask.any():
        return differentiable_zero(Q)
    cos_pos = (F.normalize(Q, dim=-1) * F.normalize(N_mu_pos, dim=-1)).sum(dim=-1)
    cos_neg = (F.normalize(Q, dim=-1) * F.normalize(N_mu_neg, dim=-1)).sum(dim=-1)
    per_patch = (
        token_rho.squeeze(-1)
        * token_omega
        * ((1.0 - cos_pos) + F.relu(margin + cos_neg - cos_pos))
    )
    return per_patch[hc_match_mask].mean()


# ------------------------------------------------------------- subset consistency


def latent_consistency_loss(u_a: torch.Tensor, u_b: torch.Tensor) -> torch.Tensor:
    """``mean_i [1 - cos(u_i^A, u_i^B)]`` (contracts §18.7)."""
    cos = (F.normalize(u_a, dim=-1) * F.normalize(u_b, dim=-1)).sum(dim=-1).clamp(-1.0, 1.0)
    return (1.0 - cos).mean()


def _bernoulli_kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p = p.clamp(1e-7, 1.0 - 1e-7)
    q = q.clamp(1e-7, 1.0 - 1e-7)
    return p * torch.log(p / q) + (1.0 - p) * torch.log((1.0 - p) / (1.0 - q))


def prob_consistency_loss(p_a: torch.Tensor, p_b: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence between Bernoulli prediction distributions."""
    m = 0.5 * (p_a + p_b)
    return (0.5 * _bernoulli_kl(p_a, m) + 0.5 * _bernoulli_kl(p_b, m)).mean()


# ------------------------------------------------------------- entropy + anchor


def entropy_floor_loss(
    importance: torch.Tensor, mask: torch.Tensor, floor: float
) -> torch.Tensor:
    """``mean_i max(0, H_min - H(I_i))`` over the effective mask (contracts §18.8)."""
    i_safe = importance.clamp_min(1e-7)
    entropy = -(i_safe * torch.log(i_safe) * mask).sum(dim=1)  # [B]
    return F.relu(floor - entropy).mean()


def encoder_anchor_loss(current: torch.Tensor, stage1: torch.Tensor) -> torch.Tensor:
    """``||theta_h - theta_h^Stage1||^2`` (contracts §18.9)."""
    if current.shape != stage1.shape:
        raise ValueError("anchor weight vectors must have identical shapes")
    return ((current - stage1) ** 2).sum()


# ------------------------------------------------------------- subset masks


def _subset_rng(
    seed: int, fold: int, epoch: int, subject_id: str, name: str, category: int, train: bool
) -> np.random.Generator:
    key = f"{seed}:{fold}:{epoch if train else 'fixed'}:{subject_id}:{name}:{category}"
    digest = int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")
    return np.random.default_rng(digest)


def generate_subset_masks(
    *,
    trial_mask: torch.Tensor,
    category_ids: torch.Tensor,
    subject_ids: Sequence[str],
    seed: int,
    fold: int,
    epoch: int,
    names: tuple[str, ...] = ("A", "B"),
    min_fraction: float = 0.5,
    max_fraction: float = 0.8,
    train: bool = True,
) -> dict[str, torch.Tensor]:
    """Category-stratified subset masks (guide §15.2).

    Per subject and present category, retains between ``min_fraction`` and
    ``max_fraction`` of the valid trials, always at least one per present
    category. Training masks vary deterministically with epoch; validation
    masks are fixed by a stable hash of ``(seed, fold, subject_id)``. Missing
    trials are never reactivated.
    """
    if not 0.0 < min_fraction < max_fraction <= 1.0:
        raise ValueError("require 0 < min_fraction < max_fraction <= 1")
    b, s = trial_mask.shape
    if category_ids.shape != (b, s):
        raise ValueError("category_ids must match trial_mask shape")
    masks: dict[str, torch.Tensor] = {
        name: torch.zeros(b, s, dtype=torch.bool, device=trial_mask.device) for name in names
    }
    mask_np = trial_mask.cpu().numpy()
    cat_np = category_ids.cpu().numpy()
    for i in range(b):
        for name in names:
            for k in range(4):
                slots = np.nonzero(mask_np[i] & (cat_np[i] == k))[0]
                if slots.size == 0:
                    continue
                rng = _subset_rng(seed, fold, epoch, str(subject_ids[i]), name, k, train)
                fraction = float(rng.uniform(min_fraction, max_fraction))
                n_keep = int(round(slots.size * fraction))
                n_keep = int(np.clip(n_keep, 1, slots.size))
                chosen = rng.choice(slots, size=n_keep, replace=False)
                masks[name][i, chosen] = True
    for name in names:
        masks[name] = masks[name] & trial_mask  # never reactivate a missing trial
    return masks


# ------------------------------------------------------------- total assembly


def compute_stage2_losses(
    *,
    loss_cfg: LossSectionConfig,
    subsets_cfg: SubsetsSectionConfig,
    labels: torch.Tensor,
    full: Stage2ForwardOutput,
    subsets: dict[str, Stage2ForwardOutput] | None,
    match_inputs: Stage2MatchInputs | None,
    anchor_current: torch.Tensor | None = None,
    anchor_stage1: torch.Tensor | None = None,
    epoch: int = 0,
) -> Stage2LossOutput:
    """Weighted total per contracts §18.10 plus every component and diagnostic."""
    zero = differentiable_zero(full.main_logit)
    subsets = subsets or {}
    subset_outputs = [subsets.get(name) for name in ("A", "B")]
    present_subset_outputs = [o for o in subset_outputs if o is not None]

    cls = subject_bce(full.main_logit, labels)
    for sub in present_subset_outputs:
        cls = cls + 0.5 * subject_bce(sub.main_logit, labels)

    aux = auxiliary_bce(full.auxiliary_logit, labels)

    if match_inputs is not None:
        trialmatch = trial_match_loss(
            match_inputs.cos_pos, match_inputs.cos_neg,
            match_inputs.hc_match_mask, loss_cfg.match_margin,
        )
        bankrank = bank_rank_loss(
            match_inputs.comparator_pos, match_inputs.comparator_neg,
            match_inputs.hc_match_mask, loss_cfg.bank_rank_margin,
        )
        if match_inputs.Q is not None:
            tokenmatch = token_match_loss(
                match_inputs.Q, match_inputs.N_mu_pos, match_inputs.N_mu_neg,
                match_inputs.token_rho, match_inputs.token_omega,
                match_inputs.hc_match_mask, loss_cfg.token_match_margin,
            )
            match = trialmatch + 0.5 * bankrank + 0.25 * tokenmatch
        else:
            tokenmatch = zero + 0.0
            match = trialmatch + 0.5 * bankrank
    else:
        trialmatch = bankrank = tokenmatch = match = zero + 0.0

    if len(present_subset_outputs) == 2:
        a, b = present_subset_outputs
        latent_cons = latent_consistency_loss(a.subject_embedding, b.subject_embedding)
        prob_cons = prob_consistency_loss(
            torch.sigmoid(a.main_logit), torch.sigmoid(b.main_logit)
        )
        cons = latent_cons + prob_cons
    else:
        latent_cons = prob_cons = cons = zero + 0.0

    entropy = entropy_floor_loss(
        full.stimulus_importance, full.trial_mask, loss_cfg.entropy_floor
    )
    anneal = 1.0 if loss_cfg.entropy_anneal_epochs == 0 else max(
        0.0, 1.0 - epoch / loss_cfg.entropy_anneal_epochs
    )

    if anchor_current is not None and anchor_stage1 is not None and loss_cfg.lambda_anchor > 0.0:
        anchor = encoder_anchor_loss(anchor_current, anchor_stage1)
    else:
        anchor = zero + 0.0

    total = (
        cls
        + loss_cfg.lambda_aux * aux
        + loss_cfg.lambda_match * match
        + loss_cfg.lambda_cons * cons
        + loss_cfg.lambda_entropy * anneal * entropy
        + loss_cfg.lambda_anchor * anchor
    )

    n_hc = int(match_inputs.hc_mask.sum()) if match_inputs is not None else 0
    n_hc_match = int(match_inputs.hc_match_mask.sum()) if match_inputs is not None else 0
    matched_cosine_mean = 0.0
    wrong_cosine_mean = 0.0
    bank_rank_accuracy = 0.0
    if match_inputs is not None and n_hc_match > 0:
        mask = match_inputs.hc_match_mask
        matched_cosine_mean = float(match_inputs.cos_pos[mask].mean().detach())
        wrong_cosine_mean = float(match_inputs.cos_neg[mask].mean().detach())
        bank_rank_accuracy = float(
            (match_inputs.comparator_pos[mask] > match_inputs.comparator_neg[mask])
            .float().mean().detach()
        )

    return Stage2LossOutput(
        total=total,
        cls=cls,
        aux=aux,
        match=match,
        trialmatch=trialmatch,
        bankrank=bankrank,
        tokenmatch=tokenmatch,
        cons=cons,
        latent_cons=latent_cons,
        prob_cons=prob_cons,
        entropy=entropy,
        anchor=anchor,
        n_hc_match_trials=n_hc_match,
        n_skipped_match_trials=n_hc - n_hc_match,
        matched_cosine_mean=matched_cosine_mean,
        wrong_cosine_mean=wrong_cosine_mean,
        bank_rank_accuracy=bank_rank_accuracy,
    )
