"""Stage-2 loss tests (guide 05 §15, §16.4)."""

from __future__ import annotations

import torch
import pytest

from stage2.losses import (
    auxiliary_bce,
    bank_rank_loss,
    compute_stage2_losses,
    differentiable_zero,
    encoder_anchor_loss,
    entropy_floor_loss,
    generate_subset_masks,
    latent_consistency_loss,
    prob_consistency_loss,
    subject_bce,
    token_match_loss,
    trial_match_loss,
)


def test_perfect_logits_beat_reversed_logits():
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    perfect = torch.tensor([-5.0, -5.0, 5.0, 5.0])
    reversed_logits = torch.tensor([5.0, 5.0, -5.0, -5.0])
    assert subject_bce(perfect, labels) < subject_bce(reversed_logits, labels)


def test_subject_bce_rejects_trial_shaped_logits():
    with pytest.raises(ValueError, match="subject BCE"):
        subject_bce(torch.randn(2, 100), torch.tensor([0.0, 1.0]))


def test_wrong_bank_increases_match_and_rank_loss():
    n = 8
    mask = torch.ones(n, dtype=torch.bool)
    good_cos_pos = torch.ones(n)
    bad_cos_pos = 0.2 * torch.ones(n)
    cos_neg = 0.5 * torch.ones(n)
    assert trial_match_loss(good_cos_pos, cos_neg, mask, 0.2) < trial_match_loss(
        bad_cos_pos, cos_neg, mask, 0.2
    )
    good_comp = torch.ones(n)
    bad_comp = -torch.ones(n)
    comp_neg = 0.5 * torch.ones(n)
    assert bank_rank_loss(good_comp, comp_neg, mask, 0.2) < bank_rank_loss(
        bad_comp, comp_neg, mask, 0.2
    )


def test_sz_trials_never_enter_hc_matching_loss():
    """Only the hc_mask slots contribute — the mask is the single selector,
    built by the caller from subject labels mapped through flat slots."""
    cos_pos = torch.tensor([1.0, 0.5, 0.2, 0.0, 0.0, 0.0], requires_grad=True)
    cos_neg = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], requires_grad=True)
    hc_mask = torch.tensor([True, True, True, False, False, False])
    loss = trial_match_loss(cos_pos, cos_neg, hc_mask, 0.2)
    # Manual mean over the three HC slots only; SZ slots (with "perfect"
    # cos_neg=1 that would inflate the loss) must not enter.
    s_pos, s_neg = cos_pos[:3], cos_neg[:3]
    expected = (1.0 - s_pos).mean() + torch.relu(0.2 + s_neg - s_pos).mean()
    assert torch.allclose(loss, expected)
    # A fully excluded mask yields the differentiable zero.
    zero = trial_match_loss(cos_pos, cos_neg, torch.zeros(6, dtype=torch.bool), 0.2)
    assert zero.item() == 0.0
    assert zero.requires_grad  # differentiable zero


def test_subset_masks_respect_categories_and_missingness():
    torch.manual_seed(0)
    mask = torch.zeros(2, 100, dtype=torch.bool)
    mask[0, [0, 1, 2, 3, 4, 5, 6, 7]] = True  # all four categories, 2 each
    mask[1, [0, 1, 4, 5, 8]] = True  # categories 0, 1 only
    cats = torch.arange(100).expand(2, -1) % 4
    masks = generate_subset_masks(
        trial_mask=mask, category_ids=cats, subject_ids=["000", "101"],
        seed=2026, fold=0, epoch=0, train=True,
    )
    for name, sub in masks.items():
        # Never reactivate a missing trial.
        assert torch.equal(sub & ~mask, torch.zeros(2, 100, dtype=torch.bool))
        # At least one retained trial per present category; the per-category
        # retained fraction stays within [min_fraction, max_fraction] whenever
        # the category has at least two valid trials (a single-trial category
        # is always fully retained by the min-one rule).
        for i in range(2):
            for k in range(4):
                present = mask[i] & (cats[i] == k)
                n_k = int(present.sum())
                kept = int((sub[i] & present).sum())
                if n_k == 0:
                    assert kept == 0
                elif n_k == 1:
                    assert kept == 1
                else:
                    assert kept >= 1, (name, i, k)
                    assert 0.5 - 1e-6 <= kept / n_k <= 0.8 + 1e-6, (name, i, k)


def test_subset_masks_vary_with_epoch_but_fixed_for_validation():
    mask = torch.zeros(1, 100, dtype=torch.bool)
    mask[0, :40] = True
    cats = torch.arange(100).expand(1, -1) % 4
    kw = dict(trial_mask=mask, category_ids=cats, subject_ids=["000"],
              seed=2026, fold=0, min_fraction=0.5, max_fraction=0.8)
    train0 = generate_subset_masks(**kw, epoch=0, train=True)
    train1 = generate_subset_masks(**kw, epoch=1, train=True)
    assert not torch.equal(train0["A"], train1["A"]) or not torch.equal(
        train0["B"], train1["B"]
    )
    val0 = generate_subset_masks(**kw, epoch=0, train=False)
    val9 = generate_subset_masks(**kw, epoch=9, train=False)
    assert torch.equal(val0["A"], val9["A"]) and torch.equal(val0["B"], val9["B"])


def test_identical_subset_embeddings_give_zero_latent_consistency():
    u = torch.randn(4, 128)
    assert latent_consistency_loss(u, u).item() == pytest.approx(0.0, abs=1e-6)
    different = torch.randn(4, 128)
    assert latent_consistency_loss(u, different) > latent_consistency_loss(u, u)


def test_prob_consistency_is_jsd():
    p = torch.tensor([0.9, 0.1])
    q = torch.tensor([0.9, 0.1])
    assert prob_consistency_loss(p, q).item() == pytest.approx(0.0, abs=1e-6)
    r = torch.tensor([0.1, 0.9])
    assert prob_consistency_loss(p, r) > 0.0


def test_entropy_loss_activates_only_below_floor():
    # Uniform importance over K valid slots -> entropy = log K. The loss is
    # max(0, floor - H), so it activates only when H < floor.
    importance = torch.zeros(1, 100)
    mask = torch.zeros(1, 100, dtype=torch.bool)
    k = 10
    importance[0, :k] = 1.0 / k
    mask[0, :k] = True
    entropy = float(torch.tensor(k, dtype=torch.float64).log())
    assert entropy_floor_loss(importance, mask, floor=entropy + 0.1).item() > 0.0
    assert entropy_floor_loss(importance, mask, floor=entropy - 0.1).item() == 0.0


def test_anchor_loss_zero_at_stage1_weights():
    weights = torch.randn(100)
    assert encoder_anchor_loss(weights, weights).item() == 0.0
    assert encoder_anchor_loss(weights + 0.1, weights).item() > 0.0


def test_token_match_loss_uses_hc_mask_and_margin():
    n = 4
    Q = torch.randn(n, 8, 128, requires_grad=True)
    N_pos = torch.randn(n, 8, 128)
    N_neg = torch.randn(n, 8, 128) + 5.0  # far away -> margin term saturates
    rho = torch.rand(n, 8, 1)
    omega = torch.rand(n, 8)
    hc = torch.tensor([True, True, False, False])
    loss = token_match_loss(Q, N_pos, N_neg, rho, omega, hc, 0.2)
    # Manual mean over the two HC trials only.
    cos_pos = (torch.nn.functional.normalize(Q, dim=-1)
               * torch.nn.functional.normalize(N_pos, dim=-1)).sum(dim=-1)
    cos_neg = (torch.nn.functional.normalize(Q, dim=-1)
               * torch.nn.functional.normalize(N_neg, dim=-1)).sum(dim=-1)
    per_patch = rho.squeeze(-1) * omega * (
        (1.0 - cos_pos) + torch.relu(0.2 + cos_neg - cos_pos)
    )
    assert torch.allclose(loss, per_patch[hc].mean())
    # A fully excluded mask yields the differentiable zero.
    zero = token_match_loss(Q, N_pos, N_neg, rho, omega, torch.zeros(n, dtype=torch.bool), 0.2)
    assert zero.item() == 0.0
    assert zero.requires_grad


def test_disabled_token_loss_is_differentiable_zero():
    """compute_stage2_losses with trial-only matching yields a differentiable
    zero tokenmatch, and the total still backpropagates."""
    from stage2.contracts import Stage2ForwardOutput, Stage2MatchInputs

    torch.manual_seed(0)
    b = 4
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])

    def make_output(scale: float = 1.0):
        leaf_logit = torch.randn(b, requires_grad=True)
        leaf_aux = torch.randn(b, requires_grad=True)
        leaf_emb = torch.randn(b, 128, requires_grad=True)
        mask = torch.ones(b, 100, dtype=torch.bool)
        out = Stage2ForwardOutput(
            main_logit=scale * leaf_logit,
            auxiliary_logit=scale * leaf_aux,
            subject_embedding=scale * leaf_emb,
            trial_embeddings=torch.zeros(b, 100, 128),
            trial_mask=mask,
            query_patch_attention=torch.zeros(b, 100, 192),
            stimulus_attention=torch.zeros(b, 100),
            stimulus_importance=torch.ones(b, 100) / 100.0,
            stimulus_evidence=torch.zeros(b, 100),
            stimulus_contribution=torch.zeros(b, 100),
            semantic_compatibility=torch.zeros(b, 100),
            normative_deviation=torch.zeros(b, 100),
            weighted_normative_deviation=torch.zeros(b, 100),
            semantic_patch_map=None,
            diagnostics={},
        )
        return out, leaf_logit

    full, leaf_logit = make_output()
    subset_a, _ = make_output(0.5)
    subset_b, _ = make_output(0.7)
    n = 8
    match_inputs = Stage2MatchInputs(
        hc_mask=torch.ones(n, dtype=torch.bool),
        hc_match_mask=torch.ones(n, dtype=torch.bool),
        negative_stimulus_indices=torch.arange(n),
        cos_pos=torch.randn(n),
        cos_neg=torch.randn(n),
        comparator_pos=torch.randn(n),
        comparator_neg=torch.randn(n),
        rho=torch.rand(n, 1),
    )
    from stage2.config import LossSectionConfig, SubsetsSectionConfig

    losses = compute_stage2_losses(
        loss_cfg=LossSectionConfig(),
        subsets_cfg=SubsetsSectionConfig(),
        labels=labels,
        full=full,
        subsets={"A": subset_a, "B": subset_b},
        match_inputs=match_inputs,
        epoch=0,
    )
    assert losses.tokenmatch.item() == 0.0
    assert losses.tokenmatch.requires_grad
    losses.total.backward()
    assert leaf_logit.grad is not None


def test_total_loss_weights_follow_contract():
    """total = cls + 0.3 aux + 0.1 match + 0.1 cons + 0.01 entropy."""
    from stage2.config import LossSectionConfig, SubsetsSectionConfig
    from stage2.contracts import Stage2ForwardOutput, Stage2MatchInputs

    b = 2
    labels = torch.tensor([0.0, 1.0])
    mask = torch.ones(b, 100, dtype=torch.bool)
    full = Stage2ForwardOutput(
        main_logit=torch.tensor([1.0, -1.0]),
        auxiliary_logit=torch.tensor([1.0, -1.0]),
        subject_embedding=torch.randn(b, 128),
        trial_embeddings=torch.zeros(b, 100, 128),
        trial_mask=mask,
        query_patch_attention=torch.zeros(b, 100, 192),
        stimulus_attention=torch.zeros(b, 100),
        stimulus_importance=torch.ones(b, 100) / 100.0,
        stimulus_evidence=torch.zeros(b, 100),
        stimulus_contribution=torch.zeros(b, 100),
        semantic_compatibility=torch.zeros(b, 100),
        normative_deviation=torch.zeros(b, 100),
        weighted_normative_deviation=torch.zeros(b, 100),
        semantic_patch_map=None,
        diagnostics={},
    )
    n = 4
    mi = Stage2MatchInputs(
        hc_mask=torch.tensor([True, True, False, False]),
        hc_match_mask=torch.tensor([True, True, False, False]),
        negative_stimulus_indices=torch.tensor([1, 0, 1, 0]),
        cos_pos=torch.tensor([0.9, 0.9, 0.0, 0.0]),
        cos_neg=torch.tensor([0.4, 0.4, 0.0, 0.0]),
        comparator_pos=torch.tensor([0.7, 0.7, 0.0, 0.0]),
        comparator_neg=torch.tensor([0.3, 0.3, 0.0, 0.0]),
        rho=torch.rand(n, 1),
    )
    losses = compute_stage2_losses(
        loss_cfg=LossSectionConfig(lambda_entropy=0.0),
        subsets_cfg=SubsetsSectionConfig(),
        labels=labels,
        full=full,
        subsets=None,
        match_inputs=mi,
        epoch=0,
    )
    expected = (
        losses.cls + 0.3 * losses.aux + 0.1 * losses.match
        + 0.1 * losses.cons + 0.0 * losses.entropy + 0.0 * losses.anchor
    )
    assert torch.allclose(losses.total, expected)
    # match = trialmatch + 0.5 * bankrank
    assert torch.allclose(losses.match, losses.trialmatch + 0.5 * losses.bankrank)
    assert losses.n_hc_match_trials == 2
    assert losses.n_skipped_match_trials == 0
    assert losses.matched_cosine_mean == pytest.approx(0.9)
    assert losses.wrong_cosine_mean == pytest.approx(0.4)
    assert losses.bank_rank_accuracy == pytest.approx(1.0)
