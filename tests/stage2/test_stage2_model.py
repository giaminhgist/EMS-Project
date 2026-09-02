"""Stage-2 model tests (guide 05 §16.1-§16.2, §17): base forward shapes,
gradient audit, single-encode guarantee, bank gather correctness and the
optional fused-token mode."""

from __future__ import annotations

import numpy as np
import torch
import pytest

from conftest import make_full_model_fixture, make_synthetic_batch
from stage2.collate import flatten_valid_trials
from stage2.losses import compute_stage2_losses, generate_subset_masks
from stage2.model import Stage2Model, Stage2ModelError
from stage2.bank import NormativeBankStore


@pytest.fixture()
def stack(tmp_path):
    return make_full_model_fixture(tmp_path)


def make_batch_with_missing(batch_size=4):
    """B=4 with missing trials in at least two subjects (guide §16.1)."""
    return make_synthetic_batch(
        batch_size, valid_counts=[20, 20, 16, 17], seed=3,
        labels=[0.0, 0.0, 1.0, 1.0], bank_split_ids=[0, 1, 0, 1],
    )


def test_base_forward_shapes_and_finiteness(stack):
    model: Stage2Model = stack["model"]
    batch = make_batch_with_missing()
    result = model(batch, "train")
    out = result.full
    assert out.main_logit.shape == (4,)
    assert out.auxiliary_logit.shape == (4,)
    assert out.subject_embedding.shape == (4, 128)
    assert out.trial_embeddings.shape == (4, 100, 128)
    assert out.trial_mask.shape == (4, 100) and out.trial_mask.dtype == torch.bool
    assert out.query_patch_attention.shape == (4, 100, 192)
    for name in (
        "stimulus_attention", "stimulus_importance", "stimulus_evidence",
        "stimulus_contribution", "semantic_compatibility",
        "normative_deviation", "weighted_normative_deviation",
    ):
        assert tuple(getattr(out, name).shape) == (4, 100), name
    assert out.semantic_patch_map is None  # trial-only bank mode
    for value in (out.main_logit, out.auxiliary_logit, out.subject_embedding,
                  out.trial_embeddings, out.stimulus_importance):
        assert torch.isfinite(value).all()
    # Missing trials stay missing in every panel.
    assert torch.equal(out.trial_mask, batch.trial_mask)


def test_frozen_encoder_single_encode_with_subsets(stack):
    model: Stage2Model = stack["model"]
    batch = make_batch_with_missing()
    counts: list[int] = []
    model.transferred_encoder.encoder.register_forward_hook(
        lambda m, args, out: counts.append(1)
    )
    masks = generate_subset_masks(
        trial_mask=batch.trial_mask, category_ids=batch.category_ids,
        subject_ids=list(batch.subject_ids), seed=2026, fold=0, epoch=0, train=True,
    )
    result = model(batch, "train", subset_masks=masks)
    assert set(result.subsets) == {"A", "B"}
    assert len(counts) == 1, "the frozen encoder must run exactly once per batch"


def test_train_keeps_frozen_encoder_in_eval(stack):
    model: Stage2Model = stack["model"]
    model.train()
    assert model.transferred_encoder.training is False
    assert model.transferred_encoder.encoder.training is False
    model.eval()


def test_gradient_audit(stack):
    model: Stage2Model = stack["model"]
    cfg = stack["cfg"]
    batch = make_batch_with_missing()
    masks = generate_subset_masks(
        trial_mask=batch.trial_mask, category_ids=batch.category_ids,
        subject_ids=list(batch.subject_ids), seed=cfg.seed, fold=0, epoch=0, train=True,
    )
    result = model(batch, "train", subset_masks=masks)
    enc = model.encode_trials(batch, "train")
    match_inputs = model.matching_inputs(batch, enc=enc, epoch=0)

    bank_before = {
        k: v.clone() for k, v in (
            ("mu_trial", model.bank_store.mu_trial),
            ("sigma_trial", model.bank_store.sigma_trial),
        )
    }
    losses = compute_stage2_losses(
        loss_cfg=cfg.loss, subsets_cfg=cfg.subsets, labels=batch.labels,
        full=result.full, subsets=result.subsets, match_inputs=match_inputs, epoch=0,
    )
    losses.total.backward()

    # Frozen encoder: no gradients at all.
    for name, param in model.transferred_encoder.encoder.named_parameters():
        assert param.grad is None, f"frozen encoder {name} received a gradient"

    # Every new Stage-2 module receives finite gradients, and each module has
    # at least one nonzero gradient. (The comparator bias cancels exactly by
    # the +r-/-r+ symmetry of the rank loss — inherent, not a defect.)
    required = [
        ("pooler", model.pooler),
        ("relation.query_projection", model.relation.query_projection),
        ("relation.bank_mean_adapter", model.relation.bank_mean_adapter),
        ("relation.bank_sigma_adapter", model.relation.bank_sigma_adapter),
        ("relation.reliability", model.relation.reliability),
        ("relation.relation", model.relation.relation),
        ("relation.comparator", model.relation.comparator),
        ("aggregator.gated_attention", model.aggregator.gated_attention),
        ("aggregator.transformer", model.aggregator.transformer),
        ("aggregator.main_head", model.aggregator.main_head),
        ("aggregator.evidence_head", model.aggregator.evidence_head),
    ]
    for name, module in required:
        max_abs = 0.0
        for pname, param in module.named_parameters():
            assert param.grad is not None, f"{name}.{pname} has no gradient"
            assert torch.isfinite(param.grad).all(), f"{name}.{pname} non-finite gradient"
            max_abs = max(max_abs, float(param.grad.abs().max()))
        assert max_abs > 0, f"{name} received only zero gradients"

    # Bank tensors: unchanged, no gradient.
    assert not model.bank_store.mu_trial.requires_grad
    assert torch.equal(model.bank_store.mu_trial, bank_before["mu_trial"])
    assert torch.equal(model.bank_store.sigma_trial, bank_before["sigma_trial"])


def make_distinctive_bank_rows(stack):
    """Orthogonal per-stimulus directions so relations differ across stimuli."""
    store: NormativeBankStore = stack["bank_store"]
    distinctive = torch.eye(100, 128) * 5.0
    np.save(stack["bank_root"] / "fold_0" / "mu_trial.npy", distinctive.numpy())
    for j in store.split_arrays:
        np.save(
            stack["bank_root"] / "fold_0" / "crossfit" / f"split_{j}" / "mu_trial.npy",
            distinctive.numpy(),
        )
    store.mu_trial = distinctive.clone()
    for j in store.split_tensors:
        store.split_tensors[j]["mu_trial"] = distinctive.clone()
    return distinctive


def test_correct_stimulus_bank_is_gathered(stack):
    """With distinctive per-stimulus bank rows, the gathered mean must be the
    row of the trial's own stimulus slot."""
    store: NormativeBankStore = stack["bank_store"]
    distinctive = make_distinctive_bank_rows(stack)
    model: Stage2Model = stack["model"]
    batch = make_batch_with_missing(batch_size=2)
    flat = flatten_valid_trials(batch)
    gathered = store.gather_trials(
        flat.stimulus_indices, flat.subject_slots, batch.bank_split_ids, "train"
    )
    assert torch.allclose(gathered.mu_trial, distinctive[flat.stimulus_indices])
    enc = model.encode_trials(batch, "train")
    assert enc.n_trials == flat.stimulus_indices.numel()
    # A wrong-stimulus gather must produce different bank means.
    wrong_idx = (flat.stimulus_indices + 1) % 100
    wrong = store.gather_trials(wrong_idx, flat.subject_slots, batch.bank_split_ids, "train")
    assert not torch.allclose(gathered.mu_trial, wrong.mu_trial)


def test_wrong_bank_changes_relation_for_controlled_fixture(stack):
    make_distinctive_bank_rows(stack)
    model: Stage2Model = stack["model"]
    batch = make_synthetic_batch(
        2, valid_counts=[20, 20], seed=5, labels=[0.0, 0.0], bank_split_ids=[0, 1]
    )
    match = model.matching_inputs(batch, epoch=0)
    assert (match.hc_match_mask == 1).all()
    assert (match.negative_stimulus_indices != flatten_valid_trials(batch).stimulus_indices).all()
    assert not torch.allclose(match.cos_pos, match.cos_neg)
    assert not torch.allclose(match.comparator_pos, match.comparator_neg)


def test_token_mode_forward_and_semantic_map(tmp_path):
    stack = make_full_model_fixture(
        tmp_path, include_token=True, model={"bank_mode": "trial_and_fused_token"}
    )
    model: Stage2Model = stack["model"]
    assert model.token_branch is not None
    batch = make_batch_with_missing()
    result = model(batch, "train", debug_token_attention=True)
    out = result.full
    assert out.semantic_patch_map is not None
    assert out.semantic_patch_map.shape == (4, 100, 12, 16)
    assert (out.semantic_patch_map[~out.trial_mask] == 0).all()
    enc = model.encode_trials(batch, "train", debug_token_attention=True)
    assert enc.token_attention_weights is not None
    assert enc.token_attention_weights.shape == (enc.n_trials, 4, 192, 9)
    # Token matching tensors are produced for the loss.
    match = model.matching_inputs(batch, enc=enc, epoch=0)
    assert match.Q is not None and match.N_mu_pos is not None and match.N_mu_neg is not None
    assert match.token_rho is not None and match.token_omega is not None


def test_token_mode_fails_without_token_arrays(tmp_path):
    stack = make_full_model_fixture(
        tmp_path,
        include_token=False,
        model={"bank_mode": "trial_and_fused_token"},
        build_model=False,
    )
    with pytest.raises(Stage2ModelError, match="no fused token banks"):
        Stage2Model(stack["cfg"], stack["bank_store"])


def test_parameter_report_and_encoder_metadata(stack):
    model: Stage2Model = stack["model"]
    report = model.parameter_report()
    assert report["total"] == report["trainable"] + report["frozen"]
    assert report["encoder"]["trainable"] == 0
    assert report["encoder"]["frozen"] == report["encoder"]["total"]
    assert model.transferred_encoder.checkpoint_sha256 == model.transferred_encoder.expected_sha256
    assert model.transferred_encoder.fold == 0
