"""Ablation framework tests (guide 06 §6-§10): registry specs, overlay-diff
validation, controlled implementations and per-ablation forward/backward."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from conftest import make_full_model_fixture, make_synthetic_batch
from stage2.ablations import (
    ABLATIONS,
    BASE_CONFIG_DEFAULT,
    build_wrong_bank_permutation,
    compute_global_bank_stats,
    resolve_ablation_config,
    validate_bank_capabilities,
)
from stage2.bank import FULL_BANK_ID, NormativeBankStore
from stage2.config import ConfigError, Stage2Config
from stage2.losses import compute_stage2_losses, generate_subset_masks
from stage2.model import Stage2Model, Stage2ModelError
from stage2.pooling import MeanQueryPooler
from stage2.subject_aggregation import SubjectAggregator
from stage2.contracts import EncodedTrials

RUNNABLE = [name for name in ABLATIONS if name != "same_space_heat_bank"]


# ------------------------------------------------------- registry and diffs


def test_registry_specs_are_complete():
    for name, spec in ABLATIONS.items():
        assert spec.name == name
        assert spec.scientific_question
        assert isinstance(spec.declared_changes, tuple)
        assert isinstance(spec.required_bank_capabilities, tuple)
        assert isinstance(spec.forbidden_with, tuple)
        assert isinstance(spec.is_negative_control, bool)
        assert spec.reference in ABLATIONS


def test_every_overlay_changes_exactly_its_declared_keys():
    for name in RUNNABLE:
        resolved = resolve_ablation_config(BASE_CONFIG_DEFAULT, name, fold=0)
        assert set(resolved.changed_keys) == set(
            ABLATIONS[name].declared_changes
        ), name
    # Child token ablations reference the token model, not base.
    for name, spec in ABLATIONS.items():
        if name in ("single_token_attention", "no_spatial_bridge"):
            assert spec.reference == "token_bank_serial_attention", name
        else:
            assert spec.reference == "base", name


def test_same_space_heat_bank_rejected_cleanly():
    with pytest.raises(ConfigError, match="not runnable"):
        resolve_ablation_config(BASE_CONFIG_DEFAULT, "same_space_heat_bank", fold=0)


def test_token_ablations_fail_without_token_arrays(tmp_path):
    stack = make_full_model_fixture(
        tmp_path, include_token=False, build_model=False
    )
    store = stack["bank_store"]
    for name in ("token_bank_serial_attention", "single_token_attention", "no_spatial_bridge"):
        resolved = resolve_ablation_config(BASE_CONFIG_DEFAULT, name, fold=0)
        with pytest.raises(ConfigError, match="fused token banks"):
            validate_bank_capabilities(resolved.spec, store)
        with pytest.raises(Stage2ModelError, match="no fused token banks"):
            Stage2Model(resolved.config, store)


# ------------------------------------------------------- controlled helpers


def test_wrong_bank_permutation_properties():
    categories = [i % 4 for i in range(100)]
    perm = build_wrong_bank_permutation(categories, seed=2026, fold=0)
    assert set(perm) == set(range(100))
    # No fixed point for categories with >= 2 members; same-category mapping.
    for s, t in perm.items():
        assert categories[t] == categories[s]
        members = [i for i in range(100) if categories[i] == categories[s]]
        if len(members) >= 2:
            assert t != s
    # Deterministic in (seed, fold); changes with seed.
    again = build_wrong_bank_permutation(categories, seed=2026, fold=0)
    assert perm == again
    other = build_wrong_bank_permutation(categories, seed=2027, fold=0)
    assert perm != other
    # Single-member category maps to itself.
    solo = build_wrong_bank_permutation([0, 0, 1], seed=1, fold=0)
    assert solo[2] == 2


def test_global_bank_stats_match_hand_computed_fixture():
    mu = torch.tensor([[1.0, 0.0], [3.0, 0.0]])
    sigma = torch.ones(2, 2)
    count = torch.tensor([1, 3], dtype=torch.int64)
    gmu, gsigma, gcount = compute_global_bank_stats(mu, sigma, count)
    assert gcount == 4
    assert torch.allclose(gmu, torch.tensor([2.5, 0.0]), atol=1e-5)
    # pooled variance dim0: (1*(1+1) + 3*(1+9))/4 - 2.5^2 = 8 - 6.25 = 1.75
    assert torch.allclose(gsigma, torch.tensor([1.75 ** 0.5, 1.0]), atol=1e-5)


def test_no_bank_preserves_relation_width():
    """no_bank neutralizes bank inputs but keeps the 770-dim relation."""
    from stage2.relation import TrialRelationBlock

    neutral = TrialRelationBlock(active=False).eval()
    q0 = torch.randn(4, 128)
    mu = torch.randn(4, 128)
    sigma = torch.rand(4, 128) + 0.5
    count = torch.tensor([8, 8, 8, 8])
    out_neutral = neutral(q0)
    out_ignored = neutral(q0, mu, sigma, count)  # bank tensors are ignored
    assert torch.allclose(out_ignored["z_trial"], out_neutral["z_trial"])
    # The neutral path equals an active block fed exactly zero mean / unit
    # sigma / unit count — identical weights, identical relation width.
    active = TrialRelationBlock(active=True).eval()
    active.load_state_dict(neutral.state_dict())
    out_active = active(q0, mu, sigma, count)
    manual = active(q0, torch.zeros_like(q0), torch.ones_like(q0),
                    torch.ones(4, dtype=torch.int64))
    assert out_active["z_trial"].shape == out_neutral["z_trial"].shape
    for key in ("q", "n_mu", "uncertainty_context", "rho", "cosine", "comparator"):
        assert out_active[key].shape == out_neutral[key].shape
    assert torch.allclose(out_neutral["z_trial"], manual["z_trial"])
    assert torch.allclose(out_neutral["cosine"], manual["cosine"])
    # The neutral cosine is deterministic and independent of any bank value:
    # different supplied bank rows produce bitwise-identical outputs.
    assert torch.equal(out_neutral["cosine"], out_ignored["cosine"])
    assert torch.equal(out_neutral["z_trial"], out_ignored["z_trial"])


def test_no_bank_never_reads_bank_values(tmp_path):
    stack = make_full_model_fixture(tmp_path, build_model=False)
    cfg = resolve_ablation_config(BASE_CONFIG_DEFAULT, "no_bank", fold=0).config
    model = Stage2Model(cfg, stack["bank_store"])
    calls: list[int] = []
    original = stack["bank_store"].gather_trials

    def spy(*args, **kwargs):
        calls.append(1)
        raise AssertionError("no_bank must not gather bank values")

    stack["bank_store"].gather_trials = spy
    batch = make_synthetic_batch(4, valid_counts=[8, 8, 8, 8], seed=1,
                                 bank_split_ids=[0, 1, 0, 1])
    result = model(batch, "train")
    assert calls == []
    assert result.full.main_logit.shape == (4,)
    assert torch.isfinite(result.full.main_logit).all()
    assert model.matching_inputs(batch) is None
    stack["bank_store"].gather_trials = original


def test_unfreeze_last_block_exposes_exactly_last_block(tmp_path):
    stack = make_full_model_fixture(tmp_path, build_model=False)
    cfg = resolve_ablation_config(BASE_CONFIG_DEFAULT, "unfreeze_last_block", fold=0).config
    assert cfg.loss.lambda_anchor > 0.0
    model = Stage2Model(cfg, stack["bank_store"])
    trainable = [
        name for name, p in model.transferred_encoder.encoder.named_parameters()
        if p.requires_grad
    ]
    assert trainable, "last block must be trainable"
    assert all(name.startswith("residual_blocks.1.") for name in trainable)
    for name, p in model.transferred_encoder.encoder.named_parameters():
        if not name.startswith("residual_blocks.1."):
            assert not p.requires_grad, name
    current, reference = model.transferred_encoder.anchor_vectors()
    assert current.numel() > 0 and current.shape == reference.shape
    assert torch.allclose(current, reference)  # untouched Stage-1 weights


def test_mean_query_pooling_is_uniform_mean():
    pooler = MeanQueryPooler()
    tokens = torch.randn(3, 192, 128)
    weights, q0 = pooler(tokens)
    assert torch.allclose(weights, torch.full((3, 192), 1.0 / 192.0))
    # float32 sum vs mean differ by reduction rounding (~1e-7): absolute tol.
    assert torch.allclose(q0, tokens.mean(dim=1), atol=1e-4, rtol=1e-4)


def test_no_category_balance_uses_one_global_softmax():
    # Categories with unequal sizes: the unbalanced softmax mass follows the
    # categories' trial counts rather than equal per-category mass.
    from stage2.subject_aggregation import CategoryBalancedGatedAttention

    balanced = CategoryBalancedGatedAttention(balanced=True)
    unbalanced = CategoryBalancedGatedAttention(balanced=False)
    # Zero tokens make every gated score equal: softmax mass is exactly
    # proportional to trial counts (deterministic comparison).
    z = torch.zeros(1, 100, 128)
    cats = (torch.arange(100) % 4).unsqueeze(0)
    mask = torch.zeros(1, 100, dtype=torch.bool)
    mask[0, :10] = True  # categories 0,1 have 3 trials; categories 2,3 have 2
    _, imp_bal, _, _ = balanced(z, cats, mask)
    _, imp_unbal, _, _ = unbalanced(z, cats, mask)
    assert torch.allclose((imp_bal * mask).sum(dim=1), torch.ones(1), atol=1e-4)
    assert torch.allclose((imp_unbal * mask).sum(dim=1), torch.ones(1), atol=1e-4)
    for k in range(4):
        assert torch.allclose(
            imp_bal[0, mask[0] & (cats[0] == k)].sum(), torch.tensor(0.25), atol=1e-4
        )
    # Unbalanced: category 0 (3 trials) receives more mass than category 3 (2).
    mass0 = imp_unbal[0, mask[0] & (cats[0] == 0)].sum()
    mass3 = imp_unbal[0, mask[0] & (cats[0] == 3)].sum()
    assert mass0 > mass3


def test_mean_subject_pooling_equals_mean_of_present_category_tokens():
    from stage2.subject_aggregation import SubjectTransformerAggregator

    agg0 = SubjectTransformerAggregator(layers=0)
    agg1 = SubjectTransformerAggregator(layers=1)
    tokens = torch.randn(2, 4, 128)
    present = torch.tensor([[True, True, True, True], [True, True, False, True]])
    subject0, out0 = agg0(tokens, present)
    expected = torch.stack(
        [tokens[0].mean(dim=0), tokens[1][present[1]].mean(dim=0)]
    )
    assert torch.allclose(subject0, expected)
    assert out0.shape == (2, 5, 128)
    assert torch.allclose(out0[:, 0], subject0)
    assert not hasattr(agg0, "layer")  # no Transformer parameters exist
    subject1, out1 = agg1(tokens, present)
    assert out1.shape == (2, 5, 128)


# ------------------------------------------------------- model-level controls


def _make_enc_for_aggregator():
    batch = 2
    slots = list(range(12))
    subj = torch.tensor([i for i in range(batch) for _ in slots], dtype=torch.int64)
    stim = torch.tensor(slots * batch, dtype=torch.int64)
    n = subj.numel()
    mask = torch.zeros(batch, 100, dtype=torch.bool)
    mask[subj, stim] = True
    return EncodedTrials(
        batch_size=batch, subject_slots=subj, stimulus_slots=stim,
        category_ids=stim % 4, trial_mask=mask,
        category_ids_panel=(torch.arange(100) % 4).expand(batch, -1).clone(),
        heatmap_tokens=torch.randn(n, 192, 128),
        patch_attention=torch.rand(n, 192).softmax(-1),
        q0=torch.randn(n, 128), q=torch.randn(n, 128), n_mu=torch.randn(n, 128),
        uncertainty_context=torch.randn(n, 128), rho=torch.rand(n, 1),
        cosine=torch.rand(n, 1) * 0.5, z_trial=torch.randn(n, 128),
        comparator=torch.randn(n), bank_ids=torch.zeros(n, dtype=torch.int64),
    )


def test_missing_trials_zero_attention_for_mean_variants():
    enc = _make_enc_for_aggregator()
    for layers, balanced in ((0, True), (0, False), (1, True), (1, False)):
        agg = SubjectAggregator(
            transformer_layers=layers, category_balanced=balanced
        )
        out = agg(enc)
        assert (out.stimulus_attention[~out.trial_mask] == 0).all(), (layers, balanced)
        assert (out.stimulus_importance[~out.trial_mask] == 0).all(), (layers, balanced)
        assert torch.isfinite(out.main_logit).all()


def test_wrong_bank_model_uses_the_permutation(tmp_path):
    stack = make_full_model_fixture(tmp_path, build_model=False)
    cfg = resolve_ablation_config(
        BASE_CONFIG_DEFAULT, "wrong_stimulus_bank", fold=0
    ).config
    model = Stage2Model(cfg, stack["bank_store"])
    perm = model.wrong_bank_permutation
    assert perm is not None
    # Distinctive orthogonal bank rows: the gathered mean must be the row of
    # the permuted stimulus.
    distinctive = torch.eye(100, 128) * 5.0
    store: NormativeBankStore = stack["bank_store"]
    store.mu_trial = distinctive.clone()
    for j in store.split_tensors:
        store.split_tensors[j]["mu_trial"] = distinctive.clone()
    batch = make_synthetic_batch(2, valid_counts=[8, 8], seed=2,
                                 labels=[0.0, 0.0], bank_split_ids=[0, 1])
    enc = model.encode_trials(batch, "train")
    from stage2.collate import flatten_valid_trials

    flat = flatten_valid_trials(batch)
    expected_rows = distinctive[[perm[int(s)] for s in flat.stimulus_indices.tolist()]]
    assert torch.allclose(enc.n_mu, model.relation.bank_mean_adapter(expected_rows))
    # The wrong-bank mapping is applied to matching inputs as well.
    match = model.matching_inputs(batch, enc=enc, epoch=0)
    assert match is not None
    assert not torch.allclose(match.cos_pos, match.cos_neg)


def test_global_bank_model_repeats_one_global_entry(tmp_path):
    stack = make_full_model_fixture(tmp_path, build_model=False)
    cfg = resolve_ablation_config(BASE_CONFIG_DEFAULT, "global_bank", fold=0).config
    model = Stage2Model(cfg, stack["bank_store"])
    batch = make_synthetic_batch(2, valid_counts=[8, 8], seed=3,
                                 labels=[0.0, 0.0], bank_split_ids=[0, 1])
    enc = model.encode_trials(batch, "train")
    store: NormativeBankStore = stack["bank_store"]
    # Every trial of a given bank id sees the identical global row.
    for bank_id in enc.bank_ids.unique().tolist():
        rows = enc.bank_ids == bank_id
        if bank_id == FULL_BANK_ID:
            mu, sigma, count = store.mu_trial, store.sigma_trial, store.count_trial
        else:
            arr = store.split_tensors[int(bank_id)]
            mu, sigma, count = arr["mu_trial"], arr["sigma_trial"], arr["count_trial"]
        gmu, gsigma, gcount = compute_global_bank_stats(mu, sigma, count)
        assert torch.allclose(enc.n_mu[rows], model.relation.bank_mean_adapter(gmu.expand(int(rows.sum()), -1)))
        assert enc.n_mu[rows].unique(dim=0).shape[0] == 1
        assert int(enc.bank_ids[rows][0]) == bank_id


def test_every_runnable_ablation_forward_backward_finite(tmp_path):
    stack = make_full_model_fixture(tmp_path, include_token=True)
    store = stack["bank_store"]
    batch = make_synthetic_batch(4, valid_counts=[8, 8, 8, 8], seed=4,
                                 labels=[0.0, 0.0, 1.0, 1.0], bank_split_ids=[0, 1, 0, 1])
    for name in RUNNABLE:
        resolved = resolve_ablation_config(BASE_CONFIG_DEFAULT, name, fold=0)
        validate_bank_capabilities(resolved.spec, store)
        cfg = resolved.config
        model = Stage2Model(cfg, store)
        subset_masks = None
        if cfg.subsets.enabled:
            subset_masks = generate_subset_masks(
                trial_mask=batch.trial_mask, category_ids=batch.category_ids,
                subject_ids=list(batch.subject_ids), seed=cfg.seed, fold=0,
                epoch=0, train=True,
            )
        result = model(batch, "train", subset_masks=subset_masks)
        enc = model.encode_trials(batch, "train")
        match_inputs = model.matching_inputs(batch, enc=enc, epoch=0)
        anchor_current, anchor_stage1 = model.transferred_encoder.anchor_vectors()
        if cfg.loss.lambda_anchor == 0.0 or anchor_current.numel() == 0:
            anchor_current = anchor_stage1 = None
        losses = compute_stage2_losses(
            loss_cfg=cfg.loss, subsets_cfg=cfg.subsets, labels=batch.labels,
            full=result.full, subsets=result.subsets, match_inputs=match_inputs,
            anchor_current=anchor_current, anchor_stage1=anchor_stage1, epoch=0,
        )
        assert torch.isfinite(losses.total), name
        losses.total.backward()
        grads_finite = all(
            p.grad is None or torch.isfinite(p.grad).all()
            for p in model.parameters() if p.requires_grad
        )
        assert grads_finite, name
        # Missing trials carry zero attention under every variant.
        assert (result.full.stimulus_attention[~batch.trial_mask] == 0).all(), name
        assert (result.full.stimulus_importance[~batch.trial_mask] == 0).all(), name
        if model.token_branch is not None:
            assert result.full.semantic_patch_map is not None
            assert result.full.semantic_patch_map.shape == (4, 100, 12, 16)
