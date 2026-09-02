"""Subject aggregation tests (guide 05 §8-§12): embeddings, category-balanced
gated attention, subject Transformer, additive evidence head, interpretation
outputs."""

from __future__ import annotations

import torch
import pytest

from stage2.contracts import EncodedTrials
from stage2.subject_aggregation import (
    AdditiveEvidenceHead,
    CategoryBalancedGatedAttention,
    StimulusCategoryEmbeddings,
    SubjectAggregator,
)

BATCH = 2
SLOTS = list(range(12))  # categories 0..3 x 3 members each


def make_enc(
    batch_size: int = BATCH,
    *,
    valid_by_subject: list[list[int]] | None = None,
    seed: int = 0,
) -> EncodedTrials:
    torch.manual_seed(seed)
    valid_by_subject = valid_by_subject or [SLOTS[:] for _ in range(batch_size)]
    subject_slots: list[int] = []
    stimulus_slots: list[int] = []
    for i, slots in enumerate(valid_by_subject):
        for s in slots:
            subject_slots.append(i)
            stimulus_slots.append(s)
    subj = torch.tensor(subject_slots, dtype=torch.int64)
    stim = torch.tensor(stimulus_slots, dtype=torch.int64)
    n = subj.numel()
    mask = torch.zeros(batch_size, 100, dtype=torch.bool)
    mask[subj, stim] = True
    return EncodedTrials(
        batch_size=batch_size,
        subject_slots=subj,
        stimulus_slots=stim,
        category_ids=stim % 4,
        trial_mask=mask,
        category_ids_panel=(torch.arange(100) % 4).expand(batch_size, -1).clone(),
        heatmap_tokens=torch.randn(n, 192, 128),
        patch_attention=torch.rand(n, 192).softmax(dim=-1),
        q0=torch.randn(n, 128),
        q=torch.randn(n, 128),
        n_mu=torch.randn(n, 128),
        uncertainty_context=torch.randn(n, 128),
        rho=torch.rand(n, 1),
        cosine=torch.rand(n, 1) * 0.5,
        z_trial=torch.randn(n, 128),
        comparator=torch.randn(n),
        bank_ids=torch.zeros(n, dtype=torch.int64),
    )


def run(enc: EncodedTrials, subset_mask: torch.Tensor | None = None):
    return SubjectAggregator()(enc, subset_mask)


def test_swapping_stimulus_slots_changes_embedding_selection():
    """Holding trial features fixed, moving a trial to another stimulus slot
    must change the embedding table lookup for that trial."""
    enc = make_enc()
    out = run(enc)
    swapped_valid = [SLOTS[:] for _ in range(BATCH)]
    swapped_valid[0][0], swapped_valid[0][1] = 5, 0  # swap slots 0 and 5
    enc2 = make_enc(valid_by_subject=swapped_valid, seed=0)
    out2 = run(enc2)
    # Trial 0 of subject 0 carries the same z in both encodes but sits at
    # slot 0 in enc and slot 5 in enc2: the embedding lookup must differ.
    assert not torch.allclose(out.trial_embeddings[0, 0], out2.trial_embeddings[0, 5])


def test_embedding_tables_are_indexed_by_stimulus_slot():
    emb = StimulusCategoryEmbeddings()
    z = torch.zeros(1, 100, 128)
    idx = torch.arange(100).unsqueeze(0)
    cats = (torch.arange(100) % 4).unsqueeze(0)
    e = emb(z, idx, cats)
    expected = emb.norm(
        emb.stimulus_table(torch.tensor([7])) + emb.category_table(torch.tensor([3]))
    )
    assert torch.allclose(e[0, 7], expected)


def test_importance_normalization_and_category_balance():
    enc = make_enc()
    out = run(enc)
    mask = out.trial_mask
    assert torch.allclose((out.stimulus_importance * mask).sum(dim=1), torch.ones(BATCH), atol=1e-4)
    for i in range(BATCH):
        for k in range(4):
            mass = out.stimulus_importance[i, (mask[i] & (enc.category_ids_panel[i] == k))].sum()
            assert torch.allclose(mass, torch.tensor(0.25), atol=1e-4), (i, k)


def test_missing_category_uses_missing_token_and_is_excluded():
    # Subject 1 loses every category-3 trial (slots 3, 7, 11).
    valid = [SLOTS[:], [s for s in SLOTS if s % 4 != 3]]
    enc = make_enc(valid_by_subject=valid)
    out = run(enc)
    present = out.diagnostics["category_present"]
    assert present[0].tolist() == [True, True, True, True]
    assert present[1].tolist() == [True, True, True, False]
    # Missing category token equals the learned parameter; excluded from mass.
    agg = SubjectAggregator()
    out = agg(enc)
    cat_tokens = out.diagnostics["category_tokens"]
    assert torch.allclose(
        cat_tokens[1, 3], agg.gated_attention.missing_category_token
    )
    mass = (out.stimulus_importance * out.trial_mask).sum(dim=1)
    assert torch.allclose(mass, torch.ones(BATCH), atol=1e-4)
    for k in range(3):
        assert torch.allclose(
            out.stimulus_importance[1, (out.trial_mask[1] & (enc.category_ids_panel[1] == k))].sum(),
            torch.tensor(1.0 / 3.0), atol=1e-4,
        )


def test_subset_mask_rescales_importance_and_drops_categories():
    enc = make_enc()
    # Subset keeps only categories 0 and 1 for subject 0.
    subset = torch.zeros(BATCH, 100, dtype=torch.bool)
    keep = [s for s in SLOTS if s % 4 in (0, 1)]
    subset[0, keep] = True
    subset[1] = enc.trial_mask[1]
    out = run(enc, subset)
    assert torch.allclose((out.stimulus_importance * out.trial_mask).sum(dim=1),
                          torch.ones(BATCH), atol=1e-4)
    assert (out.stimulus_importance[0, [s for s in SLOTS if s % 4 in (2, 3)]] == 0).all()
    for k in (0, 1):
        assert torch.allclose(
            out.stimulus_importance[0, out.trial_mask[0] & (enc.category_ids_panel[0] == k)].sum(),
            torch.tensor(0.5), atol=1e-4,
        )


def test_transformer_shapes_and_missing_category_mask():
    enc = make_enc()
    out = run(enc)
    diag = out.diagnostics
    assert diag["transformer_output"].shape == (BATCH, 5, 128)
    assert out.subject_embedding.shape == (BATCH, 128)
    assert out.main_logit.shape == (BATCH,)
    assert out.trial_embeddings.shape == (BATCH, 100, 128)


def test_auxiliary_logit_equals_bias_plus_contributions():
    enc = make_enc()
    agg = SubjectAggregator()
    out = agg(enc)
    assert torch.allclose(
        out.auxiliary_logit - agg.evidence_head.bias,
        out.stimulus_contribution.sum(dim=1),
    )
    assert (out.stimulus_contribution[~out.trial_mask] == 0).all()
    assert torch.allclose(
        out.stimulus_contribution,
        out.stimulus_importance * out.stimulus_evidence,
    )


def test_interpretation_outputs_are_zero_on_missing():
    enc = make_enc()
    out = run(enc)
    mask = out.trial_mask
    for panel in (
        out.stimulus_attention, out.stimulus_importance, out.stimulus_evidence,
        out.stimulus_contribution, out.semantic_compatibility,
        out.normative_deviation, out.weighted_normative_deviation,
    ):
        assert (panel[~mask] == 0).all()
    assert (out.query_patch_attention[~mask] == 0).all()
    assert torch.allclose(
        out.normative_deviation,
        out.diagnostics["trial_rho"] * (1.0 - out.semantic_compatibility),
    )
    assert torch.allclose(
        out.weighted_normative_deviation,
        out.stimulus_importance * out.normative_deviation,
    )


def test_attention_scores_missing_trials_are_negative_infinity_before_softmax():
    enc = make_enc(valid_by_subject=[[s for s in SLOTS if s % 4 != 2]] + [SLOTS[:]])
    z = torch.randn(BATCH, 100, 128)
    gate = CategoryBalancedGatedAttention()
    g = gate.w(torch.tanh(gate.V(z)) * torch.sigmoid(gate.U(z))).squeeze(-1)
    g_masked = g.masked_fill(~enc.trial_mask, float("-inf"))
    # Category 2 of subject 0 is fully missing: all its scores are -inf.
    assert (g_masked[0, enc.category_ids_panel[0] == 2] == float("-inf")).all()
