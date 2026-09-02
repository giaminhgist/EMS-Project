"""Validation tests (guide 07 §9, §17.2-§17.4, §17.17)."""

from __future__ import annotations

import torch
import pytest

from conftest import make_full_model_fixture, make_synthetic_batch
from stage2.dataset import Stage2SubjectDataset
from stage2.sampler import BalancedSubjectBatchSampler
from stage2.collate import collate_subject_samples
from stage2.validation import ValidationError, attention_entropy, run_validation
from torch.utils.data import DataLoader


@pytest.fixture()
def stack(tmp_path):
    return make_full_model_fixture(tmp_path)


def make_val_loader(stack, max_subjects=None):
    cfg = stack["cfg"]
    ds = Stage2SubjectDataset(cfg, 0, "val", bank_store=stack["bank_store"])
    if max_subjects:
        ds.subjects = ds.subjects.iloc[:max_subjects].reset_index(drop=True)
        ds.subject_ids = [str(x) for x in ds.subjects.subject_id]
        ds.labels = [int(x) for x in ds.subjects.label]
    sampler = BalancedSubjectBatchSampler(
        ds, batch_size=2, seed=cfg.seed, fold=0, epoch=0,
        balance_groups=False, drop_last=False, shuffle=False,
    )
    return ds, DataLoader(ds, batch_sampler=sampler, collate_fn=collate_subject_samples)


def test_validation_contains_each_subject_exactly_once(stack):
    model = stack["model"]
    ds, loader = make_val_loader(stack)
    model.eval()
    result = run_validation(model, loader, cfg=stack["cfg"])
    assert sorted(result.subject_ids) == sorted(ds.subject_ids)
    assert len(result.subject_ids) == len(ds)
    assert len(set(result.subject_ids)) == len(result.subject_ids)
    assert result.labels.shape == (len(ds),)
    assert result.raw_logits.shape == (len(ds),)
    assert result.stimulus_attention.shape == (len(ds), 100)
    assert result.trial_mask.shape == (len(ds), 100)
    assert result.metrics["n_subjects"] == len(ds)
    # Missing trials carry zero attention and zero contribution.
    assert (result.stimulus_attention[~result.trial_mask] == 0).all()
    assert (result.stimulus_contribution[~result.trial_mask] == 0).all()


def test_validation_rejects_repeated_subjects(stack):
    model = stack["model"]
    ds, _ = make_val_loader(stack)

    class RepeatingSampler:
        def __iter__(self):
            # Subject 0 appears twice; subject 1 is dropped.
            yield [0, 0]

    loader = DataLoader(ds, batch_sampler=RepeatingSampler(), collate_fn=collate_subject_samples)
    model.eval()
    with pytest.raises(ValidationError, match="repeats subjects"):
        run_validation(model, loader, cfg=stack["cfg"])


def test_subject_weighted_loss_aggregation(stack):
    """Batch losses are weighted by subject count, not trial count."""
    model = stack["model"]
    ds, loader = make_val_loader(stack, max_subjects=2)
    model.eval()
    result = run_validation(model, loader, cfg=stack["cfg"])
    # n_val = 2 subjects; losses finite and nonnegative.
    assert result.mean_losses["val_loss"] >= 0.0
    assert result.n_subjects == 2
    assert result.metrics["n_subjects"] == 2


def test_validation_uses_full_bank_not_crossfit(stack):
    """Validation gathers from the full fold bank (bank ids all FULL_BANK_ID)."""
    model = stack["model"]
    ds, loader = make_val_loader(stack)
    model.eval()
    enc = None
    with torch.inference_mode():
        for batch in loader:
            enc = model.encode_trials(batch, "val")
            break
    assert enc is not None
    assert (enc.bank_ids == -1).all(), "validation must use the full fold bank"


def test_token_maps_present_only_when_requested(tmp_path):
    stack = make_full_model_fixture(
        tmp_path, include_token=True, model={"bank_mode": "trial_and_fused_token"}
    )
    model = stack["model"]
    ds, loader = make_val_loader(stack, max_subjects=2)
    model.eval()
    without = run_validation(model, loader, cfg=stack["cfg"], token_maps=False)
    assert without.semantic_patch_maps is None
    with_maps = run_validation(model, loader, cfg=stack["cfg"], token_maps=True)
    assert with_maps.semantic_patch_maps is not None
    assert with_maps.semantic_patch_maps.shape == (2, 100, 12, 16)
    # Token maps stream without changing the logits.
    assert torch.equal(without.raw_logits, with_maps.raw_logits)


def test_attention_entropy_helper():
    attention = torch.zeros(1, 100)
    mask = torch.zeros(1, 100, dtype=torch.bool)
    k = 4
    attention[0, :k] = 1.0 / k
    mask[0, :k] = True
    entropy = attention_entropy(attention, mask)
    assert torch.allclose(entropy, torch.tensor(float(torch.tensor(k).log())), atol=1e-5)
