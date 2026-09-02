"""Collation, batch validation and flatten/scatter tests (guide 04 §6-§8, §12)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from stage2.collate import (
    Stage2Batch,
    collate_subject_samples,
    flatten_valid_trials,
    scatter_valid_trials,
)
from stage2.contracts import N_STIMULI, Stage2SubjectSample


def make_sample(subject_id: str, label: int, observed: list[int], seed: int = 0) -> Stage2SubjectSample:
    rng = np.random.default_rng(seed)
    heatmaps = np.zeros((N_STIMULI, 3, 48, 64), np.float32)
    mask = np.zeros(N_STIMULI, bool)
    uids: list[str | None] = [None] * N_STIMULI
    for si in observed:
        heatmaps[si] = rng.normal(size=(3, 48, 64)).astype(np.float32)
        mask[si] = True
        uids[si] = f"{subject_id}_{si:03d}"
    return Stage2SubjectSample(
        subject_id=subject_id,
        label=label,
        heatmaps=torch.from_numpy(heatmaps),
        trial_mask=torch.from_numpy(mask),
        stimulus_indices=torch.arange(N_STIMULI, dtype=torch.int64),
        category_ids=torch.tensor([si % 4 for si in range(N_STIMULI)], dtype=torch.int64),
        trial_uids=tuple(uids),
        bank_split_id=label,  # 0 for HC, 1 for SZ, None if label None
    )


def make_batch() -> Stage2Batch:
    samples = [
        make_sample("000", 0, list(range(100)), seed=0),  # complete panel
        make_sample("001", 0, [0, 2, 5, 99], seed=1),
        make_sample("101", 1, list(range(50)), seed=2),
        make_sample("103", 1, [7, 8, 9, 10, 11, 12], seed=3),
    ]
    return collate_subject_samples(samples)


class TestCollate:
    def test_shapes_and_validation(self):
        batch = make_batch()
        batch.validate()
        assert batch.n_subjects == 4
        assert batch.n_valid_trials == 100 + 4 + 50 + 6
        assert batch.heatmaps.shape == (4, 100, 3, 48, 64)
        assert batch.labels.tolist() == [0.0, 0.0, 1.0, 1.0]
        assert batch.bank_split_ids is not None
        assert batch.bank_split_ids.tolist() == [0, 0, 1, 1]
        assert len(batch.trial_uids) == 4 and all(len(u) == 100 for u in batch.trial_uids)

    def test_all_none_bank_split_ids(self):
        samples = [make_sample("000", 0, [0, 1], seed=0), make_sample("001", 0, [2, 3], seed=1)]
        for s in samples:
            s = Stage2SubjectSample(**{**s.__dict__, "bank_split_id": None})
        batch = collate_subject_samples(
            [
                Stage2SubjectSample(
                    subject_id=s.subject_id,
                    label=s.label,
                    heatmaps=s.heatmaps,
                    trial_mask=s.trial_mask,
                    stimulus_indices=s.stimulus_indices,
                    category_ids=s.category_ids,
                    trial_uids=s.trial_uids,
                    bank_split_id=None,
                )
                for s in samples
            ]
        )
        assert batch.bank_split_ids is None
        batch.validate()

    def test_mixed_bank_split_ids_rejected(self):
        base = make_sample("000", 0, [0], seed=0)
        with_none = Stage2SubjectSample(
            subject_id="001",
            label=0,
            heatmaps=base.heatmaps,
            trial_mask=base.trial_mask,
            stimulus_indices=base.stimulus_indices,
            category_ids=base.category_ids,
            trial_uids=base.trial_uids,
            bank_split_id=None,
        )
        with pytest.raises(ValueError, match="mixed"):
            collate_subject_samples([base, with_none])

    def test_validate_rejects_noncanonical_slots(self):
        batch = make_batch()
        batch.stimulus_indices = torch.zeros_like(batch.stimulus_indices)
        with pytest.raises(ValueError, match="canonical"):
            batch.validate()

    def test_validate_rejects_subject_without_trials(self):
        batch = make_batch()
        batch.trial_mask[0, :] = False
        with pytest.raises(ValueError, match="at least one valid trial"):
            batch.validate()


class TestFlattenScatter:
    def test_complete_panel_flatten(self):
        sample = make_sample("000", 0, list(range(100)), seed=0)
        batch = collate_subject_samples([sample])
        flat = flatten_valid_trials(batch)
        assert flat.heatmaps.shape == (100, 3, 48, 64)
        assert flat.subject_slots.tolist() == [0] * 100
        assert flat.stimulus_slots.tolist() == list(range(100))
        assert flat.stimulus_indices.tolist() == list(range(100))

    def test_arbitrary_missing_mask_and_row_major_order(self):
        batch = make_batch()
        flat = flatten_valid_trials(batch)
        n = batch.n_valid_trials
        assert flat.heatmaps.shape[0] == n
        # Row-major [B,100] ordering from torch.nonzero.
        mask = batch.trial_mask.numpy()
        rows = np.argwhere(mask)
        assert flat.subject_slots.tolist() == rows[:, 0].tolist()
        assert flat.stimulus_slots.tolist() == rows[:, 1].tolist()

    def test_no_cross_subject_mixing(self):
        batch = make_batch()
        flat = flatten_valid_trials(batch)
        for k in range(4):
            src = batch.heatmaps[k][batch.trial_mask[k]]
            sub = flat.subject_slots == k
            assert torch.equal(flat.heatmaps[sub], src)

    def test_exact_round_trip(self):
        batch = make_batch()
        flat = flatten_valid_trials(batch)
        padded, mask = scatter_valid_trials(
            flat.heatmaps, flat.subject_slots, flat.stimulus_slots, batch.n_subjects
        )
        assert padded.shape == (4, 100, 3, 48, 64)
        assert torch.equal(mask, batch.trial_mask)
        assert torch.equal(padded[mask], flat.heatmaps)
        # Missing slots remain exactly zero.
        assert float(padded[~mask].abs().sum()) == 0.0

    def test_labels_per_trial_for_hc_selection(self):
        batch = make_batch()
        flat = flatten_valid_trials(batch)
        labels = batch.labels.long()
        assert flat.labels_per_trial.tolist() == labels[flat.subject_slots].tolist()
        assert set(flat.labels_per_trial.tolist()) <= {0, 1}

    def test_scatter_slot_length_mismatch(self):
        flat = make_batch().heatmaps[0:1]
        with pytest.raises(ValueError, match="slot tensors"):
            scatter_valid_trials(flat, torch.tensor([0]), torch.tensor([0, 1]), 4)

    def test_bank_split_ids_flattened(self):
        batch = make_batch()
        flat = flatten_valid_trials(batch)
        assert flat.bank_split_ids is not None
        assert flat.bank_split_ids.tolist() == batch.bank_split_ids[flat.subject_slots].tolist()
