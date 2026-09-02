"""Balanced subject sampler tests (guide 04 §12)."""

from __future__ import annotations

import numpy as np
import pytest

from stage2.sampler import BalancedSubjectBatchSampler, SamplerError


class FakeDataset:
    def __init__(self, labels: list[int]):
        self._labels = labels

    def __len__(self) -> int:
        return len(self._labels)

    def subject_labels(self) -> list[int]:
        return self._labels


def balanced_labels(n_hc: int = 64, n_sz: int = 64) -> list[int]:
    return [0] * n_hc + [1] * n_sz


def all_batches(sampler: BalancedSubjectBatchSampler) -> list[list[int]]:
    return [batch for batch in sampler]


class TestBalancedComposition:
    def test_perfect_pairs(self):
        ds = FakeDataset(balanced_labels(64, 64))
        sampler = BalancedSubjectBatchSampler(ds, batch_size=4, seed=2026, fold=0)
        batches = all_batches(sampler)
        assert len(batches) == 32
        assert all(len(b) == 4 for b in batches)
        assert all(comp == (2, 2) for comp in sampler.last_batch_compositions)

    def test_every_subject_exactly_once(self):
        ds = FakeDataset(balanced_labels(64, 64))
        sampler = BalancedSubjectBatchSampler(ds, batch_size=4, seed=2026, fold=0)
        seen = [i for b in all_batches(sampler) for i in b]
        assert sorted(seen) == list(range(128))

    def test_batch_size_8_uses_four_per_group(self):
        ds = FakeDataset(balanced_labels(64, 64))
        sampler = BalancedSubjectBatchSampler(ds, batch_size=8, seed=2026, fold=1)
        batches = all_batches(sampler)
        assert all(comp == (4, 4) for comp in sampler.last_batch_compositions)

    def test_odd_batch_size_rejected_in_balanced_mode(self):
        ds = FakeDataset(balanced_labels(8, 8))
        with pytest.raises(SamplerError, match="even"):
            BalancedSubjectBatchSampler(ds, batch_size=3, seed=1, fold=0)

    def test_subject_indices_only(self):
        ds = FakeDataset(balanced_labels(10, 10))
        sampler = BalancedSubjectBatchSampler(ds, batch_size=4, seed=5, fold=2)
        for batch in all_batches(sampler):
            assert all(isinstance(i, (int, np.integer)) and 0 <= i < 20 for i in batch)

    def test_imbalanced_population_not_discarded(self):
        # 10 HC + 2 SZ: the final batches top up from the HC side; nothing is dropped.
        ds = FakeDataset([0] * 10 + [1] * 2)
        sampler = BalancedSubjectBatchSampler(ds, batch_size=4, seed=3, fold=0)
        seen = [i for b in all_batches(sampler) for i in b]
        assert sorted(seen) == list(range(12))

    def test_drop_last(self):
        ds = FakeDataset(balanced_labels(5, 5))
        sampler = BalancedSubjectBatchSampler(ds, batch_size=4, seed=1, fold=0, drop_last=True)
        batches = all_batches(sampler)
        assert all(len(b) == 4 for b in batches)
        assert len(sampler) == len(batches) == 2


class TestDeterminism:
    def test_same_seed_epoch_deterministic(self):
        ds = FakeDataset(balanced_labels(64, 64))
        a = BalancedSubjectBatchSampler(ds, batch_size=4, seed=2026, fold=0, epoch=0)
        b = BalancedSubjectBatchSampler(ds, batch_size=4, seed=2026, fold=0, epoch=0)
        assert all_batches(a) == all_batches(b)

    def test_next_epoch_different_order(self):
        ds = FakeDataset(balanced_labels(64, 64))
        sampler = BalancedSubjectBatchSampler(ds, batch_size=4, seed=2026, fold=0, epoch=0)
        first_epoch = all_batches(sampler)
        sampler.set_epoch(1)
        second_epoch = all_batches(sampler)
        assert first_epoch != second_epoch
        assert sorted(i for b in second_epoch for i in b) == list(range(128))

    def test_state_dict_round_trip(self):
        ds = FakeDataset(balanced_labels(64, 64))
        a = BalancedSubjectBatchSampler(ds, batch_size=4, seed=2026, fold=0, epoch=7)
        b = BalancedSubjectBatchSampler(ds, batch_size=4, seed=2026, fold=0, epoch=0)
        b.load_state_dict(a.state_dict())
        assert all_batches(a) == all_batches(b)

    def test_fixed_validation_order(self):
        ds = FakeDataset(balanced_labels(3, 3))
        sampler = BalancedSubjectBatchSampler(ds, batch_size=4, seed=2026, fold=0, shuffle=False)
        batches = all_batches(sampler)
        assert batches == [[0, 1, 2, 3], [4, 5]]
        # Deterministic across instances.
        sampler2 = BalancedSubjectBatchSampler(ds, batch_size=4, seed=2026, fold=0, shuffle=False)
        assert all_batches(sampler2) == batches

    def test_len_matches_iteration(self):
        for (n_hc, n_sz, bs) in [(64, 64, 4), (64, 64, 8), (10, 2, 4), (5, 5, 4), (7, 3, 6)]:
            ds = FakeDataset(balanced_labels(n_hc, n_sz))
            sampler = BalancedSubjectBatchSampler(ds, batch_size=bs, seed=1, fold=0)
            assert len(sampler) == len(all_batches(sampler)), (n_hc, n_sz, bs)

    def test_no_balance_mode(self):
        ds = FakeDataset(balanced_labels(5, 5))
        sampler = BalancedSubjectBatchSampler(
            ds, batch_size=4, seed=1, fold=0, balance_groups=False
        )
        seen = [i for b in all_batches(sampler) for i in b]
        assert sorted(seen) == list(range(10))
