"""Crossfit assignment and integration tests for the Stage-2 bank builder.

Integration fixture: 8 HC + 4 SZ synthetic subjects, 4 stimuli, 2 crossfit
splits; bank statistics are compared against direct NumPy calculations.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from stage2.bank_builder import (
    BankBuildError,
    BankConfig,
    assign_crossfit_splits,
    build_fold_banks,
)


def make_assignments(
    hc_ids: list[str], sz_ids: list[str], seed: int = 2026, fold: int = 0, n_splits: int = 2
) -> pd.DataFrame:
    rows = []
    for sid in hc_ids:
        rows.append({"subject_id": sid, "label": 0, "panel_trial_count": 4})
    for sid in sz_ids:
        rows.append({"subject_id": sid, "label": 1, "panel_trial_count": 3})
    train = pd.DataFrame(rows)
    return assign_crossfit_splits(
        train, seed=seed, fold=fold, n_splits=n_splits, forbidden_subject_ids=set()
    )


class TestCrossfitAssignment:
    def test_deterministic_for_same_seed_and_fold(self):
        hc = [f"{i:03d}" for i in range(8)]
        sz = [f"{i:03d}" for i in range(100, 104)]
        a1 = make_assignments(hc, sz)
        a2 = make_assignments(hc, sz)
        pd.testing.assert_frame_equal(a1, a2)

    def test_balanced_and_stratified(self):
        hc = [f"{i:03d}" for i in range(8)]
        sz = [f"{i:03d}" for i in range(100, 104)]
        a = make_assignments(hc, sz, n_splits=2)
        counts = a.groupby(["label", "bank_split_id"]).size().unstack(fill_value=0)
        assert list(counts.loc[0]) == [4, 4]  # 8 HC over 2 splits
        assert list(counts.loc[1]) == [2, 2]  # 4 SZ over 2 splits

    def test_completeness_balanced_when_feasible(self):
        hc = [f"{i:03d}" for i in range(8)]
        sz = [f"{i:03d}" for i in range(100, 104)]
        a = make_assignments(hc, sz, n_splits=2)
        for label in (0, 1):
            per_split = a[a.label == label].groupby("bank_split_id").panel_trial_count.sum()
            assert abs(per_split.iloc[0] - per_split.iloc[1]) <= 4

    def test_forbidden_subject_rejected(self):
        train = pd.DataFrame([{"subject_id": "000", "label": 0, "panel_trial_count": 4}])
        with pytest.raises(BankBuildError, match="validation subjects"):
            assign_crossfit_splits(
                train, seed=1, fold=0, n_splits=2, forbidden_subject_ids={"000"}
            )

    def test_hc_never_contributes_to_assigned_bank(self):
        hc = [f"{i:03d}" for i in range(8)]
        sz = [f"{i:03d}" for i in range(100, 104)]
        a = make_assignments(hc, sz, n_splits=2)
        hc_only = a[a.label == 0]
        for split in (0, 1):
            assigned = set(hc_only[hc_only.bank_split_id == split].subject_id)
            contributors = set(hc) - assigned
            assert not (assigned & contributors)


# ---------------------------------------------------------------- integration


class FakeModel:
    """Synthetic Stage-1-like model with deterministic per-row outputs."""

    def __init__(self, emb: np.ndarray, fused: np.ndarray | None = None):
        self.emb = emb
        self.fused = fused

    def __call__(self, batch: Any, token_mask: Any = None, **kwargs: Any) -> Any:
        rows = batch.indices
        return SimpleNamespace(
            trial_embedding=self.emb[rows],
            fused_tokens=self.fused[rows] if self.fused is not None else None,
            heatmap_tokens=None,
        )


class FakeDataset:
    """Duck-typed Stage-1 train dataset over synthetic trial rows."""

    def __init__(self, trial_rows: pd.DataFrame, emb: np.ndarray, fused: np.ndarray | None = None):
        self.trial_rows = trial_rows.reset_index(drop=True)
        self._model = FakeModel(emb, fused)

    def row_subject_ids(self) -> list[str]:
        return [str(x) for x in self.trial_rows.subject_id]

    def collate_from_indices(self, indices: list[int]) -> Any:
        return SimpleNamespace(indices=indices, to_device=lambda device: None)


def make_synthetic_trials(hc_ids: list[str], n_stimuli: int = 4) -> pd.DataFrame:
    rows = []
    for sid in hc_ids:
        for si in range(n_stimuli):
            rows.append(
                {
                    "subject_id": sid,
                    "stimulus_index": si,
                    "group": "HC",
                }
            )
    return pd.DataFrame(rows)


def embed_values(subject_idx: int, stimulus_index: int, dim: int) -> np.ndarray:
    """Deterministic embedding: value = subject*10 + stimulus + position*1e-3."""
    return (subject_idx * 10.0 + stimulus_index) + np.arange(dim) * 1e-3


def token_values(subject_idx: int, stimulus_index: int, n_cells: int, dim: int) -> np.ndarray:
    base = embed_values(subject_idx, stimulus_index, dim)
    return base[None, :] + np.arange(n_cells)[:, None] * 1e-5


class TestIntegration:
    def test_exact_means_full_and_crossfit(self):
        n_stimuli, dim, n_cells = 4, 8, 12
        hc_ids = [f"{i:03d}" for i in range(8)]
        sz_ids = [f"{i:03d}" for i in range(100, 104)]
        trials = make_synthetic_trials(hc_ids, n_stimuli)
        n = len(trials)
        emb = np.stack(
            [embed_values(int(r.subject_id), int(r.stimulus_index), dim) for r in trials.itertuples()]
        ).astype(np.float64)
        fused = np.stack(
            [token_values(int(r.subject_id), int(r.stimulus_index), n_cells, dim) for r in trials.itertuples()]
        ).astype(np.float64)

        dataset = FakeDataset(trials, emb, fused)
        assignments = make_assignments(hc_ids, sz_ids, n_splits=2)
        cfg = BankConfig(
            seed=2026,
            batch_size=5,
            include_fused_token_bank=True,
            include_heatmap_token_bank=False,
            crossfit_splits=2,
            crossfit_enabled=True,
            output_root=None,
        )
        build = build_fold_banks(
            dataset._model,
            dataset,
            fold=0,
            cfg=cfg,
            n_stimuli=n_stimuli,
            forbidden_subject_ids=set(),
            assignments=assignments,
        )

        # Full bank: all 8 HC subjects.
        np.testing.assert_allclose(
            build.full.mu_trial[2],
            np.mean(
                [embed_values(int(s), 2, dim) for s in hc_ids], axis=0
            ),
            rtol=1e-6,
            atol=1e-6,
        )
        assert build.n_trials_full == n

        # Crossfit banks: split j excludes the 4 HC assigned to j.
        for split in (0, 1):
            excluded = build.excluded_crossfit[split]
            assert len(excluded) == 4
            contributors = [s for s in hc_ids if s not in excluded]
            assert len(build.contributors_crossfit[split]) == 4
            np.testing.assert_allclose(
                build.crossfit[split].mu_trial[1],
                np.mean([embed_values(int(s), 1, dim) for s in contributors], axis=0),
                rtol=1e-6,
                atol=1e-6,
            )
            np.testing.assert_allclose(
                build.crossfit[split].mu_token[3][7],
                np.mean([token_values(int(s), 3, n_cells, dim)[7] for s in contributors], axis=0),
                rtol=1e-6,
                atol=1e-5,
            )
            assert build.n_trials_crossfit[split] == 4 * n_stimuli

    def test_missing_crossfit_assignment_raises(self):
        trials = make_synthetic_trials([f"{i:03d}" for i in range(4)])
        emb = np.stack(
            [embed_values(int(r.subject_id), int(r.stimulus_index), 8) for r in trials.itertuples()]
        )
        dataset = FakeDataset(trials, emb)
        cfg = BankConfig(batch_size=4, include_fused_token_bank=False, crossfit_splits=2, output_root=None)
        partial = pd.DataFrame([{"subject_id": "000", "label": 0, "bank_split_id": 0}])
        with pytest.raises(BankBuildError, match="missing crossfit assignment"):
            build_fold_banks(
                dataset._model,
                dataset,
                fold=0,
                cfg=cfg,
                n_stimuli=4,
                forbidden_subject_ids=set(),
                assignments=partial,
            )

    def test_forbidden_contributor_raises(self):
        hc_ids = [f"{i:03d}" for i in range(4)]
        trials = make_synthetic_trials(hc_ids)
        emb = np.stack(
            [embed_values(int(r.subject_id), int(r.stimulus_index), 8) for r in trials.itertuples()]
        )
        dataset = FakeDataset(trials, emb)
        cfg = BankConfig(batch_size=4, include_fused_token_bank=False, crossfit_enabled=False, output_root=None)
        with pytest.raises(BankBuildError, match="forbidden"):
            build_fold_banks(
                dataset._model,
                dataset,
                fold=0,
                cfg=cfg,
                n_stimuli=4,
                forbidden_subject_ids={"001"},
                assignments=None,
            )

    def test_crossfit_disabled_builds_full_only(self):
        hc_ids = [f"{i:03d}" for i in range(4)]
        trials = make_synthetic_trials(hc_ids)
        emb = np.stack(
            [embed_values(int(r.subject_id), int(r.stimulus_index), 8) for r in trials.itertuples()]
        )
        dataset = FakeDataset(trials, emb)
        cfg = BankConfig(batch_size=4, include_fused_token_bank=False, crossfit_enabled=False, output_root=None)
        build = build_fold_banks(
            dataset._model,
            dataset,
            fold=0,
            cfg=cfg,
            n_stimuli=4,
            forbidden_subject_ids=set(),
            assignments=None,
        )
        assert build.crossfit == {}
        assert build.n_trials_full == len(trials)
