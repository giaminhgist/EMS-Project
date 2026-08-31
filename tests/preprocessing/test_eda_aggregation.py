"""EDA aggregation tests on the synthetic processed dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from preprocessing.eda import (
    channel_statistics,
    compute_eda_summary,
    group_comparisons,
    inventory_statistics,
    mass_consistency,
    subject_level_aggregation,
    trial_frame,
)
from preprocessing.storage import TrialStore


@pytest.fixture
def store(synthetic_processed):
    return TrialStore(synthetic_processed)


def test_subject_level_aggregation_one_row_per_subject(store):
    sub = subject_level_aggregation(store)
    assert len(sub) == 5
    assert sub.subject_id.is_unique
    assert set(sub.group) == {"HC", "SZ"}
    # Aggregations are subject-level: one value per subject, not per trial.
    assert len(sub) == len(store.subject_manifest)


def test_missing_trials_counted_not_imputed(store):
    sub = subject_level_aggregation(store)
    s005 = sub[sub.subject_id == "005"].iloc[0]
    assert s005.n_trials == 2  # a2.jpg stays missing
    assert s005.n_trials_ok == 2
    sm = store.subject_manifest.set_index("subject_id")
    assert int(sm.loc["005", "n_missing_expected_stimuli"]) == 1
    # No synthetic trial rows anywhere.
    tm = store.trial_manifest
    assert len(tm[(tm.subject_id == "005") & (tm.stimulus_id == "a2.jpg")]) == 0


def test_trial_frame_joins_by_explicit_ids(store):
    tf = trial_frame(store)
    assert len(tf) == 10  # 3 + 2 + 1 (HC) + 2 + 2 (SZ) ok trials
    # Cross-check a random row through the explicit-key accessor.
    row = tf.sample(1, random_state=0).iloc[0]
    rec = store.get_trial(row.subject_id, store.trial_manifest.loc[row.trial_uid, "stimulus_id"])
    assert rec.stimulus_index == row.stimulus_index
    assert abs(rec.heatmap[0].sum() - row.mass_ch0) < 1e-3
    assert abs(rec.heatmap[1].sum() - row.mass_ch1) < 1e-3


def test_channel_statistics_match_brute_force(store):
    stats = channel_statistics(store)
    # Brute force: load everything (small synthetic dataset only).
    all_ch: dict[int, list[np.ndarray]] = {0: [], 1: [], 2: []}
    mass0, mass1 = [], []
    for subject_dir in sorted((store.root / "subjects").iterdir()):
        hm = np.load(subject_dir / "heatmaps.npy")
        for c in range(3):
            all_ch[c].append(hm[:, c].astype(np.float64))
        mass0.extend(hm[:, 0].sum(axis=(1, 2)))
        mass1.extend(hm[:, 1].sum(axis=(1, 2)))
    for c in range(3):
        flat = np.concatenate([a.ravel() for a in all_ch[c]])
        d = stats["per_channel"][c]
        assert d["min"] == pytest.approx(flat.min())
        assert d["max"] == pytest.approx(flat.max())
        assert d["mean"] == pytest.approx(flat.mean(), abs=1e-12)
        assert d["std"] == pytest.approx(flat.std(), abs=1e-12)
        assert d["finite_fraction"] == 1.0
    mass0 = np.asarray(mass0)
    mass1 = np.asarray(mass1)
    m0 = stats["per_trial_mass"]["channel_0_fixation_density"]
    m1 = stats["per_trial_mass"]["channel_1_transition_density"]
    assert m0["mean"] == pytest.approx(mass0.mean())
    assert m1["mean"] == pytest.approx(mass1.mean())


def test_mass_consistency_tight(store):
    mc = mass_consistency(trial_frame(store))
    assert mc["channel_0"]["mean_abs_error"] < 1e-3
    assert mc["channel_1"]["mean_abs_error"] < 1e-3


def test_group_comparisons_subject_level(store):
    sub = subject_level_aggregation(store)
    gc = group_comparisons(sub)
    for r in gc["results"]:
        assert r["unit"] == "subject"
        assert r["n_hc"] == 3
        assert r["n_sz"] == 2
        assert r["p_value"] is not None
        assert -1.0 <= r["rank_biserial_r"] <= 1.0


def test_inventory_statistics_reconcile(store):
    inv = inventory_statistics(store)
    assert inv["n_subjects"] == 5
    assert inv["subjects_by_group"] == {"HC": 3, "SZ": 2}
    assert inv["n_trials_observed"] == 11  # 7 + 2 + 2
    assert inv["n_trials_excluded"] == 1
    assert inv["n_excluded_or_warning_trials"] == 1
    assert inv["stimuli_by_category"] == {"CatA": 2, "CatB": 1}


def test_eda_summary_is_json_serializable_and_complete(store):
    summary = compute_eda_summary(store.root, "test command")
    import json

    text = json.dumps(summary)  # must not raise
    assert "inventory" in summary
    assert "channel_stats" in summary
    assert "group_comparisons" in summary
    assert "representative_trials" in summary
    assert set(summary["representative_trials"]) == {"HC", "SZ"}
    assert "example_heatmaps" in summary
