"""Grouped HC batch sampler tests."""

from __future__ import annotations

import collections

import pytest

from stage1.config import Stage1Config
from stage1.dataset import Stage1Dataset
from stage1.sampler import StimulusGroupedHCBatchSampler


def _ds_and_sampler(stage1_cfg_dict, epoch: int = 0):
    cfg = Stage1Config.from_dict(stage1_cfg_dict)
    ds = Stage1Dataset(
        cfg.paths.processed_root, cfg.paths.dino_root, cfg.paths.cv_fold_dir, "train"
    )
    sampler = StimulusGroupedHCBatchSampler(
        ds.stimulus_groups(),
        stimuli_per_batch=2,
        hc_per_stimulus=2,
        min_hc_per_stimulus=2,
        seed=cfg.seed,
        fold=cfg.fold,
        epoch=epoch,
        subject_by_row=ds.row_subject_ids(),
    )
    return ds, sampler


def test_sampler_groups_distinct_hc_subjects_per_stimulus(stage1_cfg_dict):
    ds, sampler = _ds_and_sampler(stage1_cfg_dict)
    assert sampler.n_skipped_stimuli == 1  # a2 has a single HC train row
    assert len(sampler) == 1  # 2 eligible stimuli (a1, b1) / 2 per batch
    batches = list(sampler)
    assert len(batches) == 1
    batch = batches[0]
    assert len(batch) == 4  # 2 stimuli x 2 trials
    stimuli_of = {i: ds[i].stimulus_id for i in batch}
    by_stimulus: dict[str, set[str]] = {}
    for i in batch:
        by_stimulus.setdefault(stimuli_of[i], set()).add(ds[i].subject_id)
    assert set(by_stimulus) == {"a1.jpg", "b1.jpg"}
    for stimulus, subjects in by_stimulus.items():
        assert len(subjects) == 2  # distinct HC subjects, no replacement


def test_sampler_deterministic_per_epoch_and_varies_across_epochs(stage1_cfg_dict):
    ds, s0 = _ds_and_sampler(stage1_cfg_dict, epoch=0)
    _, s0b = _ds_and_sampler(stage1_cfg_dict, epoch=0)
    batches0 = [tuple(b) for b in s0]
    batches0b = [tuple(b) for b in s0b]
    assert batches0 == batches0b  # reproducible per (seed, fold, epoch)
    orderings = set()
    for epoch in range(12):
        ds, s = _ds_and_sampler(stage1_cfg_dict, epoch=epoch)
        orderings.add(tuple(tuple(b) for b in s))
    assert len(orderings) >= 2  # epoch seed actually changes sampling


def test_missing_trials_not_synthesized_or_duplicated(stage1_cfg_dict):
    ds, sampler = _ds_and_sampler(stage1_cfg_dict)
    for batch in sampler:
        assert len(batch) == len(set(batch))  # no duplicated trial rows
        for i in batch:
            assert 0 <= i < len(ds)  # only real dataset rows


def test_replacement_is_refused(stage1_cfg_dict):
    ds, _ = _ds_and_sampler(stage1_cfg_dict)
    with pytest.raises(ValueError, match="replacement"):
        StimulusGroupedHCBatchSampler(
            ds.stimulus_groups(),
            stimuli_per_batch=2,
            hc_per_stimulus=2,
            min_hc_per_stimulus=2,
            replacement=True,
        )


def _synthetic_rotation_data(n_stimuli: int, n_subjects: int, rows_per_subject: int = 1):
    """stimulus -> rows; each subject contributes rows_per_subject rows."""
    groups: dict[int, list[int]] = {}
    subject_by_row: dict[int, str] = {}
    idx = 0
    for si in range(n_stimuli):
        rows = []
        for subj in range(n_subjects):
            for _ in range(rows_per_subject):
                rows.append(idx)
                subject_by_row[idx] = f"subj_{subj}"
                idx += 1
        groups[si] = rows
    return groups, subject_by_row


def test_rotation_covers_every_trial_uniformly():
    # 3 stimuli x 8 subjects x 1 row; H=2, 8 epochs -> each row exactly
    # floor/ceil(8*2/8) = 2 times, and at most once per epoch.
    groups, subject_by_row = _synthetic_rotation_data(n_stimuli=3, n_subjects=8)
    counts: collections.Counter = collections.Counter()
    for epoch in range(8):
        sampler = StimulusGroupedHCBatchSampler(
            groups,
            stimuli_per_batch=3,
            hc_per_stimulus=2,
            min_hc_per_stimulus=2,
            seed=2026,
            fold=0,
            epoch=epoch,
            subject_by_row=subject_by_row,
        )
        seen: set[int] = set()
        for batch in sampler:
            assert not seen & set(batch)  # no trial twice within one epoch
            seen |= set(batch)
        counts.update(seen)
    assert set(counts.values()) == {2}


def test_remainder_batch_keeps_all_stimuli_each_epoch():
    # 5 stimuli x 8 subjects; S=2 -> 3 batches (2 full + 1 remainder of 1
    # stimulus); every stimulus participates in every epoch.
    groups, subject_by_row = _synthetic_rotation_data(n_stimuli=5, n_subjects=8)
    row_to_stimulus = {r: si for si, rows in groups.items() for r in rows}
    sampler = StimulusGroupedHCBatchSampler(
        groups,
        stimuli_per_batch=2,
        hc_per_stimulus=2,
        min_hc_per_stimulus=2,
        seed=2026,
        fold=0,
        epoch=0,
        subject_by_row=subject_by_row,
    )
    assert len(sampler) == 3  # ceil(5 / 2)
    batches = list(sampler)
    assert [len(b) for b in batches] == [4, 4, 2]
    for epoch in range(4):
        sampler.set_epoch(epoch)
        stimuli_seen: set[int] = set()
        for batch in sampler:
            stimuli_seen |= {row_to_stimulus[r] for r in batch}
        assert stimuli_seen == set(groups)  # all 5 stimuli every epoch


def test_rotation_never_duplicates_subject_within_batch():
    # One stimulus; subject A holds 2 rows, B/C/D hold 1 each. A window of 3
    # slots must pick 3 distinct subjects even though A has multiple rows.
    groups = {0: [0, 1, 2, 3, 4]}
    subject_by_row = {0: "A", 1: "A", 2: "B", 3: "C", 4: "D"}
    sampler = StimulusGroupedHCBatchSampler(
        groups,
        stimuli_per_batch=1,
        hc_per_stimulus=3,
        min_hc_per_stimulus=3,
        seed=2026,
        fold=0,
        epoch=0,
        subject_by_row=subject_by_row,
    )
    batch = next(iter(sampler))
    subjects = {subject_by_row[r] for r in batch}
    assert len(batch) == 3
    assert len(subjects) == 3  # distinct subjects within the batch

    # Over 8 epochs every row (including both rows of subject A) is trained:
    # 8*3/4 = 6 window visits per slot; A's two rows split them -> >= 2 each.
    counts: collections.Counter = collections.Counter()
    for epoch in range(8):
        sampler.set_epoch(epoch)
        for batch in sampler:
            counts.update(batch)
    assert min(counts[r] for r in groups[0]) >= 2


def test_subject_by_row_length_is_validated():
    groups, _ = _synthetic_rotation_data(n_stimuli=1, n_subjects=2)
    with pytest.raises(ValueError, match="subject_by_row"):
        StimulusGroupedHCBatchSampler(
            groups,
            stimuli_per_batch=1,
            hc_per_stimulus=2,
            min_hc_per_stimulus=2,
            seed=2026,
            fold=0,
            epoch=0,
            subject_by_row=["only_one_subject"],
        )
