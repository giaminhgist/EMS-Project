"""Grouped HC batch sampler tests."""

from __future__ import annotations

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
