"""Normative-bank builder tests (contract §9)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from stage1.config import Stage1Config
from stage1.dataset import Stage1Dataset
from stage1.model import Stage1Model
from stage1.normative_bank import (
    NormBankConfig,
    NormBankError,
    build_normative_bank,
    save_normative_bank,
)


@pytest.fixture(scope="module")
def env_model_dataset(stage1_cfg_dict):
    cfg = Stage1Config.from_dict(stage1_cfg_dict)
    ds = Stage1Dataset(
        cfg.paths.processed_root, cfg.paths.dino_root, cfg.paths.cv_fold_dir, "train"
    )
    model = Stage1Model(cfg)
    model.eval()
    return cfg, model, ds


def test_bank_uses_only_training_hc_and_refuses_validation(env_model_dataset):
    cfg, model, ds = env_model_dataset
    bank_cfg = NormBankConfig(min_samples=1, epsilon=1e-6)
    result = build_normative_bank(
        model, ds, fold=0, seed=2026, n_stimuli=3,
        checkpoint_sha256="0" * 64,
        processed_checksums={"subject_manifest_sha256": "0" * 64},
        dino_checksum="0" * 64,
        config=bank_cfg,
        forbidden_subject_ids={"013", "200", "201"},
    )
    assert set(result.metadata["subject_ids"]) == {"000", "005"}  # train HC only
    assert result.metadata["n_trials"] == 5
    assert result.count_trial.tolist() == [2, 1, 2]  # a1, a2, b1
    # Validation subjects are rejected when they appear.
    with pytest.raises(NormBankError, match="forbidden"):
        build_normative_bank(
            model, ds, fold=0, seed=2026, n_stimuli=3,
            checkpoint_sha256="0" * 64,
            processed_checksums={}, dino_checksum="0" * 64,
            config=bank_cfg,
            forbidden_subject_ids={"000"},  # a legit train subject flagged
        )


def test_insufficient_samples_are_reported_and_policy_enforced(env_model_dataset):
    cfg, model, ds = env_model_dataset
    with pytest.raises(NormBankError, match="below min_samples"):
        build_normative_bank(
            model, ds, fold=0, seed=2026, n_stimuli=3,
            checkpoint_sha256="0" * 64, processed_checksums={}, dino_checksum="0" * 64,
            config=NormBankConfig(min_samples=2),
        )
    result = build_normative_bank(
        model, ds, fold=0, seed=2026, n_stimuli=3,
        checkpoint_sha256="0" * 64, processed_checksums={}, dino_checksum="0" * 64,
        config=NormBankConfig(min_samples=2, on_insufficient="warn"),
    )
    assert result.metadata["insufficient_stimuli"] == [1]


def test_bank_statistics_match_manual_computation(env_model_dataset):
    cfg, model, ds = env_model_dataset
    # Manual unmasked forward for the two a1 trials.
    indices = [i for i in range(len(ds)) if ds[i].stimulus_id == "a1.jpg"]
    emb = []
    for i in indices:
        batch = ds.collate_from_indices([i])
        with torch.inference_mode():
            out = model(batch, token_mask=None)
        emb.append(out.trial_embedding.detach().numpy().squeeze(0))
        assert not out.mask.any()  # unmasked inference enforced
    stacked = np.stack(emb).astype(np.float64)  # float64 to avoid cancellation
    manual_mu = np.mean(stacked, axis=0)
    manual_sigma = np.std(stacked, axis=0)

    result = build_normative_bank(
        model, ds, fold=0, seed=2026, n_stimuli=3,
        checkpoint_sha256="0" * 64, processed_checksums={}, dino_checksum="0" * 64,
        # batch_size=1 matches the manual loop exactly: CPU GEMM tiling is
        # batch-shape dependent at ~1e-6, so the reference must use the same
        # batching.
        config=NormBankConfig(min_samples=1, batch_size=1),
    )
    a1_index = 0  # a1.jpg has stimulus_index 0 in the synthetic manifest
    assert np.allclose(result.mu_trial[a1_index], manual_mu, atol=1e-6)
    assert np.allclose(
        result.sigma_trial[a1_index],
        np.maximum(manual_sigma, 1e-6),
        atol=1e-6,
    )
    # Missing stimuli are recorded, not silently filled.
    assert result.metadata["missing_stimuli"] == []


def test_save_bank_roundtrip(env_model_dataset, tmp_path):
    cfg, model, ds = env_model_dataset
    result = build_normative_bank(
        model, ds, fold=0, seed=2026, n_stimuli=3,
        checkpoint_sha256="0" * 64, processed_checksums={}, dino_checksum="0" * 64,
        config=NormBankConfig(min_samples=1, include_token_level=True),
    )
    out_dir = tmp_path / "bank"
    metadata = save_normative_bank(result, out_dir, stimulus_ids=["a1.jpg", "a2.jpg", "b1.jpg"])
    assert (out_dir / "mu_trial.npy").is_file()
    assert (out_dir / "sigma_trial.npy").is_file()
    assert (out_dir / "count_trial.npy").is_file()
    assert (out_dir / "mu_token.npy").is_file()
    assert (out_dir / "sigma_token.npy").is_file()
    assert (out_dir / "metadata.json").is_file()
    assert (out_dir / "feature_manifest.csv").is_file()
    mu = np.load(out_dir / "mu_trial.npy")
    assert mu.shape == (3, 128) and mu.dtype == np.float32
    counts = np.load(out_dir / "count_trial.npy")
    assert counts.dtype == np.int32 and counts.tolist() == [2, 1, 2]
    meta = json.loads((out_dir / "metadata.json").read_text())
    assert meta["fold"] == 0
    assert set(meta["subject_ids"]) == {"000", "005"}
    assert len(meta["array_sha256"]["mu_trial"]) == 64
    fm = pd.read_csv(out_dir / "feature_manifest.csv", dtype=str)
    assert list(fm.stimulus_id) == ["a1.jpg", "a2.jpg", "b1.jpg"]
