"""Synthetic Stage-1 fixtures: processed dataset + fake DINO + CV partitions."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests" / "preprocessing"))

from tests.preprocessing.conftest import (  # noqa: E402
    make_synthetic_raw,
    synthetic_config_dict,
    write_workbook,
)

N_STIMULI = 3
DINO_TOKENS = 768
DINO_DIM = 384


def make_fake_dino(root: Path, processed_dir: Path, n_stimuli: int = N_STIMULI, seed: int = 11) -> Path:
    """Deterministic fake DINO artifacts (patch_tokens.npy + manifests)."""
    dino_dir = root / "dino_vits16"
    dino_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    tokens = rng.standard_normal((n_stimuli, DINO_TOKENS, DINO_DIM)).astype(np.float32)
    np.save(dino_dir / "patch_tokens.npy", tokens)
    image_manifest = pd.read_csv(processed_dir / "image_manifest.csv", dtype=str)
    rows = []
    for i, r in image_manifest.iterrows():
        token_bytes = np.ascontiguousarray(tokens[i]).tobytes()
        rows.append(
            {
                "stimulus_index": int(r.stimulus_index),
                "stimulus_id": str(r.stimulus_id),
                "feature_row_index": int(r.stimulus_index),
                "image_sha256": str(r.sha256),
                "feature_shape": f"[{DINO_TOKENS}, {DINO_DIM}]",
                "feature_dtype": "float32",
                "feature_sha256": hashlib.sha256(token_bytes).hexdigest(),
            }
        )
    pd.DataFrame(rows).to_csv(dino_dir / "feature_manifest.csv", index=False)
    return dino_dir


def make_fold_dir(processed: Path, fold: int = 0) -> Path:
    """CV fold partition dir: train HC = {000, 005}, val HC = {013}; SZ
    subjects stay in the partitions (Stage 1 filters them)."""
    fold_dir = processed / "CV" / "5fold_seed2026" / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    subjects = pd.read_csv(processed / "subject_manifest.csv", dtype={"subject_id": str})
    train_hc = {"000", "005"}
    val_hc = {"013"}
    train_subjects = subjects[
        (subjects.group == "HC") & (subjects.subject_id.isin(train_hc))
        | (subjects.group == "SZ")
    ]
    val_subjects = subjects[
        (subjects.group == "HC") & (subjects.subject_id.isin(val_hc))
        | (subjects.group == "SZ")
    ]
    train_subjects.to_csv(fold_dir / "train_subjects.csv", index=False)
    val_subjects.to_csv(fold_dir / "val_subjects.csv", index=False)

    trials = pd.read_parquet(processed / "trial_manifest.parquet")
    trials["subject_id"] = trials["subject_id"].astype("string")
    train_trials = trials[trials.subject_id.isin(set(train_subjects.subject_id))]
    val_trials = trials[trials.subject_id.isin(set(val_subjects.subject_id))]
    train_trials.to_parquet(fold_dir / "train_trials.parquet", index=False)
    val_trials.to_parquet(fold_dir / "val_trials.parquet", index=False)

    # Minimal cv_metadata.json so checksum verification succeeds.
    import json

    def sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    (fold_dir.parent).mkdir(parents=True, exist_ok=True)
    (fold_dir.parent / "cv_metadata.json").write_text(
        json.dumps(
            {
                "input_checksums": {
                    "subject_manifest_sha256": sha256(processed / "subject_manifest.csv"),
                    "trial_manifest_sha256": sha256(processed / "trial_manifest.parquet"),
                    "source_inventory_sha256": sha256(processed / "source_inventory.json"),
                },
                "config_hash": "synthetic",
            }
        ),
        encoding="utf-8",
    )
    return fold_dir


@pytest.fixture(scope="module")
def stage1_env(tmp_path_factory):
    """Module-scoped synthetic environment: processed dataset, fake DINO, CV."""
    from preprocessing.config import PreprocessingConfig
    from preprocessing.pipeline import PipelineOptions, run_pipeline

    base = tmp_path_factory.mktemp("stage1_raw")
    raw = make_synthetic_raw(base / "rawbase")
    # Add two SZ subjects (>= 200) to exercise the HC filter.
    write_workbook(
        raw / "EMS" / "All_Data" / "Fixations" / "200.xlsx",
        [
            ("a1.jpg", 1, 220.0, 300.0, 400.0, 1000.0),
            ("a1.jpg", 2, 260.0, 700.0, 300.0, 1000.0),
            ("b1.jpg", 1, 180.0, 500.0, 384.0, 1000.0),
        ],
    )
    write_workbook(
        raw / "EMS" / "All_Data" / "Fixations" / "201.xlsx",
        [
            ("a2.jpg", 1, 240.0, 200.0, 200.0, 1000.0),
            ("b1.jpg", 1, 300.0, 400.0, 300.0, 1000.0),
        ],
    )
    processed = tmp_path_factory.mktemp("stage1_processed")
    cfg = synthetic_config_dict(raw, processed)
    run_pipeline(PreprocessingConfig.from_dict(cfg), PipelineOptions())

    dino_dir = make_fake_dino(tmp_path_factory.mktemp("stage1_dino"), processed)
    fold_dir = make_fold_dir(processed)
    return {"processed": processed, "dino": dino_dir, "fold_dir": fold_dir, "raw": raw}


@pytest.fixture(scope="module")
def stage1_cfg_dict(stage1_env, tmp_path_factory):
    e = stage1_env
    tmp_path = tmp_path_factory.mktemp("stage1_outputs")
    return {
        "experiment_name": "synthetic_stage1",
        "seed": 2026,
        "fold": 0,
        "model": {
            "d_model": 128,
            "heatmap_patch_size": 4,
            "heatmap_residual_blocks": 2,
            "semantic_source": "dino",
            "semantic_adapter": "learned_depthwise_2x2",
            "fusion": "serial_attention_spatial_attention",
            "attention_heads": 4,
            "attention_dropout": 0.1,
            "semantic_gamma_total_init": 0.1,
            "share_cross_attention_weights": False,
            "spatial_bridge": "residual_dwconv_ffn",
            "spatial_bridge_expansion_ratio": 2.0,
            "spatial_bridge_kernel_size": 3,
            "spatial_bridge_dropout": 0.1,
            "spatial_bridge_eta_init": 0.1,
            "pooling": "attention",
            "positional_encoding": "fixed_2d_sincos",
        },
        "masking": {
            "train_mask_ratio": 0.35,
            "validation_mask_ratio": 0.35,
            "reconstruction_scope": "masked",
        },
        "loss": {
            "reconstruction": "smooth_l1",
            "channel_weights": [1.0, 1.0, 1.0],
            "normative_metric": "loo_cosine",
            "lambda_norm": 0.1,
            "norm_start_epoch": 10,
            "norm_ramp_epochs": 5,
        },
        "sampler": {
            "stimuli_per_batch": 2,
            "hc_per_stimulus": 2,
            "replacement": False,
            "min_hc_per_stimulus": 2,
        },
        "optimization": {
            "epochs": 100,
            "optimizer": "adamw",
            "learning_rate": 3.0e-4,
            "weight_decay": 5.0e-4,
            "scheduler": "linear_warmup_cosine",
            "lr_warmup_epochs": 5,
            "gradient_clip_norm": 5.0,
            "amp": True,
        },
        "validation": {
            "selection_metric": "val_loss",
            "best_eligible_after_norm_ramp": True,
            "early_stopping_patience": 20,
        },
        "runtime": {
            "num_workers": 0,
            "pin_memory": True,
            "persistent_workers": True,
            "deterministic_validation": True,
        },
        "paths": {
            "processed_root": str(e["processed"]),
            "dino_root": str(e["dino"]),
            "cv_fold_dir": str(e["fold_dir"]),
            "output_root": str(tmp_path / "outputs"),
        },
    }
