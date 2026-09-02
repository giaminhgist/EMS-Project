"""Shared synthetic fixtures for Stage-2 Phase-2/3 tests (no real EMS data).

Helpers build a minimal processed dataset, CV partitions, DINO placeholder
files, a synthetic Stage-1 checkpoint/registry and a Phase-1-style normative
bank root on demand.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from stage2.bank_builder import (
    BankArrays,
    BankConfig,
    build_metadata,
    publish_bank_dir,
    write_root_manifest,
)
from stage2.contracts import expected_array_shapes

N_STIMULI = 100


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_image_manifest(processed_root: Path) -> pd.DataFrame:
    categories = ["Manipulated Images", "Natural Scenes", "Social Scenes", "Synthetic Images"]
    df = pd.DataFrame(
        {
            "stimulus_index": list(range(N_STIMULI)),
            "stimulus_id": [f"s{i:03d}.jpg" for i in range(N_STIMULI)],
            "category": [categories[i % 4] for i in range(N_STIMULI)],
        }
    )
    df.to_csv(processed_root / "image_manifest.csv", index=False)
    return df


def make_subject_manifest(processed_root: Path, subjects: dict[str, int]) -> pd.DataFrame:
    """subjects: subject_id -> label."""
    df = pd.DataFrame(
        {
            "subject_id": list(subjects),
            "label": list(subjects.values()),
            "group": ["HC" if v == 0 else "SZ" for v in subjects.values()],
        }
    )
    df.to_csv(processed_root / "subject_manifest.csv", index=False)
    return df


def make_trial_manifest(
    processed_root: Path,
    subjects: dict[str, int],
    observed: dict[str, list[int]],
) -> pd.DataFrame:
    """observed: subject_id -> list of observed stimulus indices (no dups)."""
    rows = []
    for sid, label in subjects.items():
        for row_idx, si in enumerate(sorted(observed.get(sid, []))):
            rows.append(
                {
                    "trial_uid": f"{sid}_{si:03d}",
                    "subject_id": sid,
                    "stimulus_id": f"s{si:03d}.jpg",
                    "subject_numeric_id": int(sid),
                    "stimulus_index": si,
                    "group": "HC" if label == 0 else "SZ",
                    "label": label,
                    "category": "Natural Scenes",
                    "subject_array_path": f"subjects/{sid}/heatmaps.npy",
                    "subject_row_index": row_idx,
                    "qc_status": "ok",
                }
            )
    df = pd.DataFrame(rows)
    df.to_parquet(processed_root / "trial_manifest.parquet", index=False)
    return df


def make_heatmap_arrays(
    processed_root: Path, subjects: dict[str, int], observed: dict[str, list[int]]
) -> None:
    for sid, stimuli in observed.items():
        n = len(stimuli)
        rng = np.random.default_rng(int(sid) * 7 + n)
        arr = rng.normal(size=(n, 3, 48, 64)).astype(np.float32)
        arr[:, :2] = np.abs(arr[:, :2])  # density/mass channels non-negative
        np.save(processed_root / "subjects" / sid / "heatmaps.npy", arr)


def make_cv_root(
    root: Path,
    subjects: dict[str, int],
    train_ids: list[str],
    val_ids: list[str],
    *,
    fold: int = 0,
) -> Path:
    cv_root = root / "CV" / "5fold_seed2026"
    fold_dir = cv_root / f"fold_{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    for split, ids in (("train", train_ids), ("val", val_ids)):
        pd.DataFrame({"subject_id": ids, "label": [subjects[i] for i in ids]}).to_csv(
            fold_dir / f"{split}_subjects.csv", index=False
        )
    return cv_root


def make_dino_placeholder(root: Path, processed_root: Path) -> Path:
    dino_root = root / "stimulus_features" / "dino_vits16"
    dino_root.mkdir(parents=True, exist_ok=True)
    (dino_root / "patch_tokens.npy").write_bytes(b"placeholder dino tokens")
    image = pd.read_csv(processed_root / "image_manifest.csv", dtype=str)
    pd.DataFrame(
        {
            "stimulus_index": image.stimulus_index,
            "stimulus_id": image.stimulus_id,
            "feature_row_index": list(range(N_STIMULI)),
        }
    ).to_csv(dino_root / "feature_manifest.csv", index=False)
    return dino_root


def make_dataset_metadata(processed_root: Path, subjects: dict[str, int]) -> dict:
    meta = {"completed": True, "n_subjects": len(subjects)}
    (processed_root / "dataset_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def make_cv_metadata(cv_root: Path, processed_root: Path) -> dict:
    meta = {
        "seed": 2026,
        "n_splits": 5,
        "input_checksums": {
            "subject_manifest_sha256": sha256_of(processed_root / "subject_manifest.csv"),
            "trial_manifest_sha256": sha256_of(processed_root / "trial_manifest.parquet"),
        },
    }
    (cv_root / "cv_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def make_processed_fixture(
    root: Path,
    subjects: dict[str, int],
    observed: dict[str, list[int]],
    *,
    with_dino: bool = False,
) -> Path:
    """Build a minimal synthetic processed dataset under ``root/processed_dataset``."""
    processed_root = root / "processed_dataset"
    (processed_root / "subjects").mkdir(parents=True, exist_ok=True)
    for sid in subjects:
        (processed_root / "subjects" / sid).mkdir(exist_ok=True)
    make_image_manifest(processed_root)
    make_subject_manifest(processed_root, subjects)
    make_trial_manifest(processed_root, subjects, observed)
    make_heatmap_arrays(processed_root, subjects, observed)
    make_dataset_metadata(processed_root, subjects)
    if with_dino:
        make_dino_placeholder(root, processed_root)
    return processed_root


def make_stage1_checkpoint(root: Path, fold: int = 0, seed: int = 7) -> tuple[Path, str]:
    """Write a real-format synthetic Stage-1 checkpoint under ``root``.

    Uses a randomly initialized ``HeatmapPatchEncoder`` (real architecture,
    synthetic weights) plus non-encoder keys that the transferred-encoder
    wrapper must ignore. Returns ``(path, sha256)``.
    """
    from stage1.heatmap_encoder import HeatmapPatchEncoder

    torch.manual_seed(seed)
    encoder = HeatmapPatchEncoder(
        in_channels=3, d_model=128, patch_size=4, grid_h=12, grid_w=16, n_residual_blocks=2
    )
    state = {f"heatmap_encoder.{k}": v for k, v in encoder.state_dict().items()}
    state["semantic_adapter.adapter.depthwise.weight"] = torch.zeros(2, 2)
    state["pooling.W1.weight"] = torch.zeros(2, 2)
    payload = {
        "model_state": state,
        "optimizer_state": None,
        "scheduler_state": None,
        "scaler_state": None,
        "epoch": 0,
        "global_step": 0,
        "best_metric": 0.0,
        "best_epoch": 0,
        "fold": fold,
        "run_id": "synthetic_stage1_run",
        "config_resolved": {
            "model": {
                "d_model": 128,
                "heatmap_patch_size": 4,
                "heatmap_residual_blocks": 2,
                "positional_encoding": "fixed_2d_sincos",
                "input_channels": 3,
                "active_channels": [0, 1, 2],
            }
        },
        "config_hash": "a" * 64,
        "source_checksums": {},
        "rng_state": {},
        "sampler_epoch": 0,
    }
    path = root / f"stage1_checkpoint_fold{fold}.pt"
    torch.save(payload, path)
    return path, sha256_of(path)


def make_checkpoint_registry(
    root: Path, checkpoints: dict[int, tuple[Path, str]]
) -> Path:
    """Write a Stage-1 checkpoint registry mapping each fold to one checkpoint."""
    registry = {
        "schema_version": 1,
        "evaluation_regime": "pilot_existing_stage1",
        "folds": {
            str(f): {"checkpoint": str(path), "sha256": sha}
            for f, (path, sha) in checkpoints.items()
        },
    }
    registry_path = root / "stage1_checkpoints.yaml"
    registry_path.write_text(yaml.safe_dump(registry), encoding="utf-8")
    return registry_path


def make_bank_fixture(
    root: Path,
    *,
    processed_root: Path,
    cv_root: Path,
    hc_ids: list[str],
    sz_ids: list[str],
    forbidden_ids: list[str],
    crossfit_splits: int = 2,
    include_token: bool = False,
    registry_path: Path | None = None,
    dino_root: Path | None = None,
    checkpoint: Path | None = None,
    checkpoint_sha: str | None = None,
) -> tuple[Path, Path]:
    """Build a synthetic normative bank root + checkpoint registry under ``root``.

    ``checkpoint``/``checkpoint_sha`` supply a real-format Stage-1 checkpoint;
    when omitted a dummy byte file is registered instead. Returns
    ``(bank_root, registry_path)``.
    """
    bank_root = root / "normative_bank"
    if checkpoint is None:
        checkpoint = root / "dummy_checkpoint.pt"
        checkpoint.write_bytes(b"dummy stage1 checkpoint")
    checkpoint_sha = checkpoint_sha or sha256_of(checkpoint)
    if registry_path is None:
        registry_path = root / "stage1_checkpoints.yaml"
    registry = {
        "schema_version": 1,
        "evaluation_regime": "pilot_existing_stage1",
        "folds": {
            str(f): {"checkpoint": str(checkpoint), "sha256": checkpoint_sha} for f in range(5)
        },
    }
    registry_path.write_text(yaml.safe_dump(registry), encoding="utf-8")

    source_checksums = {
        "processed_subject_manifest_sha256": sha256_of(processed_root / "subject_manifest.csv"),
        "processed_trial_manifest_sha256": sha256_of(processed_root / "trial_manifest.parquet"),
        "processed_dataset_metadata_sha256": sha256_of(processed_root / "dataset_metadata.json"),
        "cv_metadata_sha256": sha256_of(cv_root / "cv_metadata.json"),
    }
    if dino_root is not None:
        source_checksums["dino_patch_tokens_sha256"] = sha256_of(dino_root / "patch_tokens.npy")
    provenance = {
        "stage1_checkpoint_path": str(checkpoint),
        "stage1_checkpoint_sha256": checkpoint_sha,
        "stage1_run_id": "synthetic_run",
        "stage1_config_hash": "c" * 64,
        "processed_dataset_metadata_sha256": source_checksums["processed_dataset_metadata_sha256"],
        "dino_feature_manifest_sha256": "d" * 64,
        "cv_partition_sha256": {"train_trials": "e" * 64, "val_trials": "f" * 64},
        "source_checksums": source_checksums,
    }

    feature_manifest = pd.DataFrame(
        {
            "stimulus_index": list(range(N_STIMULI)),
            "stimulus_id": [f"s{i:03d}.jpg" for i in range(N_STIMULI)],
            "category_id": [i % 4 for i in range(N_STIMULI)],
        }
    )
    shapes = expected_array_shapes(include_token, False)
    excluded = {
        j: {sid for i, sid in enumerate(hc_ids) if i % crossfit_splits == j}
        for j in range(crossfit_splits)
    }

    def make_arrays(seed: int) -> BankArrays:
        rng = np.random.default_rng(seed)
        arrays = BankArrays(
            mu_trial=rng.normal(size=(100, 128)).astype(np.float32),
            sigma_trial=(1.0 + rng.random((100, 128))).astype(np.float32),
            count_trial=np.full(100, len(hc_ids), dtype=np.int32),
        )
        if include_token:
            arrays.mu_token = rng.normal(size=(100, 192, 128)).astype(np.float32)
            arrays.sigma_token = (1.0 + rng.random((100, 192, 128))).astype(np.float32)
        return arrays

    def make_meta(
        crossfit_split: int | None,
        contributors: list[str],
        excluded_ids: list[str] | None,
        n_trials: int,
    ) -> dict:
        return build_metadata(
            fold=0,
            crossfit_split=crossfit_split,
            seed=2026,
            evaluation_regime="pilot_existing_stage1",
            estimator="mean",
            epsilon=1e-6,
            min_samples=2,
            include_fused_token_bank=include_token,
            include_heatmap_token_bank=False,
            provenance=provenance,
            contributing_hc_subject_ids=sorted(contributors),
            forbidden_validation_subject_ids=sorted(forbidden_ids),
            n_contributing_subjects=len(contributors),
            n_trials=n_trials,
            array_shapes={},
            array_sha256={},
            stimulus_manifest_sha256="",
            crossfit_excluded_hc_subject_ids=excluded_ids,
        )

    fold_dir = bank_root / "fold_0"
    publish_bank_dir(
        fold_dir,
        make_arrays(0),
        feature_manifest=feature_manifest,
        metadata=make_meta(None, hc_ids, None, 100 * len(hc_ids)),
        expected_shapes=shapes,
        min_samples=2,
        epsilon=1e-6,
    )
    crossfit_dir = fold_dir / "crossfit"
    crossfit_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, sid in enumerate(hc_ids):
        rows.append(
            {
                "subject_id": sid,
                "label": 0,
                "bank_split_id": i % crossfit_splits,
                "is_hc_bank_contributor": True,
                "panel_trial_count": 100,
            }
        )
    for i, sid in enumerate(sz_ids):
        rows.append(
            {
                "subject_id": sid,
                "label": 1,
                "bank_split_id": i % crossfit_splits,
                "is_hc_bank_contributor": False,
                "panel_trial_count": 90,
            }
        )
    assignment = pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)
    assignment.to_csv(crossfit_dir / "subject_assignment.csv", index=False)
    for j in range(crossfit_splits):
        publish_bank_dir(
            crossfit_dir / f"split_{j}",
            make_arrays(j + 1),
            feature_manifest=feature_manifest,
            metadata=make_meta(
                j,
                [s for s in hc_ids if s not in excluded[j]],
                sorted(excluded[j]),
                100 * len([s for s in hc_ids if s not in excluded[j]]),
            ),
            expected_shapes=shapes,
            min_samples=2,
            epsilon=1e-6,
        )
    bank_cfg = BankConfig(
        output_root=bank_root,
        include_fused_token_bank=include_token,
        include_heatmap_token_bank=False,
        crossfit_splits=crossfit_splits,
        crossfit_enabled=True,
    )
    write_root_manifest(
        bank_root,
        cfg=bank_cfg,
        evaluation_regime="pilot_existing_stage1",
        checkpoint_registry=registry,
    )
    return bank_root, registry_path


DEFAULT_HC_IDS = ["000", "002", "004", "008"]
DEFAULT_SZ_IDS = ["101", "103"]
DEFAULT_FORBIDDEN_IDS = ["010", "105"]


def make_full_model_fixture(
    tmp_path: Path,
    *,
    crossfit_splits: int = 2,
    include_token: bool = False,
    model: dict[str, Any] | None = None,
    loss: dict[str, Any] | None = None,
    subsets: dict[str, Any] | None = None,
    seed: int = 7,
    build_model: bool = True,
) -> dict[str, Any]:
    """Full synthetic stack for Phase-3 model tests: processed dataset, CV
    partitions, DINO placeholder, real-format Stage-1 checkpoint + registry,
    normative banks, config, bank store and Stage2Model."""
    from stage2.bank import NormativeBankStore
    from stage2.model import Stage2Model

    subjects: dict[str, int] = {
        **{sid: 0 for sid in DEFAULT_HC_IDS},
        **{sid: 1 for sid in DEFAULT_SZ_IDS},
        **{sid: 0 for sid in DEFAULT_FORBIDDEN_IDS[:1]},
        **{sid: 1 for sid in DEFAULT_FORBIDDEN_IDS[1:]},
    }
    train_ids = DEFAULT_HC_IDS + DEFAULT_SZ_IDS
    observed = {sid: list(range(20)) for sid in subjects}
    processed = make_processed_fixture(tmp_path, subjects, observed, with_dino=True)
    cv_root = make_cv_root(tmp_path, subjects, train_ids, DEFAULT_FORBIDDEN_IDS)
    make_cv_metadata(cv_root, processed)
    dino_root = make_dino_placeholder(tmp_path, processed)
    checkpoint, checkpoint_sha = make_stage1_checkpoint(tmp_path, fold=0, seed=seed)
    bank_root, registry = make_bank_fixture(
        tmp_path,
        processed_root=processed,
        cv_root=cv_root,
        hc_ids=DEFAULT_HC_IDS,
        sz_ids=DEFAULT_SZ_IDS,
        forbidden_ids=DEFAULT_FORBIDDEN_IDS,
        crossfit_splits=crossfit_splits,
        include_token=include_token,
        dino_root=dino_root,
        checkpoint=checkpoint,
        checkpoint_sha=checkpoint_sha,
    )
    cfg = make_stage2_config(
        processed_root=processed,
        cv_root=cv_root,
        bank_root=bank_root,
        registry_path=registry,
        model=model,
        loss=loss,
        subsets=subsets,
    )
    bank_store = NormativeBankStore(cfg, 0)
    stage2_model = Stage2Model(cfg, bank_store) if build_model else None
    return {
        "processed": processed,
        "cv_root": cv_root,
        "bank_root": bank_root,
        "registry": registry,
        "checkpoint": checkpoint,
        "cfg": cfg,
        "bank_store": bank_store,
        "model": stage2_model,
    }


def make_synthetic_batch(
    batch_size: int = 4,
    *,
    valid_counts: list[int] | None = None,
    seed: int = 0,
    labels: list[float] | None = None,
    bank_split_ids: list[int] | None = None,
    device: str = "cpu",
):
    """A hand-built :class:`Stage2Batch` with random heatmaps and missing slots.

    Valid trials occupy the first ``valid_counts[i]`` canonical slots of each
    subject (stimuli 0..19 cover all four categories), so at least two
    subjects can be given missing trials by construction.
    """
    from stage2.collate import Stage2Batch

    torch.manual_seed(seed)
    if valid_counts is None:
        valid_counts = [20, 20, 16, 17]
    valid_counts = list(valid_counts)[:batch_size]
    if labels is None:
        labels = [0.0, 0.0, 1.0, 1.0]
    labels = list(labels)[:batch_size]
    if bank_split_ids is None:
        bank_split_ids = [i % 2 for i in range(batch_size)]
    bank_split_ids = list(bank_split_ids)[:batch_size]
    heatmaps = torch.randn(batch_size, 100, 3, 48, 64)
    heatmaps[:, :2] = heatmaps[:, :2].abs()
    mask = torch.zeros(batch_size, 100, dtype=torch.bool)
    for i, n in enumerate(valid_counts):
        mask[i, :n] = True
    stimulus_indices = torch.arange(100, dtype=torch.int64).expand(batch_size, -1).clone()
    category_ids = torch.arange(100, dtype=torch.int64) % 4
    category_ids = category_ids.expand(batch_size, -1).clone()
    uids = tuple(
        tuple((f"{i:03d}_{s:03d}" if bool(mask[i, s]) else None) for s in range(100))
        for i in range(batch_size)
    )
    return Stage2Batch(
        subject_ids=tuple(str(i).zfill(3) for i in range(batch_size)),
        labels=torch.tensor(labels, dtype=torch.float32),
        heatmaps=heatmaps,
        trial_mask=mask,
        stimulus_indices=stimulus_indices,
        category_ids=category_ids,
        bank_split_ids=torch.tensor(bank_split_ids, dtype=torch.int64),
        trial_uids=uids,
    ).to_device(device)


def make_stage2_config(
    *,
    processed_root: Path,
    cv_root: Path,
    bank_root: Path,
    registry_path: Path,
    train_mode: str = "crossfit",
    model: dict[str, Any] | None = None,
    loss: dict[str, Any] | None = None,
    subsets: dict[str, Any] | None = None,
    **overrides: Any,
):
    from stage2.config import Stage2Config

    raw: dict[str, Any] = {
        "seed": 2026,
        "fold": 0,
        "bank": {
            "root": str(bank_root),
            "train_mode": train_mode,
            "verify_checksums": True,
            "checkpoint_registry": str(registry_path),
        },
        "sampler": {"subject_batch_size": 4, "balance_groups": True, "drop_last": False},
        "runtime": {"num_workers": 0, "pin_memory": False, "persistent_workers": False},
        "paths": {
            "processed_root": str(processed_root),
            "cv_root": str(cv_root),
            "normative_bank_root": str(bank_root),
        },
    }
    if model is not None:
        raw["model"] = dict(model)
    if loss is not None:
        raw["loss"] = dict(loss)
    if subsets is not None:
        raw["subsets"] = dict(subsets)
    for key, value in overrides.items():
        if key in ("seed", "fold"):
            raw[key] = value
        elif key in ("batch_size",):
            raw["sampler"]["subject_batch_size"] = value
    return Stage2Config.from_dict(raw)
