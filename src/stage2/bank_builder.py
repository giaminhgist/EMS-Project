"""Phase-1 Stage-2 normative bank builder (guide 03, contracts §6-§10).

Builds, for each fold, the full outer-training-HC normative bank and four
cross-fitted training banks, using unmasked Stage-1 inference over
post-fusion trial embeddings and fused tokens. float64 accumulation,
float32 storage, atomic publication, audited before rename.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import yaml

from preprocessing.storage import (
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_npy,
    sha256_of_file,
)

from .contracts import (
    ARRAY_DTYPES,
    BANK_SCHEMA_VERSION,
    EVALUATION_REGIMES,
    REGISTRY_SCHEMA_VERSION,
    build_feature_manifest,
    expected_array_shapes,
    feature_manifest_csv_bytes,
    sha256_of_bytes,
    verify_stimulus_order,
)

log = logging.getLogger("stage2.bank_builder")


class BankBuildError(ValueError):
    pass


class BankVerifyError(ValueError):
    pass


# --------------------------------------------------------------------- config


@dataclass(frozen=True)
class BankConfig:
    seed: int = 2026
    estimator: str = "mean"
    epsilon: float = 1e-6
    min_samples: int = 2
    batch_size: int = 64
    include_fused_token_bank: bool = True
    include_heatmap_token_bank: bool = False
    crossfit_splits: int = 4
    crossfit_enabled: bool = True
    device: str = "cpu"
    output_root: Path | None = None
    processed_root: Path | None = None
    dino_root: Path | None = None
    cv_root: Path | None = None

    def validate(self) -> None:
        if self.seed < 0:
            raise BankBuildError("seed must be non-negative")
        if self.estimator != "mean":
            raise BankBuildError("only estimator='mean' is supported")
        if self.epsilon <= 0 or self.min_samples < 1:
            raise BankBuildError("epsilon must be > 0; min_samples >= 1")
        if self.batch_size <= 0:
            raise BankBuildError("batch_size must be positive")
        if self.device not in ("cpu", "cuda"):
            raise BankBuildError("device must be 'cpu' or 'cuda'")
        if self.crossfit_enabled and self.crossfit_splits < 2:
            raise BankBuildError("crossfit_splits must be >= 2 when crossfit is enabled")
        if self.output_root is None:
            raise BankBuildError("output_root is required")

    @classmethod
    def from_dict(cls, raw: dict[str, Any], source: str = "<dict>") -> "BankConfig":
        known = {
            "seed", "estimator", "epsilon", "min_samples", "batch_size",
            "include_fused_token_bank", "include_heatmap_token_bank",
            "crossfit_splits", "crossfit_enabled", "device", "output_root",
            "processed_root", "dino_root", "cv_root",
        }
        unknown = sorted(set(raw) - known)
        if unknown:
            raise BankBuildError(f"{source}: unknown bank config fields: {unknown}")
        cfg = cls(
            seed=int(raw.get("seed", 2026)),
            estimator=str(raw.get("estimator", "mean")),
            epsilon=float(raw.get("epsilon", 1e-6)),
            min_samples=int(raw.get("min_samples", 2)),
            batch_size=int(raw.get("batch_size", 64)),
            include_fused_token_bank=bool(raw.get("include_fused_token_bank", True)),
            include_heatmap_token_bank=bool(raw.get("include_heatmap_token_bank", False)),
            crossfit_splits=int(raw.get("crossfit_splits", 4)),
            crossfit_enabled=bool(raw.get("crossfit_enabled", True)),
            device=str(raw.get("device", "cpu")),
            output_root=Path(raw["output_root"]) if raw.get("output_root") else None,
            processed_root=Path(raw["processed_root"]) if raw.get("processed_root") else None,
            dino_root=Path(raw["dino_root"]) if raw.get("dino_root") else None,
            cv_root=Path(raw["cv_root"]) if raw.get("cv_root") else None,
        )
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml(cls, path: Path | str) -> "BankConfig":
        p = Path(path)
        if not p.is_file():
            raise BankBuildError(f"bank config not found: {p}")
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise BankBuildError(f"{p}: bank configuration must be a mapping")
        return cls.from_dict(raw, source=str(p))

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "seed": self.seed,
            "estimator": self.estimator,
            "epsilon": self.epsilon,
            "min_samples": self.min_samples,
            "batch_size": self.batch_size,
            "include_fused_token_bank": self.include_fused_token_bank,
            "include_heatmap_token_bank": self.include_heatmap_token_bank,
            "crossfit_splits": self.crossfit_splits,
            "crossfit_enabled": self.crossfit_enabled,
            "device": self.device,
        }
        for name in ("output_root", "processed_root", "dino_root", "cv_root"):
            value = getattr(self, name)
            out[name] = str(value) if value is not None else None
        return out


# --------------------------------------------------------- checkpoint registry


def load_checkpoint_registry(path: Path | str) -> dict[str, Any]:
    """Load and validate the Stage-1 checkpoint registry (guide §3)."""
    p = Path(path)
    if not p.is_file():
        raise BankBuildError(f"checkpoint registry not found: {p}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise BankBuildError(f"{p}: registry must be a mapping")
    if raw.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise BankBuildError(
            f"registry schema_version must be {REGISTRY_SCHEMA_VERSION}, got {raw.get('schema_version')!r}"
        )
    regime = raw.get("evaluation_regime")
    if regime not in EVALUATION_REGIMES:
        raise BankBuildError(f"unknown evaluation_regime: {regime!r}")
    folds_raw = raw.get("folds")
    if not isinstance(folds_raw, dict):
        raise BankBuildError("registry must define exactly folds 0-4")
    folds = {str(k): v for k, v in folds_raw.items()}
    if set(folds) != {"0", "1", "2", "3", "4"}:
        raise BankBuildError("registry must define exactly folds 0-4")
    for key, entry in folds.items():
        if not isinstance(entry, dict) or not entry.get("checkpoint") or not entry.get("sha256"):
            raise BankBuildError(f"registry fold {key}: checkpoint and sha256 are required")
        if len(str(entry["sha256"])) != 64:
            raise BankBuildError(f"registry fold {key}: sha256 must be 64 hex characters")
    return {"schema_version": REGISTRY_SCHEMA_VERSION, "evaluation_regime": regime, "folds": folds}


# --------------------------------------------------------- streaming statistics


class StreamingAccumulator:
    """Grouped float64 sum/sumsq/count accumulation of ``[n, ...]`` features.

    Statistics are accumulated in float64 and cast to float32 only at
    finalization, matching the Stage-2 contract §6.1.
    """

    def __init__(self, n_stimuli: int, feature_shape: tuple[int, ...]):
        self.n_stimuli = n_stimuli
        self.feature_shape = tuple(feature_shape)
        self.sum = np.zeros((n_stimuli,) + self.feature_shape, dtype=np.float64)
        self.sumsq = np.zeros_like(self.sum)
        self.count = np.zeros(n_stimuli, dtype=np.int64)

    def add(self, stimulus_indices: np.ndarray, features: np.ndarray) -> None:
        si = np.asarray(stimulus_indices)
        feat = np.asarray(features, dtype=np.float64)
        if si.ndim != 1 or len(si) != feat.shape[0]:
            raise BankBuildError(
                f"stimulus_indices length {len(si)} does not match feature rows {feat.shape[0]}"
            )
        if feat.shape[1:] != self.feature_shape:
            raise BankBuildError(f"feature shape {feat.shape[1:]} != {self.feature_shape}")
        if not np.all(np.isfinite(feat)):
            raise BankBuildError("non-finite feature values during accumulation")
        if np.any(si < 0) or np.any(si >= self.n_stimuli):
            raise BankBuildError("stimulus index out of manifest range")
        si64 = si.astype(np.int64)
        np.add.at(self.sum, si64, feat)
        np.add.at(self.sumsq, si64, feat * feat)
        np.add.at(self.count, si64, np.ones(len(si64), dtype=np.int64))

    @property
    def n_rows(self) -> int:
        return int(self.count.sum())

    def finalize(self, *, epsilon: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(mu float32, sigma float32, count int32)``.

        Zero-count stimuli keep mu=0 and sigma=epsilon (never gathered by
        downstream code; coverage is asserted by the caller).
        """
        ndim = len(self.feature_shape)
        denom = np.maximum(self.count, 1).reshape((self.n_stimuli,) + (1,) * ndim)
        mu64 = self.sum / denom
        var = self.sumsq / denom - mu64 * mu64
        var = np.maximum(var, epsilon**2)
        return mu64.astype(np.float32), np.sqrt(var).astype(np.float32), self.count.astype(np.int32)


# ----------------------------------------------------------- crossfit assignment


def assign_crossfit_splits(
    train_subjects: pd.DataFrame,
    *,
    seed: int,
    fold: int,
    n_splits: int,
    forbidden_subject_ids: set[str],
) -> pd.DataFrame:
    """Deterministic label-stratified crossfit assignment (guide §10.1).

    ``train_subjects`` needs columns ``subject_id`` (str), ``label`` (0/1) and
    ``panel_trial_count`` (ok trials in the fold training partition). Subjects
    are sorted by (panel_trial_count desc, subject_id) and rotated into splits,
    balancing both group counts and panel completeness across splits.
    """
    required = {"subject_id", "label", "panel_trial_count"}
    if not required <= set(train_subjects.columns):
        raise BankBuildError(f"train_subjects must have columns {sorted(required)}")
    forbidden = forbidden_subject_ids & set(train_subjects.subject_id.astype(str))
    if forbidden:
        raise BankBuildError(f"validation subjects in training assignment: {sorted(forbidden)}")

    rng = np.random.default_rng(seed * 31 + fold)
    rows: list[dict[str, Any]] = []
    for label in (0, 1):
        group = train_subjects[train_subjects.label.astype(int) == label].copy()
        group["subject_id"] = group.subject_id.astype(str)
        group["panel_trial_count"] = group.panel_trial_count.astype(int)
        order = np.lexsort((group.subject_id.to_numpy(), -group.panel_trial_count.to_numpy()))
        ids = group.subject_id.to_numpy()[order]
        rotation = int(rng.integers(0, n_splits))
        for i, sid in enumerate(ids):
            rows.append(
                {
                    "subject_id": str(sid),
                    "label": int(label),
                    "bank_split_id": int((i + rotation) % n_splits),
                    "is_hc_bank_contributor": bool(label == 0),
                    "panel_trial_count": int(group.panel_trial_count.iloc[order[i]]),
                }
            )
    assignment = pd.DataFrame(rows).sort_values("subject_id").reset_index(drop=True)
    counts = assignment.groupby(["label", "bank_split_id"]).size().unstack(fill_value=0)
    if (counts.to_numpy().max(axis=1) - counts.to_numpy().min(axis=1) > 1).any():
        raise BankBuildError(f"crossfit split sizes are not balanced:\n{counts}")
    return assignment


# ----------------------------------------------------------------- bank arrays


@dataclass
class BankArrays:
    mu_trial: np.ndarray
    sigma_trial: np.ndarray
    count_trial: np.ndarray
    mu_token: np.ndarray | None = None
    sigma_token: np.ndarray | None = None
    mu_heat_token: np.ndarray | None = None
    sigma_heat_token: np.ndarray | None = None

    def array_items(self) -> dict[str, np.ndarray | None]:
        return {
            "mu_trial": self.mu_trial,
            "sigma_trial": self.sigma_trial,
            "count_trial": self.count_trial,
            "mu_token": self.mu_token,
            "sigma_token": self.sigma_token,
            "mu_heat_token": self.mu_heat_token,
            "sigma_heat_token": self.sigma_heat_token,
        }


@dataclass
class FoldBankBuild:
    fold: int
    full: BankArrays
    crossfit: dict[int, BankArrays]
    contributors_full: set[str]
    contributors_crossfit: dict[int, set[str]]
    excluded_crossfit: dict[int, set[str]]
    assignments: pd.DataFrame
    n_trials_full: int
    n_trials_crossfit: dict[int, int]


def _to_float64(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    return np.asarray(value, dtype=np.float64)


def build_fold_banks(
    model: Any,
    dataset: Any,
    *,
    fold: int,
    cfg: BankConfig,
    n_stimuli: int,
    forbidden_subject_ids: set[str],
    assignments: pd.DataFrame,
) -> FoldBankBuild:
    """One streaming pass over all training-HC trials builds the full bank and
    every crossfit bank simultaneously (guide §7-§10).

    ``dataset`` must expose ``trial_rows`` (DataFrame with ``stimulus_index``),
    ``row_subject_ids()`` aligned with row order and ``collate_from_indices``.
    ``model`` must accept ``(batch, token_mask=None, return_fused=..., return_heatmap_tokens=...)``.
    """
    trial_rows = dataset.trial_rows
    indices = list(range(len(trial_rows)))
    subject_ids = [str(x) for x in dataset.row_subject_ids()]
    if "group" in trial_rows.columns and (trial_rows.group != "HC").any():
        raise BankBuildError("bank building requires HC-only training rows")
    contributors = set(subject_ids)
    if contributors & forbidden_subject_ids:
        raise BankBuildError(
            f"forbidden (validation) subjects contributed to the bank: "
            f"{sorted(contributors & forbidden_subject_ids)}"
        )

    n_splits = cfg.crossfit_splits if cfg.crossfit_enabled else 0
    split_of: dict[str, int] = {}
    excluded: dict[int, set[str]] = {}
    if n_splits:
        if assignments is None or not {"subject_id", "label", "bank_split_id"} <= set(assignments.columns):
            raise BankBuildError("crossfit enabled but no valid subject assignment was provided")
        label_of = {str(r.subject_id): int(r.label) for r in assignments.itertuples()}
        split_of = {str(r.subject_id): int(r.bank_split_id) for r in assignments.itertuples()}
        missing_assignment = contributors - set(split_of)
        if missing_assignment:
            raise BankBuildError(
                f"HC contributors missing crossfit assignment: {sorted(missing_assignment)}"
            )
        excluded = {
            j: {sid for sid, jj in split_of.items() if jj == j and label_of.get(sid) == 0}
            for j in range(n_splits)
        }

    full_acc: StreamingAccumulator | None = None
    full_token_acc: StreamingAccumulator | None = None
    full_heat_acc: StreamingAccumulator | None = None
    split_accs: dict[int, StreamingAccumulator] = {}
    split_token_accs: dict[int, StreamingAccumulator] = {}
    split_heat_accs: dict[int, StreamingAccumulator] = {}
    split_contributors: dict[int, set[str]] = {j: set() for j in range(n_splits)}
    trial_shape: tuple[int, ...] | None = None
    token_shape: tuple[int, ...] | None = None

    with torch.inference_mode():
        for start in range(0, len(indices), cfg.batch_size):
            batch_indices = indices[start : start + cfg.batch_size]
            batch = dataset.collate_from_indices(batch_indices)
            if hasattr(batch, "to_device"):
                batch.to_device(cfg.device)
            out = model(
                batch,
                token_mask=None,
                return_fused=cfg.include_fused_token_bank,
                return_heatmap_tokens=cfg.include_heatmap_token_bank,
            )
            emb = _to_float64(out.trial_embedding)
            fused = _to_float64(out.fused_tokens)
            heat_tok = _to_float64(out.heatmap_tokens)
            if emb is None or emb.ndim < 2:
                raise BankBuildError("model returned no usable trial embeddings")

            # Accumulator shapes derive from the first observed batch.
            if full_acc is None:
                trial_shape = tuple(emb.shape[1:])
                full_acc = StreamingAccumulator(n_stimuli, trial_shape)
                if cfg.include_fused_token_bank:
                    if fused is None or fused.ndim < 3:
                        raise BankBuildError("fused-token bank requested but model returned no fused tokens")
                    token_shape = tuple(fused.shape[1:])
                    full_token_acc = StreamingAccumulator(n_stimuli, token_shape)
                if cfg.include_heatmap_token_bank:
                    if heat_tok is None or heat_tok.ndim < 3:
                        raise BankBuildError("heatmap-token bank requested but model returned no heatmap tokens")
                    token_shape = tuple(heat_tok.shape[1:])
                    full_heat_acc = StreamingAccumulator(n_stimuli, token_shape)
                split_accs = {j: StreamingAccumulator(n_stimuli, trial_shape) for j in range(n_splits)}
                if cfg.include_fused_token_bank:
                    split_token_accs = {j: StreamingAccumulator(n_stimuli, token_shape) for j in range(n_splits)}
                if cfg.include_heatmap_token_bank:
                    split_heat_accs = {j: StreamingAccumulator(n_stimuli, token_shape) for j in range(n_splits)}

            if emb.shape[1:] != trial_shape:
                raise BankBuildError(f"trial embedding shape changed mid-pass: {emb.shape[1:]} != {trial_shape}")
            if fused is not None and (token_shape is None or fused.shape[1:] != token_shape):
                raise BankBuildError(f"fused token shape changed mid-pass: {None if fused is None else fused.shape[1:]}")
            si = np.array([int(trial_rows.iloc[i].stimulus_index) for i in batch_indices])
            subs = [subject_ids[i] for i in batch_indices]

            full_acc.add(si, emb)
            if full_token_acc is not None and fused is not None:
                full_token_acc.add(si, fused)
            if full_heat_acc is not None and heat_tok is not None:
                full_heat_acc.add(si, heat_tok)
            for j in range(n_splits):
                keep = np.array([s not in excluded[j] for s in subs], dtype=bool)
                if not keep.any():
                    continue
                split_accs[j].add(si[keep], emb[keep])
                if split_token_accs and fused is not None:
                    split_token_accs[j].add(si[keep], fused[keep])
                if split_heat_accs and heat_tok is not None:
                    split_heat_accs[j].add(si[keep], heat_tok[keep])
                split_contributors[j].update(s for s, k in zip(subs, keep) if k)

    if full_acc is None:
        raise BankBuildError("no training rows were available for bank building")

    def finalize(acc: StreamingAccumulator, token_acc: StreamingAccumulator | None,
                 heat_acc: StreamingAccumulator | None, label: str) -> tuple[BankArrays, int]:
        mu, sigma, count = acc.finalize(epsilon=cfg.epsilon)
        mu_tok = sigma_tok = mu_heat = sigma_heat = None
        if token_acc is not None:
            mu_tok, sigma_tok, _ = token_acc.finalize(epsilon=cfg.epsilon)
        if heat_acc is not None:
            mu_heat, sigma_heat, _ = heat_acc.finalize(epsilon=cfg.epsilon)
        n = int(count.sum())
        if np.any(count < cfg.min_samples):
            bad = [int(i) for i in range(n_stimuli) if count[i] < cfg.min_samples]
            raise BankBuildError(f"{label}: stimuli below min_samples={cfg.min_samples}: {bad}")
        return BankArrays(mu, sigma, count, mu_tok, sigma_tok, mu_heat, sigma_heat), n

    full_arrays, n_full = finalize(full_acc, full_token_acc, full_heat_acc, f"fold {fold} full bank")
    crossfit_arrays: dict[int, BankArrays] = {}
    n_crossfit: dict[int, int] = {}
    for j in range(n_splits):
        crossfit_arrays[j], n_crossfit[j] = finalize(
            split_accs[j], split_token_accs.get(j), split_heat_accs.get(j),
            f"fold {fold} crossfit split {j}",
        )
        if split_contributors[j] & excluded[j]:
            raise BankBuildError(
                f"fold {fold} split {j}: excluded HC subjects contributed: "
                f"{sorted(split_contributors[j] & excluded[j])}"
            )

    return FoldBankBuild(
        fold=fold,
        full=full_arrays,
        crossfit=crossfit_arrays,
        contributors_full=set(contributors),
        contributors_crossfit=split_contributors,
        excluded_crossfit=excluded,
        assignments=assignments,
        n_trials_full=n_full,
        n_trials_crossfit=n_crossfit,
    )


# ----------------------------------------------------- real checkpoint resolution


def build_fold_from_checkpoint(
    *,
    fold: int,
    registry_entry: dict[str, str],
    evaluation_regime: str,
    cfg: BankConfig,
) -> tuple[FoldBankBuild, dict[str, Any]]:
    """Resolve the approved checkpoint and build all banks for one fold (guide §6)."""
    from stage1.checkpoint import load_checkpoint
    from stage1.config import Stage1Config
    from stage1.dataset import Stage1Dataset
    from stage1.experiment import source_checksums
    from stage1.model import Stage1Model

    checkpoint_path = Path(registry_entry["checkpoint"])
    if not checkpoint_path.is_file():
        raise BankBuildError(f"checkpoint not found: {checkpoint_path}")
    actual_sha = sha256_of_file(checkpoint_path)
    expected_sha = str(registry_entry["sha256"])
    if actual_sha != expected_sha:
        raise BankBuildError(
            f"checkpoint SHA-256 mismatch for {checkpoint_path}:\n"
            f"  registry {expected_sha}\n  actual   {actual_sha}"
        )
    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        raise BankBuildError("--device cuda requested but CUDA is not available")

    contents = load_checkpoint(checkpoint_path, device)
    meta = contents.meta
    if meta.get("fold") != fold:
        raise BankBuildError(
            f"checkpoint fold {meta.get('fold')} does not match requested fold {fold}"
        )
    stage1_cfg = Stage1Config.from_dict(meta["config_resolved"], source=str(checkpoint_path))

    current_checksums = source_checksums(stage1_cfg)
    recorded = meta.get("source_checksums") or {}
    for key, expected in current_checksums.items():
        if recorded.get(key) != expected:
            raise BankBuildError(f"checkpoint source checksum {key} does not match current inputs")

    kwargs = dict(
        verify_checksums=True,
        fold=fold,
        active_channels=tuple(stage1_cfg.model.active_channels),
    )
    train_ds = Stage1Dataset(
        stage1_cfg.paths.processed_root, stage1_cfg.paths.dino_root,
        stage1_cfg.paths.cv_fold_dir, "train", **kwargs,
    )
    val_ds = Stage1Dataset(
        stage1_cfg.paths.processed_root, stage1_cfg.paths.dino_root,
        stage1_cfg.paths.cv_fold_dir, "val", **kwargs,
    )
    train_ids = set(train_ds.row_subject_ids())
    val_ids = set(val_ds.row_subject_ids())
    if train_ids & val_ids:
        raise BankBuildError(f"train/validation subject overlap in fold {fold}")

    partition_dir = stage1_cfg.paths.cv_fold_dir
    train_partition = pd.read_parquet(partition_dir / "train_trials.parquet")
    train_partition["subject_id"] = train_partition.subject_id.astype(str)
    val_partition = pd.read_parquet(partition_dir / "val_trials.parquet")
    val_partition["subject_id"] = val_partition.subject_id.astype(str)
    forbidden = set(val_partition.subject_id)  # all held-out subjects, HC and SZ

    ok_rows = train_partition[train_partition.qc_status == "ok"]
    panel = ok_rows.groupby("subject_id").agg(
        label=("label", "first"), panel_trial_count=("trial_uid", "size")
    ).reset_index()
    assignments = assign_crossfit_splits(
        panel,
        seed=cfg.seed,
        fold=fold,
        n_splits=cfg.crossfit_splits if cfg.crossfit_enabled else 1,
        forbidden_subject_ids=forbidden,
    ) if cfg.crossfit_enabled else None

    model = Stage1Model(stage1_cfg)
    model.load_state_dict(contents.state_dict)
    model.to(device)
    model.eval()

    image_manifest = pd.read_csv(
        stage1_cfg.paths.processed_root / "image_manifest.csv",
        dtype={"stimulus_id": str, "category": str},
    )
    dino_feature_manifest = pd.read_csv(
        stage1_cfg.paths.dino_root / "feature_manifest.csv", dtype=str
    )
    feature_manifest = build_feature_manifest(image_manifest)
    verify_stimulus_order(feature_manifest, dino_feature_manifest, image_manifest)

    n_stimuli = len(image_manifest)
    build = build_fold_banks(
        model,
        train_ds,
        fold=fold,
        cfg=cfg,
        n_stimuli=n_stimuli,
        forbidden_subject_ids=forbidden,
        assignments=assignments,
    )

    provenance: dict[str, Any] = {
        "stage1_checkpoint_path": str(checkpoint_path),
        "stage1_checkpoint_sha256": actual_sha,
        "stage1_run_id": meta.get("run_id"),
        "stage1_config_hash": meta.get("config_hash"),
        "source_checksums": dict(recorded),
        "processed_dataset_metadata_sha256": recorded.get("processed_dataset_metadata_sha256"),
        "dino_patch_tokens_sha256": recorded.get("dino_patch_tokens_sha256"),
        "dino_feature_manifest_sha256": sha256_of_file(stage1_cfg.paths.dino_root / "feature_manifest.csv"),
        "cv_partition_sha256": {
            "train_trials": sha256_of_file(partition_dir / "train_trials.parquet"),
            "val_trials": sha256_of_file(partition_dir / "val_trials.parquet"),
        },
        "feature_manifest": feature_manifest,
        "forbidden_validation_subject_ids": sorted(forbidden),
    }
    return build, provenance


# ------------------------------------------------------------------ publication


def _is_complete_bank_dir(bank_dir: Path, expected_shapes: dict[str, tuple[int, ...]]) -> bool:
    if not bank_dir.is_dir():
        return False
    if not (bank_dir / "metadata.json").is_file() or not (bank_dir / "audit.json").is_file():
        return False
    for name in expected_shapes:
        if not (bank_dir / f"{name}.npy").is_file():
            return False
    return True


def build_metadata(
    *,
    fold: int,
    crossfit_split: int | None,
    seed: int,
    evaluation_regime: str,
    estimator: str,
    epsilon: float,
    min_samples: int,
    include_fused_token_bank: bool,
    include_heatmap_token_bank: bool,
    provenance: dict[str, Any],
    contributing_hc_subject_ids: list[str],
    forbidden_validation_subject_ids: list[str],
    n_contributing_subjects: int,
    n_trials: int,
    array_shapes: dict[str, list[int]],
    array_sha256: dict[str, str],
    stimulus_manifest_sha256: str,
    crossfit_excluded_hc_subject_ids: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": BANK_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fold": fold,
        "crossfit_split": crossfit_split,
        "seed": seed,
        "evaluation_regime": evaluation_regime,
        "estimator": estimator,
        "epsilon": epsilon,
        "min_samples": min_samples,
        "include_fused_token_bank": include_fused_token_bank,
        "include_heatmap_token_bank": include_heatmap_token_bank,
        "stage1_checkpoint_path": provenance["stage1_checkpoint_path"],
        "stage1_checkpoint_sha256": provenance["stage1_checkpoint_sha256"],
        "stage1_run_id": provenance["stage1_run_id"],
        "stage1_config_hash": provenance["stage1_config_hash"],
        "processed_manifest_sha256": provenance.get("processed_dataset_metadata_sha256"),
        "dino_feature_sha256": provenance.get("dino_feature_manifest_sha256"),
        "cv_partition_sha256": provenance.get("cv_partition_sha256"),
        "source_checksums": provenance.get("source_checksums"),
        "contributing_hc_subject_ids": contributing_hc_subject_ids,
        "forbidden_validation_subject_ids": forbidden_validation_subject_ids,
        "crossfit_excluded_hc_subject_ids": crossfit_excluded_hc_subject_ids or [],
        "n_contributing_subjects": n_contributing_subjects,
        "n_trials": n_trials,
        "array_shapes": array_shapes,
        "array_sha256": array_sha256,
        "stimulus_manifest_sha256": stimulus_manifest_sha256,
    }


def reload_and_audit(
    bank_dir: Path,
    *,
    expected_shapes: dict[str, tuple[int, ...]],
    min_samples: int,
    epsilon: float,
    forbidden_validation_subject_ids: list[str] | None = None,
    crossfit_excluded_hc_subject_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Reload saved arrays from ``bank_dir`` and run every Phase-1 assertion."""
    shapes: dict[str, list[int]] = {}
    shas: dict[str, str] = {}
    checks: dict[str, Any] = {"arrays_finite": True, "sigma_at_least_epsilon": True}
    for name, shape in expected_shapes.items():
        path = bank_dir / f"{name}.npy"
        if not path.is_file():
            raise BankVerifyError(f"missing array: {path}")
        arr = np.load(path, mmap_mode="r")
        if tuple(arr.shape) != tuple(shape):
            raise BankVerifyError(f"{path}: shape {arr.shape} != {shape}")
        if str(arr.dtype) != str(ARRAY_DTYPES[name]):
            raise BankVerifyError(f"{path}: dtype {arr.dtype} != {ARRAY_DTYPES[name]}")
        if not np.isfinite(arr).all():
            checks["arrays_finite"] = False
        if name == "sigma_trial":
            if np.any(arr < epsilon - 1e-12):
                checks["sigma_at_least_epsilon"] = False
        shapes[name] = list(shape)
        shas[name] = sha256_of_file(path)

    metadata_path = bank_dir / "metadata.json"
    if not metadata_path.is_file():
        raise BankVerifyError(f"missing metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    recorded_shas = metadata.get("array_sha256") or {}
    if set(recorded_shas) != set(shas) or any(recorded_shas[k] != shas[k] for k in shas):
        raise BankVerifyError(f"{bank_dir}: array checksums do not match metadata")

    counts = np.load(bank_dir / "count_trial.npy", mmap_mode="r")
    checks["all_stimuli_covered"] = bool(np.all(counts >= min_samples))
    checks["n_trials"] = int(counts.sum())
    contributors = set(metadata.get("contributing_hc_subject_ids") or [])
    forbidden = set(forbidden_validation_subject_ids or metadata.get("forbidden_validation_subject_ids") or [])
    checks["no_forbidden_contributors"] = not bool(contributors & forbidden)
    excluded = set(crossfit_excluded_hc_subject_ids or metadata.get("crossfit_excluded_hc_subject_ids") or [])
    checks["crossfit_self_exclusion_ok"] = not bool(contributors & excluded)
    if (metadata.get("crossfit_split") is not None) and excluded:
        checks["crossfit_exclusion_expected"] = True
    checks["complete"] = all(
        [
            checks["arrays_finite"],
            checks["sigma_at_least_epsilon"],
            checks["all_stimuli_covered"],
            checks["no_forbidden_contributors"],
            checks["crossfit_self_exclusion_ok"],
        ]
    )
    return {"shapes": shapes, "sha256": shas, "metadata": metadata, "checks": checks}


def publish_bank_dir(
    bank_dir: Path,
    arrays: BankArrays,
    *,
    feature_manifest: pd.DataFrame,
    metadata: dict[str, Any],
    expected_shapes: dict[str, tuple[int, ...]],
    min_samples: int,
    epsilon: float,
    overwrite_incomplete: bool = False,
) -> dict[str, Any]:
    """Atomically publish one bank directory (guide §11).

    All arrays, metadata and the reloaded audit are staged in a temporary
    sibling directory that is renamed into place only after every check
    passes. A verified complete bank is never silently overwritten.
    """
    bank_dir = Path(bank_dir)
    if bank_dir.exists():
        if _is_complete_bank_dir(bank_dir, expected_shapes):
            raise BankBuildError(
                f"bank already complete: {bank_dir} (remove it manually to rebuild)"
            )
        if not overwrite_incomplete:
            raise BankBuildError(
                f"bank exists but is incomplete: {bank_dir} (use --overwrite-incomplete to rebuild)"
            )
    staging = bank_dir.parent / f".{bank_dir.name}.{os.getpid()}.tmp"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=False)
    try:
        shas: dict[str, str] = {}
        for name, arr in arrays.array_items().items():
            if arr is None:
                continue
            shas[name] = atomic_write_npy(staging / f"{name}.npy", arr)
        manifest_bytes = feature_manifest_csv_bytes(feature_manifest)
        atomic_write_bytes(staging / "feature_manifest.csv", manifest_bytes)
        metadata["array_shapes"] = {k: list(v) for k, v in expected_shapes.items()}
        metadata["array_sha256"] = shas
        metadata["stimulus_manifest_sha256"] = sha256_of_bytes(manifest_bytes)
        atomic_write_json(staging / "metadata.json", metadata)

        audit = reload_and_audit(
            staging,
            expected_shapes=expected_shapes,
            min_samples=min_samples,
            epsilon=epsilon,
            forbidden_validation_subject_ids=metadata.get("forbidden_validation_subject_ids"),
            crossfit_excluded_hc_subject_ids=metadata.get("crossfit_excluded_hc_subject_ids"),
        )
        if not audit["checks"]["complete"]:
            raise BankVerifyError(f"{bank_dir}: audit failed: {audit['checks']}")
        audit["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(staging / "audit.json", audit)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if bank_dir.exists():
        old = bank_dir.parent / f".{bank_dir.name}.{os.getpid()}.old"
        os.replace(bank_dir, old)
        try:
            os.replace(staging, bank_dir)
        except BaseException:
            os.replace(old, bank_dir)
            raise
        shutil.rmtree(old, ignore_errors=True)
    else:
        os.replace(staging, bank_dir)
    log.info("published bank: %s", bank_dir)
    return audit


def is_fold_complete(fold_dir: Path, expected_shapes: dict[str, tuple[int, ...]], n_crossfit: int) -> bool:
    if not fold_dir.is_dir():
        return False
    if not _is_complete_bank_dir(fold_dir, expected_shapes):
        return False
    if not (fold_dir / "feature_manifest.csv").is_file():
        return False
    if n_crossfit:
        crossfit_dir = fold_dir / "crossfit"
        if not (crossfit_dir / "subject_assignment.csv").is_file():
            return False
        for j in range(n_crossfit):
            if not _is_complete_bank_dir(crossfit_dir / f"split_{j}", expected_shapes):
                return False
    return True


def write_root_manifest(
    output_root: Path,
    *,
    cfg: BankConfig,
    evaluation_regime: str,
    checkpoint_registry: dict[str, Any],
) -> dict[str, Any]:
    """Reconcile the on-disk state of every fold into the root manifest.

    This is the last file published; incomplete or missing folds are marked
    as such rather than claiming all-fold success (guide §11).
    """
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    expected_shapes = expected_array_shapes(
        cfg.include_fused_token_bank, cfg.include_heatmap_token_bank
    )
    n_crossfit = cfg.crossfit_splits if cfg.crossfit_enabled else 0
    manifest: dict[str, Any] = {
        "schema_version": BANK_SCHEMA_VERSION,
        "evaluation_regime": evaluation_regime,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bank_config": cfg.to_dict(),
        "folds": {},
    }
    for fold in range(5):
        fold_dir = output_root / f"fold_{fold}"
        entry: dict[str, Any] = {
            "path": str(fold_dir.relative_to(output_root)),
            "stage1_checkpoint": checkpoint_registry["folds"][str(fold)]["checkpoint"],
            "stage1_checkpoint_sha256": checkpoint_registry["folds"][str(fold)]["sha256"],
        }
        if is_fold_complete(fold_dir, expected_shapes, n_crossfit):
            entry["status"] = "complete"
            audit = reload_and_audit(
                fold_dir,
                expected_shapes=expected_shapes,
                min_samples=cfg.min_samples,
                epsilon=cfg.epsilon,
            )
            entry["n_contributing_hc_subjects"] = audit["metadata"].get("n_contributing_subjects")
            entry["n_trials"] = audit["checks"]["n_trials"]
            entry["arrays"] = {
                name: {"shape": shape, "sha256": audit["sha256"][name]}
                for name, shape in expected_shapes.items()
            }
            if n_crossfit:
                entry["crossfit_splits"] = n_crossfit
                entry["crossfit"] = {}
                for j in range(n_crossfit):
                    split_dir = fold_dir / "crossfit" / f"split_{j}"
                    split_audit = reload_and_audit(
                        split_dir,
                        expected_shapes=expected_shapes,
                        min_samples=cfg.min_samples,
                        epsilon=cfg.epsilon,
                    )
                    entry["crossfit"][f"split_{j}"] = {
                        "arrays": {
                            name: {"shape": shape, "sha256": split_audit["sha256"][name]}
                            for name, shape in expected_shapes.items()
                        }
                    }
        elif fold_dir.exists():
            entry["status"] = "incomplete"
        else:
            entry["status"] = "missing"
        manifest["folds"][str(fold)] = entry
    atomic_write_json(output_root / "manifest.json", manifest)
    return manifest


# -------------------------------------------------------------------- run/build


def build_all_folds(
    *,
    registry_path: Path,
    cfg: BankConfig,
    folds: list[int],
    overwrite_incomplete: bool = False,
) -> list[dict[str, Any]]:
    """Build the requested folds into ``cfg.output_root`` (guide §5, §6)."""
    registry = load_checkpoint_registry(registry_path)
    expected_shapes = expected_array_shapes(
        cfg.include_fused_token_bank, cfg.include_heatmap_token_bank
    )
    n_crossfit = cfg.crossfit_splits if cfg.crossfit_enabled else 0
    reports: list[dict[str, Any]] = []
    for fold in folds:
        fold_dir = cfg.output_root / f"fold_{fold}"
        if is_fold_complete(fold_dir, expected_shapes, n_crossfit):
            metadata_path = fold_dir / "metadata.json"
            existing_sha = json.loads(metadata_path.read_text(encoding="utf-8")).get(
                "stage1_checkpoint_sha256"
            )
            if existing_sha == str(registry["folds"][str(fold)]["sha256"]):
                log.info("fold %d already complete with the approved checkpoint; skipping", fold)
                reports.append({"fold": fold, "status": "skipped_complete"})
                continue
            raise BankBuildError(
                f"fold {fold} bank exists but was built from checkpoint {existing_sha}; "
                f"remove {fold_dir} manually to rebuild from the approved checkpoint"
            )
        if fold_dir.exists() and not overwrite_incomplete:
            raise BankBuildError(
                f"fold {fold} bank exists but is incomplete: {fold_dir} "
                f"(use --overwrite-incomplete to rebuild)"
            )
        if fold_dir.exists() and overwrite_incomplete:
            log.info("removing incomplete fold dir before rebuild: %s", fold_dir)
            shutil.rmtree(fold_dir)

        entry = registry["folds"][str(fold)]
        build, provenance = build_fold_from_checkpoint(
            fold=fold,
            registry_entry=entry,
            evaluation_regime=registry["evaluation_regime"],
            cfg=cfg,
        )
        # publish_bank_dir stages and renames fold_dir into place itself.
        feature_manifest = provenance["feature_manifest"]

        def _meta(crossfit_split: int | None, arrays: BankArrays,
                  contributors: set[str], n_trials: int,
                  excluded: list[str] | None = None) -> dict[str, Any]:
            return build_metadata(
                fold=fold,
                crossfit_split=crossfit_split,
                seed=cfg.seed,
                evaluation_regime=registry["evaluation_regime"],
                estimator=cfg.estimator,
                epsilon=cfg.epsilon,
                min_samples=cfg.min_samples,
                include_fused_token_bank=cfg.include_fused_token_bank,
                include_heatmap_token_bank=cfg.include_heatmap_token_bank,
                provenance=provenance,
                contributing_hc_subject_ids=sorted(contributors),
                forbidden_validation_subject_ids=provenance["forbidden_validation_subject_ids"],
                n_contributing_subjects=len(contributors),
                n_trials=n_trials,
                array_shapes={},
                array_sha256={},
                stimulus_manifest_sha256="",
                crossfit_excluded_hc_subject_ids=excluded,
            )

        publish_bank_dir(
            fold_dir,
            build.full,
            feature_manifest=feature_manifest,
            metadata=_meta(None, build.full, build.contributors_full, build.n_trials_full),
            expected_shapes=expected_shapes,
            min_samples=cfg.min_samples,
            epsilon=cfg.epsilon,
        )

        if n_crossfit:
            crossfit_dir = fold_dir / "crossfit"
            crossfit_dir.mkdir(parents=True, exist_ok=True)
            assignment_csv = build.assignments.to_csv(index=False).encode("utf-8")
            atomic_write_bytes(crossfit_dir / "subject_assignment.csv", assignment_csv)
            for j in range(n_crossfit):
                publish_bank_dir(
                    crossfit_dir / f"split_{j}",
                    build.crossfit[j],
                    feature_manifest=feature_manifest,
                    metadata=_meta(
                        j,
                        build.crossfit[j],
                        build.contributors_crossfit[j],
                        build.n_trials_crossfit[j],
                        excluded=sorted(build.excluded_crossfit[j]),
                    ),
                    expected_shapes=expected_shapes,
                    min_samples=cfg.min_samples,
                    epsilon=cfg.epsilon,
                )
        reports.append(
            {
                "fold": fold,
                "status": "built",
                "n_trials_full": build.n_trials_full,
                "n_trials_crossfit": build.n_trials_crossfit,
                "n_contributors": len(build.contributors_full),
            }
        )
    write_root_manifest(
        cfg.output_root,
        cfg=cfg,
        evaluation_regime=registry["evaluation_regime"],
        checkpoint_registry=registry,
    )
    return reports


# --------------------------------------------------------------------- verify


def verify_fold(
    fold: int,
    *,
    output_root: Path,
    cfg: BankConfig,
    registry: dict[str, Any],
) -> dict[str, Any]:
    """Read-only verification of one fold (guide §12)."""
    expected_shapes = expected_array_shapes(
        cfg.include_fused_token_bank, cfg.include_heatmap_token_bank
    )
    n_crossfit = cfg.crossfit_splits if cfg.crossfit_enabled else 0
    fold_dir = output_root / f"fold_{fold}"
    result: dict[str, Any] = {"fold": fold, "status": "ok", "checks": {}}
    if not fold_dir.is_dir():
        raise BankVerifyError(f"fold {fold}: directory missing: {fold_dir}")

    entry = registry["folds"][str(fold)]
    checkpoint_path = Path(entry["checkpoint"])
    actual_sha = sha256_of_file(checkpoint_path) if checkpoint_path.is_file() else None
    result["checks"]["checkpoint_sha256"] = actual_sha == str(entry["sha256"])
    if not result["checks"]["checkpoint_sha256"]:
        raise BankVerifyError(f"fold {fold}: checkpoint SHA-256 does not match the registry")

    full_audit = reload_and_audit(
        fold_dir,
        expected_shapes=expected_shapes,
        min_samples=cfg.min_samples,
        epsilon=cfg.epsilon,
    )
    result["full"] = {"shapes": full_audit["shapes"], "sha256": full_audit["sha256"]}
    result["checks"]["full_audit"] = full_audit["checks"]

    metadata = full_audit["metadata"]
    if metadata.get("fold") != fold:
        raise BankVerifyError(f"fold {fold}: metadata fold {metadata.get('fold')} mismatch")
    if metadata.get("stage1_checkpoint_sha256") != str(entry["sha256"]):
        raise BankVerifyError(f"fold {fold}: metadata checkpoint sha256 does not match registry")

    feature_manifest_path = fold_dir / "feature_manifest.csv"
    if not feature_manifest_path.is_file():
        raise BankVerifyError(f"fold {fold}: missing feature_manifest.csv")
    result["checks"]["stimulus_manifest_sha256"] = (
        sha256_of_file(feature_manifest_path) == metadata.get("stimulus_manifest_sha256")
    )
    if not result["checks"]["stimulus_manifest_sha256"]:
        raise BankVerifyError(f"fold {fold}: feature_manifest.csv checksum mismatch")

    subject_manifest = pd.read_csv(
        cfg.processed_root / "subject_manifest.csv", dtype={"subject_id": str}
    )
    label_by_id = dict(zip(subject_manifest.subject_id, subject_manifest.label.astype(int)))
    contributors = set(metadata.get("contributing_hc_subject_ids") or [])
    non_hc = {s for s in contributors if label_by_id.get(s) != 0}
    result["checks"]["contributors_all_hc"] = not non_hc
    if non_hc:
        raise BankVerifyError(f"fold {fold}: non-HC bank contributors: {sorted(non_hc)}")

    if n_crossfit:
        crossfit_dir = fold_dir / "crossfit"
        if not (crossfit_dir / "subject_assignment.csv").is_file():
            raise BankVerifyError(f"fold {fold}: missing crossfit/subject_assignment.csv")
        assignment = pd.read_csv(crossfit_dir / "subject_assignment.csv", dtype=str)
        result["crossfit"] = {}
        for j in range(n_crossfit):
            split_dir = crossfit_dir / f"split_{j}"
            audit = reload_and_audit(
                split_dir,
                expected_shapes=expected_shapes,
                min_samples=cfg.min_samples,
                epsilon=cfg.epsilon,
            )
            result["crossfit"][f"split_{j}"] = {"checks": audit["checks"]}
            if not audit["checks"]["complete"]:
                raise BankVerifyError(f"fold {fold}: crossfit split {j} audit failed")
            split_meta = audit["metadata"]
            if split_meta.get("fold") != fold or split_meta.get("crossfit_split") != j:
                raise BankVerifyError(f"fold {fold}: split {j} metadata identity mismatch")
            excluded = set(split_meta.get("crossfit_excluded_hc_subject_ids") or [])
            contrib = set(split_meta.get("contributing_hc_subject_ids") or [])
            if contrib & excluded:
                raise BankVerifyError(f"fold {fold} split {j}: self-inclusion detected")
            assigned = set(assignment[assignment.bank_split_id.astype(int) == j].subject_id)
            hc_assigned = assigned & contributors
            if hc_assigned != excluded:
                raise BankVerifyError(
                    f"fold {fold} split {j}: assignment/exclusion sets disagree "
                    f"({len(hc_assigned)} assigned vs {len(excluded)} excluded)"
                )
    return result


def verify_all(
    *,
    registry_path: Path,
    cfg: BankConfig,
    folds: list[int],
) -> dict[str, Any]:
    """Read-only verification of all requested folds; raises on any mismatch."""
    registry = load_checkpoint_registry(registry_path)
    results: dict[str, Any] = {}
    manifest_path = cfg.output_root / "manifest.json"
    expected_shapes = expected_array_shapes(
        cfg.include_fused_token_bank, cfg.include_heatmap_token_bank
    )
    n_crossfit = cfg.crossfit_splits if cfg.crossfit_enabled else 0
    if manifest_path.is_file():
        raw_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        results["root_manifest"] = raw_manifest
        # The root manifest must report the same content found on disk.
        for fold in folds:
            fold_dir = cfg.output_root / f"fold_{fold}"
            disk_status = (
                "complete"
                if is_fold_complete(fold_dir, expected_shapes, n_crossfit)
                else ("incomplete" if fold_dir.exists() else "missing")
            )
            recorded = raw_manifest.get("folds", {}).get(str(fold), {}).get("status")
            if recorded != disk_status:
                raise BankVerifyError(
                    f"root manifest status for fold {fold} is {recorded!r}, disk status is {disk_status!r}"
                )
    for fold in folds:
        results[str(fold)] = verify_fold(
            fold, output_root=cfg.output_root, cfg=cfg, registry=registry
        )
    # Stimulus manifests must be identical across the five folds.
    shas = {
        fold: sha256_of_file(cfg.output_root / f"fold_{fold}" / "feature_manifest.csv")
        for fold in folds
        if (cfg.output_root / f"fold_{fold}" / "feature_manifest.csv").is_file()
    }
    if len(set(shas.values())) > 1:
        raise BankVerifyError(f"feature manifests differ across folds: {shas}")
    results["feature_manifest_sha256_by_fold"] = shas
    return results
