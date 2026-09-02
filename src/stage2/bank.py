"""Read-only normative bank store for Stage 2 (Phase 2, guide 04 §10-§11).

Loads one outer fold's full + crossfit banks, verifies provenance, checksums,
shapes, contributor/forbidden sets and crossfit self-exclusion, then serves
vectorized per-trial gathers. Bank arrays are plain non-trainable tensors or
read-only memmaps; nothing is wrapped in ``nn.Parameter``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .bank_builder import BankVerifyError, load_checkpoint_registry
from .config import Stage2Config
from .contracts import (
    ARRAY_DTYPES,
    BANK_SCHEMA_VERSION,
    N_STIMULI,
    build_feature_manifest,
    expected_array_shapes,
    sha256_of_file,
    verify_stimulus_order,
)

FULL_BANK_ID = -1


@dataclass
class BankGather:
    """Per-trial bank rows gathered for a flattened valid-trial batch."""

    mu_trial: torch.Tensor  # [N,128] float32
    sigma_trial: torch.Tensor  # [N,128] float32
    count_trial: torch.Tensor  # [N] int64
    mu_token: torch.Tensor | None  # [N,192,128] float32 or None
    sigma_token: torch.Tensor | None
    mu_heat_token: torch.Tensor | None
    sigma_heat_token: torch.Tensor | None
    bank_ids: torch.Tensor  # [N] int64; FULL_BANK_ID (-1) = full fold bank


def _verify_array(
    path: Path,
    expected_shape: tuple[int, ...],
    dtype: str,
    *,
    epsilon: float | None = None,
) -> np.ndarray:
    if not path.is_file():
        raise BankVerifyError(f"missing bank array: {path}")
    arr = np.load(path, mmap_mode="r")
    if tuple(arr.shape) != tuple(expected_shape):
        raise BankVerifyError(f"{path}: shape {arr.shape} != {expected_shape}")
    if str(arr.dtype) != str(ARRAY_DTYPES[dtype]):
        raise BankVerifyError(f"{path}: dtype {arr.dtype} != {ARRAY_DTYPES[dtype]}")
    if not np.isfinite(arr).all():
        raise BankVerifyError(f"{path}: non-finite values")
    if epsilon is not None and np.any(arr < epsilon - 1e-12):
        raise BankVerifyError(f"{path}: values below epsilon={epsilon}")
    return arr


class NormativeBankStore:
    """One outer fold's normative banks, verified and memory-mapped.

    Bank selection rules (guide §10.2):
      train + crossfit     -> the subject's assigned crossfit bank
      train + full mode    -> the full fold bank
      validation           -> the full fold bank
    """

    def __init__(
        self,
        cfg: Stage2Config,
        fold: int,
        *,
        require_token_banks: bool | None = None,
        device: str = "cpu",
    ):
        if not 0 <= fold <= 4:
            raise BankVerifyError(f"fold must be in [0, 4], got {fold}")
        self.cfg = cfg
        self.fold = fold
        self.device = device
        self.train_mode = cfg.bank.train_mode
        self.verify_checksums = cfg.bank.verify_checksums
        if require_token_banks is None:
            require_token_banks = cfg.bank.require_token_banks
        self.require_token_banks = require_token_banks
        self.bank_root = Path(cfg.bank.root)

        registry = load_checkpoint_registry(cfg.bank.checkpoint_registry)
        self.registry_entry = registry["folds"][str(fold)]
        self.evaluation_regime = registry["evaluation_regime"]
        if cfg.evaluation_regime != self.evaluation_regime:
            raise BankVerifyError(
                f"config evaluation_regime {cfg.evaluation_regime!r} does not match the "
                f"checkpoint registry regime {self.evaluation_regime!r}"
            )

        manifest_path = self.bank_root / "manifest.json"
        if not manifest_path.is_file():
            raise BankVerifyError(f"bank root manifest missing: {manifest_path}")
        root_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if root_manifest.get("schema_version") != BANK_SCHEMA_VERSION:
            raise BankVerifyError(
                f"bank manifest schema {root_manifest.get('schema_version')!r} != {BANK_SCHEMA_VERSION!r}"
            )
        fold_entry = root_manifest.get("folds", {}).get(str(fold))
        if fold_entry is None or fold_entry.get("status") != "complete":
            raise BankVerifyError(f"fold {fold} is not complete in the bank manifest")

        self.fold_dir = self.bank_root / f"fold_{fold}"
        self.metadata = self._load_metadata(self.fold_dir, expected_crossfit_split=None)
        self._verify_provenance()

        include_token = self.metadata.get("include_fused_token_bank", False)
        include_heat = self.metadata.get("include_heatmap_token_bank", False)
        self.epsilon = float(self.metadata.get("epsilon", 1e-6))
        shapes = expected_array_shapes(include_token, include_heat)

        self.mu_trial_np = _verify_array(
            self.fold_dir / "mu_trial.npy", shapes["mu_trial"], "mu_trial"
        )
        self.sigma_trial_np = _verify_array(
            self.fold_dir / "sigma_trial.npy", shapes["sigma_trial"], "sigma_trial",
            epsilon=self.epsilon,
        )
        self.count_trial_np = _verify_array(
            self.fold_dir / "count_trial.npy", shapes["count_trial"], "count_trial"
        )
        if include_token:
            self.mu_token_np = _verify_array(
                self.fold_dir / "mu_token.npy", shapes["mu_token"], "mu_token"
            )
            self.sigma_token_np = _verify_array(
                self.fold_dir / "sigma_token.npy", shapes["sigma_token"], "sigma_token",
                epsilon=self.epsilon,
            )
        else:
            self.mu_token_np = self.sigma_token_np = None
        if include_heat:
            self.mu_heat_token_np = _verify_array(
                self.fold_dir / "mu_heat_token.npy", shapes["mu_heat_token"], "mu_heat_token"
            )
            self.sigma_heat_token_np = _verify_array(
                self.fold_dir / "sigma_heat_token.npy", shapes["sigma_heat_token"], "sigma_heat_token",
                epsilon=self.epsilon,
            )
        else:
            self.mu_heat_token_np = self.sigma_heat_token_np = None

        if self.require_token_banks and self.mu_token_np is None:
            raise BankVerifyError(f"fold {fold}: token banks required but not present in metadata")
        self.has_token_banks = self.mu_token_np is not None
        self.has_heat_token_banks = self.mu_heat_token_np is not None

        # Gather-time tensors: bank arrays become plain (non-trainable) torch
        # tensors once, and all batched gathers use torch indexing only.
        self.mu_trial = torch.from_numpy(np.array(self.mu_trial_np, dtype=np.float32))
        self.sigma_trial = torch.from_numpy(np.array(self.sigma_trial_np, dtype=np.float32))
        self.count_trial = torch.from_numpy(np.array(self.count_trial_np, dtype=np.int64))
        self.mu_token = (
            torch.from_numpy(np.array(self.mu_token_np, dtype=np.float32))
            if self.mu_token_np is not None
            else None
        )
        self.sigma_token = (
            torch.from_numpy(np.array(self.sigma_token_np, dtype=np.float32))
            if self.sigma_token_np is not None
            else None
        )
        self.mu_heat_token = (
            torch.from_numpy(np.array(self.mu_heat_token_np, dtype=np.float32))
            if self.mu_heat_token_np is not None
            else None
        )
        self.sigma_heat_token = (
            torch.from_numpy(np.array(self.sigma_heat_token_np, dtype=np.float32))
            if self.sigma_heat_token_np is not None
            else None
        )

        self.contributors_full = set(self.metadata.get("contributing_hc_subject_ids") or [])
        self.forbidden = set(self.metadata.get("forbidden_validation_subject_ids") or [])
        if self.contributors_full & self.forbidden:
            raise BankVerifyError(
                f"fold {fold}: validation subjects contributed to the bank: "
                f"{sorted(self.contributors_full & self.forbidden)}"
            )
        self._verify_contributors_are_hc()

        # Crossfit banks.
        self.assignment: pd.DataFrame | None = None
        self.excluded_by_split: dict[int, set[str]] = {}
        self.contributors_by_split: dict[int, set[str]] = {}
        self.split_arrays: dict[int, dict[str, np.ndarray]] = {}
        self.split_tensors: dict[int, dict[str, torch.Tensor]] = {}
        self.crossfit_splits = int(fold_entry.get("crossfit_splits", 0) or 0)
        if self.train_mode == "crossfit":
            if not self.crossfit_splits:
                raise BankVerifyError(
                    f"fold {fold}: bank.train_mode=crossfit but the bank has no crossfit splits"
                )
            crossfit_dir = self.fold_dir / "crossfit"
            assignment_path = crossfit_dir / "subject_assignment.csv"
            if not assignment_path.is_file():
                raise BankVerifyError(f"fold {fold}: missing {assignment_path}")
            self.assignment = pd.read_csv(assignment_path, dtype=str)
            required_cols = {"subject_id", "label", "bank_split_id"}
            if not required_cols <= set(self.assignment.columns):
                raise BankVerifyError(f"fold {fold}: crossfit assignment missing columns {sorted(required_cols)}")
            self._verify_assignment_completeness()
            hc_assigned_by_split: dict[int, set[str]] = {}
            for j in range(self.crossfit_splits):
                hc_assigned_by_split[j] = set(
                    self.assignment[
                        (self.assignment.label.astype(int) == 0)
                        & (self.assignment.bank_split_id.astype(int) == j)
                    ].subject_id.astype(str)
                )
            for j in range(self.crossfit_splits):
                split_dir = crossfit_dir / f"split_{j}"
                meta = self._load_metadata(split_dir, expected_crossfit_split=j)
                split_shapes = expected_array_shapes(
                    meta.get("include_fused_token_bank", False),
                    meta.get("include_heatmap_token_bank", False),
                )
                excluded = set(meta.get("crossfit_excluded_hc_subject_ids") or [])
                contributors = set(meta.get("contributing_hc_subject_ids") or [])
                if contributors & excluded:
                    raise BankVerifyError(f"fold {fold} split {j}: self-inclusion detected")
                if excluded != hc_assigned_by_split[j]:
                    raise BankVerifyError(
                        f"fold {fold} split {j}: assignment and metadata exclusion sets disagree"
                    )
                self.excluded_by_split[j] = excluded
                self.contributors_by_split[j] = contributors
                self.split_arrays[j] = {
                    "mu_trial": _verify_array(
                        split_dir / "mu_trial.npy", split_shapes["mu_trial"], "mu_trial"
                    ),
                    "sigma_trial": _verify_array(
                        split_dir / "sigma_trial.npy", split_shapes["sigma_trial"], "sigma_trial",
                        epsilon=self.epsilon,
                    ),
                    "count_trial": _verify_array(
                        split_dir / "count_trial.npy", split_shapes["count_trial"], "count_trial"
                    ),
                }
                if meta.get("include_fused_token_bank", False):
                    self.split_arrays[j]["mu_token"] = _verify_array(
                        split_dir / "mu_token.npy", split_shapes["mu_token"], "mu_token"
                    )
                    self.split_arrays[j]["sigma_token"] = _verify_array(
                        split_dir / "sigma_token.npy", split_shapes["sigma_token"], "sigma_token",
                        epsilon=self.epsilon,
                    )
                if meta.get("include_heatmap_token_bank", False):
                    self.split_arrays[j]["mu_heat_token"] = _verify_array(
                        split_dir / "mu_heat_token.npy", split_shapes["mu_heat_token"], "mu_heat_token"
                    )
                    self.split_arrays[j]["sigma_heat_token"] = _verify_array(
                        split_dir / "sigma_heat_token.npy", split_shapes["sigma_heat_token"], "sigma_heat_token",
                        epsilon=self.epsilon,
                    )
            # Gather-time tensors for the crossfit banks (pure torch indexing).
            self.split_tensors: dict[int, dict[str, torch.Tensor]] = {}
            for j, arrays in self.split_arrays.items():
                self.split_tensors[j] = {
                    "mu_trial": torch.from_numpy(np.array(arrays["mu_trial"], dtype=np.float32)),
                    "sigma_trial": torch.from_numpy(np.array(arrays["sigma_trial"], dtype=np.float32)),
                    "count_trial": torch.from_numpy(np.array(arrays["count_trial"], dtype=np.int64)),
                }
                if arrays.get("mu_token") is not None:
                    self.split_tensors[j]["mu_token"] = torch.from_numpy(
                        np.array(arrays["mu_token"], dtype=np.float32)
                    )
                    self.split_tensors[j]["sigma_token"] = torch.from_numpy(
                        np.array(arrays["sigma_token"], dtype=np.float32)
                    )
                if arrays.get("mu_heat_token") is not None:
                    self.split_tensors[j]["mu_heat_token"] = torch.from_numpy(
                        np.array(arrays["mu_heat_token"], dtype=np.float32)
                    )
                    self.split_tensors[j]["sigma_heat_token"] = torch.from_numpy(
                        np.array(arrays["sigma_heat_token"], dtype=np.float32)
                    )
            counts = {
                j: len(self.contributors_by_split[j]) for j in range(self.crossfit_splits)
            }
            if len(set(counts.values())) > 1:
                raise BankVerifyError(
                    f"fold {fold}: crossfit banks have different contributor counts: {counts}"
                )
        elif self.crossfit_splits and self.train_mode == "full_self_included":
            # Crossfit artifacts exist but are not used; allowed, recorded.
            pass

        self.feature_manifest = pd.read_csv(self.fold_dir / "feature_manifest.csv")
        self._verify_stimulus_manifest()

    # ------------------------------------------------------------------ helpers

    def _load_metadata(self, bank_dir: Path, *, expected_crossfit_split: int | None) -> dict[str, Any]:
        path = bank_dir / "metadata.json"
        if not path.is_file():
            raise BankVerifyError(f"missing metadata: {path}")
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != BANK_SCHEMA_VERSION:
            raise BankVerifyError(f"{path}: schema_version {metadata.get('schema_version')!r}")
        if metadata.get("fold") != self.fold:
            raise BankVerifyError(f"{path}: metadata fold {metadata.get('fold')} != {self.fold}")
        if metadata.get("crossfit_split") != expected_crossfit_split:
            raise BankVerifyError(
                f"{path}: crossfit_split {metadata.get('crossfit_split')!r} != {expected_crossfit_split!r}"
            )
        return metadata

    def _split_metadata(self, j: int) -> dict[str, Any]:
        return json.loads(
            (self.fold_dir / "crossfit" / f"split_{j}" / "metadata.json").read_text(encoding="utf-8")
        )

    def _verify_provenance(self) -> None:
        """Registry SHA-256 plus processed/DINO/CV input checksums (guide §10.1)."""
        checkpoint_path = Path(self.registry_entry["checkpoint"])
        if not checkpoint_path.is_file():
            raise BankVerifyError(f"registry checkpoint missing: {checkpoint_path}")
        registry_sha = str(self.registry_entry["sha256"])
        actual_sha = sha256_of_file(checkpoint_path)
        if actual_sha != registry_sha:
            raise BankVerifyError(f"checkpoint SHA-256 does not match the registry: {checkpoint_path}")
        if self.metadata.get("stage1_checkpoint_sha256") != registry_sha:
            raise BankVerifyError(f"fold {self.fold}: bank was built from a different checkpoint")
        if not self.verify_checksums:
            return
        source_checksums = self.metadata.get("source_checksums") or {}
        paths = self.cfg.paths
        # DINO root is the canonical sibling of processed_root (fixed project paths).
        dino_root = paths.processed_root.parent / "stimulus_features" / "dino_vits16"
        expected = {
            "processed_subject_manifest_sha256": paths.processed_root / "subject_manifest.csv",
            "processed_trial_manifest_sha256": paths.processed_root / "trial_manifest.parquet",
            "processed_dataset_metadata_sha256": paths.processed_root / "dataset_metadata.json",
            "dino_patch_tokens_sha256": dino_root / "patch_tokens.npy",
            "cv_metadata_sha256": paths.cv_root / "cv_metadata.json",
        }
        for key, path in expected.items():
            if not Path(path).is_file():
                raise BankVerifyError(f"input file for checksum verification missing: {path}")
            actual = sha256_of_file(Path(path))
            recorded = source_checksums.get(key)
            if recorded is None:
                continue
            if actual != recorded:
                raise BankVerifyError(f"checksum mismatch for {key}: recorded {recorded}, actual {actual}")

    def _verify_contributors_are_hc(self) -> None:
        subject_manifest = pd.read_csv(
            self.cfg.paths.processed_root / "subject_manifest.csv", dtype={"subject_id": str}
        )
        label_by_id = dict(zip(subject_manifest.subject_id, subject_manifest.label.astype(int)))
        non_hc = {s for s in self.contributors_full if label_by_id.get(s) != 0}
        if non_hc:
            raise BankVerifyError(f"fold {self.fold}: non-HC bank contributors: {sorted(non_hc)}")

    def _verify_assignment_completeness(self) -> None:
        train_subjects = pd.read_csv(
            self.cfg.paths.cv_root / f"fold_{self.fold}" / "train_subjects.csv", dtype={"subject_id": str}
        )
        assigned = set(self.assignment.subject_id.astype(str))
        expected = set(train_subjects.subject_id.astype(str))
        if assigned != expected:
            missing = sorted(expected - assigned)
            extra = sorted(assigned - expected)
            raise BankVerifyError(
                f"fold {self.fold}: crossfit assignment does not cover the training partition "
                f"(missing {missing}, extra {extra})"
            )

    def _verify_stimulus_manifest(self) -> None:
        image_manifest = pd.read_csv(
            self.cfg.paths.processed_root / "image_manifest.csv",
            dtype={"stimulus_id": str, "category": str},
        )
        canonical = build_feature_manifest(image_manifest)
        if not self.feature_manifest.stimulus_id.to_numpy().tolist() == canonical.stimulus_id.to_numpy().tolist():
            raise BankVerifyError(f"fold {self.fold}: bank feature manifest stimulus order differs from processed manifest")
        if self.feature_manifest.stimulus_index.astype(int).to_numpy().tolist() != list(range(N_STIMULI)):
            raise BankVerifyError(f"fold {self.fold}: bank feature manifest indices are not canonical 0..99")

    # ------------------------------------------------------------- bank lookup

    def bank_for_subject(self, subject_id: str, split: str, bank_split_id: int | None) -> int:
        """Return the bank id a subject uses: FULL_BANK_ID (-1) or a crossfit split."""
        if split == "val":
            return FULL_BANK_ID
        if self.train_mode == "full_self_included":
            return FULL_BANK_ID
        if bank_split_id is None:
            raise BankVerifyError(f"train subject {subject_id} has no bank_split_id in crossfit mode")
        if bank_split_id not in self.split_arrays:
            raise BankVerifyError(f"subject {subject_id} assigned unknown bank split {bank_split_id}")
        return int(bank_split_id)

    def bank_split_id_for(self, subject_id: str, split: str) -> int | None:
        """Dataset-side lookup: the assigned crossfit split, or None for the full bank."""
        if split == "val" or self.train_mode == "full_self_included":
            return None
        row = self.assignment[self.assignment.subject_id.astype(str) == str(subject_id)]
        if row.empty:
            raise BankVerifyError(f"subject {subject_id} missing from the crossfit assignment")
        return int(row.bank_split_id.iloc[0])

    # ------------------------------------------------------------------ gather

    def gather_trials(
        self,
        stimulus_indices: Any,
        subject_slots: Any,
        subject_bank_ids: Any,
        split: str,
        *,
        device: str | None = None,
    ) -> BankGather:
        """Gather bank rows for every flattened valid trial (guide §10.3).

        ``subject_bank_ids`` is ``[B]`` or None; ``split`` selects the bank
        policy (validation always uses the full fold bank).
        """
        device = device or self.device
        si = torch.as_tensor(stimulus_indices, dtype=torch.int64, device=device)
        slots = torch.as_tensor(subject_slots, dtype=torch.int64, device=device)
        n = si.numel()
        if slots.numel() != n:
            raise BankVerifyError("stimulus_indices and subject_slots length mismatch")
        if torch.any(si < 0) or torch.any(si >= N_STIMULI):
            raise BankVerifyError("stimulus index out of bank manifest range")

        if split == "val" or self.train_mode == "full_self_included" or subject_bank_ids is None:
            bank_ids = torch.full((n,), FULL_BANK_ID, dtype=torch.int64, device=device)
        else:
            subject_ids = torch.as_tensor(subject_bank_ids, dtype=torch.int64, device=device)
            bank_ids = subject_ids[slots]
            unknown = {int(x) for x in bank_ids.tolist()} - set(self.split_tensors)
            if unknown:
                raise BankVerifyError(f"unknown bank ids in gather: {sorted(unknown)}")

        mu_trial = torch.empty((n, 128), dtype=torch.float32, device=device)
        sigma_trial = torch.empty((n, 128), dtype=torch.float32, device=device)
        count_trial = torch.empty(n, dtype=torch.int64, device=device)
        mu_token = torch.empty((n, 192, 128), dtype=torch.float32, device=device) if self.has_token_banks else None
        sigma_token = torch.empty((n, 192, 128), dtype=torch.float32, device=device) if self.has_token_banks else None
        mu_heat = torch.empty((n, 192, 128), dtype=torch.float32, device=device) if self.has_heat_token_banks else None
        sigma_heat = torch.empty((n, 192, 128), dtype=torch.float32, device=device) if self.has_heat_token_banks else None

        for bank_id in sorted({int(x) for x in bank_ids.tolist()}):
            rows = torch.nonzero(bank_ids == bank_id, as_tuple=True)[0]
            idx = si[rows]
            if bank_id == FULL_BANK_ID:
                mu_trial[rows] = self.mu_trial.to(device)[idx]
                sigma_trial[rows] = self.sigma_trial.to(device)[idx]
                count_trial[rows] = self.count_trial.to(device)[idx]
                if mu_token is not None:
                    mu_token[rows] = self.mu_token.to(device)[idx]
                    sigma_token[rows] = self.sigma_token.to(device)[idx]
                if mu_heat is not None:
                    mu_heat[rows] = self.mu_heat_token.to(device)[idx]
                    sigma_heat[rows] = self.sigma_heat_token.to(device)[idx]
            else:
                arrays = self.split_tensors[bank_id]
                mu_trial[rows] = arrays["mu_trial"].to(device)[idx]
                sigma_trial[rows] = arrays["sigma_trial"].to(device)[idx]
                count_trial[rows] = arrays["count_trial"].to(device)[idx]
                if mu_token is not None and "mu_token" in arrays:
                    mu_token[rows] = arrays["mu_token"].to(device)[idx]
                    sigma_token[rows] = arrays["sigma_token"].to(device)[idx]
                if mu_heat is not None and "mu_heat_token" in arrays:
                    mu_heat[rows] = arrays["mu_heat_token"].to(device)[idx]
                    sigma_heat[rows] = arrays["sigma_heat_token"].to(device)[idx]

        return BankGather(
            mu_trial=mu_trial,
            sigma_trial=sigma_trial,
            count_trial=count_trial,
            mu_token=mu_token,
            sigma_token=sigma_token,
            mu_heat_token=mu_heat,
            sigma_heat_token=sigma_heat,
            bank_ids=bank_ids,
        )


def audit_data_boundary(
    train_dataset: Any,
    val_dataset: Any,
    bank_store: NormativeBankStore,
) -> dict[str, Any]:
    """In-memory leakage/consistency audit (guide §11).

    Returns a dict of named booleans; the trainer treats ``status != "ok"``
    as fatal. No files are written.
    """
    train_ids = set(train_dataset.subject_ids)
    val_ids = set(val_dataset.subject_ids)
    checks: dict[str, Any] = {
        "train_val_disjoint": not bool(train_ids & val_ids),
        "labels_binary": all(
            label in (0, 1) for label in train_dataset.subject_labels() + val_dataset.subject_labels()
        ),
        "no_val_subject_in_bank": not bool(val_ids & bank_store.contributors_full),
        "stimulus_manifest_unique": (
            bank_store.feature_manifest.stimulus_index.astype(int).to_numpy().tolist()
            == list(range(N_STIMULI))
        ),
        "crossfit_self_exclusion": True,
        "bank_contributor_counts_comparable": True,
    }
    if bank_store.train_mode == "crossfit":
        checks["crossfit_self_exclusion"] = all(
            not (bank_store.excluded_by_split[j] & bank_store.contributors_by_split[j])
            for j in bank_store.excluded_by_split
        )
        checks["bank_contributor_counts_comparable"] = (
            len(set(len(c) for c in bank_store.contributors_by_split.values())) == 1
        )
    checks["status"] = "ok" if all(v for k, v in checks.items() if k != "status") else "fail"
    return checks
