"""Subject-level Stage-2 dataset (Phase 2, guide 04 §4-§5, §13).

One item per subject: a fixed 100-stimulus panel with an explicit missing-trial
mask. Labels come from the canonical manifests, subject ordering from the CV
partition files, and per-subject heatmap arrays are memory-mapped through a
small LRU cache.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from stage1.dataset import _SubjectArrayCache, verify_input_checksums

from .bank import NormativeBankStore, audit_data_boundary
from .collate import collate_subject_samples, flatten_valid_trials
from .config import ConfigError, Stage2Config
from .contracts import N_STIMULI, Stage2SubjectSample, build_feature_manifest
from .sampler import BalancedSubjectBatchSampler

log = logging.getLogger("stage2.dataset")

HEATMAP_SHAPE = (3, 48, 64)


class Stage2DataError(ValueError):
    pass


def fixed_heatmap_transform(heatmap: np.ndarray) -> np.ndarray:
    """The Stage-1 fixed leakage-free transform (guide §5.2), reused exactly.

    channel 0: log1p(max(x,0)); channel 1: log1p(max(x,0)); channel 2: clip(x,-1,1)
    """
    out = heatmap.astype(np.float32, copy=True)
    out[0] = np.log1p(np.maximum(out[0], 0.0))
    out[1] = np.log1p(np.maximum(out[1], 0.0))
    out[2] = np.clip(out[2], -1.0, 1.0)
    return out


class Stage2SubjectDataset(Dataset):
    """Map-style dataset over the subjects of one CV fold partition.

    ``__len__`` is the number of subjects; ``__getitem__`` assembles the
    100-slot panel from the subject's observed (qc-ok) trial rows.
    """

    def __init__(
        self,
        cfg: Stage2Config,
        fold: int,
        split: str,
        *,
        bank_store: NormativeBankStore | None = None,
        verify_checksums: bool = True,
    ):
        if split not in ("train", "val"):
            raise Stage2DataError(f"split must be 'train' or 'val', got {split!r}")
        self.cfg = cfg
        self.fold = fold
        self.split = split
        processed_root = Path(cfg.paths.processed_root)
        cv_fold_dir = Path(cfg.paths.cv_root) / f"fold_{fold}"

        if verify_checksums:
            dino_root = processed_root.parent / "stimulus_features" / "dino_vits16"
            verify_input_checksums(processed_root, cv_fold_dir, dino_root)

        subject_manifest = pd.read_csv(
            processed_root / "subject_manifest.csv", dtype={"subject_id": str, "group": str}
        )
        manifest_label = dict(zip(subject_manifest.subject_id, subject_manifest.label.astype(int)))

        partition_path = cv_fold_dir / f"{split}_subjects.csv"
        if not partition_path.is_file():
            raise Stage2DataError(f"missing partition file: {partition_path}")
        partition = pd.read_csv(partition_path, dtype={"subject_id": str})
        if partition.subject_id.duplicated().any():
            raise Stage2DataError(f"duplicate subjects in {partition_path.name}")
        partition_ids = set(partition.subject_id)

        trial_manifest = pd.read_parquet(processed_root / "trial_manifest.parquet")
        trial_manifest["subject_id"] = trial_manifest.subject_id.astype(str)
        trials = trial_manifest[trial_manifest.subject_id.isin(partition_ids)]

        rows_by_subject: dict[str, list[tuple[int, int, str]]] = {}
        subject_array_path: dict[str, str] = {}
        n_observed: dict[str, int] = {}
        for sid, group in trials[trials.qc_status == "ok"].groupby("subject_id"):
            seen: set[int] = set()
            rows: list[tuple[int, int, str]] = []
            for rec in group.itertuples():
                si = int(rec.stimulus_index)
                if not 0 <= si < N_STIMULI:
                    raise Stage2DataError(
                        f"subject {sid}: noncanonical stimulus index {si} in trial manifest"
                    )
                if si in seen:
                    raise Stage2DataError(f"subject {sid}: duplicate trial row for stimulus {si}")
                seen.add(si)
                rows.append((si, int(rec.subject_row_index), str(rec.trial_uid)))
            rows_by_subject[sid] = sorted(rows, key=lambda r: r[0])
            n_observed[sid] = len(rows)
            path = str(group.subject_array_path.iloc[0])
            if pd.isna(path):
                raise Stage2DataError(f"subject {sid}: missing subject_array_path")
            subject_array_path[sid] = path

        for sid in sorted(partition_ids - set(rows_by_subject)):
            raise Stage2DataError(f"subject {sid}: no observed (qc-ok) trials in the {split} partition")

        subjects = partition.copy()
        subjects["label"] = subjects.subject_id.map(manifest_label)
        if subjects.label.isna().any():
            missing = sorted(subjects[subjects.label.isna()].subject_id)
            raise Stage2DataError(f"subjects missing from the canonical manifest: {missing}")
        # Labels must agree with the partition file's own label column.
        if "label" in partition.columns:
            disagree = subjects[
                subjects.label.astype(int) != partition.label.astype(int)
            ].subject_id.tolist()
            if disagree:
                raise Stage2DataError(
                    f"partition labels disagree with the canonical manifest: {disagree}"
                )
        labels = subjects.label.astype(int).to_numpy()
        if not set(np.unique(labels)) <= {0, 1}:
            raise Stage2DataError("subject labels must be binary {0, 1}")

        subjects["subject_array_path"] = subjects.subject_id.map(subject_array_path)
        subjects["number_of_observed_trials"] = subjects.subject_id.map(n_observed)
        for sid, path in subject_array_path.items():
            p = processed_root / path
            if not p.is_file():
                raise Stage2DataError(f"subject array not found: {p}")
            arr = np.load(p, mmap_mode="r")
            if arr.ndim != 4 or arr.shape[1:] != HEATMAP_SHAPE:
                raise Stage2DataError(f"{p}: heatmap shape {arr.shape} != [n, 3, 48, 64]")
            if arr.dtype != np.float32:
                raise Stage2DataError(f"{p}: heatmap dtype {arr.dtype} != float32")
            if len(arr) < n_observed[sid]:
                raise Stage2DataError(
                    f"{p}: array has {len(arr)} rows but {n_observed[sid]} observed trials"
                )

        bank_split: dict[str, int | None] = {}
        if split == "train" and bank_store is not None and bank_store.train_mode == "crossfit":
            for sid in partition_ids:
                bank_split[sid] = bank_store.bank_split_id_for(sid, split)
            subjects["bank_split_id"] = subjects.subject_id.map(bank_split)
        else:
            subjects["bank_split_id"] = None

        image_manifest = pd.read_csv(
            processed_root / "image_manifest.csv",
            dtype={"stimulus_id": str, "category": str},
        )
        self.category_ids = np.array(
            build_feature_manifest(image_manifest).category_id, dtype=np.int64
        )  # writable copy; converted to a tensor per item

        self.processed_root = processed_root
        self.subjects = subjects.reset_index(drop=True)
        self._rows_by_subject = rows_by_subject
        self._array_cache = _SubjectArrayCache(capacity=16)
        self.subject_ids: list[str] = [str(x) for x in self.subjects.subject_id]
        self.labels: list[int] = [int(x) for x in self.subjects.label]

    # ------------------------------------------------------------- properties

    def subject_labels(self) -> list[int]:
        return self.labels

    def n_missing_trials(self) -> int:
        return int((N_STIMULI - self.subjects.number_of_observed_trials).sum())

    def __len__(self) -> int:
        return len(self.subjects)

    def __getitem__(self, index: int) -> Stage2SubjectSample:
        row = self.subjects.iloc[index]
        sid = str(row.subject_id)
        arr = self._array_cache.get(self.processed_root / str(row.subject_array_path))
        heatmaps = np.zeros((N_STIMULI,) + HEATMAP_SHAPE, dtype=np.float32)
        mask = np.zeros(N_STIMULI, dtype=bool)
        uids: list[str | None] = [None] * N_STIMULI
        for si, row_index, uid in self._rows_by_subject[sid]:
            heatmap = np.array(arr[row_index], dtype=np.float32)  # owned copy
            if heatmap.shape != HEATMAP_SHAPE:
                raise Stage2DataError(f"subject {sid}: heatmap row shape {heatmap.shape}")
            if not np.isfinite(heatmap).all():
                raise Stage2DataError(f"subject {sid} stimulus {si}: non-finite heatmap")
            heatmaps[si] = fixed_heatmap_transform(heatmap)
            mask[si] = True
            uids[si] = uid
        bank_split_id = row.bank_split_id
        return Stage2SubjectSample(
            subject_id=sid,
            label=int(row.label),
            heatmaps=torch.from_numpy(heatmaps),
            trial_mask=torch.from_numpy(mask),
            stimulus_indices=torch.arange(N_STIMULI, dtype=torch.int64),
            category_ids=torch.from_numpy(self.category_ids),
            trial_uids=tuple(uids),
            bank_split_id=int(bank_split_id) if bank_split_id is not None and not pd.isna(bank_split_id) else None,
        )


# ------------------------------------------------------------------- dry run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage-2 data-layer dry run (no training)")
    parser.add_argument(
        "--config", default=str(Path(__file__).resolve().parents[2] / "configs" / "stage2" / "base.yaml")
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="print data/bank summary and exit")
    return parser


def dry_run(cfg: Stage2Config, fold: int) -> dict[str, Any]:
    bank_store = NormativeBankStore(cfg, fold)
    train_ds = Stage2SubjectDataset(cfg, fold, "train", bank_store=bank_store)
    val_ds = Stage2SubjectDataset(cfg, fold, "val", bank_store=bank_store)
    audit = audit_data_boundary(train_ds, val_ds, bank_store)

    print(f"fold {fold} | bank regime {bank_store.evaluation_regime} | train_mode {bank_store.train_mode}")
    print(
        f"train subjects: {len(train_ds)} (HC {sum(1 for l in train_ds.subject_labels() if l == 0)}, "
        f"SZ {sum(1 for l in train_ds.subject_labels() if l == 1)})"
    )
    print(
        f"val subjects:   {len(val_ds)} (HC {sum(1 for l in val_ds.subject_labels() if l == 0)}, "
        f"SZ {sum(1 for l in val_ds.subject_labels() if l == 1)})"
    )
    print(f"observed train trials: {sum(train_ds.subjects.number_of_observed_trials)}")
    print(f"observed val trials:   {sum(val_ds.subjects.number_of_observed_trials)}")
    print(f"missing train trials:  {train_ds.n_missing_trials()}")
    print(f"missing val trials:    {val_ds.n_missing_trials()}")
    print(f"leakage audit: {audit}")

    train_sampler = BalancedSubjectBatchSampler(
        train_ds,
        batch_size=cfg.sampler.subject_batch_size,
        seed=cfg.seed,
        fold=fold,
        epoch=0,
        balance_groups=cfg.sampler.balance_groups,
        drop_last=cfg.sampler.drop_last,
        shuffle=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_sampler=train_sampler,
        collate_fn=collate_subject_samples,
        num_workers=cfg.runtime.num_workers,
        pin_memory=cfg.runtime.pin_memory,
        persistent_workers=cfg.runtime.persistent_workers,
    )
    first_batch = next(iter(train_loader))
    print(f"train batches: {len(train_loader)} | first batch subjects: {first_batch.subject_ids}")
    print(
        f"first batch: labels {first_batch.labels.tolist()} | bank_split_ids "
        f"{None if first_batch.bank_split_ids is None else first_batch.bank_split_ids.tolist()}"
    )
    print(
        f"batch tensor shapes: heatmaps {tuple(first_batch.heatmaps.shape)} "
        f"mask {tuple(first_batch.trial_mask.shape)}"
    )
    flat = flatten_valid_trials(first_batch)
    print(f"flattened valid trials: {tuple(flat.heatmaps.shape)}")
    gathered = bank_store.gather_trials(
        stimulus_indices=flat.stimulus_indices,
        subject_slots=flat.subject_slots,
        subject_bank_ids=first_batch.bank_split_ids,
        split="train",
    )
    print(
        f"gathered bank: mu_trial {tuple(gathered.mu_trial.shape)} "
        f"mu_token {None if gathered.mu_token is None else tuple(gathered.mu_token.shape)} "
        f"bank_ids {gathered.bank_ids.unique().tolist()}"
    )
    val_sampler = BalancedSubjectBatchSampler(
        val_ds,
        batch_size=cfg.sampler.subject_batch_size,
        seed=cfg.seed,
        fold=fold,
        epoch=0,
        balance_groups=cfg.sampler.balance_groups,
        drop_last=False,
        shuffle=False,
    )
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler, collate_fn=collate_subject_samples)
    print(f"val batches (fixed order): {len(val_loader)}")
    return {"train": train_ds, "val": val_ds, "bank_store": bank_store, "audit": audit}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        cfg = Stage2Config.from_yaml(args.config)
        cfg = Stage2Config.from_dict({**cfg.to_dict(), "fold": args.fold})
        if not args.dry_run:
            print("use --dry-run to inspect the data boundary (no training exists yet)")
            return 0
        dry_run(cfg, args.fold)
        return 0
    except (ConfigError, Stage2DataError, ValueError, OSError) as exc:
        print(f"stage2 data check failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
