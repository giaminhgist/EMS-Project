"""HC-only Stage-1 dataset and collator (contract §2)."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .types import Stage1Batch, Stage1Sample

LOG1P_CLIP_TRANSFORM = "log1p_clip"


class Stage1DataError(ValueError):
    pass


class _SubjectArrayCache:
    """Small LRU of memory-mapped per-subject heatmap arrays."""

    def __init__(self, capacity: int = 16):
        self.capacity = capacity
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def get(self, path: Path) -> np.ndarray:
        key = str(path)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        arr = np.load(path, mmap_mode="r")
        self._cache[key] = arr
        if len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
        return arr


def verify_input_checksums(
    processed_root: Path, cv_fold_dir: Path, dino_root: Path
) -> None:
    """Validate processed/CV/DINO checksums before any trial is served."""
    from preprocessing.storage import StorageError, sha256_of_file, TrialStore

    try:
        store = TrialStore(processed_root)
        store.verify_manifest_checksums()
    except StorageError as exc:
        raise Stage1DataError(str(exc)) from exc

    cv_meta_path = cv_fold_dir.parent / "cv_metadata.json"
    if not cv_meta_path.is_file():
        raise Stage1DataError(f"missing CV metadata: {cv_meta_path}")
    cv_meta = json.loads(cv_meta_path.read_text(encoding="utf-8"))
    input_checksums = cv_meta.get("input_checksums", {})
    for key, fname in [
        ("subject_manifest_sha256", "subject_manifest.csv"),
        ("trial_manifest_sha256", "trial_manifest.parquet"),
    ]:
        recorded = input_checksums.get(key)
        if recorded is None:
            continue
        actual = sha256_of_file(processed_root / fname)
        if actual != recorded:
            raise Stage1DataError(f"{fname} checksum mismatch vs CV metadata")

    # DINO artifacts: manifest consistent with image manifest; tokens present.
    from stimulus_features.storage import FeaturePaths

    paths = FeaturePaths.for_dir(dino_root)
    if not paths.patch_tokens.is_file():
        raise Stage1DataError(f"missing DINO tokens: {paths.patch_tokens}")
    feature_manifest = pd.read_csv(paths.feature_manifest, dtype=str)
    image_manifest = pd.read_csv(
        processed_root / "image_manifest.csv", dtype={"stimulus_id": str}
    )
    if len(feature_manifest) != len(image_manifest):
        raise Stage1DataError("DINO feature manifest does not match image manifest")
    if not (feature_manifest.stimulus_id.to_numpy() == image_manifest.stimulus_id.to_numpy()).all():
        raise Stage1DataError("DINO feature manifest stimulus order differs from image manifest")


class Stage1Dataset(Dataset):
    """Map-style dataset over one fold partition, filtered to HC + ok trials.

    ``transform="fixed"`` applies the leakage-free transform
    ``x[0]=log1p(x[0]); x[1]=log1p(x[1]); x[2]=clip(x[2],-1,1)``.
    """

    def __init__(
        self,
        processed_root: Path | str,
        dino_root: Path | str,
        cv_fold_dir: Path | str,
        split: str,  # "train" | "val"
        *,
        group_filter: str = "HC",
        transform: str = "fixed",
        verify_checksums: bool = True,
        fold: int | None = None,
        array_cache_capacity: int = 16,
        active_channels: tuple[int, ...] = (0, 1, 2),
    ):
        if split not in ("train", "val"):
            raise Stage1DataError(f"split must be 'train' or 'val', got {split!r}")
        self.processed_root = Path(processed_root)
        self.dino_root = Path(dino_root)
        self.cv_fold_dir = Path(cv_fold_dir)
        self.split = split
        self.group_filter = group_filter
        self.transform = transform
        self.fold = fold
        self.active_channels = tuple(active_channels)

        if verify_checksums:
            verify_input_checksums(self.processed_root, self.cv_fold_dir, self.dino_root)

        partition_path = self.cv_fold_dir / f"{split}_trials.parquet"
        if not partition_path.is_file():
            raise Stage1DataError(f"missing partition file: {partition_path}")
        trials = pd.read_parquet(partition_path)

        # The canonical CV partitions retain both HC and SZ for future Stage 2;
        # Stage 1 must filter HC explicitly and never serve a non-HC or
        # non-heatmap-eligible row. Filtered counts are recorded.
        self.n_rows_filtered_non_hc = int((trials.group != group_filter).sum())
        self.n_rows_filtered_non_ok = int(
            ((trials.group == group_filter) & (trials.qc_status != "ok")).sum()
        )
        self.trial_rows = trials[
            (trials.group == group_filter) & (trials.qc_status == "ok")
        ].reset_index(drop=True)
        if self.trial_rows.empty:
            raise Stage1DataError(
                f"partition {partition_path.name} has no {group_filter} heatmap-eligible rows"
            )

        # DINO tokens via the feature manifest (never by array position).
        feature_manifest = pd.read_csv(
            self.dino_root / "feature_manifest.csv", dtype=str
        )
        self.dino_row_by_stimulus_index = {
            int(r.stimulus_index): int(r.feature_row_index)
            for r in feature_manifest.itertuples()
        }
        self.dino_tokens = np.load(self.dino_root / "patch_tokens.npy", mmap_mode="r")
        self.dino_tensor_cache: dict[int, torch.Tensor] = {}

        self._array_cache = _SubjectArrayCache(capacity=array_cache_capacity)
        self._row_indices_by_stimulus: dict[int, list[int]] = {}
        for pos, row in enumerate(self.trial_rows.itertuples()):
            self._row_indices_by_stimulus.setdefault(int(row.stimulus_index), []).append(pos)

    @property
    def n_trials(self) -> int:
        return len(self.trial_rows)

    def stimulus_groups(self) -> dict[int, list[int]]:
        """stimulus_index -> dataset row indices (actual training trials)."""
        return dict(self._row_indices_by_stimulus)

    def trial_uids(self) -> list[str]:
        return list(self.trial_rows.trial_uid)

    def _heatmap_for_row(self, row) -> torch.Tensor:
        heatmaps = self._array_cache.get(self.processed_root / str(row.subject_array_path))
        idx = int(row.subject_row_index)
        heatmap = np.array(heatmaps[idx], dtype=np.float32)  # [3, 48, 64]
        if self.transform == "fixed":
            out = heatmap.copy()
            out[0] = np.log1p(np.maximum(out[0], 0.0))
            out[1] = np.log1p(np.maximum(out[1], 0.0))
            out[2] = np.clip(out[2], -1.0, 1.0)
            heatmap = out
        # Channel ablations select a declared subset of the three channels.
        if self.active_channels != (0, 1, 2):
            heatmap = heatmap[list(self.active_channels)]
        return torch.from_numpy(heatmap)

    def _dino_for_stimulus(self, stimulus_index: int) -> torch.Tensor:
        if stimulus_index not in self.dino_tensor_cache:
            row = self.dino_row_by_stimulus_index[stimulus_index]
            tokens = np.array(
                self.dino_tokens[row], dtype=np.float32
            )  # [768, 384]
            self.dino_tensor_cache[stimulus_index] = torch.from_numpy(tokens)
        return self.dino_tensor_cache[stimulus_index]

    def __len__(self) -> int:
        return self.n_trials

    def __getitem__(self, index: int) -> Stage1Sample:
        row = self.trial_rows.iloc[index]
        return Stage1Sample(
            heatmap=self._heatmap_for_row(row),
            subject_id=str(row.subject_id),
            stimulus_id=str(row.stimulus_id),
            stimulus_index=int(row.stimulus_index),
            trial_uid=str(row.trial_uid),
            group=str(row.group),
        )

    def collate(self, samples: list[Stage1Sample]) -> Stage1Batch:
        """Deduplicate stimuli within the batch (contract §2).

        The adapter runs only on the unique stimulus tensors; raw DINO tensors
        are never materialized 64 times.
        """
        heatmaps = torch.stack([s.heatmap for s in samples])
        # Deterministic slot assignment by order of first appearance.
        slot_by_stimulus: dict[int, int] = {}
        unique_stimulus_indices: list[int] = []
        for s in samples:
            if s.stimulus_index not in slot_by_stimulus:
                slot_by_stimulus[s.stimulus_index] = len(unique_stimulus_indices)
                unique_stimulus_indices.append(s.stimulus_index)
        unique_dino = torch.stack(
            [self._dino_for_stimulus(si) for si in unique_stimulus_indices]
        )
        slots = torch.tensor(
            [slot_by_stimulus[s.stimulus_index] for s in samples], dtype=torch.int64
        )
        return Stage1Batch(
            heatmaps=heatmaps,
            unique_dino_tokens=unique_dino,
            trial_to_stimulus_slot=slots,
            stimulus_indices=torch.tensor(unique_stimulus_indices, dtype=torch.int64),
            subject_ids=[s.subject_id for s in samples],
            stimulus_ids=[s.stimulus_id for s in samples],
            trial_uids=[s.trial_uid for s in samples],
            groups=[s.group for s in samples],
        )

    def collate_from_indices(self, indices: list[int]) -> Stage1Batch:
        return self.collate([self[i] for i in indices])
