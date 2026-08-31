"""Atomic storage, manifests, and the indexed trial accessor."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .identifiers import sha256_hex, trial_uid

TRIAL_MANIFEST_COLUMNS: list[str] = [
    "trial_uid",
    "subject_id",
    "stimulus_id",
    "subject_numeric_id",
    "stimulus_index",
    "group",
    "label",
    "category",
    "subject_array_path",
    "subject_row_index",
    "n_fixations_raw",
    "n_fixations_used",
    "n_transitions_used",
    "total_duration_ms_raw",
    "total_duration_ms_used",
    "n_off_canvas",
    "n_nonfinite",
    "n_nonpositive_duration",
    "n_below_duration_threshold",
    "fix_index_has_gap",
    "qc_status",
]

TRIAL_QC_EXTRA_COLUMNS: list[str] = [
    "fix_index_has_duplicates",
    "fix_index_non_monotonic",
    "temporal_undefined",
    "n_malformed_fix_index",
    "n_malformed_duration",
    "n_malformed_pupil",
    "mass_density",
    "mass_transition",
    "mass_density_error",
    "mass_transition_error",
]


class StorageError(ValueError):
    """Raised on storage contract violations."""


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically (temp file + fsync + replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, payload: Any) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    atomic_write_text(path, text)


def sha256_of_file(path: Path) -> str:
    return sha256_hex(path.read_bytes())


def atomic_write_npy(path: Path, array: np.ndarray) -> str:
    """Write an array atomically; return its byte-level SHA-256."""
    buf = tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npy.tmp", delete=False)
    tmp = Path(buf.name)
    try:
        np.save(buf, array, allow_pickle=False)
        buf.close()
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return sha256_of_file(path)


def validate_heatmap_array(heatmaps: np.ndarray, height: int, width: int) -> None:
    """Validate a per-subject heatmap stack before publishing."""
    if not isinstance(heatmaps, np.ndarray):
        raise StorageError("heatmaps must be a numpy array")
    if heatmaps.ndim != 4 or heatmaps.shape[1:] != (3, height, width):
        raise StorageError(f"heatmaps shape must be [N, 3, {height}, {width}], got {heatmaps.shape}")
    if heatmaps.dtype != np.float32:
        raise StorageError(f"heatmaps dtype must be float32, got {heatmaps.dtype}")
    if not np.all(np.isfinite(heatmaps)):
        raise StorageError("heatmaps contain non-finite values")
    if np.any(heatmaps[:, :2] < 0):
        raise StorageError("density/transition channels contain negative values")
    ch2 = heatmaps[:, 2]
    if np.any(np.abs(ch2) > 1.0 + 1e-6):
        raise StorageError("temporal channel exceeds [-1, 1] beyond tolerance")


def validate_stimulus_indices(
    stimulus_indices: np.ndarray, n_rows: int, n_stimuli: int
) -> None:
    if stimulus_indices.ndim != 1 or len(stimulus_indices) != n_rows:
        raise StorageError("stimulus_indices length must match heatmap rows")
    if stimulus_indices.dtype not in (np.int32, np.int64):
        raise StorageError("stimulus_indices dtype must be integer")
    if np.any(stimulus_indices < 0) or np.any(stimulus_indices >= n_stimuli):
        raise StorageError("stimulus_indices out of manifest range")
    if not np.all(np.diff(stimulus_indices.astype(np.int64)) > 0):
        raise StorageError("stimulus_indices must be strictly increasing (stimulus_index order)")


def publish_subject(
    output_root: Path,
    subject_id: str,
    heatmaps: np.ndarray,
    stimulus_indices: np.ndarray,
    trial_qc: pd.DataFrame,
    artifact_meta: dict[str, Any],
    height: int,
    width: int,
    n_stimuli: int,
    force: bool,
) -> None:
    """Atomically publish one subject directory.

    Files are written into a staging directory that is renamed into place, so
    a partially written subject never appears complete.
    """
    output_root = Path(output_root)
    subject_dir = output_root / "subjects" / subject_id
    validate_heatmap_array(heatmaps, height, width)
    validate_stimulus_indices(stimulus_indices, len(heatmaps), n_stimuli)
    if len(trial_qc) == 0:
        raise StorageError(f"refusing to publish subject {subject_id} with no trials")

    if subject_dir.exists():
        if not force:
            raise StorageError(
                f"subject directory already exists: {subject_dir} "
                "(use --force to replace or --resume to skip)"
            )
        # Replace: stage first, then swap via a unique staging name.
        staging = output_root / "subjects" / f".tmp_{subject_id}_{os.getpid()}"
        if staging.exists():
            staging = output_root / "subjects" / f".tmp_{subject_id}_{os.getpid()}_{id(staging)}"
    else:
        staging = output_root / "subjects" / f".tmp_{subject_id}_{os.getpid()}"

    staging.mkdir(parents=True, exist_ok=False)
    try:
        # Record file-level checksums (of the .npy bytes actually written) in
        # the artifact metadata so resume/verification compares like with like.
        heatmaps_sha = atomic_write_npy(staging / "heatmaps.npy", heatmaps)
        stim_sha = atomic_write_npy(
            staging / "stimulus_indices.npy", stimulus_indices.astype(np.int64)
        )
        artifact_meta = dict(artifact_meta)
        artifact_meta["heatmaps_sha256"] = heatmaps_sha
        artifact_meta["stimulus_indices_sha256"] = stim_sha
        trial_qc.to_parquet(staging / "trial_qc.parquet", index=False)
        atomic_write_json(staging / "artifact_meta.json", artifact_meta)
    except BaseException:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        raise

    if subject_dir.exists():
        # Atomic swap for --force: rename old aside, rename staging in.
        old = output_root / "subjects" / f".old_{subject_id}_{os.getpid()}"
        os.replace(subject_dir, old)
        try:
            os.replace(staging, subject_dir)
        except BaseException:
            os.replace(old, subject_dir)
            raise
        import shutil

        shutil.rmtree(old, ignore_errors=True)
    else:
        os.replace(staging, subject_dir)


def verify_subject_dir(
    subject_dir: Path, expected_config_hash: str
) -> tuple[dict[str, Any], str, str] | None:
    """Return ``(artifact_meta, heatmaps_sha256, stimulus_indices_sha256)`` if
    the subject directory is compatible with the current configuration, else
    raise or return ``None`` when absent."""
    subject_dir = Path(subject_dir)
    meta_path = subject_dir / "artifact_meta.json"
    if not subject_dir.exists():
        return None
    if not meta_path.is_file():
        raise StorageError(f"subject directory missing artifact_meta.json: {subject_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("config_hash") != expected_config_hash:
        raise StorageError(
            f"subject {subject_dir.name} was produced with config hash "
            f"{meta.get('config_hash')!r}, current config hash is {expected_config_hash!r}"
        )
    heatmaps_path = subject_dir / "heatmaps.npy"
    stim_path = subject_dir / "stimulus_indices.npy"
    if not heatmaps_path.is_file() or not stim_path.is_file():
        raise StorageError(f"subject directory incomplete: {subject_dir}")
    current_h = sha256_of_file(heatmaps_path)
    current_s = sha256_of_file(stim_path)
    if current_h != meta.get("heatmaps_sha256") or current_s != meta.get("stimulus_indices_sha256"):
        raise StorageError(f"subject arrays do not match recorded checksums: {subject_dir}")
    return meta, current_h, current_s


# --------------------------------------------------------------------- accessor


@dataclass(frozen=True)
class TrialRecord:
    subject_id: str
    stimulus_id: str
    stimulus_index: int
    trial_uid: str
    group: str
    label: int
    category: str
    n_fixations_raw: int
    n_fixations_used: int
    n_transitions_used: int
    qc_status: str
    heatmap: np.ndarray  # float32 [3, H, W] view into the subject array


class TrialStore:
    """Indexed accessor over a completed processed dataset.

    Resolution happens exclusively through manifests; array positions are
    never inferred from identifier magnitude.
    """

    def __init__(self, processed_root: Path | str, verify_completion: bool = True) -> None:
        self.root = Path(processed_root)
        if verify_completion and not (self.root / "dataset_metadata.json").is_file():
            raise StorageError(
                f"processed dataset has no completion record: {self.root}"
                " (the dataset is incomplete or was never finalized)"
            )
        self._image_manifest = pd.read_csv(
            self.root / "image_manifest.csv", dtype={"stimulus_id": str, "category": str}
        )
        self._subject_manifest = pd.read_csv(
            self.root / "subject_manifest.csv",
            dtype={"subject_id": str, "group": str, "source_workbook": str},
        )
        self._trial_manifest = pd.read_parquet(self.root / "trial_manifest.parquet")
        self._trial_manifest = self._trial_manifest.set_index("trial_uid", drop=False)
        self._memmaps: dict[str, np.ndarray] = {}
        self._stim_indices: dict[str, np.ndarray] = {}

    @property
    def trial_manifest(self) -> pd.DataFrame:
        return self._trial_manifest

    @property
    def image_manifest(self) -> pd.DataFrame:
        return self._image_manifest

    @property
    def subject_manifest(self) -> pd.DataFrame:
        return self._subject_manifest

    def verify_manifest_checksums(self) -> None:
        """Verify manifest checksums recorded in the completion record."""
        meta = json.loads((self.root / "dataset_metadata.json").read_text(encoding="utf-8"))
        for fname, key in [
            ("image_manifest.csv", "image_manifest_sha256"),
            ("subject_manifest.csv", "subject_manifest_sha256"),
            ("trial_manifest.parquet", "trial_manifest_sha256"),
            ("qc_summary.json", "qc_summary_sha256"),
            ("source_inventory.json", "source_inventory_sha256"),
        ]:
            recorded = meta.get(key)
            if recorded is None:
                continue
            actual = sha256_of_file(self.root / fname)
            if actual != recorded:
                raise StorageError(
                    f"{fname} checksum mismatch: recorded {recorded}, actual {actual}"
                )

    def _subject_arrays(self, array_path: str) -> tuple[np.ndarray, np.ndarray]:
        if array_path in self._memmaps:
            return self._memmaps[array_path], self._stim_indices[array_path]
        path = self.root / array_path
        if not path.is_file():
            raise StorageError(f"subject array not found: {path}")
        heatmaps = np.load(path, mmap_mode="r")
        stim_idx_path = path.parent / "stimulus_indices.npy"
        stim_indices = np.load(stim_idx_path, mmap_mode="r")
        if heatmaps.ndim != 4:
            raise StorageError(f"corrupt subject array shape: {heatmaps.shape}")
        self._memmaps[array_path] = heatmaps
        self._stim_indices[array_path] = stim_indices
        return heatmaps, stim_indices

    def get_trial(self, subject_id: str, stimulus_id: str) -> TrialRecord:
        """Resolve the composite key ``(subject_id, stimulus_id)`` to its exact
        stored array row through the manifests."""
        uid = trial_uid(subject_id, stimulus_id)
        if uid not in self._trial_manifest.index:
            raise KeyError(f"trial not in manifest: ({subject_id}, {stimulus_id})")
        return self._record(uid)

    def get_trial_by_uid(self, uid: str) -> TrialRecord:
        if uid not in self._trial_manifest.index:
            raise KeyError(f"trial uid not in manifest: {uid}")
        return self._record(uid)

    def _record(self, uid: str) -> TrialRecord:
        row = self._trial_manifest.loc[uid]
        if row.qc_status != "ok":
            raise StorageError(
                f"trial {uid} is not heatmap-eligible (qc_status={row.qc_status})"
            )
        heatmaps, stim_indices = self._subject_arrays(str(row.subject_array_path))
        idx = int(row.subject_row_index)
        if int(stim_indices[idx]) != int(row.stimulus_index):
            raise StorageError(
                f"round-trip mismatch for trial {uid}: stimulus_indices[{idx}]="
                f"{stim_indices[idx]} but manifest stimulus_index={row.stimulus_index}"
            )
        heatmap = np.array(heatmaps[idx], dtype=np.float32)
        return TrialRecord(
            subject_id=str(row.subject_id),
            stimulus_id=str(row.stimulus_id),
            stimulus_index=int(row.stimulus_index),
            trial_uid=str(row.trial_uid),
            group=str(row.group),
            label=int(row.label),
            category=str(row.category),
            n_fixations_raw=int(row.n_fixations_raw),
            n_fixations_used=int(row.n_fixations_used),
            n_transitions_used=int(row.n_transitions_used),
            qc_status=str(row.qc_status),
            heatmap=heatmap,
        )
