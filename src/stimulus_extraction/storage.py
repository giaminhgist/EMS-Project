"""Storage and verification for frozen DINO feature artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from preprocessing.storage import atomic_write_bytes, atomic_write_json, atomic_write_text

FEATURE_MANIFEST_COLUMNS = [
    "stimulus_index",
    "stimulus_id",
    "feature_row_index",
    "image_sha256",
    "feature_shape",
    "feature_dtype",
    "feature_sha256",
]


class StorageError(ValueError):
    pass


def token_sha256(tokens: np.ndarray) -> str:
    """SHA-256 of a feature row's raw float32 bytes."""
    import hashlib

    arr = np.ascontiguousarray(tokens, dtype=np.float32)
    if arr.ndim != 2:
        raise StorageError(f"feature row must be 2-D, got shape {arr.shape}")
    return hashlib.sha256(arr.tobytes()).hexdigest()


@dataclass(frozen=True)
class FeaturePaths:
    output_dir: Path
    patch_tokens: Path
    feature_manifest: Path
    extraction_config: Path
    model_metadata: Path
    validation_report: Path

    @classmethod
    def for_dir(cls, output_dir: Path) -> "FeaturePaths":
        return cls(
            output_dir=Path(output_dir),
            patch_tokens=Path(output_dir) / "patch_tokens.npy",
            feature_manifest=Path(output_dir) / "feature_manifest.csv",
            extraction_config=Path(output_dir) / "extraction_config.json",
            model_metadata=Path(output_dir) / "model_metadata.json",
            validation_report=Path(output_dir) / "validation_report.json",
        )


def open_memmap(path: Path, n_rows: int, n_tokens: int, dim: int) -> np.ndarray:
    """Create (or open r+) a float32 memmap for patch tokens."""
    return np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(n_rows, n_tokens, dim)
    )


def write_feature_manifest(rows: list[dict[str, Any]], paths: FeaturePaths) -> None:
    df = pd.DataFrame(rows, columns=FEATURE_MANIFEST_COLUMNS)
    for col in ["stimulus_id", "image_sha256", "feature_shape", "feature_dtype", "feature_sha256"]:
        df[col] = df[col].astype("string")
    import io

    buf = io.StringIO()
    df.to_csv(buf, index=False)
    atomic_write_bytes(paths.feature_manifest, buf.getvalue().encode("utf-8"))


def read_feature_manifest(paths: FeaturePaths) -> pd.DataFrame | None:
    if not paths.feature_manifest.is_file():
        return None
    return pd.read_csv(
        paths.feature_manifest,
        dtype={
            "stimulus_index": int,
            "stimulus_id": str,
            "feature_row_index": int,
            "image_sha256": str,
            "feature_shape": str,
            "feature_dtype": str,
            "feature_sha256": str,
        },
    )


def write_extraction_config(
    cfg: Any, paths: FeaturePaths, actual_image_preprocessing: dict[str, Any]
) -> None:
    payload = {
        "resolved_config": cfg.to_dict(),
        "config_hash": cfg.config_hash(),
        "actual_image_preprocessing": actual_image_preprocessing,
    }
    atomic_write_json(paths.extraction_config, payload)


def write_model_metadata(metadata: Any, paths: FeaturePaths) -> None:
    atomic_write_json(paths.model_metadata, metadata.to_dict() if hasattr(metadata, "to_dict") else metadata)


def write_validation_report(report: dict[str, Any], paths: FeaturePaths) -> None:
    atomic_write_json(paths.validation_report, report)


def verify_row(memmap: np.ndarray, row_index: int, expected_sha256: str) -> None:
    """Verify one completed feature row against its recorded checksum."""
    if row_index >= len(memmap):
        raise StorageError(f"row {row_index} beyond memmap length {len(memmap)}")
    row = np.ascontiguousarray(memmap[row_index], dtype=np.float32)
    if token_sha256(row) != expected_sha256:
        raise StorageError(f"row {row_index} checksum mismatch (recorded {expected_sha256})")
    if not np.all(np.isfinite(row)):
        raise StorageError(f"row {row_index} contains non-finite values")


def verify_model_identity(cfg: Any, paths: FeaturePaths, metadata: Any) -> None:
    """Refuse to resume/verify against a different model or configuration."""
    if not paths.model_metadata.is_file():
        raise StorageError("model_metadata.json missing; cannot verify model identity")
    stored = json.loads(paths.model_metadata.read_text(encoding="utf-8"))
    if stored.get("hub_source") != cfg.hub_source or stored.get("model_name") != cfg.model_name:
        raise StorageError(
            f"model identity mismatch: stored {stored.get('hub_source')}/{stored.get('model_name')} "
            f"vs configured {cfg.hub_source}/{cfg.model_name}"
        )
    if stored.get("checkpoint_sha256") and metadata.checkpoint_sha256:
        if stored["checkpoint_sha256"] != metadata.checkpoint_sha256:
            raise StorageError("checkpoint SHA-256 mismatch between runs")
    if not paths.extraction_config.is_file():
        raise StorageError("extraction_config.json missing; cannot verify configuration")
    ext_cfg = json.loads(paths.extraction_config.read_text(encoding="utf-8"))
    if ext_cfg.get("config_hash") != cfg.config_hash():
        raise StorageError("extraction config hash mismatch between runs")
