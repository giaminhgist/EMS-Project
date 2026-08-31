"""Synthetic CV fixtures: a 12-subject processed dataset (7 HC / 5 SZ)."""

from __future__ import annotations

import sys
from pathlib import Path

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

HC_IDS = ["000", "001", "002", "003", "004", "005", "013"]
SZ_IDS = ["200", "201", "202", "203", "204"]
ALL_IDS = HC_IDS + SZ_IDS

_TRIAL_PLANS = {
    "000": [("a1.jpg", 3), ("a2.jpg", 2), ("b1.jpg", 2)],
    "001": [("a1.jpg", 2), ("b1.jpg", 1)],
    "002": [("a2.jpg", 2), ("b1.jpg", 2)],
    "003": [("a1.jpg", 1), ("a2.jpg", 1), ("b1.jpg", 1)],
    "004": [("a1.jpg", 2), ("a2.jpg", 2)],
    "005": [("a1.jpg", 2), ("b1.jpg", 1)],
    "013": [("a2.jpg", 1)],  # substantially incomplete
    "200": [("a1.jpg", 2), ("b1.jpg", 2)],
    "201": [("a2.jpg", 1), ("b1.jpg", 1)],
    "202": [("a1.jpg", 1), ("a2.jpg", 1), ("b1.jpg", 1)],
    "203": [("a1.jpg", 2)],
    "204": [("b1.jpg", 2), ("a2.jpg", 2)],
}


def make_cv_raw(root: Path) -> Path:
    """Raw tree with 12 subjects; subject 005 has an off-canvas row, and 013
    sees only one stimulus."""
    raw = make_synthetic_raw(root / "rawbase")
    fix_root = raw / "EMS" / "All_Data" / "Fixations"
    # Overwrite the three conftest subjects with controlled plans.
    import shutil

    for sid in ["000", "005", "013"]:
        (fix_root / f"{sid}.xlsx").unlink()
    counter = {"n": 0}

    def rows_for(image: str, n: int) -> list[tuple]:
        out = []
        for i in range(1, n + 1):
            counter["n"] += 1
            x = 100.0 + (counter["n"] * 137) % 800
            y = 100.0 + (counter["n"] * 211) % 600
            out.append((image, i, 150.0 + 30 * i, x, y, 900.0 + i))
        return out

    for sid, plan in _TRIAL_PLANS.items():
        rows = []
        for image, n in plan:
            rows.extend(rows_for(image, n))
        write_workbook(fix_root / f"{sid}.xlsx", rows)
    return raw


def cv_config_dict(processed_root: Path, output_root: Path, **overrides) -> dict:
    cfg = {
        "n_splits": 5,
        "shuffle": True,
        "random_state": 2026,
        "stratify_column": "label",
        "group_column": "subject_id",
        "subject_manifest": str(processed_root / "subject_manifest.csv"),
        "trial_manifest": str(processed_root / "trial_manifest.parquet"),
        "source_inventory": str(processed_root / "source_inventory.json"),
        "output_root": str(output_root),
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture(scope="module")
def cv_processed(tmp_path_factory):
    """Module-scoped synthetic processed dataset with HC and SZ subjects."""
    from preprocessing.config import PreprocessingConfig
    from preprocessing.pipeline import PipelineOptions, run_pipeline

    base = tmp_path_factory.mktemp("cv_raw")
    raw = make_cv_raw(base)
    processed = tmp_path_factory.mktemp("cv_processed")
    cfg = synthetic_config_dict(raw, processed)
    run_pipeline(PreprocessingConfig.from_dict(cfg), PipelineOptions())
    return processed
