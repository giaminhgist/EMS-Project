"""Synthetic fixtures for preprocessing tests (no full EMS data required)."""

from __future__ import annotations

from pathlib import Path

import pytest
from openpyxl import Workbook
from PIL import Image

HEADER = ["IMAGE", "FIX_INDEX", "FIX_DURATION", "FIX_X", "FIX_Y", "FIX_PUPIL"]


def write_image(path: Path, size: tuple[int, int] = (32, 24), color: tuple[int, int, int] = (120, 80, 40)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, format="JPEG")


def write_workbook(path: Path, rows: list[tuple[str, int, float, float, float, float]]) -> None:
    """Write a subject workbook with the Free_viewing sheet and a header row.

    Each row is ``(image, fix_index, duration_ms, x, y, pupil)``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    wb = Workbook()
    ws = wb.active
    ws.title = "Free_viewing"
    ws.append(HEADER)
    for row in rows:
        ws.append(list(row))
    wb.save(path)


def make_synthetic_raw(tmp_path: Path) -> Path:
    """Create a tiny synthetic EMS source tree.

    Images: ``CatA/a1.jpg``, ``CatA/a2.jpg``, ``CatB/b1.jpg``
    Subjects: 000 (HC), 005 (HC), 013 (HC) — non-contiguous IDs.
    """
    raw_root = tmp_path / "raw"
    fix_root = raw_root / "EMS" / "All_Data" / "Fixations"
    img_root = raw_root / "EMS" / "Images"
    for name, color in [("a1.jpg", (10, 20, 30)), ("a2.jpg", (40, 50, 60)), ("b1.jpg", (70, 80, 90))]:
        cat = "CatA" if name.startswith("a") else "CatB"
        write_image(img_root / cat / name, color=color)

    # Subject 000: a1 with shuffled FIX_INDEX row order, a2 with one off-canvas
    # fixation, b1 with one zero-duration fixation.
    write_workbook(
        fix_root / "000.xlsx",
        [
            ("a1.jpg", 2, 200.0, 600.0, 400.0, 1000.0),
            ("a1.jpg", 1, 150.0, 300.0, 400.0, 1000.0),
            ("a1.jpg", 3, 250.0, 800.0, 200.0, 1000.0),
            ("a2.jpg", 1, 300.0, 200.0, 300.0, 1000.0),
            ("a2.jpg", 2, 300.0, 1500.0, 900.0, 1000.0),  # off-canvas
            ("b1.jpg", 1, 0.0, 500.0, 384.0, 1000.0),  # zero duration
            ("b1.jpg", 2, 400.0, 512.0, 384.0, 1000.0),
            ("b1.jpg", 3, 400.0, 700.0, 500.0, 1000.0),
        ],
    )
    # Subject 005: only a1 and b1 observed (a2 missing -> must stay missing).
    write_workbook(
        fix_root / "005.xlsx",
        [
            ("a1.jpg", 1, 200.0, 100.0, 100.0, 900.0),
            ("a1.jpg", 2, 200.0, 900.0, 700.0, 900.0),
            ("b1.jpg", 1, 500.0, 400.0, 300.0, 900.0),
        ],
    )
    # Subject 013: a1 entirely off-canvas (excluded), a2 fine.
    write_workbook(
        fix_root / "013.xlsx",
        [
            ("a1.jpg", 1, 200.0, 1500.0, -200.0, 800.0),
            ("a1.jpg", 2, 200.0, 1800.0, 900.0, 800.0),
            ("a2.jpg", 1, 250.0, 512.0, 384.0, 800.0),
        ],
    )
    return raw_root


def synthetic_config_dict(raw_root: Path, output_root: Path, **overrides) -> dict:
    """Config dict for the synthetic tree (defaults filled by the schema)."""
    cfg = {
        "raw_root": str(raw_root),
        "fixation_root": str(raw_root / "EMS" / "All_Data" / "Fixations"),
        "image_root": str(raw_root / "EMS" / "Images"),
        "output_root": str(output_root),
        "num_workers": 0,
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture
def synthetic_raw(tmp_path: Path) -> Path:
    return make_synthetic_raw(tmp_path)


@pytest.fixture
def synthetic_processed(synthetic_raw: Path, tmp_path: Path) -> Path:
    """A complete synthetic processed dataset with HC and SZ subjects."""
    from preprocessing.config import PreprocessingConfig
    from preprocessing.pipeline import PipelineOptions, run_pipeline

    # Add two SZ subjects (numeric IDs >= 200) to the synthetic tree.
    write_workbook(
        synthetic_raw / "EMS" / "All_Data" / "Fixations" / "200.xlsx",
        [
            ("a1.jpg", 1, 220.0, 300.0, 400.0, 1000.0),
            ("a1.jpg", 2, 260.0, 700.0, 300.0, 1000.0),
            ("b1.jpg", 1, 180.0, 500.0, 384.0, 1000.0),
        ],
    )
    write_workbook(
        synthetic_raw / "EMS" / "All_Data" / "Fixations" / "201.xlsx",
        [
            ("a2.jpg", 1, 240.0, 200.0, 200.0, 1000.0),
            ("b1.jpg", 1, 300.0, 400.0, 300.0, 1000.0),
        ],
    )
    out = tmp_path / "processed"
    cfg = synthetic_config_dict(synthetic_raw, out)
    run_pipeline(PreprocessingConfig.from_dict(cfg), PipelineOptions())
    return out
