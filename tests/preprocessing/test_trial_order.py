"""Trial-order tests: FIX_INDEX sorting, missing stimuli, index hygiene."""

from __future__ import annotations

import numpy as np

from preprocessing.heatmaps import HeatmapParams, build_trial_heatmap
from preprocessing.qc import compute_trial_qc
from preprocessing.trials import parse_workbook_trials
from tests.preprocessing.conftest import write_workbook

PARAMS = HeatmapParams(height=48, width=64, sigma=2.0, truncate=4.0, transition_step=0.5, temporal_epsilon=1e-8)
COLS = {
    "image": "IMAGE",
    "fix_index": "FIX_INDEX",
    "fix_duration": "FIX_DURATION",
    "fix_x": "FIX_X",
    "fix_y": "FIX_Y",
    "fix_pupil": "FIX_PUPIL",
}


def _trial_heatmap(trial):
    events = trial.sorted_by_fix_index()
    xs = np.array([e.x for e in events], dtype=np.float64)
    ys = np.array([e.y for e in events], dtype=np.float64)
    spatial = np.array(
        [e.coord_status == "ok" and 0 <= e.x <= 1024 and 0 <= e.y <= 768 for e in events]
    )
    durations = np.array([e.duration for e in events], dtype=np.float64)
    dur_ok = np.array([e.duration_status == "ok" and e.duration > 0 for e in events])
    heatmap, _ = build_trial_heatmap(
        xs, ys, spatial, durations, dur_ok,
        source_width=1024, source_height=768, params=PARAMS,
    )
    return heatmap


def test_shuffled_workbook_rows_yield_same_maps(tmp_path):
    events = [
        ("a1.jpg", 3, 250.0, 800.0, 200.0, 1000.0),
        ("a1.jpg", 1, 150.0, 300.0, 400.0, 1000.0),
        ("a1.jpg", 2, 200.0, 600.0, 400.0, 1000.0),
    ]
    p1 = tmp_path / "shuffled.xlsx"
    p2 = tmp_path / "ordered.xlsx"
    write_workbook(p1, events)
    write_workbook(p2, sorted(events, key=lambda r: r[1]))
    t1 = parse_workbook_trials(p1, "000", COLS, "Free_viewing")[0]["a1.jpg"]
    t2 = parse_workbook_trials(p2, "000", COLS, "Free_viewing")[0]["a1.jpg"]
    assert [e.fix_index for e in t1.sorted_by_fix_index()] == [1, 2, 3]
    h1 = _trial_heatmap(t1)
    h2 = _trial_heatmap(t2)
    assert np.array_equal(h1, h2)


def test_fix_index_sorting_not_row_order(tmp_path):
    # Rows are written in reverse FIX_INDEX order; sorting must restore order.
    rows = [
        ("b1.jpg", 3, 100.0, 700.0, 500.0, 900.0),
        ("b1.jpg", 1, 100.0, 100.0, 100.0, 900.0),
        ("b1.jpg", 2, 100.0, 400.0, 300.0, 900.0),
    ]
    p = tmp_path / "rev.xlsx"
    write_workbook(p, rows)
    trial = parse_workbook_trials(p, "000", COLS, "Free_viewing")[0]["b1.jpg"]
    assert [e.fix_index for e in trial.sorted_by_fix_index()] == [1, 2, 3]
    # Original row order is non-monotonic and flagged in QC.
    qc = compute_trial_qc(
        "000", "b1.jpg", trial.sorted_by_fix_index(),
        source_width=1024, source_height=768, min_fix_duration_ms=0,
        drop_nonpositive_duration=True, zero_spatial_policy="exclude_no_spatial",
    )
    assert qc.fix_index_non_monotonic is True
    assert qc.fix_index_has_gap is False
    assert qc.fix_index_has_duplicates is False


def test_missing_stimuli_do_not_generate_synthetic_trials(tmp_path):
    rows = [
        ("a1.jpg", 1, 100.0, 100.0, 100.0, 900.0),
        ("b1.jpg", 1, 100.0, 400.0, 300.0, 900.0),
    ]
    p = tmp_path / "subset.xlsx"
    write_workbook(p, rows)
    trials, n_rows = parse_workbook_trials(p, "000", COLS, "Free_viewing")
    assert n_rows == 2
    assert set(trials) == {"a1.jpg", "b1.jpg"}  # no a2.jpg row created


def test_duplicate_and_gap_fix_index_flags(tmp_path):
    rows = [
        ("a1.jpg", 1, 100.0, 100.0, 100.0, 900.0),
        ("a1.jpg", 1, 100.0, 400.0, 300.0, 900.0),  # duplicate index
        ("a1.jpg", 3, 100.0, 700.0, 500.0, 900.0),  # gap (2 missing)
    ]
    p = tmp_path / "dup_gap.xlsx"
    write_workbook(p, rows)
    trial = parse_workbook_trials(p, "000", COLS, "Free_viewing")[0]["a1.jpg"]
    qc = compute_trial_qc(
        "000", "a1.jpg", trial.sorted_by_fix_index(),
        source_width=1024, source_height=768, min_fix_duration_ms=0,
        drop_nonpositive_duration=True, zero_spatial_policy="exclude_no_spatial",
    )
    assert qc.fix_index_has_duplicates is True
    assert qc.fix_index_has_gap is True
    assert qc.qc_status == "ok"  # spatial fixations still usable


def test_zero_spatial_trial_excluded(tmp_path):
    rows = [
        ("a1.jpg", 1, 100.0, 1500.0, 900.0, 900.0),
        ("a1.jpg", 2, 100.0, -100.0, 400.0, 900.0),
    ]
    p = tmp_path / "zero_spatial.xlsx"
    write_workbook(p, rows)
    trial = parse_workbook_trials(p, "000", COLS, "Free_viewing")[0]["a1.jpg"]
    qc = compute_trial_qc(
        "000", "a1.jpg", trial.sorted_by_fix_index(),
        source_width=1024, source_height=768, min_fix_duration_ms=0,
        drop_nonpositive_duration=True, zero_spatial_policy="exclude_no_spatial",
    )
    assert qc.qc_status == "excluded_no_spatial_fixations"
    assert qc.n_off_canvas == 2
    assert qc.n_fixations_used == 0
