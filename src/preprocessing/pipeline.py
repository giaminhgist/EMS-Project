"""Preprocessing pipeline orchestration (inventory -> parse -> heatmaps -> store)."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import PreprocessingConfig
from .heatmaps import HeatmapParams, build_trial_heatmap
from .identifiers import trial_uid
from .inventory import (
    StimulusInfo,
    SubjectFileInfo,
    estimate_output_size,
    inventory_stimuli,
    inventory_subjects,
    source_inventory_json,
    validate_image_references,
    verify_free_space,
)
from .qc import TrialQC, compute_trial_qc
from .storage import (
    TRIAL_MANIFEST_COLUMNS,
    StorageError,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    publish_subject,
    sha256_of_file,
    verify_subject_dir,
)
from .trials import parse_workbook_trials


@dataclass
class PipelineOptions:
    dry_run: bool = False
    force: bool = False
    resume: bool = False
    subjects: list[str] | None = None
    output_root_override: Path | None = None


@dataclass
class SubjectOutput:
    subject_id: str
    n_fixation_rows: int
    # Filled during assembly (published in the write phase):
    heatmaps: np.ndarray | None = None
    stimulus_indices: np.ndarray | None = None
    qc_df: pd.DataFrame | None = None
    artifact_meta: dict[str, Any] | None = None


@dataclass
class PipelineReport:
    config_hash: str
    n_subjects: int
    n_trials_observed: int
    n_trials_ok: int
    n_trials_excluded: int
    dry_run: bool
    notes: list[str] = field(default_factory=list)


def _process_one_subject(task: dict[str, Any]) -> dict[str, Any]:
    """Worker entry point (module-level for pickling)."""
    subject_file = Path(task["path"])
    subject_id: str = task["subject_id"]
    columns: dict[str, str] = task["columns"]
    sheet_name: str = task["sheet_name"]
    source_width: int = task["source_width"]
    source_height: int = task["source_height"]
    min_fix_duration_ms: float = task["min_fix_duration_ms"]
    drop_nonpositive_duration: bool = task["drop_nonpositive_duration"]
    zero_spatial_policy: str = task["zero_spatial_policy"]
    stimulus_index_by_id: dict[str, int] = task["stimulus_index_by_id"]
    compute_heatmaps: bool = task["compute_heatmaps"]
    hp = HeatmapParams(
        height=task["heatmap_height"],
        width=task["heatmap_width"],
        sigma=task["sigma"],
        truncate=task["truncate"],
        transition_step=task["transition_step"],
        temporal_epsilon=task["temporal_epsilon"],
    )

    trials_by_image, n_fixation_rows = parse_workbook_trials(
        subject_file, subject_id, columns, sheet_name
    )
    outputs: list[dict[str, Any]] = []
    for image in sorted(trials_by_image):  # deterministic: image string order
        trial = trials_by_image[image]
        if image not in stimulus_index_by_id:
            raise StorageError(f"subject {subject_id}: image {image!r} not in stimulus inventory")
        stimulus_index = stimulus_index_by_id[image]
        sorted_events = trial.sorted_by_fix_index()
        qc = compute_trial_qc(
            subject_id,
            image,
            sorted_events,
            source_width=source_width,
            source_height=source_height,
            min_fix_duration_ms=min_fix_duration_ms,
            drop_nonpositive_duration=drop_nonpositive_duration,
            zero_spatial_policy=zero_spatial_policy,
        )
        heatmap = None
        heatmap_stats = None
        if qc.qc_status == "ok" and compute_heatmaps:
            xs = np.array([e.x if e.x is not None else np.nan for e in sorted_events])
            ys = np.array([e.y if e.y is not None else np.nan for e in sorted_events])
            spatial_valid = np.array(
                [
                    e.coord_status == "ok"
                    and e.x is not None
                    and e.y is not None
                    and 0.0 <= e.x <= source_width
                    and 0.0 <= e.y <= source_height
                    for e in sorted_events
                ]
            )
            durations = np.array(
                [e.duration if e.duration is not None else np.nan for e in sorted_events]
            )
            duration_valid = np.array(
                [e.duration_status == "ok" and (e.duration or 0.0) > min_fix_duration_ms for e in sorted_events]
            )
            heatmap, heatmap_stats = build_trial_heatmap(
                xs,
                ys,
                spatial_valid,
                durations,
                duration_valid,
                source_width=source_width,
                source_height=source_height,
                params=hp,
            )
        outputs.append(
            {
                "subject_id": subject_id,
                "stimulus_id": image,
                "stimulus_index": stimulus_index,
                "qc": qc,
                "heatmap": heatmap,
                "heatmap_stats": heatmap_stats,
            }
        )
    return {"subject_id": subject_id, "n_fixation_rows": n_fixation_rows, "trials": outputs}


def _library_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    try:
        import pyarrow

        versions["pyarrow"] = pyarrow.__version__
    except ImportError:  # pragma: no cover
        versions["pyarrow"] = "unavailable"
    try:
        import openpyxl

        versions["openpyxl"] = openpyxl.__version__
    except ImportError:  # pragma: no cover
        versions["openpyxl"] = "unavailable"
    try:
        import PIL

        versions["pillow"] = PIL.__version__
    except ImportError:  # pragma: no cover
        versions["pillow"] = "unavailable"
    return versions


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd="/root/EMS-Project",
            check=False,
            timeout=10,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        pass
    return None


def run_pipeline(
    config: PreprocessingConfig, options: PipelineOptions | None = None
) -> PipelineReport:
    options = options or PipelineOptions()
    output_root = Path(options.output_root_override or config.output_root)

    # 1. Inventory (validates source layout before any write).
    stimuli = inventory_stimuli(config.image_root)
    stimulus_index_by_id = {s.stimulus_id: s.stimulus_index for s in stimuli}
    all_subjects, other_files = inventory_subjects(
        config.fixation_root,
        config.subject_glob,
        config.subject_filename_regex,
        split=config.hc_sz_split,
    )
    if options.subjects:
        wanted = set(options.subjects)
        known = {s.identity.subject_id for s in all_subjects}
        unknown = sorted(wanted - known)
        if unknown:
            raise StorageError(f"requested subjects not in inventory: {unknown}")
        selected = [s for s in all_subjects if s.identity.subject_id in wanted]
    else:
        selected = all_subjects

    config_hash = config.config_hash()

    # 2. Parse + QC (+ heatmaps) per subject.
    tasks = []
    for s in selected:
        tasks.append(
            {
                "path": str(s.path),
                "subject_id": s.identity.subject_id,
                "columns": config.columns,
                "sheet_name": config.sheet_name,
                "source_width": config.source_width,
                "source_height": config.source_height,
                "min_fix_duration_ms": config.min_fix_duration_ms,
                "drop_nonpositive_duration": config.drop_nonpositive_duration,
                "zero_spatial_policy": config.zero_spatial_policy,
                "stimulus_index_by_id": stimulus_index_by_id,
                "compute_heatmaps": not options.dry_run,
                "heatmap_height": config.heatmap_height,
                "heatmap_width": config.heatmap_width,
                "sigma": config.gaussian_sigma_cells,
                "truncate": config.gaussian_truncate_sigma,
                "transition_step": config.transition_sample_step_cells,
                "temporal_epsilon": config.temporal_epsilon,
            }
        )

    use_pool = config.num_workers > 1 and len(tasks) > 1
    if use_pool and not options.dry_run:
        with ProcessPoolExecutor(max_workers=config.num_workers) as ex:
            raw_results = list(ex.map(_process_one_subject, tasks))
    else:
        raw_results = [_process_one_subject(t) for t in tasks]

    # Map back to subject identity order and validate image references.
    results_by_id = {r["subject_id"]: r for r in raw_results}
    if len(results_by_id) != len(selected):
        raise StorageError("duplicate subject results from workers")
    for s in selected:
        validate_image_references(
            {t["stimulus_id"]: None for t in results_by_id[s.identity.subject_id]["trials"]},
            stimuli,
        )

    # 3. Assemble per-subject arrays and trial rows in deterministic order.
    subject_outputs: dict[str, SubjectOutput] = {}
    category_by_id = {s.stimulus_id: s.category for s in stimuli}
    all_trial_rows: list[dict[str, Any]] = []
    anomaly = {
        "n_fixation_rows": 0,
        "n_trials_observed": 0,
        "n_off_canvas_rows": 0,
        "n_nonfinite_rows": 0,
        "n_nonpositive_duration_rows": 0,
        "n_below_duration_threshold_rows": 0,
        "n_trials_fix_index_gaps": 0,
        "n_trials_fix_index_duplicates": 0,
        "n_trials_fix_index_non_monotonic": 0,
        "n_trials_zero_spatial": 0,
        "n_temporal_undefined": 0,
        "n_malformed_fix_index": 0,
        "n_malformed_duration": 0,
        "n_malformed_pupil": 0,
    }
    excluded_trials: list[dict[str, str]] = []

    for s in selected:
        ident = s.identity
        raw = results_by_id[ident.subject_id]
        anomaly["n_fixation_rows"] += raw["n_fixation_rows"]
        subject_trials = raw["trials"]
        subject_trials.sort(key=lambda t: t["stimulus_index"])
        ok_trials = [t for t in subject_trials if t["qc"].qc_status == "ok"]
        heatmaps_rows = [t["heatmap"] for t in ok_trials]
        stimulus_indices = np.array([t["stimulus_index"] for t in ok_trials], dtype=np.int64)
        heatmaps = (
            np.stack(heatmaps_rows).astype(np.float32) if heatmaps_rows else np.empty((0, 3, config.heatmap_height, config.heatmap_width), dtype=np.float32)
        )

        row_index_by_uid: dict[str, int] = {}
        for pos, t in enumerate(ok_trials):
            row_index_by_uid[trial_uid(ident.subject_id, t["stimulus_id"])] = pos

        subject_qc_rows: list[dict[str, Any]] = []
        for t in subject_trials:
            qc: TrialQC = t["qc"]
            anomaly["n_trials_observed"] += 1
            anomaly["n_off_canvas_rows"] += qc.n_off_canvas
            anomaly["n_nonfinite_rows"] += qc.n_nonfinite
            anomaly["n_nonpositive_duration_rows"] += qc.n_nonpositive_duration
            anomaly["n_below_duration_threshold_rows"] += qc.n_below_duration_threshold
            anomaly["n_trials_fix_index_gaps"] += int(qc.fix_index_has_gap)
            anomaly["n_trials_fix_index_duplicates"] += int(qc.fix_index_has_duplicates)
            anomaly["n_trials_fix_index_non_monotonic"] += int(qc.fix_index_non_monotonic)
            anomaly["n_temporal_undefined"] += int(qc.temporal_undefined)
            anomaly["n_malformed_fix_index"] += qc.n_malformed_fix_index
            anomaly["n_malformed_duration"] += qc.n_malformed_duration
            anomaly["n_malformed_pupil"] += qc.n_malformed_pupil
            if qc.qc_status != "ok":
                anomaly["n_trials_zero_spatial"] += 1
                excluded_trials.append(
                    {"subject_id": ident.subject_id, "stimulus_id": t["stimulus_id"]}
                )

            uid = trial_uid(ident.subject_id, t["stimulus_id"])
            row_index = row_index_by_uid.get(uid)
            stats = t["heatmap_stats"]
            all_trial_rows.append(
                {
                    "trial_uid": uid,
                    "subject_id": ident.subject_id,
                    "stimulus_id": t["stimulus_id"],
                    "subject_numeric_id": ident.subject_numeric_id,
                    "stimulus_index": t["stimulus_index"],
                    "group": ident.group,
                    "label": ident.label,
                    "category": category_by_id[t["stimulus_id"]],
                    "subject_array_path": (
                        f"subjects/{ident.subject_id}/heatmaps.npy" if row_index is not None else None
                    ),
                    "subject_row_index": row_index,
                    "n_fixations_raw": qc.n_fixations_raw,
                    "n_fixations_used": qc.n_fixations_used,
                    "n_transitions_used": qc.n_transitions_used,
                    "total_duration_ms_raw": qc.total_duration_ms_raw,
                    "total_duration_ms_used": qc.total_duration_ms_used,
                    "n_off_canvas": qc.n_off_canvas,
                    "n_nonfinite": qc.n_nonfinite,
                    "n_nonpositive_duration": qc.n_nonpositive_duration,
                    "n_below_duration_threshold": qc.n_below_duration_threshold,
                    "fix_index_has_gap": qc.fix_index_has_gap,
                    "qc_status": qc.qc_status,
                }
            )
            subject_qc_rows.append(
                {
                    "trial_uid": uid,
                    "stimulus_id": t["stimulus_id"],
                    "stimulus_index": t["stimulus_index"],
                    "subject_row_index": row_index,
                    "qc_status": qc.qc_status,
                    "median_fixation_duration_ms": qc.median_fixation_duration_ms,
                    **{k: v for k, v in qc.to_dict().items() if k != "qc_status"},
                    "mass_density": (stats.mass_density if stats else None),
                    "mass_transition": (stats.mass_transition if stats else None),
                    "mass_density_error": (
                        abs(stats.mass_density - stats.mass_density_expected) if stats else None
                    ),
                    "mass_transition_error": (
                        abs(stats.mass_transition - stats.mass_transition_expected) if stats else None
                    ),
                }
            )

        subject_outputs[ident.subject_id] = SubjectOutput(
            subject_id=ident.subject_id,
            n_fixation_rows=raw["n_fixation_rows"],
        )
        subject_qc_df = pd.DataFrame(subject_qc_rows)
        subject_qc_df["subject_row_index"] = subject_qc_df["subject_row_index"].astype("Int64")
        for col in [
            "median_fixation_duration_ms",
            "mass_density",
            "mass_transition",
            "mass_density_error",
            "mass_transition_error",
        ]:
            subject_qc_df[col] = subject_qc_df[col].astype("Float64")
        artifact_meta = {
            "subject_id": ident.subject_id,
            "config_hash": config_hash,
            "source_workbook_sha256": s.sha256,
            "n_trials_total": len(subject_trials),
            "n_trials_ok": len(ok_trials),
        }
        # Store arrays and meta for the write phase.
        subject_outputs[ident.subject_id].heatmaps = heatmaps
        subject_outputs[ident.subject_id].stimulus_indices = stimulus_indices
        subject_outputs[ident.subject_id].qc_df = subject_qc_df
        subject_outputs[ident.subject_id].artifact_meta = artifact_meta

    n_trials_ok = sum(1 for r in all_trial_rows if r["qc_status"] == "ok")
    report = PipelineReport(
        config_hash=config_hash,
        n_subjects=len(selected),
        n_trials_observed=anomaly["n_trials_observed"],
        n_trials_ok=n_trials_ok,
        n_trials_excluded=anomaly["n_trials_zero_spatial"],
        dry_run=options.dry_run,
    )

    # 4. Dry-run: report only, write nothing.
    if options.dry_run:
        estimated = estimate_output_size(
            n_trials_ok, config.heatmap_height, config.heatmap_width
        )
        free = verify_free_space(output_root, estimated)
        report.notes.append(
            f"dry-run: would write {len(selected)} subject dirs, {n_trials_ok} heatmaps "
            f"({estimated / 1e6:.1f} MB), {free / 1e9:.2f} GB free"
        )
        return report

    # 5. Publish per-subject directories atomically.
    verify_free_space(
        output_root,
        estimate_output_size(n_trials_ok, config.heatmap_height, config.heatmap_width),
    )
    for s in selected:
        so = subject_outputs[s.identity.subject_id]
        heatmaps = so.heatmaps
        stimulus_indices = so.stimulus_indices
        qc_df = so.qc_df
        meta = dict(so.artifact_meta or {})
        assert heatmaps is not None and stimulus_indices is not None and qc_df is not None
        subject_dir = output_root / "subjects" / s.identity.subject_id
        if subject_dir.exists() and not options.force:
            # A rerun with unchanged config and sources verifies and skips
            # compatible outputs; incompatible or incomplete outputs fail
            # (--force replaces after target validation).
            verify_subject_dir(subject_dir, config_hash)
            report.notes.append(f"skipped existing subject {s.identity.subject_id}")
            continue
        publish_subject(
            output_root,
            s.identity.subject_id,
            heatmaps,
            stimulus_indices,
            qc_df,
            meta,
            height=config.heatmap_height,
            width=config.heatmap_width,
            n_stimuli=len(stimuli),
            force=options.force,
        )

    # 6. Global artifacts (manifests last, completion record very last).
    is_canonical = (
        options.subjects is None
        and (options.output_root_override is None or Path(options.output_root_override) == Path(config.output_root))
    )
    if is_canonical:
        _write_global_artifacts(
            config=config,
            config_hash=config_hash,
            output_root=output_root,
            stimuli=stimuli,
            selected=selected,
            other_files=other_files,
            all_trial_rows=all_trial_rows,
            anomaly=anomaly,
            excluded_trials=excluded_trials,
        )
    else:
        report.notes.append(
            "partial run: global completion record NOT written (smoke output only)"
        )
    return report


def _write_global_artifacts(
    config: PreprocessingConfig,
    config_hash: str,
    output_root: Path,
    stimuli: list[StimulusInfo],
    selected: list[SubjectFileInfo],
    other_files: list[str],
    all_trial_rows: list[dict[str, Any]],
    anomaly: dict[str, Any],
    excluded_trials: list[dict[str, str]],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    # image_manifest.csv (atomic)
    import io

    image_manifest = pd.DataFrame([s.to_manifest_row() for s in stimuli])
    buf = io.StringIO()
    image_manifest.to_csv(buf, index=False)
    atomic_write_bytes(output_root / "image_manifest.csv", buf.getvalue().encode("utf-8"))

    # subject_manifest.csv (atomic)
    subject_rows = []
    for s in selected:
        ident = s.identity
        trials_here = [r for r in all_trial_rows if r["subject_id"] == ident.subject_id]
        subject_rows.append(
            {
                "subject_id": ident.subject_id,
                "subject_numeric_id": ident.subject_numeric_id,
                "group": ident.group,
                "label": ident.label,
                "source_workbook": ident.source_workbook,
                "n_fixation_rows": sum(r["n_fixations_raw"] for r in trials_here),
                "n_trials": len(trials_here),
                "n_missing_expected_stimuli": len(stimuli) - len(trials_here),
                "source_sha256": s.sha256,
            }
        )
    subject_manifest = pd.DataFrame(subject_rows)
    buf = io.StringIO()
    subject_manifest.to_csv(buf, index=False)
    atomic_write_bytes(output_root / "subject_manifest.csv", buf.getvalue().encode("utf-8"))

    # trial_manifest.parquet (+ optional CSV export)
    all_trial_rows.sort(key=lambda r: (r["subject_numeric_id"], r["stimulus_index"]))
    trial_manifest = pd.DataFrame(all_trial_rows, columns=TRIAL_MANIFEST_COLUMNS)
    for col in ["trial_uid", "subject_id", "stimulus_id", "group", "category", "qc_status"]:
        trial_manifest[col] = trial_manifest[col].astype("string")
    trial_manifest["subject_array_path"] = trial_manifest["subject_array_path"].astype("string")
    trial_manifest["subject_row_index"] = trial_manifest["subject_row_index"].astype("Int64")
    trial_manifest.to_parquet(output_root / "trial_manifest.parquet", index=False)
    if config.write_trial_manifest_csv:
        trial_manifest.to_csv(output_root / "trial_manifest.csv", index=False)

    # qc_summary.json
    qc_summary = {
        "config_hash": config_hash,
        "n_subjects": len(selected),
        "n_subjects_hc": sum(1 for s in selected if s.identity.label == 0),
        "n_subjects_sz": sum(1 for s in selected if s.identity.label == 1),
        **anomaly,
        "excluded_trials": excluded_trials,
        "notes": [
            "off_canvas_policy=drop: off-canvas fixations are excluded from spatial "
            "maps but retained in QC counts",
            "zero_spatial_policy=exclude_no_spatial_fixations: trials with no usable "
            "spatial fixation are recorded in the trial manifest with qc_status "
            "'excluded_no_spatial_fixations' and have no heatmap row",
        ],
    }
    atomic_write_json(output_root / "qc_summary.json", qc_summary)

    # source_inventory.json (reproducible: no timestamps)
    atomic_write_text(
        output_root / "source_inventory.json",
        source_inventory_json(stimuli, selected, other_files, anomaly),
    )

    # Resolved configuration
    atomic_write_text(output_root / "preprocessing_config.json", config.to_json())

    # dataset_metadata.json completion record (last).
    n_ok = sum(1 for r in all_trial_rows if r["qc_status"] == "ok")
    subjects_meta: dict[str, Any] = {}
    for s in selected:
        subject_dir = output_root / "subjects" / s.identity.subject_id
        meta_path = subject_dir / "artifact_meta.json"
        if not meta_path.is_file():
            raise StorageError(f"missing artifact metadata for subject {s.identity.subject_id}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        subjects_meta[s.identity.subject_id] = {
            "heatmaps_sha256": meta["heatmaps_sha256"],
            "stimulus_indices_sha256": meta["stimulus_indices_sha256"],
            "n_trials_total": meta["n_trials_total"],
            "n_trials_ok": meta["n_trials_ok"],
            "source_workbook_sha256": meta["source_workbook_sha256"],
        }
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_hash": config_hash,
        "seed": config.seed,
        "git_commit": _git_commit(),
        "library_versions": _library_versions(),
        "image_manifest_sha256": sha256_of_file(output_root / "image_manifest.csv"),
        "subject_manifest_sha256": sha256_of_file(output_root / "subject_manifest.csv"),
        "trial_manifest_sha256": sha256_of_file(output_root / "trial_manifest.parquet"),
        "trial_manifest_csv_sha256": (
            sha256_of_file(output_root / "trial_manifest.csv")
            if config.write_trial_manifest_csv
            else None
        ),
        "qc_summary_sha256": sha256_of_file(output_root / "qc_summary.json"),
        "source_inventory_sha256": sha256_of_file(output_root / "source_inventory.json"),
        "preprocessing_config_sha256": sha256_of_file(output_root / "preprocessing_config.json"),
        "counts": {
            "n_subjects": len(selected),
            "n_fixation_rows": anomaly["n_fixation_rows"],
            "n_trials_observed": anomaly["n_trials_observed"],
            "n_trials_ok": n_ok,
            "n_trials_excluded_no_spatial_fixations": anomaly["n_trials_zero_spatial"],
        },
        "subjects": subjects_meta,
    }
    atomic_write_json(output_root / "dataset_metadata.json", metadata)
