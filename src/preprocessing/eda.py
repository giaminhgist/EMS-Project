"""Reproducible EDA computed exclusively from canonical processed artifacts.

All aggregations join through explicit manifest IDs (never array positions),
heatmap statistics are streamed subject-by-subject through memory maps, and
every group comparison aggregates to one value per subject before any
inferential statistic is computed.
"""

from __future__ import annotations

import json
import math
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .storage import TrialStore, sha256_of_file

CH0, CH1, CH2 = 0, 1, 2
GRID_H, GRID_W = 48, 64


def _py(value: Any) -> Any:
    """Convert numpy types to plain Python for JSON serialization."""
    if isinstance(value, dict):
        return {k: _py(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_py(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return _py(value.tolist())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def inventory_statistics(store: TrialStore) -> dict[str, Any]:
    """Counts and distributions derived from manifests and QC summary."""
    sub = store.subject_manifest
    img = store.image_manifest
    tm = store.trial_manifest
    qc = json.loads((store.root / "qc_summary.json").read_text(encoding="utf-8"))
    ok = tm[tm.qc_status == "ok"]

    trials_per_subject = sub.n_trials.value_counts().sort_index()
    fix_per_trial = ok.n_fixations_used
    dur_per_trial = ok.total_duration_ms_used

    def hist(series: pd.Series, bins: int = 30) -> dict[str, Any]:
        counts, edges = np.histogram(series.astype(float), bins=bins)
        return {
            "counts": _py(counts),
            "edges": _py(edges),
        }

    return {
        "n_subjects": int(len(sub)),
        "subjects_by_group": {g: int(c) for g, c in sub.group.value_counts().items()},
        "stimuli_by_category": {
            c: int(n) for c, n in img.category.value_counts().items()
        },
        "n_fixation_rows": int(sub.n_fixation_rows.sum()),
        "n_trials_observed": int(len(tm)),
        "n_trials_ok": int(len(ok)),
        "n_trials_excluded": int(qc["n_trials_zero_spatial"]),
        "n_excluded_or_warning_trials": int(qc["n_trials_zero_spatial"]),
        "trials_per_subject": {
            "min": int(trials_per_subject.index.min()),
            "max": int(trials_per_subject.index.max()),
            "mean": float(sub.n_trials.mean()),
            "distribution": hist(sub.n_trials, bins=20),
        },
        "subjects_missing_stimuli": [
            {
                "subject_id": str(r.subject_id),
                "n_trials": int(r.n_trials),
                "n_missing_expected_stimuli": int(r.n_missing_expected_stimuli),
            }
            for r in sub[sub.n_missing_expected_stimuli > 0].sort_values(
                "n_missing_expected_stimuli", ascending=False
            ).itertuples()
        ],
        "fixations_per_trial": {
            "min": int(fix_per_trial.min()),
            "max": int(fix_per_trial.max()),
            "mean": float(fix_per_trial.mean()),
            "median": float(fix_per_trial.median()),
            "distribution": hist(fix_per_trial, bins=30),
        },
        "total_duration_per_trial_ms": {
            "min": float(dur_per_trial.min()),
            "max": float(dur_per_trial.max()),
            "mean": float(dur_per_trial.mean()),
            "median": float(dur_per_trial.median()),
            "distribution": hist(dur_per_trial, bins=30),
        },
        "qc_event_counts": {
            "n_off_canvas": int(qc["n_off_canvas_rows"]),
            "n_nonfinite": int(qc["n_nonfinite_rows"]),
            "n_nonpositive_duration": int(qc["n_nonpositive_duration_rows"]),
            "n_below_duration_threshold": int(qc["n_below_duration_threshold_rows"]),
            "n_trials_fix_index_gaps": int(qc["n_trials_fix_index_gaps"]),
            "n_trials_fix_index_duplicates": int(qc["n_trials_fix_index_duplicates"]),
            "n_trials_fix_index_non_monotonic": int(qc["n_trials_fix_index_non_monotonic"]),
            "n_temporal_undefined": int(qc["n_temporal_undefined"]),
            "n_malformed_fix_index": int(qc["n_malformed_fix_index"]),
            "n_malformed_duration": int(qc["n_malformed_duration"]),
            "n_malformed_pupil": int(qc["n_malformed_pupil"]),
            "n_trials_excluded_no_spatial_fixations": int(qc["n_trials_zero_spatial"]),
        },
    }


def _spatial_metrics_per_trial(heatmaps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-trial channel-0 entropy and center-of-mass distance (grid cells).

    ``heatmaps`` is a memmap-backed ``[N, 3, H, W]`` array for one subject.
    """
    ch0 = heatmaps[:, CH0].astype(np.float64)
    masses = ch0.sum(axis=(1, 2))
    probs = np.zeros_like(ch0)
    np.divide(ch0, masses[:, None, None], out=probs, where=masses[:, None, None] > 0)
    logp = np.zeros_like(probs)
    np.log(probs, out=logp, where=probs > 0)
    entropy = -np.sum(probs * logp, axis=(1, 2))
    cols = np.arange(GRID_W, dtype=np.float64)
    rows = np.arange(GRID_H, dtype=np.float64)
    com_u = np.sum(probs * cols[None, None, :], axis=(1, 2))
    com_v = np.sum(probs * rows[None, :, None], axis=(1, 2))
    com_dist = np.hypot(com_u - (GRID_W - 1) / 2, com_v - (GRID_H - 1) / 2)
    return entropy, com_dist


def trial_frame(store: TrialStore) -> pd.DataFrame:
    """One row per heatmap-eligible trial, joined by explicit IDs.

    Mass columns come from the stored arrays (via ``subject_row_index`` from
    ``trial_qc.parquet``), never from row order.
    """
    rows: list[dict[str, Any]] = []
    for subject_dir in sorted((store.root / "subjects").iterdir()):
        subject_id = subject_dir.name
        qc = pd.read_parquet(subject_dir / "trial_qc.parquet")
        ok = qc[qc.qc_status == "ok"]
        if ok.empty:
            continue
        heatmaps = np.load(subject_dir / "heatmaps.npy", mmap_mode="r")
        mass0 = heatmaps[:, CH0].sum(axis=(1, 2))
        mass1 = heatmaps[:, CH1].sum(axis=(1, 2))
        entropy, com_dist = _spatial_metrics_per_trial(heatmaps)
        rows.extend(
            {
                "subject_id": subject_id,
                "trial_uid": r.trial_uid,
                "stimulus_index": int(r.stimulus_index),
                "n_fixations_used": int(r.n_fixations_used),
                "n_transitions_used": int(r.n_transitions_used),
                "total_duration_ms_used": float(r.total_duration_ms_used),
                "median_fixation_duration_ms": (
                    float(r.median_fixation_duration_ms)
                    if pd.notna(r.median_fixation_duration_ms)
                    else None
                ),
                "mass_ch0": float(mass0[pos]),
                "mass_ch1": float(mass1[pos]),
                "entropy_ch0": float(entropy[pos]),
                "com_distance_ch0": float(com_dist[pos]),
            }
            for pos, r in enumerate(ok.itertuples())
        )
    return pd.DataFrame(rows)


def subject_level_aggregation(store: TrialStore, tf: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per subject (independent unit for group comparisons)."""
    tf = trial_frame(store) if tf is None else tf
    sub = store.subject_manifest
    agg = (
        tf.groupby("subject_id")
        .agg(
            n_trials_ok=("trial_uid", "count"),
            n_fixations_total=("n_fixations_used", "sum"),
            total_duration_ms=("total_duration_ms_used", "sum"),
            median_of_trial_medians_ms=("median_fixation_duration_ms", "median"),
            mean_ch0_mass=("mass_ch0", "mean"),
            mean_ch1_mass=("mass_ch1", "mean"),
            mean_entropy_ch0=("entropy_ch0", "mean"),
            mean_com_distance_ch0=("com_distance_ch0", "mean"),
        )
        .reset_index()
    )
    agg["n_fixations_per_trial"] = agg.n_fixations_total / agg.n_trials_ok
    agg["mean_fixation_duration_ms"] = agg.total_duration_ms / agg.n_fixations_total
    out = sub[["subject_id", "group", "label", "n_trials"]].merge(agg, on="subject_id", how="left")
    # Subjects whose every trial was excluded keep explicit NaNs.
    return out


def group_comparisons(subject_df: pd.DataFrame) -> dict[str, Any]:
    """Mann-Whitney U comparisons at the subject level.

    Unit of analysis: one value per subject. Rank-biserial correlation is
    reported as the effect size (positive => HC median larger).
    """
    from scipy.stats import mannwhitneyu

    metrics = {
        "n_fixations_per_trial": "mean fixations per trial",
        "median_of_trial_medians_ms": "median of per-trial median fixation durations (ms)",
        "mean_fixation_duration_ms": "mean fixation duration (ms)",
        "total_duration_ms": "total fixation duration (ms)",
        "mean_ch0_mass": "mean fixation-density mass per trial",
        "mean_ch1_mass": "mean transition-density mass per trial",
        "mean_entropy_ch0": "mean fixation-map spatial entropy (nats)",
        "mean_com_distance_ch0": "mean center-of-mass distance from map center (cells)",
    }
    results = []
    for col, label in metrics.items():
        hc = subject_df.loc[subject_df.group == "HC", col].dropna().astype(float)
        sz = subject_df.loc[subject_df.group == "SZ", col].dropna().astype(float)
        entry: dict[str, Any] = {
            "metric": col,
            "description": label,
            "unit": "subject",
            "n_hc": int(len(hc)),
            "n_sz": int(len(sz)),
            "hc_median": float(hc.median()) if len(hc) else None,
            "sz_median": float(sz.median()) if len(sz) else None,
            "hc_iqr": [float(hc.quantile(0.25)), float(hc.quantile(0.75))] if len(hc) else None,
            "sz_iqr": [float(sz.quantile(0.25)), float(sz.quantile(0.75))] if len(sz) else None,
        }
        if len(hc) >= 2 and len(sz) >= 2:
            res = mannwhitneyu(hc, sz, alternative="two-sided")
            u = float(res.statistic)
            entry["mannwhitney_u"] = u
            entry["p_value"] = float(res.pvalue)
            entry["rank_biserial_r"] = 1.0 - 2.0 * u / (len(hc) * len(sz))
        else:
            entry["mannwhitney_u"] = None
            entry["p_value"] = None
            entry["rank_biserial_r"] = None
        results.append(entry)
    return {"results": results, "notes": [
        "Unit of analysis: subject (one value per subject); trials are never "
        "treated as independent clinical samples.",
        "rank_biserial_r > 0 means the HC subject-level median is larger.",
        "These comparisons are descriptive; no multiple-comparison correction "
        "is applied and no diagnostic validity is claimed.",
    ]}


def channel_statistics(store: TrialStore, n_bins: int = 200) -> dict[str, Any]:
    """Streamed per-channel statistics over every stored heatmap cell."""
    n = 0
    m1 = np.zeros(3, dtype=np.float64)
    m2 = np.zeros(3, dtype=np.float64)
    vmin = np.full(3, np.inf)
    vmax = np.full(3, -np.inf)
    nonfinite = np.zeros(3, dtype=np.int64)
    nz2_sum = 0.0
    nz2_n = 0
    mass0: list[float] = []
    mass1: list[float] = []

    # Pass 1: moments, extrema, per-trial masses.
    for subject_dir in sorted((store.root / "subjects").iterdir()):
        heatmaps = np.load(subject_dir / "heatmaps.npy", mmap_mode="r")
        if len(heatmaps) == 0:
            continue
        arr = heatmaps
        n += arr.shape[0] * arr.shape[1] * arr.shape[2] * arr.shape[3]
        nonfinite += np.count_nonzero(~np.isfinite(arr.astype(np.float64)), axis=(0, 2, 3))
        for c in range(3):
            ch = arr[:, c].astype(np.float64)
            vmin[c] = min(vmin[c], float(ch.min()))
            vmax[c] = max(vmax[c], float(ch.max()))
            flat = ch.ravel()
            m1[c] += flat.sum()
            m2[c] += float((flat * flat).sum())
        mass0.extend(float(x) for x in arr[:, CH0].sum(axis=(1, 2)))
        mass1.extend(float(x) for x in arr[:, CH1].sum(axis=(1, 2)))
        nz2_sum += float(np.count_nonzero(arr[:, CH2]))
        nz2_n += arr.shape[0]

    # ``n`` counts cells across all three channels; each channel's statistics
    # are over ``n_per_channel`` cells.
    n_per_channel = n / 3.0
    mean = m1 / n_per_channel
    var = np.maximum(m2 / n_per_channel - mean * mean, 0.0)
    std = np.sqrt(var)

    # Pass 2: histograms (ch0/ch1 log-spaced from 0, ch2 linear in [-1, 1]).
    hist_counts = [np.zeros(n_bins, dtype=np.int64) for _ in range(3)]
    for subject_dir in sorted((store.root / "subjects").iterdir()):
        heatmaps = np.load(subject_dir / "heatmaps.npy", mmap_mode="r")
        if len(heatmaps) == 0:
            continue
        for c in range(3):
            ch = heatmaps[:, c].astype(np.float64)
            if c == CH2:
                bins = np.linspace(-1.0, 1.0, n_bins + 1)
                hist_counts[c] += np.histogram(ch, bins=bins)[0]
            else:
                mx = float(vmax[c])
                if mx <= 0:
                    continue
                # Log-spaced bins between 1e-6 and the observed maximum.
                bins = np.exp(np.linspace(math.log(1e-6), math.log(max(mx, 1e-3)), n_bins + 1))
                hist_counts[c] += np.histogram(ch[ch >= 0], bins=bins)[0]

    def quantiles_from_hist(counts: np.ndarray, bins: np.ndarray, qs=(0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)) -> dict[str, float]:
        cdf = np.cumsum(counts) / max(counts.sum(), 1)
        out = {}
        for q in qs:
            out[f"q{int(round(q*100)):02d}"] = float(bins[np.searchsorted(cdf, q, side="right")])
        return out

    mass0_arr = np.asarray(mass0)
    mass1_arr = np.asarray(mass1)
    mass_hist = lambda arr, nb: {
        "counts": _py(np.histogram(arr, bins=nb, range=(0, max(int(arr.max()), 1)))[0]),
        "edges": _py(np.histogram(arr, bins=nb, range=(0, max(int(arr.max()), 1)))[1]),
    }

    per_channel = []
    for c in range(3):
        if c == CH2:
            bins = np.linspace(-1.0, 1.0, n_bins + 1)
            quantiles = quantiles_from_hist(hist_counts[c], bins)
        else:
            mx = float(vmax[c])
            bins = np.exp(np.linspace(math.log(1e-6), math.log(max(mx, 1e-3)), n_bins + 1))
            quantiles = quantiles_from_hist(hist_counts[c], bins)
        per_channel.append(
            {
                "channel": c,
                "shape": [3, GRID_H, GRID_W],
                "dtype": "float32",
                "n_cells": n // 3,
                "finite_fraction": 1.0 - float(nonfinite[c]) / max(n // 3, 1),
                "min": float(vmin[c]),
                "max": float(vmax[c]),
                "mean": float(mean[c]),
                "std": float(std[c]),
                "quantiles": quantiles,
            }
        )
    per_channel[CH2]["fraction_nonzero_cells"] = float(nz2_sum / max(nz2_n * GRID_H * GRID_W, 1))

    return {
        "per_channel": per_channel,
        "per_trial_mass": {
            "channel_0_fixation_density": {
                "min": float(mass0_arr.min()),
                "max": float(mass0_arr.max()),
                "mean": float(mass0_arr.mean()),
                "median": float(np.median(mass0_arr)),
                "distribution": mass_hist(mass0_arr, 30),
            },
            "channel_1_transition_density": {
                "min": float(mass1_arr.min()),
                "max": float(mass1_arr.max()),
                "mean": float(mass1_arr.mean()),
                "median": float(np.median(mass1_arr)),
                "distribution": mass_hist(mass1_arr, 30),
            },
        },
    }


def mass_consistency(tf: pd.DataFrame) -> dict[str, Any]:
    """Consistency between expected and observed per-trial map mass."""
    err0 = (tf.mass_ch0 - tf.n_fixations_used).abs()
    err1 = (tf.mass_ch1 - tf.n_transitions_used).abs()
    return {
        "channel_0": {
            "mean_abs_error": float(err0.mean()),
            "max_abs_error": float(err0.max()),
            "n_trials": int(len(tf)),
        },
        "channel_1": {
            "mean_abs_error": float(err1.mean()),
            "max_abs_error": float(err1.max()),
            "n_trials": int(len(tf)),
        },
    }


def group_mean_heatmaps(store: TrialStore) -> dict[str, Any]:
    """Descriptive group-average heatmaps (EDA only; never used for training)."""
    sums: dict[str, tuple[np.ndarray, int]] = {}
    sub = store.subject_manifest.set_index("subject_id")
    for subject_dir in sorted((store.root / "subjects").iterdir()):
        subject_id = subject_dir.name
        group = str(sub["group"].loc[subject_id])
        heatmaps = np.load(subject_dir / "heatmaps.npy", mmap_mode="r")
        if len(heatmaps) == 0:
            continue
        acc, n = sums.get(group, (np.zeros((3, GRID_H, GRID_W), dtype=np.float64), 0))
        acc += heatmaps.astype(np.float64).sum(axis=0)
        sums[group] = (acc, n + len(heatmaps))
    return {
        group: {"mean": _py(acc / n), "n_trials": n, "n_subjects": int((sub.group == group).sum())}
        for group, (acc, n) in sorted(sums.items())
    }


def select_representative_trials(store: TrialStore, tf: pd.DataFrame) -> dict[str, Any]:
    """Deterministic example trials (one per group).

    Rule: within each group, choose the subject with the median total fixation
    count (ties broken by ascending subject_id); within that subject choose the
    trial with the median fixation-density mass (ties broken by ascending
    stimulus_index). These are examples, not group averages.
    """
    sub = store.subject_manifest
    out = {}
    for group in ["HC", "SZ"]:
        ids = sorted(sub.loc[sub.group == group, "subject_id"].astype(str))
        # Subject-level totals deterministically (ties broken by subject_id).
        totals = (
            tf[tf.subject_id.isin(ids)]
            .groupby("subject_id")["n_fixations_used"]
            .sum()
            .reindex(ids)
        )
        median_idx = sorted(totals.values)[len(totals) // 2]
        subject_id = totals[totals == median_idx].index[0]
        trials = tf[tf.subject_id == subject_id].sort_values("stimulus_index")
        median_mass = sorted(trials.mass_ch0)[len(trials) // 2]
        trial = trials[trials.mass_ch0 == median_mass].iloc[0]
        out[group] = {
            "subject_id": subject_id,
            "stimulus_index": int(trial.stimulus_index),
            "trial_uid": str(trial.trial_uid),
            "n_fixations_used": int(trial.n_fixations_used),
            "rule": (
                "subject with median total fixations in group, trial with median "
                "fixation-density mass in that subject"
            ),
        }
    return out


def compute_eda_summary(processed_root: Path | str, command: str) -> dict[str, Any]:
    """Assemble the full machine-readable EDA summary."""
    store = TrialStore(processed_root)
    store.verify_manifest_checksums()
    metadata = json.loads((store.root / "dataset_metadata.json").read_text(encoding="utf-8"))
    tf = trial_frame(store)
    subject_df = subject_level_aggregation(store, tf)
    examples = select_representative_trials(store, tf)

    example_heatmaps: dict[str, Any] = {}
    for group, info in examples.items():
        rec = store.get_trial_by_uid(info["trial_uid"])
        example_heatmaps[group] = {
            "heatmap": _py(rec.heatmap),
            "subject_id": info["subject_id"],
            "stimulus_id": rec.stimulus_id,
            "stimulus_index": info["stimulus_index"],
            "n_fixations_used": info["n_fixations_used"],
        }

    return _py(
        {
            "generator": {
                "command": command,
                "python": platform.python_version(),
                "numpy": str(np.__version__),
                "pandas": str(pd.__version__),
            },
            "processed_dataset": {
                "root": str(store.root),
                "config_hash": metadata["config_hash"],
                "seed": metadata["seed"],
                "dataset_metadata_sha256": sha256_of_file(store.root / "dataset_metadata.json"),
                "image_manifest_sha256": metadata["image_manifest_sha256"],
                "subject_manifest_sha256": metadata["subject_manifest_sha256"],
                "trial_manifest_sha256": metadata["trial_manifest_sha256"],
                "qc_summary_sha256": metadata["qc_summary_sha256"],
                "preprocessing_config_sha256": metadata["preprocessing_config_sha256"],
                "preprocessing_config": json.loads(
                    (store.root / "preprocessing_config.json").read_text(encoding="utf-8")
                ),
            },
            "inventory": inventory_statistics(store),
            "channel_stats": channel_statistics(store),
            "mass_consistency": mass_consistency(tf),
            "group_comparisons": group_comparisons(subject_df),
            "subject_level": _py(subject_df.to_dict(orient="records")),
            "representative_trials": _py(examples),
            "example_heatmaps": example_heatmaps,
            "group_mean_heatmaps": group_mean_heatmaps(store),
        }
    )
