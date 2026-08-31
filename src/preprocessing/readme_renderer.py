"""Render the preprocessed-dataset README and EDA figures from the summary.

Everything rendered here is derived from ``eda_summary.json`` data, so the
output is deterministic given the same summary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Palette (validated default, light surface) - see docs in the dataviz skill.
C_HC = "#2a78d6"  # categorical slot 1 (blue)
C_SZ = "#eb6834"  # categorical slot 2 (orange)
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
             "#0d366b"]
RED_RAMP = ["#f7cdcb", "#f3a9a4", "#ee817b", "#e34948", "#c03a3a", "#a32424"]
CMAP_BLUE_SEQ = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
CMAP_DIVERG = LinearSegmentedColormap.from_list(
    "temporal", ["#104281", "#2a78d6", "#9ec5f4", "#f0efec", "#f7cdcb", "#e34948", "#a32424"]
)

FIG_FILES = {
    "subject_trial_counts": "figures/fig_subject_trial_counts.png",
    "trials_per_subject": "figures/fig_trials_per_subject.png",
    "subject_fixations_by_group": "figures/fig_subject_fixations_by_group.png",
    "qc_events": "figures/fig_qc_events.png",
    "example_trials": "figures/fig_example_trials.png",
    "group_average_heatmaps": "figures/fig_group_average_heatmaps.png",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def _imshow_channel(ax, data: np.ndarray, channel: int, title: str) -> None:
    if channel in (0, 1):
        cmap, vmin, vmax = CMAP_BLUE_SEQ, 0.0, float(np.max(data))
        label = "density (log1p shown)" if channel == 0 else "transition density"
        shown = np.log1p(np.maximum(data, 0))
        im = ax.imshow(shown, cmap=cmap, vmin=0.0, vmax=np.log1p(vmax), origin="lower")
    else:
        cmap, vmin, vmax = CMAP_DIVERG, -1.0, 1.0
        label = "temporal progression τ"
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(label, fontsize=7)
    cbar.ax.tick_params(labelsize=6, colors=MUTED)


def render_figures(summary: dict[str, Any], output_dir: Path) -> list[str]:
    """Create all figure PNGs; return their relative paths."""
    _style()
    out = Path(output_dir)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    inv = summary["inventory"]

    # 1. Subject and trial counts by group.
    import pandas as pd

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    metrics = ["Subjects", "Observed trials"]
    sl0 = pd.DataFrame(summary["subject_level"])
    trials_by_group = {
        g: int(sl0.loc[sl0.group == g, "n_trials"].sum()) for g in ["HC", "SZ"]
    }
    hc_vals = [inv["subjects_by_group"]["HC"], trials_by_group["HC"]]
    sz_vals = [inv["subjects_by_group"]["SZ"], trials_by_group["SZ"]]
    x = np.arange(len(metrics))
    w = 0.38
    b1 = ax.bar(x - w / 2, hc_vals, w, color=C_HC, label="HC")
    b2 = ax.bar(x + w / 2, sz_vals, w, color=C_SZ, label="SZ")
    for bars in (b1, b2):
        for b in bars:
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height(),
                f"{int(b.get_height()):,}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=INK2,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("count")
    ax.set_title("Subjects and observed trials by diagnostic group")
    ax.legend()
    ax.grid(axis="y")
    ax.set_ylim(0, max(hc_vals + sz_vals) * 1.12)
    fig.tight_layout()
    fig.savefig(figs / Path(FIG_FILES["subject_trial_counts"]).name)
    plt.close(fig)
    written.append(FIG_FILES["subject_trial_counts"])

    # 2. Distribution of observed trials per subject.
    d = inv["trials_per_subject"]["distribution"]
    centers = (np.asarray(d["edges"][:-1]) + np.asarray(d["edges"][1:])) / 2
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    ax.bar(centers, d["counts"], width=np.diff(d["edges"]) * 0.9, color="#5598e7")
    ax.set_xlabel("observed trials per subject")
    ax.set_ylabel("number of subjects")
    ax.set_title("Distribution of observed trials per subject (n = 160)")
    ax.grid(axis="y")
    fig.tight_layout()
    fig.savefig(figs / Path(FIG_FILES["trials_per_subject"]).name)
    plt.close(fig)
    written.append(FIG_FILES["trials_per_subject"])

    # 3. Subject-level fixation count and median duration by group.
    import pandas as pd

    sl = pd.DataFrame(summary["subject_level"])
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for ax, col, title, unit in [
        (axes[0], "n_fixations_per_trial", "Mean fixations per trial", "fixations"),
        (axes[1], "median_of_trial_medians_ms", "Median of per-trial median\nfixation durations", "ms"),
    ]:
        data = [sl.loc[sl.group == g, col].dropna().values for g in ["HC", "SZ"]]
        bp = ax.boxplot(
            data, positions=[0, 1], widths=0.4, patch_artist=True,
            showfliers=False, medianprops=dict(color=INK, linewidth=1.5),
            boxprops=dict(linewidth=1.0), whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
        )
        for patch, color in zip(bp["boxes"], [C_HC, C_SZ]):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
        for pos, vals, color in zip([0, 1], data, [C_HC, C_SZ]):
            # Deterministic horizontal jitter (index-based, no RNG).
            jitter = (np.arange(len(vals)) - len(vals) / 2.0) / max(len(vals) * 8, 1)
            ax.plot(pos + jitter, vals, "o", markersize=3, color=color, alpha=0.55)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["HC", "SZ"])
        ax.set_ylabel(unit)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y")
    fig.suptitle("Subject-level fixation statistics by group (one point per subject)", fontsize=10)
    fig.tight_layout()
    fig.savefig(figs / Path(FIG_FILES["subject_fixations_by_group"]).name)
    plt.close(fig)
    written.append(FIG_FILES["subject_fixations_by_group"])

    # 4. QC event counts.
    qc = inv["qc_event_counts"]
    labels = {
        "n_off_canvas": "off-canvas fixations",
        "n_nonpositive_duration": "nonpositive-duration fixations",
        "n_below_duration_threshold": "fixations below duration threshold",
        "n_trials_excluded_no_spatial_fixations": "trials with no usable spatial fixation",
        "n_trials_fix_index_gaps": "trials with FIX_INDEX gaps",
        "n_trials_fix_index_duplicates": "trials with duplicate FIX_INDEX",
        "n_trials_fix_index_non_monotonic": "trials with non-monotonic FIX_INDEX order",
        "n_temporal_undefined": "trials with undefined temporal clock",
        "n_malformed_duration": "malformed duration cells",
        "n_malformed_fix_index": "malformed FIX_INDEX cells",
        "n_malformed_pupil": "malformed pupil cells",
        "n_nonfinite": "non-finite coordinate cells",
    }
    nonzero = [(labels[k], qc[k]) for k in labels if qc.get(k, 0) > 0]
    nonzero.sort(key=lambda kv: kv[1])
    names = [n for n, _ in nonzero]
    vals = [v for _, v in nonzero]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.barh(np.arange(len(vals)), vals, color="#5598e7")
    ax.set_yticks(np.arange(len(vals)))
    ax.set_yticklabels(names, fontsize=8)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:,}", va="center", fontsize=8, color=INK2)
    ax.set_xlabel("count")
    ax.set_title("QC event counts (log-free, nonzero counts only)")
    ax.grid(axis="x")
    fig.tight_layout()
    fig.savefig(figs / Path(FIG_FILES["qc_events"]).name)
    plt.close(fig)
    written.append(FIG_FILES["qc_events"])

    # 5. Representative trial visualizations (one per group).
    ch_titles = {0: "fixation density", 1: "transition density", 2: "temporal progression"}
    fig, axes = plt.subplots(2, 3, figsize=(9.4, 5.6))
    for r, group in enumerate(["HC", "SZ"]):
        hm = np.asarray(summary["example_heatmaps"][group]["heatmap"], dtype=np.float64)
        sid = summary["example_heatmaps"][group]["subject_id"]
        nfix = summary["example_heatmaps"][group]["n_fixations_used"]
        for c in range(3):
            _imshow_channel(
                axes[r, c], hm[c], c,
                f"{group} subject {sid} — {ch_titles[c]}\n({nfix} fixations, example, not a group average)",
            )
    fig.suptitle("Representative three-channel trials (deterministic examples)", fontsize=11)
    fig.tight_layout()
    fig.savefig(figs / Path(FIG_FILES["example_trials"]).name)
    plt.close(fig)
    written.append(FIG_FILES["example_trials"])

    # 6. Group-average heatmaps (descriptive only).
    fig, axes = plt.subplots(2, 3, figsize=(9.4, 5.6))
    for r, group in enumerate(["HC", "SZ"]):
        gm = np.asarray(summary["group_mean_heatmaps"][group]["mean"], dtype=np.float64)
        n_trials = summary["group_mean_heatmaps"][group]["n_trials"]
        for c in range(3):
            _imshow_channel(
                axes[r, c], gm[c], c,
                f"{group} group mean — {ch_titles[c]}\n(n = {n_trials} trials; descriptive EDA only)",
            )
    fig.suptitle("Group-average heatmaps (descriptive artifact — never used for training)", fontsize=11)
    fig.tight_layout()
    fig.savefig(figs / Path(FIG_FILES["group_average_heatmaps"]).name)
    plt.close(fig)
    written.append(FIG_FILES["group_average_heatmaps"])

    return written


def _fmt(value: Any, nd: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{nd}f}"
    return str(value)


def _fmt_p(value: Any) -> str:
    if value is None:
        return "n/a"
    if value < 0.0001:
        return f"{value:.2e}"
    return f"{value:.4f}"


def render_readme(summary: dict[str, Any]) -> str:
    """Render the processed-dataset README from the EDA summary."""
    inv = summary["inventory"]
    ch = summary["channel_stats"]
    mc = summary["mass_consistency"]
    gc = summary["group_comparisons"]
    cfg = summary["processed_dataset"]["preprocessing_config"]
    ds = summary["processed_dataset"]

    def ch_row(c: int) -> str:
        d = ch["per_channel"][c]
        return (
            f"| {c} | float32 | [3, 48, 64] | {_fmt(d['min'])} | {_fmt(d['max'])} | "
            f"{_fmt(d['mean'], 4)} | {_fmt(d['std'], 4)} | "
            f"q01 {_fmt(d['quantiles']['q01'], 4)} / q50 {_fmt(d['quantiles']['q50'], 4)} / "
            f"q99 {_fmt(d['quantiles']['q99'], 4)} | {_fmt(d['finite_fraction'], 6)} |"
        )

    gcmp = "\n".join(
        f"- **{r['description']}** (unit: subject): HC n = {r['n_hc']}, median = "
        f"{_fmt(r['hc_median'], 2)}; SZ n = {r['n_sz']}, median = {_fmt(r['sz_median'], 2)}; "
        f"Mann–Whitney U = {_fmt(r['mannwhitney_u'], 1)}, p = {_fmt_p(r['p_value'])}, "
        f"rank-biserial r = {_fmt(r['rank_biserial_r'], 3)} "
        "(r > 0 ⇒ HC subject-level median larger)"
        for r in gc["results"]
    )
    gnotes = "\n".join(f"- {n}" for n in gc["notes"])

    missing = "\n".join(
        f"- `{m['subject_id']}`: {m['n_trials']} trials observed "
        f"({m['n_missing_expected_stimuli']} of 100 stimuli missing)"
        for m in inv["subjects_missing_stimuli"]
    )

    figs = "\n".join(
        f"![{p}]({p})\n\n*{cap}*"
        for p, cap in [
            (FIG_FILES["subject_trial_counts"], "Figure 1 — Subjects and observed trials by diagnostic group."),
            (FIG_FILES["trials_per_subject"], "Figure 2 — Distribution of observed trials per subject."),
            (FIG_FILES["subject_fixations_by_group"], "Figure 3 — Subject-level fixation count and median duration by group (one point per subject)."),
            (FIG_FILES["qc_events"], "Figure 4 — QC event counts."),
            (FIG_FILES["example_trials"], "Figure 5 — Deterministically selected example trials (one HC, one SZ); examples, not group averages."),
            (FIG_FILES["group_average_heatmaps"], "Figure 6 — Group-average heatmaps; descriptive EDA artifact, never used for model training."),
        ]
    )

    example_cmd = (
        "```python\n"
        "import sys\n"
        "sys.path.insert(0, \"/root/EMS-Project/src\")\n"
        "from preprocessing.storage import TrialStore\n"
        "\n"
        "store = TrialStore(\"/root/EMS-Project/processed_dataset\")\n"
        "record = store.get_trial(subject_id=\"000\", stimulus_id=\"outman_054.jpg\")\n"
        "print(record.heatmap.shape)   # (3, 48, 64) float32\n"
        "print(record.group)           # 'HC'\n"
        "```"
    )

    return f"""# EMS processed dataset — preprocessing, schema, and EDA

Generated reproducibly from the canonical processed artifacts by:

```bash
{summary['generator']['command']}
```

Python {summary['generator']['python']}, NumPy {summary['generator']['numpy']}, pandas {summary['generator']['pandas']}.

## 1. Purpose and provenance

The processed dataset under `/root/EMS-Project/processed_dataset` is a
**deterministic cache** for later modeling stages (DINO features, five-fold
subject CV, Stage-1 training). It is built from raw EMS fixation-event
workbooks under `original_dataset/EMS/All_Data/Fixations` and stimulus images
under `original_dataset/EMS/Images`; **the original dataset is never
modified**.

The cache contains **no fitted model, no CV statistics, no normative bank,
and no population normalization**. Preprocessing operates on individual
trials only.

Generating configuration:

- preprocessing config SHA-256: `{ds['preprocessing_config_sha256']}`
- resolved config hash: `{ds['config_hash']}` (seed {ds['seed']})
- source inventory / dataset metadata checksums are recorded in
  `source_inventory.json` and `dataset_metadata.json`

## 2. Directory and file schema

```text
processed_dataset/
├── dataset_metadata.json        # completion record (written last; readers require it)
├── preprocessing_config.json    # fully resolved configuration
├── source_inventory.json        # image/subject SHA-256 inventory + anomaly measurements
├── image_manifest.csv           # stimulus_index, stimulus_id, source_image_name, category,
│                                #   relative_image_path, width, height, sha256
├── subject_manifest.csv         # subject_id, subject_numeric_id, group, label, source_workbook,
│                                #   n_fixation_rows, n_trials, n_missing_expected_stimuli, source_sha256
├── trial_manifest.parquet       # trial_uid, subject_id, stimulus_id, subject_numeric_id,
│                                #   stimulus_index, group, label, category, subject_array_path,
│                                #   subject_row_index, n_fixations_raw, n_fixations_used,
│                                #   n_transitions_used, total_duration_ms_raw, total_duration_ms_used,
│                                #   n_off_canvas, n_nonfinite, n_nonpositive_duration,
│                                #   n_below_duration_threshold, fix_index_has_gap, qc_status
├── trial_manifest.csv           # optional human-readable export
├── qc_summary.json              # global QC counts + excluded-trial list
└── subjects/
    └── <subject_id>/            # canonical string ID with leading zeros ("000", "203")
        ├── heatmaps.npy         # float32 [n_subject_trials, 3, 48, 64]
        ├── stimulus_indices.npy # int64 [n_subject_trials], ascending stimulus_index order
        ├── trial_qc.parquet     # per-trial QC incl. mass and median-duration columns
        └── artifact_meta.json   # config hash, source checksum, array checksums
```

Identifier rules: `subject_id` is the workbook stem with leading zeros
preserved and is never an array offset; `stimulus_index` is a contiguous
integer assigned only through `image_manifest.csv` (ordered by category, then
basename); the trial key is `(subject_id, stimulus_id)` with a SHA-256-derived
`trial_uid`. The pair `(subject_array_path, subject_row_index)` retrieves the
exact array row for a trial.

Minimal retrieval example:

{example_cmd}

## 3. Exact channel definitions

Heatmaps are `float32 [3, 48, 64]` with this immutable channel order:

| channel | content | stored range |
|---|---|---|
| 0 | fixation density | ≥ 0, mass ≈ number of used fixations |
| 1 | consecutive-fixation transition density | ≥ 0, mass ≈ number of used transitions |
| 2 | temporal progression τ | approximately [−1, 1] |

Coordinate transform for an accepted fixation `(x, y)` on the
{cfg['source_width']} × {cfg['source_height']} canvas:

```text
u = x * (W − 1) / (1024 − 1)      v = y * (H − 1) / (768 − 1)      W = 64, H = 48
```

Event filtering policy: off-canvas fixations are excluded from spatial maps
(policy `{cfg['off_canvas_policy']}`) but keep their QC counts; fixations with
nonpositive duration are dropped (`min_fix_duration_ms={cfg['min_fix_duration_ms']}`,
`drop_nonpositive_duration={cfg['drop_nonpositive_duration']}`); trials with no
usable spatial fixation are recorded with
`qc_status = "excluded_no_spatial_fixations"` and have **no heatmap row**.

Channel 0: H_F = Σᵢ Gᵢ, one unit-mass truncated Gaussian per used fixation
(σ = {cfg['gaussian_sigma_cells']} cells, truncation {cfg['gaussian_truncate_sigma']}σ,
border-truncated kernels renormalized to unit sum). Mass is deliberately **not**
normalized to one or to any group statistic.

Channel 1: for every consecutive pair in original `FIX_INDEX` order with both
endpoints spatially valid, the segment is rasterized with sub-cell sampling
(max spacing {cfg['transition_sample_step_cells']} cells), normalized to unit
mass, and Gaussian-smoothed. Rejected intermediate fixations are never
bridged; zero-length transitions deposit unit mass at their endpoint.

Channel 2: with no timestamps available, elapsed time is reconstructed from
fixation durations. For duration-valid events in original order (off-canvas
events advance the clock but deposit nothing):

```text
tᵢᵐⁱᵈ = (Σ_{{k<i}} d_k + dᵢ/2) / Σ_k d_k        τᵢ = 2 tᵢᵐⁱᵈ − 1
H_P(u,v) = Σᵢ τᵢ Gᵢ(u,v) / (H_F(u,v) + ε)
```

Unvisited locations are exactly zero. See
`src/preprocessing/heatmaps.py` for the implementation.

## 4. Population and trial inventory

Measured from the processed artifacts:

- **Subjects**: {inv['n_subjects']} ({', '.join(f"{g}: {n}" for g, n in inv['subjects_by_group'].items())})
- **Stimuli**: {sum(inv['stimuli_by_category'].values())} images — {', '.join(f"{c}: {n}" for c, n in inv['stimuli_by_category'].items())}
- **Fixation rows**: {inv['n_fixation_rows']:,}
- **Observed trials**: {inv['n_trials_observed']:,} ({inv['n_trials_ok']:,} heatmap-eligible; {inv['n_trials_excluded']} excluded)
- **Trials per subject**: min {inv['trials_per_subject']['min']}, max {inv['trials_per_subject']['max']}, mean {_fmt(inv['trials_per_subject']['mean'], 1)}
- **Fixations per trial**: min {inv['fixations_per_trial']['min']}, median {_fmt(inv['fixations_per_trial']['median'], 1)}, max {inv['fixations_per_trial']['max']}
- **Total positive fixation duration per trial**: median {_fmt(inv['total_duration_per_trial_ms']['median'], 0)} ms, mean {_fmt(inv['total_duration_per_trial_ms']['mean'], 0)} ms
- **Subjects with missing stimuli** ({len(inv['subjects_missing_stimuli'])}):
{missing}
- **QC event counts**: off-canvas rows {inv['qc_event_counts']['n_off_canvas']:,};
  nonpositive durations {inv['qc_event_counts']['n_nonpositive_duration']:,};
  below-threshold rows {inv['qc_event_counts']['n_below_duration_threshold']:,};
  non-finite/malformed cells {inv['qc_event_counts']['n_nonfinite']:,};
  FIX_INDEX gaps/duplicates/non-monotonic {inv['qc_event_counts']['n_trials_fix_index_gaps']}/{inv['qc_event_counts']['n_trials_fix_index_duplicates']}/{inv['qc_event_counts']['n_trials_fix_index_non_monotonic']}
- **Excluded or warning-status trials**: {inv['n_excluded_or_warning_trials']}
  (zero usable spatial fixations after the approved policy)

## 5. Heatmap-channel statistics

Computed over every stored heatmap cell (float32, `[3, 48, 64]` per trial):

| channel | dtype | shape | min | max | mean | std | quantiles | finite |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(ch_row(c) for c in (0, 1, 2))}

Per-trial total mass:

- Channel 0 (fixation density): min {_fmt(ch['per_trial_mass']['channel_0_fixation_density']['min'], 1)}, median {_fmt(ch['per_trial_mass']['channel_0_fixation_density']['median'], 1)}, max {_fmt(ch['per_trial_mass']['channel_0_fixation_density']['max'], 1)}
- Channel 1 (transition density): min {_fmt(ch['per_trial_mass']['channel_1_transition_density']['min'], 1)}, median {_fmt(ch['per_trial_mass']['channel_1_transition_density']['median'], 1)}, max {_fmt(ch['per_trial_mass']['channel_1_transition_density']['max'], 1)}
- Channel 2: fraction of nonzero cells {_fmt(ch['per_channel'][2]['fraction_nonzero_cells'], 4)}; range {_fmt(ch['per_channel'][2]['min'], 4)}…{_fmt(ch['per_channel'][2]['max'], 4)}

Mass-consistency (expected vs observed):

- Channel 0: mean |H_F − n_fixations_used| = {mc['channel_0']['mean_abs_error']:.2e}, max = {mc['channel_0']['max_abs_error']:.2e} over {mc['channel_0']['n_trials']:,} trials
- Channel 1: mean |H_T − n_transitions_used| = {mc['channel_1']['mean_abs_error']:.2e}, max = {mc['channel_1']['max_abs_error']:.2e} over {mc['channel_1']['n_trials']:,} trials

## 6. Basic HC/SZ EDA (descriptive)

Every comparison is aggregated to **one value per subject** first; trials are
never treated as independent clinical samples.

{gcmp}

{gnotes}

## 7. Caveats

- Subject and stimulus identifiers are non-contiguous; array positions must
  be resolved through manifests, never by ID magnitude.
- {inv['n_trials_observed']:,} observed trials instead of a complete
  160 × 100 rectangle; missing trials remain missing everywhere downstream.
- No timestamps or raw gaze samples exist; fixation order and duration are the
  only temporal information, so the temporal channel is a reconstructed clock.
- Viewing duration is right-censored by trial end; total durations are
  observational aggregates, not exposure times.
- Pupil units are unknown and unvalidated (`FIX_PUPIL` is retained in QC only).
- Off-canvas and short-duration events are handled by the approved policies
  and counted in QC; the kernel σ = {cfg['gaussian_sigma_cells']} and the
  duration threshold are ablation candidates.
- The five excluded trials, the two substantially incomplete subjects
  (`216`, `259`), and the descriptive group differences above do **not**
  establish diagnostic validity. Stage-1 modeling is HC-only and does not
  consume the SZ comparisons shown here.

## Figures

{figs}
"""


def write_artifacts(summary: dict[str, Any], output_dir: Path) -> tuple[Path, Path, list[str]]:
    """Write README.md, figures, and eda_summary.json; return their paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    figure_paths = render_figures(summary, out)
    readme = render_readme(summary)
    (out / "README.md").write_text(readme, encoding="utf-8")
    (out / "eda_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out / "README.md", out / "eda_summary.json", figure_paths
