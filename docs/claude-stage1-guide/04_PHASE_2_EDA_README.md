# Phase 2 — Processed-dataset EDA and README

## Goal

Generate reproducible basic EDA from the canonical processed artifacts and
write:

```text
/root/EMS-Project/readme/preprocessed/README.md
```

The README must reflect measured results from the processed dataset. Do not copy
expected numbers into result tables without recomputing them.

## 1. Suggested files

```text
generate_preprocessed_readme.py
src/preprocessing/eda.py
src/preprocessing/readme_renderer.py
tests/preprocessing/test_eda_aggregation.py
tests/preprocessing/test_preprocessed_readme.py
readme/preprocessed/README.md
readme/preprocessed/figures/*.png
readme/preprocessed/eda_summary.json
```

The JSON summary is the machine-readable source for rendered tables. The
Markdown file should be regenerable from code and should state the command used.

## 2. Required README sections

### 2.1 Dataset purpose and provenance

Explain:

- the processed dataset is a deterministic cache for later modeling;
- the raw source is fixation-event EMS data;
- the processed cache contains no fitted model, no CV statistics, no normative
  bank, and no population normalization;
- the original dataset remains unchanged;
- which resolved preprocessing configuration and source-inventory checksum
  generated this version.

### 2.2 Directory and file schema

Document every canonical file, array shape, dtype, manifest field, and lookup
path. Include a minimal Python example retrieving a trial by explicit
`subject_id` and `stimulus_id`.

### 2.3 Exact channel definitions

Document the three channels, coordinate transform, Gaussian sigma, event
filtering policy, mass-preservation behavior, and temporal range. Include the
equations or link to code functions by repository-relative path.

### 2.4 Population and trial inventory

Report measured:

- subjects by HC/SZ group;
- stimuli by category;
- fixation rows and observed trials;
- trials per subject distribution;
- subjects with missing stimuli;
- fixations per trial distribution;
- total positive fixation duration per trial;
- off-canvas, non-finite, nonpositive-duration, short-duration, duplicate-index,
  and index-gap counts;
- number of excluded or warning-status trials.

### 2.5 Heatmap-channel statistics

For each channel report at least:

- shape and dtype;
- finite fraction;
- minimum, maximum, mean, standard deviation, and selected quantiles;
- per-trial total mass distribution for fixation and transition density;
- temporal-progression range and fraction of nonzero cells;
- consistency errors between expected and observed mass.

### 2.6 Basic HC/SZ EDA

Provide descriptive comparisons for:

- number of fixations per trial;
- median fixation duration;
- total fixation duration;
- fixation- and transition-map mass;
- heatmap spatial center of mass or entropy, if implemented correctly.

Avoid pseudoreplication. Inferential comparisons must first aggregate to one
value per subject. If Mann–Whitney U or an effect size is reported, clearly state
the subject-level unit and provide group sample sizes. Do not treat thousands of
trials as independent clinical samples.

### 2.7 Caveats

Document:

- non-contiguous identifiers;
- missing subject-stimulus trials;
- no timestamps or raw gaze samples;
- right-censored viewing duration when observed;
- unknown or risky pupil units;
- off-canvas and short events;
- preprocessing choices that require ablation;
- no claim that descriptive HC/SZ differences establish diagnostic validity.

## 3. Required figures

Create compact, publication-readable PNGs with titles, units, legends, and
captions:

1. subject and trial counts by group;
2. distribution of observed trials per subject;
3. subject-level fixation count and median duration by group;
4. QC event counts;
5. one representative three-channel trial visualization for HC and one for SZ,
   selected deterministically and labeled as examples, not group averages;
6. optional group-average heatmaps computed only for EDA and clearly marked as
   descriptive artifacts that are never used for model training.

Do not overwrite source stimuli or render identifiable participant information.

## 4. Reproducible command

Provide:

```bash
python generate_preprocessed_readme.py \
  --processed-root /root/EMS-Project/processed_dataset \
  --output-dir /root/EMS-Project/readme/preprocessed
```

Support `--verify-only`, which checks whether README metadata and summary hashes
match the current processed dataset.

## 5. Required tests

- EDA joins use explicit IDs and do not assume array order.
- Subject-level aggregation returns one row per subject.
- Missing trials are counted, not imputed.
- Heatmap stats are streamed or chunked rather than loading unnecessary copies.
- A small synthetic processed dataset renders a README containing all required
  headings and valid figure links.
- `--verify-only` detects a modified manifest or preprocessing config.

## 6. Acceptance criteria

- README and figures are generated from canonical processed artifacts.
- Reported counts reconcile with manifests and QC.
- Statistical comparisons use subjects as independent units.
- Every path and example command is valid from the repository root.
- No CV or model-fitting output is introduced.

## 7. Gate

Finish with the standard report and ask:

> Phase 2 EDA and the preprocessed-data README are complete. Would you like to
> change any analysis or documentation, or should I continue to Phase 3 frozen
> DINO feature extraction?

Then stop.

