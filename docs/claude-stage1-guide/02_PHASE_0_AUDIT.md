# Phase 0 — Read-only audit and decision gate

## Goal

Understand the real repository and raw EMS layout before writing code. This
phase is read-only. Do not create source files, directories, virtual
environments, generated data, or configuration files.

## 1. Repository audit

From `/root/EMS-Project`, inspect and report:

- existing Git state and current branch, without changing either;
- existing top-level files and directories;
- existing Python package layout, configuration system, test framework, and
  dependency manager;
- any existing preprocessing, DINO, CV, Stage-1, logging, or checkpoint code;
- unrelated modified or untracked files that must be preserved;
- Python, PyTorch, CUDA, GPU, NumPy, pandas, scikit-learn, Pillow, SciPy,
  openpyxl, and pyarrow availability;
- disk space required for raw, processed, DINO, and output artifacts.

Do not print secrets or full environment variables.

## 2. Raw EMS audit

Inspect `/root/EMS-Project/original_dataset` without modifying it. Resolve the
actual paths for:

- `All_Data.xlsx` or equivalently named split/index file;
- the 160 expected subject workbooks;
- the `Free_viewing` sheet;
- the 100 stimulus images and their four category directories;
- any README or metadata shipped with the dataset.

Report actual rather than assumed values for:

- number and exact filename format of subject workbooks;
- subject IDs as strings and numeric values;
- workbook sheets and columns;
- number of rows, trials, and unique images;
- duplicate `(subject_id, IMAGE, FIX_INDEX)` keys;
- missing values and malformed numeric values;
- off-canvas coordinates relative to 1024 × 768;
- `FIX_INDEX` monotonicity, duplicate indices, and gaps per trial;
- fixation-duration range and the number below candidate thresholds;
- image filenames referenced by workbooks but missing on disk;
- disk images never referenced by a workbook;
- duplicate image basenames across category folders;
- image dimensions and aspect ratios;
- how `All_Data.xlsx` represents subject IDs, folds, and diagnostic groups;
- whether the available 160 subjects are the cross-validation subset described
  by the dataset documentation, and whether any separate test data are actually
  present.

Expected values are validation targets, not values to hard-code:

- 160 subjects: 80 HC and 80 SZ;
- non-contiguous numeric IDs between 0 and 303;
- 100 images of 1024 × 768;
- approximately 15,912 observed subject-image trials;
- approximately 225,159 fixation rows;
- some subjects missing stimuli, including two strongly incomplete subjects.

If actual data disagree, report the difference and stop; do not force the raw
data to match expectations.

## 3. Required decisions to ask the user

Present a recommended default for each item and request explicit confirmation:

1. **Python package directory**
   - Requested path: `src/Stage-1`.
   - Recommended importable path: `src/stage1` because `Stage-1` is not a valid
     normal Python package identifier.
   - Do not rename silently.

2. **DINO variant**
   - Current methodology default: original pretrained `DINO ViT-S/16`, frozen.
   - Verify the original 1024×768-pixel stimulus, then resize it
     deterministically to `[3,384,512]` in C×H×W tensor order for DINO. This
     preserves the 4:3 aspect ratio and the normalized image coordinate system.
     Do not crop, pad, flip, or rotate.
   - This produces a 24×32 patch grid and `[768,384]` patch-token features;
     these numbers mean `number_of_patches × embedding_dimension`, not image
     width and height.
   - If the user wants DINOv2 or another checkpoint, update the tensor contract
     before implementation.

3. **Off-canvas policy**
   - Recommended: exclude off-canvas fixations from spatial map construction but
     retain their counts in trial QC.
   - Alternatives: clip to the boundary or reject the entire trial.

4. **Short-fixation policy**
   - Recommended first reproducibility version: retain parsed fixations with
     positive duration and expose `--min-fix-duration-ms` as a configurable
     ablation/QC threshold; drop zero or negative duration events.

5. **Heatmap kernel**
   - Recommended: Gaussian sigma `2.0` cells on the 48 × 64 grid, stored in the
     resolved preprocessing configuration and tested by ablation later.

6. **Manifest format**
   - Recommended: CSV for human-readable subject/image manifests and Parquet for
     the trial manifest, with an optional CSV export.
   - If pyarrow is unavailable, ask before changing the canonical format.

7. **Label source**
   - Recommended: parse and validate labels from the dataset-provided metadata
     when present. Use the documented numeric-ID rule only to construct the
     initial subject manifest, then store and consume the explicit `group` and
     `label` columns everywhere downstream.

## 4. Phase 0 deliverable

Provide the audit in the required phase-end report. Include a compact proposed
repository tree reflecting existing code. Do not create that tree yet.

End with:

> Phase 0 is complete. Please confirm or modify the seven decisions above. Once
> confirmed, should I continue to Phase 1 preprocessing?

Then stop.
