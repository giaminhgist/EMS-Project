# Phase 0 — read-only Stage-2 audit and decision gate

## Goal

Audit the real `/root/EMS-Project` repository, five CV folds, processed data, Stage-1 code, trained checkpoints and available disk space before implementing Stage 2.

This phase is strictly read-only.

## 1. Required reading

Read completely:

```text
00_MASTER_WORKFLOW.md
01_STAGE2_CONTRACTS.md
/root/EMS-Project/readme/Stage1/README.md
/root/EMS-Project/configs/stage1/base.yaml
/root/EMS-Project/src/stage1/heatmap_encoder.py
/root/EMS-Project/src/stage1/model.py
/root/EMS-Project/src/stage1/normative_bank.py
/root/EMS-Project/src/stage1/checkpoint.py
/root/EMS-Project/src/stage1/dataset.py
/root/EMS-Project/src/stage1/config.py
/root/EMS-Project/stage1_trainer.py
```

If any path differs, find the implemented equivalent and report it.

## 2. Forbidden operations in Phase 0

Do not:

- create or edit files;
- make directories;
- generate a bank;
- run training;
- install packages;
- modify Git state;
- delete caches or outputs;
- create a virtual environment;
- download new model weights.

Read-only commands and small in-memory Python inspections are allowed.

## 3. Repository audit

Report:

- current branch and commit SHA;
- `git status --short` without modifying anything;
- top-level directory structure;
- Python package layout;
- test framework and configuration;
- dependency manager and active environment;
- Python, PyTorch, NumPy, pandas, scikit-learn, pyarrow and CUDA versions;
- available CPU RAM, GPU memory and disk space;
- whether existing Stage-2 files already exist;
- unrelated user modifications that must be preserved.

## 4. Processed-data audit

Verify without loading all arrays into RAM:

- processed manifest paths and checksums;
- exactly 160 subjects with explicit labels;
- exactly 100 canonical stimuli;
- category mapping and counts;
- per-subject heatmap arrays and row indices;
- heatmap shape `[3,48,64]`, dtype and finite values;
- number and IDs of incomplete subjects;
- missing-trial representation;
- trial UID uniqueness;
- subject/stimulus ID consistency across manifests.

Report observed values rather than assuming the documentation is current.

## 5. Five-fold CV audit

For `fold_0` through `fold_4`, report:

```text
number of train subjects
number of validation subjects
train HC/SZ counts
validation HC/SZ counts
subject overlap count
partition filenames
partition checksums
```

Required assertions:

- no train/validation subject overlap within a fold;
- the five validation sets are mutually disjoint;
- their union equals all available subjects;
- labels agree with the canonical processed manifest;
- stimulus metadata is not inferred from filename prefixes.

## 6. Stage-1 architecture/API audit

Confirm the actual implemented values and signatures:

```text
HeatmapPatchEncoder class path
input shape
patch size
token grid
hidden dimension
number of residual blocks
forward signature
checkpoint state-dict key prefix
Stage1Model forward flags
trial embedding shape
fused token return flag
heatmap token return flag
normative-bank estimator and epsilon
```

Confirm that unmasked Stage-1 inference is expressed by `token_mask=None` or report the actual API.

## 7. Stage-1 checkpoint inventory

Recursively inspect `outputs/stage1/` for:

```text
best_stage1_fold0.pt ... best_stage1_fold4.pt
last_stage1_fold0.pt ... last_stage1_fold4.pt
```

For every candidate best checkpoint, report:

- absolute path;
- file size and SHA-256;
- fold and run ID stored inside;
- best epoch and best metric;
- resolved config hash;
- input/checksum metadata;
- whether the state dict contains the expected encoder, semantic, fusion and pooling keys;
- whether the checkpoint loads on CPU without mutation.

If more than one eligible best checkpoint exists for a fold, do not choose automatically. Present a table and ask the user to select.

## 8. Existing bank audit

Inspect any banks currently stored under `outputs/stage1/` or elsewhere.

For each candidate bank, report:

```text
fold
checkpoint SHA-256
contributing subject IDs
forbidden validation overlap
array names and shapes
stimulus manifest
include_token_level status
epsilon and estimator
array checksums
```

Do not copy or move existing banks in this phase.

## 9. Disk estimate

Estimate space for:

- five full trial banks;
- five full fused-token banks;
- four crossfit banks per outer fold;
- optional heatmap-token banks;
- Stage-2 histories, checkpoints and interpretation outputs;
- temporary atomic-write files.

Use actual dtype/shape calculations. Flag if free disk is less than twice the estimated bank build peak plus expected Stage-2 outputs.

## 10. Required user decisions

At the end of the audit, present these decisions with a recommendation.

### Decision A — evaluation regime

```text
A1: pilot_existing_stage1
    Reuse existing five Stage-1 fold checkpoints.
    Faster, but may not be a strict untouched-test estimate if checkpoint
    selection used the later Stage-2 validation HC subjects.

A2: strict_nested_stage1 (recommended for final paper estimate)
    Stage-1 selection occurs only inside each Stage-2 outer-training fold,
    followed by refit and bank creation.
    Requires additional Stage-1 training not implemented by these Stage-2 phases.
```

If A2 is selected but strict checkpoints do not exist, stop and request a separate Stage-1 nested-retraining task before Phase 1.

### Decision B — bank contents

```text
B1: trial bank only
    Lowest storage; stimulus importance and trial-level semantic compatibility.

B2: trial + fused-token bank (recommended)
    Enables serial token cross-attention and 12x16 semantic maps.

B3: trial + fused-token + heatmap-token bank
    Adds direct same-space heatmap cosine/z-score; approximately doubles
    token-bank storage relative to B2.
```

### Decision C — training-bank self-inclusion

```text
C1: four-way crossfit (recommended)
    Each training subject uses a bank whose HC contributor subset excludes
    the subject's assigned split.

C2: full bank for all training subjects
    Simpler but training HCs contribute to their own reference bank.
```

### Decision D — base encoder policy

```text
D1: frozen Stage-1 heatmap encoder (recommended base)
D2: unfreeze final residual block as the base
```

Even if D2 is chosen, preserve D1 as an ablation and disable direct heat-token z-scores after unfreezing.

### Decision E — Stage-2 base model

```text
E1: trial-bank primary model (recommended)
E2: fused-token serial-attention model as base
```

Both should remain runnable through named configurations when B2/B3 artifacts exist.

## 11. Phase-0 output

Return one audit report in chat containing:

- all observations above;
- the proposed exact checkpoint path for each fold;
- blockers;
- the five decision questions;
- recommended choices and tradeoffs.

Do not create `stage1_checkpoints.yaml` until the user confirms the choices in the next message.

## 12. Gate

End with exactly:

> Phase 0 read-only audit is complete. Please confirm the checkpoint selected for each fold and Decisions A–E. Would you like to change anything, or should I continue to Phase 1 and build the five normative-bank artifacts?

Then stop.

