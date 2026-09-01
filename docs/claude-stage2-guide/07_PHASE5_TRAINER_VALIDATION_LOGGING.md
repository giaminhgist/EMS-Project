# Phase 5 — implement training, validation, history and checkpointing

## Goal

Create the root training entry point:

```text
/root/EMS-Project/stage2_trainer.py
```

and the reusable Stage-2 training engine under `src/stage2/`. The implementation must train and validate at subject level, support all approved ablations, write a durable history immediately after every epoch, select and preserve the best checkpoint, resume exactly, and export auditable subject/stimulus results.

## 1. Preconditions

Require explicit Phase-4 approval. Before editing:

1. inspect `git status --short`;
2. run the complete Stage-2 unit test suite;
3. verify banks for the requested folds;
4. verify each Stage-1 checkpoint hash against its bank metadata;
5. resolve the Phase-0 evaluation regime;
6. confirm that every requested ablation passes capability validation.

Stop on any mismatch. Never fall back to a different checkpoint or bank automatically.

## 2. Files to create or change

```text
stage2_trainer.py
src/stage2/trainer.py
src/stage2/metrics.py
src/stage2/history.py
src/stage2/checkpoint.py
src/stage2/validation.py
src/stage2/calibration.py
src/stage2/attribution.py
tests/stage2/test_stage2_history.py
tests/stage2/test_stage2_checkpoint.py
tests/stage2/test_stage2_metrics.py
tests/stage2/test_stage2_validation.py
tests/stage2/test_stage2_trainer_smoke.py
```

Keep `stage2_trainer.py` thin: argument parsing, configuration resolution, fold iteration and exit codes. Place testable behavior inside `src/stage2/`.

## 3. Command-line interface

Provide at least:

```text
--config PATH
--fold {0,1,2,3,4,all}
--ablation NAME
--seed INT
--evaluation-regime {pilot_existing_stage1,strict_nested_stage1}
--stage1-checkpoint PATH
--bank-root PATH
--output-root PATH
--resume PATH
--load-stage2-weights PATH
--device DEVICE
--num-workers INT
--deterministic
--dry-run
--verify-only
--max-train-subjects INT
--max-val-subjects INT
--max-train-batches INT
--max-val-batches INT
```

Rules:

- `--resume` means exact continuation and is mutually exclusive with `--load-stage2-weights`.
- `--load-stage2-weights` loads model tensors only into a new run with a fresh optimizer, scheduler and history.
- smoke-limit flags must be rejected for a non-smoke production run unless `--dry-run` is present.
- `--fold all` runs folds sequentially by default so one fold failure cannot corrupt concurrent output.
- a completed run directory is never overwritten.
- CLI overrides and their previous/resolved values are written to metadata.

Example target commands:

```bash
python stage2_trainer.py --config configs/stage2/base.yaml --fold 0 --dry-run

python stage2_trainer.py --config configs/stage2/base.yaml --fold 0

python stage2_trainer.py --config configs/stage2/base.yaml --fold all

python stage2_trainer.py --config configs/stage2/base.yaml \
  --fold all --ablation no_bank

python stage2_trainer.py --resume \
  outputs/stage2/<run_id>/fold_0/checkpoints/last_stage2_fold0.pt
```

## 4. Evaluation-regime behavior

The Phase-0 decision must affect both data construction and the claims in output metadata.

### 4.1 `pilot_existing_stage1`

- Reuse the approved existing Stage-1 fold checkpoint.
- The fold validation partition may be used for per-epoch validation and checkpoint selection.
- Mark the result `exploratory` because the Stage-1 checkpoint may already have been selected using that same fold's validation HC data.
- Do not label this as a strict held-out estimate.

### 4.2 `strict_nested_stage1`

- Require a Stage-1 checkpoint trained and selected without the outer held-out fold.
- Persist an inner, subject-level selection split inside the outer-training partition.
- Per-epoch `val_*` fields refer to the inner selection split.
- Select the checkpoint, calibration and decision threshold without the outer held-out subjects.
- Evaluate the outer fold once after all choices are frozen, under an `outer_test/` directory.
- Never use the outer score for early stopping, scheduler decisions or ablation selection.

Record `validation_scope=outer_fold_exploratory` or `validation_scope=inner_selection` in every history row. In strict mode, store final outer-fold metrics separately and never append them as another training epoch.

## 5. Run initialization

Before the first optimizer step:

1. resolve and validate configuration;
2. create a unique run ID containing a UTC timestamp, experiment name, ablation, seed and short config hash;
3. create output directories without overwriting existing content;
4. write atomically:
   - `config_resolved.yaml`;
   - `environment.json`;
   - `run_metadata.json`;
   - `source_checksums.json`;
5. record git commit plus dirty-state summary without changing the worktree;
6. record dataset, split, Stage-1 and bank checksums;
7. seed Python, NumPy, PyTorch, CUDA and DataLoader workers;
8. instantiate subject-level datasets/loaders;
9. construct the model and assert trainable/frozen parameter groups;
10. run one no-gradient shape check;
11. write `audit/tensor_shapes.json` and `audit/leakage_checks.json`.

If initialization fails, leave an explicit failure log but do not create a fake history row.

## 6. Optimizer and scheduler

Use AdamW by default. Separate parameter groups:

```text
group 0  Stage-2 trainable parameters   LR = 1e-4, WD = 5e-4
group 1  final encoder block, optional  LR = 1e-5, WD = 5e-4
```

Exclude biases and normalization scale/shift from weight decay only if this behavior is explicit in config and metadata. Do not silently inherit framework defaults.

Use a phase-aware linear-warmup/cosine schedule. Recommended behavior:

- Phase 2A has its own alignment optimizer/scheduler and local epoch counter;
- at Phase 2B, recreate the optimizer over all Stage-2 trainable modules and start its warmup/cosine schedule;
- global epoch remains monotonically increasing across both phases;
- checkpoint stores both `epoch` and `phase_epoch`;
- Phase-2A epochs have `eligible_for_best=false`.

If the selected ablation sets `alignment_epochs=0`, begin directly in Phase 2B.

## 7. Phase 2A — bank alignment warm-up

Data:

- outer-training HC subjects only;
- crossfit bank per training subject when approved;
- all available stimulus trials, masked correctly.

Train:

- query attention pooler/projection;
- bank adapter/projection;
- relation/comparator parameters used by the match objective;
- optional token branch.

Freeze:

- transferred heatmap encoder;
- every normative-bank tensor;
- diagnostic classifier and subject aggregation parameters not required by `L_match`.

Objective:

```text
trial bank       L_match = L_trialmatch + 0.5 * L_bankrank
token bank       L_match += 0.25 * L_tokenmatch
```

Track alignment validation on HC subjects from the allowed selection partition. Do not select the final diagnostic checkpoint in this phase.

## 8. Phase 2B — diagnostic training

Data:

- HC and SZ training subjects;
- subject-balanced sampler;
- full or masked 100-stimulus panels;
- two category-stratified subset views only when consistency loss is active.

Base objective:

```text
L_total = L_cls
        + 0.3 * L_aux
        + 0.1 * L_match
        + 0.1 * L_cons
        + 0.01 * L_ent
        + lambda_anchor * L_anchor
```

Calculate losses exactly as defined in `01_STAGE2_CONTRACTS.md` and implemented in Phase 3. All diagnostic BCE terms use subject logits `[B]` and subject labels `[B]`.

### 8.1 One training step

For each subject batch:

1. transfer tensors asynchronously only when pinned memory/CUDA are active;
2. zero gradients with `set_to_none=True`;
3. run autocast when AMP is enabled;
4. forward one full/subset subject panel;
5. compute typed loss output;
6. reject non-finite component or total loss;
7. scale/backpropagate;
8. unscale gradients before norm measurement;
9. calculate global pre-clip gradient norm;
10. clip at configured norm;
11. record whether clipping occurred;
12. optimizer step and scaler update;
13. scheduler step at the configured granularity;
14. accumulate loss sums weighted by the number of subjects, not by trials;
15. release transient token maps unless they are requested for audit output.

Do not skip a non-finite batch and continue as if training were valid. Save a failure diagnostic and stop the fold.

## 9. Deterministic validation

Validation must:

- set `model.eval()` and use inference mode;
- contain each allowed validation subject exactly once;
- use deterministic subject order and no augmentation;
- use the full fold bank, never a crossfit shard;
- mask missing stimuli before every softmax/reduction;
- aggregate logits and labels by subject ID;
- reject duplicate or missing subject IDs;
- calculate losses as subject-weighted means;
- restore training mode afterward when appropriate.

Return a typed validation result containing:

```text
mean component losses
subject IDs
labels [N]
raw logits [N]
probabilities [N]
predictions [N]
stimulus attention [N,100]
stimulus evidence [N,100]
stimulus contribution [N,100]
semantic compatibility [N,100]
normative deviation [N,100]
trial mask [N,100]
optional semantic patch maps [N,100,12,16]
```

The optional patch maps may be streamed to a compressed file to control memory.

## 10. Metrics

Implement metrics without assuming both classes appear in every mini-batch. Compute them once from the complete validation subject set.

Required:

```text
accuracy
balanced_accuracy
AUROC
F1
sensitivity
specificity
Brier score
confusion matrix
number of HC subjects
number of SZ subjects
```

Use HC=`0`, SZ=`1`. Define and test the zero-denominator behavior. If AUROC is undefined because the complete validation set contains one class, store `null` plus a warning; do not substitute zero.

The default threshold during training is 0.5. Any calibrated threshold must be fitted only on permitted training/inner-selection predictions, saved with provenance, and frozen before outer evaluation.

## 11. Best-epoch rule

Base selection metric:

```text
maximize val_balanced_accuracy
```

Deterministic tie breakers in order:

1. higher `val_auroc`;
2. lower `val_loss`;
3. earlier epoch.

Only Phase-2B/2C epochs are eligible. Set:

```text
eligible_for_best
is_best_epoch
best_epoch_so_far
```

before committing the epoch. Early stopping monitors the same ordered rule. Do not use training loss to break a validation tie.

## 12. Durable per-epoch history

Path:

```text
outputs/stage2/<run_id>/fold_<k>/history.csv
```

Use the complete stable column set from Section 22 of `01_STAGE2_CONTRACTS.md`, plus these execution-state fields:

```text
phase_epoch
global_step
validation_scope
optimizer_step_count
skipped_optimizer_step_count
```

Important semantics:

- `learning_rate` and `encoder_learning_rate` are end-of-epoch values;
- `learning_rate_min/max` cover all optimizer steps in that epoch;
- `weight_decay` is the active Stage-2 group value;
- `grad_norm_mean/max` use pre-clip, unscaled norms;
- `grad_clip_fraction` is clipped optimizer steps divided by attempted optimizer steps;
- unused loss components are empty/null but columns remain present;
- `epoch` is global and zero-based or one-based consistently; document the choice;
- no column may appear or disappear after epoch 0.

### 12.1 Epoch commit protocol

At the end of every successfully completed epoch:

1. finish deterministic validation;
2. calculate metrics and the best-epoch decision;
3. construct the complete history row in memory;
4. atomically write `last_stage2_fold<k>.pt` to a temporary sibling, flush, `fsync`, and replace;
5. if best, atomically write `best_stage2_fold<k>.pt`;
6. rewrite `history.csv` atomically through a temporary sibling, flush, `fsync`, and replace;
7. atomically write `epoch_commit.json` last, containing epoch, history-row hash and checkpoint hash;
8. `fsync` the containing directory where supported;
9. only then print `epoch committed`.

This ensures history is persisted immediately after every epoch and is never ahead of the resumable checkpoint.

On resume, if the last checkpoint is one epoch ahead of history because of a crash, rerun validation deterministically from that checkpoint and reconstruct the missing row. If history is ahead of the checkpoint, stop as corrupted state. Never truncate silently.

## 13. Checkpoints and exact resume

Both best and last checkpoints must satisfy Section 23 of the contract. Additionally store:

```text
phase_epoch
global_step
optimizer step counters
history row hash
best-rule tuple
calibration state, if fitted
ablation specification and diff
```

Exact resume must restore:

- model;
- optimizer and all parameter groups;
- scheduler;
- AMP scaler;
- epoch/global step/training phase;
- best metric, best epoch and early-stopping counter;
- Python, NumPy, CPU/CUDA PyTorch RNG states;
- sampler state and DataLoader epoch seed;
- subset-view generator state.

Reject resume when any critical checksum, fold, schema, architecture, ablation or split differs. A user who intentionally wants different configuration must use `--load-stage2-weights` and a new run.

## 14. Validation artifacts

For the selected best checkpoint, write atomically:

```text
validation/metrics.json
validation/subject_predictions.parquet
validation/stimulus_attributions.npz
validation/calibration.json
audit/bank_match_metrics.json
```

`subject_predictions.parquet` contains one row per subject:

```text
subject_id
fold
split_scope
label
raw_logit
uncalibrated_probability
calibrated_probability
threshold
prediction
is_correct
num_available_stimuli
```

`stimulus_attributions.npz` must align arrays with explicit `subject_ids` and `stimulus_ids`. Store at least attention, evidence, contribution, semantic compatibility, normative deviation and trial mask.

Never interpret attention alone as causality. The report should distinguish:

- importance: category-balanced stimulus attention and additive contribution;
- deviation: normative distance/z features;
- semantic information: matched query–bank compatibility and optional token-level semantic map.

## 15. Calibration

Fit one positive temperature using only permitted non-test predictions:

- pilot mode: use a documented inner split or out-of-fold predictions within the training partition;
- strict mode: use inner-selection or inner-fold out-of-fold predictions;
- never fit temperature or threshold on the outer held-out fold.

Save pre/post calibration Brier score, negative log-likelihood, temperature, sample IDs, fit scope and checksum. If calibration is disabled, write a metadata file saying so rather than omitting provenance.

## 16. Logging and failure records

Write a human-readable `train.log` and structured run metadata. Include:

- start/end UTC times;
- host/device and library versions;
- resolved fold/seed/ablation;
- subject and trial counts;
- trainable/frozen parameters;
- epoch summaries;
- checkpoint/history commit results;
- early stopping reason;
- warnings and fatal errors.

Do not print raw subject medical data. Subject IDs may appear only where needed for audit/prediction alignment.

On fatal failure, atomically write `run_failure.json` with exception type, phase, epoch, last committed epoch and safe diagnostic context, then exit non-zero.

## 17. Tests

At minimum, add tests for:

1. one CPU synthetic epoch trains and validates;
2. losses and metrics are aggregated by subject count;
3. validation contains each subject exactly once;
4. missing stimuli receive zero attention/contribution;
5. history is committed after every epoch with the stable column set;
6. `is_best_epoch` and tie breakers are correct;
7. alignment epochs are ineligible for best;
8. grad norms are measured after AMP unscale and before clipping;
9. non-finite loss causes a hard failure;
10. atomic checkpoint replacement leaves no partial target;
11. exact resume produces the same next-epoch parameters and metrics as uninterrupted training;
12. incompatible resume fails;
13. weight-only initialization starts a fresh history;
14. pilot and strict validation scopes cannot be confused;
15. outer subjects are never used for strict-mode early stopping/calibration;
16. ablation metadata and config diff reach every checkpoint;
17. subject prediction and attribution arrays share the same ID order;
18. token maps stream without changing logits;
19. `--fold all` creates five isolated fold directories;
20. a one-batch `--dry-run` writes audit outputs but no production best checkpoint.

Run unit tests on CPU. If CUDA is available, also run a small AMP test and report it separately.

## 18. Required smoke commands

After unit tests, run from `/root/EMS-Project`:

```bash
python stage2_trainer.py \
  --config configs/stage2/base.yaml \
  --fold 0 \
  --verify-only

python stage2_trainer.py \
  --config configs/stage2/base.yaml \
  --fold 0 \
  --dry-run \
  --max-train-subjects 4 \
  --max-val-subjects 4 \
  --max-train-batches 1 \
  --max-val-batches 1
```

Then perform a two-epoch synthetic or explicitly marked smoke run to verify history, best/last checkpoints and resume. Do not present smoke metrics as experimental results.

## 19. Phase-5 completion gate

After all verification, stop all repository changes and use the standard completion report.

The report must include:

- exact root commands tested;
- one history-row excerpt with no real subject data;
- best/last checkpoint paths;
- resume equivalence result;
- evaluation regime and validation scope;
- runnable versus unavailable ablations.

Ask exactly:

> Phase 5 is complete. Would you like to inspect or change the trainer, validation, history or checkpoint behavior, or should I continue to Phase 6 and generate the Stage-2 README, run final QA and prepare the handoff?

Do not start Phase 6 until the user explicitly approves.
