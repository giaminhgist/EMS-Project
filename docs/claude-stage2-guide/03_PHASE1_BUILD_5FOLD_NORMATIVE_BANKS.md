# Phase 1 — build and verify five fold-specific HC normative banks

## Goal

Implement a reproducible root command that loads the user-approved Stage-1 checkpoint for each fold, performs unmasked inference on that fold's training HC trials, and writes audited bank artifacts under:

```text
/root/EMS-Project/normative_bank/fold_0
...
/root/EMS-Project/normative_bank/fold_4
```

Build full banks and, when approved in Phase 0, four cross-fitted training banks per outer fold.

Do not implement the Stage-2 dataset or classifier in this phase.

## 1. Preconditions

Require explicit user approval of:

- one Stage-1 checkpoint path for each fold;
- evaluation regime;
- trial/token bank contents;
- crossfit policy;
- encoder freeze/base-model policy.

If any checkpoint is missing or ambiguous, stop.

## 2. Files to create or change

Recommended files:

```text
configs/stage2/stage1_checkpoints.yaml
configs/stage2/bank.yaml
build_stage2_normative_banks.py
src/stage2/__init__.py
src/stage2/contracts.py
src/stage2/bank_builder.py
tests/stage2/test_bank_builder.py
tests/stage2/test_bank_crossfit.py
tests/stage2/test_bank_cli.py
```

Reuse existing atomic-write and checksum utilities from preprocessing/Stage 1 when their behavior meets this guide. Do not copy those utilities into a second implementation.

## 3. Checkpoint registry

Write only the paths explicitly approved by the user:

```yaml
schema_version: 1
evaluation_regime: <approved value>
folds:
  0:
    checkpoint: /root/EMS-Project/outputs/stage1/<run>/fold_0/checkpoints/best_stage1_fold0.pt
    sha256: <measured>
  1:
    checkpoint: ...
    sha256: ...
  2:
    checkpoint: ...
    sha256: ...
  3:
    checkpoint: ...
    sha256: ...
  4:
    checkpoint: ...
    sha256: ...
```

The loader must verify the stored SHA-256 before inference.

## 4. Bank configuration

Create a validated configuration similar to:

```yaml
seed: 2026
estimator: mean
epsilon: 1.0e-6
min_samples: 2
batch_size: 64
include_fused_token_bank: true
include_heatmap_token_bank: false
crossfit_splits: 4
crossfit_enabled: true
device: cpu
output_root: /root/EMS-Project/normative_bank
```

Use the approved Phase-0 choices, not the example values when they differ.

## 5. Root CLI contract

Implement:

```bash
python build_stage2_normative_banks.py \
  --checkpoint-registry configs/stage2/stage1_checkpoints.yaml \
  --config configs/stage2/bank.yaml \
  --fold all \
  --device cuda
```

Required options:

```text
--fold 0|1|2|3|4|all
--device cpu|cuda
--checkpoint-registry PATH
--config PATH
--output-root PATH
--include-fused-token-bank / --no-include-fused-token-bank
--include-heatmap-token-bank / --no-include-heatmap-token-bank
--crossfit-splits INT
--no-crossfit
--verify-only
--overwrite-incomplete
```

Do not offer an unrestricted `--force` that silently overwrites a verified complete bank. A config/checksum change must use a new bank version or require explicit removal by the user.

## 6. Fold input resolution

For fold `k`:

1. load the checkpoint registry entry;
2. verify path, SHA-256, stored fold, run ID and config;
3. instantiate the checkpoint's resolved `Stage1Config`;
4. load the Stage-1 training and validation datasets using existing Stage-1 code;
5. assert training data are HC only for bank creation;
6. collect validation subject IDs as a forbidden set;
7. verify train/validation subject disjointness;
8. verify processed, CV and DINO checksums stored in the checkpoint;
9. instantiate `Stage1Model` and load its complete state dict;
10. set `eval()` and use inference mode.

The full Stage-1 model is required here because `mu_trial` and `mu_token` are post-semantic-fusion bank artifacts.

## 7. Unmasked inference

For every training HC trial:

```python
out = stage1_model(
    batch,
    token_mask=None,
    return_fused=include_fused_token_bank,
    return_heatmap_tokens=include_heatmap_token_bank,
)
```

Collect:

```text
out.trial_embedding         [n,128]
out.fused_tokens            [n,192,128] when enabled
out.heatmap_tokens          [n,192,128] when enabled
stimulus_index              [n]
subject_id                  length n
trial_uid                   length n
```

Never use reconstructed heatmaps or validation embeddings to build the bank.

## 8. Streaming statistics

Implement one reusable accumulator for `[n,...]` features grouped by stimulus index.

Internally store:

```text
sum       float64 [100,...]
sumsq     float64 [100,...]
count     int64   [100]
```

Finalize with the equations in `01_STAGE2_CONTRACTS.md` and clamp standard deviation at `epsilon`.

Required output shapes:

```text
mu_trial             [100,128]
sigma_trial          [100,128]
count_trial          [100]
mu_token             [100,192,128] when enabled
sigma_token          [100,192,128] when enabled
mu_heat_token        [100,192,128] when enabled
sigma_heat_token     [100,192,128] when enabled
```

Use the canonical `stimulus_index`; never group by lexicographic filename order.

## 9. Full fold bank

Build the full bank using all training-partition HC subjects.

Before saving, assert:

- no forbidden validation subject contributed;
- every contributor is explicitly HC;
- all 100 stimuli have at least `min_samples`;
- all arrays are finite;
- every standard deviation is at least `epsilon`;
- feature-manifest stimulus order matches DINO and processed manifests;
- observed count sum equals the included trial count.

## 10. Four-way crossfit banks

When enabled:

### 10.1 Subject assignments

Create deterministic assignments using `(seed, outer_fold)`.

For HC subjects, balance:

- number of subjects per split;
- complete versus incomplete panel status when feasible.

For SZ subjects, independently stratify subject assignments so each split contains similar SZ counts.

Write:

```text
fold_<k>/crossfit/subject_assignment.csv
```

Columns:

```text
subject_id,label,bank_split_id,is_hc_bank_contributor,panel_trial_count
```

### 10.2 Crossfit bank construction

For split `j`, exclude HC subjects assigned to `j`, then build statistics from all remaining outer-training HC subjects.

Every training subject assigned to `j`, including SZ subjects, later uses this same bank.

Metadata must record included and excluded HC IDs.

### 10.3 Required crossfit assertions

For every training subject:

```python
if label == HC:
    assert subject_id not in contributors_of_assigned_bank
```

Also verify:

- all four banks contain all 100 stimuli;
- counts are finite and above `min_samples`;
- assignment is deterministic;
- HC and SZ counts across bank split IDs differ by at most one where possible;
- no validation subject is assigned or included.

## 11. Atomic publication

Build each bank in a temporary sibling directory. Only rename it to its canonical path after all arrays, metadata and audits pass.

Each bank directory contains:

```text
mu_trial.npy
sigma_trial.npy
count_trial.npy
mu_token.npy / sigma_token.npy             # configured
mu_heat_token.npy / sigma_heat_token.npy   # configured
feature_manifest.csv
metadata.json
```

Write `audit.json` only after reloading saved arrays and verifying shapes/checksums.

The root `manifest.json` is the last file published after all requested folds finish.

If one fold fails, leave verified earlier folds intact and mark the root manifest incomplete. Do not publish a false all-fold success.

## 12. `--verify-only`

Verification mode performs no writes and checks:

- all requested directories/files exist;
- registry/checkpoint/bank hashes match;
- metadata fold identity matches paths;
- contributor and forbidden ID sets are valid;
- array shapes/dtypes/finiteness;
- stimulus manifests are identical across five folds;
- crossfit exclusion properties;
- root manifest reports the same content found on disk.

Return nonzero on any mismatch.

## 13. Tests

### Unit tests

Use small synthetic tensors to test:

- float64 mean/variance accumulation;
- epsilon clamping;
- grouping by explicit stimulus index;
- wrong or duplicated stimulus indices;
- missing stimulus failure;
- forbidden contributor failure;
- crossfit assignment determinism;
- crossfit self-exclusion;
- metadata/checksum generation;
- atomic publication after success only.

### Integration test

Construct a tiny synthetic Stage-1-like model/dataset with:

```text
8 HC subjects
4 SZ metadata-only subjects
4 stimuli
2 crossfit splits
trial embeddings [n,128]
tokens [n,192,128]
```

Verify exact bank means against a direct NumPy calculation.

### Real smoke test

After unit/integration tests pass, build one approved fold using trial bank only into a dedicated smoke output root. Do not overwrite the canonical bank.

Then, if the user-approved config requires tokens, build the canonical fold and verify the expected storage estimate.

## 14. Recommended verification commands

Adapt to the repository's test runner:

```bash
python -m pytest tests/stage2/test_bank_builder.py -q
python -m pytest tests/stage2/test_bank_crossfit.py -q
python -m pytest tests/stage2/test_bank_cli.py -q

python build_stage2_normative_banks.py \
  --checkpoint-registry configs/stage2/stage1_checkpoints.yaml \
  --config configs/stage2/bank.yaml \
  --fold 0 \
  --output-root /root/EMS-Project/outputs/stage2_bank_smoke \
  --no-include-fused-token-bank

python build_stage2_normative_banks.py \
  --checkpoint-registry configs/stage2/stage1_checkpoints.yaml \
  --config configs/stage2/bank.yaml \
  --fold all

python build_stage2_normative_banks.py \
  --checkpoint-registry configs/stage2/stage1_checkpoints.yaml \
  --config configs/stage2/bank.yaml \
  --fold all \
  --verify-only
```

## 15. Acceptance criteria

- Five fold directories exist under root `normative_bank/`.
- Every full fold bank uses only that fold's training HCs.
- Every full fold has all 100 stimuli.
- Requested token arrays have exact shapes.
- Crossfit banks and assignments satisfy self-exclusion.
- Checkpoint and source checksums are recorded and verified.
- A root manifest reconciles all fold artifacts.
- Tests and `--verify-only` pass.
- No Stage-2 model or trainer code beyond shared contracts/bank utilities is implemented.

## 16. Gate

End with exactly:

> Phase 1 normative-bank generation is complete for all approved folds and configurations. Would you like to inspect or change the banks, contributor policy, or token-bank options, or should I continue to Phase 2 and implement the subject-level Stage-2 Dataset/DataLoader?

Then stop.

