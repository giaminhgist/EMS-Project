# Phase 2 — subject-level Stage-2 Dataset, DataLoader and bank loader

## Goal

Implement the complete Stage-2 data boundary under `src/stage2/`:

- one dataset item per subject;
- fixed 100-stimulus panel with an explicit missing-trial mask;
- deterministic balanced HC/SZ subject batching;
- validated full/crossfit normative-bank loading;
- typed batch objects and valid-trial flatten/scatter utilities.

Do not implement the Stage-2 neural network, losses or trainer in this phase.

## 1. Preconditions

Require:

- Phase 1 completion and user approval;
- five verified banks under `normative_bank/`;
- exact processed/CV paths confirmed by Phase 0;
- no unresolved stimulus-index mismatch.

Run bank `--verify-only` before starting.

## 2. Files to create or change

Recommended files:

```text
src/stage2/contracts.py
src/stage2/config.py
src/stage2/bank.py
src/stage2/dataset.py
src/stage2/sampler.py
src/stage2/collate.py
tests/stage2/test_bank_loader.py
tests/stage2/test_subject_dataset.py
tests/stage2/test_subject_sampler.py
tests/stage2/test_collate_and_flatten.py
```

Preserve existing Phase-1 classes and extend them rather than creating competing bank schemas.

## 3. Stage-2 configuration skeleton

Implement typed validated configuration sections needed by the data layer:

```text
seed
fold
paths.processed_root
paths.cv_root
paths.normative_bank_root
bank.train_mode
bank.verify_checksums
sampler.subject_batch_size
sampler.balance_groups
sampler.drop_last
runtime.num_workers
runtime.pin_memory
runtime.persistent_workers
```

Reject:

- fold outside `0..4`;
- odd balanced batch size;
- nonexistent paths;
- `persistent_workers=true` with `num_workers=0`;
- crossfit mode when crossfit artifacts are absent;
- bank schema/version mismatch.

Do not yet create the full model/loss config; Phase 4 will complete it.

## 4. Canonical subject index

Build an in-memory subject table from explicit manifests and CV partitions:

```text
subject_id
label
split                 train or val
fold
subject_array_path
number_of_observed_trials
bank_split_id         train only when crossfit is enabled
```

Requirements:

- exactly one row per subject in a requested split;
- no train/validation overlap;
- labels come from the canonical processed/CV manifest;
- every subject array path exists;
- subject IDs remain strings;
- do not assume subject IDs are contiguous array indices;
- do not derive labels during `__getitem__` from `<200` or `>=200`.

## 5. `Stage2SubjectDataset`

Implement a PyTorch `Dataset` whose `__len__` is the number of subjects in the split.

One `__getitem__` returns:

```python
Stage2SubjectSample(
    subject_id: str,
    label: int,
    heatmaps: Tensor,             # [100,3,48,64], float32
    trial_mask: Tensor,           # [100], bool
    stimulus_indices: Tensor,     # [100], int64
    category_ids: Tensor,         # [100], int64
    trial_uids: tuple[str | None, ...],
    bank_split_id: int | None,
)
```

### 5.1 Panel assembly

Initialize padded storage:

```text
heatmaps            zeros [100,3,48,64]
trial_mask          false [100]
stimulus_indices    arange(100)
category_ids        canonical category for every stimulus
trial_uids          None for missing slots
```

For every observed trial row:

1. resolve the exact subject-array row from the manifest;
2. verify stored `stimulus_index`;
3. write the heatmap into its canonical stimulus slot;
4. set that mask element true;
5. store the trial UID.

Reject duplicate subject–stimulus rows.

### 5.2 Heatmap transform

Reuse exactly the Stage-1 fixed leakage-free transform unless the audit finds a different approved contract:

```text
channel 0: log1p(max(x,0))
channel 1: log1p(max(x,0))
channel 2: clip(x,-1,1)
```

Do not fit a global Stage-2 normalization across all subjects. Any learned/fitted statistics must come from the current fold's training subjects and be recorded; the base uses the existing fixed transform only.

### 5.3 Memory behavior

- memory-map per-subject `.npy` arrays;
- reuse or adapt the existing Stage-1 small LRU array cache;
- do not retain all 160 full subject panels in each worker;
- do not share unsafe mutable NumPy memmaps across worker processes;
- return owned tensors when required by PyTorch worker semantics.

## 6. `Stage2Batch` typed contract

Implement a dataclass with:

```python
subject_ids: tuple[str, ...]
labels: Tensor                 # [B], float32
heatmaps: Tensor               # [B,100,3,48,64]
trial_mask: Tensor             # [B,100], bool
stimulus_indices: Tensor       # [B,100], int64
category_ids: Tensor           # [B,100], int64
bank_split_ids: Tensor | None  # [B], int64
trial_uids: tuple[tuple[str | None, ...], ...]
```

Provide:

```text
to_device(device)
n_subjects
n_valid_trials
validate()
```

Validation checks exact shapes/dtypes, binary labels, canonical stimulus slots and at least one valid trial per subject.

## 7. Collation

The collator stacks subject samples without flattening trials.

For `B=4`:

```text
labels                [4]
heatmaps              [4,100,3,48,64]
trial_mask            [4,100]
stimulus_indices      [4,100]
category_ids          [4,100]
bank_split_ids        [4]
```

Do not retrieve bank tensors in DataLoader workers. The model/training process loads banks once and gathers them after flattening valid trials.

## 8. Valid-trial flatten and scatter metadata

Implement one tested utility:

```python
flat = flatten_valid_trials(batch)
```

Return:

```text
heatmaps                 [N,3,48,64]
stimulus_indices         [N]
category_ids             [N]
subject_slots            [N]
stimulus_slots           [N]
labels_per_trial         [N]      # only for selecting HC matching loss, never trial BCE
bank_split_ids           [N] or null
```

Implement `scatter_valid_trials(flat_tensor, subject_slots, stimulus_slots, B, S=100)` returning a padded tensor plus the original mask.

Tests must prove round-trip correctness with arbitrary missing patterns.

## 9. Balanced subject sampler

Implement a deterministic `BalancedSubjectBatchSampler`.

Base behavior:

```text
batch_size=4 -> target 2 HC + 2 SZ
batch_size=8 -> target 4 HC + 4 SZ
```

Requirements:

- operate on subject dataset indices;
- shuffle independently within HC and SZ lists using `(seed,fold,epoch)`;
- never create batches by sampling trials;
- expose `set_epoch(epoch)`;
- report how final imbalance is handled;
- support exact resume by storing/reconstructing epoch state;
- deterministic iteration for a fixed seed/epoch;
- do not silently discard many subjects merely to obtain perfect class balance.

Recommended final-batch policy:

- use each subject once where possible;
- allow a smaller final batch;
- never duplicate a validation subject;
- record final-batch composition.

## 10. Normative bank loader

Implement a read-only `NormativeBankStore` that loads one outer fold.

### 10.1 On initialization

Verify:

- root and fold metadata schema;
- fold identity;
- Stage-1 checkpoint checksum against the approved registry;
- processed, DINO and CV checksums;
- stimulus manifest equality;
- required arrays for configured bank mode;
- all array shapes, dtypes and finite values;
- contributor/forbidden subject sets;
- crossfit assignment completeness.

Use memory mapping for large token arrays where safe.

Set tensors/buffers as non-trainable. Do not wrap bank arrays in `nn.Parameter`.

### 10.2 Bank selection

API:

```python
bank_key = bank_store.bank_for_subject(
    subject_id=...,
    split="train" | "val",
    bank_split_id=...,
)
```

Rules:

```text
train + crossfit     -> assigned crossfit bank
train + full mode    -> full fold bank
validation           -> full fold bank
```

### 10.3 Batched gather

Support different crossfit bank IDs inside one subject batch:

```python
gathered = bank_store.gather_trials(
    stimulus_indices=flat.stimulus_indices,  # [N]
    subject_slots=flat.subject_slots,        # [N]
    subject_bank_ids=batch.bank_split_ids,   # [B]
    split=batch_split,
)
```

Return a typed object:

```text
mu_trial              [N,128]
sigma_trial           [N,128]
count_trial           [N]
mu_token              [N,192,128] or None
sigma_token           [N,192,128] or None
mu_heat_token         [N,192,128] or None
sigma_heat_token      [N,192,128] or None
bank_ids              [N]
```

Gather by explicit stimulus and bank IDs. Avoid Python loops over 192 tokens.

## 11. Leakage and consistency assertions

At dataset/bank initialization, write an in-memory audit result proving:

- no train/validation subject overlap;
- no validation subject in any bank contributor list;
- in crossfit mode, no training HC belongs to its assigned bank contributors;
- HC and SZ subjects receive banks with comparable contributor counts;
- every dataset stimulus index exists exactly once in bank manifest;
- every subject label is explicit and binary.

Expose this audit to the trainer later; do not write Phase-5 output files yet.

## 12. Tests

### Dataset tests

- complete 100-trial subject;
- subject missing one trial;
- subjects 216/259-like severe missingness fixtures;
- duplicate stimulus row failure;
- noncanonical stimulus index failure;
- wrong heatmap shape/dtype/non-finite failure;
- explicit label use;
- deterministic fixed transform;
- LRU/memmap round trip.

### Sampler tests

- balanced composition;
- every subject seen;
- deterministic same seed/epoch;
- different order at next epoch;
- odd batch-size rejection in balanced mode;
- fixed validation order;
- no trial-level indices.

### Bank loader tests

- exact stimulus gather;
- mixed crossfit IDs within one batch;
- full bank for validation;
- checksum mismatch failure;
- missing token arrays failure only when required;
- self-inclusion failure;
- manifest-order mismatch failure;
- bank tensors require no gradient.

### Flatten/scatter tests

- complete panel;
- arbitrary missing mask;
- no cross-subject mixing;
- exact round trip;
- trial labels exposed only for HC selection metadata.

## 13. Dry-run utility

Add a small module or test command that, for one fold, prints without training:

```text
train/val subject counts and labels
observed trial counts
batch tensor shapes
first train batch subject IDs
assigned bank IDs
gathered bank tensor shapes
missing trial counts
leakage audit status
```

Do not create `stage2_trainer.py` yet. A module command such as this is sufficient:

```bash
python -m stage2.dataset --config configs/stage2/base.yaml --fold 0 --dry-run
```

If the full config does not exist yet, use a minimal data config fixture and update it in Phase 4.

## 14. Acceptance criteria

- Dataset length equals subject count, not trial count.
- One item contains a 100-slot masked panel.
- DataLoader batches subjects and maintains class balance.
- Missing trials are excluded by mask.
- Full and crossfit banks are selected correctly.
- Bank gather returns exact shapes and stimulus rows.
- All leakage/checksum assertions pass.
- Unit tests and one real fold dry-run pass.
- No Stage-2 neural network, loss or training loop is implemented.

## 15. Gate

End with exactly:

> Phase 2 subject-level Dataset/DataLoader and normative-bank loader are complete. Would you like to inspect or change the batching, missing-trial, or crossfit-bank behavior, or should I continue to Phase 3 and implement the Stage-2 model and losses?

Then stop.

