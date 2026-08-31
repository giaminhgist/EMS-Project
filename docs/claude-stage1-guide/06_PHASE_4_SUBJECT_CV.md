# Phase 4 — Five-fold subject-level cross-validation

## Goal

Create a deterministic, stratified, five-fold outer cross-validation assignment
for all subjects in the audited `All_Data` population and save it under:

```text
/root/EMS-Project/CV
```

The split is subject-level. Trial-level splitting is forbidden.

## 1. Suggested files

```text
configs/cv_5fold.yaml
build_cv.py
src/cv/__init__.py
src/cv/config.py
src/cv/build_subject_folds.py
src/cv/validate_folds.py
tests/cv/test_subject_level_split.py
tests/cv/test_split_determinism.py
tests/cv/test_missing_trials.py
tests/cv/test_cv_manifest_compatibility.py
```

## 2. Input population

Use explicit IDs and labels from `processed_dataset/subject_manifest.csv`, then
reconcile them with the audited `All_Data.xlsx` population.

Before splitting, assert:

- every `All_Data` subject resolves to exactly one subject-manifest row;
- every included subject has exactly one explicit label;
- no subject is duplicated after preserving leading zeros;
- no unapproved external/held-out test subject is mixed into CV;
- trial completeness is not used to determine fold membership.

If `All_Data.xlsx` contains provided four-fold columns, preserve those columns as
source metadata but do not reuse them as the requested five-fold assignment.

## 3. Split algorithm

Default configuration:

```yaml
n_splits: 5
shuffle: true
random_state: 2026
stratify_column: label
group_column: subject_id
```

Use `sklearn.model_selection.StratifiedKFold` on one row per subject.

For a balanced 80 HC / 80 SZ population, each validation fold should normally
contain 32 subjects, approximately 16 HC and 16 SZ. Measure actual counts rather
than hard-coding them.

Assign a single `validation_fold` value `0..4` to each subject. For fold `k`:

```text
validation: validation_fold == k
training:   validation_fold != k
```

## 4. Output schema

Use a versioned directory, for example:

```text
CV/5fold_seed2026/
├── cv_config.json
├── cv_metadata.json
├── fold_assignments.csv
├── validation_report.json
├── fold_0/
│   ├── train_subjects.csv
│   ├── val_subjects.csv
│   ├── train_trials.parquet
│   └── val_trials.parquet
├── fold_1/
│   └── ...
└── fold_4/
    └── ...
```

`fold_assignments.csv` minimum columns:

```text
subject_id, subject_numeric_id, group, label, validation_fold,
n_observed_trials
```

Subject files contain explicit IDs and labels. Trial files are created by joining
the subject files to `trial_manifest`; they contain only actual observed trials.
Do not fill missing stimuli.

Save checksums for:

- input subject manifest;
- input trial manifest;
- audited `All_Data` identity;
- CV config;
- fold assignments.

## 5. HC-only Stage-1 views

The canonical CV files retain both HC and SZ for future use. Also report the
Stage-1 subset counts for each fold:

```text
stage1_train_hc_subjects
stage1_val_hc_subjects
stage1_train_hc_trials
stage1_val_hc_trials
```

Do not create a separate random split of HC subjects. Stage 1 must filter HC from
the same outer fold assignment.

## 6. CLI

Provide:

```bash
python build_cv.py \
  --config configs/cv_5fold.yaml \
  --subject-manifest /root/EMS-Project/processed_dataset/subject_manifest.csv \
  --trial-manifest /root/EMS-Project/processed_dataset/trial_manifest.parquet \
  --output-root /root/EMS-Project/CV
```

Also support:

```text
--verify-only
--force
```

Never regenerate an existing split under the same directory with a different
seed. A changed configuration requires a new versioned directory unless the user
explicitly approves replacement.

## 7. Required tests

- Each subject occurs exactly once in `fold_assignments.csv`.
- Validation subject sets are mutually disjoint.
- The union of validation subject sets equals the complete input population.
- For every fold, train and validation subject intersections are empty.
- Every trial in a partition belongs to a subject in the corresponding subject
  partition.
- No subject contributes trials to both train and validation in one fold.
- Labels remain stratified within feasible integer limits.
- The same seed and input checksums reproduce identical assignments.
- A different seed changes at least one assignment while remaining valid.
- Non-contiguous and leading-zero IDs survive all file round trips.
- Incomplete subjects retain only their real trials.
- Stage-1 HC views contain no SZ trial.

## 8. Acceptance criteria

- Five folds pass all leakage and completeness checks.
- All `All_Data` subjects are assigned exactly once as validation subjects.
- Class and trial counts are reported per fold.
- Files resolve every subject/trial through explicit identifiers.
- The split is immutable and checksum-tracked.

## 9. Gate

Finish with the standard report and ask:

> Phase 4 five-fold subject-level CV is complete. Would you like to change the
> seed or split artifacts, or should I continue to Phase 5 Stage-1 core
> implementation?

Then stop.

