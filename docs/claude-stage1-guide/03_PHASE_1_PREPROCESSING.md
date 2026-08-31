# Phase 1 — EMS preprocessing and three-channel heatmaps

## Goal

Implement deterministic preprocessing from fixation workbooks under
`/root/EMS-Project/original_dataset` to explicit manifests and per-subject
heatmap arrays under `/root/EMS-Project/processed_dataset`.

Do not implement EDA figures, DINO, CV, or Stage 1 in this phase.

## 1. Suggested files

Adapt names to the audited repository structure while preserving these
responsibilities:

```text
configs/preprocessing.yaml
preprocess_ems.py
src/preprocessing/__init__.py
src/preprocessing/config.py
src/preprocessing/inventory.py
src/preprocessing/identifiers.py
src/preprocessing/heatmaps.py
src/preprocessing/qc.py
src/preprocessing/storage.py
src/preprocessing/pipeline.py
tests/preprocessing/test_identifiers.py
tests/preprocessing/test_heatmaps.py
tests/preprocessing/test_trial_order.py
tests/preprocessing/test_storage_roundtrip.py
tests/preprocessing/test_preprocessing_smoke.py
```

Do not create a second implementation inside a notebook.

## 2. Configuration schema

Use a validated dataclass, Pydantic model, or equivalent typed schema. Minimum
resolved fields:

```yaml
raw_root: /root/EMS-Project/original_dataset
output_root: /root/EMS-Project/processed_dataset
subject_glob: "*.xlsx"
subject_filename_regex: "^[0-9]+[.]xlsx$"
sheet_name: Free_viewing
source_width: 1024
source_height: 768
heatmap_height: 48
heatmap_width: 64
gaussian_sigma_cells: 2.0
gaussian_truncate_sigma: 4.0
off_canvas_policy: drop
min_fix_duration_ms: 0
drop_nonpositive_duration: true
transition_sample_step_cells: 0.5
temporal_epsilon: 1.0e-8
dtype: float32
num_workers: <explicit integer>
seed: 2026
write_trial_manifest_csv: true
```

Reject unknown fields and invalid values. Write the completely resolved config,
not merely the user-supplied subset.

## 3. Inventory before conversion

The preprocessing command must perform a validation pass before writing any
heatmap:

1. inventory all stimulus files and create a unique image mapping;
2. inventory subject workbooks and canonicalize IDs as strings;
3. validate required sheet and column names;
4. validate diagnostic group resolution;
5. validate workbook image references against disk images;
6. measure anomalies described in Phase 0;
7. compute source-file SHA-256 values;
8. determine deterministic `subject_id` and `stimulus_index` order;
9. estimate output size and verify free space;
10. refuse to continue on ambiguous image basenames or duplicate trial keys.

The filename regex must exclude metadata workbooks such as `All_Data.xlsx` from
the subject-workbook inventory.

The inventory result becomes `source_inventory.json` and must be reproducible
when the source files do not change.

## 4. Trial parsing

For each subject workbook:

1. read only the required sheet and columns;
2. retain original row number for QC;
3. group by the exact `IMAGE` value without numeric assumptions;
4. stable-sort every trial by `FIX_INDEX` and then original row number;
5. verify duplicate or missing fixation indices and record flags;
6. coerce numeric fields explicitly and count conversion failures;
7. preserve the raw event counts and raw total positive duration;
8. apply the user-approved duration and coordinate policies;
9. never reorder stimuli to match a presumed numeric image ID;
10. never create a row for an unobserved subject-stimulus pair.

The time position for temporal progression must be computed on the original
ordered sequence of temporally valid positive-duration fixations. An off-canvas
fixation contributes elapsed duration but does not deposit spatial density when
the approved policy is `drop`.

## 5. Heatmap implementation

Implement the formulas in `01_DATA_MODEL_CONTRACTS.md` as independent pure
functions.

### 5.1 Gaussian splatting

- Work at floating-point grid coordinates.
- Evaluate only a local window of radius
  `ceil(gaussian_truncate_sigma * sigma)` for efficiency.
- Renormalize a border-truncated kernel to unit sum.
- Return finite float64 working arrays and cast to float32 only when finalized.
- Reject negative or non-finite output.

### 5.2 Fixation density

- Deposit one unit-mass Gaussian per spatially valid fixation.
- Verify output mass approximately equals the number of used fixations.
- Do not divide by fixation count, maximum, subject total, or group total.

### 5.3 Consecutive transition density

- Form pairs using adjacent events in original `FIX_INDEX` order.
- A pair is used only if both endpoints pass the spatial policy.
- Never bridge across a rejected event.
- Sample each line with maximum spacing 0.5 grid cells by default.
- Bilinearly splat sampled points, normalize that segment to unit mass, and then
  smooth it with the configured Gaussian.
- Renormalize after border cropping so each accepted segment contributes unit
  mass.
- Handle a zero-length pair as a point transition.

### 5.4 Temporal progression

- Calculate cumulative timing from positive fixation durations.
- Deposit `tau_i * G_i` only for spatially valid fixation positions.
- Divide by fixation density plus epsilon.
- Set cells with negligible density to zero.
- Clip only small floating-point overshoots outside `[-1, 1]` and count larger
  violations as implementation errors.

### 5.5 Trial validity

If a real trial has no usable spatial fixation after the approved policy, do not
silently create a training example. Record the trial and reason in QC, report
the count, and stop for user direction if this occurs in the real dataset.

## 6. Storage and lookup

Write one consolidated array per subject:

```text
subjects/<subject_id>/heatmaps.npy
```

Requirements:

- order rows by `stimulus_index`, not filename lexicographic order;
- write matching `stimulus_indices.npy`;
- store `subject_row_index` in the global trial manifest;
- avoid NumPy object arrays and pickle;
- validate shape, dtype, finite values, channel ranges, and row mappings before
  atomically publishing each subject directory;
- a rerun with unchanged config and source checksums must either verify and skip
  compatible output or produce byte-equivalent numeric arrays;
- incompatible existing output must fail unless `--force` is given.

Write the top-level `dataset_metadata.json` completion record last, only after
all arrays and manifests validate. Downstream readers must reject a dataset
without a successful completion record.

Provide an indexed accessor used by tests:

```python
record = store.get_trial(subject_id="000", stimulus_id="<canonical-id>")
assert record.heatmap.shape == (3, 48, 64)
```

The accessor resolves the composite key through manifests. It must not scan
arrays or parse numeric IDs at query time.

## 7. Command-line interface

The root command should support at least:

```bash
python preprocess_ems.py --config configs/preprocessing.yaml --dry-run
python preprocess_ems.py --config configs/preprocessing.yaml
python preprocess_ems.py --config configs/preprocessing.yaml --resume
python preprocess_ems.py --config configs/preprocessing.yaml --subjects 000 001
python preprocess_ems.py --config configs/preprocessing.yaml --force
```

`--dry-run` performs the inventory and validation without writing processed
arrays. `--subjects` is for smoke tests and must write to a user-specified test
output rather than make a partial canonical dataset look complete.

## 8. Required tests

Unit tests must verify:

- leading-zero subject IDs survive a CSV/Parquet round trip;
- non-contiguous IDs do not index arrays;
- a center fixation creates a symmetric map with mass one;
- a border fixation also has mass one after truncation;
- two fixations produce fixation mass two;
- one transition produces transition mass one;
- a zero-length transition is finite and has mass one;
- an invalid middle fixation prevents bridging two valid endpoints;
- temporal progression is negative for an early location and positive for a
  later location;
- temporal output is finite and in `[-1, 1]`;
- shuffled workbook rows yield the same maps after `FIX_INDEX` sorting;
- missing stimuli do not generate synthetic trials;
- composite-key lookup returns the exact stored row;
- repeated seeded preprocessing produces identical arrays on synthetic data.

Run a real-data smoke test with at least one HC and one SZ subject to validate
file parsing, but do not interpret group differences in this phase.

## 9. Acceptance criteria

- All tests pass.
- Canonical preprocessing completes for every valid observed trial.
- Manifests have unique IDs and composite trial keys.
- Every manifest row round-trips to the expected array row.
- Heatmaps are float32 `[3,48,64]`, finite, and follow the channel contract.
- Counts reconcile across source inventory, manifests, arrays, and QC.
- Missing trials remain absent.
- The raw directory is unchanged.

## 10. Gate

Finish with the standard report and ask:

> Phase 1 preprocessing is complete. Would you like to inspect or change the
> heatmap/QC outputs, or should I continue to Phase 2 EDA and documentation?

Then stop.
