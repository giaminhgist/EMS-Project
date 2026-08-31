# EMS data and Stage-1 model contracts

This file defines interfaces that must remain stable across phases. If the raw
audit makes a contract impossible, Claude Code must report the conflict and ask
the user to approve a revision before continuing.

## 1. Known EMS data facts

The project description reports:

- fixation-event data from free viewing of static scenes;
- raw columns `IMAGE`, `FIX_INDEX`, `FIX_DURATION`, `FIX_X`, `FIX_Y`, and
  `FIX_PUPIL` in a `Free_viewing` sheet;
- 160 available subjects, balanced as 80 HC and 80 SZ;
- non-contiguous subject IDs represented by workbook filenames;
- 100 stimuli, normally 1024 × 768 pixels, from four categories;
- 15,912 observed subject-stimulus trials rather than a complete 16,000-trial
  rectangle;
- no timestamps; fixation order and duration are the only temporal information;
- some short, zero-duration, and off-canvas fixations;
- 12 subjects with missing stimuli, including two substantially incomplete
  subjects.

These are audit expectations. The implementation must use values measured from
the actual files and must record discrepancies.

## 2. Identifier contract

### 2.1 Subject identifier

- `subject_id` is a canonical string taken from the workbook stem, preserving
  leading zeros, for example `"000"`.
- `subject_numeric_id` is a separate integer used only for validation and, when
  necessary, initial label derivation.
- `subject_id` must never be converted to an array offset.
- Downstream code consumes the explicit `group` and `label` fields from the
  subject manifest; it must not repeatedly infer diagnosis from the ID.

### 2.2 Stimulus identifier

- `stimulus_id` is a stable canonical string.
- Prefer the exact workbook `IMAGE` value if it uniquely identifies one disk
  image. If basenames collide across category directories, use the normalized
  relative POSIX image path and store the workbook value separately as
  `source_image_name`.
- `stimulus_index` is an explicitly generated contiguous integer `0..N-1` used
  only for efficient array storage.
- The only valid conversion between `stimulus_id` and `stimulus_index` is the
  `image_manifest.csv` mapping.

### 2.3 Trial identifier

The primary trial key is:

```text
(subject_id, stimulus_id)
```

It must be unique in `trial_manifest`. A deterministic `trial_uid` may be the
first 20 hexadecimal characters of:

```text
SHA256(subject_id + "\0" + stimulus_id)
```

Do not use Python's `hash()`.

## 3. Raw-to-grid coordinate contract

Default source canvas:

```text
width  = 1024
height = 768
```

Output grid:

```text
H = 48
W = 64
```

For an accepted fixation coordinate `(x, y)`:

\[
u=x\frac{W-1}{1024-1},\qquad
v=y\frac{H-1}{768-1}.
\]

Use floating-point grid positions during Gaussian splatting. Do not round to a
single cell before kernel generation. If source image dimensions differ,
resolve coordinate-system metadata during Phase 0 rather than silently scaling
from the JPEG dimensions.

## 4. Three-channel heatmap contract

Every observed trial produces:

```text
heatmap: float32 [3, 48, 64]
```

Channels are stored in this immutable order:

```text
0 = fixation_density
1 = consecutive_fixation_transition_density
2 = temporal_progression
```

No per-subject, per-group, or population z-score is applied during preprocessing.

### 4.1 Shared Gaussian kernel

For valid fixation `i` at grid position `p_i=(u_i,v_i)`, construct a Gaussian
kernel `G_i` on the 48 × 64 grid. Normalize each truncated kernel so its discrete
sum is one. The initial recommended sigma is `2.0` grid cells and must be stored
in `preprocessing_config.json`.

### 4.2 Channel 0: fixation density

\[
H_F=\sum_{i=1}^{n}G_i.
\]

Because each kernel has unit mass:

\[
\sum_{u,v}H_F(u,v)\approx n.
\]

This deliberately preserves the number of valid fixations. Do not normalize
each trial to unit mass or maximum one in the stored representation.

### 4.3 Channel 1: consecutive-fixation transition density

For every consecutive pair in original `FIX_INDEX` order, include a transition
only when both endpoints are spatially valid. Do not connect across a removed
intermediate fixation.

Rasterize the line segment with sub-cell sampling. Normalize the contribution
of each segment to unit mass before Gaussian smoothing, so long transitions do
not dominate merely because they contain more raster samples:

\[
H_T=\sum_{i=1}^{n-1}\mathcal{G}_{\sigma}
\left(\operatorname{UnitMassLine}(p_i,p_{i+1})\right).
\]

A zero-length transition deposits unit mass at its endpoint. With all endpoints
valid, total mass should be approximately `n-1`.

### 4.4 Channel 2: temporal progression

Because EMS has no timestamps, reconstruct elapsed time from fixation duration.
Compute the cumulative clock across all temporally valid positive-duration
events in original order, including elapsed time from an off-canvas event. Only
spatially valid events deposit a Gaussian. For such a fixation:

\[
t_i^{mid}=
\frac{\sum_{k<i}d_k+d_i/2}{\sum_k d_k},
\qquad
\tau_i=2t_i^{mid}-1.
\]

Thus early viewing is near `-1`, the middle near `0`, and late viewing near `+1`.
Compute the density-conditioned local temporal mean:

\[
H_P(u,v)=
\frac{\sum_i \tau_iG_i(u,v)}{H_F(u,v)+\epsilon}.
\]

Set unvisited locations to exactly zero. The expected numerical range is
approximately `[-1, 1]`. Fixation density remains available in channel 0, so
channel 2 need not duplicate density magnitude.

### 4.5 Model-side fixed transforms

The stored heatmaps remain raw according to the definitions above. The Stage-1
dataset may apply the fixed, leakage-free transform:

```text
x[0] = log1p(max(x[0], 0))
x[1] = log1p(max(x[1], 0))
x[2] = clip(x[2], -1, 1)
```

Optional channel standardization is an ablation and, if enabled, its statistics
must be fitted only on the current fold's training HC trials and saved with the
fold checkpoint.

## 5. Processed dataset storage contract

```text
processed_dataset/
├── dataset_metadata.json
├── preprocessing_config.json
├── source_inventory.json
├── image_manifest.csv
├── subject_manifest.csv
├── trial_manifest.parquet
├── trial_manifest.csv                 # optional human-readable export
├── qc_summary.json
└── subjects/
    └── <subject_id>/
        ├── heatmaps.npy               # [n_subject_trials, 3, 48, 64]
        ├── stimulus_indices.npy       # [n_subject_trials]
        └── trial_qc.parquet
```

`image_manifest.csv` minimum columns:

```text
stimulus_index, stimulus_id, source_image_name, category,
relative_image_path, width, height, sha256
```

`subject_manifest.csv` minimum columns:

```text
subject_id, subject_numeric_id, group, label, source_workbook,
n_fixation_rows, n_trials, n_missing_expected_stimuli, source_sha256
```

Use `label=0` for HC and `label=1` for SZ.

`trial_manifest` minimum columns:

```text
trial_uid, subject_id, stimulus_id, subject_numeric_id, stimulus_index,
group, label, category, subject_array_path, subject_row_index,
n_fixations_raw, n_fixations_used, n_transitions_used,
total_duration_ms_raw, total_duration_ms_used,
n_off_canvas, n_nonfinite, n_nonpositive_duration,
n_below_duration_threshold, fix_index_has_gap, qc_status
```

The pair `(subject_array_path, subject_row_index)` must retrieve the same trial
identified by `(subject_id, stimulus_id)`. Tests must verify this round trip.

## 6. Frozen DINO feature contract

Default model after Phase-0 confirmation:

```text
model              = pretrained DINO ViT-S/16
trainable           = false
source image        = 1024 × 768 pixels (W × H), verified from manifest
DINO input image    = 512 × 384 pixels (W × H), deterministic resize
image tensor        = float32 [3, 384, 512] (C × H × W)
patch grid          = [24, 32]
number patch tokens = 768
token dimension     = 384
```

Image preprocessing:

1. RGB conversion;
2. verify the source dimensions and checksum against `image_manifest.csv`;
3. resize the complete image deterministically to 384 × 512 with recorded
   interpolation and antialias settings;
4. no center crop, padding, flip, rotation, or random geometric augmentation;
5. official pretrained-model normalization;
6. extract final normalized patch tokens;
7. remove the CLS token before storage.

Because source and DINO input have the same 4:3 aspect ratio and the full image
is resized without cropping or padding, normalized positions remain aligned
with the 48 × 64 heatmap grid.

Canonical storage:

```text
stimulus_features/dino_vits16/
├── patch_tokens.npy             # float32 [100, 768, 384]
├── cls_tokens.npy               # optional float32 [100, 384]
├── feature_manifest.csv
├── extraction_config.json
├── model_metadata.json
└── validation_report.json
```

The first axis follows `stimulus_index` in `image_manifest.csv`; it never follows
a numeric interpretation of `stimulus_id`.

Stage 1 reshapes and adapts tokens as follows:

```text
[N, 768, 384]
-> [N, 384, 24, 32]
-> DepthwiseConv2d(384, 384, kernel_size=2, stride=2)
-> Conv2d(384, 128, kernel_size=1)
-> [N, 128, 12, 16]
-> [N, 192, 128]
```

## 7. Five-fold split contract

- Split only subjects listed in the audited `All_Data` population.
- Use `StratifiedKFold(n_splits=5, shuffle=True, random_state=<saved seed>)` on
  explicit subject labels.
- Assign each subject to exactly one validation fold.
- No trial-level or stimulus-level random split is permitted.
- Missing trials remain absent and do not affect subject membership.
- Save the split seed, input-manifest SHA-256, library versions, and class counts.

For fold `k`:

```text
train subjects = all subjects whose assigned validation fold != k
val subjects   = all subjects whose assigned validation fold == k
```

Stage 1 further filters both partitions to HC only. SZ subjects remain in the CV
artifacts for future Stage 2 but are not loaded by Stage 1.

## 8. Stage-1 tensor contract

Example grouped training batch:

```text
unique stimuli per batch        S = 8
HC trials per stimulus          H = 8
trial batch size                N = S * H = 64

heatmaps                        [64, 3, 48, 64]
unique DINO patch tokens        [8, 768, 384]
trial-to-stimulus slot          [64]
heatmap patch tokens            [64, 192, 128]
adapted unique semantic tokens  [8, 192, 128]
adapted semantic tokens         [64, 192, 128]
attention-1 fused tokens        [64, 192, 128]
spatial-bridge tokens           [64, 192, 128]
attention-2 fused tokens        [64, 192, 128]
reconstruction                  [64, 3, 48, 64]
trial embedding                 [64, 128]
```

The model contains:

1. a trainable heatmap patch encoder;
2. a trainable DINO semantic adapter;
3. a serial residual semantic fusion module with two independent
   cross-attention layers separated by one residual spatial NN bridge;
4. a trainable masked-reconstruction decoder;
5. attention or mean token pooling;
6. no classifier and no VICReg projector.

The canonical fusion order is immutable unless an explicitly named ablation is
selected:

```text
heatmap tokens
-> cross-attention 1 with DINO key/value
-> residual spatial NN bridge
-> cross-attention 2 with the same DINO tensor as key/value
-> final LayerNorm
```

The two cross-attention modules must have independent parameters. They are not
parallel branches and their outputs are not concatenated or averaged.

Training loss:

\[
\mathcal{L}_{stage1}=
\mathcal{L}_{masked\_reconstruction}
+\lambda_N\mathcal{L}_{HC\_norm}.
\]

Use reconstruction-only warm-up before ramping `lambda_N` from zero to its
configured value. Setting `lambda_N=0` is an ablation.

## 9. Normative-bank contract

After selecting the best fold checkpoint, run unmasked inference on all
outer-training HC trials and compute per-stimulus statistics:

```text
mu_trial       float32 [100, 128]
sigma_trial    float32 [100, 128]
count_trial    int32   [100]
```

Optional token-level output:

```text
mu_token       float32 [100, 192, 128]
sigma_token    float32 [100, 192, 128]
```

Every bank must include fold ID, subject IDs used, model checkpoint SHA-256,
processed-manifest checksum, DINO-feature checksum, estimator, epsilon, and
minimum sample count. Never use validation HC subjects in the bank.

## 10. Leakage contract

| Artifact or operation | Allowed data |
|---|---|
| Event-to-heatmap conversion | Individual trial only |
| Fixed `log1p` transform | Individual trial only |
| Frozen pretrained DINO extraction | Stimulus image only |
| Five-fold assignment | Subject IDs and explicit labels |
| Optional channel mean/std | Current fold's training HC trials only |
| Stage-1 parameter optimization | Current fold's training HC trials only |
| Early stopping/model selection | Current fold's validation HC trials only |
| Fold normative bank | Current fold's training HC trials only |
| Future Stage-2 training | Not in scope |
