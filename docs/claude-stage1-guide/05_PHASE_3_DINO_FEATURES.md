# Phase 3 — Frozen DINO stimulus feature extraction

## Goal

Extract one spatial patch-token tensor for each unique stimulus using the
user-approved frozen pretrained DINO model and save it under:

```text
/root/EMS-Project/stimulus_features
```

Do not train or fine-tune DINO. Do not implement the trainable semantic adapter
in this phase; it belongs to Stage 1 and must be fold-specific.

## 1. Suggested files

```text
configs/dino_vits16.yaml
extract_dino_features.py
src/stimulus_extraction/__init__.py
src/stimulus_extraction/config.py
src/stimulus_extraction/dino_extractor.py
src/stimulus_extraction/storage.py
src/stimulus_extraction/validate.py
tests/stimulus_features/test_dino_preprocessing.py
tests/stimulus_features/test_dino_token_shapes.py
tests/stimulus_features/test_feature_id_mapping.py
tests/stimulus_features/test_feature_reproducibility.py
```

## 2. Default resolved configuration

After explicit Phase-0 confirmation, the default is:

```yaml
model_family: dino
model_name: dino_vits16
pretrained: true
frozen: true
input_height: 384
input_width: 512
resize_mode: exact
interpolation: bicubic
antialias: true
center_crop: false
random_augmentation: false
output_layer: final_normalized_patch_tokens
include_cls_token: false
output_dtype: float32
batch_size: 1
num_workers: 4
device: cuda
seed: 2026
```

Record the actual model source, resolved code revision when available,
checkpoint cache path or identifier, checkpoint SHA-256 when accessible,
preprocessing mean/std, PyTorch version, and extraction device. Do not claim a
revision is pinned unless it actually is.

If weights cannot be downloaded, stop and report the blocker. Do not silently
substitute a random model, a different DINO version, or a supervised ViT.

## 3. Image preprocessing

Resolve stimuli only through `processed_dataset/image_manifest.csv`.

For every stimulus:

1. verify its current SHA-256 matches the manifest;
2. open and convert to RGB;
3. verify the audited source size, normally 768 pixels high × 1024 pixels wide;
4. resize the complete image deterministically to 384 pixels high × 512 pixels
   wide using the configured interpolation and antialias settings;
5. do not crop, pad, flip, rotate, or randomly augment;
6. convert to float tensor;
7. apply the official normalization expected by the chosen pretrained DINO
   checkpoint.

The expected source size is 1024×768 in W×H image notation. The DINO input is
512×384 in W×H notation and `[3,384,512]` in C×H×W tensor notation. The resize
preserves the 4:3 aspect ratio and therefore preserves normalized positions
relative to the 48×64 heatmap. If any audited source stimulus differs, do not
warp it silently; report and request a policy.

## 4. Patch-token extraction

Use:

```python
model.eval()
for parameter in model.parameters():
    parameter.requires_grad_(False)
```

Run under `torch.inference_mode()`.

For DINO ViT-S/16:

```text
input                         [B, 3, 384, 512]
patch size                    16 × 16
patch grid                    24 × 32
tokens including CLS          [B, 769, 384]
stored patch tokens           [B, 768, 384]
```

Use the model API that exposes final normalized intermediate tokens. A normal
classification-style `model(image)` call may return only a global/CLS embedding
and is not sufficient. Encapsulate API-specific extraction in a wrapper and
assert the exact token count and dimension. The wrapper must use the selected
checkpoint's native positional-embedding interpolation for the non-square
24×32 grid; do not manually truncate, tile, or reshape positional embeddings.

Preserve row-major spatial patch order. Verify it by reshaping:

```python
tokens.reshape(batch, 24, 32, 384)
```

Do not average patch tokens and do not apply the Stage-1 384→128 adapter here.

## 5. Storage

Canonical output:

```text
stimulus_features/dino_vits16/
├── patch_tokens.npy
├── feature_manifest.csv
├── extraction_config.json
├── model_metadata.json
└── validation_report.json
```

`patch_tokens.npy`:

```text
dtype: float32
shape: [n_stimuli, 768, 384]
axis 0: exact stimulus_index order from image_manifest.csv
```

Minimum `feature_manifest.csv` columns:

```text
stimulus_index, stimulus_id, feature_row_index, image_sha256,
feature_shape, feature_dtype, feature_sha256
```

The writer must use a temporary array or memory map, validate every completed
row, and publish atomically. `--resume` must verify completed-row checksums and
the exact model/config identity before continuing.

## 6. CLI

Provide:

```bash
python extract_dino_features.py \
  --config configs/dino_vits16.yaml \
  --image-manifest /root/EMS-Project/processed_dataset/image_manifest.csv \
  --output-root /root/EMS-Project/stimulus_features
```

Also support:

```text
--device cpu|cuda
--batch-size N
--num-workers N
--resume
--force
--verify-only
--stimulus-limit N       # smoke-test output only, never canonical output
```

## 7. Required tests and validation

- Image preprocessing deterministically produces `[3,384,512]` without crop,
  padding, flip, or rotation, and records interpolation/antialias settings.
- Model parameters have `requires_grad=False` and remain unchanged.
- Patch-token shape is exactly `[768,384]` per stimulus. Here 768 is the
  number of spatial patches and 384 is the embedding dimension; neither value
  is the stimulus width or height.
- Positional embeddings are interpolated by the model's supported API to the
  24×32 grid, and a non-square real-image smoke test succeeds.
- No NaN or infinity is present.
- Feature variance across tokens and across stimuli is nonzero.
- Manifest mapping round-trips `stimulus_id -> stimulus_index -> feature row`.
- Stimulus order in the image manifest does not change lookup correctness.
- Repeated CPU extraction of the same image is numerically equal within a stated
  tolerance; GPU reproducibility tolerance is recorded.
- A deliberately changed image checksum is detected by `--verify-only`.
- A real-data smoke test extracts at least two different categories and confirms
  their tensors are not identical.

## 8. Acceptance criteria

- Exactly one valid feature row exists for every image-manifest row.
- Output shapes, dtype, mapping, checksums, and metadata satisfy the contract.
- DINO was pretrained, frozen, and run in evaluation/inference mode.
- No subject gaze or diagnostic label was used.
- No trainable Stage-1 adapter was precomputed.

## 9. Gate

Finish with the standard report and ask:

> Phase 3 frozen DINO feature extraction is complete. Would you like to change
> the DINO checkpoint or feature contract, or should I continue to Phase 4
> five-fold subject-level cross-validation?

Then stop.
