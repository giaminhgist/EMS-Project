# Phase 7 — End-to-end QA and handoff

## Goal

Verify that all phases form one reproducible and leakage-safe pipeline. Make only
small fixes necessary to satisfy existing contracts. Any architectural or data
contract change requires a return to the relevant earlier phase and user
approval.

## 1. Full verification sequence

Run from `/root/EMS-Project` using the actual repository environment.

### 1.1 Static and unit checks

Run the configured formatter/linter/type checker if present, followed by all
unit tests. At minimum:

```bash
pytest -q tests/preprocessing
pytest -q tests/stimulus_features
pytest -q tests/cv
pytest -q tests/stage1
```

Do not hide warnings by globally disabling them. Resolve actionable numerical,
data-type, deprecation, and resource warnings or document why they are safe.

### 1.2 Artifact verification modes

```bash
python preprocess_ems.py --config configs/preprocessing.yaml --dry-run

python generate_preprocessed_readme.py \
  --processed-root /root/EMS-Project/processed_dataset \
  --output-dir /root/EMS-Project/readme/preprocessed \
  --verify-only

python extract_dino_features.py \
  --config configs/dino_vits16.yaml \
  --image-manifest /root/EMS-Project/processed_dataset/image_manifest.csv \
  --output-root /root/EMS-Project/stimulus_features \
  --verify-only

python build_cv.py \
  --config configs/cv_5fold.yaml \
  --subject-manifest /root/EMS-Project/processed_dataset/subject_manifest.csv \
  --trial-manifest /root/EMS-Project/processed_dataset/trial_manifest.parquet \
  --output-root /root/EMS-Project/CV \
  --verify-only

python stage1_trainer.py --config configs/stage1/base.yaml --fold 0 --dry-run
```

Adjust only filenames explicitly approved in prior phases.

### 1.3 Training smoke checks

Run at least:

```bash
python stage1_trainer.py --config configs/stage1/base.yaml --fold 0 \
  --max-epochs 2 --max-train-batches 3 --max-val-batches 3 \
  --run-name final_smoke
```

Then verify resume for one additional epoch from the generated last checkpoint.

Do not run a long multi-fold experiment merely to finish QA unless the user asks.
Perform `--dry-run` on all five folds and verify disjoint subjects, input shapes,
eligible sampler groups, and writable isolated output paths.

## 2. Cross-artifact consistency matrix

Verify and record:

| Check | Required result |
|---|---|
| Raw source checksums | Unchanged since Phase 1 inventory |
| Subject IDs | Same canonical strings in processed and CV manifests |
| Stimulus IDs | Same canonical mapping in image, trial, and DINO manifests |
| Trial keys | Unique and resolvable to exact per-subject array rows |
| Heatmaps | `[3,48,64]`, float32, finite |
| DINO input | `[3,384,512]`, deterministic full-image resize with no crop/pad/flip |
| DINO tokens | `[768,384]`, float32, finite; patch count × embedding dimension |
| DINO rows | Exactly one per stimulus index |
| CV validation folds | Mutually disjoint; union equals All_Data subjects |
| Fold partitions | No subject overlap within any train/validation pair |
| Stage-1 groups | HC only in train and validation |
| Training shapes | Match `[N,192,128]` fusion contract |
| Fusion order | Exactly Attention 1 -> spatial NN bridge -> Attention 2; no parallel branch |
| Attention parameters | Attention 1 and 2 are independent; both receive gradients |
| Attention-2 query | Comes from spatial-bridge output, not original heatmap tokens |
| DINO gradient | Absent |
| History | One durable row per completed epoch |
| Checkpoints | Best and last independently loadable |
| Norm bank | Only outer-training HC contributors |
| Missing trials | Remain absent throughout pipeline |

## 3. Reproducibility checks

- Repeat a small preprocessing fixture and compare numeric arrays exactly.
- Repeat DINO extraction for selected images and compare within recorded
  tolerance.
- Rebuild CV with the same seed and verify identical assignment checksum.
- Run two deterministic CPU mini-training tests and compare histories.
- For GPU training, state which operations may remain nondeterministic and the
  observed tolerance; do not promise bitwise identity if it is not achieved.
- Verify resumed epoch order, learning rate, sampler selection, and history are
  consistent with uninterrupted training.

## 4. Leakage audit

Inspect code paths, not only output files. Confirm:

- no preprocessing function accepts diagnosis-group population statistics;
- no fold normalizer is fitted on validation data;
- Stage-1 dataset filters HC before sampling;
- model-selection metrics use validation HC but never update the model;
- normative-bank builder reads only `train_subjects.csv` and asserts exclusion
  of validation IDs;
- DINO extraction uses only images and a pretrained frozen checkpoint;
- no SZ label, BCE, classification accuracy, AUC, VICReg, SupCon, or Stage-2
  module appears in Stage-1 optimization.

Search for suspicious code and explain every legitimate occurrence of terms
such as `label`, `val`, `mean`, `std`, `fit`, `norm`, `SZ`, and `classifier`.

## 5. Resource and robustness report

Measure during the final smoke run:

- preprocessing wall time and generated size;
- DINO feature size and extraction time;
- Stage-1 trainable parameter count;
- batch CPU/GPU memory;
- average train and validation epoch time;
- data-loader worker behavior;
- any I/O bottleneck from per-subject memory maps;
- behavior with `num_workers=0` for debugging;
- behavior when one manifest checksum is deliberately mismatched.

Report measurements as environment-specific observations, not universal
benchmarks.

## 6. Documentation verification

Confirm both exist and match code:

```text
/root/EMS-Project/readme/preprocessed/README.md
/root/EMS-Project/readme/Stage1/README.md
```

Run every documented command in `--help`, `--dry-run`, `--verify-only`, or smoke
mode as appropriate. Verify relative links and figures. Remove stale options and
update tensor tables from the implemented model.

## 7. Final implementation status

Create or update a concise repository document such as:

```text
/root/EMS-Project/IMPLEMENTATION_STATUS.md
```

Include:

- completed phases and dates;
- canonical commands;
- artifact locations and checksums;
- test summary;
- approved deviations from this guide;
- unresolved limitations;
- recommended next experiments;
- explicit statement that Stage 2 remains unimplemented.

Do not include full training claims from a smoke run.

## 8. Final acceptance criteria

- All required unit and integration tests pass.
- Canonical artifacts validate against each other.
- A two-epoch real-data smoke run and one-epoch resume succeed.
- All five folds pass dry-run and leakage checks.
- History, checkpoints, and norm bank are recoverable and traceable.
- Documentation commands and schemas are accurate.
- Raw EMS data remain unchanged.
- No out-of-scope Stage-2 code was introduced.

## 9. Final gate

Finish with the standard report and ask:

> Phase 7 end-to-end QA is complete. Would you like any corrective changes, or
> should I consider the EMS preprocessing, DINO, five-fold CV, and Stage-1
> implementation ready for full experiments?

Then stop and wait for the user's decision.
