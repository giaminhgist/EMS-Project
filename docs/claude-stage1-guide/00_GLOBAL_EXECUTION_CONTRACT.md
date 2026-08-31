# Global execution contract

## 1. Objective

Build a reproducible, fold-safe pipeline that:

1. converts fixation-event records into one three-channel heatmap tensor for
   each observed `(subject_id, stimulus_id)` trial;
2. stores data through explicit manifests so non-contiguous identifiers are
   never used as array positions;
3. extracts spatial patch tokens from a frozen pretrained DINO model;
4. creates five subject-level stratified folds from `All_Data`;
5. trains and validates an HC-only Stage-1 semantic-conditioned normative
   encoder;
6. writes crash-safe epoch histories, checkpoints, and fold-specific normative
   banks;
7. exposes controlled Stage-1 ablations;
8. generates method and usage documentation from the implemented code and
   resolved configurations.

## 2. Immutable paths

Unless the user explicitly approves a change, use:

```text
PROJECT_ROOT=/root/EMS-Project
RAW_ROOT=/root/EMS-Project/original_dataset
PROCESSED_ROOT=/root/EMS-Project/processed_dataset
STIMULUS_FEATURE_ROOT=/root/EMS-Project/stimulus_features
CV_ROOT=/root/EMS-Project/CV
OUTPUT_ROOT=/root/EMS-Project/outputs
PREPROCESSED_README=/root/EMS-Project/readme/preprocessed/README.md
STAGE1_README=/root/EMS-Project/readme/Stage1/README.md
TRAINER=/root/EMS-Project/stage1_trainer.py
```

`RAW_ROOT` is read-only. Generated files must never be placed inside it.

## 3. Phase-gated workflow

| Phase | Scope | May mutate repository? |
|---:|---|---|
| 0 | Audit repository, raw data, environment, and unresolved decisions | No |
| 1 | Preprocessing code, manifests, three-channel heatmaps, QC | Yes |
| 2 | EDA and processed-dataset README | Yes |
| 3 | Frozen DINO feature extraction | Yes |
| 4 | Five-fold subject-level split generation | Yes |
| 5 | Stage-1 dataset, sampler, model, losses, and normative bank | Yes |
| 6 | Trainer, validation, logging, checkpointing, ablations, Stage-1 README | Yes |
| 7 | End-to-end tests, reproducibility audit, and handoff | Only fixes in scope |

Claude Code must not combine phase completion reports. It must stop after each
phase and wait for user approval.

## 4. Required phase-end report

Use this exact structure:

```markdown
## Phase <N> completion report

### Implemented
- ...

### Files changed
- `path`: purpose

### Verification performed
- `command`
  - Result: ...

### Data or scientific checks
- ...

### Open decisions or discrepancies
- ...

### Gate
Phase <N> is complete. Would you like to modify anything in this phase, or
should I continue to Phase <N+1>?
```

After printing the gate, stop. Do not start the next phase in the same response.

## 5. Engineering rules

- Inspect existing files before editing and preserve unrelated user changes.
- Prefer small modules with typed public interfaces and explicit dataclasses or
  configuration schemas.
- Use repository-root-relative imports and commands. Do not rely on the shell's
  current working directory after startup.
- Use `pathlib.Path`; validate every configured path.
- Use a single seeded utility for Python, NumPy, PyTorch, CUDA, data-loader
  workers, samplers, and deterministic validation masks.
- Never use Python's process-randomized `hash()` for persistent IDs or seeds.
  Use a stable SHA-256-derived integer instead.
- Write large generated arrays through temporary files followed by atomic
  `os.replace` after validation.
- Write small metadata atomically as well. A process interruption must not leave
  a file that looks complete but is truncated.
- Store resolved configuration, software versions, Git commit when available,
  source-manifest checksum, and random seed with every generated artifact.
- A `--force` flag may replace a generated artifact only after target validation;
  normal execution should refuse incompatible existing outputs.
- A `--resume` mode must verify configuration and manifest compatibility before
  continuing.
- Unit tests must use synthetic fixtures and must not require the full EMS data.
- Integration and smoke tests may use a small selected subset of the real data.

## 6. Scientific rules

- The independent unit for cross-validation is the subject, not the trial.
- Clinical labels are subject-level; no subject may appear in both train and
  validation partitions of a fold.
- Missing subject-stimulus trials remain missing. Never create fake zero trials.
- Raw fixation order must be established with `FIX_INDEX`, not assumed from row
  order.
- Off-canvas, short-duration, duplicate, and malformed events must be counted in
  QC before applying an approved policy.
- Preprocessing must not calculate group means, fold means, PCA, global z-score,
  or HC norms.
- Frozen DINO features may be extracted once for all folds because no EMS subject
  information is used to fit DINO.
- The DINO adapter, heatmap encoder, serial
  `cross-attention 1 -> spatial NN bridge -> cross-attention 2` fusion module,
  pooling layer, and decoder are learned separately inside each outer fold.
- Stage 1 uses only HC training trials for optimization and HC validation trials
  for model selection.
- A fold's normative bank uses only its outer-training HC subjects.
- Every result must identify whether it is an observation, implementation
  decision, or unverified hypothesis.

## 7. No silent scope expansion

Do not implement Stage 2, SZ/HC classification, VICReg, SupCon, external test-set
evaluation, hyperparameter search, cloud uploads, or Git pushes unless the user
adds that scope explicitly.
