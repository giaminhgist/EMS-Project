# EMS Stage 2 — Claude Code master workflow

Use this guide set to implement Stage 2 inside the existing repository at:

```text
/root/EMS-Project
```

The implementation classifies subjects as Healthy Control (HC) or Schizophrenia (SZ) by comparing each subject's heatmap response with a fold-specific HC normative bank for the same stimulus.

## 1. How Claude Code must use this guide set

Run exactly one implementation phase at a time.

Before every phase, Claude Code must read:

1. this file;
2. `01_STAGE2_CONTRACTS.md`;
3. the requested phase file.

The phase order is immutable:

| Phase | Guide | Scope |
|---:|---|---|
| 0 | `02_PHASE0_AUDIT_AND_DECISIONS.md` | Read-only repository and artifact audit |
| 1 | `03_PHASE1_BUILD_5FOLD_NORMATIVE_BANKS.md` | Build five fold-specific HC banks |
| 2 | `04_PHASE2_DATASET_DATALOADER.md` | Subject-level dataset, bank loader, sampler and collator |
| 3 | `05_PHASE3_MODEL_AND_LOSSES.md` | Stage-2 architecture and objectives |
| 4 | `06_PHASE4_ABLATION_FRAMEWORK.md` | Configuration and ablation system |
| 5 | `07_PHASE5_TRAINER_VALIDATION_LOGGING.md` | Root CLI, training, validation, history and checkpoints |
| 6 | `08_PHASE6_README_QA_HANDOFF.md` | Stage-2 README, end-to-end QA and handoff |

Do not start Phase `n+1` until the user explicitly approves continuing after Phase `n`.

### Copy-paste prompt to start Claude Code

Place this guide directory somewhere Claude Code can read, then start with:

```text
Work in /root/EMS-Project. Read 00_MASTER_WORKFLOW.md,
01_STAGE2_CONTRACTS.md and 02_PHASE0_AUDIT_AND_DECISIONS.md in full.
Perform Phase 0 only. Do not make implementation changes during Phase 0.
At the end, use the required completion report, ask all Phase-0 decisions,
then stop and wait for my explicit approval before Phase 1.
```

For every later phase, ask Claude Code to read the master, contract and that phase file in full. Do not paste all phase prompts at once because the required user gates are part of the implementation protocol.

## 2. Mandatory stop gate after every phase

After completing a phase, Claude Code must:

1. stop making repository changes;
2. provide the phase completion report defined below;
3. ask the phase-specific continuation question;
4. wait for the user's answer.

Approval to implement Stage 2 as a whole is not approval to silently run all phases.

Never combine multiple phase completion reports.

## 3. Standard phase completion report

Use this exact structure:

```markdown
## Phase <N> completion report

### Implemented
- ...

### Files created or changed
- `<path>`: purpose

### Verification performed
- `<command>`
  - Result: ...

### Tensor and data contracts verified
- ...

### Deviations or unresolved items
- None
  OR
- ...

### Next gate
Phase <N> is complete. Would you like to inspect or change anything, or should I continue to Phase <N+1>?
```

If a test fails, an input is missing, or a contract cannot be met, report the blocker and stop. Do not mark the phase complete.

## 4. Repository mutation rules

Before writing anything:

- inspect `git status --short`;
- preserve every existing user change;
- do not reset, checkout, clean, stash, delete or overwrite unrelated work;
- do not modify `original_dataset/`;
- do not regenerate Stage-1 artifacts unless the current phase explicitly requires it;
- do not push, merge or create a pull request unless the user separately requests it.

Use existing repository utilities and conventions when they satisfy the Stage-2 contract. Do not create duplicate preprocessing, CV or Stage-1 implementations.

## 5. Fixed project paths

Canonical paths:

```text
REPO_ROOT             /root/EMS-Project
PROCESSED_ROOT        /root/EMS-Project/processed_dataset
DINO_ROOT             /root/EMS-Project/stimulus_features/dino_vits16
CV_ROOT               /root/EMS-Project/CV/5fold_seed2026
STAGE1_OUTPUT_ROOT    /root/EMS-Project/outputs/stage1
NORM_BANK_ROOT        /root/EMS-Project/normative_bank
STAGE2_OUTPUT_ROOT    /root/EMS-Project/outputs/stage2
STAGE2_SOURCE_ROOT    /root/EMS-Project/src/stage2
STAGE2_README         /root/EMS-Project/readme/Stage2/README.md
```

If the audit finds different real paths, report them in Phase 0 and request approval before changing these contracts.

## 6. Target repository layout

Adapt small filename details to existing conventions, but preserve these responsibilities:

```text
/root/EMS-Project/
├── build_stage2_normative_banks.py
├── stage2_trainer.py
├── generate_stage2_readme.py
├── configs/
│   └── stage2/
│       ├── base.yaml
│       ├── stage1_checkpoints.yaml
│       └── ablations/
│           └── *.yaml
├── src/
│   └── stage2/
│       ├── __init__.py
│       ├── config.py
│       ├── contracts.py
│       ├── bank_builder.py
│       ├── bank.py
│       ├── dataset.py
│       ├── sampler.py
│       ├── collate.py
│       ├── transferred_encoder.py
│       ├── pooling.py
│       ├── relation.py
│       ├── token_attention.py
│       ├── subject_aggregation.py
│       ├── model.py
│       ├── losses.py
│       ├── metrics.py
│       ├── checkpoint.py
│       ├── history.py
│       ├── validation.py
│       └── trainer.py
├── tests/
│   └── stage2/
│       └── test_*.py
├── normative_bank/
│   ├── manifest.json
│   └── fold_0/ ... fold_4/
├── outputs/
│   └── stage2/
└── readme/
    └── Stage2/
        └── README.md
```

It is acceptable to combine very small modules when this matches the current repository style. It is not acceptable to place Stage-2 implementation inside `src/stage1/`.

## 7. Scientific invariants

The following are hard requirements:

1. The independent diagnostic sample is the subject, not the trial.
2. HC/SZ labels are never assigned as independent targets to subject–stimulus trials.
3. Every batch contains complete or masked subject panels.
4. Missing trials remain missing and are excluded from every softmax and loss.
5. Every normative comparison uses the bank entry for the correct `stimulus_index` unless a named wrong-bank ablation is active.
6. Every fold bank uses HC subjects from that fold's training partition only.
7. No validation subject contributes to the full fold bank.
8. The transferred Stage-1 heatmap encoder is frozen in the base model.
9. Stage 2 does not run DINO or Stage-1 semantic fusion for a query subject.
10. Stage-2 bank tensors always have `requires_grad=False`.
11. Attention weights are model attributions, not automatically explanations.
12. All evaluation, bootstrap and confidence intervals use subjects as the unit.

## 8. Stage-1 reuse boundary

Stage 2 may reuse:

- `src/stage1/heatmap_encoder.py` architecture;
- `heatmap_encoder.*` weights from the approved fold checkpoint;
- the fold-specific bank generated by full Stage-1 unmasked inference.

The Stage-2 query path must not reuse:

- DINO patch tokens;
- Stage-1 semantic adapter;
- Stage-1 semantic fusion;
- Stage-1 decoder;
- Stage-1 trial pooling head;
- cached Stage-1 embeddings of a query HC or SZ subject.

The bank-building command is allowed to load the complete Stage-1 model because the bank itself is a precomputed Stage-1 artifact.

## 9. No silent design changes

Stop and ask the user before changing any of the following:

- heatmap shape `[3,48,64]`;
- Stage-1 token grid `12×16` and token count `192`;
- hidden dimension `128`;
- five subject-level folds;
- HC=`0`, SZ=`1` label convention;
- root `normative_bank/` location;
- root `stage2_trainer.py` location;
- `src/stage2/` package boundary;
- subject-level batching;
- base-model frozen encoder policy;
- phase-gated workflow.

## 10. Implementation quality rules

- Use type hints for public functions and dataclasses for batch/output contracts.
- Validate shapes, dtypes, IDs, checksums and fold identity at boundaries.
- Use row-major token ordering for every `12×16 -> 192` conversion.
- Prefer stable vectorized operations and float64 accumulation for bank statistics.
- Use atomic writes for bank arrays, metadata, history, metrics and checkpoints.
- Set seeds for Python, NumPy, PyTorch and DataLoader workers.
- Validation must be deterministic.
- Unit tests must use small synthetic fixtures and must not require the full EMS dataset.
- Smoke tests may use a few real subjects only when Phase 0 confirms paths.
- No placeholder `pass`, TODO-only implementation or fake metric is accepted as phase completion.

## 11. Failure behavior

Fail with a clear error rather than silently continuing when:

- a Stage-1 fold checkpoint is missing or ambiguous;
- checkpoint fold/config does not match the requested bank fold;
- a bank contains a forbidden subject;
- bank stimulus IDs do not match processed/DINO manifests;
- expected tensor shapes differ;
- a subject appears in both train and validation partitions;
- a subject label is missing or inconsistent;
- history and checkpoint resume state disagree;
- a non-finite loss or gradient occurs;
- an ablation changes more factors than its declared overlay.

## 12. Final deliverables

The completed repository must provide:

- five audited HC normative banks under `normative_bank/fold_0` through `fold_4`;
- Stage-2 package under `src/stage2/`;
- `stage2_trainer.py` at repository root;
- configuration and named ablation overlays;
- durable per-epoch histories under `outputs/stage2/`;
- best and last checkpoints with exact resume;
- subject-level validation predictions and metrics;
- interpretation outputs for stimulus importance and normative deviation;
- optional token-level semantic maps;
- generated `readme/Stage2/README.md`;
- unit, integration and smoke tests;
- a final handoff report listing commands and known limitations.
