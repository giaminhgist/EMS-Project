# Phase 6 — Stage-1 trainer, ablations, checkpoints, and README

## Goal

Implement `/root/EMS-Project/stage1_trainer.py`, complete train/validation
execution, crash-safe per-epoch history, checkpoints, controlled ablations, and:

```text
/root/EMS-Project/readme/Stage1/README.md
```

## 1. Suggested files

```text
configs/stage1/base.yaml
configs/stage1/ablations/*.yaml
stage1_trainer.py
src/stage1/trainer.py
src/stage1/validation.py
src/stage1/metrics.py
src/stage1/checkpoint.py
src/stage1/history.py
src/stage1/experiment.py
src/stage1/reproducibility.py
generate_stage1_readme.py
tests/stage1/test_training_smoke.py
tests/stage1/test_history_commit.py
tests/stage1/test_checkpoint_resume.py
tests/stage1/test_best_checkpoint.py
tests/stage1/test_ablation_configs.py
tests/stage1/test_fold_isolation.py
tests/stage1/test_norm_bank_after_training.py
```

## 2. Base training configuration

Use explicit, validated configuration. Recommended starting values are design
defaults, not claimed optimal values:

```yaml
experiment_name: dino_hc_normative_stage1
seed: 2026
fold: 0

model:
  d_model: 128
  heatmap_patch_size: 4
  heatmap_residual_blocks: 2
  semantic_source: dino
  semantic_adapter: learned_depthwise_2x2
  fusion: serial_attention_spatial_attention
  attention_heads: 4
  attention_dropout: 0.1
  semantic_gamma_total_init: 0.1
  share_cross_attention_weights: false
  spatial_bridge: residual_dwconv_ffn
  spatial_bridge_expansion_ratio: 2.0
  spatial_bridge_kernel_size: 3
  spatial_bridge_dropout: 0.1
  spatial_bridge_eta_init: 0.1
  pooling: attention
  positional_encoding: fixed_2d_sincos

masking:
  train_mask_ratio: 0.35
  validation_mask_ratio: 0.35
  reconstruction_scope: masked

loss:
  reconstruction: smooth_l1
  channel_weights: [1.0, 1.0, 1.0]
  normative_metric: loo_cosine
  lambda_norm: 0.1
  norm_start_epoch: 10
  norm_ramp_epochs: 5

sampler:
  stimuli_per_batch: 8
  hc_per_stimulus: 8
  replacement: false

optimization:
  epochs: 100
  optimizer: adamw
  learning_rate: 3.0e-4
  weight_decay: 5.0e-4
  scheduler: linear_warmup_cosine
  lr_warmup_epochs: 5
  gradient_clip_norm: 5.0
  amp: true

validation:
  selection_metric: val_loss
  best_eligible_after_norm_ramp: true
  early_stopping_patience: 20

runtime:
  num_workers: 4
  pin_memory: true
  persistent_workers: true
  deterministic_validation: true
```

The implementation must derive:

```text
gamma_1_init = semantic_gamma_total_init / 2
gamma_2_init = semantic_gamma_total_init / 2
```

The base configuration requires `share_cross_attention_weights=false`.
Reject silent weight sharing or any parallel execution path; such changes need
a separately named ablation configuration.

Save the resolved configuration with every run. Reject invalid combinations,
unknown options, and unrecognized ablation names.

## 3. Training phases

### Phase A: reconstruction warm-up

For epochs before `norm_start_epoch`:

\[
\mathcal{L}=\mathcal{L}_{rec},\qquad\lambda_N=0.
\]

### Phase B: normative ramp

Increase `lambda_N` linearly from 0 to its configured value over
`norm_ramp_epochs`.

### Phase C: joint objective

After the ramp:

\[
\mathcal{L}=\mathcal{L}_{rec}+0.1\mathcal{L}_{HC-norm}
\]

for the default configuration.

Do not compare a warm-up epoch with a full-objective epoch for best-checkpoint
selection when their validation loss definitions differ. With the default
policy, mark epochs as `eligible_for_best=false` until the normative ramp is
complete.

## 4. Training loop requirements

For each epoch:

1. call `sampler.set_epoch(epoch)`;
2. set train mode only on trainable Stage-1 modules;
3. run autocast only when configured and supported;
4. zero gradients with `set_to_none=True`;
5. calculate component losses and effective `lambda_norm`;
6. detect non-finite inputs, losses, outputs, and gradients;
7. unscale gradients before calculating and clipping gradient norm;
8. update optimizer and scheduler in a documented order;
9. aggregate sample-weighted metrics rather than unweighted batch means when
   batch sizes can vary;
10. run complete deterministic HC validation;
11. determine best-checkpoint eligibility and status;
12. save last/best checkpoint atomically;
13. commit the history row immediately;
14. print a concise epoch summary containing both total and component losses.

If an epoch fails, retain the previous complete checkpoint/history and write a
clear error report. Do not append a false completed epoch.

## 5. Validation

Validation uses only HC trials from the current fold's validation subjects.

- Use fixed deterministic masks for reconstruction.
- Compute reconstruction metrics over all validation trials.
- Collect all trial embeddings and IDs.
- Group embeddings by stimulus and calculate leave-one-out normative consistency
  from all available validation HC subjects for that stimulus.
- Report group counts and skip/fail rules explicitly.
- Do not use validation embeddings to update training parameters, preprocessing
  statistics, or the final normative bank.

The canonical `val_loss` uses the same current epoch `lambda_norm` as training.
Also log `val_recon_loss` and `val_norm_loss` separately.

## 6. Per-epoch history

Required path:

```text
outputs/stage1/<run_id>/fold_<k>/history.csv
```

The CSV must be durably updated immediately after every completed epoch. Because
the file is small, rewrite the full table to a temporary file, flush and
`fsync`, then atomically replace `history.csv`. Do not wait until training ends.

Required columns:

```text
run_id
fold
epoch
training_phase
eligible_for_best
is_best_epoch
best_epoch_so_far
learning_rate
learning_rate_min
learning_rate_max
weight_decay
lambda_norm
train_loss
train_recon_loss
train_recon_fixation
train_recon_transition
train_recon_temporal
train_norm_loss
train_spread_loss
val_loss
val_recon_loss
val_recon_fixation
val_recon_transition
val_recon_temporal
val_norm_loss
val_spread_loss
train_within_stimulus_dispersion
train_between_stimulus_dispersion
val_within_stimulus_dispersion
val_between_stimulus_dispersion
grad_norm_mean
grad_norm_max
grad_clip_fraction
semantic_gamma_attention1
semantic_gamma_attention2
spatial_bridge_eta
train_mask_ratio_realized
val_mask_ratio_realized
num_train_trials
num_val_trials
n_train_batches
n_val_batches
n_train_stimulus_groups
n_val_stimulus_groups
n_skipped_norm_groups_train
n_skipped_norm_groups_val
nonfinite_batch_count
epoch_time_seconds
peak_gpu_memory_mb
seed
```

If a metric is not applicable, store an empty/NaN field with a documented
meaning rather than changing columns between runs. Validate uniqueness of
`(fold, epoch)` and monotonic epoch order after every write.

Optional TensorBoard or JSONL logging may be added, but `history.csv` remains the
canonical required artifact.

## 7. Output structure

```text
outputs/stage1/<run_id>/
├── run_metadata.json
├── config_resolved.yaml
├── environment.json
├── source_checksums.json
├── fold_0/
│   ├── history.csv
│   ├── train.log
│   ├── checkpoints/
│   │   ├── best_stage1_fold0.pt
│   │   └── last_stage1_fold0.pt
│   ├── validation/
│   │   ├── best_val_embeddings.npz
│   │   └── metrics.json
│   └── normative_bank/
│       ├── mu_trial.npy
│       ├── sigma_trial.npy
│       ├── count_trial.npy
│       ├── feature_manifest.csv
│       └── metadata.json
└── fold_4/
    └── ...
```

Do not overwrite a completed run. New configurations receive new deterministic
or timestamped run IDs that include an ablation name.

## 8. Checkpoints and resume

Every last checkpoint must include:

```text
model state
optimizer state
scheduler state
AMP scaler state
epoch and global step
best metric and best epoch
resolved config and config hash
fold and run ID
processed/CV/DINO manifest checksums
Python/NumPy/PyTorch CPU and CUDA RNG states
sampler state or sufficient deterministic epoch state
channel-normalizer state when enabled
```

Required CLI semantics:

```bash
# Resume exactly, including optimizer and scheduler
python stage1_trainer.py --resume \
  outputs/stage1/<run>/fold_0/checkpoints/last_stage1_fold0.pt

# Initialize model weights only; fresh optimizer/scheduler/run
python stage1_trainer.py --config configs/stage1/base.yaml --fold 0 \
  --load-stage1-weights \
  outputs/stage1/<run>/fold_0/checkpoints/best_stage1_fold0.pt
```

Resume must reject fold, architecture, manifest, or resolved-config mismatch
unless a narrowly defined safe override is explicitly approved. Weight-only load
must report missing/unexpected keys and require exact architecture by default.

## 9. Trainer CLI

Provide:

```bash
# One fold
python stage1_trainer.py --config configs/stage1/base.yaml --fold 0

# All folds sequentially with isolated outputs
python stage1_trainer.py --config configs/stage1/base.yaml --fold all

# Smoke test
python stage1_trainer.py --config configs/stage1/base.yaml --fold 0 \
  --max-epochs 2 --max-train-batches 3 --max-val-batches 3 \
  --run-name smoke

# Verify inputs and tensor contracts without optimization
python stage1_trainer.py --config configs/stage1/base.yaml --fold 0 --dry-run

# Rebuild fold norm bank from best checkpoint
python stage1_trainer.py --build-norm-bank \
  --checkpoint outputs/stage1/<run>/fold_0/checkpoints/best_stage1_fold0.pt
```

CLI overrides must be saved into the resolved configuration. Do not allow a CLI
flag to alter behavior without recording it.

## 10. Controlled ablations

Implement named configuration files rather than scattered booleans. Every
ablation changes one primary factor relative to `base.yaml` unless explicitly
labeled factorial.

Minimum required ablations:

| Name | Change | Scientific question |
|---|---|---|
| `no_semantic` | Bypass DINO and both cross-attentions while retaining the spatial bridge | Does stimulus semantics help? |
| `aligned_add_fusion` | Replace both attention injections with independently projected aligned additions while retaining the bridge | Is dynamic cross-attention better than position-aligned addition? |
| `concat_fusion` | Replace both attention injections with position-aligned concatenation/projection while retaining the bridge | Is attention necessary beyond learned dense fusion? |
| `single_cross_attention` | Remove Attention 2 but retain Attention 1 and the spatial bridge | Does semantic re-attention add value after spatial contextualization? |
| `no_spatial_bridge` | Replace the middle spatial bridge with identity | Is the NN bridge necessary between the two attentions? |
| `token_mlp_bridge` | Replace the depthwise-convolution bridge with a token-wise MLP of matched width | Does explicit spatial mixing add value? |
| `no_fusion_residual` | Remove the residual paths around both cross-attentions | Do residual paths preserve gaze information and stabilize training? |
| `fixed_fusion_gates` | Fix `gamma_1=gamma_2=0.05` and `eta=0.1` | Does learning the three residual gates help beyond their initial scales? |
| `mean_pooling` | Replace attention pooling with mean pooling | Are specific spatial tokens more informative? |
| `no_norm_loss` | Set `lambda_norm=0` | Does HC normative consistency improve the representation? |
| `full_reconstruction` | No token mask; reconstruct the full map | Does masking force better semantic usage? |
| `fixation_only` | Use only channel 0; zero/drop other channels via declared adapter | Do transitions and temporal order add value? |
| `no_transition_channel` | Remove channel 1 | Contribution of consecutive transitions |
| `no_temporal_channel` | Remove channel 2 | Contribution of temporal progression |
| `avgpool_semantic_adapter` | Fixed 2×2 pooling plus 1×1 projection | Does learned spatial aggregation help? |

All variants must preserve a common output interface and produce a complete
resolved config/history. The input channel count may change only through a
clearly implemented model config; never silently leave removed channels in use.

Invalid or misleading combinations must fail validation. In particular, do not
allow masked-only loss with a zero mask ratio, and do not run a norm-only model
without reconstruction unless the user explicitly requests an unsafe collapse
control experiment.

## 11. Diagnostic metrics for semantic usage and collapse

Log or calculate during validation:

- learned `gamma_1`, bridge `eta`, and `gamma_2` residual gates;
- Attention-1 and Attention-2 entropy on a fixed validation diagnostic subset;
- Jensen-Shannon divergence and cosine similarity between the head-averaged
  Attention-1 and Attention-2 maps as descriptive redundancy diagnostics,
  without assuming that either high or low disagreement is inherently better;
- residual-output norm ratios for Attention 1, the spatial bridge, and
  Attention 2;
- within-stimulus HC embedding dispersion;
- between-stimulus centroid dispersion;
- per-dimension embedding standard deviation;
- reconstruction difference after shuffling DINO stimulus features across
  trials as an evaluation-only semantic-dependence test.

The shuffled-DINO test must not update weights. If shuffling has no effect, flag
that the model may be ignoring semantic information rather than claiming success.

## 12. Stage-1 README

Generate `/root/EMS-Project/readme/Stage1/README.md` with:

1. research objective and Stage-1-only scope;
2. HC-only and no-VICReg statement;
3. input and artifact dependencies;
4. DINO ViT-S/16 extraction and exact spatial alignment;
5. architecture diagram and tensor-shape table showing the exact serial order
   `Attention 1 -> spatial NN bridge -> Attention 2`, independent attention
   parameters, shared DINO key/value tensor, and all three residual gates;
6. masked reconstruction and leave-one-out cosine normative loss equations;
7. training phases and best-checkpoint policy;
8. CV/leakage controls;
9. normative-bank definition and output schema;
10. base training, smoke test, all-fold, resume, weight-load, norm-bank, and
    ablation commands;
11. output directory and `history.csv` schema;
12. troubleshooting for OOM, missing IDs, checksum mismatch, resume mismatch,
    non-finite loss, insufficient HC groups, and failed DINO semantic-use test;
13. explicit statement that Stage 2 classification is future work.

The README must match implemented defaults. Generate architecture parameter
counts and example tensor shapes from a real dry-run rather than typing values
that can drift from code.

## 13. Required tests

- Two-epoch synthetic smoke training yields finite train/validation losses.
- `history.csv` exists after epoch 1, before training completes.
- Every completed epoch adds exactly one valid history row.
- Simulated interruption leaves the previous history and last checkpoint valid.
- Resume reproduces the next epoch's sampling and scheduler state within the
  documented determinism level.
- Best checkpoint is not selected before full-objective eligibility by default.
- `is_best_epoch` and `best_epoch_so_far` are consistent.
- Best and last checkpoints have the required names per fold.
- Weight-only loading does not restore optimizer state.
- Fold 0 cannot load fold 1 split artifacts silently.
- Every ablation config parses, changes the intended field, and preserves all
  other base fields.
- Every supported fusion/pooling/channel ablation passes a forward/backward test.
- The default fusion executes Attention 1, the spatial bridge, and Attention 2
  in that order; Attention 2 consumes bridge output, not the original heatmap
  tokens.
- Attention 1 and Attention 2 have independent parameters and both receive
  finite gradients.
- The single-attention ablation retains the spatial bridge and preserves the
  default output interface.
- Stage-1 train and validation datasets contain HC only.
- Norm-bank generation excludes validation subject IDs.
- README commands pass parser/dry-run validation.

## 14. Acceptance criteria

- One real fold completes a short smoke run.
- History is saved after every epoch and includes all required columns.
- Best/last checkpoints and exact resume work.
- The best checkpoint builds a fold-safe HC norm bank.
- Named ablations are executable and traceable.
- Stage-1 README accurately describes code and commands.
- No VICReg, contrastive, SZ classification, or Stage-2 logic is introduced.

## 15. Gate

Finish with the standard report and ask:

> Phase 6 trainer, ablations, checkpoints, and Stage-1 README are complete.
> Would you like to change any training behavior or documentation, or should I
> continue to Phase 7 end-to-end QA and handoff?

Then stop.
