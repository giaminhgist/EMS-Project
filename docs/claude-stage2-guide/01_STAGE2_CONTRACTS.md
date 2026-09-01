# EMS Stage 2 — stable data, model, loss and artifact contracts

This file is authoritative for every Stage-2 implementation phase. If the repository audit makes a contract impossible, stop and ask the user before revising it.

## 1. Research objective

For subject `i` and stimulus `s`, Stage 2:

1. encodes the three-channel subject heatmap with the transferred Stage-1 heatmap encoder;
2. retrieves the HC normative statistics for the same stimulus;
3. learns an aligned query–bank comparison;
4. creates one trial deviation token;
5. assigns category-balanced importance across stimuli;
6. aggregates to one subject representation;
7. predicts HC versus SZ at subject level.

The base model must also return:

- per-stimulus importance;
- per-stimulus signed contribution;
- learned semantic compatibility with the HC bank;
- reliability-weighted normative deviation.

The optional token-bank model additionally returns one `12×16` semantic-deviation map per observed trial.

## 2. Fixed dataset facts

```text
subjects                    160 = 80 HC + 80 SZ
stimuli                     100
observed trials             15,912
heatmap                     float32 [3,48,64]
stimulus categories         4
diagnostic target           subject-level binary label
```

Stimulus category counts:

```text
Manipulated Images          32
Natural Scenes              31
Social Scenes               22
Synthetic Images            15
```

Missing trials are represented only by a boolean mask. Do not insert a fake observed trial.

## 3. Naming and notation

| Symbol | Default | Meaning |
|---|---:|---|
| `B` | 4–8 | subjects per Stage-2 batch |
| `S` | 100 | maximum stimuli per subject |
| `N` | `≤B×S` | valid trials after flattening |
| `T` | 192 | spatial tokens on a `12×16` grid |
| `D` | 128 | hidden feature dimension |
| `H` | 4 | attention heads |
| `K` | 9 | tokens in a local `3×3` bank window |
| `K_cat` | 4 | stimulus categories |

Row-major spatial index:

```text
t = row * 16 + column
row = t // 16
column = t % 16
```

## 4. Stage-1 checkpoint contract

The current Stage-1 encoder is `stage1.heatmap_encoder.HeatmapPatchEncoder`:

```text
heatmap                               [N,3,48,64]
Conv2d(3,128,kernel=4,stride=4)       [N,128,12,16]
GroupNorm(32) + GELU                  [N,128,12,16]
flatten row-major                     [N,192,128]
optional mask-token replacement       [N,192,128]
fixed 2-D sine/cosine position        [N,192,128]
two residual spatial blocks           [N,192,128]
```

Stage 2 always calls it unmasked:

```python
H = heatmap_encoder(heatmaps, token_mask=None)
```

Only keys under the approved encoder prefix may initialize the query encoder. Loading DINO adapter, fusion, decoder or Stage-1 pooling weights into the Stage-2 query path is forbidden.

The checkpoint registry must map each fold to exactly one explicit file:

```yaml
fold_0: /root/EMS-Project/outputs/stage1/<run>/fold_0/checkpoints/best_stage1_fold0.pt
fold_1: /root/EMS-Project/outputs/stage1/<run>/fold_1/checkpoints/best_stage1_fold1.pt
fold_2: /root/EMS-Project/outputs/stage1/<run>/fold_2/checkpoints/best_stage1_fold2.pt
fold_3: /root/EMS-Project/outputs/stage1/<run>/fold_3/checkpoints/best_stage1_fold3.pt
fold_4: /root/EMS-Project/outputs/stage1/<run>/fold_4/checkpoints/best_stage1_fold4.pt
```

Never select the newest checkpoint implicitly when several candidates exist.

## 5. Evaluation-regime metadata

Every run and bank records one of:

```text
pilot_existing_stage1
strict_nested_stage1
```

`pilot_existing_stage1` means an existing fold checkpoint may have been selected using the HC validation subjects that later form the Stage-2 held-out fold. It is useful for development but must not be described as a strict untouched-test estimate.

`strict_nested_stage1` means Stage-1 epoch/hyperparameter selection occurred entirely inside the Stage-2 outer-training subjects, followed by a fixed-schedule refit on outer-training HCs.

In `strict_nested_stage1`, per-epoch validation and checkpoint selection use a persisted inner subject-level selection split. The outer held-out fold is evaluated exactly once after model, threshold and calibration choices are frozen. In `pilot_existing_stage1`, per-epoch validation may use the outer fold, but every output must label it `outer_fold_exploratory`.

Phase 0 must ask the user which regime applies and store the answer. Do not infer or silently upgrade the claim.

## 6. Normative-bank statistics

### 6.1 Required trial-level bank

For each fold and stimulus:

```text
mu_trial       float32 [100,128]
sigma_trial    float32 [100,128]
count_trial    int32   [100]
```

The statistics come from unmasked Stage-1 **post-fusion trial embeddings** of training-partition HC subjects.

Use population diagonal variance, matching the existing Stage-1 bank builder:

\[
\mu_s=\frac1{n_s}\sum_h z_{h,s},
\]

\[
\sigma_s
=
\sqrt{\max\left(
\frac1{n_s}\sum_hz_{h,s}^2-\mu_s^2,
\epsilon^2
\right)}.
\]

Accumulate sums and squared sums in float64; cast stored arrays to float32.

### 6.2 Required fused token bank for the full semantic model

When `include_fused_token_bank=true`:

```text
mu_token       float32 [100,192,128]
sigma_token    float32 [100,192,128]
```

These are statistics of Stage-1 final fused semantic–gaze tokens, not raw DINO tokens.

### 6.3 Optional same-space heatmap-token bank

When `include_heatmap_token_bank=true`:

```text
mu_heat_token       float32 [100,192,128]
sigma_heat_token    float32 [100,192,128]
```

These are statistics directly from Stage-1 `heatmap_tokens`. They permit direct cosine and standardized residuals only while the transferred encoder remains frozen at the exact checkpoint used to build the bank.

This artifact is optional because it approximately doubles token-bank storage.

## 7. Five-fold bank directory contract

Canonical layout:

```text
normative_bank/
├── manifest.json
├── stage1_checkpoints_resolved.yaml
├── fold_0/
│   ├── mu_trial.npy
│   ├── sigma_trial.npy
│   ├── count_trial.npy
│   ├── mu_token.npy                 # when enabled
│   ├── sigma_token.npy              # when enabled
│   ├── mu_heat_token.npy            # optional
│   ├── sigma_heat_token.npy         # optional
│   ├── feature_manifest.csv
│   ├── metadata.json
│   ├── audit.json
│   └── crossfit/
│       ├── subject_assignment.csv
│       ├── split_0/
│       │   ├── mu_trial.npy
│       │   ├── sigma_trial.npy
│       │   ├── count_trial.npy
│       │   ├── mu_token.npy         # when enabled
│       │   ├── sigma_token.npy
│       │   └── metadata.json
│       └── split_1/ ... split_3/
├── fold_1/
└── fold_4/
```

`manifest.json` lists all five folds, paths, array shapes, checksums, build status and evaluation regime.

`feature_manifest.csv` contains exactly:

```text
stimulus_index,stimulus_id,category_id
```

The `stimulus_index` order must match processed, DINO, CV and Stage-1 artifacts.

## 8. Cross-fitted training banks

Self-inclusion can make a training HC artificially close to a bank it helped create. The strict default is four-way HC cross-fitting inside each outer-training fold.

Procedure for outer fold `k`:

1. deterministically split outer-training HC subjects into four approximately equal groups, stratified only by completeness if necessary;
2. assign every outer-training subject, HC and SZ, to one of the same four split IDs using a deterministic label-stratified assignment;
3. build `crossfit/split_j` using all outer-training HC subjects except HC members assigned to `j`;
4. during Stage-2 training, every subject assigned to split `j` uses bank `j`;
5. during validation/test, every subject uses the full outer-training-HC bank.

Assigning both HC and SZ subjects to crossfit bank IDs prevents bank contributor count from becoming a diagnosis cue.

Required checks:

- a training HC never contributes to the crossfit bank it receives;
- HC and SZ bank-count distributions are matched by assignment;
- validation subjects never contribute to any full or crossfit bank;
- every crossfit bank contains all 100 stimuli with `count >= min_samples`;
- subject assignments are stable for the same seed.

If the user declines cross-fitting in Phase 0, store `bank_train_mode=full_self_included` and report it as a limitation.

## 9. Bank metadata contract

Each `metadata.json` stores at least:

```text
schema_version
created_at_utc
fold
crossfit_split or null
seed
evaluation_regime
estimator
epsilon
min_samples
include_fused_token_bank
include_heatmap_token_bank
stage1_checkpoint_path
stage1_checkpoint_sha256
stage1_run_id
stage1_config_hash
processed_manifest_sha256
dino_feature_sha256
cv_partition_sha256
contributing_hc_subject_ids
forbidden_validation_subject_ids
n_contributing_subjects
n_trials
array_shapes
array_sha256
stimulus_manifest_sha256
```

Paths may be absolute for provenance, but checksums are authoritative.

## 10. Approximate bank storage

For float32 arrays:

```text
one mu_token or sigma_token          100*192*128*4 bytes ≈ 9.83 MB
one fused mean+sigma token bank      ≈ 19.66 MB
five full fused token banks          ≈ 98.3 MB
five full + four crossfit/fold       ≈ 491.5 MB
```

Enabling `mu_heat_token/sigma_heat_token` approximately doubles token-bank storage. Phase 0 must check disk space before enabling it.

## 11. Stage-2 subject dataset contract

One `Dataset.__getitem__` call returns one subject, never one trial:

```python
{
    "subject_id": str,
    "label": int,                         # HC=0, SZ=1
    "heatmaps": Tensor[100,3,48,64],      # missing slots zero-filled only as storage padding
    "trial_mask": BoolTensor[100],        # defines which slots are real
    "stimulus_indices": LongTensor[100],  # canonical 0..99 slot identity
    "category_ids": LongTensor[100],      # 0..3
    "trial_uids": list[str | None],
    "bank_split_id": int | None,
}
```

Zero-filled missing slots are never passed as valid trials. Every consumer must mask them.

Subject ordering must come from explicit CV partition files, not numeric ID thresholds. Diagnostic labels must come from the canonical manifest, even if the current dataset uses ID ranges.

## 12. Subject batch contract

Collated batch:

```text
subject_ids                list[str], length B
labels                     float32 [B]
heatmaps                   float32 [B,100,3,48,64]
trial_mask                 bool    [B,100]
stimulus_indices           int64   [B,100]
category_ids               int64   [B,100]
bank_split_ids             int64   [B] or null
```

The model flattens only valid trials:

```text
valid_heatmaps             [N,3,48,64]
flat_stimulus_indices      [N]
flat_category_ids          [N]
flat_subject_slots         [N]
flat_stimulus_slots        [N]
```

For `B=8` with complete panels, `N=800`.

## 13. Subject sampler contract

The base sampler is a deterministic balanced subject sampler:

```text
batch_size=4    -> 2 HC + 2 SZ when possible
batch_size=8    -> 4 HC + 4 SZ when possible
```

Requirements:

- no subject repeats within a batch unless explicitly configured for the final incomplete batch;
- epoch shuffling depends on `(seed, fold, epoch)`;
- distributed ranks, if supported, receive disjoint subjects;
- the sampler never balances by duplicating individual trials;
- report dropped or repeated final-batch subjects;
- validation order is fixed and includes each subject once.

## 14. Base Stage-2 model contract

### 14.1 Transferred encoder

```text
valid heatmaps                         [N,3,48,64]
frozen Stage-1 HeatmapPatchEncoder     [N,192,128]
```

Do not reuse the Stage-1 trial pooling module. Stage 2 learns a new query pooler.

### 14.2 New query attention pooler

```text
H                                      [N,192,128]
pool logits                            [N,192]
patch attention                        [N,192]
q0                                     [N,128]
query projection q                     [N,128]
```

### 14.3 Trial bank lookup and adapter

```text
mu_raw                                 [N,128]
sigma_raw                              [N,128]
count                                  [N]
n_mu                                   [N,128]
uncertainty_context                    [N,128]
rho                                    [N,1]
```

Query tokens are pre-fusion; the trial bank is post-fusion. Raw subtraction or raw cosine is invalid. Use learned query and bank projections.

### 14.4 Explicit normative relation

```text
q                                      [N,128]
n_mu                                   [N,128]
uncertainty_context                    [N,128]
delta                                  [N,128]
absolute delta                         [N,128]
elementwise product                    [N,128]
cosine                                 [N,1]
reliability                            [N,1]
concatenated relation                  [N,770]
MLP 770 -> 256 -> 128                  [N,128]
z_trial                                [N,128]
```

Use a residual query shortcut and LayerNorm.

### 14.5 Subject-panel reconstruction

```text
flat z_trial                           [N,128]
scatter                                [B,100,128]
trial mask                             [B,100]
```

### 14.6 Category-balanced gated attention

Use gated attention within each of the four stimulus categories:

\[
g_{i,s}=w^\top[\tanh(Vz_{i,s})\odot\sigma(Uz_{i,s})].
\]

Softmax separately over valid stimuli in category `k`:

\[
\alpha^{(k)}_{i,s}
=
\frac{\exp(g_{i,s})}
{\sum_{j\in\mathcal S_k,M_{i,j}=1}\exp(g_{i,j})}.
\]

Category token:

\[
c_{i,k}=\sum_{s\in\mathcal S_k}\alpha^{(k)}_{i,s}z_{i,s}.
\]

Global stimulus importance gives equal total mass to each present category:

\[
I_{i,s}=\alpha^{(k(s))}_{i,s}/K_i.
\]

Shapes:

```text
within-category alpha                  [B,100]
global importance I                    [B,100]
category tokens                        [B,4,128]
```

### 14.7 Subject Transformer and classifier

```text
learned subject token                  [B,1,128]
four category tokens                   [B,4,128]
one-layer Transformer input/output     [B,5,128]
subject embedding                      [B,128]
main logit                             [B]
```

Default Transformer:

```text
d_model=128
heads=4
ffn_dim=256
layers=1
dropout=0.25
```

### 14.8 Additive evidence head

```text
evidence score e                       [B,100]
auxiliary logit                        [B]
signed contribution C=I*e              [B,100]
```

Positive contribution supports SZ; negative contribution supports HC under the auxiliary additive classifier.

## 15. Base model interpretation outputs

```text
stimulus_importance                    [B,100]
stimulus_evidence                      [B,100]
stimulus_contribution                  [B,100]
semantic_compatibility                 [B,100]
normative_deviation                    [B,100]
weighted_normative_deviation           [B,100]
```

Definitions:

\[
\text{compatibility}_{i,s}=\cos(q_{i,s},n_{\mu,s}),
\]

\[
\text{deviation}_{i,s}=\rho_s[1-\cos(q_{i,s},n_{\mu,s})],
\]

\[
\text{weighted deviation}_{i,s}=I_{i,s}\rho_s[1-\cos(q_{i,s},n_{\mu,s})].
\]

Do not call these scores patch-level semantic explanations.

## 16. Optional fused-token normative branch

Enabled only when token arrays exist and `model.bank_mode=trial_and_fused_token`.

### 16.1 Token projections and aligned relation

```text
query heatmap tokens Q                 [N,192,128]
fused bank mean N_mu                   [N,192,128]
fused uncertainty context N_ctx        [N,192,128]
token reliability                      [N,192,1]
aligned relation R0                    [N,192,128]
```

Because query tokens are pre-fusion and bank tokens are post-fusion, use learned projections and a token matching objective before interpreting cosine.

### 16.2 Serial local attention

Canonical optional topology:

```text
R0
-> local cross-attention 1, 3x3 bank window
-> residual spatial NN bridge
-> local cross-attention 2, 3x3 bank window
-> token attention pooling
```

Shapes:

```text
local bank windows                      [N,192,9,128]
attention-1 weights                     [N,4,192,9]
H1                                      [N,192,128]
bridge spatial map                      [N,128,12,16]
Hb                                      [N,192,128]
attention-2 weights                     [N,4,192,9]
H2                                      [N,192,128]
d_token                                 [N,128]
```

Attention layers have independent parameters. Attention 2 consumes bridge output, not `R0`.

### 16.3 Token/trial fusion and map

```text
concat(z_trial,d_token)                 [N,256]
fusion MLP                              [N,128]
z_extended                              [N,128]
semantic patch map                      [B,100,12,16]
```

Map definition may combine normalized attention, token reliability and aligned mismatch:

\[
M_{i,s,t}
=I_{i,s}\bar A_{i,s,t}\rho_{s,t}[1-c_{i,s,t}].
\]

Overlay on the original image only after inference. Image pixels are not Stage-2 model inputs.

## 17. Optional same-space heat-token branch

When the frozen query encoder and `mu_heat_token/sigma_heat_token` share the exact checkpoint:

\[
Z^{heat}
=\operatorname{clip}
\left(
\frac{H-\mu^{heat}}{\sigma^{heat}+\epsilon},-5,5
\right).
\]

Direct heat-token cosine and z-score are valid in this branch. Disable this branch automatically if the encoder is unfrozen.

## 18. Loss contract

All diagnostic BCE losses operate on subject logits `[B]` and labels `[B]`.

### 18.1 Classification with category-stratified subsets

Run the aggregator on the full panel and two independently sampled category-stratified masks containing `50–80%` of valid stimuli:

\[
L_{cls}
=BCE(\ell^{full},y)
+\frac12[BCE(\ell^A,y)+BCE(\ell^B,y)].
\]

Validation subset masks are fixed deterministically per `(seed,fold,subject_id)`.

### 18.2 Auxiliary additive-evidence loss

\[
L_{aux}=BCE(\ell^{aux},y).
\]

### 18.3 Trial-bank matching, HC trials only

Use correct stimulus bank `n_pos` and a different same-category stimulus bank `n_neg`:

\[
s^+=\cos(q^{match},n^{pos}),
\qquad s^-=\cos(q^{match},n^{neg}),
\]

\[
L_{trialmatch}
=\operatorname{mean}_{HC}(1-s^+)
+\operatorname{mean}_{HC}\max(0,m+s^- -s^+).
\]

Default margin `m=0.2`.

### 18.4 Full comparator bank ranking

Run the relation/comparator with correct and wrong banks:

\[
L_{bankrank}
=\operatorname{mean}_{HC}\max(0,m_b-r^+ +r^-).
\]

### 18.5 Optional token matching

Only for the fused-token model:

\[
L_{tokenmatch}
=\operatorname{mean}_{HC,t}\rho_{s,t}\omega_{i,s,t}
\left[
1-\cos(Q_t,N_t)
+\max(0,m_t+\cos(Q_t,N_{t^-})-\cos(Q_t,N_t))
\right].
\]

### 18.6 Matching objectives

Base:

\[
L_{match}=L_{trialmatch}+0.5L_{bankrank}.
\]

Token model:

\[
L_{match}=L_{trialmatch}+0.5L_{bankrank}+0.25L_{tokenmatch}.
\]

### 18.7 Subject subset consistency

\[
L_{latent}=\frac1B\sum_i[1-\cos(u_i^A,u_i^B)],
\]

\[
L_{prob}=JSD(Bern(p^A),Bern(p^B)),
\]

\[
L_{cons}=L_{latent}+L_{prob}.
\]

### 18.8 Early anti-collapse entropy

\[
L_{ent}=\frac1B\sum_i\max(0,H_{min}-H(I_i)).
\]

Anneal its weight to zero after the configured early epochs.

### 18.9 Optional encoder anchor

Only if the final encoder block is unfrozen:

\[
L_{anchor}=\|\theta_h-\theta_h^{Stage1}\|_2^2.
\]

### 18.10 Total objective

Starting values:

\[
L_{total}
=L_{cls}
+0.3L_{aux}
+0.1L_{match}
+0.1L_{cons}
+0.01L_{ent}
+\lambda_{anchor}L_{anchor}.
\]

For the frozen base model, `lambda_anchor=0`.

## 19. Training phases

### Stage 2A — bank alignment warm-up

```text
data                    outer-training HC subjects only
frozen                  transferred encoder and all bank arrays
trained                 query pooler/projection, bank adapter,
                        relation/comparator and optional token branch
objective               L_match
default duration        5–15 epochs
```

### Stage 2B — diagnostic training

```text
data                    outer-training HC and SZ subjects
frozen                  encoder in base model; bank always frozen
trained                 all Stage-2 modules
objective               L_total
default maximum         40–60 epochs
```

### Stage 2C — optional final-block fine-tuning

```text
unfreeze                final heatmap encoder residual block only
encoder LR              approximately 0.1 * Stage-2 LR
objective               L_total + lambda_anchor*L_anchor
```

This is a named ablation, not the base configuration.

### Stage 2D — calibration

Fit one temperature using inner or cross-validated training predictions. Never fit calibration or threshold on the held-out fold.

## 20. Base configuration contract

Starting configuration, not a claimed optimum:

```yaml
experiment_name: hc_normative_stage2
ablation: base
seed: 2026
fold: 0
evaluation_regime: pilot_existing_stage1

model:
  d_model: 128
  encoder_source: stage1_heatmap_encoder
  freeze_encoder: true
  query_pooling: attention
  relation_hidden: 256
  bank_mode: trial
  attention_heads: 4
  token_local_window: 3
  token_attention_layers: 2
  token_spatial_bridge: residual_dwconv_ffn
  category_balanced_attention: true
  subject_transformer_layers: 1
  subject_transformer_ffn: 256
  dropout: 0.25
  auxiliary_evidence_head: true

bank:
  root: /root/EMS-Project/normative_bank
  train_mode: crossfit
  require_fused_token_bank: false
  require_heatmap_token_bank: false
  verify_checksums: true

loss:
  lambda_aux: 0.3
  lambda_match: 0.1
  lambda_cons: 0.1
  lambda_entropy: 0.01
  entropy_anneal_epochs: 10
  lambda_anchor: 0.0
  match_margin: 0.2
  bank_rank_margin: 0.2

optimization:
  alignment_epochs: 10
  classification_epochs: 50
  optimizer: adamw
  learning_rate: 1.0e-4
  encoder_learning_rate: 1.0e-5
  weight_decay: 5.0e-4
  scheduler: linear_warmup_cosine
  warmup_epochs: 5
  gradient_clip_norm: 1.0
  amp: true

sampler:
  subject_batch_size: 4
  balance_groups: true
  drop_last: false

subsets:
  enabled: true
  min_fraction: 0.5
  max_fraction: 0.8
  category_stratified: true

validation:
  selection_metric: val_balanced_accuracy
  secondary_metric: val_auroc
  early_stopping_patience: 10
  calibrate: true

runtime:
  num_workers: 0
  pin_memory: true
  persistent_workers: false
  deterministic_validation: true

paths:
  processed_root: /root/EMS-Project/processed_dataset
  cv_root: /root/EMS-Project/CV/5fold_seed2026
  stage1_output_root: /root/EMS-Project/outputs/stage1
  output_root: /root/EMS-Project/outputs/stage2
```

## 21. Output directory contract

```text
outputs/stage2/<run_id>/
├── config_resolved.yaml
├── environment.json
├── run_metadata.json
├── source_checksums.json
└── fold_0/
    ├── history.csv
    ├── epoch_commit.json
    ├── train.log
    ├── checkpoints/
    │   ├── best_stage2_fold0.pt
    │   └── last_stage2_fold0.pt
    ├── validation/
    │   ├── metrics.json
    │   ├── subject_predictions.parquet
    │   ├── stimulus_attributions.npz
    │   └── calibration.json
    └── audit/
        ├── bank_match_metrics.json
        ├── tensor_shapes.json
        └── leakage_checks.json
```

Do not overwrite a completed run. Create a new run ID for any resolved configuration change.

## 22. Per-epoch history contract

Append one durable row immediately after every successfully completed epoch. Rewrite atomically through a temporary file, flush, `fsync` and replace.

Required columns:

```text
run_id
fold
epoch
phase_epoch
global_step
training_phase
validation_scope
eligible_for_best
is_best_epoch
best_epoch_so_far
learning_rate
encoder_learning_rate
learning_rate_min
learning_rate_max
weight_decay
lambda_aux
lambda_match
lambda_cons
lambda_entropy
lambda_anchor
train_loss
train_cls_loss
train_aux_loss
train_match_loss
train_trialmatch_loss
train_bankrank_loss
train_tokenmatch_loss
train_cons_loss
train_entropy_loss
train_anchor_loss
val_loss
val_cls_loss
val_aux_loss
val_match_loss
val_cons_loss
train_accuracy
train_balanced_accuracy
val_accuracy
val_balanced_accuracy
val_auroc
val_f1
val_sensitivity
val_specificity
val_brier
train_attention_entropy
val_attention_entropy
train_matched_cosine
train_wrong_cosine
val_matched_cosine
val_wrong_cosine
bank_rank_accuracy
grad_norm_mean
grad_norm_max
grad_clip_fraction
optimizer_step_count
skipped_optimizer_step_count
num_train_subjects
num_val_subjects
num_train_trials
num_val_trials
n_train_batches
n_val_batches
nonfinite_batch_count
epoch_time_seconds
peak_gpu_memory_mb
seed
```

Use empty fields only when a component is not applicable, such as `tokenmatch` in the trial-only model. Never change the CSV column set between epochs.

## 23. Checkpoint contract

Both best and last checkpoints store:

```text
model state
optimizer state
scheduler state
AMP scaler state
epoch
global step
best metric
best epoch
early-stopping state
resolved config
config hash
fold
run ID
training phase
Stage-1 checkpoint path and SHA-256
normative-bank paths and SHA-256
processed/CV manifest checksums
Python/NumPy/PyTorch/CUDA RNG states
sampler epoch/state
```

Exact resume rejects fold, architecture, bank checksum, Stage-1 checkpoint or config mismatches.

Weight-only initialization is a separate CLI operation with a fresh optimizer, scheduler, history and run ID.

## 24. Validation and metrics contract

Validation includes each held-out subject exactly once and uses the full outer-training-HC bank.

Report:

```text
accuracy
balanced accuracy
AUROC
F1
sensitivity
specificity
Brier score
confusion matrix
number of HC/SZ subjects
number of observed trials
```

Select the best checkpoint by the configured subject-level metric. If several epochs tie within numeric tolerance, choose the earlier epoch.

Do not select by trial-level accuracy or trial count.

## 25. Interpretation faithfulness contract

Store model scores, but do not claim explanation faithfulness without controls:

- delete top-k important stimuli and compare with category-matched random deletion;
- occlude top semantic patches and compare with spatially matched random patches;
- use wrong-stimulus and shuffled-token banks;
- report the change in subject logit and performance.

## 26. Required implementation checks

At minimum, automated tests verify:

1. dataset returns one subject and the correct mask;
2. missing trials never enter softmax or loss;
3. balanced sampler operates on subject IDs;
4. bank gather retrieves the exact stimulus row;
5. crossfit subjects do not use a bank containing themselves;
6. Stage-1 encoder load includes only allowed keys;
7. frozen encoder parameters receive no gradients;
8. base forward shapes match the contracts;
9. token branch shapes and serial execution order match the contracts;
10. HC/SZ BCE is subject-level;
11. `L_match` uses HC trials only;
12. wrong-stimulus bank produces a different relation for a controlled fixture;
13. history is written after one epoch;
14. best and last checkpoints independently load;
15. resume reproduces the next optimization step;
16. ablation overlays change only declared fields;
17. validation includes each subject exactly once;
18. output predictions reconcile with metrics.
