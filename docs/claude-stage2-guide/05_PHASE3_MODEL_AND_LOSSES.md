# Phase 3 — implement the Stage-2 model and loss functions

## Goal

Implement the Stage-2 neural architecture under `src/stage2/` using:

- transferred Stage-1 heatmap encoder;
- newly trained query pooling/projection;
- per-stimulus normative-bank comparison;
- category-balanced stimulus attention;
- one subject Transformer layer;
- HC/SZ subject classifier;
- faithful additive evidence head;
- optional fused-token serial normative cross-attention;
- all loss components in the stable contract.

Do not implement the full optimizer/training loop, history or production validation in this phase.

## 1. Preconditions

Require Phase 2 completion and approval. Run:

- bank verification;
- data dry-run for one fold;
- Stage-2 data tests.

Do not change the processed dataset or bank schema in this phase.

## 2. Files to create or change

Recommended files:

```text
src/stage2/transferred_encoder.py
src/stage2/pooling.py
src/stage2/relation.py
src/stage2/token_attention.py
src/stage2/subject_aggregation.py
src/stage2/model.py
src/stage2/losses.py
src/stage2/contracts.py
tests/stage2/test_transferred_encoder.py
tests/stage2/test_trial_relation.py
tests/stage2/test_subject_aggregation.py
tests/stage2/test_token_attention.py
tests/stage2/test_stage2_model.py
tests/stage2/test_stage2_losses.py
```

## 3. Typed forward contracts

Define named dataclasses rather than returning positional tuples.

Recommended output fields:

```python
Stage2ForwardOutput(
    main_logit,                    # [B]
    auxiliary_logit,               # [B]
    subject_embedding,             # [B,128]
    trial_embeddings,              # [B,100,128]
    trial_mask,                    # [B,100]
    query_patch_attention,         # [B,100,192]
    stimulus_attention,            # [B,100]
    stimulus_importance,           # [B,100]
    stimulus_evidence,             # [B,100]
    stimulus_contribution,         # [B,100]
    semantic_compatibility,        # [B,100]
    normative_deviation,           # [B,100]
    weighted_normative_deviation,  # [B,100]
    semantic_patch_map,            # [B,100,12,16] or None
    diagnostics,                   # scalar/tensor diagnostics
)
```

Define a separate loss-output dataclass containing every component and diagnostic count.

## 4. Transferred encoder wrapper

Implement a wrapper around the existing Stage-1 `HeatmapPatchEncoder`; do not copy its architecture.

Responsibilities:

1. instantiate the exact architecture from the checkpoint's resolved config;
2. load only `heatmap_encoder.*` keys;
3. reject missing/unexpected encoder keys;
4. verify the checkpoint fold and SHA-256 against bank metadata;
5. freeze all encoder parameters in the base mode;
6. expose an explicit `unfreeze_last_block()` only for the named fine-tuning ablation;
7. always use `token_mask=None` in Stage 2;
8. preserve train/eval behavior for dropout while frozen;
9. report parameter counts and trainable names.

Important: when the full Stage-2 model enters `train()`, force the frozen encoder to remain in `eval()` if it contains stochastic or running-statistic layers. Its current GroupNorm/residual implementation has no BatchNorm, but enforce the wrapper invariant.

Input/output:

```text
[N,3,48,64] -> [N,192,128]
```

## 5. Query attention pooler

Implement a new Stage-2 attention pooler:

```text
LayerNorm(128)
Linear(128,64)
GELU or tanh
Linear(64,1)
masked spatial softmax over 192 tokens
weighted sum
```

There is no missing patch mask inside an observed heatmap; all 192 patches participate. A patch may contain zero gaze density and is still a real patch.

Return:

```text
patch attention [N,192]
q0              [N,128]
```

## 6. Query and bank projections

Implement independent modules:

```text
query_projection:  LayerNorm -> Linear(128,128)
bank_mean_adapter: LayerNorm -> Linear(128,128)
bank_sigma_adapter:
    concat(LayerNorm(log_sigma), log_count) [N,129]
    -> Linear(129,256) -> GELU -> Linear(256,128)
reliability head:
    [mean(-log_sigma), log_count] [N,2]
    -> MLP -> sigmoid [N,1]
```

Never calculate raw `q0 - mu_trial` or raw `cos(q0,mu_trial)`.

## 7. Explicit trial relation

Construct:

```text
q                        [N,128]
n_mu                     [N,128]
uncertainty_context      [N,128]
q-n_mu                   [N,128]
abs(q-n_mu)              [N,128]
q*n_mu                   [N,128]
cos(q,n_mu)              [N,1]
rho                      [N,1]
concat                   [N,770]
```

Relation MLP:

```text
Linear(770,256)
GELU
Dropout
Linear(256,128)
residual query projection
LayerNorm
```

Return `z_trial [N,128]`, cosine and reliability.

Do not replace the explicit aligned relation with cross-attention alone.

## 8. Stimulus/category embeddings

After scattering to `[B,100,128]`, add:

```text
learned stimulus embedding table     [100,128]
learned category embedding table     [4,128]
```

Apply LayerNorm after addition. Mask missing trial slots before downstream use.

Test that swapping stimulus IDs while holding heatmaps fixed changes the correct embedding and bank selection.

## 9. Category-balanced gated attention

Implement gated attention:

```text
tanh(Vz)                  [B,100,64]
sigmoid(Uz)               [B,100,64]
elementwise product       [B,100,64]
Linear(64,1) score        [B,100]
```

For each category independently:

- set missing scores to `-inf`;
- softmax only over valid members;
- produce one category token `[B,128]`;
- handle a category with no observed stimulus by using a learned missing-category token and exclude it from importance normalization.

Return:

```text
alpha within category     [B,100]
global importance         [B,100]
category tokens           [B,4,128]
category-present mask     [B,4]
```

Assert that importance sums to `1` over valid trials per subject and each present category receives equal total mass within tolerance.

## 10. Subject Transformer and main head

Prepend one learned subject token to the four category tokens.

Base:

```text
input/output               [B,5,128]
Transformer layers         1
heads                      4
head dimension             32
FFN                        256
dropout                    0.25
subject embedding          [B,128]
main logit                 [B]
```

Pass an appropriate category-present mask if any category is absent. Never expose missing-category padding as a real category response.

## 11. Additive evidence head

Shared MLP:

```text
trial token                [B,100,128]
evidence                   [B,100]
```

Compute:

```text
aux_logit = bias + masked_sum(importance * evidence)  [B]
contribution = importance * evidence                  [B,100]
```

Set missing-slot contribution to zero.

Test exact equality between `aux_logit-bias` and the sum of stored contributions.

## 12. Base interpretation outputs

Scatter trial-level cosine and reliability back to `[B,100]`:

```text
compatibility = cosine
deviation = rho * (1-cosine)
weighted_deviation = importance * deviation
```

Missing slots must be zero in stored outputs and accompanied by `trial_mask`.

## 13. Optional fused-token branch

Implement behind `bank_mode=trial_and_fused_token`. If arrays are missing, fail at initialization.

### 13.1 Token adapters

```text
H query tokens             [N,192,128]
mu_token                   [N,192,128]
sigma_token                [N,192,128]
Q                          [N,192,128]
N_mu                       [N,192,128]
N_ctx                      [N,192,128]
rho_token                  [N,192,1]
```

Use independent query/bank projections. Do not subtract raw pre/post-fusion tokens.

### 13.2 Token relation

At each aligned patch concatenate:

```text
Q, N_mu, abs(Q-N_mu), Q*N_mu, cosine, rho
dimension = 4*128+2 = 514
MLP 514 -> 256 -> 128
R0 [N,192,128]
```

### 13.3 Local bank windows

Precompute or efficiently gather a padded `3×3` neighbor index for each of 192 grid positions.

Required tensors:

```text
local K/V                  [N,192,9,128]
neighbor-valid mask        [192,9]
relative-position bias     [4,9]
```

Do not wrap horizontally between the last column of one row and the first column of the next.

### 13.4 Serial topology

Implement exactly:

```text
R0
-> LocalMHA1 + residual + LayerNorm = H1
-> spatial bridge + gated residual + LayerNorm = Hb
-> LocalMHA2 + residual + LayerNorm = H2
```

Spatial bridge:

```text
[N,192,128]
-> [N,128,12,16]
-> depthwise Conv3x3
-> pointwise 128->256
-> GELU/dropout
-> pointwise 256->128
-> [N,192,128]
```

Attention 1 and 2 must not share parameters. Hooks/tests must prove Attention 2 consumes `Hb`.

### 13.5 Token pooling and trial fusion

Pool `H2` to `d_token [N,128]`, concatenate with base `z_trial`, and project:

```text
[z_trial,d_token]          [N,256]
Linear(256,128) + GELU
residual z_trial
LayerNorm
z_extended                 [N,128]
```

The same subject aggregator consumes `z_extended`.

### 13.6 Semantic patch map

Return reduced, normalized maps; do not retain full debug attention for every training sample.

```text
per-trial map              [N,192]
subject panel map          [B,100,12,16]
```

Expose full per-head weights only when an explicit debug flag is active on a small audit subset.

## 14. Optional same-space heat-token branch

Implement only if Phase 0 approved heat-token banks.

Requirements:

- exact checkpoint hash match;
- encoder frozen;
- direct standardized residual clipped to `[-5,5]`;
- direct heat-token cosine stored separately from learned fused-bank cosine;
- disable with a clear error if the encoder is unfrozen.

Fuse the branch through a learned scalar initialized to zero so it cannot dominate at initialization.

## 15. Loss implementation

Implement pure, independently testable functions/classes for:

```text
classification loss
additive evidence BCE
trial bank match
full comparator bank rank
optional token match
subset consistency
early entropy floor
optional encoder anchor
weighted total
```

### 15.1 Subject-level BCE

Accept only logits `[B]` and labels `[B]`. Assert against `[B,100]` diagnostic BCE inputs.

### 15.2 Subset generation

Generate two category-stratified subset masks for each subject:

- retain between configured min/max fractions;
- retain at least one trial from each present category when possible;
- never reactivate a missing trial;
- training masks vary deterministically with epoch;
- validation masks are fixed by stable hash.

The model should avoid rerunning the frozen encoder three times. Encode valid trials once, then rerun only subject aggregation for full/A/B masks.

### 15.3 Matching negatives

Sample a different stimulus from the same category. Record and expose negative stimulus indices for tests/audits.

Never accidentally select the positive stimulus as its own negative.

Use matching loss on HC trials only. Select HC using subject labels mapped through flat subject slots; do not infer from IDs.

### 15.4 Total loss output

Return:

```text
total
cls
aux
match
trialmatch
bankrank
tokenmatch
cons
latent_cons
prob_cons
entropy
anchor
n_hc_match_trials
n_skipped_match_trials
matched_cosine_mean
wrong_cosine_mean
bank_rank_accuracy
```

Use differentiable zeros for disabled components.

## 16. Forward/backward tests

### Base shape test

Synthetic example:

```text
B=4
S=100
missing trials in at least two subjects
bank mode=trial
```

Assert every output shape in the contract and finite values.

### Gradient test

After backward:

- frozen encoder parameters have no gradients;
- query pooler/projection receive finite nonzero gradients;
- bank adapter and relation receive finite nonzero gradients;
- stimulus attention and subject Transformer receive gradients;
- evidence head receives gradients through `L_aux`;
- bank tensors remain unchanged and have no gradient.

### Token topology test

- LocalMHA1 and LocalMHA2 have distinct parameter storage;
- both receive gradients;
- Attention 2 input hook equals bridge output;
- neighbor windows do not wrap grid edges;
- local attention weights have `[N,4,192,9]`;
- semantic map has `[B,100,12,16]`.

### Loss tests

- perfect logits produce lower BCE than reversed logits;
- wrong bank increases match/rank loss on a controlled fixture;
- SZ trials do not enter HC matching loss;
- subset masks respect categories/missingness;
- identical subset embeddings yield zero latent consistency;
- contribution sums reproduce auxiliary logit;
- entropy loss activates only below its floor;
- anchor loss is zero at the Stage-1 weights;
- disabled token loss returns differentiable zero.

## 17. Model dry-run command

Provide a module dry-run, not the final trainer:

```bash
python -m stage2.model \
  --config configs/stage2/base.yaml \
  --fold 0 \
  --device cpu \
  --dry-run
```

Print:

```text
total/trainable/frozen parameters
loaded encoder checkpoint and hash
bank mode and paths
one batch input shapes
every major intermediate/output shape
one forward total loss and components
gradient audit
```

Use a real fold batch only after synthetic tests pass.

## 18. Acceptance criteria

- Base model forward/backward passes with exact shapes.
- Only the Stage-1 heatmap encoder is transferred.
- Encoder is frozen in base mode.
- Correct stimulus bank is gathered.
- Category importance is normalized and balanced.
- HC/SZ loss is subject-level.
- Every loss component is independently tested.
- Optional serial token branch passes topology/gradient tests when enabled.
- Interpretation outputs are returned with masks.
- No full training loop, output history or checkpoint lifecycle is implemented yet.

## 19. Gate

End with exactly:

> Phase 3 Stage-2 model and loss implementation is complete. Would you like to inspect or change the architecture, loss weights, or token-attention branch, or should I continue to Phase 4 and implement the configuration-driven ablation framework?

Then stop.

