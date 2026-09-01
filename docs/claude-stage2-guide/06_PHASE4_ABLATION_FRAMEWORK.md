# Phase 4 — implement the configuration and ablation framework

## Goal

Implement a reproducible configuration layer and a registry of named Stage-2 ablations. An ablation must change only its declared factor, must preserve the subject-level evaluation protocol, and must leave an auditable resolved configuration.

Do not implement the production training loop in this phase. A one-batch forward/backward dry-run is required, but epoch training belongs to Phase 5.

## 1. Preconditions

Require explicit approval of Phase 3. Before editing:

1. run `git status --short` and preserve unrelated work;
2. run all `tests/stage2/` tests from Phases 1–3;
3. verify the base model completes a deterministic forward pass;
4. verify the selected fold bank and Stage-1 checkpoint pass checksum checks.

If the base model is not green, stop. Do not hide a Phase-3 defect behind an ablation flag.

## 2. Files to create or change

```text
configs/stage2/base.yaml
configs/stage2/ablations/no_bank.yaml
configs/stage2/ablations/wrong_stimulus_bank.yaml
configs/stage2/ablations/global_bank.yaml
configs/stage2/ablations/random_encoder.yaml
configs/stage2/ablations/unfreeze_last_block.yaml
configs/stage2/ablations/mean_query_pooling.yaml
configs/stage2/ablations/no_category_balance.yaml
configs/stage2/ablations/mean_subject_pooling.yaml
configs/stage2/ablations/no_match_loss.yaml
configs/stage2/ablations/no_aux_loss.yaml
configs/stage2/ablations/no_consistency_loss.yaml
configs/stage2/ablations/no_attention_entropy.yaml
configs/stage2/ablations/token_bank_serial_attention.yaml
configs/stage2/ablations/single_token_attention.yaml
configs/stage2/ablations/no_spatial_bridge.yaml
configs/stage2/ablations/same_space_heat_bank.yaml
src/stage2/config.py
src/stage2/ablations.py
tests/stage2/test_stage2_config.py
tests/stage2/test_stage2_ablations.py
```

Add more named overlays only if they test one scientific question and their required artifact already exists.

## 3. Configuration resolution order

Resolve configurations in this exact order:

```text
base.yaml
  -> one optional named ablation overlay
  -> explicit command-line overrides
  -> runtime-derived immutable fields
```

Runtime-derived fields include:

- absolute repository paths;
- resolved fold;
- Stage-1 checkpoint path and SHA-256;
- bank paths and SHA-256;
- bank schema version;
- git commit and dirty-state flag;
- resolved device and numerical precision.

Never mutate the loaded base dictionary in place. Deep-copy, merge, validate, freeze into typed dataclasses, calculate a canonical config hash, and write `config_resolved.yaml` before training.

Reject:

- unknown keys;
- duplicate YAML keys;
- type coercions such as the string `"false"` becoming truthy;
- an overlay that changes undeclared keys;
- an incompatible bank/model combination;
- multiple named ablations in the same run unless the user explicitly requests a factorial experiment.

## 4. Typed configuration groups

Use dataclasses or an equally strict typed model for:

```text
ExperimentConfig
ModelConfig
BankConfig
LossConfig
OptimizationConfig
SamplerConfig
SubsetConfig
ValidationConfig
RuntimeConfig
PathConfig
AblationSpec
```

`AblationSpec` must contain at least:

```python
name: str
scientific_question: str
declared_changes: tuple[str, ...]
required_bank_capabilities: tuple[str, ...]
forbidden_with: tuple[str, ...]
interpretation: str
is_negative_control: bool
```

Store this specification in the resolved run metadata. The filename alone is not sufficient provenance.

## 5. Base configuration

Populate `configs/stage2/base.yaml` from Section 20 of `01_STAGE2_CONTRACTS.md`. The base run must have these defining properties:

```yaml
ablation: base
model:
  encoder_source: stage1_heatmap_encoder
  freeze_encoder: true
  query_pooling: attention
  bank_mode: trial
  category_balanced_attention: true
  subject_transformer_layers: 1
  auxiliary_evidence_head: true
bank:
  train_mode: crossfit
  require_fused_token_bank: false
  require_heatmap_token_bank: false
loss:
  lambda_aux: 0.3
  lambda_match: 0.1
  lambda_cons: 0.1
  lambda_entropy: 0.01
```

Do not silently change the base because an optional token bank is present. Optional artifacts must be activated only by a named configuration.

## 6. Required ablation registry

Each row below is a separate run family, not a list of switches to combine.

| Name | Single scientific question | Declared change | Artifact requirement |
|---|---|---|---|
| `base` | Full trial-bank model | None | trial bank |
| `no_bank` | Does normative information add value? | Neutralize every bank-derived relation feature while preserving relation width | none beyond schema metadata |
| `wrong_stimulus_bank` | Is stimulus matching important? | Apply one fixed, category-preserving derangement to bank indices | trial bank |
| `global_bank` | Is per-stimulus normalization important? | Replace each stimulus entry by one training-HC global bank vector | trial bank |
| `random_encoder` | Do transferred Stage-1 heatmap features add value? | Same encoder architecture with seeded random weights, frozen | trial bank; learned projections remain |
| `unfreeze_last_block` | Does limited encoder adaptation help? | Unfreeze final residual block and enable anchor loss | trial bank |
| `mean_query_pooling` | Does learned patch pooling help? | Replace query attention pooling with masked mean | trial bank |
| `no_category_balance` | Does category balancing prevent count dominance? | One masked softmax over all stimuli | trial bank |
| `mean_subject_pooling` | Does inter-stimulus contextualization help? | Remove subject Transformer and use category-balanced weighted mean | trial bank |
| `no_match_loss` | Is explicit query/bank alignment useful? | `lambda_match=0` in diagnostic phase and skip alignment warm-up | trial bank |
| `no_aux_loss` | Does faithful additive evidence supervision help? | `lambda_aux=0`; retain the head only for dimension parity but detach it from training | trial bank |
| `no_consistency_loss` | Does subset consistency improve robustness? | `lambda_cons=0` and disable the second subset view | trial bank |
| `no_attention_entropy` | Does early attention regularization help? | `lambda_entropy=0` | trial bank |
| `token_bank_serial_attention` | Does local normative semantic structure add value? | Enable fused-token serial cross-attention | fused token bank |
| `single_token_attention` | Are two serial token layers necessary? | Set token attention layers from 2 to 1 | fused token bank; compare against token model, not trial base |
| `no_spatial_bridge` | Does spatial token propagation add value? | Replace residual depthwise-convolution bridge with identity | fused token bank; compare against token model |
| `same_space_heat_bank` | Is direct same-encoder token deviation sufficient? | Use heatmap-token bank and same-space cosine/z features | heatmap-token bank and frozen encoder |

`single_token_attention` and `no_spatial_bridge` are child ablations of `token_bank_serial_attention`. Their reference run must therefore be the token model, not `base`. Encode the reference configuration in metadata.

## 7. Controlled implementations

### 7.1 `no_bank`

Keep the base relation tensor width unchanged. Retain the query vector and set all bank-dependent blocks to deterministic neutral values after normalization. Do not replace the entire model with a new classifier because that would confound capacity and architecture.

Record a binary `bank_features_active=false` field. Assert that gradients do not flow into any path that reads bank values.

### 7.2 `wrong_stimulus_bank`

Create the permutation once per fold and seed, before any labels or predictions are read.

Requirements:

1. no stimulus maps to itself;
2. permute within the same coarse stimulus category when a category has at least two members;
3. use the same permutation for every subject in the run;
4. save `wrong_bank_permutation.json` in the run audit directory;
5. reuse the correct bank variance/count belonging to the permuted mean;
6. never tune the permutation on validation performance.

### 7.3 `global_bank`

Compute a count-weighted mean and pooled diagonal variance using the fold's training-HC bank statistics. Repeat that one global entry over 100 stimulus slots. Do not recompute it from raw validation trials.

Save the formula, contributing counts and resulting checksum.

### 7.4 `random_encoder`

Instantiate the exact Stage-1 heatmap encoder architecture, initialize it from the run seed, and keep it frozen. The normative bank remains a Stage-1 fused space, so query and bank projections must still be learned by `L_match`.

Do not pretend this is a same-space comparison. Report it as a transfer negative control.

### 7.5 `unfreeze_last_block`

Only the final heatmap residual block may become trainable. Use a separate optimizer group:

```text
Stage-2 parameters LR       1e-4
encoder final block LR      1e-5
all earlier encoder layers  frozen
```

Set `lambda_anchor > 0`, retain the original Stage-1 tensors in memory or a detached reference, and calculate `L_anchor` only for unfrozen parameters.

### 7.6 token-bank variants

Require bank metadata to declare `has_fused_tokens=true` or `has_heatmap_tokens=true` as appropriate. Fail before model construction if the required arrays are absent.

For fused-token attention:

```text
query heat tokens        [B,100,192,128]
normative fused tokens   [B,100,192,128]
local attention output   [B,100,192,128]
semantic patch map       [B,100,12,16]
```

For a heatmap-token bank, direct cosine/z comparisons are allowed only while the transferred encoder remains bitwise frozen and matches the bank checkpoint hash.

## 8. Keep training/evaluation fixed

An ablation overlay must not change any of these unless that item is its declared factor:

- outer fold assignments;
- crossfit-bank assignment;
- random seeds;
- subject sampler;
- batch size;
- epoch budget;
- optimizer and scheduler;
- early-stopping policy;
- validation subject set;
- metric implementation;
- calibration procedure;
- saved prediction schema.

Use at least three seeds for comparative reporting if compute permits. Pair runs by fold and seed. Statistical comparisons must use subject-level predictions or fold/seed scores, never 15,912 trials as independent samples.

## 9. Overlay-diff validation

Implement a utility that flattens base and resolved configurations to dotted keys. For every named ablation:

1. calculate the changed dotted keys;
2. compare them with `declared_changes`;
3. reject extra or missing changes;
4. validate required bank capabilities;
5. write the diff to run metadata.

Example:

```text
ablation=no_match_loss
declared_changes:
  - loss.lambda_match
  - optimization.alignment_epochs
actual_changes:
  - loss.lambda_match: 0.1 -> 0.0
  - optimization.alignment_epochs: 10 -> 0
status: valid
```

Paths, checksums, fold and run ID are runtime-derived and excluded from the scientific overlay diff, but still stored.

## 10. Tests

At minimum, test:

1. base YAML resolves into typed configuration;
2. unknown and duplicate keys fail;
3. boolean/string ambiguity fails;
4. every registered overlay changes exactly its declared keys;
5. multiple named ablations fail by default;
6. token ablations fail without the required bank arrays;
7. `same_space_heat_bank` fails when the encoder is unfrozen or hashes differ;
8. wrong-bank mapping is deterministic, category-preserving and has no fixed point;
9. global-bank pooled statistics match a hand-computed synthetic fixture;
10. base and `no_bank` relation widths are identical;
11. `unfreeze_last_block` exposes exactly the intended parameters;
12. config hashing is invariant to YAML key order;
13. every ablation produces a forward/backward pass with finite values;
14. missing trials have zero attention under every aggregation variant.

Run the complete existing Stage-2 test suite after the new tests.

## 11. Required dry-run report

For `base` plus every available ablation, report:

```text
name
reference configuration
changed dotted keys
required bank capabilities
trainable parameter count
frozen parameter count
input/output tensor shapes
one-batch total and component losses
finite forward/backward status
```

Do not fabricate token-ablation results if token banks were not approved in Phase 0. Mark them `not runnable: required artifact absent` and demonstrate that configuration validation rejects them cleanly.

## 12. Phase-4 completion gate

After implementation and verification, stop all repository changes and use the standard completion report from `00_MASTER_WORKFLOW.md`.

Ask exactly:

> Phase 4 is complete. Would you like to inspect or change the ablation definitions, or should I continue to Phase 5 and implement the trainer, validation, history and checkpoint system?

Do not start Phase 5 until the user explicitly approves.
