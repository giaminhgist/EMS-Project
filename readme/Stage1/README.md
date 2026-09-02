# Stage-1 — HC-only semantic-conditioned normative encoder

Static artifact: generated once from the implemented code and resolved
configuration; the generator script was removed by explicit user request, so
update the affected sections manually after code/config changes.

## 1. Research objective and Stage-1-only scope

Stage 1 learns an HC-only, semantic-conditioned normative representation of
eye-movement patterns: a per-trial 128-d embedding that reconstructs the
masked three-channel fixation heatmap while staying consistent across healthy
viewers of the same stimulus (leave-one-out cosine consistency). **Stage 2
(SZ/HC classification) is future work and is not implemented.**

## 2. HC-only and no-VICReg statement

Stage 1 trains exclusively on HC trials from the current outer fold; SZ rows
are filtered out at the dataset boundary. The objective is masked
reconstruction + leave-one-out cosine normative consistency, plus a
between-stimulus dispersion-floor hinge on stimulus centroids (approved
contract amendment, 2026-08-31) that prevents collapse-to-mean of the
embedding. **No VICReg, no contrastive loss (InfoNCE/SupCon), no SZ
classification loss, and no diagnostic classifier exist anywhere in this
package.**

## 3. Input and artifact dependencies

- `processed_dataset/` — heatmaps, subject/trial manifests, QC (Phase 1/2)
- `stimulus_features/dino_vits16/` — frozen DINO ViT-S/16 patch tokens (Phase 3)
- `CV/5fold_seed2026/fold_<k>/` — subject-level fold partitions (Phase 4)
- all three are checksum-verified at dataset initialization

## 4. DINO ViT-S/16 extraction and spatial alignment

Each stimulus (1024×768) was deterministically resized to 512×384 and passed
through frozen pretrained DINO ViT-S/16 (Phase 3). Patch tokens
`[768, 384]` on the 24×32 grid are reshaped and adapted:

```text
[N, 768, 384]
-> [N, 384, 24, 32]
-> DepthwiseConv2d(384, 384, kernel_size=2, stride=2)
-> Conv2d(384, 128, kernel_size=1)
-> [N, 128, 12, 16] -> [N, 192, 128]
```

One DINO patch covers 16×16 resized pixels = 2×2 heatmap cells; the adapter
aggregates 2×2 DINO patches = 32×32 resized pixels; the heatmap encoder
aggregates 4×4 heatmap cells = the same 32×32 pixels. Both branches therefore
produce an exactly aligned 12×16 grid.

## 5. Architecture

```text
heatmap [N,3,48,64]
  -> Conv2d(3,128,k=4,s=4) -> GN+GELU -> [N,192,128]
  -> masked-token replacement -> fixed 2-D sincos positions
  -> 2 residual blocks          H0
  -> cross-attention 1 (Q=H0, K/V=adapted DINO) + gamma_1 residual     H1
  -> spatial NN bridge (LN -> 128->256 GELU -> dwconv 3x3 -> GN+GELU -> 256->128) + eta residual   M
  -> cross-attention 2 (Q=M, K/V=same adapted DINO) + gamma_2 residual H2
  -> LayerNorm                                            Z  [N,192,128]
  -> decoder (ConvT(128,64,k4,s4) -> res block -> 1x1 -> 3 ch)
  -> attention pooling (softmax(w2^T tanh(W1 z_j)))        z  [N,128]
```

The canonical order `Attention 1 -> spatial NN bridge -> Attention 2` is
immutable in the base configuration. The two cross-attentions have fully
independent parameters and share only the same adapted DINO key/value tensor.
Residual gates are initialized as `gamma_1 = gamma_2 = 0.1/2`
and `eta = 0.1` (learned by default).

### Measured tensor shapes (S=8, H=8, N=S*H)

| tensor | shape |
|---|---|
| heatmaps | [64, 3, 48, 64] |
| unique DINO patch tokens | [8, 768, 384] |
| trial-to-stimulus slot | [64] |
| heatmap patch tokens | [64, 192, 128] |
| adapted unique semantic tokens | [8, 192, 128] |
| adapted semantic tokens | [64, 192, 128] |
| attention-1 fused tokens | [64, 192, 128] |
| spatial-bridge tokens | [64, 192, 128] |
| attention-2 fused tokens | [64, 192, 128] |
| reconstruction | [64, 3, 48, 64] |
| trial embedding | [64, 128] |

### Measured parameter counts

- total parameters: **617,479** (all trainable)
- heatmap_encoder: 128
- fusion: 3

## 6. Losses

Masked reconstruction (channel-aware Smooth L1 on masked pixels; the token
mask on the 12×16 grid is upsampled by repeating each token over its 4×4
patch; equal channel weights by default):

```text
L_rec = sum_c w_c * mean_{masked pixels} SmoothL1(recon_c, target_c)
```

Leave-one-HC-out normative consistency (stop-gradiented centroid):

```text
mu_{-h,s} = (1/(H-1)) sum_{h' != h} z_{h',s}
L_norm = (1/N) sum_{h,s} [1 - cos(z_{h,s}, sg(mu_{-h,s}))]
L_stage1 = L_rec + lambda_norm * L_norm
```

## 7. Training phases and best-checkpoint policy

- Phase A (epochs 0..9): reconstruction warm-up, lambda_norm = 0
- Phase B (epochs 10..14): linear ramp of lambda_norm to 0.1
- Phase C: joint objective
- Best-checkpoint eligibility: only epochs >= 15 are eligible
  (`best_eligible_after_norm_ramp=True`); warm-up and ramp
  epochs are never compared against full-objective epochs.
- Model selection metric: `val_loss` (same lambda_norm as training; components logged separately).

## 8. CV/leakage controls

- The independent unit is the subject; validation subjects never appear in
  the training partition of their fold.
- Stage 1 loads HC trials only from the current fold's partitions.
- Optional channel statistics would be fitted on the fold's training HC
  trials only (not implemented in the base configuration).
- Early stopping/model selection uses validation HC trials only.
- The fold normative bank uses outer-training HC subjects only (validation
  IDs are asserted absent).
- DINO features are pretrained and frozen; no EMS gaze or label was used to
  produce them.

## 9. Normative bank

After best-checkpoint selection, unmasked inference runs over all
outer-training HC trials and stores per-stimulus statistics:

```text
normative_bank/
├── mu_trial.npy        float32 [100, 128]
├── sigma_trial.npy     float32 [100, 128]   (diagonal, clamped at epsilon)
├── count_trial.npy     int32   [100]
├── feature_manifest.csv
└── metadata.json       fold, subject IDs, checkpoint SHA-256, checksums,
                        estimator, epsilon, min sample count
```

## 10. Commands

```bash
# One fold
python stage1_trainer.py --config configs/stage1/base.yaml --fold 1
python stage1_trainer.py --config configs/stage1/base.yaml --fold 0
python stage1_trainer.py --config configs/stage1/base.yaml --fold 2
python stage1_trainer.py --config configs/stage1/base.yaml --fold 3
python stage1_trainer.py --config configs/stage1/base.yaml --fold 4
# All folds sequentially with isolated outputs
python stage1_trainer.py --config configs/stage1/base.yaml --fold all

# Smoke test
python stage1_trainer.py --config configs/stage1/base.yaml --fold 0 \
  --max-epochs 2 --max-train-batches 3 --max-val-batches 3 --run-name smoke

# Verify inputs and tensor contracts without optimization
python stage1_trainer.py --config configs/stage1/base.yaml --fold 0 --dry-run

# Resume exactly (optimizer/scheduler/RNG restored)
python stage1_trainer.py --resume outputs/stage1/<run>/fold_0/checkpoints/last_stage1_fold0.pt

# Initialize model weights only; fresh optimizer/scheduler/run
python stage1_trainer.py --config configs/stage1/base.yaml --fold 0 \
  --load-stage1-weights outputs/stage1/<run>/fold_0/checkpoints/best_stage1_fold0.pt

# Rebuild the fold norm bank from the best checkpoint
python stage1_trainer.py --build-norm-bank \
  --checkpoint outputs/stage1/<run>/fold_0/checkpoints/best_stage1_fold0.pt

# Named ablations (one primary factor changed relative to base.yaml)
python stage1_trainer.py --config configs/stage1/base.yaml --ablation no_semantic --fold 0
```

Available ablations: `no_semantic`, `aligned_add_fusion`, `concat_fusion`,
`single_cross_attention`, `no_spatial_bridge`, `token_mlp_bridge`,
`no_fusion_residual`, `fixed_fusion_gates`, `mean_pooling`, `no_norm_loss`,
`full_reconstruction`, `fixation_only`, `no_transition_channel`,
`no_temporal_channel`, `avgpool_semantic_adapter`.

## 11. Output directory and history.csv

```text
outputs/stage1/<run_id>/
├── run_metadata.json / config_resolved.yaml / environment.json / source_checksums.json
└── fold_<k>/
    ├── history.csv          # durably rewritten after every completed epoch
    ├── checkpoints/best_stage1_fold<k>.pt  /  last_stage1_fold<k>.pt
    ├── validation/best_val_embeddings.npz / metrics.json
    └── normative_bank/ mu_trial.npy sigma_trial.npy count_trial.npy
                        feature_manifest.csv metadata.json
```

`history.csv` contains one row per completed epoch with ~50 columns covering
phases, eligibility, best status, learning rate, lambda_norm, train/val total
and per-channel losses, dispersions, gate values, realized mask ratios,
batch/trial/group counts, gradient statistics, non-finite batch count, epoch
time, and the seed. Empty fields mean "not applicable" and never change the
column set.

## 12. Troubleshooting

- **OOM**: reduce `sampler.stimuli_per_batch`/`hc_per_stimulus`, or run on GPU.
- **Missing IDs**: run `python build_cv.py --verify-only` and the
  `extract_dino_features.py --verify-only` command; Stage 1 refuses to start
  on mismatched manifests.
- **Checksum mismatch**: regenerate the artifact with the matching phase
  command, or use its `--verify-only` mode to locate the drift.
- **Resume mismatch**: resumes are rejected on fold, run ID, config hash,
  architecture, or source-checksum differences by design.
- **Non-finite loss**: check `nonfinite_batch_count` in history.csv; the
  epoch remains valid and the previous checkpoint is retained.
- **Insufficient HC groups**: the sampler reports skipped stimuli; the
  normative loss skips groups below 2 examples and counts them.
- **Failed DINO semantic-use test**: `shuffled_dino_recon_delta` near zero
  suggests the model may be ignoring semantics; inspect the attention
  entropies and gate magnitudes in `validation/metrics.json`.

## 13. Stage-2 statement

**Stage 2 (SZ/HC classification and the diagnostic evaluation of these
representations) is explicitly future work and is not implemented.**
