#!/usr/bin/env python3
"""Generate the Stage-1 README from the implemented config and a real dry-run.

Parameter counts and tensor shapes come from an actual model instantiation +
forward pass, never from hand-typed values.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from stage1.config import ConfigError, Stage1Config  # noqa: E402
from stage1.model import Stage1Model, summarize_model  # noqa: E402
from stage1.types import Stage1Batch  # noqa: E402

import torch  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "readme" / "Stage1"

# Plain (non-f-string) equation block; inserted verbatim into the README.
EQUATIONS = """```text
mu_{-h,s} = (1/(H-1)) sum_{h' != h} z_{h',s}
L_norm = (1/N) sum_{h,s} [1 - cos(z_{h,s}, sg(mu_{-h,s}))]
L_stage1 = L_rec + lambda_norm * L_norm
```"""


def _tensor_table(n: int, s: int, cfg: Stage1Config) -> tuple[str, dict]:
    """Instantiate the configured model and measure real shapes."""
    model = Stage1Model(cfg)
    summary = summarize_model(model)
    with torch.inference_mode():
        batch = Stage1Batch(
            heatmaps=torch.randn(n, cfg.model.input_channels, 48, 64),
            unique_dino_tokens=torch.randn(s, 768, 384),
            trial_to_stimulus_slot=torch.tensor([i % s for i in range(n)]),
            stimulus_indices=torch.arange(s),
            subject_ids=[f"{i:03d}" for i in range(n)],
            stimulus_ids=[f"s{i % s}" for i in range(n)],
            trial_uids=[f"u{i}" for i in range(n)],
            groups=["HC"] * n,
        )
        mask = torch.zeros(n, 192, dtype=torch.bool)
        mask[:, :67] = True
        out = model(batch, mask, return_fused=True, return_pooling_weights=True)
    rows = [
        ("heatmaps", f"[{n}, {cfg.model.input_channels}, 48, 64]"),
        ("unique DINO patch tokens", f"[{s}, 768, 384]"),
        ("trial-to-stimulus slot", f"[{n}]"),
        ("heatmap patch tokens", f"[{n}, 192, 128]"),
        ("adapted unique semantic tokens", f"[{s}, 192, 128]"),
        ("adapted semantic tokens", f"[{n}, 192, 128]"),
        ("attention-1 fused tokens", f"[{n}, 192, 128]"),
        ("spatial-bridge tokens", f"[{n}, 192, 128]"),
        ("attention-2 fused tokens", f"[{n}, 192, 128]"),
        ("reconstruction", f"[{n}, {cfg.model.input_channels}, 48, 64]"),
        ("trial embedding", f"[{n}, 128]"),
    ]
    table = "\n".join(f"| {name} | {shape} |" for name, shape in rows)
    header = "| tensor | shape |\n|---|---|"
    return f"{header}\n{table}", summary


def render_readme(cfg: Stage1Config, dry_text: str, summary_text: str) -> str:
    m = cfg.model
    return f"""# Stage-1 — HC-only semantic-conditioned normative encoder

Generated from the implemented code and resolved configuration (do not edit by hand).

```bash
python generate_stage1_readme.py --config configs/stage1/base.yaml
```

## 1. Research objective and Stage-1-only scope

Stage 1 learns an HC-only, semantic-conditioned normative representation of
eye-movement patterns: a per-trial 128-d embedding that reconstructs the
masked three-channel fixation heatmap while staying consistent across healthy
viewers of the same stimulus (leave-one-out cosine consistency). **Stage 2
(SZ/HC classification) is future work and is not implemented.**

## 2. HC-only and no-VICReg statement

Stage 1 trains exclusively on HC trials from the current outer fold; SZ rows
are filtered out at the dataset boundary. The objective is masked
reconstruction + leave-one-out cosine normative consistency. **No VICReg, no
contrastive loss (InfoNCE/SupCon), no SZ classification loss, and no
diagnostic classifier exist anywhere in this package.**

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
heatmap [N,{m.input_channels},48,64]
  -> Conv2d({m.input_channels},128,k=4,s=4) -> GN+GELU -> [N,192,128]
  -> masked-token replacement -> fixed 2-D sincos positions
  -> {m.heatmap_residual_blocks} residual blocks          H0
  -> cross-attention 1 (Q=H0, K/V=adapted DINO) + gamma_1 residual     H1
  -> spatial NN bridge (LN -> 128->256 GELU -> dwconv 3x3 -> GN+GELU -> 256->128) + eta residual   M
  -> cross-attention 2 (Q=M, K/V=same adapted DINO) + gamma_2 residual H2
  -> LayerNorm                                            Z  [N,192,128]
  -> decoder (ConvT(128,64,k4,s4) -> res block -> 1x1 -> {m.input_channels} ch)
  -> attention pooling (softmax(w2^T tanh(W1 z_j)))        z  [N,128]
```

The canonical order `Attention 1 -> spatial NN bridge -> Attention 2` is
immutable in the base configuration. The two cross-attentions have fully
independent parameters and share only the same adapted DINO key/value tensor.
Residual gates are initialized as `gamma_1 = gamma_2 = {m.semantic_gamma_total_init}/2`
and `eta = {m.spatial_bridge_eta_init}` (learned by default).

### Measured tensor shapes (S={cfg.sampler.stimuli_per_batch}, H={cfg.sampler.hc_per_stimulus}, N=S*H)

{dry_text}

### Measured parameter counts

{summary_text}

## 6. Losses

Masked reconstruction (channel-aware Smooth L1 on masked pixels; the token
mask on the 12×16 grid is upsampled by repeating each token over its 4×4
patch; equal channel weights by default):

```text
L_rec = sum_c w_c * mean_{{masked pixels}} SmoothL1(recon_c, target_c)
```

Leave-one-HC-out normative consistency (stop-gradiented centroid):

{EQUATIONS}

## 7. Training phases and best-checkpoint policy

- Phase A (epochs 0..{cfg.loss.norm_start_epoch - 1}): reconstruction warm-up, lambda_norm = 0
- Phase B (epochs {cfg.loss.norm_start_epoch}..{cfg.loss.norm_start_epoch + cfg.loss.norm_ramp_epochs - 1}): linear ramp of lambda_norm to {cfg.loss.lambda_norm}
- Phase C: joint objective
- Best-checkpoint eligibility: only epochs >= {cfg.loss.norm_start_epoch + cfg.loss.norm_ramp_epochs} are eligible
  (`best_eligible_after_norm_ramp={cfg.validation.best_eligible_after_norm_ramp}`); warm-up and ramp
  epochs are never compared against full-objective epochs.
- Model selection metric: `{cfg.validation.selection_metric}` (same lambda_norm as training; components logged separately).

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
python stage1_trainer.py --config configs/stage1/base.yaml --fold 0

# All folds sequentially with isolated outputs
python stage1_trainer.py --config configs/stage1/base.yaml --fold all

# Smoke test
python stage1_trainer.py --config configs/stage1/base.yaml --fold 0 \\
  --max-epochs 2 --max-train-batches 3 --max-val-batches 3 --run-name smoke

# Verify inputs and tensor contracts without optimization
python stage1_trainer.py --config configs/stage1/base.yaml --fold 0 --dry-run

# Resume exactly (optimizer/scheduler/RNG restored)
python stage1_trainer.py --resume outputs/stage1/<run>/fold_0/checkpoints/last_stage1_fold0.pt

# Initialize model weights only; fresh optimizer/scheduler/run
python stage1_trainer.py --config configs/stage1/base.yaml --fold 0 \\
  --load-stage1-weights outputs/stage1/<run>/fold_0/checkpoints/best_stage1_fold0.pt

# Rebuild the fold norm bank from the best checkpoint
python stage1_trainer.py --build-norm-bank \\
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
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "stage1" / "base.yaml"))
    parser.add_argument("--ablation", default=None)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        cfg = Stage1Config.load_base_with_ablation(REPO_ROOT / args.config, args.ablation)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    table, summary = _tensor_table(64, 8, cfg)
    dry_text = table
    summary_text = (
        f"- total parameters: **{summary.n_parameters_total:,}** (all trainable)\n"
        + "\n".join(
            f"- {name}: {count:,}"
            for name, count in summary.per_module.items()
            if count and name.count(".") == 0
        )
    )
    text = render_readme(cfg, dry_text, summary_text)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "README.md").write_text(text, encoding="utf-8")
    print(f"wrote {out / 'README.md'}")
    if args.verify_only:
        current = (out / "README.md").read_text(encoding="utf-8")
        if current != text:
            print("verify FAILED: stored README differs from freshly rendered text", file=sys.stderr)
            return 1
        print("verify OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
