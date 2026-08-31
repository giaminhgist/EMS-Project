# Phase 5 — Stage-1 core implementation

## Goal

Implement the HC-only Stage-1 dataset, data loaders, grouped sampler, model,
losses, and fold-safe normative-bank builder. Do not implement the full training
loop, CSV history, checkpoints, or ablation runner until Phase 6.

Use the importable package directory approved in Phase 0. This guide denotes it
as `src/stage1`; if the user explicitly retained `src/Stage-1`, document and
implement a safe import strategy rather than pretending the hyphen is a normal
Python package name.

## 1. Suggested files

```text
src/stage1/__init__.py
src/stage1/config.py
src/stage1/dataset.py
src/stage1/sampler.py
src/stage1/masking.py
src/stage1/heatmap_encoder.py
src/stage1/semantic_adapter.py
src/stage1/cross_attention.py
src/stage1/spatial_bridge.py
src/stage1/semantic_fusion.py
src/stage1/decoder.py
src/stage1/pooling.py
src/stage1/model.py
src/stage1/losses.py
src/stage1/normative_bank.py
src/stage1/types.py
tests/stage1/test_dataset.py
tests/stage1/test_sampler.py
tests/stage1/test_masking.py
tests/stage1/test_model_shapes.py
tests/stage1/test_semantic_fusion.py
tests/stage1/test_losses.py
tests/stage1/test_gradient_flow.py
tests/stage1/test_normative_bank.py
```

## 2. Dataset contract

Create a map-style dataset initialized with:

```text
processed dataset root
DINO feature root
one fold partition file
split = train | val
group filter = HC
heatmap transform config
manifest checksum requirements
```

Every sample returns a typed dictionary or dataclass equivalent to:

```python
{
    "heatmap": Tensor[3, 48, 64],          # float32
    "subject_id": str,
    "stimulus_id": str,
    "stimulus_index": int,
    "trial_uid": str,
    "group": "HC",
}
```

Requirements:

- reject any non-HC row in Stage 1 rather than silently retaining it;
- resolve heatmaps through `(subject_array_path, subject_row_index)` and verify
  the expected `stimulus_index`;
- resolve DINO features through `feature_manifest.csv`;
- memory-map large `.npy` files where appropriate;
- use a small per-worker LRU cache for subject arrays rather than reopening a
  file for every trial;
- never return fake data for a missing trial;
- validate processed, CV, and DINO checksums at initialization;
- apply only fixed `log1p`/clipping transforms by default;
- return IDs as metadata and never encode their numeric magnitude as a feature.

Implement a Stage-1 collator that deduplicates stimuli within a batch. For the
default grouped batch it returns:

```text
heatmaps                    [64, 3, 48, 64]
unique_dino_tokens          [8, 768, 384]
trial_to_stimulus_slot      [64]
subject/stimulus metadata   length 64
```

The model runs the semantic adapter only on eight unique stimulus tensors and
then gathers adapted tokens to 64 trials. Do not materialize 64 redundant copies
of the raw DINO tensor.

## 3. Grouped HC sampler

The normative regularizer requires multiple HC trials for the same stimulus in a
batch. Implement `StimulusGroupedHCBatchSampler`.

Default conceptual batch:

```text
S = 8 unique stimuli
H = 8 distinct HC trials per stimulus
N = S × H = 64 trials
```

Algorithm:

1. build `stimulus_index -> dataset row indices` from actual training trials;
2. retain stimuli with at least the configured `min_hc_per_stimulus`;
3. seed a local generator from global seed, fold, and epoch;
4. sample `S` stimuli;
5. for each stimulus sample `H` distinct subjects without replacement;
6. concatenate the groups and optionally apply a deterministic within-batch
   permutation while retaining stimulus metadata;
7. implement `set_epoch(epoch)` for reproducible changes across epochs;
8. report skipped stimuli and effective batch counts.

Do not require the same HC subjects across all selected stimuli; incomplete
subjects are valid. Never duplicate a trial to fill a group unless the user
explicitly approves replacement sampling as an ablation.

Validation should use all valid validation-HC trials deterministically. The
trainer can collect embeddings for the full validation split and calculate the
normative metric per stimulus without relying on random validation batches.

## 4. Patch masking

Use a spatial token mask at the shared 12 × 16 grid:

```text
mask: bool [N, 192]
default mask ratio: 0.35
same spatial mask for all three heatmap channels
```

Create heatmap patch tokens first with a non-overlapping 4 × 4 patch embedding,
then replace selected tokens with a learned mask token before spatial mixing.
This avoids treating a raw zero-valued heatmap region as an unmistakable mask
and prevents direct leakage from masked pixels through overlapping convolutions.

Training masks vary reproducibly by epoch. Validation masks are fixed by stable
`SHA256(seed, fold, trial_uid)` so validation losses are comparable across
epochs and resumes.

Return the token mask from `forward` so reconstruction loss can upsample it to
the 48 × 64 pixel grid.

## 5. Heatmap patch encoder

Recommended compact architecture:

```text
heatmap                           [N,   3, 48, 64]
Conv2d(3,128,kernel=4,stride=4)   [N, 128, 12, 16]
GroupNorm + GELU                  [N, 128, 12, 16]
flatten                           [N, 192, 128]
replace masked tokens             [N, 192, 128]
add fixed 2-D sine/cos position    [N, 192, 128]
two lightweight residual blocks   [N, 192, 128]
```

Residual blocks may use pre-normalized MLP plus depthwise 3 × 3 convolution on
the 12 × 16 grid. Keep the encoder compact and record its parameter count.

## 6. Frozen-DINO semantic adapter

Input is already extracted frozen DINO output for `S` unique batch stimuli:

```text
DINO tokens                       [S, 768, 384]
reshape                           [S, 384, 24, 32]
DepthwiseConv2d(384,384,k=2,s=2)  [S, 384, 12, 16]
Conv2d(384,128,kernel=1,stride=1) [S, 128, 12, 16]
GroupNorm + GELU                  [S, 128, 12, 16]
flatten                           [S, 192, 128]
gather by trial_to_stimulus_slot  [N, 192, 128]
```

Each DINO patch covers `16 × 16` pixels on the resized 512×384 stimulus, which
corresponds to `2 × 2` cells on the 48×64 (H×W) heatmap. The adapter aggregates
`2 × 2` DINO patches, corresponding to `32 × 32` resized stimulus pixels. The
heatmap encoder aggregates `4 × 4` heatmap cells, also corresponding to
`32 × 32` resized stimulus pixels. Both branches therefore produce an exactly
aligned 12×16 grid as long as the complete stimulus is resized without crop,
padding, flip, or rotation.

The adapter is trainable and fold-specific. Raw DINO tokens remain immutable and
must never receive gradients.

## 7. Serial Attention–Spatial NN–Attention fusion

Use exactly two independent pre-normalized cross-attention modules connected in
series by one residual spatial neural-network bridge:

```text
heatmap tokens H0                  [N, 192, 128]
cross-attention 1 + residual       [N, 192, 128]
spatial NN bridge + residual       [N, 192, 128]
cross-attention 2 + residual       [N, 192, 128]
final LayerNorm                    [N, 192, 128]
```

This is not a parallel design. Attention 2 must use the output of the spatial
bridge as its query, so it performs semantic re-attention after the first
gaze–semantic alignment has been spatially contextualized. Both attention
modules reuse the same adapted DINO tensor as key/value, but they must have
independent query, key, value, output-projection, and normalization parameters.

### 7.1 First semantic cross-attention

Let heatmap tokens be `H0=T_H` and adapted DINO tokens be `T_I`. Use four heads
with head dimension 32:

\[
A_1=\operatorname{MHA}_1
\left(
\operatorname{LN}_{q1}(H_0),
\operatorname{LN}_{kv1}(T_I),
\operatorname{LN}_{kv1}(T_I)
\right),
\]

\[
H_1=H_0+\gamma_1\operatorname{Dropout}(A_1).
\]

Shapes:

```text
query/key/value tokens             [N, 192, 128]
attention map per head             [N, 192, 192]
attention heads                    4
attention dropout                  0.1
```

### 7.2 Residual spatial NN bridge

The bridge must mix both feature channels and neighboring positions. Do not use
only an independent token-wise MLP in the primary model. Recommended compact
bridge:

```text
LayerNorm                          [N, 192, 128]
Linear 128 -> 256 + GELU           [N, 192, 256]
reshape                            [N, 256, 12, 16]
DepthwiseConv2d 3x3, padding=1     [N, 256, 12, 16]
GroupNorm + GELU                   [N, 256, 12, 16]
flatten                            [N, 192, 256]
Linear 256 -> 128                  [N, 192, 128]
```

With `F_spatial` denoting this transformation:

\[
B=F_{spatial}(\operatorname{LN}_{bridge}(H_1)),
\]

\[
M=H_1+\eta\operatorname{Dropout}(B).
\]

Use dropout `0.1` on `B` in the residual branch and a learnable scalar bridge
gate `eta`, initialized to `0.1`. Preserve the row-major 12×16 token ordering
during every reshape. The bridge is applied after
Attention 1 because it is intended to propagate the newly injected semantic
information across neighboring gaze regions before Attention 2.

### 7.3 Second semantic cross-attention

Attention 2 receives `M`, not `H0`, as its query:

\[
A_2=\operatorname{MHA}_2
\left(
\operatorname{LN}_{q2}(M),
\operatorname{LN}_{kv2}(T_I),
\operatorname{LN}_{kv2}(T_I)
\right),
\]

\[
H_2=M+\gamma_2\operatorname{Dropout}(A_2),
\qquad
Z=\operatorname{LN}_{out}(H_2).
\]

Use a total initial semantic residual budget of `0.1`:

\[
\gamma_1=\gamma_2=0.05.
\]

Derive both initial values as
`semantic_gamma_total_init / 2`; do not hard-code `0.05` independently of the
resolved configuration. Both gates are learned independently after
initialization. Do not add another MLP after Attention 2 in the primary model;
the reconstruction decoder and trial pooling consume `Z` directly. A
post-attention MLP, shared attention weights, parallel attention branches, or an
additional third attention layer requires an explicitly named ablation and is
not part of the default model.

Return Attention-1 and Attention-2 weights only in debug/evaluation mode to
avoid unnecessary training memory.

## 8. Reconstruction decoder

Recommended compact decoder:

```text
tokens                            [N, 192, 128]
reshape                           [N, 128, 12, 16]
ConvTranspose2d(128,64,k=4,s=4)   [N,  64, 48, 64]
residual Conv block               [N,  64, 48, 64]
Conv2d(64,3,k=1)                  [N,   3, 48, 64]
```

Do not constrain temporal output with the same activation as nonnegative density
channels. The loss compares reconstructed output to the transformed input
channels and may use separate final activations or an unconstrained output with
channel-aware loss handling.

## 9. Trial pooling

Default attention pooling:

\[
a_j=\operatorname{softmax}
\left(w_2^\top\tanh(W_1Z_j)\right),
\qquad
z=\sum_{j=1}^{192}a_jZ_j.
\]

Shapes:

```text
token scores    [N, 192, 1]
pool weights    [N, 192, 1]
trial embedding [N, 128]
```

Return pooling weights in evaluation/debug mode. Mean pooling is an ablation.

## 10. Losses

### 10.1 Masked reconstruction

Upsample the token mask by repeating each 12 × 16 token over its corresponding
4 × 4 output patch:

```text
token mask   [N, 192]
pixel mask   [N, 1, 48, 64]
```

Compute channel-aware Smooth L1 or L1 only on masked pixels. Default channel
weights are equal and explicitly configured. Report both total and per-channel
reconstruction losses.

The loss must fail clearly when configured for masked-only reconstruction with
zero masked tokens. A full-reconstruction scope is a separate ablation.

### 10.2 HC leave-one-out normative consistency

Within each stimulus group, calculate the leave-one-HC-out centroid:

\[
\mu_{-h,s}=\frac{1}{H-1}\sum_{h'\neq h}z_{h',s}.
\]

The simple default regularizer is cosine consistency:

\[
\mathcal{L}_{HC-norm}=
\frac{1}{N}\sum_{h,s}
\left[1-\cos\left(z_{h,s},\operatorname{sg}(\mu_{-h,s})\right)\right],
\]

where `sg` denotes stop-gradient. Layer normalization and reconstruction prevent
the loss from becoming the sole representation objective. Track within-stimulus
and between-stimulus dispersion to detect collapse.

Do not use a batch z-score-to-zero loss as the default: scaling both embeddings
and their batch standard deviation can make that objective poorly identified.

Total loss:

\[
\mathcal{L}_{stage1}=\mathcal{L}_{rec}
+\lambda_N\mathcal{L}_{HC-norm}.
\]

No VICReg, InfoNCE, SupCon, diagnosis cross-entropy, or SZ label is allowed.

## 11. Forward output contract

Return a named structure containing:

```text
reconstruction      [N, 3, 48, 64]
trial_embedding     [N, 128]
fused_tokens        [N, 192, 128]  # optional by flag
attention1_tokens   [N, 192, 128]  # optional by flag
bridge_tokens       [N, 192, 128]  # optional by flag
mask                [N, 192]
pooling_weights     [N, 192]       # optional by flag
attention1_weights  [N, 4, 192, 192]  # debug only
attention2_weights  [N, 4, 192, 192]  # debug only
```

Do not return a diagnosis probability.

## 12. Normative-bank builder

Given a selected fold checkpoint:

1. force unmasked inference;
2. load only that fold's training HC trials;
3. compute trial embeddings;
4. group by explicit `stimulus_index`;
5. compute mean, diagonal standard deviation, and sample count;
6. clamp standard deviation only at a recorded epsilon;
7. optionally compute token-level statistics;
8. store IDs of all contributing training HC subjects;
9. assert no validation subject contributed;
10. atomically save arrays and metadata.

Do not silently fill a missing stimulus norm. Report insufficient sample counts
and fail or apply an explicitly configured fallback approved by the user.

## 13. Required tests

- Dataset returns exact shapes and composite-key-matched trials.
- Collation deduplicates DINO features and gathers each trial from the correct
  explicit stimulus slot.
- Stage-1 dataset rejects SZ rows.
- Sampler groups the configured number of distinct HC subjects by stimulus.
- Missing trials are not synthesized or duplicated.
- Mask count matches configured ratio within integer rounding.
- Validation masks are stable across calls and process restarts.
- All model tensor shapes match this document for `N=2` and `N=64` where
  feasible.
- Reconstruction loss changes only when masked target pixels change.
- Normative loss never mixes different stimuli.
- Normative loss requires at least two HC examples and reports skipped groups.
- Backpropagation reaches heatmap encoder, semantic adapter, fusion, decoder, and
  pooling parameters.
- Precomputed DINO tensors have no gradient and are unchanged.
- The execution order is Attention 1, spatial bridge, then Attention 2; hooks or
  controlled fixtures verify that Attention 2 receives the bridge output as its
  query rather than the original heatmap tokens.
- Attention 1 and Attention 2 do not share parameter objects or storage, and
  both receive finite gradients in a backward test.
- Setting `gamma_1=gamma_2=0` makes the fusion output independent of DINO tokens
  within numerical tolerance.
- Setting `eta=0` makes the spatial bridge residual an identity mapping.
- The spatial bridge preserves `[N,192,128]`, maintains row-major 12x16 ordering,
  and an impulse-style unit test verifies neighboring-position mixing.
- Initialization satisfies
  `gamma_1 + gamma_2 == semantic_gamma_total_init` within numerical tolerance.
- The single-attention ablation removes Attention 2 but retains Attention 1 and
  the spatial bridge, preserving the output interface.
- Mean-pooling and no-semantic modes satisfy the same output interface.
- Normative-bank metadata contains only training HC subject IDs.

## 14. Acceptance criteria

- Core modules and tests pass without a full training run.
- A synthetic forward/backward pass is finite.
- A one-batch real-data forward pass has correct IDs and shapes.
- Parameter counts and trainable/frozen component lists are reported.
- No Stage-2 or diagnostic code exists in the Stage-1 package.

## 15. Gate

Finish with the standard report and ask:

> Phase 5 Stage-1 core implementation is complete. Would you like to modify the
> model, losses, or tensor interfaces, or should I continue to Phase 6 trainer,
> ablations, and Stage-1 documentation?

Then stop.
