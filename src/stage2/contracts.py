"""Shared Stage-2 constants and artifact contract helpers (contract §2, §6, §7).

Pure data contract: shapes, dtypes, manifest columns, category mapping and
stimulus-order checks. No model or dataset code lives here.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

N_STIMULI = 100
N_TOKEN_CELLS = 192
TOKEN_GRID = (12, 16)
D_MODEL = 128
TRIAL_FEATURE_SHAPE = (D_MODEL,)
TOKEN_FEATURE_SHAPE = (N_TOKEN_CELLS, D_MODEL)
HEATMAP_SHAPE = (3, 48, 64)

# Canonical category order (first appearance in the processed image manifest,
# matching the contract §2 listing).
CATEGORY_NAMES = (
    "Manipulated Images",
    "Natural Scenes",
    "Social Scenes",
    "Synthetic Images",
)
CATEGORY_ID_BY_NAME = {name: i for i, name in enumerate(CATEGORY_NAMES)}

BANK_SCHEMA_VERSION = "1"
REGISTRY_SCHEMA_VERSION = 1
EVALUATION_REGIMES = ("pilot_existing_stage1", "strict_nested_stage1")

FEATURE_MANIFEST_COLUMNS = ("stimulus_index", "stimulus_id", "category_id")
CROSSFIT_ASSIGNMENT_COLUMNS = (
    "subject_id",
    "label",
    "bank_split_id",
    "is_hc_bank_contributor",
    "panel_trial_count",
)

TRIAL_ARRAYS = ("mu_trial", "sigma_trial", "count_trial")
TOKEN_ARRAYS = ("mu_token", "sigma_token")
HEAT_TOKEN_ARRAYS = ("mu_heat_token", "sigma_heat_token")

ARRAY_DTYPES: dict[str, np.dtype] = {
    "mu_trial": np.dtype("float32"),
    "sigma_trial": np.dtype("float32"),
    "count_trial": np.dtype("int32"),
    "mu_token": np.dtype("float32"),
    "sigma_token": np.dtype("float32"),
    "mu_heat_token": np.dtype("float32"),
    "sigma_heat_token": np.dtype("float32"),
}


def expected_array_shapes(include_token: bool, include_heat_token: bool) -> dict[str, tuple[int, ...]]:
    shapes: dict[str, tuple[int, ...]] = {
        "mu_trial": (N_STIMULI, D_MODEL),
        "sigma_trial": (N_STIMULI, D_MODEL),
        "count_trial": (N_STIMULI,),
    }
    if include_token:
        shapes["mu_token"] = (N_STIMULI, N_TOKEN_CELLS, D_MODEL)
        shapes["sigma_token"] = (N_STIMULI, N_TOKEN_CELLS, D_MODEL)
    if include_heat_token:
        shapes["mu_heat_token"] = (N_STIMULI, N_TOKEN_CELLS, D_MODEL)
        shapes["sigma_heat_token"] = (N_STIMULI, N_TOKEN_CELLS, D_MODEL)
    return shapes


def build_feature_manifest(image_manifest: pd.DataFrame) -> pd.DataFrame:
    """Canonical feature manifest: stimulus_index, stimulus_id, category_id.

    The stimulus order comes from the processed image manifest (which DINO,
    CV and Stage-1 artifacts are verified against), never from filenames.
    """
    rows: list[dict[str, Any]] = []
    for rec in image_manifest.itertuples():
        category_id = CATEGORY_ID_BY_NAME.get(str(rec.category))
        if category_id is None:
            raise ValueError(f"unknown stimulus category: {rec.category!r}")
        rows.append(
            {
                "stimulus_index": int(rec.stimulus_index),
                "stimulus_id": str(rec.stimulus_id),
                "category_id": category_id,
            }
        )
    manifest = pd.DataFrame(rows, columns=list(FEATURE_MANIFEST_COLUMNS))
    indices = manifest.stimulus_index.to_numpy()
    if len(indices) != N_STIMULI or not np.array_equal(np.sort(indices), np.arange(N_STIMULI)):
        raise ValueError(
            f"image manifest must define exactly stimulus indices 0..{N_STIMULI - 1}"
        )
    return manifest


def feature_manifest_csv_bytes(manifest: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    manifest.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_stimulus_order(
    feature_manifest: pd.DataFrame, dino_feature_manifest: pd.DataFrame, image_manifest: pd.DataFrame
) -> None:
    """The bank stimulus order must match processed, DINO and CV artifacts."""
    if not np.array_equal(
        feature_manifest.stimulus_id.to_numpy(), image_manifest.stimulus_id.to_numpy()
    ):
        raise ValueError("feature manifest stimulus order differs from processed image manifest")
    if not np.array_equal(
        dino_feature_manifest.stimulus_id.to_numpy(), image_manifest.stimulus_id.to_numpy()
    ):
        raise ValueError("DINO feature manifest stimulus order differs from image manifest")


# ------------------------------------------------------------- dataset contract


@dataclass(frozen=True)
class Stage2SubjectSample:
    """One dataset item = one subject's full masked 100-stimulus panel."""

    subject_id: str
    label: int  # HC=0, SZ=1
    heatmaps: torch.Tensor  # [100,3,48,64] float32; missing slots zero-filled storage padding
    trial_mask: torch.Tensor  # [100] bool; True = real observed trial
    stimulus_indices: torch.Tensor  # [100] int64 canonical 0..99 slot identity
    category_ids: torch.Tensor  # [100] int64 0..3
    trial_uids: tuple[str | None, ...]  # length 100; None for missing slots
    bank_split_id: int | None  # crossfit split for train subjects, else None


# ------------------------------------------------- model forward/output contracts


@dataclass
class Stage2ForwardOutput:
    """Per-mask output of one subject-aggregation pass (guide 05 §3, §16.1).

    All ``[B, 100, ...]`` panels are zero-filled on missing slots and always
    accompanied by ``trial_mask``.
    """

    main_logit: torch.Tensor  # [B]
    auxiliary_logit: torch.Tensor  # [B]  (bias + masked_sum(importance * evidence))
    subject_embedding: torch.Tensor  # [B,128]
    trial_embeddings: torch.Tensor  # [B,100,128]
    trial_mask: torch.Tensor  # [B,100]
    query_patch_attention: torch.Tensor  # [B,100,192]
    stimulus_attention: torch.Tensor  # [B,100] within-category alpha
    stimulus_importance: torch.Tensor  # [B,100] global importance I
    stimulus_evidence: torch.Tensor  # [B,100]
    stimulus_contribution: torch.Tensor  # [B,100]  C = I * e
    semantic_compatibility: torch.Tensor  # [B,100]
    normative_deviation: torch.Tensor  # [B,100]
    weighted_normative_deviation: torch.Tensor  # [B,100]
    semantic_patch_map: torch.Tensor | None  # [B,100,12,16] or None
    diagnostics: dict[str, Any]  # per-trial scalars/tensors and category tokens


@dataclass
class Stage2ForwardResult:
    """One full-panel aggregation plus optional same-encode subset reruns."""

    full: Stage2ForwardOutput
    subsets: dict[str, Stage2ForwardOutput]  # name (e.g. "A"/"B") -> output


@dataclass
class EncodedTrials:
    """Flattened per-trial tensors after one encoder/pooler/relation pass.

    The subject aggregator consumes this cache for the full panel and every
    subset mask, so the frozen encoder runs exactly once per batch.
    """

    batch_size: int
    subject_slots: torch.Tensor  # [N] int64
    stimulus_slots: torch.Tensor  # [N] int64
    category_ids: torch.Tensor  # [N] int64
    trial_mask: torch.Tensor  # [B,100] bool
    category_ids_panel: torch.Tensor  # [B,100] int64
    heatmap_tokens: torch.Tensor  # [N,192,128] pre-fusion encoder output H
    patch_attention: torch.Tensor  # [N,192]
    q0: torch.Tensor  # [N,128]
    q: torch.Tensor  # [N,128] query projection
    n_mu: torch.Tensor  # [N,128] bank mean projection
    uncertainty_context: torch.Tensor  # [N,128]
    rho: torch.Tensor  # [N,1]
    cosine: torch.Tensor  # [N,1]
    z_trial: torch.Tensor  # [N,128]
    comparator: torch.Tensor  # [N]
    bank_ids: torch.Tensor  # [N] int64
    # Optional fused-token branch outputs (all None when bank_mode == "trial").
    z_extended: torch.Tensor | None = None  # [N,128]
    token_attention_weights: torch.Tensor | None = None  # [N,4,192,9] attention-2
    token_omega: torch.Tensor | None = None  # [N,192] mean-head center attention
    token_map_flat: torch.Tensor | None = None  # [N,192] reduced semantic map
    token_cosine: torch.Tensor | None = None  # [N,192]
    token_rho: torch.Tensor | None = None  # [N,192,1]
    Q: torch.Tensor | None = None  # [N,192,128] query token projection
    N_mu: torch.Tensor | None = None  # [N,192,128] bank mean token projection

    @property
    def n_trials(self) -> int:
        return int(self.subject_slots.numel())


@dataclass
class Stage2MatchInputs:
    """Per-trial matching tensors for the HC-only trial/bank/token match losses.

    Correct-bank tensors reuse the encoded pass; wrong-bank tensors come from
    a different same-category stimulus of the same bank. Ineligible HC trials
    (category with a single stimulus) are excluded via ``hc_match_mask`` and
    their negative rows are placeholders equal to the positive rows.
    """

    hc_mask: torch.Tensor  # [N] bool — HC subject slots (never trial labels)
    hc_match_mask: torch.Tensor  # [N] bool — HC and category has a negative
    negative_stimulus_indices: torch.Tensor  # [N] int64
    cos_pos: torch.Tensor  # [N]
    cos_neg: torch.Tensor  # [N]
    comparator_pos: torch.Tensor  # [N]
    comparator_neg: torch.Tensor  # [N]
    rho: torch.Tensor  # [N,1]
    Q: torch.Tensor | None = None  # [N,192,128]
    N_mu_pos: torch.Tensor | None = None  # [N,192,128]
    N_mu_neg: torch.Tensor | None = None  # [N,192,128]
    token_rho: torch.Tensor | None = None  # [N,192,1]
    token_omega: torch.Tensor | None = None  # [N,192]


@dataclass
class Stage2LossOutput:
    """Every weighted loss component and diagnostic count (guide 05 §15.4)."""

    total: torch.Tensor
    cls: torch.Tensor
    aux: torch.Tensor
    match: torch.Tensor
    trialmatch: torch.Tensor
    bankrank: torch.Tensor
    tokenmatch: torch.Tensor
    cons: torch.Tensor
    latent_cons: torch.Tensor
    prob_cons: torch.Tensor
    entropy: torch.Tensor
    anchor: torch.Tensor
    n_hc_match_trials: int
    n_skipped_match_trials: int
    matched_cosine_mean: float
    wrong_cosine_mean: float
    bank_rank_accuracy: float
