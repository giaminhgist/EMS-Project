"""Subject-level aggregation: stimulus/category embeddings, category-balanced
gated attention, one subject Transformer layer, main and additive-evidence
heads (guide 05 §8-§12, contracts §14.5-§14.8 and §15).

The aggregator consumes an :class:`EncodedTrials` cache so the frozen encoder
is never rerun for subset passes; only this module is rerun for full/A/B.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .contracts import (
    D_MODEL,
    N_STIMULI,
    N_TOKEN_CELLS,
    EncodedTrials,
    Stage2ForwardOutput,
)

N_CATEGORIES = 4


class StimulusCategoryEmbeddings(nn.Module):
    """Learned stimulus (100) and category (4) embedding tables + LayerNorm.

    Tables are added to the scattered trial tokens; swapping stimulus IDs
    while holding heatmaps fixed must change the embedding selection.
    """

    def __init__(self, dim: int = D_MODEL):
        super().__init__()
        self.stimulus_table = nn.Embedding(N_STIMULI, dim)
        self.category_table = nn.Embedding(N_CATEGORIES, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(
        self, z_scattered: torch.Tensor, stimulus_slots: torch.Tensor, category_ids: torch.Tensor
    ) -> torch.Tensor:
        """``z_scattered [B,100,D]`` + tables -> LayerNorm -> ``[B,100,D]``."""
        z = (
            z_scattered
            + self.stimulus_table(stimulus_slots)
            + self.category_table(category_ids)
        )
        return self.norm(z)


class CategoryBalancedGatedAttention(nn.Module):
    """Per-category gated attention with equal mass per present category.

    ``g = w^T [tanh(Vz) * sigmoid(Uz)]``; softmax separately over valid
    stimuli of each category; global importance ``I = alpha / K_i`` where
    ``K_i`` is the number of present categories. A category with no observed
    stimulus uses a learned missing-category token and is excluded from
    importance normalization.

    ``balanced=False`` is the ``no_category_balance`` ablation: one masked
    softmax over all valid stimuli, and category tokens are the same global
    weights restricted to each category's members.
    """

    def __init__(self, dim: int = D_MODEL, gate_dim: int = 64, balanced: bool = True):
        super().__init__()
        self.balanced = bool(balanced)
        self.V = nn.Linear(dim, gate_dim)
        self.U = nn.Linear(dim, gate_dim)
        self.w = nn.Linear(gate_dim, 1)
        self.missing_category_token = nn.Parameter(torch.zeros(dim))

    def forward(
        self,
        z: torch.Tensor,  # [B,100,D]
        category_ids: torch.Tensor,  # [B,100] int64
        mask: torch.Tensor,  # [B,100] bool — effective trial mask
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gate = torch.tanh(self.V(z)) * torch.sigmoid(self.U(z))  # [B,100,64]
        g = self.w(gate).squeeze(-1)  # [B,100]
        g = g.masked_fill(~mask, float("-inf"))

        b = z.shape[0]
        category_present = torch.zeros(b, N_CATEGORIES, dtype=torch.bool, device=z.device)
        for k in range(N_CATEGORIES):
            category_present[:, k] = (mask & (category_ids == k)).any(dim=1)

        if not self.balanced:
            # One masked softmax over all valid stimuli.
            alpha = torch.softmax(g, dim=1)  # [B,100]
            importance = alpha * mask
            category_tokens = torch.empty(
                b, N_CATEGORIES, z.shape[-1], device=z.device, dtype=z.dtype
            )
            for k in range(N_CATEGORIES):
                in_cat = category_ids == k
                token = ((alpha.unsqueeze(-1) * z) * in_cat.unsqueeze(-1)).sum(dim=1)
                token = torch.where(
                    category_present[:, k].unsqueeze(-1),
                    token,
                    self.missing_category_token.unsqueeze(0),
                )
                category_tokens[:, k] = token
            return alpha, importance, category_tokens, category_present

        alpha = torch.zeros_like(g)  # [B,100] within-category attention
        category_tokens = torch.empty(b, N_CATEGORIES, z.shape[-1], device=z.device, dtype=z.dtype)
        for k in range(N_CATEGORIES):
            in_cat = category_ids == k
            present = in_cat & mask
            has_any = present.any(dim=1)
            scores = torch.where(present, g, torch.full_like(g, float("-inf")))
            # softmax over an all--inf row is NaN; absent categories must be
            # exactly zero instead of polluting the attention panels.
            w_cat = torch.where(
                has_any.unsqueeze(1), torch.softmax(scores, dim=1), torch.zeros_like(scores)
            )  # [B,100]
            alpha = alpha + torch.where(in_cat, w_cat, torch.zeros_like(w_cat))
            token = (w_cat.unsqueeze(-1) * z).sum(dim=1)  # [B,D]
            token = torch.where(
                has_any.unsqueeze(-1), token, self.missing_category_token.unsqueeze(0)
            )
            category_tokens[:, k] = token

        n_present = category_present.sum(dim=1, keepdim=True).clamp_min(1).to(z.dtype)  # [B,1]
        importance = alpha / n_present  # [B,100] — equal total mass per present category
        importance = importance * mask  # excluded slots are exactly zero
        return alpha, importance, category_tokens, category_present


class SubjectTransformerAggregator(nn.Module):
    """One learned subject token + four category tokens through one Transformer
    layer (guide §10). Missing-category positions are attention-masked so
    padding is never exposed as a real category response.

    ``layers=0`` is the ``mean_subject_pooling`` ablation: the subject
    embedding is the category-balanced weighted mean of the present category
    tokens and no Transformer parameters exist.
    """

    def __init__(
        self,
        dim: int = D_MODEL,
        heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.25,
        layers: int = 1,
    ):
        super().__init__()
        if layers not in (0, 1):
            raise ValueError("subject transformer layers must be 0 or 1")
        if dim % heads != 0:
            raise ValueError(f"dim {dim} not divisible by heads {heads}")
        self.layers = layers
        if layers >= 1:
            self.subject_token = nn.Parameter(torch.zeros(dim))
            self.layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )

    def forward(
        self, category_tokens: torch.Tensor, category_present: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``category_tokens [B,4,D]`` -> ``(subject_embedding [B,D], out [B,5,D])``."""
        b = category_tokens.shape[0]
        if self.layers == 0:
            n_present = category_present.sum(dim=1, keepdim=True).clamp_min(1).to(
                category_tokens.dtype
            )
            subject_embedding = (
                category_tokens * category_present.unsqueeze(-1)
            ).sum(dim=1) / n_present  # [B,D]
            out = torch.cat([subject_embedding.unsqueeze(1), category_tokens], dim=1)
            return subject_embedding, out
        subject = self.subject_token.expand(b, 1, -1)
        x = torch.cat([subject, category_tokens], dim=1)  # [B,5,D]
        # key_padding_mask: True = padded position to be ignored.
        pad = torch.cat(
            [torch.zeros(b, 1, dtype=torch.bool, device=x.device), ~category_present], dim=1
        )  # [B,5]
        x = self.layer(x, src_key_padding_mask=pad)
        return x[:, 0], x


class MainHead(nn.Module):
    """Subject embedding -> scalar main logit."""

    def __init__(self, dim: int = D_MODEL):
        super().__init__()
        self.head = nn.Linear(dim, 1)

    def forward(self, subject_embedding: torch.Tensor) -> torch.Tensor:
        return self.head(subject_embedding).squeeze(-1)  # [B]


class AdditiveEvidenceHead(nn.Module):
    """Shared MLP over trial tokens: ``aux_logit = bias + masked_sum(I*e)``.

    Positive contribution supports SZ; negative supports HC. Missing-slot
    contributions are exactly zero.
    """

    def __init__(self, dim: int = D_MODEL, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, z: torch.Tensor, importance: torch.Tensor, mask: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """``z [B,100,D]`` -> ``(aux_logit [B], evidence [B,100], contribution [B,100])``."""
        evidence = self.mlp(z).squeeze(-1)  # [B,100]
        evidence = evidence * mask
        contribution = importance * evidence  # [B,100]; missing slots stay zero
        aux_logit = self.bias + contribution.sum(dim=1)  # [B]
        return aux_logit, evidence, contribution


class SubjectAggregator(nn.Module):
    """Full subject-level aggregation over one :class:`EncodedTrials` cache.

    Produces a complete :class:`Stage2ForwardOutput` for one effective trial
    mask (the full panel or a category-stratified subset).
    """

    def __init__(
        self,
        *,
        dim: int = D_MODEL,
        heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.25,
        transformer_layers: int = 1,
        category_balanced: bool = True,
    ):
        super().__init__()
        self.embeddings = StimulusCategoryEmbeddings(dim)
        self.gated_attention = CategoryBalancedGatedAttention(dim, balanced=category_balanced)
        self.transformer = SubjectTransformerAggregator(
            dim=dim, heads=heads, ffn_dim=ffn_dim, dropout=dropout, layers=transformer_layers
        )
        self.main_head = MainHead(dim)
        self.evidence_head = AdditiveEvidenceHead(dim)

    def _scatter(
        self, flat: torch.Tensor, enc: EncodedTrials, feature: tuple[int, ...] = (D_MODEL,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = torch.zeros(
            (enc.batch_size, N_STIMULI) + feature,
            dtype=flat.dtype,
            device=flat.device,
        )
        out[enc.subject_slots, enc.stimulus_slots] = flat
        mask = torch.zeros(enc.batch_size, N_STIMULI, dtype=torch.bool, device=flat.device)
        mask[enc.subject_slots, enc.stimulus_slots] = True
        return out, mask

    def forward(
        self,
        enc: EncodedTrials,
        subset_mask: torch.Tensor | None = None,
    ) -> Stage2ForwardOutput:
        z_source = enc.z_extended if enc.z_extended is not None else enc.z_trial
        z_scattered, seen = self._scatter(z_source, enc)
        effective_mask = enc.trial_mask & (subset_mask if subset_mask is not None else True)
        if not torch.equal(enc.trial_mask, seen):
            raise ValueError("encoded trials do not reproduce the batch trial mask")
        if not torch.all(effective_mask.any(dim=1)):
            raise ValueError("every subject must keep at least one trial in the effective mask")

        panel_indices = torch.arange(N_STIMULI, device=z_scattered.device).expand(
            enc.batch_size, -1
        )
        z = self.embeddings(z_scattered, panel_indices, enc.category_ids_panel)
        # Missing slots must not leak into any downstream operation.
        z = z * effective_mask.unsqueeze(-1)

        alpha, importance, category_tokens, category_present = self.gated_attention(
            z, enc.category_ids_panel, effective_mask
        )
        subject_embedding, transformer_out = self.transformer(category_tokens, category_present)
        main_logit = self.main_head(subject_embedding)
        aux_logit, evidence, contribution = self.evidence_head(z, importance, effective_mask)

        # Interpretation outputs; missing slots are exactly zero.
        patch_attention_scattered, _ = self._scatter(
            enc.patch_attention, enc, feature=(N_TOKEN_CELLS,)
        )
        cosine_scattered, _ = self._scatter(enc.cosine.squeeze(-1), enc, feature=())
        rho_scattered, _ = self._scatter(enc.rho.squeeze(-1), enc, feature=())
        cosine_scattered = cosine_scattered * effective_mask
        rho_scattered = rho_scattered * effective_mask
        deviation = rho_scattered * (1.0 - cosine_scattered)
        weighted_deviation = importance * deviation

        semantic_patch_map = None
        if enc.token_map_flat is not None:
            token_map_scattered, _ = self._scatter(enc.token_map_flat, enc, feature=(N_TOKEN_CELLS,))
            # M = I * A_bar * rho * (1 - c) (contracts §16.3); importance is
            # already zero outside the effective mask.
            token_map_scattered = token_map_scattered * importance.unsqueeze(-1)
            semantic_patch_map = token_map_scattered.reshape(enc.batch_size, N_STIMULI, 12, 16)

        return Stage2ForwardOutput(
            main_logit=main_logit,
            auxiliary_logit=aux_logit,
            subject_embedding=subject_embedding,
            trial_embeddings=z,
            trial_mask=effective_mask,
            query_patch_attention=patch_attention_scattered * effective_mask.unsqueeze(-1),
            stimulus_attention=alpha * effective_mask,
            stimulus_importance=importance,
            stimulus_evidence=evidence,
            stimulus_contribution=contribution,
            semantic_compatibility=cosine_scattered,
            normative_deviation=deviation,
            weighted_normative_deviation=weighted_deviation,
            semantic_patch_map=semantic_patch_map,
            diagnostics={
                "category_tokens": category_tokens,
                "category_present": category_present,
                "transformer_output": transformer_out,
                "trial_rho": rho_scattered,
                "trial_cosine": cosine_scattered,
            },
        )
