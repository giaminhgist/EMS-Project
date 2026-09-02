"""Subject batch contract, collation and valid-trial flatten/scatter (guide 04 §6-§8)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .contracts import N_STIMULI, Stage2SubjectSample


@dataclass
class Stage2Batch:
    """One collated subject batch; trials are never flattened here."""

    subject_ids: tuple[str, ...]
    labels: torch.Tensor  # [B] float32
    heatmaps: torch.Tensor  # [B,100,3,48,64] float32
    trial_mask: torch.Tensor  # [B,100] bool
    stimulus_indices: torch.Tensor  # [B,100] int64 (canonical 0..99)
    category_ids: torch.Tensor  # [B,100] int64
    bank_split_ids: torch.Tensor | None  # [B] int64 or None
    trial_uids: tuple[tuple[str | None, ...], ...]

    @property
    def n_subjects(self) -> int:
        return self.heatmaps.shape[0]

    @property
    def n_valid_trials(self) -> int:
        return int(self.trial_mask.sum().item())

    def to_device(self, device: torch.device | str) -> "Stage2Batch":
        self.labels = self.labels.to(device)
        self.heatmaps = self.heatmaps.to(device)
        self.trial_mask = self.trial_mask.to(device)
        self.stimulus_indices = self.stimulus_indices.to(device)
        self.category_ids = self.category_ids.to(device)
        if self.bank_split_ids is not None:
            self.bank_split_ids = self.bank_split_ids.to(device)
        return self

    def validate(self) -> None:
        b = self.n_subjects
        if b == 0:
            raise ValueError("empty subject batch")
        if self.labels.shape != (b,) or self.labels.dtype != torch.float32:
            raise ValueError(f"labels shape/dtype invalid: {tuple(self.labels.shape)} {self.labels.dtype}")
        labels_i = self.labels.long()
        if not torch.all((labels_i == 0) | (labels_i == 1)):
            raise ValueError("labels must be binary")
        if self.heatmaps.shape != (b, N_STIMULI, 3, 48, 64) or self.heatmaps.dtype != torch.float32:
            raise ValueError(f"heatmaps shape/dtype invalid: {tuple(self.heatmaps.shape)} {self.heatmaps.dtype}")
        if self.trial_mask.shape != (b, N_STIMULI) or self.trial_mask.dtype != torch.bool:
            raise ValueError("trial_mask shape/dtype invalid")
        if not torch.all(self.trial_mask.any(dim=1)):
            raise ValueError("every subject must have at least one valid trial")
        if self.stimulus_indices.shape != (b, N_STIMULI):
            raise ValueError("stimulus_indices shape invalid")
        if not torch.equal(self.stimulus_indices, torch.arange(N_STIMULI).expand(b, -1)):
            raise ValueError("stimulus_indices must be canonical 0..99 slots")
        if self.category_ids.shape != (b, N_STIMULI):
            raise ValueError("category_ids shape invalid")
        if not torch.all((self.category_ids >= 0) & (self.category_ids <= 3)):
            raise ValueError("category_ids must be in 0..3")
        if self.bank_split_ids is not None and self.bank_split_ids.shape != (b,):
            raise ValueError("bank_split_ids shape invalid")
        if len(self.trial_uids) != b or any(len(u) != N_STIMULI for u in self.trial_uids):
            raise ValueError("trial_uids must be B tuples of length 100")


def collate_subject_samples(samples: list[Stage2SubjectSample]) -> Stage2Batch:
    """Stack subject samples into one typed batch (guide §7)."""
    heatmaps = torch.stack([s.heatmaps for s in samples])
    labels = torch.tensor([s.label for s in samples], dtype=torch.float32)
    trial_mask = torch.stack([s.trial_mask for s in samples])
    stimulus_indices = torch.stack([s.stimulus_indices for s in samples])
    category_ids = torch.stack([s.category_ids for s in samples])
    split_ids = [s.bank_split_id for s in samples]
    if any(sid is None for sid in split_ids):
        if any(sid is not None for sid in split_ids):
            raise ValueError("mixed bank_split_id presence within one batch")
        bank_split_ids = None
    else:
        bank_split_ids = torch.tensor(split_ids, dtype=torch.int64)
    return Stage2Batch(
        subject_ids=tuple(s.subject_id for s in samples),
        labels=labels,
        heatmaps=heatmaps,
        trial_mask=trial_mask,
        stimulus_indices=stimulus_indices,
        category_ids=category_ids,
        bank_split_ids=bank_split_ids,
        trial_uids=tuple(s.trial_uids for s in samples),
    )


@dataclass
class FlattenedTrials:
    """Row-major flattening of the valid trials in a subject batch."""

    heatmaps: torch.Tensor  # [N,3,48,64] float32
    stimulus_indices: torch.Tensor  # [N] int64
    category_ids: torch.Tensor  # [N] int64
    subject_slots: torch.Tensor  # [N] int64 (row index into the subject batch)
    stimulus_slots: torch.Tensor  # [N] int64 (canonical stimulus 0..99)
    labels_per_trial: torch.Tensor  # [N] int64 — only for HC selection metadata, never trial BCE
    bank_split_ids: torch.Tensor | None  # [N] int64 or None


def flatten_valid_trials(batch: Stage2Batch) -> FlattenedTrials:
    """Flatten only valid (masked) trials in row-major [B,100] order.

    Missing trials remain missing and are excluded from every downstream
    softmax, gather and loss.
    """
    batch.validate()
    subject_slots, stimulus_slots = torch.nonzero(batch.trial_mask, as_tuple=True)
    labels_expanded = batch.labels.long().unsqueeze(1).expand(-1, N_STIMULI).contiguous()
    bank_expanded = None
    if batch.bank_split_ids is not None:
        bank_expanded = batch.bank_split_ids.unsqueeze(1).expand(-1, N_STIMULI).contiguous()
    return FlattenedTrials(
        heatmaps=batch.heatmaps[batch.trial_mask],
        stimulus_indices=batch.stimulus_indices[batch.trial_mask],
        category_ids=batch.category_ids[batch.trial_mask],
        subject_slots=subject_slots,
        stimulus_slots=stimulus_slots,
        labels_per_trial=labels_expanded[batch.trial_mask],
        bank_split_ids=bank_expanded[batch.trial_mask] if bank_expanded is not None else None,
    )


def scatter_valid_trials(
    flat_tensor: torch.Tensor,
    subject_slots: torch.Tensor,
    stimulus_slots: torch.Tensor,
    batch_size: int,
    n_stimuli: int = N_STIMULI,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of the valid-trial flattening.

    Returns ``(padded [B, S, *feature], mask [B, S])`` where
    ``padded[mask] == flat_tensor`` in row-major order and missing slots stay
    exactly zero.
    """
    n = flat_tensor.shape[0]
    if subject_slots.shape != (n,) or stimulus_slots.shape != (n,):
        raise ValueError("slot tensors must have length equal to the flat leading dim")
    mask = torch.zeros(batch_size, n_stimuli, dtype=torch.bool, device=flat_tensor.device)
    mask[subject_slots, stimulus_slots] = True
    padded = torch.zeros(
        (batch_size, n_stimuli) + tuple(flat_tensor.shape[1:]),
        dtype=flat_tensor.dtype,
        device=flat_tensor.device,
    )
    padded[subject_slots, stimulus_slots] = flat_tensor
    return padded, mask
