"""Grouped HC batch sampler (contract §3).

Each batch contains ``S`` unique stimuli and ``H`` distinct HC trials per
stimulus (``N = S * H``). Stimuli are shuffled per epoch with a local
generator seeded from SHA-256(global seed, fold, epoch), so every epoch is
reproducible.

Trial selection per stimulus is a deterministic rotation, not random
sampling: a fixed seeded permutation of the stimulus's subjects (or of its
rows, when no subject ids are supplied) is sliced by ``H`` at offset
``epoch * H`` with wraparound, and within a subject the subject's rows are
rotated across epochs. Over a run of ``E`` epochs every trial of every
stimulus is therefore trained ``floor(E * H / n)`` or ``ceil(E * H / n)``
times (``n`` = number of subjects/rows for that stimulus), so no trial is
starved. When the number of eligible stimuli is not a multiple of ``S``, a
final remainder batch keeps the leftover stimuli, so every eligible stimulus
participates in every epoch.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
from torch.utils.data import Sampler

from .masking import make_rng


class StimulusGroupedHCBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        stimulus_groups: dict[int, list[int]],
        *,
        stimuli_per_batch: int = 8,
        hc_per_stimulus: int = 8,
        min_hc_per_stimulus: int = 8,
        replacement: bool = False,
        seed: int = 2026,
        fold: int = 0,
        epoch: int = 0,
        subject_by_row: list[str] | None = None,
    ):
        if replacement:
            raise ValueError("replacement sampling is not supported (see guide)")
        if subject_by_row is not None:
            max_row = max(
                (r for rows in stimulus_groups.values() for r in rows), default=-1
            )
            if max_row >= len(subject_by_row):
                raise ValueError("subject_by_row must cover every row index in stimulus_groups")
        self.stimulus_groups = stimulus_groups
        self.stimuli_per_batch = stimuli_per_batch
        self.hc_per_stimulus = hc_per_stimulus
        self.min_hc_per_stimulus = min_hc_per_stimulus
        self.seed = seed
        self.fold = fold
        self.epoch = epoch

        self.eligible_stimuli = sorted(
            si for si, rows in stimulus_groups.items() if len(rows) >= min_hc_per_stimulus
        )
        self.n_skipped_stimuli = len(stimulus_groups) - len(self.eligible_stimuli)
        if stimuli_per_batch:
            n_full = len(self.eligible_stimuli) // stimuli_per_batch
            self._len = n_full + (1 if len(self.eligible_stimuli) % stimuli_per_batch else 0)
        else:
            self._len = 0

        # Fixed per-stimulus rotation order, independent of epoch: subjects
        # (or rows, without subject ids) in a seeded permutation, each slot
        # holding that subject's rows in a seeded order. Built once; epochs
        # only slide the window, which is what guarantees uniform coverage.
        self._slots_by_stimulus: dict[int, list[list[int]]] = {}
        for si, rows in stimulus_groups.items():
            if subject_by_row is not None:
                rows_by_subject: dict[str, list[int]] = {}
                for r in rows:
                    rows_by_subject.setdefault(subject_by_row[r], []).append(r)
                subjects = sorted(rows_by_subject)
                make_rng("grouped_hc_rotation", seed, fold, si).shuffle(subjects)
                slots: list[list[int]] = []
                for s in subjects:
                    s_rows = rows_by_subject[s]
                    make_rng("grouped_hc_rotation_rows", seed, fold, si, s).shuffle(s_rows)
                    slots.append(s_rows)
            else:
                rows = list(rows)
                make_rng("grouped_hc_rotation", seed, fold, si).shuffle(rows)
                slots = [[r] for r in rows]
            self._slots_by_stimulus[si] = slots

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self._len

    def _trials_for_stimulus(self, stimulus_index: int) -> list[int]:
        slots = self._slots_by_stimulus[stimulus_index]
        n = len(slots)
        if n < self.hc_per_stimulus:
            # Fewer subjects/rows than H: skip this stimulus for this epoch
            # (counted); eligibility guarantees >= min rows, but min may be
            # below H when configured so.
            self.n_skipped_stimuli += 1
            return []
        start = (self.epoch * self.hc_per_stimulus) % n
        chosen = [slots[(start + i) % n] for i in range(self.hc_per_stimulus)]
        # Within a slot (one subject's rows), rotate rows across epochs so
        # every row of the subject is trained too.
        return [s_rows[self.epoch % len(s_rows)] for s_rows in chosen]

    def __iter__(self) -> Iterator[list[int]]:
        rng = make_rng("grouped_hc_sampler", self.seed, self.fold, self.epoch)
        order = list(self.eligible_stimuli)
        rng.shuffle(order)
        n_batches = len(order) // self.stimuli_per_batch
        n_remainder = len(order) % self.stimuli_per_batch
        self.n_batches_this_epoch = n_batches + (1 if n_remainder else 0)
        for b in range(self.n_batches_this_epoch):
            batch_stimuli = order[b * self.stimuli_per_batch : (b + 1) * self.stimuli_per_batch]
            batch: list[int] = []
            for stimulus_index in batch_stimuli:
                batch.extend(self._trials_for_stimulus(stimulus_index))
            if len(batch) != len(batch_stimuli) * self.hc_per_stimulus:
                # A stimulus fell short; drop the partial batch (report via
                # n_skipped_stimuli) rather than duplicate trials.
                continue
            # Deterministic within-batch permutation (stimulus metadata stays
            # attached to each trial through the dataset's collate).
            rng.shuffle(batch)
            yield batch
