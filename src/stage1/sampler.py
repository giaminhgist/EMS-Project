"""Grouped HC batch sampler (contract §3).

Each batch contains ``S`` unique stimuli and ``H`` distinct HC trials per
stimulus (``N = S * H``). Stimuli and per-stimulus trials are sampled without
replacement per epoch; the local generator is seeded from
SHA-256(global seed, fold, epoch) so every epoch is reproducible.
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
    ):
        if replacement:
            raise ValueError("replacement sampling is not supported (see guide)")
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
        self._len = len(self.eligible_stimuli) // stimuli_per_batch if stimuli_per_batch else 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self._len

    def __iter__(self) -> Iterator[list[int]]:
        rng = make_rng("grouped_hc_sampler", self.seed, self.fold, self.epoch)
        order = list(self.eligible_stimuli)
        rng.shuffle(order)
        n_batches = len(order) // self.stimuli_per_batch
        self.n_batches_this_epoch = n_batches
        for b in range(n_batches):
            batch: list[int] = []
            for stimulus_index in order[b * self.stimuli_per_batch : (b + 1) * self.stimuli_per_batch]:
                rows = list(self.stimulus_groups[stimulus_index])
                if len(rows) < self.hc_per_stimulus:
                    # Stimuli with fewer than H trials are skipped per epoch
                    # (counted); eligible filtering guarantees >= min, but min
                    # may be below H when configured so.
                    self.n_skipped_stimuli += 1
                    continue
                chosen = rng.choice(np.asarray(rows, dtype=np.int64), size=self.hc_per_stimulus, replace=False)
                batch.extend(int(x) for x in chosen)
            if len(batch) != self.stimuli_per_batch * self.hc_per_stimulus:
                # A stimulus fell short; drop the partial batch (report via
                # n_skipped_stimuli) rather than duplicate trials.
                continue
            # Deterministic within-batch permutation (stimulus metadata stays
            # attached to each trial through the dataset's collate).
            rng.shuffle(batch)
            yield batch
