"""Deterministic balanced subject sampler (Phase 2, guide 04 §9).

Batches subject indices only; trials are never sampled. HC and SZ subject
lists are shuffled independently with ``(seed, fold, epoch)``, paired
per-batch at ``batch_size // 2`` per group, and topped up from the remaining
group only when one side is exhausted — no subject is silently discarded
merely to obtain perfect class balance. Validation uses a fixed order.
"""

from __future__ import annotations

import math
from typing import Any, Iterator

import numpy as np


class SamplerError(ValueError):
    pass


class BalancedSubjectBatchSampler:
    """BatchSampler over subject indices with deterministic epoch shuffling.

    batch_size=4 -> 2 HC + 2 SZ per batch; batch_size=8 -> 4 HC + 4 SZ.
    """

    def __init__(
        self,
        dataset: Any,
        *,
        batch_size: int,
        seed: int,
        fold: int,
        epoch: int = 0,
        balance_groups: bool = True,
        drop_last: bool = False,
        shuffle: bool = True,
    ):
        if batch_size <= 0:
            raise SamplerError("batch_size must be positive")
        if balance_groups and batch_size % 2 != 0:
            raise SamplerError(
                f"balanced subject batching requires an even batch_size, got {batch_size}"
            )
        n = len(dataset)
        labels = np.asarray(dataset.subject_labels(), dtype=np.int64)
        if labels.shape != (n,):
            raise SamplerError("dataset.subject_labels() length does not match dataset size")
        if not set(np.unique(labels)) <= {0, 1}:
            raise SamplerError("subject labels must be binary {0, 1}")
        self.n_subjects = n
        self.batch_size = batch_size
        self.balance_groups = balance_groups
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.fold = fold
        self._epoch = epoch
        self.hc_indices = np.where(labels == 0)[0]
        self.sz_indices = np.where(labels == 1)[0]
        # Composition of every batch yielded by the last __iter__ call.
        self.last_batch_compositions: list[tuple[int, int]] = []

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    @property
    def epoch(self) -> int:
        return self._epoch

    def state_dict(self) -> dict[str, Any]:
        return {"seed": self.seed, "fold": self.fold, "epoch": self._epoch}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("seed") != self.seed or state.get("fold") != self.fold:
            raise SamplerError("sampler state seed/fold does not match")
        self.set_epoch(int(state["epoch"]))

    def _rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed * 31 + self.fold * 7 + self._epoch)

    def _iterate_balanced(self, hc: np.ndarray, sz: np.ndarray) -> Iterator[list[int]]:
        per = self.batch_size // 2
        i = j = 0
        while i < len(hc) or j < len(sz):
            batch: list[int] = []
            take_hc = min(per, len(hc) - i)
            batch += [int(x) for x in hc[i : i + take_hc]]
            i += take_hc
            take_sz = min(per, len(sz) - j)
            batch += [int(x) for x in sz[j : j + take_sz]]
            j += take_sz
            if len(batch) < self.batch_size and (i < len(hc) or j < len(sz)):
                # One group exhausted: top up from the other without duplication.
                if i < len(hc):
                    k = min(self.batch_size - len(batch), len(hc) - i)
                    batch += [int(x) for x in hc[i : i + k]]
                    i += k
                else:
                    k = min(self.batch_size - len(batch), len(sz) - j)
                    batch += [int(x) for x in sz[j : j + k]]
                    j += k
            if self.drop_last and len(batch) < self.batch_size:
                break
            yield batch

    def __iter__(self) -> Iterator[list[int]]:
        self.last_batch_compositions = []
        if not self.shuffle:
            for start in range(0, self.n_subjects, self.batch_size):
                batch = list(range(start, min(start + self.batch_size, self.n_subjects)))
                if self.drop_last and len(batch) < self.batch_size:
                    break
                yield batch
            return
        rng = self._rng()
        if not self.balance_groups:
            order = rng.permutation(self.n_subjects)
            for start in range(0, self.n_subjects, self.batch_size):
                batch = [int(x) for x in order[start : start + self.batch_size]]
                if self.drop_last and len(batch) < self.batch_size:
                    break
                yield batch
            return
        hc = self.hc_indices[rng.permutation(len(self.hc_indices))]
        sz = self.sz_indices[rng.permutation(len(self.sz_indices))]
        hc_set = set(int(x) for x in self.hc_indices)
        for batch in self._iterate_balanced(hc, sz):
            n_hc = sum(1 for i in batch if i in hc_set)
            self.last_batch_compositions.append((n_hc, len(batch) - n_hc))
            yield batch

    def _n_batches(self) -> int:
        if not self.shuffle:
            total = math.ceil(self.n_subjects / self.batch_size)
            if self.drop_last and self.n_subjects % self.batch_size:
                total -= 1
            return total
        if not self.balance_groups:
            total = math.ceil(self.n_subjects / self.batch_size)
            if self.drop_last and self.n_subjects % self.batch_size:
                total -= 1
            return total
        per = self.batch_size // 2
        hc_n, sz_n = len(self.hc_indices), len(self.sz_indices)
        i = j = count = 0
        while i < hc_n or j < sz_n:
            b = 0
            take_hc = min(per, hc_n - i)
            i += take_hc
            b += take_hc
            take_sz = min(per, sz_n - j)
            j += take_sz
            b += take_sz
            if b < self.batch_size and (i < hc_n or j < sz_n):
                if i < hc_n:
                    k = min(self.batch_size - b, hc_n - i)
                    i += k
                    b += k
                else:
                    k = min(self.batch_size - b, sz_n - j)
                    j += k
                    b += k
            if self.drop_last and b < self.batch_size:
                break
            count += 1
        return count

    def __len__(self) -> int:
        return self._n_batches()
