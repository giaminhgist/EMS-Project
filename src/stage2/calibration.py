"""Single-temperature calibration (guide 07 §15, contracts §20 Stage 2D).

One positive temperature is fitted only on permitted non-test predictions
(training-partition or inner-selection predictions under the pilot regime —
never the outer held-out fold). Provenance, sample IDs, fit scope, pre/post
Brier and NLL and a checksum are recorded. Disabled calibration still writes
an explicit metadata file.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F


class CalibrationError(ValueError):
    pass


@dataclass
class CalibrationState:
    temperature: float
    fit_scope: str
    subject_ids: list[str]
    n_subjects: int
    pre_brier: float
    post_brier: float
    pre_nll: float
    post_nll: float
    checksum: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "fit_scope": self.fit_scope,
            "subject_ids": self.subject_ids,
            "n_subjects": self.n_subjects,
            "pre_brier": self.pre_brier,
            "post_brier": self.post_brier,
            "pre_nll": self.pre_nll,
            "post_nll": self.post_nll,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CalibrationState":
        return cls(
            temperature=float(raw["temperature"]),
            fit_scope=str(raw["fit_scope"]),
            subject_ids=list(raw["subject_ids"]),
            n_subjects=int(raw["n_subjects"]),
            pre_brier=float(raw["pre_brier"]),
            post_brier=float(raw["post_brier"]),
            pre_nll=float(raw["pre_nll"]),
            post_nll=float(raw["post_nll"]),
            checksum=str(raw["checksum"]),
        )


def _nll(logits: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        logits / temperature, labels, reduction="mean"
    )


def _brier(logits: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    p = torch.sigmoid(logits / temperature)
    return ((p - labels) ** 2).mean()


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    subject_ids: list[str],
    fit_scope: str,
    grid: tuple[float, float, int] = (0.1, 5.0, 50),
) -> CalibrationState:
    """Fit one positive temperature minimizing NLL on permitted predictions.

    Never call this with outer held-out fold predictions under the pilot
    regime; ``fit_scope`` documents exactly which subjects were used.
    """
    logits = torch.as_tensor(logits, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.float32)
    if logits.shape != labels.shape or logits.dim() != 1:
        raise CalibrationError("calibration requires flat logits and labels")
    if logits.numel() == 0:
        raise CalibrationError("calibration requires at least one prediction")
    if len(subject_ids) != logits.numel():
        raise CalibrationError("subject_ids must align with predictions")

    lo, hi, steps = grid
    candidates = torch.linspace(lo, hi, steps)
    nlls = torch.stack([_nll(logits, labels, float(t)) for t in candidates])
    best = int(nlls.argmin())
    temperature = float(candidates[best])
    # Local parabolic refinement around the best grid point.
    if 0 < best < steps - 1:
        t0, t1, t2 = (float(candidates[best - 1 + k]) for k in range(3))
        n0, n1, n2 = (float(nlls[best - 1 + k]) for k in range(3))
        denom = n0 - 2.0 * n1 + n2
        if abs(denom) > 1e-12:
            shift = 0.5 * (n0 - n2) / denom * (t1 - t0)
            refined = t1 + shift
            if lo <= refined <= hi and float(_nll(logits, labels, refined)) < n1:
                temperature = refined

    pre_brier = float(_brier(logits, labels, 1.0))
    post_brier = float(_brier(logits, labels, temperature))
    pre_nll = float(_nll(logits, labels, 1.0))
    post_nll = float(_nll(logits, labels, temperature))
    payload = {
        "temperature": temperature,
        "fit_scope": fit_scope,
        "subject_ids": sorted(subject_ids),
        "n_subjects": int(logits.numel()),
        "pre_brier": pre_brier,
        "post_brier": post_brier,
        "pre_nll": pre_nll,
        "post_nll": post_nll,
    }
    checksum = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return CalibrationState(
        temperature=temperature,
        fit_scope=fit_scope,
        subject_ids=sorted(subject_ids),
        n_subjects=int(logits.numel()),
        pre_brier=pre_brier,
        post_brier=post_brier,
        pre_nll=pre_nll,
        post_nll=post_nll,
        checksum=checksum,
    )


def calibration_disabled_metadata() -> dict[str, Any]:
    """Explicit provenance when calibration is off (guide 07 §15)."""
    return {"calibration": "disabled", "reason": "validation.calibrate=false"}
