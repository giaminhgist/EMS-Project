"""Five-fold subject-level cross-validation (Phase 4)."""

from .build_subject_folds import Population, build_assignments, fold_subject_partitions, fold_trial_partitions
from .config import CVConfig

__all__ = [
    "CVConfig",
    "Population",
    "build_assignments",
    "fold_subject_partitions",
    "fold_trial_partitions",
]
