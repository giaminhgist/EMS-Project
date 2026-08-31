"""Single seeded utility for Python, NumPy, and PyTorch RNGs (global contract §5)."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def seed_all(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) from one integer."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def capture_rng_state(device: str) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if device == "cuda" and torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])
