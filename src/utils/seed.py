"""Reproducibility helpers — fix all random seeds in one call."""

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for Python, NumPy and PyTorch.

    Args:
        seed: Integer seed value (default matches params.yaml random_state).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
