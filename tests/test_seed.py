"""Testes do helper de reprodutibilidade (src/utils/seed.py)."""

import random

import numpy as np
import pytest
import torch

from src.utils.seed import set_seed


def test_set_seed_makes_random_calls_deterministic():
    set_seed(123)
    a = (random.random(), np.random.rand(), torch.rand(1).item())
    set_seed(123)
    b = (random.random(), np.random.rand(), torch.rand(1).item())
    assert a == b


def test_set_seed_seeds_cuda_when_available(monkeypatch: pytest.MonkeyPatch):
    # torch.manual_seed já dispara manual_seed_all internamente quando CUDA
    # está "disponível" — o que importa é que nosso branch explícito também
    # chama, sempre com a seed correta (pode ser >1 chamada).
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: calls.append(seed))
    set_seed(7)
    assert calls
    assert all(c == 7 for c in calls)
