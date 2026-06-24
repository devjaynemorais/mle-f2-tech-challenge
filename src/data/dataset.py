"""PyTorch Dataset para o RetailRocket recommendation pipeline."""

from __future__ import annotations

import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


class RetailRocketDataset(Dataset):
    """Dataset de interações usuário-item para treinamento do modelo.

    Args:
        df: DataFrame com features numéricas e coluna alvo.
        feature_cols: Nomes das colunas usadas como entrada do modelo.
        target_col: Nome da coluna alvo (rótulo binário 0/1).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
    ) -> None:
        self._X = torch.tensor(
            df[feature_cols].to_numpy(dtype="float32"), dtype=torch.float32
        )
        self._y = torch.tensor(
            df[target_col].to_numpy(dtype="float32"), dtype=torch.float32
        )

    def __len__(self) -> int:
        """Retorna número de amostras no dataset."""
        return len(self._y)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Retorna par (features, rótulo) para o índice dado."""
        return self._X[idx], self._y[idx]
