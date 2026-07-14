"""PyTorch Dataset para o RetailRocket recommendation pipeline.

Contrato dos tensores (specs/001-recommender-quality/data-model.md):
ids como ``long`` (índices de embedding) e contínuas como ``float32``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.data.feature_contract import CONT_COLS, TARGET_COL


class RetailRocketDataset(Dataset):
    """Dataset de interações rotuladas para o NCF.

    Cada amostra é a tupla ``(user_idx, item_idx, cat_idx, x_cont, y)`` com
    ids em ``long`` e contínuas/alvo em ``float32``.

    Args:
        users: Índices de usuário (já roteados para o vocabulário do modelo).
        items: Índices de item.
        x_cont: Matriz de features contínuas (já escaladas).
        y: Rótulos binários.
        cats: Índices de categoria do item (default: zeros).
    """

    def __init__(
        self,
        users: np.ndarray,
        items: np.ndarray,
        x_cont: np.ndarray,
        y: np.ndarray,
        cats: np.ndarray | None = None,
    ) -> None:
        self._users = torch.as_tensor(np.asarray(users), dtype=torch.long)
        self._items = torch.as_tensor(np.asarray(items), dtype=torch.long)
        cats = cats if cats is not None else np.zeros(len(users), dtype="int64")
        self._cats = torch.as_tensor(np.asarray(cats), dtype=torch.long)
        self._x_cont = torch.as_tensor(np.asarray(x_cont), dtype=torch.float32)
        self._y = torch.as_tensor(np.asarray(y), dtype=torch.float32)

    @classmethod
    def from_dataframe(
        cls, df: pd.DataFrame, target_col: str = TARGET_COL
    ) -> RetailRocketDataset:
        """Constrói o dataset a partir de um DataFrame rotulado (contrato).

        Args:
            df: DataFrame com FEATURE_COLS e a coluna alvo.
            target_col: Nome da coluna alvo.

        Returns:
            Instância de RetailRocketDataset.
        """
        return cls(
            users=df["user_idx"].to_numpy(dtype="int64"),
            items=df["item_idx"].to_numpy(dtype="int64"),
            x_cont=df[CONT_COLS].to_numpy(dtype="float32"),
            y=df[target_col].to_numpy(dtype="float32"),
        )

    def __len__(self) -> int:
        """Retorna número de amostras no dataset."""
        return len(self._y)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Retorna (user_idx, item_idx, cat_idx, x_cont, y) para o índice."""
        return (
            self._users[idx],
            self._items[idx],
            self._cats[idx],
            self._x_cont[idx],
            self._y[idx],
        )
