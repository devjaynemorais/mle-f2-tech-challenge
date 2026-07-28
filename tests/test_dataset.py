"""Testes do RetailRocketDataset (construção a partir de DataFrame)."""

import pandas as pd
import torch

from src.data.dataset import RetailRocketDataset
from src.data.feature_contract import CONT_COLS, TARGET_COL


def test_from_dataframe_builds_matching_tensors():
    df = pd.DataFrame(
        {
            "user_idx": [0, 1],
            "item_idx": [5, 6],
            **{c: [0.0, 1.0] for c in CONT_COLS},
            TARGET_COL: [1.0, 0.0],
        }
    )
    dataset = RetailRocketDataset.from_dataframe(df)
    assert len(dataset) == 2
    users, items, cats, x_cont, y, y_view = dataset[0]
    assert users.item() == 0
    assert items.item() == 5
    assert cats.item() == 0  # default: sem categoria
    assert x_cont.shape == (len(CONT_COLS),)
    assert y.item() == 1.0
    assert y_view.item() == 0.0  # default: sem multi-task
    assert users.dtype == torch.long
    assert x_cont.dtype == torch.float32
