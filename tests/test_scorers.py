"""Testes de HistoryFeatureLookup e dos scorers (make_model_scorer/popularity)."""

import numpy as np
import pandas as pd

from src.data.feature_contract import FEATURE_COLS
from src.evaluation.scorers import (
    HistoryFeatureLookup,
    make_model_scorer,
    make_popularity_scorer,
)
from src.models.baselines import LogisticRecommender
from src.models.mlp import NCFRecommender


def _history() -> pd.DataFrame:
    # user 1 viu 10 (x2) e 11; user 2 viu 11 (x2). view_counts: 11=3 > 10=2.
    return pd.DataFrame(
        {
            "user_idx": [1, 1, 1, 2, 2],
            "item_idx": [10, 10, 11, 11, 11],
            "event": ["view"] * 5,
            "weight": [1, 1, 1, 1, 1],
            "timestamp": pd.to_datetime(
                ["2015-05-01", "2015-05-02", "2015-05-03", "2015-05-04", "2015-05-05"]
            ),
        }
    )


def test_catalog_and_known_users():
    lookup = HistoryFeatureLookup(_history())
    assert lookup.catalog.tolist() == [10, 11]
    assert lookup.known_users() == {1, 2}


def test_view_counts_and_pair_features():
    lookup = HistoryFeatureLookup(_history())
    assert lookup.view_counts(np.array([10, 11, 99])).tolist() == [2.0, 3.0, 0.0]

    counts, recency = lookup.pair_features(
        np.array([1, 1]),
        np.array([10, 99]),
        pd.to_datetime(["2015-05-03", "2015-05-03"]).to_numpy(),
    )
    assert counts.tolist() == [2.0, 0.0]
    assert recency[0] == 1.0  # 05-02 -> 05-03
    assert recency[1] == -1.0  # nunca visto (sentinela)


def test_matrix_has_contract_shape():
    lookup = HistoryFeatureLookup(_history())
    matrix = lookup.matrix(np.array([1, 2]), np.array([11, 10]))
    assert matrix.shape == (2, len(FEATURE_COLS))
    assert matrix.dtype == np.float32


def test_make_model_scorer_with_ncf_uses_head_argument():
    lookup = HistoryFeatureLookup(_history())
    model = NCFRecommender(n_users=3, n_items=12, epochs=1, embedding_dim=4).fit(
        lookup.matrix(np.array([1, 2]), np.array([10, 11])),
        np.array([1.0, 0.0], dtype="float32"),
    )
    scorer = make_model_scorer(model, lookup, head="strong")
    scores = scorer(np.array([1, 2]), np.array([10, 11]))
    assert scores.shape == (2,)
    assert ((scores >= 0.0) & (scores <= 1.0)).all()


def test_make_model_scorer_with_non_ncf_model_ignores_head():
    """Baselines (sem múltiplas cabeças) usam predict_proba(matrix) direto."""
    lookup = HistoryFeatureLookup(_history())
    matrix = lookup.matrix(np.array([1, 1, 2, 2]), np.array([10, 11, 10, 11]))
    model = LogisticRecommender(random_state=0).fit(
        matrix, np.array([1.0, 0.0, 0.0, 1.0], dtype="float32")
    )
    scorer = make_model_scorer(model, lookup)
    scores = scorer(np.array([1, 2]), np.array([10, 11]))
    assert scores.shape == (2,)


def test_make_popularity_scorer_returns_view_counts():
    lookup = HistoryFeatureLookup(_history())
    scorer = make_popularity_scorer(lookup)
    scores = scorer(np.array([1, 2]), np.array([10, 11]))
    assert scores.tolist() == [2.0, 3.0]
