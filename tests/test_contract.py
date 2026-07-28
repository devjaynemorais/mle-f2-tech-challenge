"""Teste de contrato treino ↔ serving (FR-007, T035).

Garante que a matriz montada pelo serving e a montada pela avaliação
produzem o MESMO vetor para o mesmo (user, item, contexto), na ordem do
contrato de features.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.feature_contract import CONT_COLS, FEATURE_COLS, ID_COLS
from src.evaluation.scorers import HistoryFeatureLookup
from src.serving.recommender import RecommendationService
from src.serving.store import FeatureStore


class _CaptureModel:
    """Captura a matriz recebida e devolve scores constantes."""

    def __init__(self):
        self.captured = None

    def predict_proba(self, X):
        self.captured = X.copy()
        return np.linspace(1.0, 0.0, len(X))


@pytest.fixture()
def events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_idx": [1, 1, 1, 2],
            "item_idx": [10, 11, 10, 12],
            "event": ["view", "addtocart", "view", "view"],
            "weight": [1, 3, 1, 1],
            "timestamp": pd.to_datetime(
                ["2015-05-01", "2015-05-03", "2015-05-05", "2015-05-02"]
            ),
        }
    )


def test_contract_column_layout():
    assert FEATURE_COLS == ID_COLS + CONT_COLS
    assert FEATURE_COLS[:2] == ["user_idx", "item_idx"]


def test_serving_matrix_matches_history_lookup(events):
    """Mesmo (user, item) → mesmas features não-contextuais nos dois lados."""
    store = FeatureStore(events)
    model = _CaptureModel()
    service = RecommendationService(model, store, max_candidates=10)
    service.recommend(user_id=1, top_k=5)
    serving_matrix = model.captured
    assert serving_matrix is not None
    assert serving_matrix.shape[1] == len(FEATURE_COLS)

    lookup = HistoryFeatureLookup(events)
    users = serving_matrix[:, 0].astype("int64")
    items = serving_matrix[:, 1].astype("int64")
    eval_matrix = lookup.matrix(users, items)

    # hour/day_of_week são contexto (now vs. fim do histórico) — comparar o resto
    context_cols = {"hour", "day_of_week"}
    for pos, col in enumerate(FEATURE_COLS):
        if col in context_cols:
            continue
        assert serving_matrix[:, pos] == pytest.approx(eval_matrix[:, pos], rel=1e-6), (
            f"coluna divergente entre serving e avaliação: {col}"
        )


def test_serving_user_state_is_current(events):
    """Estado servido do usuário = agregação completa do histórico."""
    store = FeatureStore(events)
    freq, eng, rec = store.user_features(1)
    assert freq == 3.0  # 3 eventos do user 1
    assert eng == 5.0  # 1 + 3 + 1
    assert rec == 0.0  # último evento coincide com o fim do histórico
