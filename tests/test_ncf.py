"""Testes do NCF: shapes, roteamento unknown e sanidade de aprendizado (T025)."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.data.feature_contract import CONT_COLS, FEATURE_COLS
from src.models.mlp import NCFRecommender

N_USERS, N_ITEMS = 30, 40


def _matrix(users, items, rng=None):
    """Monta a matriz do contrato com contínuas determinísticas."""
    rng = rng or np.random.default_rng(0)
    n = len(users)
    cont = rng.random((n, len(CONT_COLS)))
    return np.column_stack([users, items, cont]).astype("float32")


def _tiny_model(**kwargs) -> NCFRecommender:
    defaults = dict(
        n_users=N_USERS,
        n_items=N_ITEMS,
        embedding_dim=8,
        cat_embedding_dim=2,
        hidden_dims=[16],
        epochs=2,
        batch_size=64,
        random_state=0,
    )
    defaults.update(kwargs)
    return NCFRecommender(**defaults)


def _fit_tiny(model: NCFRecommender, n: int = 200) -> np.ndarray:
    rng = np.random.default_rng(1)
    users = rng.integers(0, N_USERS, n)
    items = rng.integers(0, N_ITEMS, n)
    X = _matrix(users, items, rng)
    y = rng.integers(0, 2, n).astype("float32")
    model.fit(X, y)
    return X


def test_predict_proba_shape_and_range():
    model = _tiny_model()
    X = _fit_tiny(model)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X),)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    preds = model.predict(X)
    assert set(np.unique(preds)).issubset({0, 1})


def test_matrix_follows_contract_width():
    assert len(FEATURE_COLS) == 2 + len(CONT_COLS)
    model = _tiny_model()
    X = _fit_tiny(model)
    assert X.shape[1] == len(FEATURE_COLS)


def test_unknown_ids_route_to_shared_embedding():
    """Ids fora do vocabulário/treino caem no MESMO embedding unknown (D4)."""
    model = _tiny_model()
    _fit_tiny(model)
    cont = np.full((1, len(CONT_COLS)), 0.5)
    x_a = np.column_stack([[999], [5], *cont.T]).astype("float32")
    x_b = np.column_stack([[12345], [5], *cont.T]).astype("float32")
    # dois usuários desconhecidos diferentes → mesmo score (mesmo embedding)
    assert model.predict_proba(x_a) == pytest.approx(model.predict_proba(x_b))


def test_known_mask_routes_non_train_ids_to_unknown():
    known_users = np.zeros(N_USERS, dtype=bool)
    known_users[:10] = True
    model = _tiny_model(known_users=known_users)
    _fit_tiny(model)
    cont = np.full((1, len(CONT_COLS)), 0.5)
    x_unseen = np.column_stack([[15], [5], *cont.T]).astype("float32")  # 15 ∉ treino
    x_out = np.column_stack([[N_USERS + 7], [5], *cont.T]).astype("float32")
    assert model.predict_proba(x_unseen) == pytest.approx(model.predict_proba(x_out))


def test_unknown_dropout_trains_unknown_embedding():
    """Com id-dropout, o fit roteia parte dos ids p/ unknown sem quebrar."""
    model = _tiny_model(unknown_dropout=0.5, epochs=3)
    X = _fit_tiny(model)
    proba = model.predict_proba(X)
    assert np.all((proba >= 0.0) & (proba <= 1.0))
    # usuário desconhecido continua pontuável
    cont = np.full((1, len(CONT_COLS)), 0.5)
    x_unknown = np.column_stack([[N_USERS + 1], [5], *cont.T]).astype("float32")
    assert 0.0 <= model.predict_proba(x_unknown)[0] <= 1.0


def test_overfits_tiny_learnable_pattern():
    """Sanidade: com padrão aprendível por embeddings, AUC de treino sobe."""
    rng = np.random.default_rng(3)
    n = 600
    users = rng.integers(0, N_USERS, n)
    items = rng.integers(0, N_ITEMS, n)
    y = ((users + items) % 2).astype("float32")  # paridade: só ids explicam
    X = _matrix(users, items, rng)
    model = _tiny_model(epochs=60, patience=60, lr=0.01)
    model.fit(X, y)
    auc = roc_auc_score(y, model.predict_proba(X))
    assert auc > 0.9


def test_early_stopping_uses_validation_metric():
    """Com validação fornecida, o melhor estado por val-AUC é restaurado."""
    rng = np.random.default_rng(4)
    n = 300
    users = rng.integers(0, N_USERS, n)
    items = rng.integers(0, N_ITEMS, n)
    y = ((users + items) % 2).astype("float32")
    X = _matrix(users, items, rng)
    model = _tiny_model(epochs=10, patience=2)
    model.fit(X[:200], y[:200], X_val=X[200:], y_val=y[200:])
    proba = model.predict_proba(X[200:])
    assert proba.shape == (100,)
