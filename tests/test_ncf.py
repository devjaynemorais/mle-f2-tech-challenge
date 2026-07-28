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


def test_unknown_item_keeps_known_category():
    """Item ausente do treino, mas com categoria conhecida, preserva o
    cat_idx REAL na pontuação — não cai no "unknown" só porque seu
    embedding de item é o compartilhado (spec 002, FR-006)."""
    n_categories = 3
    item_categories = np.full(N_ITEMS + 1, n_categories, dtype="int64")
    item_categories[7] = 1  # item 7: categoria real conhecida (via content)
    known_items = np.zeros(N_ITEMS, dtype=bool)
    known_items[:5] = True  # item 7 NÃO apareceu no treino
    model = _tiny_model(
        n_categories=n_categories,
        item_categories=item_categories,
        known_items=known_items,
    )
    _fit_tiny(model)
    cont = np.full((1, len(CONT_COLS)), 0.5)
    x_known_cat = np.column_stack([[0], [7], *cont.T]).astype("float32")
    users, items, cats, _ = model._split_matrix(x_known_cat)
    assert items[0] == N_ITEMS  # embedding de item roteado p/ "unknown"
    assert cats[0] == 1  # mas a categoria REAL (1) foi preservada

    # item 8: também ausente do treino e SEM categoria conhecida — cai no
    # índice de categoria "unknown" (n_categories), comportamento inalterado.
    x_no_cat = np.column_stack([[0], [8], *cont.T]).astype("float32")
    _, items2, cats2, _ = model._split_matrix(x_no_cat)
    assert items2[0] == N_ITEMS
    assert cats2[0] == n_categories


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


# ---------------------------------------------------------------------------
# Multi-task: cabeça auxiliar de view (spec 002, FR-001/FR-002)
# ---------------------------------------------------------------------------


def test_predict_proba_head_param_shapes_and_range():
    """head='strong'/'view' aceitos, retornam arrays válidos do mesmo shape."""
    model = _tiny_model()
    X = _fit_tiny(model)
    strong = model.predict_proba(X, head="strong")
    view = model.predict_proba(X, head="view")
    assert strong.shape == view.shape == (len(X),)
    assert np.all((view >= 0.0) & (view <= 1.0))


def test_fit_without_y_view_keeps_old_behavior():
    """Sem y_view (default), o comportamento é idêntico ao pré-multi-task."""
    model = _tiny_model()
    assert model._multitask is False
    X = _fit_tiny(model)
    assert model._multitask is False
    proba = model.predict_proba(X)  # head="strong" default
    assert proba.shape == (len(X),)


def test_multitask_loss_trains_both_heads_on_distinct_patterns():
    """Com y_view fornecido, cada cabeça aprende seu próprio padrão (D2/D3).

    Estrutura os rótulos como o ``labeling.py`` real: positivo forte
    (``y=1`` implica ``y_view=1``) e linhas "vistas mas não convertidas"
    (``y=0, y_view=1``) — essas últimas são mascaradas da loss da cabeça
    forte (ver ``_combined_loss``), então a cabeça forte só é cobrada nas
    linhas não mascaradas (positivos + negativos puros).
    """
    rng = np.random.default_rng(5)
    n = 600
    users = rng.integers(0, N_USERS, n)
    items = rng.integers(0, N_ITEMS, n)
    X = _matrix(users, items, rng)
    strong_pattern = ((users + items) % 2).astype("float32")  # user E item
    view_pattern = (users % 2).astype("float32")  # padrão diferente, só user
    y = strong_pattern
    y_view = np.where(y == 1, 1.0, view_pattern).astype("float32")

    model = _tiny_model(epochs=60, patience=60, lr=0.02, view_loss_weight=1.0)
    model.fit(X, y, y_view=y_view)
    assert model._multitask is True

    strong_mask = ~((y_view == 1) & (y == 0))
    strong_proba = model.predict_proba(X, head="strong")
    auc_strong = roc_auc_score(y[strong_mask], strong_proba[strong_mask])
    auc_view = roc_auc_score(y_view, model.predict_proba(X, head="view"))
    assert auc_strong > 0.9
    assert auc_view > 0.9


def test_view_only_rows_do_not_pollute_strong_loss():
    """Linhas "vistas mas não convertidas" não distorcem a cabeça forte.

    Regressão direta do achado empírico da spec 002 (diagnóstico via
    ablation em dados reais): sem a máscara em ``_combined_loss``, linhas
    com ``y_view=1, y=0`` entravam como negativo também da loss forte —
    aqui elas duplicam exatamente os MESMOS pares (user, item) dos
    positivos "limpos" com rótulo contraditório (y=0), o que destruiria a
    aprendizagem da cabeça forte se não fossem mascaradas.
    """
    rng = np.random.default_rng(8)
    n_clean = 400
    users_c = rng.integers(0, N_USERS, n_clean)
    items_c = rng.integers(0, N_ITEMS, n_clean)
    y_clean = ((users_c + items_c) % 2).astype("float32")
    # Semântica real: positivo (y=1) sempre implica view_label=1; negativo
    # uniforme (y=0) nunca é view (view_label=0) — só as linhas "view-only"
    # abaixo têm y=0 COM view_label=1.
    y_view_clean = y_clean.copy()

    # Duplica os pares (user, item) dos positivos "limpos" como linhas
    # "vistas mas não convertidas": mesmos ids, rótulo forte oposto (y=0).
    is_pos = y_clean == 1
    users_dup, items_dup = users_c[is_pos], items_c[is_pos]
    y_dup = np.zeros(int(is_pos.sum()), dtype="float32")
    y_view_dup = np.ones(int(is_pos.sum()), dtype="float32")

    users = np.concatenate([users_c, users_dup])
    items = np.concatenate([items_c, items_dup])
    y = np.concatenate([y_clean, y_dup])
    y_view = np.concatenate([y_view_clean, y_view_dup])

    rng2 = np.random.default_rng(9)
    X = _matrix(users, items, rng2)
    model = _tiny_model(epochs=60, patience=60, lr=0.02, view_loss_weight=1.0)
    model.fit(X, y, y_view=y_view)

    proba = model.predict_proba(X[:n_clean], head="strong")
    auc = roc_auc_score(y_clean, proba)
    assert auc > 0.85


def test_early_stopping_ignores_view_head_in_multitask():
    """Em multi-task, o early stopping continua só sobre a validação forte."""
    rng = np.random.default_rng(6)
    n = 300
    users = rng.integers(0, N_USERS, n)
    items = rng.integers(0, N_ITEMS, n)
    y = ((users + items) % 2).astype("float32")
    y_view = (users % 2).astype("float32")
    X = _matrix(users, items, rng)
    model = _tiny_model(epochs=10, patience=2, view_loss_weight=0.5)
    # y_view só é passado para o TREINO — fit() não aceita y_view de validação.
    model.fit(X[:200], y[:200], X_val=X[200:], y_val=y[200:], y_view=y_view[:200])
    proba = model.predict_proba(X[200:])
    assert proba.shape == (100,)


def test_make_model_scorer_forwards_head_for_ncf():
    """make_model_scorer(head=) pontua com a cabeça certa (spec 002, D6)."""
    import pandas as pd

    from src.evaluation.scorers import HistoryFeatureLookup, make_model_scorer

    rng = np.random.default_rng(7)
    n = 500
    users = rng.integers(0, N_USERS, n)
    items = rng.integers(0, N_ITEMS, n)
    X = _matrix(users, items, rng)
    y_strong = ((users + items) % 2).astype("float32")
    y_view = (users % 2).astype("float32")
    model = _tiny_model(epochs=60, patience=60, lr=0.02, view_loss_weight=1.0)
    model.fit(X, y_strong, y_view=y_view)

    history = pd.DataFrame(
        {
            "user_idx": users,
            "item_idx": items,
            "event": ["view"] * n,
            "weight": [1] * n,
            "timestamp": pd.date_range("2015-05-01", periods=n, freq="h"),
        }
    )
    lookup = HistoryFeatureLookup(history)
    strong_scorer = make_model_scorer(model, lookup, head="strong")
    view_scorer = make_model_scorer(model, lookup, head="view")
    sample = slice(0, 20)
    strong_scores = strong_scorer(users[sample], items[sample])
    view_scores = view_scorer(users[sample], items[sample])
    assert not np.allclose(strong_scores, view_scores)
