"""Testes do labeling e negative sampling (FR-001/FR-002, D1) — sintéticos."""

import numpy as np
import pandas as pd
import pytest

from src.data.feature_contract import (
    FEATURE_COLS,
    NO_HISTORY_RECENCY,
    TARGET_COL,
    VIEW_TARGET_COL,
)
from src.data.feature_engineering import build_causal_features
from src.data.labeling import (
    POSITIVE_EVENTS,
    _as_of_pair_features,
    _as_of_view_counts,
    _sample_view_rows,
    build_item_view_times,
    build_labeled_dataset,
    build_user_item_view_times,
    build_user_seen,
    is_positive,
    item_popularity,
)


@pytest.fixture()
def events() -> pd.DataFrame:
    """Eventos sintéticos com features causais (20 usuários, 30 itens)."""
    rng = np.random.default_rng(7)
    n = 300
    raw = pd.DataFrame(
        {
            "timestamp": pd.to_datetime("2015-05-01")
            + pd.to_timedelta(np.sort(rng.integers(0, 3600 * 24 * 30, n)), unit="s"),
            "user_idx": rng.integers(0, 20, n),
            "item_idx": rng.integers(0, 30, n),
            "event": rng.choice(
                ["view", "addtocart", "transaction"], size=n, p=[0.8, 0.15, 0.05]
            ),
        }
    )
    return build_causal_features(raw)


def test_is_positive_only_strong_events():
    events = pd.Series(["view", "addtocart", "transaction", "view"])
    assert is_positive(events).tolist() == [False, True, True, False]
    assert set(POSITIVE_EVENTS) == {"addtocart", "transaction"}


def test_labeled_dataset_has_contract_columns(events):
    labeled = build_labeled_dataset(events, seed=1)
    assert list(labeled.columns) == FEATURE_COLS + [TARGET_COL]
    assert set(labeled[TARGET_COL].unique()) == {0.0, 1.0}


def test_positive_rows_match_positive_events(events):
    labeled = build_labeled_dataset(events, seed=1)
    n_pos_expected = int(is_positive(events["event"]).sum())
    assert (labeled[TARGET_COL] == 1.0).sum() == n_pos_expected


def test_negative_ratio_at_most_4_to_1(events):
    labeled = build_labeled_dataset(events, num_negatives=4, seed=1)
    n_pos = (labeled[TARGET_COL] == 1.0).sum()
    n_neg = (labeled[TARGET_COL] == 0.0).sum()
    assert n_neg <= 4 * n_pos
    assert n_neg >= 3 * n_pos  # catálogo sintético é grande o suficiente


def test_negatives_never_seen_by_user(events):
    labeled = build_labeled_dataset(events, seed=1)
    seen = build_user_seen(events)
    negatives = labeled[labeled[TARGET_COL] == 0.0]
    pairs = zip(negatives["user_idx"], negatives["item_idx"], strict=True)
    for user, item in pairs:
        assert int(item) not in seen[int(user)]


def test_labeling_is_deterministic_with_seed(events):
    a = build_labeled_dataset(events, seed=42)
    b = build_labeled_dataset(events, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_labeling_changes_with_seed(events):
    a = build_labeled_dataset(events, seed=1)
    b = build_labeled_dataset(events, seed=2)
    assert not a.equals(b)


def test_item_popularity_is_normalized_distribution(events):
    items, probs = item_popularity(events)
    assert len(items) == len(probs)
    assert probs.sum() == pytest.approx(1.0)
    assert (probs > 0).all()


def test_as_of_view_counts_strictly_before_timestamp():
    history = pd.DataFrame(
        {
            "item_idx": [5, 5, 5],
            "event": ["view", "view", "view"],
            "timestamp": pd.to_datetime(["2015-05-01", "2015-05-03", "2015-05-05"]),
        }
    )
    view_times = build_item_view_times(history)
    items = np.array([5, 5, 5, 9])
    ts = pd.to_datetime(
        ["2015-05-02", "2015-05-05", "2015-05-06", "2015-05-06"]
    ).to_numpy()
    counts = _as_of_view_counts(items, ts, view_times)
    # 2015-05-05 exato NÃO conta (estritamente antes); item 9 sem views → 0
    assert counts.tolist() == [1.0, 2.0, 3.0, 0.0]


def test_as_of_pair_features_strictly_before_timestamp():
    """user_item_view_count/recency do par — mesma semântica de view_count."""
    history = pd.DataFrame(
        {
            "user_idx": [1, 1, 1],
            "item_idx": [5, 5, 5],
            "event": ["view", "view", "view"],
            "timestamp": pd.to_datetime(["2015-05-01", "2015-05-03", "2015-05-05"]),
        }
    )
    pair_view_times = build_user_item_view_times(history)
    users = np.array([1, 1, 1, 1])
    items = np.array([5, 5, 5, 9])  # item 9: par nunca visto por este usuário
    ts = pd.to_datetime(
        ["2015-05-02", "2015-05-05", "2015-05-06", "2015-05-06"]
    ).to_numpy()
    counts, recency = _as_of_pair_features(users, items, ts, pair_view_times)
    assert counts.tolist() == [1.0, 2.0, 3.0, 0.0]
    assert recency[0] == pytest.approx(1.0)  # 2015-05-01 → 2015-05-02
    assert recency[3] == NO_HISTORY_RECENCY


def test_negative_pair_features_are_always_sentinel(events):
    """Negativos nunca foram vistos pelo usuário → par sempre 'nunca visto'.

    Invariante forte: como os negativos amostrados excluem por construção
    qualquer item já visto pelo usuário (``_seen_mask``), o par (usuário,
    item negativo) nunca teve view — user_item_view_count DEVE ser 0 e
    user_item_recency_days DEVE ser o sentinela, sempre.
    """
    labeled = build_labeled_dataset(events, seed=1)
    negatives = labeled[labeled[TARGET_COL] == 0.0]
    assert (negatives["user_item_view_count"] == 0.0).all()
    assert (negatives["user_item_recency_days"] == NO_HISTORY_RECENCY).all()


def test_view_sample_ratio_zero_has_no_view_label_column(events):
    """Default (0.0) preserva o comportamento anterior à spec 002."""
    labeled = build_labeled_dataset(events, seed=1)
    assert VIEW_TARGET_COL not in labeled.columns


def test_view_label_added_when_ratio_positive(events):
    """Positivos fortes e views amostradas viram view_label=1; negativos, 0."""
    labeled = build_labeled_dataset(events, seed=1, view_sample_ratio=1.0)
    assert VIEW_TARGET_COL in labeled.columns
    strong_pos = labeled[labeled[TARGET_COL] == 1.0]
    assert (strong_pos[VIEW_TARGET_COL] == 1.0).all()

    zero_target = labeled[labeled[TARGET_COL] == 0.0]
    # dentro de label=0 coexistem negativos de fato (view_label=0) e views
    # amostradas como sinal auxiliar (view_label=1).
    assert (zero_target[VIEW_TARGET_COL] == 0.0).any()
    assert (zero_target[VIEW_TARGET_COL] == 1.0).any()


def test_sample_view_rows_respects_ratio_and_cap(events):
    n_views_available = int((events["event"] == "view").sum())
    capped = _sample_view_rows(events, n_positives=1000, view_sample_ratio=10.0, seed=1)
    assert len(capped) == n_views_available
    assert (capped["event"] == "view").all()

    small = _sample_view_rows(events, n_positives=2, view_sample_ratio=1.0, seed=1)
    assert len(small) == 2


def test_raises_without_positives():
    df = pd.DataFrame(
        {
            "user_idx": [0],
            "item_idx": [0],
            "event": ["view"],
            "timestamp": pd.to_datetime(["2015-05-01"]),
            "hour": [0],
            "day_of_week": [0],
            "frequency": [0],
            "engagement_score": [0.0],
            "recency_days": [-1.0],
            "view_count": [0],
        }
    )
    with pytest.raises(ValueError, match="positivo"):
        build_labeled_dataset(df)


def test_val_labeling_uses_train_history(events):
    """Popularidade/vistos vêm do histórico (treino), não do split rotulado."""
    cut = events["timestamp"].quantile(0.7)
    train = events[events["timestamp"] <= cut]
    val = events[events["timestamp"] > cut]
    if not is_positive(val["event"]).any():
        pytest.skip("split sintético sem positivos na validação")
    labeled = build_labeled_dataset(val, history=train, seed=3)
    # negativos só podem vir do catálogo do treino
    train_items = set(train["item_idx"].astype(int))
    negs = labeled.loc[labeled[TARGET_COL] == 0.0, "item_idx"].astype(int)
    assert all(i in train_items for i in negs)
