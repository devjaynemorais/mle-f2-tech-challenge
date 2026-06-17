"""Testes de feature engineering — usam DataFrames sintéticos.

Não dependem de arquivos em disco, então passam em qualquer clone limpo.
"""

import pandas as pd
import pytest

from src.data.feature_engineering import (
    add_event_weights,
    add_temporal_features,
    build_interaction_features,
    chronological_split,
    compute_item_features,
    compute_user_features,
)


# ---------------------------------------------------------------------------
# Fixture base
# ---------------------------------------------------------------------------


@pytest.fixture()
def df_clean() -> pd.DataFrame:
    """Eventos limpos pós-preprocess: user_idx, item_idx, timestamp, event."""
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2015-05-01 10:00",
                    "2015-05-02 14:00",
                    "2015-05-03 09:00",
                    "2015-05-04 20:00",
                    "2015-05-05 11:00",
                    "2015-05-06 08:00",
                    "2015-05-07 16:00",
                    "2015-05-08 13:00",
                    "2015-05-09 18:00",
                    "2015-05-10 10:00",
                ]
            ),
            "user_idx": [0, 0, 0, 0, 0, 1, 1, 1, 2, 2],
            "item_idx": [0, 1, 1, 2, 2, 0, 1, 2, 0, 1],
            "visitorid": [10, 10, 10, 10, 10, 20, 20, 20, 30, 30],
            "itemid": [100, 200, 200, 300, 300, 100, 200, 300, 100, 200],
            "event": [
                "view",
                "view",
                "addtocart",
                "view",
                "transaction",
                "view",
                "addtocart",
                "view",
                "view",
                "view",
            ],
        }
    )


# ---------------------------------------------------------------------------
# add_event_weights
# ---------------------------------------------------------------------------


def test_event_weights_view(df_clean: pd.DataFrame) -> None:
    """view recebe peso 1."""
    result = add_event_weights(df_clean)
    view_weights = result.loc[result["event"] == "view", "weight"]
    assert (view_weights == 1).all()


def test_event_weights_addtocart(df_clean: pd.DataFrame) -> None:
    """addtocart recebe peso 3."""
    result = add_event_weights(df_clean)
    cart_weights = result.loc[result["event"] == "addtocart", "weight"]
    assert (cart_weights == 3).all()


def test_event_weights_transaction(df_clean: pd.DataFrame) -> None:
    """transaction recebe peso 5."""
    result = add_event_weights(df_clean)
    txn_weights = result.loc[result["event"] == "transaction", "weight"]
    assert (txn_weights == 5).all()


def test_event_weights_column_exists(df_clean: pd.DataFrame) -> None:
    """Coluna 'weight' adicionada ao DataFrame."""
    result = add_event_weights(df_clean)
    assert "weight" in result.columns


# ---------------------------------------------------------------------------
# add_temporal_features
# ---------------------------------------------------------------------------


def test_temporal_hour_of_day(df_clean: pd.DataFrame) -> None:
    """Coluna 'hour' extrai hora correta do timestamp."""
    result = add_temporal_features(df_clean)
    assert "hour" in result.columns
    assert result["hour"].iloc[0] == 10   # "2015-05-01 10:00"
    assert result["hour"].iloc[1] == 14   # "2015-05-02 14:00"


def test_temporal_day_of_week(df_clean: pd.DataFrame) -> None:
    """Coluna 'day_of_week' extrai dia da semana (0=seg, 6=dom)."""
    result = add_temporal_features(df_clean)
    assert "day_of_week" in result.columns
    # 2015-05-01 = sexta = 4
    assert result["day_of_week"].iloc[0] == 4


def test_temporal_does_not_drop_rows(df_clean: pd.DataFrame) -> None:
    """Número de linhas não muda após adicionar features temporais."""
    result = add_temporal_features(df_clean)
    assert len(result) == len(df_clean)


# ---------------------------------------------------------------------------
# compute_user_features
# ---------------------------------------------------------------------------


def test_compute_user_frequency(df_clean: pd.DataFrame) -> None:
    """Frequência = total de interações por user_idx."""
    df_w = add_event_weights(df_clean)
    user_feats = compute_user_features(df_w)
    # user_idx=0 tem 5 eventos, user_idx=1 tem 3, user_idx=2 tem 2
    assert user_feats.loc[user_feats["user_idx"] == 0, "frequency"].iloc[0] == 5
    assert user_feats.loc[user_feats["user_idx"] == 1, "frequency"].iloc[0] == 3
    assert user_feats.loc[user_feats["user_idx"] == 2, "frequency"].iloc[0] == 2


def test_compute_user_recency(df_clean: pd.DataFrame) -> None:
    """Recência = dias desde último evento do usuário até data de referência."""
    df_w = add_event_weights(df_clean)
    ref_date = pd.Timestamp("2015-05-11")
    user_feats = compute_user_features(df_w, reference_date=ref_date)
    # user_idx=0: último evento 2015-05-05 → 6 dias
    recency_u0 = user_feats.loc[user_feats["user_idx"] == 0, "recency_days"].iloc[0]
    assert recency_u0 == 6


def test_compute_weighted_engagement(df_clean: pd.DataFrame) -> None:
    """engagement_score = soma dos pesos por user_idx."""
    df_w = add_event_weights(df_clean)
    user_feats = compute_user_features(df_w)
    # user_idx=0: view×3 + addtocart×1 + transaction×1 = 3×1 + 3 + 5 = 11
    score_u0 = user_feats.loc[user_feats["user_idx"] == 0, "engagement_score"].iloc[0]
    assert score_u0 == 11


def test_user_features_one_row_per_user(df_clean: pd.DataFrame) -> None:
    """Uma linha por user_idx no resultado."""
    df_w = add_event_weights(df_clean)
    user_feats = compute_user_features(df_w)
    assert user_feats["user_idx"].nunique() == len(user_feats)


# ---------------------------------------------------------------------------
# compute_item_features
# ---------------------------------------------------------------------------


def test_item_view_count(df_clean: pd.DataFrame) -> None:
    """view_count = número de eventos 'view' por item_idx."""
    item_feats = compute_item_features(df_clean)
    # item_idx=0: view×3 (eventos idx 0,5,8)
    vc = item_feats.loc[item_feats["item_idx"] == 0, "view_count"].iloc[0]
    assert vc == 3


def test_item_popularity_tier_column_exists(df_clean: pd.DataFrame) -> None:
    """Coluna 'popularity_tier' gerada."""
    item_feats = compute_item_features(df_clean)
    assert "popularity_tier" in item_feats.columns


def test_item_popularity_tier_valid_values(df_clean: pd.DataFrame) -> None:
    """popularity_tier só contém: long_tail, mid_tier, top_tier."""
    item_feats = compute_item_features(df_clean)
    valid = {"long_tail", "mid_tier", "top_tier"}
    assert set(item_feats["popularity_tier"].unique()).issubset(valid)


def test_item_features_one_row_per_item(df_clean: pd.DataFrame) -> None:
    """Uma linha por item_idx no resultado."""
    item_feats = compute_item_features(df_clean)
    assert item_feats["item_idx"].nunique() == len(item_feats)


# ---------------------------------------------------------------------------
# build_interaction_features
# ---------------------------------------------------------------------------


def test_build_features_has_weight(df_clean: pd.DataFrame) -> None:
    """Resultado contém coluna 'weight'."""
    result = build_interaction_features(df_clean)
    assert "weight" in result.columns


def test_build_features_has_temporal(df_clean: pd.DataFrame) -> None:
    """Resultado contém colunas temporais."""
    result = build_interaction_features(df_clean)
    assert "hour" in result.columns
    assert "day_of_week" in result.columns


def test_build_features_has_user_features(df_clean: pd.DataFrame) -> None:
    """Resultado contém features de usuário (frequency, recency_days, engagement_score)."""
    result = build_interaction_features(df_clean)
    for col in ("frequency", "recency_days", "engagement_score"):
        assert col in result.columns


def test_build_features_has_item_features(df_clean: pd.DataFrame) -> None:
    """Resultado contém features de item (view_count, popularity_tier)."""
    result = build_interaction_features(df_clean)
    for col in ("view_count", "popularity_tier"):
        assert col in result.columns


def test_build_features_preserves_row_count(df_clean: pd.DataFrame) -> None:
    """Número de linhas = número de eventos originais."""
    result = build_interaction_features(df_clean)
    assert len(result) == len(df_clean)


# ---------------------------------------------------------------------------
# chronological_split
# ---------------------------------------------------------------------------


def test_chronological_split_order(df_clean: pd.DataFrame) -> None:
    """Todos eventos de treino precedem os de validação e teste."""
    train, val, test = chronological_split(df_clean, val_size=0.1, test_size=0.2)
    assert train["timestamp"].max() <= val["timestamp"].min()
    assert val["timestamp"].max() <= test["timestamp"].min()


def test_split_proportions(df_clean: pd.DataFrame) -> None:
    """Proporções train/val/test dentro de ±1 evento do esperado (10 linhas)."""
    train, val, test = chronological_split(df_clean, val_size=0.1, test_size=0.2)
    total = len(df_clean)
    assert abs(len(test) - round(total * 0.2)) <= 1
    assert abs(len(val) - round(total * 0.1)) <= 1
    assert len(train) + len(val) + len(test) == total


def test_no_future_leakage(df_clean: pd.DataFrame) -> None:
    """Nenhum evento de teste tem timestamp menor que o máximo do treino."""
    train, _val, test = chronological_split(df_clean, val_size=0.1, test_size=0.2)
    assert test["timestamp"].min() >= train["timestamp"].max()
