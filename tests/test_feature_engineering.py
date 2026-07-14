"""Testes de feature engineering causal — usam DataFrames sintéticos.

Não dependem de arquivos em disco, então passam em qualquer clone limpo.
Inclui o teste de ausência de vazamento temporal (AC-3).
"""

import pandas as pd
import pytest

from src.data.feature_contract import NO_HISTORY_RECENCY
from src.data.feature_engineering import (
    add_causal_item_features,
    add_causal_user_features,
    add_event_weights,
    add_temporal_features,
    build_causal_features,
    chronological_split,
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
# add_event_weights / add_temporal_features
# ---------------------------------------------------------------------------


def test_event_weights(df_clean: pd.DataFrame) -> None:
    """view=1, addtocart=3, transaction=5."""
    result = add_event_weights(df_clean)
    assert (result.loc[result["event"] == "view", "weight"] == 1).all()
    assert (result.loc[result["event"] == "addtocart", "weight"] == 3).all()
    assert (result.loc[result["event"] == "transaction", "weight"] == 5).all()


def test_temporal_features(df_clean: pd.DataFrame) -> None:
    """hour e day_of_week extraídos do timestamp, sem perder linhas."""
    result = add_temporal_features(df_clean)
    assert result["hour"].iloc[0] == 10
    assert result["day_of_week"].iloc[0] == 4  # 2015-05-01 = sexta
    assert len(result) == len(df_clean)


# ---------------------------------------------------------------------------
# Features causais de usuário (as-of)
# ---------------------------------------------------------------------------


def test_causal_frequency_counts_only_prior_events(df_clean: pd.DataFrame) -> None:
    """frequency = nº de eventos ANTERIORES do usuário (0 no primeiro)."""
    df = add_causal_user_features(add_event_weights(df_clean))
    u0 = df[df["user_idx"] == 0]
    assert u0["frequency"].tolist() == [0, 1, 2, 3, 4]


def test_causal_engagement_excludes_current_event(df_clean: pd.DataFrame) -> None:
    """engagement_score = soma dos pesos dos eventos anteriores."""
    df = add_causal_user_features(add_event_weights(df_clean))
    u0 = df[df["user_idx"] == 0]
    # pesos u0: view=1, view=1, addtocart=3, view=1, transaction=5
    assert u0["engagement_score"].tolist() == [0, 1, 2, 5, 6]


def test_causal_recency_days_from_previous_event(df_clean: pd.DataFrame) -> None:
    """recency_days = dias desde o evento anterior; sentinela no primeiro."""
    df = add_causal_user_features(add_event_weights(df_clean))
    u0 = df[df["user_idx"] == 0]
    assert u0["recency_days"].iloc[0] == NO_HISTORY_RECENCY
    # 2015-05-01 10:00 → 2015-05-02 14:00 = 1 dia e 4h
    assert u0["recency_days"].iloc[1] == pytest.approx(1 + 4 / 24)


# ---------------------------------------------------------------------------
# Features causais de item (as-of)
# ---------------------------------------------------------------------------


def test_causal_view_count_counts_only_prior_views(df_clean: pd.DataFrame) -> None:
    """view_count = views ANTERIORES do item (evento corrente excluído)."""
    df = add_causal_item_features(df_clean)
    item0 = df[df["item_idx"] == 0]
    # item 0 recebe views nas linhas 0, 5 e 8 → contagem prévia 0, 1, 2
    assert item0["view_count"].tolist() == [0, 1, 2]


def test_causal_view_count_ignores_non_view_events(df_clean: pd.DataFrame) -> None:
    """addtocart/transaction não incrementam view_count."""
    df = add_causal_item_features(df_clean)
    item1 = df[df["item_idx"] == 1]
    # item 1: view (linha 1), addtocart (2), addtocart (6), view (9)
    assert item1["view_count"].tolist() == [0, 1, 1, 1]


# ---------------------------------------------------------------------------
# Ausência de vazamento temporal (AC-3)
# ---------------------------------------------------------------------------


def test_no_temporal_leakage_features_unchanged_by_future(
    df_clean: pd.DataFrame,
) -> None:
    """Features de um prefixo temporal não mudam quando o futuro é incluído.

    Se qualquer agregação usasse eventos futuros, as features das primeiras
    linhas seriam diferentes ao truncar o dataset — aqui devem ser idênticas.
    """
    feature_cols = [
        "frequency",
        "engagement_score",
        "recency_days",
        "view_count",
    ]
    full = build_causal_features(df_clean)
    prefix = build_causal_features(df_clean.iloc[:6])
    pd.testing.assert_frame_equal(
        full.iloc[:6][feature_cols].reset_index(drop=True),
        prefix[feature_cols].reset_index(drop=True),
    )


def test_build_causal_features_has_all_columns(df_clean: pd.DataFrame) -> None:
    """Resultado contém todas as colunas do contrato + peso."""
    result = build_causal_features(df_clean)
    for col in (
        "weight",
        "hour",
        "day_of_week",
        "frequency",
        "engagement_score",
        "recency_days",
        "view_count",
    ):
        assert col in result.columns
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


def test_no_future_leakage_in_split(df_clean: pd.DataFrame) -> None:
    """Nenhum evento de teste tem timestamp menor que o máximo do treino."""
    train, _val, test = chronological_split(df_clean, val_size=0.1, test_size=0.2)
    assert test["timestamp"].min() >= train["timestamp"].max()
