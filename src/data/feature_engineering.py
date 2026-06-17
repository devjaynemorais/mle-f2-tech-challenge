"""Engenharia de features para o RetailRocket dataset.

Todas as funções são puras (DataFrame → DataFrame) e não dependem de disco.
Pesos de eventos: view=1, addtocart=3, transaction=5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EVENT_WEIGHTS: dict[str, int] = {"view": 1, "addtocart": 3, "transaction": 5}


def add_event_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona coluna 'weight' com peso por tipo de evento.

    Args:
        df: DataFrame com coluna 'event'.

    Returns:
        DataFrame com coluna 'weight' adicionada.
    """
    df = df.copy()
    df["weight"] = df["event"].map(EVENT_WEIGHTS)
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona colunas 'hour' e 'day_of_week' extraídas do timestamp.

    Args:
        df: DataFrame com coluna 'timestamp' (datetime).

    Returns:
        DataFrame com colunas temporais adicionadas.
    """
    df = df.copy()
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    return df


def compute_user_features(
    df: pd.DataFrame,
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Agrega features de comportamento por usuário.

    Args:
        df: DataFrame com colunas 'user_idx', 'timestamp', 'weight'.
        reference_date: Data de referência para cálculo de recência.
            Padrão: dia seguinte ao último evento.

    Returns:
        DataFrame com uma linha por user_idx e colunas:
        frequency, recency_days, engagement_score.
    """
    if reference_date is None:
        reference_date = df["timestamp"].max() + pd.Timedelta(days=1)
    agg = (
        df.groupby("user_idx")
        .agg(
            frequency=("user_idx", "count"),
            last_event=("timestamp", "max"),
            engagement_score=("weight", "sum"),
        )
        .reset_index()
    )
    agg["recency_days"] = (
        reference_date.normalize() - agg["last_event"].dt.normalize()
    ).dt.days
    return agg.drop(columns=["last_event"])


def compute_item_features(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega features de popularidade por item.

    Args:
        df: DataFrame com colunas 'item_idx', 'event'.

    Returns:
        DataFrame com uma linha por item_idx e colunas:
        view_count, popularity_tier (long_tail / mid_tier / top_tier).
    """
    views = df[df["event"] == "view"]
    vc = views.groupby("item_idx").size().reset_index(name="view_count")
    all_items = df[["item_idx"]].drop_duplicates()
    item_feats = all_items.merge(vc, on="item_idx", how="left").fillna({"view_count": 0})
    item_feats["view_count"] = item_feats["view_count"].astype(int)
    p50 = item_feats["view_count"].quantile(0.5)
    p90 = item_feats["view_count"].quantile(0.9)
    conditions = [item_feats["view_count"] > p90, item_feats["view_count"] > p50]
    item_feats["popularity_tier"] = np.select(
        conditions, ["top_tier", "mid_tier"], default="long_tail"
    )
    return item_feats


def build_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Constrói DataFrame de interações com todas as features.

    Combina features de evento, temporais, usuário e item em cada linha.

    Args:
        df: DataFrame limpo com colunas user_idx, item_idx, timestamp, event.

    Returns:
        DataFrame com features completas para treinamento.
    """
    df = add_event_weights(df)
    df = add_temporal_features(df)
    user_feats = compute_user_features(df)
    item_feats = compute_item_features(df)
    df = df.merge(user_feats, on="user_idx", how="left")
    df = df.merge(item_feats, on="item_idx", how="left")
    return df


def chronological_split(
    df: pd.DataFrame,
    val_size: float = 0.1,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide eventos em treino, validação e teste pela ordem temporal.

    Args:
        df: DataFrame com coluna 'timestamp'.
        val_size: Proporção do conjunto de validação.
        test_size: Proporção do conjunto de teste.

    Returns:
        Tupla (train_df, val_df, test_df) com splits cronológicos.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    n = len(df)
    n_test = round(n * test_size)
    n_val = round(n * val_size)
    test = df.iloc[n - n_test :]
    val = df.iloc[n - n_test - n_val : n - n_test]
    train = df.iloc[: n - n_test - n_val]
    return train, val, test
