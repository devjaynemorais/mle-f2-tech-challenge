"""Engenharia de features CAUSAIS para o RetailRocket dataset (FR-003, D2).

Todas as agregações são *as-of*: cada linha enxerga apenas eventos
estritamente anteriores a ela — por construção não há vazamento temporal,
independentemente do split. Funções puras (DataFrame → DataFrame).

A definição de pesos/rótulo vive em ``src.data.labeling`` (FR-002).
"""

from __future__ import annotations

import pandas as pd

from src.data.feature_contract import NO_HISTORY_RECENCY
from src.data.labeling import EVENT_WEIGHTS

_SECONDS_PER_DAY = 86400.0


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


def add_causal_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features de usuário *as-of*: excluem o evento corrente.

    - ``frequency``: nº de eventos anteriores do usuário.
    - ``engagement_score``: soma dos pesos dos eventos anteriores.
    - ``recency_days``: dias desde o evento anterior do usuário
      (``NO_HISTORY_RECENCY`` quando é o primeiro evento).

    Args:
        df: DataFrame com user_idx, timestamp e weight, ordenado por timestamp.

    Returns:
        DataFrame com as três colunas causais adicionadas.
    """
    df = df.copy()
    grouped = df.groupby("user_idx", sort=False)
    df["frequency"] = grouped.cumcount()
    df["engagement_score"] = grouped["weight"].cumsum() - df["weight"]
    prev_ts = grouped["timestamp"].shift(1)
    delta = (df["timestamp"] - prev_ts).dt.total_seconds() / _SECONDS_PER_DAY
    df["recency_days"] = delta.fillna(NO_HISTORY_RECENCY)
    return df


def add_causal_item_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features de item *as-of*: ``view_count`` = views anteriores do item.

    Args:
        df: DataFrame com item_idx e event, ordenado por timestamp.

    Returns:
        DataFrame com a coluna view_count adicionada.
    """
    df = df.copy()
    is_view = (df["event"] == "view").astype("int64")
    df["view_count"] = is_view.groupby(df["item_idx"]).cumsum() - is_view
    return df


def add_causal_pair_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features causais do par (usuário, item) — sinal específico do par.

    Dá sinal a usuários/itens com pouco histórico agregado, mas que já
    interagiram um com o outro antes (spec 002, FR-007/FR-008):

    - ``user_item_view_count``: nº de views *anteriores* deste usuário
      neste item específico.
    - ``user_item_recency_days``: dias desde a última view *anterior*
      deste usuário neste item (``NO_HISTORY_RECENCY`` se nunca visto).

    Args:
        df: DataFrame com user_idx, item_idx, event e timestamp, ordenado
            por timestamp.

    Returns:
        DataFrame com as duas colunas de par adicionadas.
    """
    df = df.copy()
    is_view = (df["event"] == "view").astype("int64")
    pair_key = [df["user_idx"], df["item_idx"]]
    df["user_item_view_count"] = is_view.groupby(pair_key).cumsum() - is_view

    view_ts = df["timestamp"].where(df["event"] == "view")
    prev_view_ts = view_ts.groupby(pair_key).transform(lambda s: s.shift(1).ffill())
    delta = (df["timestamp"] - prev_view_ts).dt.total_seconds() / _SECONDS_PER_DAY
    df["user_item_recency_days"] = delta.fillna(NO_HISTORY_RECENCY)
    return df


def build_causal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Constrói o DataFrame de interações com todas as features causais.

    Ordena por timestamp (estável) e aplica pesos, features temporais e
    agregações *as-of* de usuário, item e do par (usuário, item).

    Args:
        df: DataFrame limpo com colunas user_idx, item_idx, timestamp, event.

    Returns:
        DataFrame com features completas, uma linha por evento.
    """
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    df = add_event_weights(df)
    df = add_temporal_features(df)
    df = add_causal_user_features(df)
    df = add_causal_item_features(df)
    df = add_causal_pair_features(df)
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
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    n = len(df)
    n_test = round(n * test_size)
    n_val = round(n * val_size)
    test = df.iloc[n - n_test :]
    val = df.iloc[n - n_test - n_val : n - n_test]
    train = df.iloc[: n - n_test - n_val]
    return train, val, test
