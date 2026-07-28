"""Definição ÚNICA de rótulo e negative sampling (FR-001, FR-002, D1).

Este módulo é a única fonte da verdade para:
- pesos de evento (``EVENT_WEIGHTS``);
- o que conta como interação POSITIVA no treino (``POSITIVE_EVENTS``);
- as definições de relevância usadas nas métricas de ranking (forte/ampla);
- a amostragem de negativos por popularidade^alpha (estilo word2vec).

Nenhum outro módulo pode redeclarar threshold/rótulo (ex.: ``weight >= 3``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.feature_contract import (
    CONT_COLS,
    FEATURE_COLS,
    NO_HISTORY_RECENCY,
    TARGET_COL,
    VIEW_TARGET_COL,
)

EVENT_WEIGHTS: dict[str, int] = {"view": 1, "addtocart": 3, "transaction": 5}

# Interação positiva de TREINO: sinal forte de preferência.
POSITIVE_EVENTS: tuple[str, ...] = ("addtocart", "transaction")

# Definições de relevância para as métricas de ranking (FR-005a).
STRONG_RELEVANCE_EVENTS: tuple[str, ...] = POSITIVE_EVENTS
BROAD_RELEVANCE_EVENTS: tuple[str, ...] = ("view", "addtocart", "transaction")

# Colunas específicas do ITEM (não do usuário/contexto) — recalculadas para
# o item negativo amostrado, nunca copiadas do evento positivo original.
_ITEM_SPECIFIC_COLS: frozenset[str] = frozenset(
    {"view_count", "user_item_view_count", "user_item_recency_days"}
)

# Colunas contínuas copiadas do evento positivo para os negativos amostrados
# (descrevem o usuário/contexto no instante do evento, não o item).
_USER_CONTEXT_COLS: tuple[str, ...] = tuple(
    c for c in CONT_COLS if c not in _ITEM_SPECIFIC_COLS
)

_MAX_RESAMPLE_ROUNDS = 20
_EMPTY_SET: frozenset[int] = frozenset()


def is_positive(events: pd.Series) -> pd.Series:
    """Retorna máscara booleana de interação positiva (addtocart/transaction).

    Args:
        events: Série com o tipo de evento por linha.

    Returns:
        Série booleana alinhada ao índice de entrada.
    """
    return events.isin(POSITIVE_EVENTS)


def item_popularity(
    history: pd.DataFrame, alpha: float = 0.75
) -> tuple[np.ndarray, np.ndarray]:
    """Calcula a distribuição de amostragem de itens por popularidade^alpha.

    Args:
        history: Eventos de referência (apenas TREINO — FR-003).
        alpha: Expoente de amortecimento da popularidade (D1).

    Returns:
        Tupla (item_ids, probs) com ids e probabilidades normalizadas.
    """
    counts = history["item_idx"].value_counts()
    items = counts.index.to_numpy(dtype="int64")
    probs = counts.to_numpy(dtype="float64") ** alpha
    probs /= probs.sum()
    return items, probs


def build_user_seen(df: pd.DataFrame) -> dict[int, set[int]]:
    """Mapeia user_idx → conjunto de item_idx com que o usuário interagiu.

    Args:
        df: Eventos (qualquer tipo) com colunas user_idx e item_idx.

    Returns:
        Dicionário usuário → itens vistos.
    """
    return {
        int(u): {int(i) for i in items}
        for u, items in df.groupby("user_idx")["item_idx"]
    }


def build_item_view_times(history: pd.DataFrame) -> dict[int, np.ndarray]:
    """Mapeia item_idx → timestamps ordenados dos seus eventos de view.

    Usado para calcular ``view_count`` *as-of* dos negativos amostrados,
    mantendo a mesma semântica causal dos positivos (D2).

    Args:
        history: Eventos de referência (apenas TREINO).

    Returns:
        Dicionário item → array crescente de datetime64.
    """
    views = history.loc[history["event"] == "view", ["item_idx", "timestamp"]]
    views = views.sort_values("timestamp")
    return {
        int(i): ts.to_numpy(dtype="datetime64[ns]")
        for i, ts in views.groupby("item_idx")["timestamp"]
    }


def build_user_item_view_times(
    history: pd.DataFrame,
) -> dict[tuple[int, int], np.ndarray]:
    """Mapeia (user_idx, item_idx) → timestamps ordenados de views do par.

    Análogo a ``build_item_view_times``, mas no nível do par — usado para
    calcular ``user_item_view_count``/``user_item_recency_days`` *as-of*
    dos negativos amostrados (spec 002, FR-007/FR-008/FR-009, D5).

    Args:
        history: Eventos de referência (apenas TREINO).

    Returns:
        Dicionário (user_idx, item_idx) → array crescente de datetime64.
    """
    views = history.loc[
        history["event"] == "view", ["user_idx", "item_idx", "timestamp"]
    ]
    views = views.sort_values("timestamp")
    return {
        (int(u), int(i)): ts.to_numpy(dtype="datetime64[ns]")
        for (u, i), ts in views.groupby(["user_idx", "item_idx"])["timestamp"]
    }


def _as_of_pair_features(
    users: np.ndarray,
    items: np.ndarray,
    timestamps: np.ndarray,
    pair_view_times: dict[tuple[int, int], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Contagem e recência causais de views do par (user, item) amostrado."""
    empty = np.empty(0, dtype="datetime64[ns]")
    counts = np.empty(len(items), dtype="float64")
    recency = np.empty(len(items), dtype="float64")
    for pos, (user, item, ts) in enumerate(zip(users, items, timestamps, strict=True)):
        times = pair_view_times.get((int(user), int(item)), empty)
        idx = np.searchsorted(times, ts, side="left")
        counts[pos] = idx
        if idx == 0:
            recency[pos] = NO_HISTORY_RECENCY
        else:
            recency[pos] = (ts - times[idx - 1]) / np.timedelta64(1, "D")
    return counts, recency


def _seen_mask(
    users: np.ndarray, items: np.ndarray, seen: dict[int, set[int]]
) -> np.ndarray:
    """Máscara True onde o par (user, item) já foi visto pelo usuário."""
    return np.fromiter(
        (
            int(i) in seen.get(int(u), _EMPTY_SET)
            for u, i in zip(users, items, strict=True)
        ),
        dtype=bool,
        count=len(users),
    )


def _as_of_view_counts(
    items: np.ndarray, timestamps: np.ndarray, view_times: dict[int, np.ndarray]
) -> np.ndarray:
    """view_count causal: nº de views do item estritamente antes do timestamp."""
    empty = np.empty(0, dtype="datetime64[ns]")
    out = np.empty(len(items), dtype="float64")
    for pos, (item, ts) in enumerate(zip(items, timestamps, strict=True)):
        out[pos] = np.searchsorted(view_times.get(int(item), empty), ts, side="left")
    return out


def _sample_negative_rows(
    positives: pd.DataFrame,
    seen: dict[int, set[int]],
    item_ids: np.ndarray,
    probs: np.ndarray,
    num_negatives: int,
    view_times: dict[int, np.ndarray],
    pair_view_times: dict[tuple[int, int], np.ndarray],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Gera ``num_negatives`` negativos por positivo com rejeição de vistos."""
    n = len(positives) * num_negatives
    users = np.repeat(positives["user_idx"].to_numpy(dtype="int64"), num_negatives)
    timestamps = np.repeat(
        positives["timestamp"].to_numpy(dtype="datetime64[ns]"), num_negatives
    )
    candidates = rng.choice(item_ids, size=n, p=probs)
    for _ in range(_MAX_RESAMPLE_ROUNDS):
        bad = _seen_mask(users, candidates, seen)
        if not bad.any():
            break
        candidates[bad] = rng.choice(item_ids, size=int(bad.sum()), p=probs)
    keep = ~_seen_mask(users, candidates, seen)

    pair_count, pair_recency = _as_of_pair_features(
        users, candidates, timestamps, pair_view_times
    )
    rows = {
        "user_idx": users,
        "item_idx": candidates,
        "view_count": _as_of_view_counts(candidates, timestamps, view_times),
        "user_item_view_count": pair_count,
        "user_item_recency_days": pair_recency,
    }
    for col in _USER_CONTEXT_COLS:
        rows[col] = np.repeat(positives[col].to_numpy(), num_negatives)
    negatives = pd.DataFrame(rows).loc[keep, FEATURE_COLS].reset_index(drop=True)
    negatives[TARGET_COL] = 0.0
    return negatives


def _sample_view_rows(
    events: pd.DataFrame,
    n_positives: int,
    view_sample_ratio: float,
    seed: int,
) -> pd.DataFrame:
    """Amostra eventos ``view`` reais como sinal auxiliar (spec 002, D1).

    Reaproveita eventos causais já materializados em ``events`` — não
    recalcula nenhuma agregação. A razão é relativa ao nº de positivos
    fortes, para não deixar o volume bruto de views (tipicamente ~13x
    maior) afogar o sinal principal.

    Args:
        events: Split sendo rotulado (mesmo df de ``build_labeled_dataset``).
        n_positives: Nº de positivos fortes do split (referência da razão).
        view_sample_ratio: Views amostradas por positivo forte.
        seed: Semente — amostragem reprodutível.

    Returns:
        Subconjunto de ``events`` com apenas eventos ``view`` amostrados
        (colunas originais, ainda não recortado para FEATURE_COLS).
    """
    view_events = events.loc[events["event"] == "view"]
    n = min(round(n_positives * view_sample_ratio), len(view_events))
    if n <= 0:
        return view_events.iloc[0:0]
    return view_events.sample(n=n, random_state=seed)


def build_labeled_dataset(
    events: pd.DataFrame,
    history: pd.DataFrame | None = None,
    num_negatives: int = 4,
    popularity_alpha: float = 0.75,
    seed: int = 42,
    view_sample_ratio: float = 0.0,
) -> pd.DataFrame:
    """Constrói o dataset rotulado: positivos reais + negativos amostrados.

    Positivos = eventos addtocart/transaction de ``events`` (com suas features
    causais). Negativos = itens amostrados por popularidade^alpha do
    ``history``, rejeitando itens já vistos pelo usuário (em history ∪ events).

    Args:
        events: Split a rotular (train/val/test) com FEATURE_COLS + timestamp.
        history: Eventos usados para popularidade, views *as-of* e vistos.
            Default: os próprios ``events`` (caso do treino). Para val/test,
            passar o TREINO para não vazar estatísticas (FR-003).
        num_negatives: Razão de negativos por positivo (default 4:1).
        popularity_alpha: Expoente da amostragem por popularidade (D1).
        seed: Semente do gerador — amostragem reprodutível.
        view_sample_ratio: Views reais amostradas por positivo forte, como
            sinal auxiliar de multi-task (spec 002, FR-001/FR-003, D1).
            Default 0.0 — desligado: nenhuma coluna ``view_label`` é
            adicionada, comportamento idêntico ao anterior à spec 002.

    Returns:
        DataFrame com FEATURE_COLS + coluna ``label`` (1.0/0.0). Se
        ``view_sample_ratio > 0``, também inclui ``view_label`` (1.0 para
        positivos fortes e views amostradas, 0.0 para negativos).

    Raises:
        ValueError: Se não houver evento positivo em ``events``.
    """
    history = events if history is None else history
    positives = events.loc[is_positive(events["event"])]
    if positives.empty:
        raise ValueError("Nenhum evento positivo (addtocart/transaction) no split.")

    item_ids, probs = item_popularity(history, popularity_alpha)
    seen = build_user_seen(pd.concat([history, events], ignore_index=True))
    view_times = build_item_view_times(history)
    pair_view_times = build_user_item_view_times(history)
    rng = np.random.default_rng(seed)

    pos_rows = positives[FEATURE_COLS].reset_index(drop=True).copy()
    pos_rows[TARGET_COL] = 1.0
    neg_rows = _sample_negative_rows(
        positives,
        seen,
        item_ids,
        probs,
        num_negatives,
        view_times,
        pair_view_times,
        rng,
    )
    if view_sample_ratio <= 0:
        return pd.concat([pos_rows, neg_rows], ignore_index=True)

    pos_rows[VIEW_TARGET_COL] = 1.0
    neg_rows[VIEW_TARGET_COL] = 0.0
    view_rows = _sample_view_rows(events, len(positives), view_sample_ratio, seed)
    if view_rows.empty:
        return pd.concat([pos_rows, neg_rows], ignore_index=True)
    view_rows = view_rows[FEATURE_COLS].reset_index(drop=True).copy()
    view_rows[TARGET_COL] = 0.0
    view_rows[VIEW_TARGET_COL] = 1.0
    return pd.concat([pos_rows, neg_rows, view_rows], ignore_index=True)
