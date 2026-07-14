"""Scorers de pares (user, item) para a avaliação de ranking (FR-006, FR-007).

``HistoryFeatureLookup`` deriva o estado corrente de usuários e itens a
partir de um histórico de eventos e monta a matriz na ordem do contrato de
features — a mesma transformação vista no treino e no serving.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.feature_contract import FEATURE_COLS, NO_HISTORY_RECENCY
from src.evaluation.ranking import PairScorer
from src.models.base import RecommenderBase
from src.models.mlp import NCFRecommender

_SECONDS_PER_DAY = 86400.0
_SCORING_CHUNK = 500_000


class HistoryFeatureLookup:
    """Estado *as-of* de usuários/itens derivado do histórico de eventos.

    Args:
        history: Eventos com user_idx, item_idx, event, weight e timestamp.
        context_ts: Timestamp de contexto default para pontuação
            (default: último timestamp do histórico).
    """

    def __init__(
        self, history: pd.DataFrame, context_ts: pd.Timestamp | None = None
    ) -> None:
        self.context_ts = context_ts or history["timestamp"].max()
        self._user_state = history.groupby("user_idx").agg(
            frequency=("user_idx", "count"),
            engagement_score=("weight", "sum"),
            last_ts=("timestamp", "max"),
        )
        views = history.loc[history["event"] == "view"]
        self._item_views = views.groupby("item_idx").size()
        pair_agg = views.groupby(["user_idx", "item_idx"]).agg(
            count=("timestamp", "size"), last_ts=("timestamp", "max")
        )
        self._pair_state: dict[tuple[int, int], tuple[int, pd.Timestamp]] = {
            (int(u), int(i)): (int(c), last_ts)
            for (u, i), c, last_ts in zip(
                pair_agg.index, pair_agg["count"], pair_agg["last_ts"], strict=True
            )
        }
        self._catalog = np.sort(history["item_idx"].unique()).astype("int64")

    @property
    def catalog(self) -> np.ndarray:
        """Itens presentes no histórico (elegíveis como negativos)."""
        return self._catalog

    def known_users(self) -> set[int]:
        """Usuários com pelo menos um evento no histórico."""
        return {int(u) for u in self._user_state.index}

    def view_counts(self, items: np.ndarray) -> np.ndarray:
        """Total de views por item no histórico (0 para desconhecidos)."""
        return pd.Series(items).map(self._item_views).fillna(0.0).to_numpy("float64")

    def pair_features(
        self, users: np.ndarray, items: np.ndarray, ts: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Contagem total e recência (vs. ``ts``) de views do par (user, item).

        Mesmo estilo de ``view_counts`` (agregado sobre todo o histórico, não
        um cursor *as-of* por linha) — consistente com o par não ter menos
        precisão que o par-a-par já usado para ``view_count``/``recency_days``.

        Args:
            users: user_idx por linha.
            items: item_idx por linha (alinhado a users).
            ts: Timestamp de contexto por linha (alinhado a users/items).

        Returns:
            Tupla (user_item_view_count, user_item_recency_days).
        """
        counts = np.empty(len(users), dtype="float64")
        recency = np.empty(len(users), dtype="float64")
        for pos, (u, i, t) in enumerate(zip(users, items, ts, strict=True)):
            state = self._pair_state.get((int(u), int(i)))
            if state is None:
                counts[pos] = 0.0
                recency[pos] = NO_HISTORY_RECENCY
            else:
                count, last_ts = state
                counts[pos] = count
                recency[pos] = (t - last_ts) / np.timedelta64(1, "D")
        return counts, recency

    def matrix(
        self,
        users: np.ndarray,
        items: np.ndarray,
        context_ts_by_user: dict[int, pd.Timestamp] | None = None,
    ) -> np.ndarray:
        """Monta a matriz FEATURE_COLS para os pares (user, item) dados.

        Args:
            users: user_idx por linha.
            items: item_idx por linha (alinhado a users).
            context_ts_by_user: Timestamp de pontuação por usuário
                (ex.: primeiro evento de teste). Default: ``context_ts``.

        Returns:
            Matriz float32 na ordem do contrato de features.
        """
        state = self._user_state.reindex(users)
        ts = pd.to_datetime(pd.Series(users).map(context_ts_by_user or {}))
        ts = ts.fillna(self.context_ts)
        delta = (ts.to_numpy() - state["last_ts"].to_numpy()).astype(
            "timedelta64[s]"
        ).astype("float64") / _SECONDS_PER_DAY
        recency = np.where(np.isnan(delta), NO_HISTORY_RECENCY, delta)
        pair_count, pair_recency = self.pair_features(users, items, ts.to_numpy())
        cols = {
            "user_idx": np.asarray(users, dtype="float64"),
            "item_idx": np.asarray(items, dtype="float64"),
            "hour": ts.dt.hour.to_numpy("float64"),
            "day_of_week": ts.dt.dayofweek.to_numpy("float64"),
            "frequency": state["frequency"].fillna(0.0).to_numpy("float64"),
            "engagement_score": (
                state["engagement_score"].fillna(0.0).to_numpy("float64")
            ),
            "recency_days": recency,
            "view_count": self.view_counts(items),
            "user_item_view_count": pair_count,
            "user_item_recency_days": pair_recency,
        }
        return np.column_stack([cols[c] for c in FEATURE_COLS]).astype("float32")


def make_model_scorer(
    model: RecommenderBase,
    lookup: HistoryFeatureLookup,
    context_ts_by_user: dict[int, pd.Timestamp] | None = None,
    head: str = "strong",
) -> PairScorer:
    """Scorer que pontua pares com o modelo sobre a matriz do contrato.

    Args:
        model: Modelo treinado (predict_proba sobre FEATURE_COLS).
        lookup: Estado derivado do histórico.
        context_ts_by_user: Timestamp de contexto por usuário.
        head: Qual cabeça do NCF pontuar — ``"strong"`` (default, cenário
            principal) ou ``"view"`` (relevância ampla, spec 002, FR-004).
            Ignorado para modelos sem múltiplas cabeças (ex.: baselines).

    Returns:
        Função (users, items) → scores.
    """

    def scorer(users: np.ndarray, items: np.ndarray) -> np.ndarray:
        out = np.empty(len(users), dtype="float64")
        for start in range(0, len(users), _SCORING_CHUNK):
            end = start + _SCORING_CHUNK
            matrix = lookup.matrix(
                users[start:end], items[start:end], context_ts_by_user
            )
            if isinstance(model, NCFRecommender):
                out[start:end] = model.predict_proba(matrix, head=head)
            else:
                out[start:end] = model.predict_proba(matrix)
        return out

    return scorer


def make_popularity_scorer(lookup: HistoryFeatureLookup) -> PairScorer:
    """Scorer baseline: score do item = nº de views no histórico (FR-006).

    Args:
        lookup: Estado derivado do histórico.

    Returns:
        Função (users, items) → scores.
    """

    def scorer(users: np.ndarray, items: np.ndarray) -> np.ndarray:  # noqa: ARG001
        return lookup.view_counts(items)

    return scorer
