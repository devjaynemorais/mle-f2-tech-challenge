"""Feature store em memória para a API de serving.

Constrói tabelas de lookup a partir dos EVENTOS de data/processed/ —
o estado corrente do usuário (frequency, engagement, recency) e do item
(view_count) é agregado aqui, com a mesma semântica das features causais
do treino, avaliadas "no fim do histórico" (FR-007).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.feature_contract import NO_HISTORY_RECENCY

_SPLIT_FILES = ("train.parquet", "val.parquet", "test.parquet")
_SECONDS_PER_DAY = 86400.0


class FeatureStore:
    """Tabelas de lookup para pontuar candidatos no serving.

    Args:
        interactions: Eventos com colunas user_idx, item_idx, event,
            weight e timestamp (splits de data/processed/ concatenados).
    """

    def __init__(self, interactions: pd.DataFrame) -> None:
        self._reference_ts = interactions["timestamp"].max()
        self._users = self._build_users(interactions)
        self._item_vc = self._build_item_views(interactions)
        self._pairs = self._build_pair_views(interactions)
        self._seen = self._build_seen(interactions)
        self._popular = np.array(
            sorted(self._item_vc, key=lambda i: self._item_vc[i], reverse=True),
            dtype="int64",
        )
        self._rank = {int(item): pos + 1 for pos, item in enumerate(self._popular)}
        self._item_cat = self._build_item_categories(interactions)
        # Referência crua p/ a timeline da demo — categoria/popularidade do
        # item vêm de item_profile(), não precisam duplicar aqui.
        self._timeline_df = interactions[
            ["user_idx", "item_idx", "event", "timestamp"]
        ].copy()

    @classmethod
    def from_processed(cls, processed_dir: Path) -> FeatureStore:
        """Carrega e concatena os splits de data/processed/."""
        frames = [
            pd.read_parquet(processed_dir / f)
            for f in _SPLIT_FILES
            if (processed_dir / f).exists()
        ]
        if not frames:
            raise FileNotFoundError(f"Nenhum split encontrado em {processed_dir}")
        return cls(pd.concat(frames, ignore_index=True))

    def _build_users(self, df: pd.DataFrame) -> dict[int, tuple[float, float, float]]:
        """Mapeia user_idx → (frequency, engagement_score, recency_days)."""
        agg = df.groupby("user_idx").agg(
            frequency=("user_idx", "count"),
            engagement_score=("weight", "sum"),
            last_ts=("timestamp", "max"),
        )
        recency = (
            (self._reference_ts - agg["last_ts"]).dt.total_seconds() / _SECONDS_PER_DAY
        ).fillna(NO_HISTORY_RECENCY)
        return {
            int(u): (float(f), float(e), float(r))
            for u, f, e, r in zip(
                agg.index,
                agg["frequency"],
                agg["engagement_score"],
                recency,
                strict=True,
            )
        }

    def _build_item_views(self, df: pd.DataFrame) -> dict[int, int]:
        """Mapeia item_idx → total de views (0 para itens sem view)."""
        views = df.loc[df["event"] == "view"].groupby("item_idx").size()
        all_items = df["item_idx"].unique()
        return {int(i): int(views.get(i, 0)) for i in all_items}

    def _build_pair_views(
        self, df: pd.DataFrame
    ) -> dict[tuple[int, int], tuple[int, float]]:
        """Mapeia (user_idx, item_idx) → (views do par, recência em dias)."""
        views = df.loc[df["event"] == "view", ["user_idx", "item_idx", "timestamp"]]
        agg = views.groupby(["user_idx", "item_idx"]).agg(
            count=("timestamp", "size"), last_ts=("timestamp", "max")
        )
        recency = (
            self._reference_ts - agg["last_ts"]
        ).dt.total_seconds() / _SECONDS_PER_DAY
        return {
            (int(u), int(i)): (int(c), float(r))
            for (u, i), c, r in zip(agg.index, agg["count"], recency, strict=True)
        }

    def _build_seen(self, df: pd.DataFrame) -> dict[int, set[int]]:
        """Mapeia user_idx → conjunto de item_idx já vistos."""
        return {
            int(u): {int(x) for x in items}
            for u, items in df.groupby("user_idx")["item_idx"]
        }

    def _build_item_categories(self, df: pd.DataFrame) -> dict[int, int]:
        """Mapeia item_idx → cat_idx (mesmo espaço do embedding de categoria).

        Ausente em datasets sintéticos de teste que não passaram pelo estágio
        ``feature_eng`` (sem coluna ``cat_idx``) — nesse caso o dict fica
        vazio e ``item_category()`` devolve ``None``.
        """
        if "cat_idx" not in df.columns:
            return {}
        first_cat = df.groupby("item_idx")["cat_idx"].first()
        return {int(i): int(c) for i, c in first_cat.items()}

    @property
    def n_users(self) -> int:
        """Número de usuários conhecidos."""
        return len(self._users)

    @property
    def n_items(self) -> int:
        """Número de itens no catálogo."""
        return len(self._item_vc)

    def has_user(self, user_idx: int) -> bool:
        """Retorna True se o usuário tem histórico."""
        return int(user_idx) in self._users

    def user_features(self, user_idx: int) -> tuple[float, float, float]:
        """Retorna (frequency, engagement_score, recency_days) do usuário."""
        return self._users[int(user_idx)]

    def seen_items(self, user_idx: int) -> set[int]:
        """Retorna os itens já vistos pelo usuário (vazio se desconhecido)."""
        return self._seen.get(int(user_idx), set())

    def candidate_items(self, max_candidates: int) -> np.ndarray:
        """Retorna os item_idx candidatos, ordenados por popularidade."""
        return self._popular[:max_candidates]

    def view_counts(self, items: np.ndarray) -> np.ndarray:
        """Retorna os view_counts alinhados ao array de itens."""
        return np.array([self._item_vc.get(int(i), 0) for i in items], dtype="int64")

    def pair_features(
        self, user_idx: int, items: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retorna (user_item_view_count, user_item_recency_days) por item.

        Sentinela ``NO_HISTORY_RECENCY`` quando o par (usuário, item) nunca
        teve view (spec 002, FR-007/FR-008).
        """
        counts = np.empty(len(items), dtype="float64")
        recency = np.empty(len(items), dtype="float64")
        for pos, item in enumerate(items):
            state = self._pairs.get((int(user_idx), int(item)))
            counts[pos] = state[0] if state else 0.0
            recency[pos] = state[1] if state else NO_HISTORY_RECENCY
        return counts, recency

    def popular_items(self, k: int) -> list[tuple[int, int]]:
        """Retorna os top-k (item_idx, view_count) por popularidade."""
        return [(int(i), self._item_vc[int(i)]) for i in self._popular[:k]]

    def item_profile(self, item_idx: int) -> dict:
        """Retorna um perfil mínimo do item — pra dar identidade a um item_idx.

        O dataset é anonimizado (sem nome/título de produto), então
        ``cat_idx`` (mesmo espaço do embedding de categoria do modelo) e a
        posição no ranking de popularidade são o máximo de "quem é esse
        item" que dá pra mostrar honestamente.
        """
        item_idx = int(item_idx)
        return {
            "item_idx": item_idx,
            "cat_idx": self._item_cat.get(item_idx),
            "view_count": self._item_vc.get(item_idx, 0),
            "popularity_rank": self._rank.get(item_idx),
        }

    def timeline(self, user_idx: int, limit: int = 15) -> list[dict]:
        """Retorna os últimos ``limit`` eventos reais do usuário, em ordem.

        Cada evento já vem enriquecido com o perfil do item (categoria,
        popularidade) — é o que a demo usa pra mostrar "o que esse usuário
        fez" antes de pedir a recomendação. Vazio se o usuário é desconhecido.
        """
        user_idx = int(user_idx)
        sub = self._timeline_df.loc[self._timeline_df["user_idx"] == user_idx]
        if sub.empty:
            return []
        sub = sub.sort_values("timestamp").tail(limit)
        return [
            {
                "timestamp": row.timestamp.isoformat(),
                "event": row.event,
                **self.item_profile(row.item_idx),
            }
            for row in sub.itertuples()
        ]

    def sample_users(self) -> dict[str, dict]:
        """Retorna 3 usuários reais representativos para a demo interativa.

        ``active`` (maior frequency), ``sparse`` (menor frequency entre os
        conhecidos) e ``cold`` (id garantidamente ausente, dispara o
        fallback de popularidade no serving).
        """
        active_id = max(self._users, key=lambda u: self._users[u][0])
        sparse_id = min(self._users, key=lambda u: self._users[u][0])
        cold_id = max(self._users) + 1

        def _describe(user_idx: int) -> dict:
            freq, eng, rec = self._users[user_idx]
            return {
                "user_id": user_idx,
                "known": True,
                "frequency": freq,
                "engagement_score": eng,
                "recency_days": rec,
            }

        return {
            "active": _describe(active_id),
            "sparse": _describe(sparse_id),
            "cold": {"user_id": cold_id, "known": False},
        }
