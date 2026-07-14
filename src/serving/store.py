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
        self._seen = self._build_seen(interactions)
        self._popular = np.array(
            sorted(self._item_vc, key=lambda i: self._item_vc[i], reverse=True),
            dtype="int64",
        )

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
            (self._reference_ts - agg["last_ts"]).dt.total_seconds()
            / _SECONDS_PER_DAY
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

    def _build_seen(self, df: pd.DataFrame) -> dict[int, set[int]]:
        """Mapeia user_idx → conjunto de item_idx já vistos."""
        return {
            int(u): {int(x) for x in items}
            for u, items in df.groupby("user_idx")["item_idx"]
        }

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

    def popular_items(self, k: int) -> list[tuple[int, int]]:
        """Retorna os top-k (item_idx, view_count) por popularidade."""
        return [(int(i), self._item_vc[int(i)]) for i in self._popular[:k]]
