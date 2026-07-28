"""Serviço de recomendação: pontua candidatos e ranqueia por probabilidade.

Para usuários conhecidos, monta a matriz de features dos itens candidatos e
usa predict_proba. Para usuários desconhecidos, cai para popularidade.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from src.data.feature_contract import FEATURE_COLS
from src.models.base import RecommenderBase
from src.serving.store import FeatureStore


class RecommendationService:
    """Gera recomendações top-K a partir do modelo e do feature store."""

    def __init__(
        self,
        model: RecommenderBase,
        store: FeatureStore,
        max_candidates: int = 5000,
    ) -> None:
        self.model = model
        self.store = store
        self.max_candidates = max_candidates

    def recommend(self, user_id: int, top_k: int) -> dict:
        """Retorna as top-k recomendações e a estratégia usada."""
        recs: list[dict] = []
        if self.store.has_user(user_id):
            recs = self._model_recommend(user_id, top_k)
        if recs:
            strategy = "model"
        else:
            recs, strategy = self._popularity_recommend(top_k), "popularity"
        return {
            "user_id": int(user_id),
            "strategy": strategy,
            "count": len(recs),
            "recommendations": recs,
        }

    def _model_recommend(self, user_id: int, top_k: int) -> list[dict]:
        """Pontua os itens candidatos não vistos e ranqueia (vazio se não há)."""
        items, _, scores = self._score_candidates(user_id)
        if items.size == 0:
            return []
        order = np.argsort(scores)[::-1][:top_k]
        return [
            {"item_idx": int(items[i]), "score": round(float(scores[i]), 6)}
            for i in order
        ]

    def _score_candidates(
        self, user_id: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filtra candidatos já vistos, monta a matriz e pontua (cabeça strong).

        Retorna (items, matrix, scores) — arrays vazios se não sobrar nenhum
        candidato após excluir os já vistos.
        """
        candidates = self.store.candidate_items(self.max_candidates)
        seen = self.store.seen_items(user_id)
        items = candidates[~np.isin(candidates, list(seen))]
        if items.size == 0:
            empty_matrix = np.empty((0, len(FEATURE_COLS)), dtype="float32")
            return items, empty_matrix, np.empty(0, dtype="float32")
        matrix = self._build_matrix(user_id, items)
        scores = self.model.predict_proba(matrix)
        return items, matrix, scores

    def explain(self, user_id: int, top_k: int) -> dict:
        """Monta um payload passo a passo do que acontece em /recommend.

        Expõe o que fica escondido dentro de ``recommend()``: quem é o
        usuário (features causais), o tamanho do pool de candidatos, quantos
        foram excluídos por já vistos, a matriz de features de cada item
        pontuado (ordem de ``FEATURE_COLS``) e por que caiu no fallback de
        popularidade, quando é o caso.
        """
        user_id = int(user_id)
        known = self.store.has_user(user_id)
        seen = self.store.seen_items(user_id)
        user_block: dict = {"user_id": user_id, "known": known, "seen_count": len(seen)}
        if known:
            freq, eng, rec = self.store.user_features(user_id)
            user_block.update(frequency=freq, engagement_score=eng, recency_days=rec)

        # Filtro de vistos independe de o usuário ser conhecido (seen_items
        # já devolve vazio pra desconhecidos) — calculado uma vez aqui pra
        # "excluded_seen" ser correto em todos os ramos (não é 0 sempre que
        # o fallback é por usuário desconhecido, e sim quando não há nada
        # marcado como visto).
        candidates = self.store.candidate_items(self.max_candidates)
        pool_size = int(candidates.size)
        remaining = int((~np.isin(candidates, list(seen))).sum())

        strategy = "popularity"
        fallback_reason: str | None = None if known else "unknown_user"
        scored: list[dict] = []

        if known and remaining > 0:
            items, matrix, scores = self._score_candidates(user_id)
            strategy = "model"
            order = np.argsort(scores)[::-1][:top_k]
            scored = [
                {
                    **self.store.item_profile(items[i]),
                    "score": round(float(scores[i]), 6),
                    "features": dict(
                        zip(
                            FEATURE_COLS,
                            (float(v) for v in matrix[i]),
                            strict=True,
                        )
                    ),
                }
                for i in order
            ]
        elif known:
            fallback_reason = "no_candidates_left"

        if strategy == "popularity":
            scored = [
                {**self.store.item_profile(item), "score": float(vc)}
                for item, vc in self.store.popular_items(top_k)
            ]

        return {
            "user": user_block,
            "candidates": {
                "pool_size": pool_size,
                "excluded_seen": pool_size - remaining,
                "remaining": remaining,
            },
            "scored": scored,
            "strategy": strategy,
            "fallback_reason": fallback_reason,
        }

    def _build_matrix(self, user_id: int, items: np.ndarray) -> np.ndarray:
        """Monta a matriz na ordem do contrato de features (FR-007)."""
        freq, eng, rec = self.store.user_features(user_id)
        now = datetime.now()
        n = items.shape[0]
        pair_count, pair_recency = self.store.pair_features(user_id, items)
        cols = {
            "user_idx": np.full(n, user_id),
            "item_idx": items,
            "hour": np.full(n, now.hour),
            "day_of_week": np.full(n, now.weekday()),
            "frequency": np.full(n, freq),
            "engagement_score": np.full(n, eng),
            "recency_days": np.full(n, rec),
            "view_count": self.store.view_counts(items),
            "user_item_view_count": pair_count,
            "user_item_recency_days": pair_recency,
        }
        return np.column_stack([cols[c] for c in FEATURE_COLS]).astype("float32")

    def _popularity_recommend(self, top_k: int) -> list[dict]:
        """Retorna os itens mais populares como fallback."""
        return [
            {"item_idx": item, "score": float(vc)}
            for item, vc in self.store.popular_items(top_k)
        ]
