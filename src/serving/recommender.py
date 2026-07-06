"""Serviço de recomendação: pontua candidatos e ranqueia por probabilidade.

Para usuários conhecidos, monta a matriz de features dos itens candidatos e
usa predict_proba. Para usuários desconhecidos, cai para popularidade.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

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
        candidates = self.store.candidate_items(self.max_candidates)
        seen = self.store.seen_items(user_id)
        items = candidates[~np.isin(candidates, list(seen))]
        if items.size == 0:
            return []
        scores = self.model.predict_proba(self._build_matrix(user_id, items))
        order = np.argsort(scores)[::-1][:top_k]
        return [
            {"item_idx": int(items[i]), "score": round(float(scores[i]), 6)}
            for i in order
        ]

    def _build_matrix(self, user_id: int, items: np.ndarray) -> np.ndarray:
        """Monta a matriz de features na ordem esperada pelo modelo."""
        freq, eng, rec = self.store.user_features(user_id)
        now = datetime.now()
        n = items.shape[0]
        return np.column_stack(
            [
                np.full(n, user_id),
                items,
                np.full(n, now.hour),
                np.full(n, now.weekday()),
                np.full(n, freq),
                np.full(n, eng),
                np.full(n, rec),
                self.store.view_counts(items),
            ]
        ).astype("float32")

    def _popularity_recommend(self, top_k: int) -> list[dict]:
        """Retorna os itens mais populares como fallback."""
        return [
            {"item_idx": item, "score": float(vc)}
            for item, vc in self.store.popular_items(top_k)
        ]
