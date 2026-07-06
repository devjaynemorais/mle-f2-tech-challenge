"""Testes da API de serving (store, recommender, loader, endpoints)."""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.serving.recommender import RecommendationService
from src.serving.store import FeatureStore


def _sample_interactions() -> pd.DataFrame:
    # user 1 viewed items 10, 11; user 2 viewed item 11. view_counts: 11>10>12.
    return pd.DataFrame(
        {
            "user_idx": [1, 1, 2],
            "item_idx": [10, 11, 11],
            "frequency": [2, 2, 1],
            "engagement_score": [4.0, 4.0, 1.0],
            "recency_days": [3, 3, 5],
            "view_count": [5, 9, 9],
        }
    )


def _store_with_item12() -> FeatureStore:
    df = _sample_interactions()
    extra = pd.DataFrame(
        {
            "user_idx": [3],
            "item_idx": [12],
            "frequency": [1],
            "engagement_score": [1.0],
            "recency_days": [1],
            "view_count": [1],
        }
    )
    return FeatureStore(pd.concat([df, extra], ignore_index=True))


def test_store_user_and_item_counts():
    store = _store_with_item12()
    assert store.n_users == 3
    assert store.n_items == 3


def test_store_user_features_and_seen():
    store = _store_with_item12()
    assert store.has_user(1) is True
    assert store.has_user(99) is False
    assert store.user_features(1) == (2.0, 4.0, 3.0)
    assert store.seen_items(1) == {10, 11}


def test_store_candidates_are_popularity_sorted():
    store = _store_with_item12()
    candidates = store.candidate_items(max_candidates=10)
    assert list(candidates) == [11, 10, 12]
    assert list(store.view_counts(np.array([11, 10, 12]))) == [9, 5, 1]
    assert store.popular_items(2) == [(11, 9), (10, 5)]


class _StubModel:
    def predict_proba(self, X):
        return np.full(len(X), 0.5)


def test_load_from_local_reads_record_and_unpickles(tmp_path: Path):
    from src.serving.model_loader import _load_from_local

    run_id = "abc123"
    artifacts_dir = tmp_path / "artifacts"
    (artifacts_dir / run_id).mkdir(parents=True)
    with open(artifacts_dir / run_id / "model.pkl", "wb") as f:
        pickle.dump(_StubModel(), f)
    record = tmp_path / "promoted_model.json"
    record.write_text(json.dumps({"run_id": run_id}))

    model = _load_from_local(record, artifacts_dir)
    assert model.predict_proba(np.zeros((2, 8))).tolist() == [0.5, 0.5]


class _ItemIdxModel:
    """Modelo fake: score = item_idx (coluna 1), determinístico."""

    def predict_proba(self, X):
        return X[:, 1].astype(float)


def _service() -> RecommendationService:
    store = _store_with_item12()
    return RecommendationService(_ItemIdxModel(), store, max_candidates=10)


def test_recommend_ranks_and_excludes_seen():
    result = _service().recommend(user_id=1, top_k=5)
    assert result["strategy"] == "model"
    items = [r["item_idx"] for r in result["recommendations"]]
    # user 1 já viu 10 e 11; sobra apenas 12
    assert items == [12]
    assert result["count"] == 1


def test_recommend_unknown_user_uses_popularity():
    result = _service().recommend(user_id=999, top_k=2)
    assert result["strategy"] == "popularity"
    assert [r["item_idx"] for r in result["recommendations"]] == [11, 10]


def test_recommend_scores_are_descending():
    store = FeatureStore(_sample_interactions())  # itens 10, 11 (nenhum item 12)
    svc = RecommendationService(_ItemIdxModel(), store, max_candidates=10)
    result = svc.recommend(user_id=2, top_k=5)  # user 2 viu 11 → sobra 10
    scores = [r["score"] for r in result["recommendations"]]
    assert scores == sorted(scores, reverse=True)
