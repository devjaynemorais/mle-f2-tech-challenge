"""Testes da API de serving (store, recommender, loader, endpoints)."""

import json
import pickle
import socket
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.serving import api as api_module
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


def _client_with_service() -> TestClient:
    # TestClient(app) sem "with" NÃO dispara o lifespan → sem MLflow/disco.
    client = TestClient(api_module.app)
    api_module.state["service"] = _service()
    return client


def test_health_reports_loaded():
    client = _client_with_service()
    body = client.get("/health").json()
    assert body == {"status": "ok", "model_loaded": True, "n_users": 3, "n_items": 3}


def test_recommend_endpoint_happy_path():
    client = _client_with_service()
    resp = client.get("/recommend", params={"user_id": 1, "top_k": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["strategy"] == "model"
    assert [r["item_idx"] for r in body["recommendations"]] == [12]


def test_recommend_top_k_out_of_range_returns_422():
    client = _client_with_service()
    resp_low = client.get("/recommend", params={"user_id": 1, "top_k": 0})
    resp_high = client.get("/recommend", params={"user_id": 1, "top_k": 101})
    assert resp_low.status_code == 422
    assert resp_high.status_code == 422


def test_recommend_without_service_returns_503():
    client = TestClient(api_module.app)
    api_module.state["service"] = None
    assert client.get("/recommend", params={"user_id": 1}).status_code == 503


def test_recommend_orders_multiple_candidates_descending():
    # user 3 já viu apenas o item 12; sobram 11 e 10 (score = item_idx)
    result = _service().recommend(user_id=3, top_k=5)
    assert result["strategy"] == "model"
    items = [r["item_idx"] for r in result["recommendations"]]
    assert items == [11, 10]


def test_recommend_known_user_seen_all_falls_back_to_popularity():
    df = pd.DataFrame(
        {
            "user_idx": [7, 7, 7],
            "item_idx": [10, 11, 12],
            "frequency": [3, 3, 3],
            "engagement_score": [3.0, 3.0, 3.0],
            "recency_days": [1, 1, 1],
            "view_count": [5, 9, 1],
        }
    )
    svc = RecommendationService(_ItemIdxModel(), FeatureStore(df), max_candidates=10)
    result = svc.recommend(user_id=7, top_k=5)
    assert result["strategy"] == "popularity"


def test_registry_reachable_false_for_closed_port():
    from src.serving.model_loader import _registry_reachable

    # porta 9 (discard) quase sempre fechada → deve falhar rápido
    assert _registry_reachable("http://127.0.0.1:9", timeout=0.5) is False


def test_registry_reachable_true_for_non_http_scheme():
    from src.serving.model_loader import _registry_reachable

    assert _registry_reachable("file:///tmp/mlruns") is True


def test_registry_reachable_true_when_listening():
    from src.serving.model_loader import _registry_reachable

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    try:
        assert _registry_reachable(f"http://127.0.0.1:{port}", timeout=1.0) is True
    finally:
        srv.close()
