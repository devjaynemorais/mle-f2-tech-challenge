"""Testes da API de serving (store, recommender, loader, endpoints)."""

import json
import pickle
import socket
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.data.feature_contract import FEATURE_COLS
from src.serving import api as api_module
from src.serving.recommender import RecommendationService
from src.serving.store import FeatureStore


def _events(rows: list[tuple[int, int, str, str]]) -> pd.DataFrame:
    """Monta eventos (user, item, event, timestamp) com pesos padrão."""
    weights = {"view": 1, "addtocart": 3, "transaction": 5}
    return pd.DataFrame(
        {
            "user_idx": [r[0] for r in rows],
            "item_idx": [r[1] for r in rows],
            "event": [r[2] for r in rows],
            "weight": [weights[r[2]] for r in rows],
            "timestamp": pd.to_datetime([r[3] for r in rows]),
        }
    )


def _sample_interactions() -> pd.DataFrame:
    # user 1 viu 10, 10, 11; user 2 viu 11, 11. view_counts: 11=3 > 10=2.
    return _events(
        [
            (1, 10, "view", "2015-05-01"),
            (1, 10, "view", "2015-05-02"),
            (1, 11, "view", "2015-05-03"),
            (2, 11, "view", "2015-05-04"),
            (2, 11, "view", "2015-05-05"),
        ]
    )


def _store_with_item12() -> FeatureStore:
    extra = _events([(3, 12, "view", "2015-05-06")])
    return FeatureStore(pd.concat([_sample_interactions(), extra], ignore_index=True))


def test_from_processed_concatenates_existing_splits(tmp_path: Path):
    train, val = _sample_interactions().iloc[:3], _sample_interactions().iloc[3:]
    train.to_parquet(tmp_path / "train.parquet", index=False)
    val.to_parquet(tmp_path / "val.parquet", index=False)
    # test.parquet ausente — from_processed deve ignorar o que não existe.
    store = FeatureStore.from_processed(tmp_path)
    assert store.n_users == 2
    assert store.n_items == 2


def test_from_processed_raises_when_no_split_exists(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="Nenhum split"):
        FeatureStore.from_processed(tmp_path)


def test_store_user_and_item_counts():
    store = _store_with_item12()
    assert store.n_users == 3
    assert store.n_items == 3


def test_store_user_features_and_seen():
    store = _store_with_item12()
    assert store.has_user(1) is True
    assert store.has_user(99) is False
    # user 1: 3 eventos view (peso 1), último em 05-03; referência = 05-06
    assert store.user_features(1) == (3.0, 3.0, 3.0)
    assert store.seen_items(1) == {10, 11}


def test_store_candidates_are_popularity_sorted():
    store = _store_with_item12()
    candidates = store.candidate_items(max_candidates=10)
    assert list(candidates) == [11, 10, 12]
    assert list(store.view_counts(np.array([11, 10, 12]))) == [3, 2, 1]
    assert store.popular_items(2) == [(11, 3), (10, 2)]


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


def test_load_from_registry_raises_when_unreachable(monkeypatch: pytest.MonkeyPatch):
    from src.serving import model_loader

    monkeypatch.setattr(
        model_loader, "_registry_reachable", lambda uri, timeout=2.0: False
    )
    with pytest.raises(ConnectionError):
        model_loader._load_from_registry()


def test_load_from_registry_raises_when_no_production_version(
    monkeypatch: pytest.MonkeyPatch,
):
    import mlflow.tracking as mlflow_tracking_module

    from src.serving import model_loader

    monkeypatch.setattr(
        model_loader, "_registry_reachable", lambda uri, timeout=2.0: True
    )

    class _FakeClient:
        def __init__(self, tracking_uri: str | None = None) -> None:
            pass

        def get_latest_versions(self, name: str, stages: list[str]) -> list:
            return []

    monkeypatch.setattr(mlflow_tracking_module, "MlflowClient", _FakeClient)
    with pytest.raises(ValueError, match="Production"):
        model_loader._load_from_registry()


def test_load_from_registry_downloads_and_unpickles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    import mlflow
    import mlflow.tracking as mlflow_tracking_module

    from src.serving import model_loader

    monkeypatch.setattr(
        model_loader, "_registry_reachable", lambda uri, timeout=2.0: True
    )

    class _FakeVersion:
        run_id = "abc123"

    class _FakeClient:
        def __init__(self, tracking_uri: str | None = None) -> None:
            pass

        def get_latest_versions(self, name: str, stages: list[str]) -> list:
            return [_FakeVersion()]

    monkeypatch.setattr(mlflow_tracking_module, "MlflowClient", _FakeClient)

    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(_StubModel(), f)
    monkeypatch.setattr(
        mlflow.artifacts, "download_artifacts", lambda **kwargs: str(model_path)
    )

    model = model_loader._load_from_registry()
    assert model.predict_proba(np.zeros((2, 8))).tolist() == [0.5, 0.5]


def test_load_production_model_uses_registry_when_available(
    monkeypatch: pytest.MonkeyPatch,
):
    from src.serving import model_loader

    sentinel = _StubModel()
    monkeypatch.setattr(model_loader, "_load_from_registry", lambda: sentinel)
    assert model_loader.load_production_model() is sentinel


def test_load_production_model_falls_back_to_local_on_registry_failure(
    monkeypatch: pytest.MonkeyPatch,
):
    from src.serving import model_loader

    def _boom() -> None:
        raise RuntimeError("registry indisponível")

    sentinel = _StubModel()
    monkeypatch.setattr(model_loader, "_load_from_registry", _boom)
    monkeypatch.setattr(
        model_loader, "_load_from_local", lambda record_path, artifacts_dir: sentinel
    )
    assert model_loader.load_production_model() is sentinel


class _ItemIdxModel:
    """Modelo fake: score = item_idx (coluna 1 do contrato), determinístico."""

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
    store = FeatureStore(_sample_interactions())  # itens 10, 11 (sem item 12)
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


def test_health_reports_degraded_without_service():
    client = TestClient(api_module.app)
    api_module.state["service"] = None
    assert client.get("/health").json() == {"status": "degraded", "model_loaded": False}


def test_root_returns_service_metadata():
    client = TestClient(api_module.app)
    body = client.get("/").json()
    assert body == {"service": "RetailRocket Recommender API", "version": "0.1.0"}


def test_lifespan_loads_service_on_startup(monkeypatch: pytest.MonkeyPatch):
    """``with TestClient`` dispara o lifespan de verdade (startup + shutdown)."""
    monkeypatch.setattr(api_module, "load_production_model", lambda: _ItemIdxModel())
    monkeypatch.setattr(
        api_module.FeatureStore,
        "from_processed",
        classmethod(lambda cls, path: _store_with_item12()),
    )
    with TestClient(api_module.app) as client:
        body = client.get("/health").json()
        assert body == {
            "status": "ok",
            "model_loaded": True,
            "n_users": 3,
            "n_items": 3,
        }


def test_lifespan_degrades_when_model_loading_fails(monkeypatch: pytest.MonkeyPatch):
    def _boom() -> None:
        raise RuntimeError("sem modelo Production disponível")

    # A lifespan não reseta o estado no shutdown — sem isso, um teste
    # anterior que deixou state["service"] setado vazaria pra este.
    api_module.state["service"] = None
    monkeypatch.setattr(api_module, "load_production_model", _boom)
    with TestClient(api_module.app) as client:
        assert client.get("/health").json() == {
            "status": "degraded",
            "model_loaded": False,
        }


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
    df = _events(
        [
            (7, 10, "view", "2015-05-01"),
            (7, 11, "view", "2015-05-02"),
            (7, 12, "view", "2015-05-03"),
        ]
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


# ─── Demo interativa: sample_users + explain() ─────────────────────────────


def test_sample_users_active_sparse_and_cold():
    # user 1: 3 views (freq mais alta); user 3: 1 view (freq mais baixa).
    store = _store_with_item12()
    samples = store.sample_users()
    assert samples["active"]["user_id"] == 1
    assert samples["active"]["known"] is True
    assert samples["sparse"]["user_id"] == 3
    assert samples["cold"] == {"user_id": 4, "known": False}


def test_explain_known_user_returns_features_in_contract_order():
    result = _service().explain(user_id=3, top_k=5)
    assert result["strategy"] == "model"
    assert result["fallback_reason"] is None
    assert result["user"]["known"] is True
    assert result["candidates"]["remaining"] == 2  # sobram 11 e 10
    items = [row["item_idx"] for row in result["scored"]]
    assert items == [11, 10]
    assert list(result["scored"][0]["features"].keys()) == FEATURE_COLS


def test_explain_unknown_user_falls_back_to_popularity():
    result = _service().explain(user_id=999, top_k=2)
    assert result["strategy"] == "popularity"
    assert result["fallback_reason"] == "unknown_user"
    assert result["user"]["known"] is False
    assert [row["item_idx"] for row in result["scored"]] == [11, 10]
    # Usuário desconhecido não "viu" nada — excluded_seen não pode ser o
    # pool inteiro só porque o fallback foi por usuário desconhecido.
    assert result["candidates"]["excluded_seen"] == 0
    assert result["candidates"]["remaining"] == result["candidates"]["pool_size"]


def test_explain_known_user_seen_all_falls_back_to_popularity():
    df = _events(
        [
            (7, 10, "view", "2015-05-01"),
            (7, 11, "view", "2015-05-02"),
            (7, 12, "view", "2015-05-03"),
        ]
    )
    svc = RecommendationService(_ItemIdxModel(), FeatureStore(df), max_candidates=10)
    result = svc.explain(user_id=7, top_k=5)
    assert result["strategy"] == "popularity"
    assert result["fallback_reason"] == "no_candidates_left"
    assert result["candidates"]["remaining"] == 0
    assert result["candidates"]["excluded_seen"] == result["candidates"]["pool_size"]


def test_explain_endpoint_happy_path():
    client = _client_with_service()
    resp = client.get("/explain", params={"user_id": 3, "top_k": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["strategy"] == "model"
    assert body["scored"][0]["features"]


def test_explain_without_service_returns_503():
    client = TestClient(api_module.app)
    api_module.state["service"] = None
    assert client.get("/explain", params={"user_id": 1}).status_code == 503


def test_sample_users_endpoint_happy_path():
    client = _client_with_service()
    resp = client.get("/demo/sample-users")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"active", "sparse", "cold"}


def test_demo_page_serves_html():
    client = _client_with_service()
    resp = client.get("/demo")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "RetailRocket" in resp.text


# ─── Demo interativa: item_profile + timeline ──────────────────────────────


def _events_with_category(rows: list[tuple[int, int, str, str, int]]) -> pd.DataFrame:
    """Como ``_events``, mas com ``cat_idx`` — simula o schema real pós
    ``feature_eng`` (a coluna só existe depois desse estágio no pipeline)."""
    df = _events([r[:4] for r in rows])
    df["cat_idx"] = [r[4] for r in rows]
    return df


def test_item_profile_without_category_column_returns_none():
    # _events() (fixture padrão) não passa por feature_eng — sem cat_idx.
    store = _store_with_item12()
    profile = store.item_profile(11)
    assert profile["item_idx"] == 11
    assert profile["cat_idx"] is None
    assert profile["view_count"] == 3
    assert profile["popularity_rank"] == 1  # item 11 é o mais visto


def test_item_profile_reports_category_and_rank():
    df = _events_with_category(
        [
            (1, 10, "view", "2015-05-01", 100),
            (1, 11, "view", "2015-05-02", 200),
            (2, 11, "view", "2015-05-03", 200),
        ]
    )
    store = FeatureStore(df)
    assert store.item_profile(11) == {
        "item_idx": 11,
        "cat_idx": 200,
        "view_count": 2,
        "popularity_rank": 1,
    }
    assert store.item_profile(10)["popularity_rank"] == 2


def test_timeline_orders_chronologically_and_respects_limit():
    store = _store_with_item12()
    events = store.timeline(1, limit=2)
    assert [e["event"] for e in events] == ["view", "view"]
    assert [e["item_idx"] for e in events] == [10, 11]  # os 2 últimos, em ordem
    assert events[-1]["timestamp"] > events[0]["timestamp"]


def test_timeline_unknown_user_returns_empty():
    store = _store_with_item12()
    assert store.timeline(999) == []


def test_explain_scored_items_include_item_profile():
    result = _service().explain(user_id=3, top_k=5)
    row = result["scored"][0]
    assert "cat_idx" in row
    assert "popularity_rank" in row


def test_user_timeline_endpoint_happy_path():
    client = _client_with_service()
    resp = client.get("/demo/user-timeline", params={"user_id": 1, "limit": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["known"] is True
    assert len(body["events"]) == 3  # user 1 viu 10, 10, 11


def test_user_timeline_endpoint_unknown_user():
    client = _client_with_service()
    resp = client.get("/demo/user-timeline", params={"user_id": 999})
    assert resp.status_code == 200
    body = resp.json()
    assert body["known"] is False
    assert body["events"] == []


def test_user_timeline_without_service_returns_503():
    client = TestClient(api_module.app)
    api_module.state["service"] = None
    assert client.get("/demo/user-timeline", params={"user_id": 1}).status_code == 503
