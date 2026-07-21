"""Testes da etapa DVC `evaluate` (src/evaluation/evaluate.py).

Cobre as funções puras isoladamente e um fluxo `run()` de ponta a ponta com
dados sintéticos (sem dataset real, sem servidor MLflow — backend SQLite
temporário, mesmo padrão de tests/test_registry.py).
"""

import json
import pickle
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
import yaml

from src.config import settings as settings_module
from src.data.feature_contract import FEATURE_COLS
from src.data.feature_engineering import build_causal_features, chronological_split
from src.evaluation import evaluate
from src.models.mlp import NCFRecommender


class _StubModel:
    """Modelo picklável mínimo — classes locais não são picklable."""

    def predict_proba(self, X):
        return np.zeros(len(X))


def _synthetic_events(
    n: int = 150, n_users: int = 15, n_items: int = 20
) -> pd.DataFrame:
    rng = np.random.default_rng(11)
    raw = pd.DataFrame(
        {
            "timestamp": pd.to_datetime("2015-05-01")
            + pd.to_timedelta(np.sort(rng.integers(0, 3600 * 24 * 20, n)), unit="s"),
            "user_idx": rng.integers(0, n_users, n),
            "item_idx": rng.integers(0, n_items, n),
            "event": rng.choice(
                ["view", "addtocart", "transaction"], size=n, p=[0.7, 0.2, 0.1]
            ),
        }
    )
    return build_causal_features(raw)


# ─── Funções puras ──────────────────────────────────────────────────────────


def test_compute_metrics_returns_expected_keys():
    y_true = np.array([1, 0, 1, 0], dtype="float32")
    y_proba = np.array([0.9, 0.2, 0.8, 0.4], dtype="float32")
    metrics = evaluate.compute_metrics(y_true, y_proba)
    assert set(metrics) == {
        "roc_auc",
        "average_precision",
        "f1",
        "precision",
        "recall",
    }
    assert metrics["roc_auc"] == pytest.approx(1.0)


def test_positives_by_user_dedupes_and_caps():
    test = pd.DataFrame(
        {
            "user_idx": [1, 1, 1, 2],
            "item_idx": [10, 10, 11, 20],
            "event": ["transaction", "transaction", "addtocart", "view"],
            "timestamp": pd.to_datetime(
                ["2015-05-01", "2015-05-02", "2015-05-03", "2015-05-01"]
            ),
        }
    )
    from src.data.labeling import STRONG_RELEVANCE_EVENTS

    result = evaluate.positives_by_user(test, STRONG_RELEVANCE_EVENTS)
    assert set(result[1].tolist()) == {10, 11}  # user 2 não tem evento forte
    assert 2 not in result

    capped = evaluate.positives_by_user(test, STRONG_RELEVANCE_EVENTS, max_per_user=1)
    assert len(capped[1]) == 1


def test_context_ts_by_user_is_first_event_per_user():
    test = pd.DataFrame(
        {
            "user_idx": [1, 1, 2],
            "timestamp": pd.to_datetime(["2015-05-02", "2015-05-01", "2015-05-05"]),
        }
    )
    ctx = evaluate._context_ts_by_user(test)
    assert ctx[1] == pd.Timestamp("2015-05-01")
    assert ctx[2] == pd.Timestamp("2015-05-05")


def test_weighted_overall_zero_users_returns_empty_dict():
    assert (
        evaluate._weighted_overall({"warm": {}, "cold": {}}, {"warm": 0, "cold": 0})
        == {}
    )


def test_weighted_overall_weights_by_segment_size():
    segments = {"warm": {"ndcg": 1.0}, "cold": {"ndcg": 0.0}}
    counts = {"warm": 3, "cold": 1}
    result = evaluate._weighted_overall(segments, counts)
    assert result["ndcg"] == pytest.approx(0.75)


def test_flatten_nested_metrics_dict():
    nested = {"classification": {"roc_auc": 0.9}, "ranking": {"strong": {"ndcg": 1}}}
    flat = evaluate._flatten(nested)
    assert flat == {
        "classification_roc_auc": 0.9,
        "ranking_strong_ndcg": 1.0,
    }


def test_load_splits_reads_three_parquet_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(evaluate, "PROCESSED_DIR", tmp_path)
    for name in ("train", "val", "test"):
        pd.DataFrame({"a": [1]}).to_parquet(tmp_path / f"{name}.parquet", index=False)
    train, val, test = evaluate.load_splits()
    assert len(train) == len(val) == len(test) == 1


def test_load_model_picks_most_recent_with_pickle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(evaluate, "MODELS_DIR", tmp_path)

    older = tmp_path / "run_old"
    older.mkdir()
    with open(older / "model.pkl", "wb") as f:
        pickle.dump(_StubModel(), f)

    empty_dir = tmp_path / "run_no_model"
    empty_dir.mkdir()  # sem model.pkl — deve ser ignorado

    newer = tmp_path / "run_new"
    newer.mkdir()
    with open(newer / "model.pkl", "wb") as f:
        pickle.dump(_StubModel(), f)
    import os
    import time

    time.sleep(0.01)
    os.utime(newer / "model.pkl", None)  # garante mtime mais recente

    model = evaluate.load_model()
    assert isinstance(model, _StubModel)


def test_load_model_raises_when_no_model_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(evaluate, "MODELS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="Nenhum model.pkl"):
        evaluate.load_model()


def test_save_plots_writes_roc_and_pr_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(evaluate, "METRICS_DIR", tmp_path)
    y_true = np.array([0, 0, 1, 1], dtype="float32")
    y_proba = np.array([0.1, 0.4, 0.35, 0.8], dtype="float32")
    evaluate._save_plots(y_true, y_proba)
    roc = json.loads((tmp_path / "plots" / "roc_curve.json").read_text())
    pr = json.loads((tmp_path / "plots" / "pr_curve.json").read_text())
    assert all({"fpr", "tpr"} == set(point) for point in roc)
    assert all({"precision", "recall"} == set(point) for point in pr)


# ─── ranking_metrics / _eval_relevance (com modelo NCF real) ────────────────


def test_ranking_metrics_returns_strong_and_broad_segments():
    events = _synthetic_events()
    train, val, test = chronological_split(events, val_size=0.2, test_size=0.2)
    history = pd.concat([train, val], ignore_index=True)

    from src.data.labeling import build_labeled_dataset

    labeled = build_labeled_dataset(train, seed=1, view_sample_ratio=1.0)
    model = NCFRecommender(
        n_users=15, n_items=20, embedding_dim=4, hidden_dims=[8], epochs=2
    )
    model.fit(
        labeled[FEATURE_COLS].to_numpy(dtype="float32"),
        labeled["label"].to_numpy(dtype="float32"),
        y_view=labeled["view_label"].to_numpy(dtype="float32"),
    )

    eval_p = {
        "k_values": [2, 5],
        "num_candidate_negatives": 5,
        "max_broad_positives": 10,
        "seed": 1,
    }
    result = evaluate.ranking_metrics(model, train, history, test, eval_p)
    assert set(result) == {"strong", "broad"}
    for relevance in result.values():
        assert "model" in relevance
        assert "popularity" in relevance
        assert "n_users" in relevance


# ─── run() de ponta a ponta ─────────────────────────────────────────────────


@pytest.fixture
def evaluate_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """Monta processed/, models/artifacts/, params.yaml e MLflow SQLite tmp."""
    events = _synthetic_events()
    train, val, test = chronological_split(events, val_size=0.2, test_size=0.2)

    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    train.to_parquet(processed_dir / "train.parquet", index=False)
    val.to_parquet(processed_dir / "val.parquet", index=False)
    test.to_parquet(processed_dir / "test.parquet", index=False)

    from src.data.labeling import build_labeled_dataset

    labeled = build_labeled_dataset(train, seed=1)
    model = NCFRecommender(
        n_users=15, n_items=20, embedding_dim=4, hidden_dims=[8], epochs=2
    )
    model.fit(
        labeled[FEATURE_COLS].to_numpy(dtype="float32"),
        labeled["label"].to_numpy(dtype="float32"),
    )
    models_dir = tmp_path / "artifacts"
    run_dir = models_dir / "run1"
    run_dir.mkdir(parents=True)
    with open(run_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    metrics_dir = tmp_path / "metrics"
    params = {
        "labeling": {"num_negatives": 4, "popularity_alpha": 0.0, "seed": 1},
        "eval": {
            "k_values": [2, 5],
            "num_candidate_negatives": 5,
            "max_broad_positives": 10,
            "seed": 1,
        },
        "mlflow": {"experiment_name": "eval_test_exp"},
        "train": {"model_type": "ncf"},
    }
    params_path = tmp_path / "params.yaml"
    params_path.write_text(yaml.dump(params))

    uri = f"sqlite:///{tmp_path.as_posix()}/mlflow.db"
    monkeypatch.setattr(settings_module.settings, "mlflow_tracking_uri", uri)
    mlflow.set_tracking_uri(uri)
    artifact_loc = (tmp_path / "mlartifacts").as_uri()
    mlflow.create_experiment("eval_test_exp", artifact_location=artifact_loc)

    monkeypatch.setattr(evaluate, "PARAMS_PATH", params_path)
    monkeypatch.setattr(evaluate, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(evaluate, "MODELS_DIR", models_dir)
    monkeypatch.setattr(evaluate, "METRICS_DIR", metrics_dir)
    return {"metrics_dir": metrics_dir}


def test_run_end_to_end_writes_metrics_and_plots(evaluate_env: dict):
    evaluate.run()
    metrics_dir = evaluate_env["metrics_dir"]
    metrics = json.loads((metrics_dir / "eval_metrics.json").read_text())
    assert "classification" in metrics
    assert "ranking" in metrics
    assert (metrics_dir / "plots" / "roc_curve.json").exists()
    assert (metrics_dir / "plots" / "pr_curve.json").exists()
