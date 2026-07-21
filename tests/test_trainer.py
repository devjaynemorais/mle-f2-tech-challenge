"""Testes da etapa DVC `train` (src/training/trainer.py).

Cobre as funções puras isoladamente e um fluxo `run()` de ponta a ponta com
dados sintéticos (sem dataset real, sem servidor MLflow — backend SQLite
temporário, mesmo padrão de tests/test_registry.py e tests/test_evaluate.py).
"""

import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest
import yaml

from src.config import settings as settings_module
from src.data.feature_contract import FEATURE_COLS
from src.data.feature_engineering import build_causal_features, chronological_split
from src.models.baselines import LogisticRecommender
from src.models.mlp import NCFRecommender
from src.training import trainer


def _synthetic_events(
    n: int = 150, n_users: int = 15, n_items: int = 20
) -> pd.DataFrame:
    rng = np.random.default_rng(21)
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
    df = build_causal_features(raw)
    df["cat_idx"] = rng.integers(-1, 4, len(df))  # -1 = sem categoria (unknown)
    return df


def _train_val() -> tuple[pd.DataFrame, pd.DataFrame]:
    events = _synthetic_events()
    train, val, _test = chronological_split(events, val_size=0.2, test_size=0.2)
    return train, val


# ─── Funções puras ──────────────────────────────────────────────────────────


def test_load_splits_reads_train_and_val(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(trainer, "PROCESSED_DIR", tmp_path)
    pd.DataFrame({"a": [1]}).to_parquet(tmp_path / "train.parquet", index=False)
    pd.DataFrame({"a": [1, 2]}).to_parquet(tmp_path / "val.parquet", index=False)
    train, val = trainer.load_splits()
    assert len(train) == 1
    assert len(val) == 2


def test_build_training_arrays_without_multitask():
    train, val = _train_val()
    labeling_p = {"num_negatives": 4, "popularity_alpha": 0.0, "seed": 1}
    X_train, y_train, X_val, y_val, y_view = trainer.build_training_arrays(
        train, val, labeling_p
    )
    assert X_train.shape[1] == len(FEATURE_COLS)
    assert y_train.shape[0] == X_train.shape[0]
    assert X_val.shape[1] == len(FEATURE_COLS)
    assert y_view is None


def test_build_training_arrays_with_multitask():
    train, val = _train_val()
    labeling_p = {
        "num_negatives": 4,
        "popularity_alpha": 0.0,
        "seed": 1,
        "view_sample_ratio": 1.0,
    }
    *_, y_view = trainer.build_training_arrays(train, val, labeling_p)
    assert y_view is not None


def test_vocab_kwargs_derives_vocab_and_categories():
    train, val = _train_val()
    kwargs = trainer._vocab_kwargs(train, val)
    both = pd.concat([train, val], ignore_index=True)
    assert kwargs["n_users"] == int(both["user_idx"].max()) + 1
    assert kwargs["n_items"] == int(both["item_idx"].max()) + 1
    assert kwargs["known_users"].dtype == bool
    assert kwargs["n_categories"] > 0
    assert kwargs["item_categories"].shape == (kwargs["n_items"] + 1,)


def test_vocab_kwargs_no_categories_at_all():
    train, val = _train_val()
    train = train.copy()
    val = val.copy()
    train["cat_idx"] = -1
    val["cat_idx"] = -1
    kwargs = trainer._vocab_kwargs(train, val)
    assert kwargs["n_categories"] == 0


def test_create_model_ncf():
    train, val = _train_val()
    train_p = {
        "model_type": "ncf",
        "embedding_dim": 4,
        "cat_embedding_dim": 2,
        "hidden_dims": [8],
        "dropout": 0.1,
        "unknown_dropout": 0.0,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "epochs": 1,
        "batch_size": 32,
        "early_stopping_patience": 1,
        "random_state": 0,
    }
    model = trainer._create_model(train_p, train, val)
    assert isinstance(model, NCFRecommender)


def test_create_model_logistic():
    train, val = _train_val()
    model = trainer._create_model(
        {"model_type": "logistic", "random_state": 0}, train, val
    )
    assert isinstance(model, LogisticRecommender)


def test_create_model_dummy():
    train, val = _train_val()
    model = trainer._create_model({"model_type": "dummy"}, train, val)
    from src.models.baselines import DummyRecommender

    assert isinstance(model, DummyRecommender)


def test_val_ranking_metric_returns_zero_without_positives():
    train, val = _train_val()
    val_no_positives = val.copy()
    val_no_positives["event"] = "view"  # sem addtocart/transaction
    labeling_p = {"num_negatives": 4, "popularity_alpha": 0.0, "seed": 1}
    X_train, y_train, *_ = trainer.build_training_arrays(train, val, labeling_p)
    model = LogisticRecommender(random_state=0).fit(X_train, y_train)
    eval_p = {"k_values": [2, 5], "num_candidate_negatives": 5, "seed": 1}
    metric = trainer.val_ranking_metric(model, train, val_no_positives, eval_p)
    assert metric == 0.0


def test_val_ranking_metric_normal_case():
    train, val = _train_val()
    labeling_p = {"num_negatives": 4, "popularity_alpha": 0.0, "seed": 1}
    X_train, y_train, X_val, y_val, _ = trainer.build_training_arrays(
        train, val, labeling_p
    )
    model = LogisticRecommender(random_state=0).fit(X_train, y_train)
    eval_p = {"k_values": [2, 5], "num_candidate_negatives": 5, "seed": 1}
    metric = trainer.val_ranking_metric(model, train, val, eval_p)
    assert 0.0 <= metric <= 1.0


def test_save_model_writes_pickle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(trainer, "MODELS_DIR", tmp_path)
    model = LogisticRecommender(random_state=0)
    dest = trainer.save_model(model, "run123")
    assert (dest / "model.pkl").exists()
    assert dest == tmp_path / "run123"


def test_save_train_metrics_writes_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(trainer, "METRICS_DIR", tmp_path)
    trainer._save_train_metrics({"val_auc": 0.9})
    saved = json.loads((tmp_path / "train_metrics.json").read_text())
    assert saved == {"val_auc": 0.9}


def test_fit_ncf_passes_validation_and_view(monkeypatch: pytest.MonkeyPatch):
    train, val = _train_val()
    calls = {}

    class _FakeNCF(NCFRecommender):
        def fit(self, X, y, X_val=None, y_val=None, y_view=None):
            calls["X_val"] = X_val
            calls["y_view"] = y_view
            return self

    model = _FakeNCF(n_users=15, n_items=20)
    trainer._fit(
        model,
        np.zeros((2, 10)),
        np.zeros(2),
        np.zeros((2, 10)),
        np.zeros(2),
        np.ones(2),
    )
    assert calls["X_val"] is not None
    assert calls["y_view"] is not None


def test_fit_non_ncf_ignores_validation():
    calls = {}

    class _FakeBaseline(LogisticRecommender):
        def fit(self, X, y):
            calls["called"] = True
            return self

    model = _FakeBaseline()
    trainer._fit(model, np.zeros((2, 10)), np.zeros(2), np.zeros((2, 10)), np.zeros(2))
    assert calls["called"] is True


# ─── run() de ponta a ponta ─────────────────────────────────────────────────


@pytest.fixture
def trainer_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    train, val = _train_val()
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    train.to_parquet(processed_dir / "train.parquet", index=False)
    val.to_parquet(processed_dir / "val.parquet", index=False)

    models_dir = tmp_path / "artifacts"
    metrics_dir = tmp_path / "metrics"

    params = {
        "labeling": {"num_negatives": 4, "popularity_alpha": 0.0, "seed": 1},
        "eval": {"k_values": [2, 5], "num_candidate_negatives": 5, "seed": 1},
        "mlflow": {"experiment_name": "train_test_exp", "run_name": "train"},
        "train": {
            "model_type": "logistic",
            "random_state": 0,
        },
    }
    params_path = tmp_path / "params.yaml"
    params_path.write_text(yaml.dump(params))

    uri = f"sqlite:///{tmp_path.as_posix()}/mlflow.db"
    monkeypatch.setattr(settings_module.settings, "mlflow_tracking_uri", uri)
    mlflow.set_tracking_uri(uri)

    monkeypatch.setattr(trainer, "PARAMS_PATH", params_path)
    monkeypatch.setattr(trainer, "PROCESSED_DIR", processed_dir)
    monkeypatch.setattr(trainer, "MODELS_DIR", models_dir)
    monkeypatch.setattr(trainer, "METRICS_DIR", metrics_dir)
    return {"models_dir": models_dir, "metrics_dir": metrics_dir}


def test_run_end_to_end_trains_and_saves(trainer_env: dict):
    trainer.run()
    metrics_dir = trainer_env["metrics_dir"]
    metrics = json.loads((metrics_dir / "train_metrics.json").read_text())
    assert "val_auc" in metrics
    assert "val_ndcg_at_20" in metrics
    saved_models = list(trainer_env["models_dir"].iterdir())
    assert len(saved_models) == 1
    assert (saved_models[0] / "model.pkl").exists()
