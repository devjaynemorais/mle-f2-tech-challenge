"""Testes da lógica de registro/promoção no MLflow Model Registry.

Usam um backend SQLite temporário e artefatos em tmp_path, sem servidor
MLflow nem dados reais — passam em qualquer clone limpo do repositório.
"""

import pickle
from pathlib import Path

import mlflow
import numpy as np
import pytest
from mlflow.tracking import MlflowClient

from src.config import settings as settings_module
from src.models.mlp import MLPRecommender
from src.utils import mlflow_tracking


@pytest.fixture
def tracking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Configura um tracking URI SQLite isolado com artefatos em tmp_path."""
    uri = f"sqlite:///{tmp_path.as_posix()}/mlflow.db"
    monkeypatch.setattr(settings_module.settings, "mlflow_tracking_uri", uri)
    mlflow.set_tracking_uri(uri)
    artifact_loc = (tmp_path / "artifacts").as_uri()
    mlflow.create_experiment("test_exp", artifact_location=artifact_loc)
    mlflow.set_experiment("test_exp")
    return tmp_path


def _log_run(tmp_path: Path, val_auc: float) -> str:
    """Cria um run com métrica val_auc e artefato model/model.pkl."""
    with mlflow.start_run() as run:
        mlflow.log_metric("val_auc", val_auc)
        model = MLPRecommender(input_dim=4, hidden_dims=[8], epochs=2)
        rng = np.random.default_rng(0)
        model.fit(
            rng.random((16, 4)).astype("float32"),
            (rng.random(16) > 0.5).astype("float32"),
        )
        path = tmp_path / run.info.run_id / "model.pkl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(model, f)
        mlflow.log_artifact(str(path), artifact_path="model")
        return run.info.run_id


def test_find_best_model_run_escolhe_maior_auc(tracking: Path) -> None:
    """find_best_model_run deve retornar o run com maior val_auc."""
    _log_run(tracking, 0.70)
    best = _log_run(tracking, 0.90)
    _log_run(tracking, 0.60)
    run = mlflow_tracking.find_best_model_run("test_exp", "val_auc")
    assert run.info.run_id == best


def test_find_best_model_run_experimento_inexistente(tracking: Path) -> None:
    """Deve lançar ValueError quando o experimento não existe."""
    with pytest.raises(ValueError, match="não encontrado"):
        mlflow_tracking.find_best_model_run("nao_existe", "val_auc")


def test_register_e_promote_para_production(tracking: Path) -> None:
    """register_model + promote_model devem colocar a versão em Production."""
    run_id = _log_run(tracking, 0.80)
    version = mlflow_tracking.register_model(run_id, "test_model", stage="Staging")
    mlflow_tracking.promote_model("test_model", version, stage="Production")
    client = MlflowClient(tracking_uri=settings_module.settings.mlflow_tracking_uri)
    prod = client.get_latest_versions("test_model", stages=["Production"])[0]
    assert prod.version == version
    assert prod.run_id == run_id
