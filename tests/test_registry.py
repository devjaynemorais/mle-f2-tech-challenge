"""Testes da lógica de registro/promoção no MLflow Model Registry.

Usam um backend SQLite temporário e artefatos em tmp_path, sem servidor
MLflow nem dados reais — passam em qualquer clone limpo do repositório.
"""

import json
import pickle
from pathlib import Path

import mlflow
import numpy as np
import pytest
import yaml
from mlflow.tracking import MlflowClient

from src.config import settings as settings_module
from src.models import registry
from src.models.baselines import LogisticRecommender
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
        model = LogisticRecommender(random_state=0)
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


def test_setup_experiment_returns_experiment_id(tracking: Path) -> None:
    """setup_experiment cria/reaproveita o experimento e devolve o id."""
    experiment_id = mlflow_tracking.setup_experiment("test_exp")
    client = MlflowClient(tracking_uri=settings_module.settings.mlflow_tracking_uri)
    experiment = client.get_experiment_by_name("test_exp")
    assert experiment_id == experiment.experiment_id


def test_find_best_model_run_sem_artefato_de_modelo(tracking: Path) -> None:
    """Runs sem artefato model/model.pkl não contam — deve lançar ValueError."""
    with mlflow.start_run():
        mlflow.log_metric("val_auc", 0.99)  # sem log_artifact: sem modelo
    with pytest.raises(ValueError, match="Nenhum run"):
        mlflow_tracking.find_best_model_run("test_exp", "val_auc")


def test_register_model_reusa_modelo_ja_registrado(tracking: Path) -> None:
    """Uma 2a chamada com o mesmo nome não deve falhar (modelo já existe)."""
    run_a = _log_run(tracking, 0.70)
    run_b = _log_run(tracking, 0.90)
    version_a = mlflow_tracking.register_model(run_a, "modelo_repetido")
    version_b = mlflow_tracking.register_model(run_b, "modelo_repetido")
    assert version_a != version_b


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


# ─── src/models/registry.py — etapa "promote" do pipeline DVC ─────────────


def test_write_record_persists_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    record_path = tmp_path / "promoted_model.json"
    monkeypatch.setattr(registry, "RECORD_PATH", record_path)
    registry._write_record({"model_name": "x", "version": "1"})
    assert json.loads(record_path.read_text()) == {"model_name": "x", "version": "1"}


def test_promote_finds_registers_and_promotes(tracking: Path) -> None:
    run_id = _log_run(tracking, 0.85)
    reg = {
        "metric": "val_auc",
        "ascending": False,
        "model_name": "promo_test_model",
        "stage": "Production",
    }
    record = registry._promote(reg, "test_exp")
    assert record["run_id"] == run_id
    assert record["model_name"] == "promo_test_model"
    assert record["stage"] == "Production"
    assert record["value"] == pytest.approx(0.85)


def test_run_reads_params_yaml_and_writes_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tracking: Path
) -> None:
    run_id = _log_run(tracking, 0.77)
    params = {
        "registry": {
            "metric": "val_auc",
            "ascending": False,
            "model_name": "run_test_model",
            "stage": "Production",
        },
        "mlflow": {"experiment_name": "test_exp"},
    }
    params_path = tmp_path / "params.yaml"
    params_path.write_text(yaml.dump(params))
    record_path = tmp_path / "promoted_model.json"
    monkeypatch.setattr(registry, "PARAMS_PATH", params_path)
    monkeypatch.setattr(registry, "RECORD_PATH", record_path)

    registry.run()

    saved = json.loads(record_path.read_text())
    assert saved["run_id"] == run_id
    assert saved["model_name"] == "run_test_model"
