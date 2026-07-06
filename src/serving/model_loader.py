"""Carrega o modelo de produção para a API de serving.

Estratégia: tenta o MLflow Model Registry (stage Production); se o servidor
estiver inacessível, cai para o artefato local registrado em
models/promoted_model.json.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

from src.config.settings import settings
from src.models.base import RecommenderBase

logger = logging.getLogger(__name__)

_ARTIFACT_PATH = "model/model.pkl"
_RECORD_PATH = Path("models/promoted_model.json")


def load_production_model() -> RecommenderBase:
    """Retorna o modelo de produção (Registry-first, fallback local)."""
    try:
        return _load_from_registry()
    except Exception as exc:  # noqa: BLE001 — fallback intencional
        logger.warning("Registry indisponível (%s); usando artefato local.", exc)
        return _load_from_local(_RECORD_PATH, Path(settings.model_artifacts_path))


def _load_from_registry() -> RecommenderBase:
    """Baixa e desserializa o modelo Production do MLflow Registry."""
    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    versions = client.get_latest_versions(
        settings.production_model_name, stages=[settings.production_stage]
    )
    if not versions:
        raise ValueError("Nenhuma versão em Production no Registry.")
    local = mlflow.artifacts.download_artifacts(
        run_id=versions[0].run_id,
        artifact_path=_ARTIFACT_PATH,
        tracking_uri=settings.mlflow_tracking_uri,
    )
    with open(local, "rb") as f:
        return pickle.load(f)


def _load_from_local(record_path: Path, artifacts_dir: Path) -> RecommenderBase:
    """Desserializa o modelo apontado por promoted_model.json."""
    record = json.loads(record_path.read_text())
    model_file = artifacts_dir / record["run_id"] / "model.pkl"
    with open(model_file, "rb") as f:
        return pickle.load(f)
