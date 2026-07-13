"""Etapa de promoção do pipeline DVC: registra e promove o melhor modelo.

Executada após `evaluate`. Busca o melhor run por `val_auc`, registra o
artefato no MLflow Model Registry (Staging) e promove para Production.
Persiste os dados da promoção em models/promoted_model.json para o DVC rastrear.
"""

import json
import logging
from pathlib import Path

import mlflow
import yaml

from src.config.settings import settings
from src.utils.mlflow_tracking import (
    find_best_model_run,
    promote_model,
    register_model,
)

logger = logging.getLogger(__name__)

PARAMS_PATH = Path("params.yaml")
RECORD_PATH = Path("models/promoted_model.json")


def _write_record(record: dict) -> None:
    """Persiste os dados da promoção em models/promoted_model.json (DVC out)."""
    RECORD_PATH.parent.mkdir(parents=True, exist_ok=True)
    RECORD_PATH.write_text(json.dumps(record, indent=2))
    logger.info("Registro de promoção salvo em %s", RECORD_PATH)


def _promote(reg: dict, experiment_name: str) -> dict:
    """Encontra o melhor run, registra em Staging e promove; retorna o registro."""
    best = find_best_model_run(experiment_name, reg["metric"], reg["ascending"])
    version = register_model(best.info.run_id, reg["model_name"], stage="Staging")
    promote_model(reg["model_name"], version, stage=reg["stage"])
    return {
        "model_name": reg["model_name"],
        "version": version,
        "stage": reg["stage"],
        "run_id": best.info.run_id,
        "metric": reg["metric"],
        "value": best.data.metrics.get(reg["metric"]),
    }


def run() -> None:
    """Executa a etapa de promoção do modelo."""
    params = yaml.safe_load(open(PARAMS_PATH))
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    record = _promote(params["registry"], params["mlflow"]["experiment_name"])
    _write_record(record)
    logger.info("Promoção concluída: %s", record)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
