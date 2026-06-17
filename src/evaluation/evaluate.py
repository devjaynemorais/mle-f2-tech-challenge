"""Etapa 4 do pipeline DVC: avalia o modelo no conjunto de teste.

Calcula as métricas obrigatórias do TC2 (≥ 4 métricas) e salva
em metrics/eval_metrics.json para o DVC rastrear.

TODO: implementar load_model() e load_test_data() após definir
como o modelo é salvo na etapa de treino.
"""

import json
import logging
import pickle
from pathlib import Path

import mlflow
import numpy as np
import yaml
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config.settings import settings
from src.models.base import RecommenderBase

logger = logging.getLogger(__name__)

PARAMS_PATH = Path("params.yaml")
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models/artifacts")
METRICS_DIR = Path("metrics")


def load_model() -> RecommenderBase:
    """Carrega o modelo mais recente salvo em models/artifacts/.

    TODO: ajustar conforme a convenção de salvamento definida no trainer.
    """
    run_dirs = sorted(
        MODELS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
    )
    for d in run_dirs:
        model_file = d / "model.pkl"
        if model_file.exists():
            with open(model_file, "rb") as f:
                return pickle.load(f)
    raise FileNotFoundError(f"Nenhum model.pkl encontrado em {MODELS_DIR}")


def load_test_data() -> tuple:
    """Carrega os dados de teste de data/processed/.

    TODO: implementar conforme o formato do dataset escolhido.
    Deve retornar (X_test, y_test).
    """
    raise NotImplementedError("Implemente load_test_data() após escolher o dataset.")


def compute_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> dict:
    """Calcula as métricas de avaliação obrigatórias do TC2.

    São ≥ 4 métricas conforme o requisito: ROC-AUC, Average Precision,
    F1, Precisão e Recall.

    Args:
        y_true: Rótulos verdadeiros.
        y_proba: Probabilidades preditas da classe positiva.
        threshold: Limiar de decisão para converter probabilidade em 0/1.

    Returns:
        Dicionário com nome da métrica → valor.
    """
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "average_precision": float(average_precision_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def _log_and_save(metrics: dict, mlflow_p: dict) -> None:
    """Loga métricas no MLflow e persiste o JSON para o DVC rastrear."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_p["experiment_name"])
    with mlflow.start_run(run_name="evaluate"):
        mlflow.log_metrics(metrics)
    out = METRICS_DIR / "eval_metrics.json"
    json.dump(metrics, open(out, "w"), indent=2)
    logger.info("Métricas salvas em %s", out)


def run() -> None:
    """Executa a etapa de avaliação."""
    params = yaml.safe_load(open(PARAMS_PATH))
    model = load_model()
    X_test, y_test = load_test_data()
    metrics = compute_metrics(y_test, model.predict_proba(X_test))
    logger.info("Métricas de teste: %s", metrics)
    _log_and_save(metrics, params["mlflow"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
