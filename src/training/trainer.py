"""Etapa 3 do pipeline DVC: treina o modelo e loga os resultados no MLflow.

Fluxo:
  1. Carrega os dados de data/processed/
  2. Cria o modelo via ModelFactory (definido em params.yaml)
  3. Treina o modelo
  4. Loga parâmetros e métricas no MLflow
  5. Salva o artefato do modelo em models/artifacts/

TODO: implementar load_data() e save_model() após escolher o dataset.
"""

import logging
import pickle
from pathlib import Path

import mlflow
import yaml

from src.config.settings import settings
from src.models import ModelFactory
from src.models.base import RecommenderBase

logger = logging.getLogger(__name__)

PARAMS_PATH = Path("params.yaml")
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models/artifacts")
METRICS_DIR = Path("metrics")


def load_data() -> tuple:
    """Carrega os dados de treino e validação de data/processed/.

    TODO: implementar conforme o formato do dataset escolhido.
    Deve retornar (X_train, y_train, X_val, y_val).
    """
    # TODO: carregar os splits e retornar arrays de features e rótulos
    raise NotImplementedError("Implemente load_data() após escolher o dataset.")


def save_model(model: RecommenderBase, run_id: str) -> Path:
    """Salva o modelo treinado em disco.

    Args:
        model: Instância do modelo já treinado.
        run_id: ID do run do MLflow, usado para nomear o diretório.

    Returns:
        Caminho onde o modelo foi salvo.
    """
    dest = MODELS_DIR / run_id
    dest.mkdir(parents=True, exist_ok=True)
    with open(dest / "model.pkl", "wb") as f:
        pickle.dump(model, f)
    logger.info("Modelo salvo em %s", dest)
    return dest


def _create_model(train_p: dict, input_dim: int) -> RecommenderBase:
    """Instancia o modelo via ModelFactory com os parâmetros do params.yaml."""
    return ModelFactory.create(
        train_p["model_type"],
        input_dim=input_dim,
        epochs=train_p["epochs"],
        batch_size=train_p["batch_size"],
        lr=train_p["learning_rate"],
        patience=train_p["early_stopping_patience"],
        random_state=train_p["random_state"],
    )


def _train_and_log(train_p: dict, mlflow_p: dict) -> None:
    """Treina o modelo e registra o artefato no MLflow."""
    X_train, y_train, _X_val, _y_val = load_data()
    run_name = mlflow_p.get("run_name", train_p["model_type"])
    with mlflow.start_run(run_name=run_name) as active_run:
        mlflow.log_params(train_p)
        model = _create_model(train_p, X_train.shape[1])
        logger.info("Treinando modelo: %s", train_p["model_type"])
        model.fit(X_train, y_train)
        artifact_dir = save_model(model, active_run.info.run_id)
        mlflow.log_artifact(str(artifact_dir))
        logger.info("Run MLflow: %s", active_run.info.run_id)


def run() -> None:
    """Executa a etapa de treinamento."""
    params = yaml.safe_load(open(PARAMS_PATH))
    train_p = params["train"]
    mlflow_p = params["mlflow"]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_p["experiment_name"])
    _train_and_log(train_p, mlflow_p)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
