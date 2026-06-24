"""MLflow helpers: experiment setup and model registration."""

import logging

import mlflow
from mlflow.tracking import MlflowClient

from src.config.settings import settings

logger = logging.getLogger(__name__)


def setup_experiment(experiment_name: str | None = None) -> str:
    """Configure MLflow tracking and return experiment ID.

    Args:
        experiment_name: Override for settings.mlflow_experiment_name.

    Returns:
        MLflow experiment ID.
    """
    name = experiment_name or settings.mlflow_experiment_name
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    experiment = mlflow.set_experiment(name)
    logger.info("MLflow experiment: %s (id=%s)", name, experiment.experiment_id)
    return experiment.experiment_id


def register_model(run_id: str, model_name: str, stage: str = "Staging") -> None:
    """Register a run's model artifact in the MLflow Model Registry.

    Args:
        run_id: MLflow run ID containing the model artifact.
        model_name: Registered model name.
        stage: Target stage — 'Staging' or 'Production'.
    """
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    model_uri = f"runs:/{run_id}/model"

    try:
        client.create_registered_model(model_name)
        logger.info("Created registered model: %s", model_name)
    except Exception:
        pass  # already exists

    mv = client.create_model_version(name=model_name, source=model_uri, run_id=run_id)
    client.transition_model_version_stage(
        name=model_name, version=mv.version, stage=stage, archive_existing_versions=True
    )
    logger.info("Model '%s' v%s → %s", model_name, mv.version, stage)
