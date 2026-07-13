"""MLflow helpers: experiment setup, best-run selection and model registration."""

import logging

import mlflow
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

from src.config.settings import settings

logger = logging.getLogger(__name__)

MODEL_ARTIFACT_PATH = "model/model.pkl"


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


def _has_model_artifact(client: MlflowClient, run_id: str) -> bool:
    """Return True if the run logged a model artifact at ``model/model.pkl``."""
    artifacts = client.list_artifacts(run_id, "model")
    return any(a.path == MODEL_ARTIFACT_PATH for a in artifacts)


def find_best_model_run(
    experiment_name: str, metric: str = "val_auc", ascending: bool = False
) -> Run:
    """Return the best run (by ``metric``) that has a registrable model artifact.

    Args:
        experiment_name: Experiment to search within.
        metric: Metric used to rank runs (e.g. ``val_auc``).
        ascending: If True, smaller is better; otherwise larger is better.

    Returns:
        The best matching MLflow Run.
    """
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experimento '{experiment_name}' não encontrado.")
    order = "ASC" if ascending else "DESC"
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"metrics.{metric} > -1e30",
        order_by=[f"metrics.{metric} {order}"],
        max_results=10,
    )
    for run in runs:
        if _has_model_artifact(client, run.info.run_id):
            return run
    raise ValueError(f"Nenhum run com artefato de modelo e métrica '{metric}'.")


def register_model(run_id: str, model_name: str, stage: str = "Staging") -> str:
    """Register a run's model artifact in the MLflow Model Registry.

    Args:
        run_id: MLflow run ID containing the model artifact.
        model_name: Registered model name.
        stage: Target stage — 'Staging' or 'Production'.

    Returns:
        The created model version number.
    """
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    model_uri = f"runs:/{run_id}/{MODEL_ARTIFACT_PATH}"

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
    return mv.version


def promote_model(model_name: str, version: str, stage: str = "Production") -> None:
    """Transition an existing model version to a new stage (e.g. Production).

    Args:
        model_name: Registered model name.
        version: Model version to transition.
        stage: Target stage — typically 'Production'.
    """
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
    client.transition_model_version_stage(
        name=model_name, version=version, stage=stage, archive_existing_versions=True
    )
    logger.info("Model '%s' v%s → %s", model_name, version, stage)
