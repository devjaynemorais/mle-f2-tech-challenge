"""Etapa 3 do pipeline DVC: treina o modelo e loga os resultados no MLflow.

Fluxo:
  1. Carrega os splits de data/processed/ e rotula (labeling + negativos)
  2. Cria o modelo via ModelFactory (definido em params.yaml)
  3. Treina com early stopping por métrica de VALIDAÇÃO (FR-010)
  4. Loga val_auc e val_ndcg_at_20 no MLflow (métrica de promoção)
  5. Salva o artefato auto-contido (rede + scaler + metadados) em
     models/artifacts/ (D5)
"""

import json
import logging
import pickle
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from src.config.settings import settings
from src.data.feature_contract import FEATURE_COLS, TARGET_COL, VIEW_TARGET_COL
from src.data.labeling import (
    STRONG_RELEVANCE_EVENTS,
    build_labeled_dataset,
    build_user_seen,
)
from src.evaluation.evaluate import positives_by_user
from src.evaluation.ranking import evaluate_ranking_per_user
from src.evaluation.scorers import HistoryFeatureLookup, make_model_scorer
from src.models import ModelFactory, NCFRecommender
from src.models.base import RecommenderBase

logger = logging.getLogger(__name__)

PARAMS_PATH = Path("params.yaml")
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models/artifacts")
METRICS_DIR = Path("metrics")


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega os eventos de treino e validação de data/processed/.

    Returns:
        Tupla (train_df, val_df) com eventos e features causais.
    """
    train = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / "val.parquet")
    return train, val


def build_training_arrays(
    train_df: pd.DataFrame, val_df: pd.DataFrame, labeling_p: dict
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Rotula os splits (positivos + negativos amostrados) e vira matrizes.

    Args:
        train_df: Eventos de treino.
        val_df: Eventos de validação.
        labeling_p: Bloco ``labeling`` do params.yaml.

    Returns:
        Tupla (X_train, y_train, X_val, y_val, y_view_train) na ordem do
        contrato. ``y_view_train`` é ``None`` se
        ``labeling.view_sample_ratio`` for 0/ausente (multi-task desligado
        — spec 002, FR-001/FR-003).
    """
    kwargs = {
        "num_negatives": labeling_p["num_negatives"],
        "popularity_alpha": labeling_p["popularity_alpha"],
        "seed": labeling_p["seed"],
    }
    view_sample_ratio = labeling_p.get("view_sample_ratio", 0.0)
    labeled_train = build_labeled_dataset(
        train_df, view_sample_ratio=view_sample_ratio, **kwargs
    )
    labeled_val = build_labeled_dataset(val_df, history=train_df, **kwargs)
    X_train = labeled_train[FEATURE_COLS].to_numpy(dtype="float32")
    y_train = labeled_train[TARGET_COL].to_numpy(dtype="float32")
    X_val = labeled_val[FEATURE_COLS].to_numpy(dtype="float32")
    y_val = labeled_val[TARGET_COL].to_numpy(dtype="float32")
    y_view_train = (
        labeled_train[VIEW_TARGET_COL].to_numpy(dtype="float32")
        if view_sample_ratio > 0
        else None
    )
    return X_train, y_train, X_val, y_val, y_view_train


def _vocab_kwargs(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """Deriva vocabulários, máscaras de warm e categorias para o NCF (D4)."""
    both = pd.concat([train_df, val_df], ignore_index=True)
    n_users = int(both["user_idx"].max()) + 1
    n_items = int(both["item_idx"].max()) + 1
    known_users = np.zeros(n_users, dtype=bool)
    known_users[train_df["user_idx"].unique()] = True
    known_items = np.zeros(n_items, dtype=bool)
    known_items[train_df["item_idx"].unique()] = True

    cats = both.loc[both["cat_idx"] >= 0, ["item_idx", "cat_idx"]].drop_duplicates()
    n_categories = int(cats["cat_idx"].max()) + 1 if len(cats) else 0
    item_categories = np.full(n_items + 1, n_categories, dtype="int64")
    item_categories[cats["item_idx"].to_numpy()] = cats["cat_idx"].to_numpy()
    return {
        "n_users": n_users,
        "n_items": n_items,
        "n_categories": n_categories,
        "item_categories": item_categories,
        "known_users": known_users,
        "known_items": known_items,
    }


def _create_model(
    train_p: dict, train_df: pd.DataFrame, val_df: pd.DataFrame
) -> RecommenderBase:
    """Instancia o modelo via ModelFactory com os parâmetros do params.yaml."""
    model_type = train_p["model_type"]
    if model_type == "ncf":
        return ModelFactory.create(
            model_type,
            embedding_dim=train_p["embedding_dim"],
            cat_embedding_dim=train_p["cat_embedding_dim"],
            hidden_dims=list(train_p["hidden_dims"]),
            dropout=train_p["dropout"],
            unknown_dropout=train_p["unknown_dropout"],
            view_loss_weight=train_p.get("view_loss_weight", 0.0),
            lr=train_p["learning_rate"],
            weight_decay=train_p["weight_decay"],
            epochs=train_p["epochs"],
            batch_size=train_p["batch_size"],
            patience=train_p["early_stopping_patience"],
            random_state=train_p["random_state"],
            **_vocab_kwargs(train_df, val_df),
        )
    if model_type == "logistic":
        return ModelFactory.create(model_type, random_state=train_p["random_state"])
    return ModelFactory.create(model_type)


def val_ranking_metric(
    model: RecommenderBase,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    eval_p: dict,
) -> float:
    """NDCG@20 de VALIDAÇÃO — métrica usada na promoção (FR-010, T024).

    Args:
        model: Modelo treinado.
        train_df: Eventos de treino (histórico para features/candidatos).
        val_df: Eventos de validação (positivos fortes).
        eval_p: Bloco ``eval`` do params.yaml.

    Returns:
        NDCG@20 médio sobre os usuários da validação (0.0 se não há positivos).
    """
    positives = positives_by_user(val_df, STRONG_RELEVANCE_EVENTS)
    if not positives:
        logger.warning("Validação sem positivos fortes — val_ndcg_at_20 = 0.0")
        return 0.0
    lookup = HistoryFeatureLookup(train_df)
    context_ts = {
        int(u): ts for u, ts in val_df.groupby("user_idx")["timestamp"].min().items()
    }
    seen = build_user_seen(pd.concat([train_df, val_df], ignore_index=True))
    results = evaluate_ranking_per_user(
        {"model": make_model_scorer(model, lookup, context_ts)},
        positives,
        seen,
        lookup.catalog,
        ks=eval_p["k_values"],
        num_negatives=eval_p["num_candidate_negatives"],
        seed=eval_p["seed"],
    )
    return results["model"].get("ndcg_at_20", 0.0)


def save_model(model: RecommenderBase, run_id: str) -> Path:
    """Salva o modelo treinado (auto-contido: rede + scaler + metadados).

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


def _save_train_metrics(metrics: dict) -> None:
    """Salva métricas de treino em metrics/train_metrics.json."""
    path = METRICS_DIR / "train_metrics.json"
    path.write_text(json.dumps(metrics, indent=2))
    logger.info("Métricas de treino salvas em %s", path)


def _fit(
    model: RecommenderBase,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    y_view_train: np.ndarray | None = None,
) -> None:
    """Treina o modelo; NCF recebe a validação para o early stopping.

    ``y_view_train`` (spec 002) só é consumido pelo NCF e só ativa a loss
    auxiliar de ``view`` — early stopping continua sobre a validação da
    interação forte (FR-005), inalterado.
    """
    if isinstance(model, NCFRecommender):
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val, y_view=y_view_train)
    else:
        model.fit(X_train, y_train)


def _train_and_log(params: dict) -> None:
    """Treina o modelo e registra métricas e artefato no MLflow."""
    train_p, mlflow_p = params["train"], params["mlflow"]
    train_df, val_df = load_splits()
    X_train, y_train, X_val, y_val, y_view_train = build_training_arrays(
        train_df, val_df, params["labeling"]
    )
    run_name = f"{mlflow_p.get('run_name', 'train')}-{train_p['model_type']}"
    with mlflow.start_run(run_name=run_name) as active_run:
        mlflow.log_params(
            {k: v for k, v in train_p.items() if not isinstance(v, (list, dict))}
        )
        model = _create_model(train_p, train_df, val_df)
        logger.info("Treinando modelo: %s", train_p["model_type"])
        _fit(model, X_train, y_train, X_val, y_val, y_view_train)
        val_auc = float(roc_auc_score(y_val, model.predict_proba(X_val)))
        val_ndcg = val_ranking_metric(model, train_df, val_df, params["eval"])
        metrics = {
            "val_auc": val_auc,
            "val_ndcg_at_20": val_ndcg,
            "model_type": train_p["model_type"],
        }
        mlflow.log_metrics({"val_auc": val_auc, "val_ndcg_at_20": val_ndcg})
        _save_train_metrics(metrics)
        artifact_dir = save_model(model, active_run.info.run_id)
        mlflow.log_artifact(str(artifact_dir / "model.pkl"), artifact_path="model")
        logger.info(
            "Run MLflow: %s | val_auc=%.4f | val_ndcg@20=%.4f",
            active_run.info.run_id,
            val_auc,
            val_ndcg,
        )


def run() -> None:
    """Executa a etapa de treinamento."""
    params = yaml.safe_load(open(PARAMS_PATH))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(params["mlflow"]["experiment_name"])
    _train_and_log(params)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
