"""Etapa 4 do pipeline DVC: avalia o modelo no conjunto de teste.

Duas famílias de métricas (FR-005/005a/006, FR-011):
- Classificação (ROC-AUC, AP, F1, precisão, recall) sobre o dataset de teste
  rotulado (positivos reais + negativos amostrados — mesma tarefa do treino).
- Ranking Top-K por usuário (NDCG/Recall/Precision/HitRate @{10,20}) sob
  relevância FORTE (addtocart/transaction) e AMPLA (inclui view), comparando
  modelo vs. baseline de popularidade no MESMO conjunto de candidatos, e
  segmentando usuários warm (com histórico de treino) vs. cold.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import cast

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config.settings import settings
from src.data.feature_contract import FEATURE_COLS, TARGET_COL
from src.data.labeling import (
    BROAD_RELEVANCE_EVENTS,
    STRONG_RELEVANCE_EVENTS,
    build_labeled_dataset,
    build_user_seen,
)
from src.evaluation.ranking import PairScorer, evaluate_ranking_per_user
from src.evaluation.scorers import (
    HistoryFeatureLookup,
    make_model_scorer,
    make_popularity_scorer,
)
from src.models.base import RecommenderBase

logger = logging.getLogger(__name__)

PARAMS_PATH = Path("params.yaml")
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("models/artifacts")
METRICS_DIR = Path("metrics")


def load_model() -> RecommenderBase:
    """Carrega o modelo mais recente salvo em models/artifacts/."""
    run_dirs = sorted(
        MODELS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
    )
    for d in run_dirs:
        model_file = d / "model.pkl"
        if model_file.exists():
            with open(model_file, "rb") as f:
                return cast(RecommenderBase, pickle.load(f))
    raise FileNotFoundError(f"Nenhum model.pkl encontrado em {MODELS_DIR}")


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carrega os splits de data/processed/.

    Returns:
        Tupla (train, val, test) como DataFrames de eventos com features.
    """
    return (
        pd.read_parquet(PROCESSED_DIR / "train.parquet"),
        pd.read_parquet(PROCESSED_DIR / "val.parquet"),
        pd.read_parquet(PROCESSED_DIR / "test.parquet"),
    )


def compute_metrics(
    y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5
) -> dict:
    """Calcula as métricas de classificação sobre o teste rotulado.

    Args:
        y_true: Rótulos verdadeiros (positivo real / negativo amostrado).
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


def positives_by_user(
    test: pd.DataFrame,
    relevance_events: tuple[str, ...],
    max_per_user: int | None = None,
) -> dict[int, np.ndarray]:
    """Extrai os itens relevantes de teste por usuário (D3).

    Args:
        test: Eventos do split de teste.
        relevance_events: Tipos de evento que contam como relevantes.
        max_per_user: Teto de positivos por usuário (mais recentes primeiro);
            usado na relevância ampla para conter o custo.

    Returns:
        user_idx → array de item_idx únicos relevantes.
    """
    rel = test.loc[
        test["event"].isin(relevance_events), ["user_idx", "item_idx", "timestamp"]
    ]
    rel = rel.sort_values("timestamp", kind="stable")
    rel = rel.drop_duplicates(subset=["user_idx", "item_idx"], keep="last")
    if max_per_user is not None:
        rel = rel.groupby("user_idx").tail(max_per_user)
    return {
        int(u): items.to_numpy(dtype="int64")
        for u, items in rel.groupby("user_idx")["item_idx"]
    }


def _context_ts_by_user(test: pd.DataFrame) -> dict[int, pd.Timestamp]:
    """Timestamp de pontuação por usuário: seu primeiro evento de teste."""
    return {int(u): ts for u, ts in test.groupby("user_idx")["timestamp"].min().items()}


def _weighted_overall(
    segments: dict[str, dict[str, float]], counts: dict[str, int]
) -> dict[str, float]:
    """Média geral ponderada pelo nº de usuários de cada segmento."""
    total = sum(counts[s] for s in segments if counts[s] > 0)
    if total == 0:
        return {}
    keys = next(iter([m for m in segments.values() if m]), {})
    return {
        key: sum(segments[s].get(key, 0.0) * counts[s] for s in segments) / total
        for key in keys
    }


def _eval_relevance(
    scorers: dict[str, PairScorer],
    positives: dict[int, np.ndarray],
    seen: dict[int, set[int]],
    catalog: np.ndarray,
    train_users: set[int],
    eval_p: dict,
) -> dict:
    """Avalia uma definição de relevância, segmentando warm vs. cold (FR-011)."""
    warm = {u: p for u, p in positives.items() if u in train_users}
    cold = {u: p for u, p in positives.items() if u not in train_users}
    counts = {"warm": len(warm), "cold": len(cold)}
    results: dict[str, dict] = {name: {} for name in scorers}
    for seg_name, seg_pos in (("warm", warm), ("cold", cold)):
        seg_res = evaluate_ranking_per_user(
            scorers,
            seg_pos,
            seen,
            catalog,
            ks=eval_p["k_values"],
            num_negatives=eval_p["num_candidate_negatives"],
            seed=eval_p["seed"],
        )
        for name in scorers:
            results[name][seg_name] = seg_res[name]
    for name in scorers:
        results[name]["overall"] = _weighted_overall(
            {s: results[name][s] for s in ("warm", "cold")}, counts
        )
    return {
        "n_users": {"overall": len(positives), **counts},
        **results,
    }


# Cabeça do NCF usada para pontuar cada definição de relevância (spec 002,
# FR-004, D6): relevância forte usa a cabeça principal; ampla usa a
# auxiliar de ``view`` — cada uma otimizada para o que está sendo medido.
_RELEVANCE_HEAD = {"strong": "strong", "broad": "view"}


def ranking_metrics(
    model: RecommenderBase,
    train: pd.DataFrame,
    history: pd.DataFrame,
    test: pd.DataFrame,
    eval_p: dict,
) -> dict:
    """Métricas de ranking Top-K sob relevância forte e ampla (FR-005a).

    Args:
        model: Modelo treinado.
        train: Eventos de treino (define usuários warm — D4).
        history: Eventos anteriores ao teste (treino+val) — features/candidatos.
        test: Eventos de teste.
        eval_p: Bloco ``eval`` do params.yaml.

    Returns:
        Dict aninhado relevância → scorer → segmento → métricas.
    """
    lookup = HistoryFeatureLookup(history)
    context_ts = _context_ts_by_user(test)
    seen = build_user_seen(pd.concat([history, test], ignore_index=True))
    train_users = {int(u) for u in train["user_idx"].unique()}
    relevances = {
        "strong": positives_by_user(test, STRONG_RELEVANCE_EVENTS),
        "broad": positives_by_user(
            test, BROAD_RELEVANCE_EVENTS, eval_p["max_broad_positives"]
        ),
    }
    out = {}
    for name, positives in relevances.items():
        logger.info("Ranking (%s): %d usuários", name, len(positives))
        scorers: dict[str, PairScorer] = {
            "model": make_model_scorer(
                model, lookup, context_ts, head=_RELEVANCE_HEAD[name]
            ),
            "popularity": make_popularity_scorer(lookup),
        }
        out[name] = _eval_relevance(
            scorers, positives, seen, lookup.catalog, train_users, eval_p
        )
    return out


def _save_plots(y_true: np.ndarray, y_proba: np.ndarray) -> None:
    """Salva dados de ROC e PR curves em metrics/plots/ para o DVC rastrear."""
    plots_dir = METRICS_DIR / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_data = [
        {"fpr": float(f), "tpr": float(t)} for f, t in zip(fpr, tpr, strict=True)
    ]
    (plots_dir / "roc_curve.json").write_text(json.dumps(roc_data))
    prec, rec, _ = precision_recall_curve(y_true, y_proba)
    pr_data = [
        {"precision": float(p), "recall": float(r)}
        for p, r in zip(prec, rec, strict=True)
    ]
    (plots_dir / "pr_curve.json").write_text(json.dumps(pr_data))


def _flatten(metrics: dict, prefix: str = "") -> dict[str, float]:
    """Achata o dict aninhado em nomes ``a_b_c`` → float (para o MLflow)."""
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(_flatten(value, name))
        elif isinstance(value, (int, float)):
            flat[name] = float(value)
    return flat


def _log_and_save(metrics: dict, mlflow_p: dict, model_type: str) -> None:
    """Loga métricas no MLflow e persiste o JSON para o DVC rastrear."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_p["experiment_name"])
    with mlflow.start_run(run_name=f"evaluate-{model_type}"):
        mlflow.log_metrics(_flatten(metrics))
    out = METRICS_DIR / "eval_metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    logger.info("Métricas salvas em %s", out)


def run() -> None:
    """Executa a etapa de avaliação."""
    params = yaml.safe_load(open(PARAMS_PATH))
    labeling_p, eval_p = params["labeling"], params["eval"]
    model = load_model()
    train, val, test = load_splits()
    history = pd.concat([train, val], ignore_index=True)

    labeled_test = build_labeled_dataset(
        test,
        history=history,
        num_negatives=labeling_p["num_negatives"],
        popularity_alpha=labeling_p["popularity_alpha"],
        seed=labeling_p["seed"],
    )
    y_test = labeled_test[TARGET_COL].to_numpy(dtype="float32")
    y_proba = model.predict_proba(labeled_test[FEATURE_COLS].to_numpy(dtype="float32"))
    metrics = {
        "classification": compute_metrics(y_test, y_proba),
        "ranking": ranking_metrics(model, train, history, test, eval_p),
    }
    logger.info("Métricas de teste: %s", json.dumps(metrics["classification"]))
    _save_plots(y_test, y_proba)
    _log_and_save(metrics, params["mlflow"], params["train"]["model_type"])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
