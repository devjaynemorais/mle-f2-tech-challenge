"""Etapa 2 do pipeline DVC: dados intermediários → features prontas para treinamento.

Responsabilidades:
- Criar/transformar features a partir dos dados limpos
- Dividir em treino, validação e teste
- Salvar os splits em data/processed/

TODO: implementar a lógica específica após escolher o dataset.
"""

import logging
from pathlib import Path

import pandas as pd
import yaml

from src.data.feature_engineering import build_interaction_features, chronological_split

logger = logging.getLogger(__name__)

PARAMS_PATH = Path("params.yaml")
INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features de interação para o RetailRocket dataset.

    Args:
        df: DataFrame limpo com colunas user_idx, item_idx, timestamp, event.

    Returns:
        DataFrame com features de evento, temporais, usuário e item.
    """
    return build_interaction_features(df)


def split_data(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    random_state: int,  # noqa: ARG001 — mantido para compatibilidade de assinatura
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide os dados cronologicamente em treino, validação e teste.

    Args:
        df: DataFrame com features e coluna timestamp.
        test_size: Proporção do conjunto de teste.
        val_size: Proporção do conjunto de validação.
        random_state: Ignorado — split é determinístico por timestamp.

    Returns:
        Tupla (train_df, val_df, test_df).
    """
    return chronological_split(df, val_size=val_size, test_size=test_size)


def _save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Persiste os splits de treino, validação e teste em data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    val_df.to_parquet(PROCESSED_DIR / "val.parquet", index=False)
    test_df.to_parquet(PROCESSED_DIR / "test.parquet", index=False)
    logger.info(
        "Splits salvos — treino: %d | val: %d | teste: %d",
        len(train_df), len(val_df), len(test_df),
    )


def run() -> None:
    """Executa a etapa de feature engineering."""
    params = yaml.safe_load(open(PARAMS_PATH))
    pre_p = params["preprocess"]
    interim_file = INTERIM_DIR / "data_clean.parquet"

    if not interim_file.exists():
        logger.warning(
            "Arquivo interim não encontrado: %s — execute preprocess primeiro",
            interim_file,
        )
        return

    df = build_features(pd.read_parquet(interim_file))
    train_df, val_df, test_df = split_data(
        df, pre_p["test_size"], pre_p["val_size"], pre_p["random_state"]
    )
    _save_splits(train_df, val_df, test_df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
