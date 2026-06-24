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
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

PARAMS_PATH = Path("params.yaml")
INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features a partir dos dados limpos.

    TODO: substituir pela lógica do dataset escolhido.
    Exemplos comuns:
      - Codificar IDs de usuário/item como inteiros
      - Normalizar colunas numéricas
      - Criar features de interação ou históricas
      - Gerar amostras negativas (para feedback implícito)
    """
    # TODO: implementar feature engineering aqui
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide os dados em treino, validação e teste.

    Args:
        df: DataFrame com features e rótulos.
        test_size: Proporção do conjunto de teste.
        val_size: Proporção do conjunto de validação (sobre o treino+val).
        random_state: Semente para reprodutibilidade.

    Returns:
        Tupla (train_df, val_df, test_df).
    """
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=random_state
    )
    train, val = train_test_split(
        train_val, test_size=val_size, random_state=random_state
    )
    return train, val, test


def run() -> None:
    """Executa a etapa de feature engineering.

    Lê os dados de data/interim/, gera as features e salva
    os splits train/val/test em data/processed/.
    """
    params = yaml.safe_load(open(PARAMS_PATH))
    pre_p = params["preprocess"]
    test_size: float = pre_p["test_size"]
    val_size: float = pre_p["val_size"]
    random_state: int = pre_p["random_state"]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # TODO: ajustar o nome do arquivo conforme a etapa de preprocess
    interim_file = INTERIM_DIR / "data_clean.parquet"

    if not interim_file.exists():
        logger.warning(
            "Arquivo interim não encontrado: %s — execute a etapa preprocess primeiro",
            interim_file,
        )
        return

    df = pd.read_parquet(interim_file)
    df = build_features(df)

    train_df, val_df, test_df = split_data(df, test_size, val_size, random_state)

    train_df.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    val_df.to_parquet(PROCESSED_DIR / "val.parquet", index=False)
    test_df.to_parquet(PROCESSED_DIR / "test.parquet", index=False)

    logger.info(
        "Splits salvos — treino: %d | val: %d | teste: %d",
        len(train_df), len(val_df), len(test_df),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
