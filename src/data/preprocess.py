"""Etapa 1 do pipeline DVC: dados brutos → dados intermediários limpos."""

import logging
from pathlib import Path

import pandas as pd
import yaml

from src.data.preprocessor import RetailRocketPreprocessor

logger = logging.getLogger(__name__)

PARAMS_PATH = Path("params.yaml")
RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")
RAW_FILE = RAW_DIR / "events.csv"


def run() -> None:
    """Executa a etapa de pré-processamento.

    Lê os parâmetros de params.yaml, carrega events.csv,
    aplica RetailRocketPreprocessor e salva em data/interim/.
    """
    params = yaml.safe_load(open(PARAMS_PATH))["preprocess"]
    random_state: int = params["random_state"]
    min_interactions: int = params["min_interactions"]

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    if not RAW_FILE.exists():
        logger.warning(
            "Arquivo bruto não encontrado: %s — coloque o dataset em data/raw/",
            RAW_FILE,
        )
        return

    df = pd.read_csv(RAW_FILE)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    preprocessor = RetailRocketPreprocessor(
        min_interactions=min_interactions, random_state=random_state
    )
    df_clean = preprocessor.preprocess(df)

    df_clean.to_parquet(INTERIM_DIR / "data_clean.parquet", index=False)
    logger.info("Dados limpos salvos em %s", INTERIM_DIR / "data_clean.parquet")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
