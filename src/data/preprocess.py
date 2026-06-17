"""Etapa 1 do pipeline DVC: dados brutos → dados intermediários limpos.

Padrão Strategy: a lógica de limpeza fica em classes separadas (PreprocessStrategy),
então é possível trocar o preprocessamento sem alterar o pipeline.

TODO: implementar a lógica específica após escolher o dataset.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

PARAMS_PATH = Path("params.yaml")
RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")


# ---------------------------------------------------------------------------
# Padrão Strategy: interface de pré-processamento
# ---------------------------------------------------------------------------


class PreprocessStrategy(ABC):
    """Interface que toda estratégia de pré-processamento deve seguir."""

    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Recebe o DataFrame bruto e retorna o DataFrame limpo."""
        ...


class DefaultPreprocessor(PreprocessStrategy):
    """Estratégia padrão — adaptar ao dataset escolhido.

    Args:
        random_state: Semente para reprodutibilidade.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica limpeza básica no DataFrame.

        TODO: substituir pela lógica específica do dataset escolhido.
        Exemplos comuns:
          - Remover duplicatas
          - Filtrar usuários/itens com poucas interações
          - Converter tipos de colunas
          - Ordenar por timestamp
        """
        logger.info("Linhas brutas: %d", len(df))

        # TODO: adicionar limpeza específica do dataset aqui
        df = df.drop_duplicates()

        logger.info("Linhas após limpeza: %d", len(df))
        return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Ponto de entrada do DVC
# ---------------------------------------------------------------------------


def run() -> None:
    """Executa a etapa de pré-processamento.

    Lê os parâmetros de params.yaml, carrega o dado bruto,
    aplica o preprocessamento e salva o resultado em data/interim/.
    """
    params = yaml.safe_load(open(PARAMS_PATH))["preprocess"]
    random_state: int = params["random_state"]

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    # TODO: ajustar o nome do arquivo conforme o dataset escolhido
    raw_file = RAW_DIR / "dataset.csv"

    if not raw_file.exists():
        logger.warning(
            "Arquivo bruto não encontrado: %s — coloque o dataset em data/raw/",
            raw_file,
        )
        return

    df = pd.read_csv(raw_file)
    preprocessor = DefaultPreprocessor(random_state=random_state)
    df_clean = preprocessor.preprocess(df)

    # TODO: ajustar o nome do arquivo de saída conforme necessário
    df_clean.to_parquet(INTERIM_DIR / "data_clean.parquet", index=False)
    logger.info("Dados limpos salvos em %s", INTERIM_DIR / "data_clean.parquet")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
