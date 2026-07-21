"""Strategy pattern para pré-processamento de dados.

Uso:
    preprocessor = RetailRocketPreprocessor(min_interactions=5)
    df_clean = preprocessor.preprocess(df_raw)
"""

import logging
from abc import ABC, abstractmethod

import pandas as pd

logger = logging.getLogger(__name__)


class PreprocessStrategy(ABC):
    """Interface que toda estratégia de pré-processamento deve seguir."""

    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Recebe o DataFrame bruto e retorna o DataFrame limpo.

        Args:
            df: DataFrame com os dados brutos.

        Returns:
            DataFrame limpo pronto para feature engineering.
        """


class DefaultPreprocessor(PreprocessStrategy):
    """Estratégia genérica — remove duplicatas e reseta o índice.

    Args:
        random_state: Semente para reprodutibilidade.
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica limpeza básica: remove duplicatas e reseta índice."""
        logger.info("Linhas brutas: %d", len(df))
        df = df.drop_duplicates().reset_index(drop=True)
        logger.info("Linhas após limpeza: %d", len(df))
        return df


class RetailRocketPreprocessor(PreprocessStrategy):
    """Estratégia específica para o RetailRocket dataset.

    Filtra usuários com poucas interações, faz encoding de IDs
    e ordena eventos por timestamp.

    Args:
        min_interactions: Mínimo de interações por usuário.
        random_state: Semente para reprodutibilidade.
    """

    def __init__(self, min_interactions: int = 5, random_state: int = 42) -> None:
        self.min_interactions = min_interactions
        self.random_state = random_state

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpa e normaliza os eventos do RetailRocket.

        TODO: implementar após download do dataset em data/raw/.
        """
        logger.info("Linhas brutas: %d", len(df))
        df = self._filter_users(df)
        df = self._encode_ids(df)
        df = df.sort_values("timestamp").reset_index(drop=True)
        logger.info("Linhas após limpeza: %d", len(df))
        return df

    def _filter_users(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove usuários com menos de min_interactions eventos."""
        counts = df["visitorid"].value_counts()
        active = counts[counts >= self.min_interactions].index
        return df[df["visitorid"].isin(active)]

    def _encode_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converte visitorid e itemid para índices inteiros contíguos."""
        df = df.copy()
        df["user_idx"] = pd.factorize(df["visitorid"])[0]
        df["item_idx"] = pd.factorize(df["itemid"])[0]
        return df
