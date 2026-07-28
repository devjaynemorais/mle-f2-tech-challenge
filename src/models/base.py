"""Abstract base class for all recommender models."""

from abc import ABC, abstractmethod
from typing import Any


class RecommenderBase(ABC):
    """Contrato que todos os recomendadores devem seguir."""

    @abstractmethod
    def fit(self, X: Any, y: Any) -> "RecommenderBase":
        """Treina o modelo com os dados fornecidos."""

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """Retorna predições binárias (0 ou 1)."""

    @abstractmethod
    def predict_proba(self, X: Any) -> Any:
        """Retorna a probabilidade de interação (valor entre 0 e 1)."""
