"""Modelos baseline para comparação com o MLP principal.

Todos os baselines seguem a mesma interface (RecommenderBase)
e são registrados na Factory, então podem ser criados com
ModelFactory.create("logistic") ou ModelFactory.create("dummy").
"""

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from src.models.base import RecommenderBase
from src.models.factory import ModelFactory


@ModelFactory.register("dummy")
class DummyRecommender(RecommenderBase):
    """Baseline mais simples: sempre prediz a classe majoritária.

    Serve como lower bound — qualquer modelo real deve superar isso.
    """

    def __init__(self) -> None:
        # strategy="most_frequent": prediz sempre a classe mais comum no treino
        self._model = DummyClassifier(strategy="most_frequent")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DummyRecommender":
        """Aprende qual é a classe majoritária."""
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retorna probabilidade constante baseada na frequência da classe positiva."""
        return self._model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Retorna predições binárias."""
        return self._model.predict(X)


@ModelFactory.register("logistic")
class LogisticRecommender(RecommenderBase):
    """Baseline de regressão logística com sklearn.

    Bom ponto de partida antes de usar redes neurais:
    mais rápido de treinar e fácil de interpretar.

    Args:
        C: Força inversa da regularização (maior C = menos regularização).
        max_iter: Número máximo de iterações do solver.
        random_state: Semente para reprodutibilidade.
    """

    def __init__(
        self, C: float = 1.0, max_iter: int = 1000, random_state: int = 42
    ) -> None:
        self._model = LogisticRegression(
            C=C, max_iter=max_iter, random_state=random_state
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRecommender":
        """Treina a regressão logística."""
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Retorna a probabilidade da classe positiva."""
        return self._model.predict_proba(X)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Retorna predições binárias com base no threshold."""
        return (self.predict_proba(X) >= threshold).astype(int)
