"""Interface comum e Factory para todos os modelos de recomendação.

Padrões aplicados:
- Abstract: garante que todo modelo implemente fit(), predict() e predict_proba()
- Factory: permite criar modelos pelo nome sem importar cada classe diretamente
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any


class RecommenderBase(ABC):
    """Contrato que todos os recomendadores devem seguir."""

    @abstractmethod
    def fit(self, X: Any, y: Any) -> "RecommenderBase":
        """Treina o modelo com os dados fornecidos."""
        ...

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """Retorna predições binárias (0 ou 1)."""
        ...

    @abstractmethod
    def predict_proba(self, X: Any) -> Any:
        """Retorna a probabilidade de interação (valor entre 0 e 1)."""
        ...


class ModelFactory:
    """Registro central de modelos.

    Uso:
        # Registrar um modelo (feito com o decorador):
        @ModelFactory.register("mlp")
        class MLPRecommender(RecommenderBase): ...

        # Criar uma instância pelo nome:
        model = ModelFactory.create("mlp", input_dim=64)
    """

    # Dicionário interno: nome → classe do modelo
    _registry: dict[str, type[RecommenderBase]] = {}

    @classmethod
    def register(
        cls, name: str
    ) -> Callable[[type[RecommenderBase]], type[RecommenderBase]]:
        """Decorador que registra uma classe de modelo pelo nome."""

        def decorator(model_cls: type[RecommenderBase]) -> type[RecommenderBase]:
            cls._registry[name] = model_cls
            return model_cls

        return decorator

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> RecommenderBase:
        """Instancia o modelo registrado com o nome fornecido."""
        if name not in cls._registry:
            available = list(cls._registry)
            msg = f"Modelo '{name}' não encontrado. Disponíveis: {available}"
            raise ValueError(msg)
        return cls._registry[name](**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        """Retorna a lista de nomes de modelos registrados."""
        return list(cls._registry)
