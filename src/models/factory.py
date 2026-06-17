"""Factory pattern para criação de modelos de recomendação.

Uso:
    # Registrar (feito via decorador nos módulos de modelo):
    @ModelFactory.register("mlp")
    class MLPRecommender(RecommenderBase): ...

    # Criar instância pelo nome (lido de params.yaml):
    model = ModelFactory.create("mlp", input_dim=64)
"""

from collections.abc import Callable
from typing import Any

from src.models.base import RecommenderBase


class ModelFactory:
    """Registro central de modelos — cria qualquer recomendador pelo nome."""

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
