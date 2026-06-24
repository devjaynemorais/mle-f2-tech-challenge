"""Smoke tests — verificam imports e o funcionamento básico da Factory.

Estes testes não dependem de dados reais: usam arrays sintéticos gerados
internamente, então passam em qualquer clone limpo do repositório.
"""

import numpy as np
import pytest

from src.models import MLPRecommender, ModelFactory
from src.models.base import RecommenderBase


def test_modelos_registrados_na_factory() -> None:
    """Factory deve ter mlp, logistic e dummy registrados."""
    available = ModelFactory.available()
    assert "mlp" in available
    assert "logistic" in available
    assert "dummy" in available


def test_factory_cria_mlp() -> None:
    """ModelFactory.create deve retornar uma instância de RecommenderBase."""
    model = ModelFactory.create("mlp", input_dim=4)
    assert isinstance(model, RecommenderBase)


def test_factory_nome_invalido_lanca_erro() -> None:
    """Factory deve lançar ValueError para nomes não registrados."""
    with pytest.raises(ValueError, match="não encontrado"):
        ModelFactory.create("modelo_inexistente")


def test_mlp_fit_predict_com_dados_sinteticos() -> None:
    """MLP deve treinar e gerar predições válidas com dados sintéticos."""
    rng = np.random.default_rng(42)
    X = rng.random((100, 4)).astype(np.float32)
    y = rng.integers(0, 2, size=100).astype(np.float32)

    # Treina com poucas épocas apenas para validar o fluxo
    model = MLPRecommender(input_dim=4, hidden_dims=[8], epochs=2, batch_size=32)
    model.fit(X, y)

    proba = model.predict_proba(X)
    preds = model.predict(X)

    # Probabilidades devem estar entre 0 e 1
    assert proba.shape == (100,)
    assert all(0.0 <= p <= 1.0 for p in proba)

    # Predições devem ser apenas 0 ou 1
    assert set(np.unique(preds)).issubset({0, 1})
