"""Smoke tests — verificam imports e o funcionamento básico da Factory.

Estes testes não dependem de dados reais: usam arrays sintéticos gerados
internamente, então passam em qualquer clone limpo do repositório.
"""

import numpy as np
import pytest

from src.data.feature_contract import CONT_COLS
from src.models import ModelFactory, NCFRecommender
from src.models.base import RecommenderBase


def test_modelos_registrados_na_factory() -> None:
    """Factory deve ter ncf, logistic e dummy registrados."""
    available = ModelFactory.available()
    assert "ncf" in available
    assert "logistic" in available
    assert "dummy" in available


def test_factory_cria_ncf() -> None:
    """ModelFactory.create deve retornar uma instância de RecommenderBase."""
    model = ModelFactory.create("ncf", n_users=10, n_items=10)
    assert isinstance(model, RecommenderBase)


def test_factory_nome_invalido_lanca_erro() -> None:
    """Factory deve lançar ValueError para nomes não registrados."""
    with pytest.raises(ValueError, match="não encontrado"):
        ModelFactory.create("modelo_inexistente")


def test_ncf_fit_predict_com_dados_sinteticos() -> None:
    """NCF deve treinar e gerar predições válidas com dados sintéticos."""
    rng = np.random.default_rng(42)
    n = 100
    users = rng.integers(0, 10, n)
    items = rng.integers(0, 15, n)
    cont = rng.random((n, len(CONT_COLS)))
    X = np.column_stack([users, items, cont]).astype(np.float32)
    y = rng.integers(0, 2, size=n).astype(np.float32)

    # Treina com poucas épocas apenas para validar o fluxo
    model = NCFRecommender(
        n_users=10,
        n_items=15,
        embedding_dim=4,
        hidden_dims=[8],
        epochs=2,
        batch_size=32,
    )
    model.fit(X, y)

    proba = model.predict_proba(X)
    preds = model.predict(X)

    # Probabilidades devem estar entre 0 e 1
    assert proba.shape == (n,)
    assert all(0.0 <= p <= 1.0 for p in proba)

    # Predições devem ser apenas 0 ou 1
    assert set(np.unique(preds)).issubset({0, 1})
