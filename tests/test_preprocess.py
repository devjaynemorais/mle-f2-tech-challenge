"""Testes do pré-processamento — usam DataFrames sintéticos.

Não dependem de arquivos em disco, então passam em qualquer clone limpo.
"""

import pandas as pd
import pytest

from src.data.preprocessor import DefaultPreprocessor, RetailRocketPreprocessor


# ---------------------------------------------------------------------------
# Fixtures compartilhadas
# ---------------------------------------------------------------------------


@pytest.fixture()
def df_com_duplicatas() -> pd.DataFrame:
    """DataFrame simples com linhas duplicadas."""
    return pd.DataFrame({"a": [1, 2, 3, 1, 2], "b": ["x", "y", "z", "x", "y"]})


@pytest.fixture()
def df_limpo() -> pd.DataFrame:
    """DataFrame sem duplicatas."""
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


@pytest.fixture()
def df_eventos() -> pd.DataFrame:
    """Eventos RetailRocket sintéticos com 3 usuários e interações variadas."""
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2015-05-01",
                    "2015-05-02",
                    "2015-05-03",
                    "2015-05-04",
                    "2015-05-05",
                    "2015-05-06",
                    "2015-05-07",
                    "2015-05-08",
                    "2015-05-09",
                ]
            ),
            "visitorid": [10, 10, 10, 10, 10, 20, 20, 30, 30],
            "event": [
                "view",
                "view",
                "addtocart",
                "view",
                "transaction",
                "view",
                "view",
                "view",
                "view",
            ],
            "itemid": [100, 200, 200, 300, 300, 100, 200, 100, 200],
        }
    )


# ---------------------------------------------------------------------------
# DefaultPreprocessor
# ---------------------------------------------------------------------------


def test_remove_duplicatas(df_com_duplicatas: pd.DataFrame) -> None:
    """Preprocessador deve eliminar linhas duplicadas."""
    preprocessor = DefaultPreprocessor()
    result = preprocessor.preprocess(df_com_duplicatas)
    assert not result.duplicated().any()


def test_sem_duplicatas_mantem_tamanho(df_limpo: pd.DataFrame) -> None:
    """DataFrame sem duplicatas não deve perder linhas."""
    preprocessor = DefaultPreprocessor()
    result = preprocessor.preprocess(df_limpo)
    assert len(result) == len(df_limpo)


def test_indice_resetado(df_com_duplicatas: pd.DataFrame) -> None:
    """Índice do resultado deve ser contíguo (0, 1, 2, ...)."""
    preprocessor = DefaultPreprocessor()
    result = preprocessor.preprocess(df_com_duplicatas)
    assert list(result.index) == list(range(len(result)))


# ---------------------------------------------------------------------------
# RetailRocketPreprocessor — filtragem de usuários
# ---------------------------------------------------------------------------


def test_filter_remove_cold_users(df_eventos: pd.DataFrame) -> None:
    """Usuários com menos de min_interactions eventos são removidos."""
    preprocessor = RetailRocketPreprocessor(min_interactions=5)
    result = preprocessor.preprocess(df_eventos)
    # visitorid=20 (2 eventos) e visitorid=30 (2 eventos) devem sumir
    assert 20 not in result["visitorid"].values
    assert 30 not in result["visitorid"].values


def test_filter_keeps_active_users(df_eventos: pd.DataFrame) -> None:
    """Usuários com interações suficientes são mantidos."""
    preprocessor = RetailRocketPreprocessor(min_interactions=5)
    result = preprocessor.preprocess(df_eventos)
    assert 10 in result["visitorid"].values


def test_filter_boundary_exactly_n(df_eventos: pd.DataFrame) -> None:
    """Usuário com exatamente N interações é mantido (limiar inclusivo)."""
    preprocessor = RetailRocketPreprocessor(min_interactions=2)
    result = preprocessor.preprocess(df_eventos)
    assert 20 in result["visitorid"].values
    assert 30 in result["visitorid"].values


# ---------------------------------------------------------------------------
# RetailRocketPreprocessor — encoding de IDs
# ---------------------------------------------------------------------------


def test_encode_ids_start_from_zero(df_eventos: pd.DataFrame) -> None:
    """user_idx e item_idx começam em 0."""
    preprocessor = RetailRocketPreprocessor(min_interactions=1)
    result = preprocessor.preprocess(df_eventos)
    assert result["user_idx"].min() == 0
    assert result["item_idx"].min() == 0


def test_encode_ids_are_contiguous(df_eventos: pd.DataFrame) -> None:
    """Índices são inteiros consecutivos sem lacunas."""
    preprocessor = RetailRocketPreprocessor(min_interactions=1)
    result = preprocessor.preprocess(df_eventos)
    n_users = result["user_idx"].nunique()
    n_items = result["item_idx"].nunique()
    assert set(result["user_idx"].unique()) == set(range(n_users))
    assert set(result["item_idx"].unique()) == set(range(n_items))


def test_encode_same_original_same_idx(df_eventos: pd.DataFrame) -> None:
    """Mesmo visitorid original sempre mapeia para mesmo user_idx."""
    preprocessor = RetailRocketPreprocessor(min_interactions=1)
    result = preprocessor.preprocess(df_eventos)
    for visitor_id in result["visitorid"].unique():
        idxs = result.loc[result["visitorid"] == visitor_id, "user_idx"].unique()
        assert len(idxs) == 1


# ---------------------------------------------------------------------------
# RetailRocketPreprocessor — saída
# ---------------------------------------------------------------------------


def test_output_sorted_by_timestamp(df_eventos: pd.DataFrame) -> None:
    """DataFrame de saída deve estar ordenado por timestamp."""
    preprocessor = RetailRocketPreprocessor(min_interactions=1)
    result = preprocessor.preprocess(df_eventos)
    assert result["timestamp"].is_monotonic_increasing


def test_output_has_required_columns(df_eventos: pd.DataFrame) -> None:
    """Saída deve conter colunas: user_idx, item_idx, timestamp, event."""
    preprocessor = RetailRocketPreprocessor(min_interactions=1)
    result = preprocessor.preprocess(df_eventos)
    for col in ("user_idx", "item_idx", "timestamp", "event"):
        assert col in result.columns
