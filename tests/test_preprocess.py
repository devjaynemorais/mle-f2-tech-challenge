"""Testes do pré-processamento — usam DataFrames sintéticos.

Não dependem de arquivos em disco, então passam em qualquer clone limpo.
"""

import pandas as pd
import pytest

from src.data.preprocess import DefaultPreprocessor


@pytest.fixture()
def df_com_duplicatas() -> pd.DataFrame:
    """DataFrame simples com linhas duplicadas."""
    return pd.DataFrame({"a": [1, 2, 3, 1, 2], "b": ["x", "y", "z", "x", "y"]})


@pytest.fixture()
def df_limpo() -> pd.DataFrame:
    """DataFrame sem duplicatas."""
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


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
