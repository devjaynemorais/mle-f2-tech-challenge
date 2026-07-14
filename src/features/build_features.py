"""Etapa 2 do pipeline DVC: dados intermediários → features prontas p/ treino.

Responsabilidades (FR-003, D2):
- Aplicar features CAUSAIS (*as-of*) por evento — sem vazamento temporal.
- Anexar a categoria do item (conteúdo, cold-start — D4).
- Dividir cronologicamente em treino/validação/teste.

Nada é "fitado" aqui: scaler e estatísticas de referência são ajustados
apenas no treino, dentro do modelo (D5). As features causais são seguras
por construção — cada linha usa somente eventos anteriores a ela.
"""

import logging
from pathlib import Path

import pandas as pd
import yaml

from src.data.feature_engineering import build_causal_features, chronological_split

logger = logging.getLogger(__name__)

PARAMS_PATH = Path("params.yaml")
INTERIM_DIR = Path("data/interim")
CONTENT_DIR = Path("data/content")
PROCESSED_DIR = Path("data/processed")

# cat_idx para item sem categoria conhecida (roteado ao embedding "unknown").
UNKNOWN_CAT_IDX = -1


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria as features causais de interação para o RetailRocket dataset.

    Args:
        df: DataFrame limpo com colunas user_idx, item_idx, timestamp, event.

    Returns:
        DataFrame com features causais de evento, temporais, usuário e item.
    """
    return build_causal_features(df)


def attach_categories(df: pd.DataFrame, categories: pd.DataFrame) -> pd.DataFrame:
    """Anexa ``cat_idx`` contíguo (0..C-1) a partir do categoryid do item.

    O mapeamento é determinístico (categoryid ordenado). Itens sem categoria
    recebem ``UNKNOWN_CAT_IDX``.

    Args:
        df: Eventos com coluna itemid.
        categories: DataFrame (itemid, categoryid) do ETL de conteúdo.

    Returns:
        DataFrame com a coluna cat_idx adicionada.
    """
    df = df.copy()
    if categories.empty:
        df["cat_idx"] = UNKNOWN_CAT_IDX
        return df
    vocab = {
        cat: idx for idx, cat in enumerate(sorted(categories["categoryid"].unique()))
    }
    cat_by_item = dict(
        zip(categories["itemid"], categories["categoryid"].map(vocab), strict=True)
    )
    df["cat_idx"] = (
        df["itemid"].map(cat_by_item).fillna(UNKNOWN_CAT_IDX).astype("int64")
    )
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float,
    val_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide os dados cronologicamente em treino, validação e teste.

    Args:
        df: DataFrame com features e coluna timestamp.
        test_size: Proporção do conjunto de teste.
        val_size: Proporção do conjunto de validação.

    Returns:
        Tupla (train_df, val_df, test_df).
    """
    return chronological_split(df, val_size=val_size, test_size=test_size)


def _load_categories() -> pd.DataFrame:
    """Carrega o parquet de categorias (vazio se o ETL não rodou)."""
    content_file = CONTENT_DIR / "item_categories.parquet"
    if not content_file.exists():
        logger.warning("Sem %s — itens ficarão sem categoria.", content_file)
        return pd.DataFrame(
            {"itemid": pd.Series(dtype="int64"), "categoryid": pd.Series(dtype="int64")}
        )
    return pd.read_parquet(content_file)


def _save_splits(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> None:
    """Persiste os splits de treino, validação e teste em data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(PROCESSED_DIR / "train.parquet", index=False)
    val_df.to_parquet(PROCESSED_DIR / "val.parquet", index=False)
    test_df.to_parquet(PROCESSED_DIR / "test.parquet", index=False)
    logger.info(
        "Splits salvos — treino: %d | val: %d | teste: %d",
        len(train_df),
        len(val_df),
        len(test_df),
    )


def run() -> None:
    """Executa a etapa de feature engineering."""
    params = yaml.safe_load(open(PARAMS_PATH))
    pre_p = params["preprocess"]
    interim_file = INTERIM_DIR / "data_clean.parquet"

    if not interim_file.exists():
        logger.warning(
            "Arquivo interim não encontrado: %s — execute preprocess primeiro",
            interim_file,
        )
        return

    df = build_features(pd.read_parquet(interim_file))
    df = attach_categories(df, _load_categories())
    train_df, val_df, test_df = split_data(df, pre_p["test_size"], pre_p["val_size"])
    _save_splits(train_df, val_df, test_df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
