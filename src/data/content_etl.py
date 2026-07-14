"""Etapa de ETL de conteúdo: extrai categoryid de item_properties_*.csv (T032).

Lê os arquivos brutos em chunks (são ~850MB), filtra as linhas com
property == 'categoryid' e mantém o valor mais recente por itemid.
Saída: data/content/item_categories.parquet (itemid, categoryid).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
CONTENT_DIR = Path("data/content")
OUT_FILE = CONTENT_DIR / "item_categories.parquet"

_CHUNKSIZE = 1_000_000
_USECOLS = ["timestamp", "itemid", "property", "value"]


def _read_category_rows(csv_path: Path) -> list[pd.DataFrame]:
    """Lê um CSV de propriedades em chunks e retorna só as linhas de categoria."""
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(csv_path, usecols=_USECOLS, chunksize=_CHUNKSIZE):
        rows = chunk.loc[chunk["property"] == "categoryid"]
        if not rows.empty:
            frames.append(rows[["timestamp", "itemid", "value"]])
    return frames


def extract_item_categories(csv_paths: list[Path]) -> pd.DataFrame:
    """Extrai a categoria mais recente de cada item.

    Args:
        csv_paths: Caminhos dos item_properties_part*.csv.

    Returns:
        DataFrame com colunas itemid (int64) e categoryid (int64),
        uma linha por item.
    """
    frames: list[pd.DataFrame] = []
    for path in csv_paths:
        logger.info("Lendo propriedades de %s", path)
        frames.extend(_read_category_rows(path))
    if not frames:
        return pd.DataFrame(
            {"itemid": pd.Series(dtype="int64"), "categoryid": pd.Series(dtype="int64")}
        )
    props = pd.concat(frames, ignore_index=True)
    props["categoryid"] = pd.to_numeric(props["value"], errors="coerce")
    props = props.dropna(subset=["categoryid"])
    latest = props.sort_values("timestamp", kind="stable").groupby("itemid").tail(1)
    out = latest[["itemid", "categoryid"]].astype("int64").reset_index(drop=True)
    return out


def run() -> None:
    """Executa o ETL de conteúdo e persiste item_categories.parquet."""
    CONTENT_DIR.mkdir(parents=True, exist_ok=True)
    csv_paths = sorted(RAW_DIR.glob("item_properties_part*.csv"))
    if not csv_paths:
        logger.warning(
            "Nenhum item_properties_part*.csv em %s — gerando parquet vazio "
            "(itens ficarão sem categoria).",
            RAW_DIR,
        )
    categories = extract_item_categories(csv_paths)
    categories.to_parquet(OUT_FILE, index=False)
    logger.info("Categorias salvas em %s (%d itens)", OUT_FILE, len(categories))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
