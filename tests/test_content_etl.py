"""Testes do ETL de conteúdo (extração de categoria por item)."""

from pathlib import Path

import pandas as pd
import pytest

from src.data import content_etl
from src.data.content_etl import _read_category_rows, extract_item_categories


def _write_csv(path: Path, rows: list[tuple[int, int, str, str]]) -> Path:
    pd.DataFrame(
        {
            "timestamp": [r[0] for r in rows],
            "itemid": [r[1] for r in rows],
            "property": [r[2] for r in rows],
            "value": [r[3] for r in rows],
        }
    ).to_csv(path, index=False)
    return path


def test_read_category_rows_filters_only_categoryid(tmp_path: Path):
    csv = _write_csv(
        tmp_path / "props.csv",
        [
            (1, 100, "categoryid", "5"),
            (2, 100, "available", "1"),
            (3, 200, "categoryid", "9"),
        ],
    )
    frames = _read_category_rows(csv)
    combined = pd.concat(frames, ignore_index=True)
    assert set(combined["itemid"]) == {100, 200}
    assert list(combined.columns) == ["timestamp", "itemid", "value"]


def test_extract_item_categories_keeps_most_recent_per_item(tmp_path: Path):
    csv = _write_csv(
        tmp_path / "props.csv",
        [
            (1, 100, "categoryid", "5"),
            (2, 100, "categoryid", "7"),  # mais recente → categoryid=7 vence
            (1, 200, "categoryid", "9"),
        ],
    )
    result = extract_item_categories([csv])
    result = result.set_index("itemid")
    assert result.loc[100, "categoryid"] == 7
    assert result.loc[200, "categoryid"] == 9


def test_extract_item_categories_empty_when_no_paths():
    result = extract_item_categories([])
    assert list(result.columns) == ["itemid", "categoryid"]
    assert result.empty
    assert str(result["itemid"].dtype) == "int64"


def test_extract_item_categories_drops_non_numeric_values(tmp_path: Path):
    csv = _write_csv(
        tmp_path / "props.csv",
        [
            (1, 100, "categoryid", "n12.000"),  # valor não-numérico → descartado
            (1, 200, "categoryid", "3"),
        ],
    )
    result = extract_item_categories([csv])
    assert list(result["itemid"]) == [200]


def test_run_writes_parquet_from_raw_csvs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    content_dir = tmp_path / "content"
    out_file = content_dir / "item_categories.parquet"
    _write_csv(
        raw_dir / "item_properties_part1.csv",
        [(1, 100, "categoryid", "5")],
    )
    monkeypatch.setattr(content_etl, "RAW_DIR", raw_dir)
    monkeypatch.setattr(content_etl, "CONTENT_DIR", content_dir)
    monkeypatch.setattr(content_etl, "OUT_FILE", out_file)

    content_etl.run()

    saved = pd.read_parquet(out_file)
    assert list(saved["itemid"]) == [100]


def test_run_writes_empty_parquet_when_no_raw_csvs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    content_dir = tmp_path / "content"
    out_file = content_dir / "item_categories.parquet"
    monkeypatch.setattr(content_etl, "RAW_DIR", raw_dir)
    monkeypatch.setattr(content_etl, "CONTENT_DIR", content_dir)
    monkeypatch.setattr(content_etl, "OUT_FILE", out_file)

    content_etl.run()

    assert pd.read_parquet(out_file).empty
