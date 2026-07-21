"""Testes da etapa DVC `feature_eng` (src/features/build_features.py)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.features import build_features as stage
from src.features.build_features import (
    UNKNOWN_CAT_IDX,
    attach_categories,
    build_features,
    split_data,
)


def _interim_events(n: int = 20) -> pd.DataFrame:
    """Eventos sintéticos pós-preprocess (com itemid/visitorid crus + idx)."""
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime("2015-05-01")
            + pd.to_timedelta(np.arange(n), unit="h"),
            "visitorid": rng.integers(0, 5, n),
            "itemid": rng.integers(100, 110, n),
            "user_idx": rng.integers(0, 5, n),
            "item_idx": rng.integers(0, 10, n),
            "event": rng.choice(["view", "addtocart"], size=n, p=[0.8, 0.2]),
        }
    )


def test_build_features_wraps_causal_features():
    df = build_features(_interim_events())
    assert "frequency" in df.columns
    assert "recency_days" in df.columns


def test_attach_categories_maps_known_items():
    df = pd.DataFrame({"itemid": [100, 101, 102]})
    categories = pd.DataFrame({"itemid": [100, 101], "categoryid": [50, 20]})
    result = attach_categories(df, categories)
    # vocabulário ordenado por categoryid: 20→0, 50→1
    assert result.loc[result["itemid"] == 100, "cat_idx"].item() == 1
    assert result.loc[result["itemid"] == 101, "cat_idx"].item() == 0
    assert result.loc[result["itemid"] == 102, "cat_idx"].item() == UNKNOWN_CAT_IDX


def test_attach_categories_empty_categories_uses_unknown():
    df = pd.DataFrame({"itemid": [1, 2]})
    empty = pd.DataFrame(
        {"itemid": pd.Series(dtype="int64"), "categoryid": pd.Series(dtype="int64")}
    )
    result = attach_categories(df, empty)
    assert (result["cat_idx"] == UNKNOWN_CAT_IDX).all()


def test_split_data_returns_three_chronological_splits():
    df = _interim_events(20)
    train, val, test = split_data(df, test_size=0.2, val_size=0.1)
    assert len(train) + len(val) + len(test) == 20
    assert train["timestamp"].max() <= val["timestamp"].min()
    assert val["timestamp"].max() <= test["timestamp"].min()


def test_load_categories_missing_file_returns_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(stage, "CONTENT_DIR", tmp_path / "content")
    result = stage._load_categories()
    assert result.empty
    assert list(result.columns) == ["itemid", "categoryid"]


def test_load_categories_reads_existing_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    content_dir = tmp_path / "content"
    content_dir.mkdir()
    pd.DataFrame({"itemid": [1], "categoryid": [9]}).to_parquet(
        content_dir / "item_categories.parquet", index=False
    )
    monkeypatch.setattr(stage, "CONTENT_DIR", content_dir)
    result = stage._load_categories()
    assert list(result["itemid"]) == [1]


def test_save_splits_writes_three_parquet_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    processed_dir = tmp_path / "processed"
    monkeypatch.setattr(stage, "PROCESSED_DIR", processed_dir)
    df = _interim_events(6)
    stage._save_splits(df.iloc[:3], df.iloc[3:5], df.iloc[5:])
    assert (processed_dir / "train.parquet").exists()
    assert (processed_dir / "val.parquet").exists()
    assert (processed_dir / "test.parquet").exists()


def _patch_stage_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict:
    params_path = tmp_path / "params.yaml"
    params_path.write_text(
        yaml.dump({"preprocess": {"test_size": 0.2, "val_size": 0.1}})
    )
    paths = {
        "PARAMS_PATH": params_path,
        "INTERIM_DIR": tmp_path / "interim",
        "CONTENT_DIR": tmp_path / "content",
        "PROCESSED_DIR": tmp_path / "processed",
    }
    for attr, value in paths.items():
        monkeypatch.setattr(stage, attr, value)
    return paths


def test_run_skips_when_interim_file_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _patch_stage_paths(monkeypatch, tmp_path)
    stage.run()
    assert not paths["PROCESSED_DIR"].exists()


def test_run_end_to_end_writes_processed_splits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    paths = _patch_stage_paths(monkeypatch, tmp_path)
    paths["INTERIM_DIR"].mkdir(parents=True)
    _interim_events(20).to_parquet(
        paths["INTERIM_DIR"] / "data_clean.parquet", index=False
    )

    stage.run()

    train = pd.read_parquet(paths["PROCESSED_DIR"] / "train.parquet")
    assert "cat_idx" in train.columns  # sem categorias reais → tudo unknown
    assert (train["cat_idx"] == UNKNOWN_CAT_IDX).all()
