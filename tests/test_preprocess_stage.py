"""Testes da etapa DVC `preprocess` (src/data/preprocess.py — orquestração)."""

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.data import preprocess


def _write_params(tmp_path: Path) -> Path:
    params_path = tmp_path / "params.yaml"
    params_path.write_text(
        yaml.dump({"preprocess": {"min_interactions": 5, "random_state": 42}})
    )
    return params_path


def _patch_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    interim_dir = tmp_path / "interim"
    monkeypatch.setattr(preprocess, "PARAMS_PATH", _write_params(tmp_path))
    monkeypatch.setattr(preprocess, "RAW_DIR", raw_dir)
    monkeypatch.setattr(preprocess, "RAW_FILE", raw_dir / "events.csv")
    monkeypatch.setattr(preprocess, "INTERIM_DIR", interim_dir)
    return raw_dir, interim_dir


def test_run_skips_when_raw_file_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _, interim_dir = _patch_paths(monkeypatch, tmp_path)
    preprocess.run()
    assert interim_dir.exists()  # cria o diretório mesmo sem processar
    assert not (interim_dir / "data_clean.parquet").exists()


def test_run_cleans_and_saves_parquet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    raw_dir, interim_dir = _patch_paths(monkeypatch, tmp_path)
    # 1 visitante com 5 eventos (>= min_interactions default) sobrevive ao filtro.
    n = 5
    pd.DataFrame(
        {
            "timestamp": [1430438400000 + i * 1000 for i in range(n)],
            "visitorid": [1] * n,
            "itemid": [10, 11, 10, 12, 13],
            "event": ["view"] * n,
        }
    ).to_csv(raw_dir / "events.csv", index=False)

    preprocess.run()

    saved = pd.read_parquet(interim_dir / "data_clean.parquet")
    assert len(saved) == n
    assert "user_idx" in saved.columns
    assert "item_idx" in saved.columns
