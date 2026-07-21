"""Teste do helper genérico de download (src/data/make_dataset.py).

Módulo scaffold — não faz parte do pipeline DVC ativo (o dataset real vem
via Kaggle, ver README), mas ``download()`` é código alcançável e funcional.
"""

from pathlib import Path

import pytest

from src.data.make_dataset import download


def test_download_calls_urlretrieve_and_creates_parent_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calls = []
    monkeypatch.setattr(
        "urllib.request.urlretrieve", lambda url, dest: calls.append((url, dest))
    )
    dest = tmp_path / "nested" / "events.csv"
    download("https://example.com/events.csv", dest)
    assert dest.parent.exists()
    assert calls == [("https://example.com/events.csv", dest)]
