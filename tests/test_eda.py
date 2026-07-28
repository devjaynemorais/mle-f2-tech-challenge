"""Testes dos helpers de EDA (src/utils/eda.py) — DataFrames sintéticos."""

import pandas as pd

from src.utils.eda import (
    freq_table,
    interaction_summary,
    sanity_check,
    taxa_conversao_evento,
)


def test_sanity_check_runs_without_nulls(capsys):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    sanity_check(df, name="teste")
    out = capsys.readouterr().out
    assert "Nenhum nulo encontrado" in out


def test_sanity_check_reports_nulls(capsys):
    df = pd.DataFrame({"a": [1, None, 3]})
    sanity_check(df)
    out = capsys.readouterr().out
    assert "Nulos por coluna" in out


def test_freq_table_has_expected_columns():
    df = pd.DataFrame({"event": ["view", "view", "addtocart"]})
    table = freq_table(df, "event")
    assert list(table.columns) == [
        "Frequência Absoluta",
        "Frequência Relativa (%)",
        "Frequência Acumulada (%)",
    ]
    assert table.loc["view", "Frequência Absoluta"] == 2


def test_taxa_conversao_evento_computes_funnel_rates():
    events = pd.DataFrame(
        {"event": ["view"] * 8 + ["addtocart"] * 4 + ["transaction"] * 2}
    )
    result = taxa_conversao_evento(events)
    assert result.loc["view", "% do Evento Anterior"] == 100.0
    assert result.loc["addtocart", "% do Evento Anterior"] == 50.0
    assert result.loc["transaction", "% do Evento Anterior"] == 50.0


def test_interaction_summary_describes_counts_per_entity():
    events = pd.DataFrame({"user_idx": [1, 1, 2, 3, 3, 3]})
    summary = interaction_summary(events, "user_idx")
    assert summary.columns[0] == "Interações por user_idx"
    assert summary.loc["count"].item() == 3  # 3 usuários distintos
