"""Utility functions for exploratory data analysis."""

from __future__ import annotations

import pandas as pd


def sanity_check(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Print shape, dtypes, null counts and duplicate count."""
    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  Sanity Check — {name}")
    print(sep)
    print(f"Shape : {df.shape[0]:,} linhas × {df.shape[1]} colunas")
    print(f"\nTipos:\n{df.dtypes.to_string()}")

    nulls = df.isnull().sum()
    null_pct = (nulls / len(df) * 100).round(2)
    null_df = pd.DataFrame({"Nulos": nulls, "%": null_pct})
    nulos_existentes = null_df[null_df["Nulos"] > 0]
    print("\nNulos por coluna:")
    if nulos_existentes.empty:
        print("  Nenhum nulo encontrado.")
    else:
        print(nulos_existentes.to_string())

    print(f"\nDuplicatas: {df.duplicated().sum():,}")


def freq_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Frequency table with absolute, relative and cumulative counts."""
    freq_abs = df[col].value_counts(dropna=False)
    freq_rel = (df[col].value_counts(normalize=True, dropna=False) * 100).round(2)
    return pd.DataFrame(
        {
            "Frequência Absoluta": freq_abs.values,
            "Frequência Relativa (%)": freq_rel.values,
            "Frequência Acumulada (%)": freq_rel.cumsum().round(2).values,
        },
        index=freq_abs.index,
    )


def taxa_conversao_evento(
    events: pd.DataFrame,
    event_col: str = "event",
    order: list[str] | None = None,
) -> pd.DataFrame:
    """Conversion rates across event types (view → addtocart → transaction)."""
    if order is None:
        order = ["view", "addtocart", "transaction"]
    counts = events[event_col].value_counts().reindex(order).dropna()
    prev_rates: dict[str, float] = {}
    for i, evt in enumerate(counts.index):
        if i == 0:
            prev_rates[evt] = 100.0
        else:
            prev_count = counts.iloc[i - 1]
            prev_rates[evt] = round(counts[evt] / prev_count * 100, 2) if prev_count else 0.0
    return pd.DataFrame(
        {
            "Total": counts,
            "% do Total": (counts / counts.sum() * 100).round(2),
            "% do Evento Anterior": pd.Series(prev_rates),
        }
    )


def interaction_summary(events: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Descriptive stats for interactions per entity (user or item)."""
    counts = events.groupby(id_col).size()
    stats = counts.describe().round(2)
    stats.name = f"Interações por {id_col}"
    return stats.to_frame()
