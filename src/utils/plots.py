"""Visualization utilities for exploratory data analysis."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns
from scipy import stats


def plot_univariate(
    df: pd.DataFrame,
    col: str,
    bins: int = 30,
    figsize: tuple[int, int] = (14, 4),
) -> None:
    """Histograma + KDE + Boxplot + QQ Plot para variável numérica."""
    series = df[col].dropna()
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    sns.histplot(series, bins=bins, kde=True, ax=axes[0], color="steelblue")
    axes[0].set_title(f"Distribuição — {col}")
    axes[0].set_xlabel(col)

    sns.boxplot(x=series, ax=axes[1], color="steelblue")
    axes[1].set_title(f"Boxplot — {col}")

    stats.probplot(series, plot=axes[2])
    axes[2].set_title(f"QQ Plot — {col}")

    plt.suptitle(col, fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_event_timeline(
    events: pd.DataFrame,
    timestamp_col: str = "timestamp",
    freq: str = "D",
    figsize: tuple[int, int] = (12, 4),
) -> None:
    """Série temporal do volume de eventos por frequência."""
    daily = events.set_index(timestamp_col).resample(freq).size()

    fig, ax = plt.subplots(figsize=figsize)
    daily.plot(ax=ax, color="steelblue", linewidth=1.2)
    ax.set_title(f"Volume de Eventos por {freq}", fontsize=12)
    ax.set_xlabel("Data")
    ax.set_ylabel("Eventos")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def plot_power_law(
    series: pd.Series,
    title: str = "Distribuição",
    clip_upper: int = 100,
    figsize: tuple[int, int] = (12, 4),
) -> None:
    """Histograma (clipped) + log-log para checagem de power-law."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].hist(series.clip(upper=clip_upper), bins=50, edgecolor="white", color="steelblue")
    axes[0].set_title(f"{title} (clipped em {clip_upper})")
    axes[0].set_xlabel("Interações")
    axes[0].set_ylabel("Frequência")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    sorted_vals = series.sort_values(ascending=False).reset_index(drop=True)
    axes[1].loglog(sorted_vals.values, color="steelblue", linewidth=1.2)
    axes[1].set_title(f"{title} — Log-Log (Power Law)")
    axes[1].set_xlabel("Rank")
    axes[1].set_ylabel("Interações")
    axes[1].grid(True, which="both", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()


def _annotate_bars(ax: plt.Axes, bars: list, values: list) -> None:
    """Adiciona rótulo de valor ao lado de cada barra horizontal."""
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() * 1.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}",
            va="center",
            fontsize=10,
        )


def plot_conversion_funnel(
    events: pd.DataFrame,
    event_col: str = "event",
    order: list[str] | None = None,
    figsize: tuple[int, int] = (8, 4),
) -> None:
    """Funil horizontal de conversão entre tipos de eventos."""
    if order is None:
        order = ["view", "addtocart", "transaction"]
    counts = events[event_col].value_counts().reindex(order).dropna()
    order_rev = order[::-1]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(order_rev, counts[order_rev].values, color=["green", "orange", "steelblue"])
    _annotate_bars(ax, bars, counts[order_rev].values)
    ax.set_title("Funil de Conversão", fontsize=12)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()


def boxplots_por_evento(
    events: pd.DataFrame,
    num_cols: list[str],
    event_col: str = "event",
    n_cols: int = 1,
    figsize_per: tuple[int, int] = (10, 4),
) -> None:
    """Boxplots de variáveis numéricas separados por tipo de evento."""
    n_rows = math.ceil(len(num_cols) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per[0] * n_cols, figsize_per[1] * n_rows),
    )
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, col in enumerate(num_cols):
        sns.boxplot(data=events, x=event_col, y=col, ax=axes_flat[i], palette="Set2")
        axes_flat[i].set_title(f"{col} por tipo de evento")
        axes_flat[i].grid(True, linestyle="--", alpha=0.5)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    plt.show()
