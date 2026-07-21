"""Testes das funções de plot (src/utils/plots.py).

Backend Agg (sem display) — plt.show() vira no-op, sem travar/abrir janela.
Cada teste fecha as figuras no final pra não acumular estado entre testes.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.utils.plots import (  # noqa: E402
    boxplots_por_evento,
    plot_conversion_funnel,
    plot_event_timeline,
    plot_power_law,
    plot_univariate,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _numeric_series(n: int = 30) -> pd.Series:
    rng = np.random.default_rng(0)
    return pd.Series(rng.random(n) * 100)


def test_plot_univariate_creates_three_axes():
    df = pd.DataFrame({"col": _numeric_series()})
    plot_univariate(df, "col", bins=10)
    fig = plt.gcf()
    assert len(fig.axes) == 3


def test_plot_event_timeline_creates_single_axis():
    events = pd.DataFrame(
        {
            "timestamp": pd.date_range("2015-05-01", periods=10, freq="h"),
        }
    )
    plot_event_timeline(events, freq="h")
    fig = plt.gcf()
    assert len(fig.axes) == 1
    assert "Volume de Eventos" in fig.axes[0].get_title()


def test_plot_power_law_creates_two_axes():
    series = pd.Series(np.random.default_rng(0).integers(1, 500, 50))
    plot_power_law(series, title="Interações por usuário")
    fig = plt.gcf()
    assert len(fig.axes) == 2


def test_plot_conversion_funnel_annotates_bars():
    events = pd.DataFrame(
        {"event": ["view"] * 10 + ["addtocart"] * 4 + ["transaction"] * 2}
    )
    plot_conversion_funnel(events)
    fig = plt.gcf()
    assert len(fig.axes) == 1
    # 3 barras + 3 rótulos de texto anotados por _annotate_bars
    assert len(fig.axes[0].texts) == 3


def test_boxplots_por_evento_single_column_no_flatten_branch():
    events = pd.DataFrame(
        {"event": ["view", "addtocart"] * 5, "valor": _numeric_series(10)}
    )
    boxplots_por_evento(events, num_cols=["valor"], n_cols=1)
    fig = plt.gcf()
    assert len(fig.axes) == 1


def test_boxplots_por_evento_multiple_columns_hides_unused_axes():
    events = pd.DataFrame(
        {
            "event": ["view", "addtocart", "transaction"] * 4,
            "a": _numeric_series(12),
            "b": _numeric_series(12),
            "c": _numeric_series(12),
        }
    )
    boxplots_por_evento(events, num_cols=["a", "b", "c"], n_cols=2)
    fig = plt.gcf()
    # grade 2x2 = 4 eixos; só 3 usados → o 4o fica invisível
    assert len(fig.axes) == 4
    visible = [ax.get_visible() for ax in fig.axes]
    assert visible.count(False) == 1
