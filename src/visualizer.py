"""
visualizer.py — Generates line charts for the Ethiopia electricity analysis.

Public API:
    plot_indicator(df, column, ylabel, title, trend, config, output_filename) -> None
    generate_all_charts(df, results, config) -> None
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend; safe for scripts and tests
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

_STEP_NAME = "visualizer.plot_indicator"

# Resolve the charts/ directory relative to this source file so the path is
# correct regardless of the current working directory.
# __file__ → src/visualizer.py  →  dirname → src/  →  join(..) → project root
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SRC_DIR, "..")
_CHARTS_DIR = os.path.join(_PROJECT_ROOT, "charts")


def plot_indicator(
    df: pd.DataFrame,
    column: str,
    ylabel: str,
    title: str,
    trend: dict | None,
    config,
    output_filename: str,
) -> None:
    """
    Plot a line chart for *column* over time and save to the charts/ directory.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset with at minimum ``year`` and *column* columns.
    column : str
        Name of the value column to plot.
    ylabel : str
        Label for the y-axis.
    title : str
        Chart title.
    trend : dict | None
        When not None, a dict with keys ``slope`` and ``intercept`` used to
        overlay the linear regression trend line.
    config : module
        The project ``config`` module.  ``config.VIZ_LIBRARY`` controls
        whether matplotlib (default) or plotly is used.
    output_filename : str
        Base filename (without extension).  The PNG is saved as
        ``charts/<output_filename>.png``; when plotly is active an HTML is
        also saved as ``charts/<output_filename>.html``.

    Returns
    -------
    None
        Returns early (without creating any file) and logs a WARNING when the
        column has fewer than 2 non-NaN values.
    """
    # ------------------------------------------------------------------
    # Guard: need at least 2 non-NaN data points to draw a meaningful chart
    # ------------------------------------------------------------------
    if column not in df.columns:
        logger.warning(
            "[%s] Column '%s' not found in DataFrame — skipping chart.",
            _STEP_NAME,
            column,
        )
        return

    valid_df = df.dropna(subset=[column])
    n_valid = len(valid_df)

    if n_valid < 2:
        logger.warning(
            "[%s] Column '%s' has only %d non-NaN row(s); "
            "at least 2 are required to generate a chart — skipping.",
            _STEP_NAME,
            column,
            n_valid,
        )
        return

    # ------------------------------------------------------------------
    # Ensure charts/ directory exists
    # ------------------------------------------------------------------
    os.makedirs(_CHARTS_DIR, exist_ok=True)

    png_path = os.path.join(_CHARTS_DIR, f"{output_filename}.png")
    html_path = os.path.join(_CHARTS_DIR, f"{output_filename}.html")

    viz_library = getattr(config, "VIZ_LIBRARY", "matplotlib")

    # ------------------------------------------------------------------
    # Build trend-line data (shared by both backends)
    # ------------------------------------------------------------------
    trend_years = None
    trend_values = None
    if trend is not None:
        year_min = int(valid_df["year"].min())
        year_max = int(valid_df["year"].max())
        import numpy as np
        trend_years = np.arange(year_min, year_max + 1)
        trend_values = trend["slope"] * trend_years + trend["intercept"]

    # ------------------------------------------------------------------
    # Render chart
    # ------------------------------------------------------------------
    if viz_library == "plotly":
        _plot_plotly(
            valid_df=valid_df,
            column=column,
            ylabel=ylabel,
            title=title,
            trend_years=trend_years,
            trend_values=trend_values,
            png_path=png_path,
            html_path=html_path,
        )
    else:
        _plot_matplotlib(
            valid_df=valid_df,
            column=column,
            ylabel=ylabel,
            title=title,
            trend_years=trend_years,
            trend_values=trend_values,
            png_path=png_path,
        )

    logger.info(
        "[%s] Chart saved: '%s'.",
        _STEP_NAME,
        png_path,
    )


# ---------------------------------------------------------------------------
# Private rendering helpers
# ---------------------------------------------------------------------------

def _plot_matplotlib(
    valid_df: pd.DataFrame,
    column: str,
    ylabel: str,
    title: str,
    trend_years,
    trend_values,
    png_path: str,
) -> None:
    """Render and save the chart using matplotlib."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        valid_df["year"],
        valid_df[column],
        marker="o",
        linewidth=1.8,
        label=ylabel,
    )

    if trend_years is not None and trend_values is not None:
        ax.plot(
            trend_years,
            trend_values,
            linestyle="--",
            linewidth=1.4,
            color="red",
            label="Linear trend",
        )

    ax.set_xlabel("Year")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def _plot_plotly(
    valid_df: pd.DataFrame,
    column: str,
    ylabel: str,
    title: str,
    trend_years,
    trend_values,
    png_path: str,
    html_path: str,
) -> None:
    """Render and save the chart using plotly (PNG + HTML)."""
    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=valid_df["year"],
            y=valid_df[column],
            mode="lines+markers",
            name=ylabel,
        )
    )

    if trend_years is not None and trend_values is not None:
        fig.add_trace(
            go.Scatter(
                x=trend_years,
                y=trend_values,
                mode="lines",
                name="Linear trend",
                line=dict(dash="dash", color="red"),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Year",
        yaxis_title=ylabel,
    )

    # Save PNG via kaleido (if available); fall back gracefully
    try:
        fig.write_image(png_path)
    except Exception as exc:
        logger.warning(
            "[visualizer._plot_plotly] Could not save PNG (kaleido may be missing): %s",
            exc,
        )

    # Always save HTML
    fig.write_html(html_path)
    logger.info(
        "[visualizer._plot_plotly] HTML chart saved: '%s'.",
        html_path,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def generate_all_charts(df: pd.DataFrame, results: dict, config) -> None:
    """
    Generate all standard charts for the Ethiopia electricity analysis.

    Calls :func:`plot_indicator` for:
    - ``consumption_kwh_per_capita``
    - ``access_pct_of_population``

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset as produced by ``data_cleaner.clean_and_merge``.
    results : dict
        Analysis results dict as returned by ``trend_analyzer.analyze``,
        expected to contain a ``trends`` sub-dict with optional trend dicts
        for each value column.
    config : module
        The project ``config`` module.
    """
    trends = results.get("trends", {})

    plot_indicator(
        df=df,
        column="consumption_kwh_per_capita",
        ylabel="kWh per Capita",
        title="Ethiopia: Electric Power Consumption per Capita",
        trend=trends.get("consumption_kwh_per_capita"),
        config=config,
        output_filename="consumption_kwh_per_capita",
    )

    plot_indicator(
        df=df,
        column="access_pct_of_population",
        ylabel="% of Population",
        title="Ethiopia: Access to Electricity (% of Population)",
        trend=trends.get("access_pct_of_population"),
        config=config,
        output_filename="access_pct_of_population",
    )
