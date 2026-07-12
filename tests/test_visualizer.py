"""
tests/test_visualizer.py — Unit tests for src/visualizer.py

Tests:
  - PNG is created for >= 2 non-NaN rows
  - No file is created for 0 or 1 non-NaN row
  - HTML is also created when VIZ_LIBRARY == 'plotly'
  - Trend line overlay works (non-None trend dict raises no exception)
  - All output files are written to a temporary directory
"""

import matplotlib
matplotlib.use("Agg")  # Must be set before importing pyplot; avoids display errors

import os
import types
import pytest
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(viz_library: str = "matplotlib", charts_dir: str | None = None):
    """Return a minimal config-like namespace."""
    cfg = types.SimpleNamespace()
    cfg.VIZ_LIBRARY = viz_library
    # charts_dir is passed separately via monkeypatching _CHARTS_DIR
    return cfg


def _make_df(n_valid: int, n_nan: int = 0) -> pd.DataFrame:
    """
    Build a small cleaned DataFrame with *n_valid* real rows and *n_nan* NaN rows.
    """
    years = list(range(2000, 2000 + n_valid + n_nan))
    consumption = [float(i * 10) for i in range(n_valid)] + [float("nan")] * n_nan
    access = [float(50 + i) for i in range(n_valid)] + [float("nan")] * n_nan
    return pd.DataFrame(
        {
            "year": years,
            "consumption_kwh_per_capita": consumption,
            "access_pct_of_population": access,
        }
    )


def _make_trend() -> dict:
    return {"slope": 1.5, "intercept": -2900.0, "r_squared": 0.95}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_charts_dir(tmp_path, monkeypatch):
    """Redirect all chart output to a pytest-managed temp directory."""
    import src.visualizer as viz_module
    monkeypatch.setattr(viz_module, "_CHARTS_DIR", str(tmp_path))
    return tmp_path


# ---------------------------------------------------------------------------
# Tests: PNG creation
# ---------------------------------------------------------------------------

class TestPNGCreation:

    def test_png_created_for_two_valid_rows(self, tmp_path):
        """A PNG file should be created when the column has exactly 2 non-NaN rows."""
        from src.visualizer import plot_indicator

        df = _make_df(n_valid=2)
        cfg = _make_config()
        plot_indicator(df, "consumption_kwh_per_capita", "kWh per Capita",
                       "Test Title", None, cfg, "test_out")

        assert os.path.isfile(os.path.join(str(tmp_path), "test_out.png"))

    def test_png_created_for_many_valid_rows(self, tmp_path):
        """A PNG file should be created when the column has many non-NaN rows."""
        from src.visualizer import plot_indicator

        df = _make_df(n_valid=10)
        cfg = _make_config()
        plot_indicator(df, "consumption_kwh_per_capita", "kWh per Capita",
                       "Test Title", None, cfg, "test_many")

        assert os.path.isfile(os.path.join(str(tmp_path), "test_many.png"))

    def test_no_file_created_for_zero_valid_rows(self, tmp_path):
        """No PNG should be created when all column values are NaN."""
        from src.visualizer import plot_indicator

        df = _make_df(n_valid=0, n_nan=5)
        cfg = _make_config()
        plot_indicator(df, "consumption_kwh_per_capita", "kWh per Capita",
                       "Test Title", None, cfg, "test_zero")

        assert not os.path.isfile(os.path.join(str(tmp_path), "test_zero.png"))

    def test_no_file_created_for_one_valid_row(self, tmp_path):
        """No PNG should be created when only 1 non-NaN value exists."""
        from src.visualizer import plot_indicator

        # Build a DataFrame with exactly 1 valid row and some NaN rows
        df = pd.DataFrame(
            {
                "year": [2000, 2001, 2002],
                "consumption_kwh_per_capita": [float("nan"), 42.0, float("nan")],
                "access_pct_of_population": [float("nan"), float("nan"), float("nan")],
            }
        )
        cfg = _make_config()
        plot_indicator(df, "consumption_kwh_per_capita", "kWh per Capita",
                       "Test Title", None, cfg, "test_one")

        assert not os.path.isfile(os.path.join(str(tmp_path), "test_one.png"))

    def test_no_file_created_for_entirely_nan_column(self, tmp_path):
        """No PNG should be created when the entire column is NaN."""
        from src.visualizer import plot_indicator

        df = pd.DataFrame(
            {
                "year": [2000, 2001, 2002],
                "consumption_kwh_per_capita": [float("nan"), float("nan"), float("nan")],
                "access_pct_of_population": [50.0, 55.0, 60.0],
            }
        )
        cfg = _make_config()
        plot_indicator(df, "consumption_kwh_per_capita", "kWh per Capita",
                       "Test Title", None, cfg, "test_all_nan")

        assert not os.path.isfile(os.path.join(str(tmp_path), "test_all_nan.png"))


# ---------------------------------------------------------------------------
# Tests: HTML creation (plotly)
# ---------------------------------------------------------------------------

class TestHTMLCreation:

    def test_html_created_when_plotly_selected(self, tmp_path):
        """When VIZ_LIBRARY == 'plotly', an HTML file should also be saved."""
        from src.visualizer import plot_indicator

        df = _make_df(n_valid=5)
        cfg = _make_config(viz_library="plotly")
        plot_indicator(df, "consumption_kwh_per_capita", "kWh per Capita",
                       "Test Title", None, cfg, "test_plotly")

        assert os.path.isfile(os.path.join(str(tmp_path), "test_plotly.html"))

    def test_html_not_created_when_matplotlib_selected(self, tmp_path):
        """No HTML file should be created when VIZ_LIBRARY == 'matplotlib'."""
        from src.visualizer import plot_indicator

        df = _make_df(n_valid=5)
        cfg = _make_config(viz_library="matplotlib")
        plot_indicator(df, "consumption_kwh_per_capita", "kWh per Capita",
                       "Test Title", None, cfg, "test_mpl_no_html")

        assert not os.path.isfile(os.path.join(str(tmp_path), "test_mpl_no_html.html"))


# ---------------------------------------------------------------------------
# Tests: Trend line overlay
# ---------------------------------------------------------------------------

class TestTrendOverlay:

    def test_trend_overlay_matplotlib_no_exception(self, tmp_path):
        """Passing a non-None trend dict with matplotlib should not raise any exception."""
        from src.visualizer import plot_indicator

        df = _make_df(n_valid=8)
        cfg = _make_config(viz_library="matplotlib")
        trend = _make_trend()

        # Should complete without raising
        plot_indicator(df, "consumption_kwh_per_capita", "kWh per Capita",
                       "Test with Trend", trend, cfg, "test_trend_mpl")

        assert os.path.isfile(os.path.join(str(tmp_path), "test_trend_mpl.png"))

    def test_trend_overlay_plotly_no_exception(self, tmp_path):
        """Passing a non-None trend dict with plotly should not raise any exception."""
        from src.visualizer import plot_indicator

        df = _make_df(n_valid=8)
        cfg = _make_config(viz_library="plotly")
        trend = _make_trend()

        # Should complete without raising
        plot_indicator(df, "access_pct_of_population", "% of Population",
                       "Test with Trend (plotly)", trend, cfg, "test_trend_plotly")

        assert os.path.isfile(os.path.join(str(tmp_path), "test_trend_plotly.html"))

    def test_no_trend_overlay_when_none(self, tmp_path):
        """Passing trend=None should still produce a PNG without errors."""
        from src.visualizer import plot_indicator

        df = _make_df(n_valid=6)
        cfg = _make_config()

        plot_indicator(df, "access_pct_of_population", "% of Population",
                       "No Trend", None, cfg, "test_no_trend")

        assert os.path.isfile(os.path.join(str(tmp_path), "test_no_trend.png"))


# ---------------------------------------------------------------------------
# Tests: generate_all_charts orchestrator
# ---------------------------------------------------------------------------

class TestGenerateAllCharts:

    def test_generate_all_charts_creates_both_pngs(self, tmp_path):
        """generate_all_charts should produce PNG files for both indicators."""
        from src.visualizer import generate_all_charts

        df = _make_df(n_valid=10)
        results = {
            "trends": {
                "consumption_kwh_per_capita": _make_trend(),
                "access_pct_of_population": _make_trend(),
            }
        }
        cfg = _make_config()

        generate_all_charts(df, results, cfg)

        assert os.path.isfile(
            os.path.join(str(tmp_path), "consumption_kwh_per_capita.png")
        )
        assert os.path.isfile(
            os.path.join(str(tmp_path), "access_pct_of_population.png")
        )

    def test_generate_all_charts_no_trends(self, tmp_path):
        """generate_all_charts should work even when trends are None."""
        from src.visualizer import generate_all_charts

        df = _make_df(n_valid=5)
        results = {
            "trends": {
                "consumption_kwh_per_capita": None,
                "access_pct_of_population": None,
            }
        }
        cfg = _make_config()

        generate_all_charts(df, results, cfg)

        assert os.path.isfile(
            os.path.join(str(tmp_path), "consumption_kwh_per_capita.png")
        )
        assert os.path.isfile(
            os.path.join(str(tmp_path), "access_pct_of_population.png")
        )

    def test_generate_all_charts_skips_when_insufficient_data(self, tmp_path):
        """generate_all_charts should skip chart creation when data is insufficient."""
        from src.visualizer import generate_all_charts

        df = _make_df(n_valid=0, n_nan=3)
        results = {
            "trends": {
                "consumption_kwh_per_capita": None,
                "access_pct_of_population": None,
            }
        }
        cfg = _make_config()

        generate_all_charts(df, results, cfg)

        assert not os.path.isfile(
            os.path.join(str(tmp_path), "consumption_kwh_per_capita.png")
        )
        assert not os.path.isfile(
            os.path.join(str(tmp_path), "access_pct_of_population.png")
        )


# ---------------------------------------------------------------------------
# Property-Based Tests (Hypothesis)
# ---------------------------------------------------------------------------

# Feature: ethiopia-electricity-analysis, Property 7: Chart output file is created if and only if at least 2 non-NaN data points exist

import tempfile
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st


@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
@given(
    # Generate a list of optional float values representing the column data.
    # Each element is either a finite float or None (representing NaN).
    values=st.lists(
        st.one_of(
            st.just(None),                                         # NaN slot
            st.floats(min_value=0.0, max_value=1e6,
                      allow_nan=False, allow_infinity=False),      # valid value
        ),
        min_size=0,
        max_size=30,
    )
)
def test_property_7_chart_created_iff_two_or_more_non_nan(values):
    """
    Property 7: plot_indicator creates a PNG iff >= 2 non-NaN values exist.

    Validates: Requirements 5.4, 5.5
    """
    import src.visualizer as viz_module
    from src.visualizer import plot_indicator

    # Build a DataFrame from the generated values list
    n = len(values)
    years = list(range(2000, 2000 + n)) if n > 0 else []
    col_values = [float("nan") if v is None else v for v in values]

    df = pd.DataFrame({
        "year": years,
        "value_col": col_values,
    })

    n_non_nan = sum(1 for v in col_values if v == v)  # NaN != NaN, so v==v is False for NaN

    cfg = _make_config(viz_library="matplotlib")
    output_name = "prop7_test"

    # Use a fresh temporary directory per example so each run starts clean
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Patch the module-level _CHARTS_DIR so plot_indicator writes here
        original_charts_dir = viz_module._CHARTS_DIR
        viz_module._CHARTS_DIR = tmp_dir
        try:
            plot_indicator(df, "value_col", "Value", "Property 7 Test", None, cfg, output_name)
            png_path = os.path.join(tmp_dir, f"{output_name}.png")

            if n_non_nan >= 2:
                assert os.path.isfile(png_path), (
                    f"Expected PNG to be created for {n_non_nan} non-NaN values "
                    f"(values={values}), but no file was found."
                )
            else:
                assert not os.path.isfile(png_path), (
                    f"Expected NO PNG for {n_non_nan} non-NaN values "
                    f"(values={values}), but a file was unexpectedly created."
                )
        finally:
            viz_module._CHARTS_DIR = original_charts_dir
