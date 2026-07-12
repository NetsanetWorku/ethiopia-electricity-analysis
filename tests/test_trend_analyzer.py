"""
Unit tests for src/trend_analyzer.py — Task 5.1: compute_summary
"""

import math

import numpy as np
import pandas as pd
import pytest

from src.trend_analyzer import compute_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_df(consumption, access, years=None):
    """Build a minimal cleaned DataFrame for testing."""
    n = max(len(consumption), len(access))
    if years is None:
        years = list(range(2000, 2000 + n))
    return pd.DataFrame(
        {
            "year": years,
            "consumption_kwh_per_capita": pd.array(consumption, dtype=float),
            "access_pct_of_population": pd.array(access, dtype=float),
        }
    )


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

class TestComputeSummaryHappyPath:
    def test_returns_both_column_keys(self):
        df = make_df([10.0, 20.0, 30.0], [50.0, 60.0, 70.0])
        result = compute_summary(df)
        assert "consumption_kwh_per_capita" in result
        assert "access_pct_of_population" in result

    def test_mean(self):
        df = make_df([10.0, 20.0, 30.0], [50.0, 60.0, 70.0])
        result = compute_summary(df)
        assert math.isclose(result["consumption_kwh_per_capita"]["mean"], 20.0)
        assert math.isclose(result["access_pct_of_population"]["mean"], 60.0)

    def test_median(self):
        df = make_df([10.0, 20.0, 30.0, 40.0], [5.0, 15.0, 25.0, 35.0])
        result = compute_summary(df)
        assert math.isclose(result["consumption_kwh_per_capita"]["median"], 25.0)
        assert math.isclose(result["access_pct_of_population"]["median"], 20.0)

    def test_min_max(self):
        df = make_df([10.0, 20.0, 30.0], [50.0, 60.0, 70.0])
        result = compute_summary(df)
        assert math.isclose(result["consumption_kwh_per_capita"]["min"], 10.0)
        assert math.isclose(result["consumption_kwh_per_capita"]["max"], 30.0)
        assert math.isclose(result["access_pct_of_population"]["min"], 50.0)
        assert math.isclose(result["access_pct_of_population"]["max"], 70.0)

    def test_std(self):
        values = [10.0, 20.0, 30.0]
        df = make_df(values, [1.0, 2.0, 3.0])
        result = compute_summary(df)
        expected_std = float(pd.Series(values).std())
        assert math.isclose(result["consumption_kwh_per_capita"]["std"], expected_std)

    def test_peak_year(self):
        df = make_df([10.0, 50.0, 30.0], [5.0, 15.0, 25.0], years=[2000, 2001, 2002])
        result = compute_summary(df)
        assert result["consumption_kwh_per_capita"]["peak_year"] == 2001
        assert result["access_pct_of_population"]["peak_year"] == 2002

    def test_low_year(self):
        df = make_df([10.0, 50.0, 30.0], [5.0, 15.0, 25.0], years=[2000, 2001, 2002])
        result = compute_summary(df)
        assert result["consumption_kwh_per_capita"]["low_year"] == 2000
        assert result["access_pct_of_population"]["low_year"] == 2000

    def test_nan_count_zero_when_no_nans(self):
        df = make_df([10.0, 20.0, 30.0], [1.0, 2.0, 3.0])
        result = compute_summary(df)
        assert result["consumption_kwh_per_capita"]["nan_count"] == 0
        assert result["access_pct_of_population"]["nan_count"] == 0


# ---------------------------------------------------------------------------
# NaN handling tests — requirement 8.3
# ---------------------------------------------------------------------------

class TestComputeSummaryNaNHandling:
    def test_nan_count_reflects_missing_rows(self):
        df = make_df([10.0, float("nan"), 30.0], [1.0, 2.0, float("nan")])
        result = compute_summary(df)
        assert result["consumption_kwh_per_capita"]["nan_count"] == 1
        assert result["access_pct_of_population"]["nan_count"] == 1

    def test_stats_computed_on_non_nan_rows(self):
        """NaN rows must be excluded from statistics computation."""
        df = make_df([10.0, float("nan"), 30.0], [1.0, 2.0, 3.0])
        result = compute_summary(df)
        # Only 10.0 and 30.0 should be used
        assert math.isclose(result["consumption_kwh_per_capita"]["mean"], 20.0)
        assert math.isclose(result["consumption_kwh_per_capita"]["min"], 10.0)
        assert math.isclose(result["consumption_kwh_per_capita"]["max"], 30.0)

    def test_peak_and_low_year_ignore_nan(self):
        """peak_year and low_year must only consider non-NaN rows."""
        df = make_df(
            [100.0, float("nan"), 50.0],
            [float("nan"), 70.0, 30.0],
            years=[2000, 2001, 2002],
        )
        result = compute_summary(df)
        # consumption: valid rows are year=2000 (100.0) and year=2002 (50.0)
        assert result["consumption_kwh_per_capita"]["peak_year"] == 2000
        assert result["consumption_kwh_per_capita"]["low_year"] == 2002
        # access: valid rows are year=2001 (70.0) and year=2002 (30.0)
        assert result["access_pct_of_population"]["peak_year"] == 2001
        assert result["access_pct_of_population"]["low_year"] == 2002

    def test_all_nan_column_returns_nan_stats(self):
        df = make_df([float("nan"), float("nan")], [10.0, 20.0])
        result = compute_summary(df)
        col_stats = result["consumption_kwh_per_capita"]
        assert math.isnan(col_stats["mean"])
        assert math.isnan(col_stats["min"])
        assert math.isnan(col_stats["max"])
        assert math.isnan(col_stats["std"])
        assert col_stats["peak_year"] is None
        assert col_stats["low_year"] is None
        assert col_stats["nan_count"] == 2


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

class TestComputeSummaryEdgeCases:
    def test_single_row(self):
        df = make_df([42.0], [88.0], years=[2010])
        result = compute_summary(df)
        c = result["consumption_kwh_per_capita"]
        assert math.isclose(c["mean"], 42.0)
        assert math.isclose(c["min"], 42.0)
        assert math.isclose(c["max"], 42.0)
        assert math.isclose(c["median"], 42.0)
        # std of a single element is NaN with ddof=1
        assert math.isnan(c["std"])
        assert c["peak_year"] == 2010
        assert c["low_year"] == 2010
        assert c["nan_count"] == 0

    def test_missing_column_is_skipped(self):
        """If a value column is absent from the DataFrame, it should not appear in result."""
        df = pd.DataFrame(
            {
                "year": [2000, 2001],
                "consumption_kwh_per_capita": [10.0, 20.0],
                # 'access_pct_of_population' is intentionally absent
            }
        )
        result = compute_summary(df)
        assert "consumption_kwh_per_capita" in result
        assert "access_pct_of_population" not in result

    def test_empty_dataframe_returns_empty_dict_or_nan_stats(self):
        """An empty DataFrame has no valid rows for any column."""
        df = pd.DataFrame(
            {
                "year": pd.Series([], dtype=int),
                "consumption_kwh_per_capita": pd.Series([], dtype=float),
                "access_pct_of_population": pd.Series([], dtype=float),
            }
        )
        result = compute_summary(df)
        for col in ("consumption_kwh_per_capita", "access_pct_of_population"):
            assert col in result
            assert math.isnan(result[col]["mean"])
            assert result[col]["nan_count"] == 0
            assert result[col]["peak_year"] is None
            assert result[col]["low_year"] is None

    def test_many_nans_with_one_valid_value(self):
        """Only one non-NaN value — stats should still work (std is NaN with ddof=1)."""
        df = make_df(
            [float("nan"), float("nan"), 99.0],
            [float("nan"), float("nan"), float("nan")],
            years=[2000, 2001, 2002],
        )
        result = compute_summary(df)
        c = result["consumption_kwh_per_capita"]
        assert math.isclose(c["mean"], 99.0)
        assert math.isclose(c["min"], 99.0)
        assert math.isclose(c["max"], 99.0)
        assert c["peak_year"] == 2002
        assert c["low_year"] == 2002
        assert c["nan_count"] == 2


# ---------------------------------------------------------------------------
# Tests for compute_yoy_change — Task 5.2
# ---------------------------------------------------------------------------

from src.trend_analyzer import compute_yoy_change


class TestComputeYoYChangeHappyPath:
    def test_returns_new_dataframe_not_in_place(self):
        """compute_yoy_change must return a copy, not mutate the original."""
        df = make_df([100.0, 120.0, 90.0], [40.0, 50.0, 60.0])
        result = compute_yoy_change(df)
        assert result is not df
        assert "consumption_yoy_pct" not in df.columns
        assert "access_yoy_pct" not in df.columns

    def test_yoy_columns_added(self):
        """Both YoY columns should be present in the result."""
        df = make_df([100.0, 120.0], [40.0, 50.0])
        result = compute_yoy_change(df)
        assert "consumption_yoy_pct" in result.columns
        assert "access_yoy_pct" in result.columns

    def test_basic_formula_positive_change(self):
        """YoY: (120 - 100) / abs(100) * 100 = 20.0"""
        df = make_df([100.0, 120.0], [40.0, 60.0])
        result = compute_yoy_change(df)
        assert math.isclose(result["consumption_yoy_pct"].iloc[1], 20.0)
        assert math.isclose(result["access_yoy_pct"].iloc[1], 50.0)

    def test_basic_formula_negative_change(self):
        """YoY: (80 - 100) / abs(100) * 100 = -20.0"""
        df = make_df([100.0, 80.0], [50.0, 40.0])
        result = compute_yoy_change(df)
        assert math.isclose(result["consumption_yoy_pct"].iloc[1], -20.0)
        assert math.isclose(result["access_yoy_pct"].iloc[1], -20.0)

    def test_three_row_chain(self):
        """Verify formula across a 3-row sequence."""
        df = make_df([100.0, 200.0, 150.0], [10.0, 10.0, 10.0])
        result = compute_yoy_change(df)
        # Row 1: (200 - 100) / 100 * 100 = 100.0
        assert math.isclose(result["consumption_yoy_pct"].iloc[1], 100.0)
        # Row 2: (150 - 200) / 200 * 100 = -25.0
        assert math.isclose(result["consumption_yoy_pct"].iloc[2], -25.0)


class TestComputeYoYChangeFirstRowIsNaN:
    def test_first_row_consumption_yoy_is_nan(self):
        df = make_df([100.0, 120.0, 140.0], [50.0, 60.0, 70.0])
        result = compute_yoy_change(df)
        assert math.isnan(result["consumption_yoy_pct"].iloc[0])

    def test_first_row_access_yoy_is_nan(self):
        df = make_df([100.0, 120.0], [50.0, 60.0])
        result = compute_yoy_change(df)
        assert math.isnan(result["access_yoy_pct"].iloc[0])

    def test_single_row_both_nan(self):
        df = make_df([100.0], [50.0], years=[2010])
        result = compute_yoy_change(df)
        assert math.isnan(result["consumption_yoy_pct"].iloc[0])
        assert math.isnan(result["access_yoy_pct"].iloc[0])


class TestComputeYoYChangeNaNHandling:
    def test_nan_in_source_is_skipped(self):
        """A NaN source row should produce NaN yoy and not break the chain."""
        df = make_df([100.0, float("nan"), 150.0], [1.0, 2.0, 3.0])
        result = compute_yoy_change(df)
        # Row 0: first non-NaN → NaN
        assert math.isnan(result["consumption_yoy_pct"].iloc[0])
        # Row 1: source is NaN → yoy is NaN
        assert math.isnan(result["consumption_yoy_pct"].iloc[1])
        # Row 2: previous non-NaN is 100.0 → (150 - 100) / 100 * 100 = 50.0
        assert math.isclose(result["consumption_yoy_pct"].iloc[2], 50.0)

    def test_leading_nan_in_source(self):
        """When the first row is NaN, the first valid row is still NaN (no prior value)."""
        df = make_df([float("nan"), 100.0, 110.0], [1.0, 2.0, 3.0])
        result = compute_yoy_change(df)
        # Row 0: source NaN → yoy NaN
        assert math.isnan(result["consumption_yoy_pct"].iloc[0])
        # Row 1: first valid value, no prior non-NaN → yoy NaN
        assert math.isnan(result["consumption_yoy_pct"].iloc[1])
        # Row 2: (110 - 100) / 100 * 100 = 10.0
        assert math.isclose(result["consumption_yoy_pct"].iloc[2], 10.0)

    def test_multiple_consecutive_nans_skipped(self):
        """Multiple consecutive NaN rows are all skipped; next valid row uses last valid as base."""
        df = make_df(
            [100.0, float("nan"), float("nan"), 200.0],
            [1.0, 2.0, 3.0, 4.0],
        )
        result = compute_yoy_change(df)
        assert math.isnan(result["consumption_yoy_pct"].iloc[0])
        assert math.isnan(result["consumption_yoy_pct"].iloc[1])
        assert math.isnan(result["consumption_yoy_pct"].iloc[2])
        # Row 3: previous non-NaN is 100.0 → (200 - 100) / 100 * 100 = 100.0
        assert math.isclose(result["consumption_yoy_pct"].iloc[3], 100.0)

    def test_all_nan_source_column_all_nan_yoy(self):
        """All-NaN source column → all-NaN YoY column."""
        df = make_df([float("nan"), float("nan"), float("nan")], [1.0, 2.0, 3.0])
        result = compute_yoy_change(df)
        assert result["consumption_yoy_pct"].isna().all()


class TestComputeYoYChangeDivisionByZero:
    def test_zero_previous_value_yields_nan(self):
        """When previous non-NaN value is 0, YoY for the next row should be NaN."""
        df = make_df([0.0, 100.0], [1.0, 2.0])
        result = compute_yoy_change(df)
        # Row 0: first value → NaN
        assert math.isnan(result["consumption_yoy_pct"].iloc[0])
        # Row 1: prev=0, division by zero → NaN (not inf)
        assert math.isnan(result["consumption_yoy_pct"].iloc[1])
        assert not math.isinf(result["consumption_yoy_pct"].iloc[1])

    def test_zero_previous_value_does_not_raise(self):
        """Zero previous value must not raise an exception."""
        df = make_df([0.0, 50.0, 75.0], [1.0, 2.0, 3.0])
        result = compute_yoy_change(df)  # Should not raise
        assert math.isnan(result["consumption_yoy_pct"].iloc[1])
        # Row 2 uses prev=50.0: (75 - 50) / 50 * 100 = 50.0
        assert math.isclose(result["consumption_yoy_pct"].iloc[2], 50.0)

    def test_zero_in_access_column(self):
        """Same zero-division guard works for access_pct_of_population."""
        df = make_df([1.0, 2.0, 3.0], [0.0, 80.0, 90.0])
        result = compute_yoy_change(df)
        assert math.isnan(result["access_yoy_pct"].iloc[0])
        assert math.isnan(result["access_yoy_pct"].iloc[1])
        # Row 2: prev=80.0 → (90 - 80) / 80 * 100 = 12.5
        assert math.isclose(result["access_yoy_pct"].iloc[2], 12.5)


class TestComputeYoYChangeMissingColumn:
    def test_missing_source_column_not_in_result(self):
        """If a source column is absent, its YoY column should also be absent."""
        df = pd.DataFrame(
            {
                "year": [2000, 2001],
                "consumption_kwh_per_capita": [100.0, 120.0],
                # 'access_pct_of_population' intentionally absent
            }
        )
        result = compute_yoy_change(df)
        assert "consumption_yoy_pct" in result.columns
        assert "access_yoy_pct" not in result.columns


# ---------------------------------------------------------------------------
# Tests for fit_linear_trend — Task 5.3
# ---------------------------------------------------------------------------

from src.trend_analyzer import fit_linear_trend


class TestFitLinearTrendReturnNone:
    def test_exactly_4_non_nan_rows_returns_none(self):
        """Exactly 4 non-NaN rows must return None (below the 5-row threshold)."""
        df = make_df(
            [10.0, 20.0, 30.0, 40.0],
            [1.0, 2.0, 3.0, 4.0],
            years=[2000, 2001, 2002, 2003],
        )
        result = fit_linear_trend(df, "consumption_kwh_per_capita")
        assert result is None

    def test_exactly_4_non_nan_rows_logs_warning(self, caplog):
        """Exactly 4 non-NaN rows must emit a WARNING log."""
        import logging
        df = make_df(
            [10.0, 20.0, 30.0, 40.0],
            [1.0, 2.0, 3.0, 4.0],
            years=[2000, 2001, 2002, 2003],
        )
        with caplog.at_level(logging.WARNING, logger="src.trend_analyzer"):
            fit_linear_trend(df, "consumption_kwh_per_capita")
        assert any("WARNING" in r.levelname or r.levelno >= logging.WARNING
                   for r in caplog.records)

    def test_all_nan_column_returns_none(self):
        """A column with all NaN values must return None."""
        df = make_df(
            [float("nan"), float("nan"), float("nan"), float("nan"), float("nan")],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            years=[2000, 2001, 2002, 2003, 2004],
        )
        result = fit_linear_trend(df, "consumption_kwh_per_capita")
        assert result is None

    def test_zero_rows_returns_none(self):
        """Empty DataFrame must return None."""
        df = pd.DataFrame(
            {
                "year": pd.Series([], dtype=int),
                "consumption_kwh_per_capita": pd.Series([], dtype=float),
                "access_pct_of_population": pd.Series([], dtype=float),
            }
        )
        result = fit_linear_trend(df, "consumption_kwh_per_capita")
        assert result is None


class TestFitLinearTrendReturnsDict:
    def test_exactly_5_non_nan_rows_returns_dict(self):
        """Exactly 5 non-NaN rows must return a non-None dict."""
        df = make_df(
            [10.0, 20.0, 30.0, 40.0, 50.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            years=[2000, 2001, 2002, 2003, 2004],
        )
        result = fit_linear_trend(df, "consumption_kwh_per_capita")
        assert result is not None

    def test_result_dict_has_correct_keys(self):
        """Returned dict must contain exactly slope, intercept, and r_squared."""
        df = make_df(
            [10.0, 20.0, 30.0, 40.0, 50.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            years=[2000, 2001, 2002, 2003, 2004],
        )
        result = fit_linear_trend(df, "consumption_kwh_per_capita")
        assert set(result.keys()) == {"slope", "intercept", "r_squared"}

    def test_r_squared_between_0_and_1(self):
        """r_squared must be in [0.0, 1.0] for any valid regression."""
        df = make_df(
            [10.0, 22.0, 31.0, 45.0, 50.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
            years=[2000, 2001, 2002, 2003, 2004],
        )
        result = fit_linear_trend(df, "consumption_kwh_per_capita")
        assert 0.0 <= result["r_squared"] <= 1.0

    def test_perfect_linear_data_r_squared_is_1(self):
        """A perfectly linear sequence must give r_squared == 1.0."""
        years = [2000, 2001, 2002, 2003, 2004]
        values = [float(10 * (y - 2000) + 5) for y in years]  # 5, 15, 25, 35, 45
        df = make_df(values, [1.0] * 5, years=years)
        result = fit_linear_trend(df, "consumption_kwh_per_capita")
        assert math.isclose(result["r_squared"], 1.0, rel_tol=1e-9)

    def test_slope_and_intercept_values(self):
        """Slope and intercept must match scipy reference values."""
        from scipy.stats import linregress as _lr
        years = [2000, 2001, 2002, 2003, 2004]
        values = [10.0, 22.0, 31.0, 45.0, 50.0]
        df = make_df(values, [1.0] * 5, years=years)
        result = fit_linear_trend(df, "consumption_kwh_per_capita")
        ref = _lr(years, values)
        assert math.isclose(result["slope"], ref.slope, rel_tol=1e-9)
        assert math.isclose(result["intercept"], ref.intercept, rel_tol=1e-9)

    def test_nans_in_column_are_excluded_from_regression(self):
        """NaN rows must be dropped before fitting; only non-NaN rows are used."""
        years = [2000, 2001, 2002, 2003, 2004, 2005]
        # Insert a NaN at position 2; valid rows = 5
        values = [10.0, 20.0, float("nan"), 40.0, 50.0, 60.0]
        df = make_df(values, [1.0] * 6, years=years)
        result = fit_linear_trend(df, "consumption_kwh_per_capita")
        # Should return a dict (5 valid rows remain)
        assert result is not None
        assert set(result.keys()) == {"slope", "intercept", "r_squared"}
        assert 0.0 <= result["r_squared"] <= 1.0


# ---------------------------------------------------------------------------
# Tests for analyze — Task 5.4
# ---------------------------------------------------------------------------

import os
import types

from src.trend_analyzer import analyze


def _make_config():
    """Return a minimal config stub with PATHS."""
    cfg = types.SimpleNamespace()
    cfg.PATHS = {
        "raw": "data/raw",
        "processed": "data/processed",
        "charts": "charts",
        "reports": "reports",
        "notebook": "notebooks",
    }
    return cfg


def make_analyze_df():
    """Build a reasonably-sized DataFrame suitable for all analyze sub-steps."""
    years = list(range(2000, 2010))  # 10 years → enough for regression
    consumption = [100.0 + i * 10 for i in range(10)]
    access = [20.0 + i * 5 for i in range(10)]
    return pd.DataFrame(
        {
            "year": years,
            "consumption_kwh_per_capita": pd.array(consumption, dtype=float),
            "access_pct_of_population": pd.array(access, dtype=float),
        }
    )


class TestAnalyzeReturnStructure:
    def test_returns_dict_with_required_keys(self):
        """analyze must return a dict containing 'summary', 'trends', 'df_with_yoy'."""
        df = make_analyze_df()
        result = analyze(df, _make_config())
        assert isinstance(result, dict)
        assert "summary" in result
        assert "trends" in result
        assert "df_with_yoy" in result

    def test_summary_contains_both_column_keys(self):
        """result['summary'] must have keys for both indicator columns."""
        df = make_analyze_df()
        result = analyze(df, _make_config())
        assert "consumption_kwh_per_capita" in result["summary"]
        assert "access_pct_of_population" in result["summary"]

    def test_trends_contains_both_column_keys(self):
        """result['trends'] must have keys for both indicator columns."""
        df = make_analyze_df()
        result = analyze(df, _make_config())
        assert "consumption_kwh_per_capita" in result["trends"]
        assert "access_pct_of_population" in result["trends"]

    def test_df_with_yoy_is_dataframe(self):
        """result['df_with_yoy'] must be a pandas DataFrame."""
        df = make_analyze_df()
        result = analyze(df, _make_config())
        assert isinstance(result["df_with_yoy"], pd.DataFrame)

    def test_df_with_yoy_has_yoy_columns(self):
        """df_with_yoy must contain both YoY percentage-change columns."""
        df = make_analyze_df()
        result = analyze(df, _make_config())
        yoy_df = result["df_with_yoy"]
        assert "consumption_yoy_pct" in yoy_df.columns
        assert "access_yoy_pct" in yoy_df.columns


class TestAnalyzeSummaryCSV:
    def test_summary_csv_is_created(self, tmp_path, monkeypatch):
        """analyze must write ethiopia_electricity_summary.csv to data/processed/."""
        # Point the resolved processed dir to tmp_path so we don't pollute the repo
        # We monkeypatch os.makedirs and pd.DataFrame.to_csv via checking the file
        # directly in the real project folder (simpler and more realistic).
        df = make_analyze_df()
        analyze(df, _make_config())

        src_dir = os.path.dirname(os.path.abspath(
            __import__("src.trend_analyzer", fromlist=["trend_analyzer"]).__file__
        ))
        project_root = os.path.join(src_dir, "..")
        summary_path = os.path.normpath(
            os.path.join(project_root, "data", "processed", "ethiopia_electricity_summary.csv")
        )
        assert os.path.isfile(summary_path), f"Summary CSV not found at {summary_path}"

    def test_summary_csv_has_correct_columns(self):
        """The saved CSV must contain 'metric', 'consumption_kwh_per_capita', 'access_pct_of_population'."""
        df = make_analyze_df()
        analyze(df, _make_config())

        import importlib
        mod = importlib.import_module("src.trend_analyzer")
        src_dir = os.path.dirname(os.path.abspath(mod.__file__))
        project_root = os.path.join(src_dir, "..")
        summary_path = os.path.normpath(
            os.path.join(project_root, "data", "processed", "ethiopia_electricity_summary.csv")
        )
        loaded = pd.read_csv(summary_path)
        assert "metric" in loaded.columns
        assert "consumption_kwh_per_capita" in loaded.columns
        assert "access_pct_of_population" in loaded.columns

    def test_summary_csv_has_correct_metric_rows(self):
        """The saved CSV must have a row for each expected metric name."""
        df = make_analyze_df()
        analyze(df, _make_config())

        import importlib
        mod = importlib.import_module("src.trend_analyzer")
        src_dir = os.path.dirname(os.path.abspath(mod.__file__))
        project_root = os.path.join(src_dir, "..")
        summary_path = os.path.normpath(
            os.path.join(project_root, "data", "processed", "ethiopia_electricity_summary.csv")
        )
        loaded = pd.read_csv(summary_path)
        expected_metrics = {"mean", "median", "min", "max", "std", "nan_count", "peak_year", "low_year"}
        assert expected_metrics == set(loaded["metric"].tolist())


class TestAnalyzeTrendsValues:
    def test_trends_are_dict_or_none(self):
        """Each value in result['trends'] must be a dict or None."""
        df = make_analyze_df()
        result = analyze(df, _make_config())
        for col, trend in result["trends"].items():
            assert trend is None or isinstance(trend, dict), (
                f"Trend for '{col}' must be dict or None, got {type(trend)}"
            )

    def test_trends_dict_has_regression_keys(self):
        """When a trend dict is returned it must have slope, intercept, r_squared."""
        df = make_analyze_df()
        result = analyze(df, _make_config())
        for col, trend in result["trends"].items():
            if trend is not None:
                assert set(trend.keys()) == {"slope", "intercept", "r_squared"}

    def test_trends_none_when_insufficient_data(self):
        """With fewer than 5 non-NaN rows, fit_linear_trend returns None."""
        df = make_df(
            [10.0, 20.0, 30.0, 40.0],  # only 4 rows
            [1.0, 2.0, 3.0, 4.0],
            years=[2000, 2001, 2002, 2003],
        )
        result = analyze(df, _make_config())
        assert result["trends"]["consumption_kwh_per_capita"] is None
        assert result["trends"]["access_pct_of_population"] is None


# ---------------------------------------------------------------------------
# Property-Based Tests — Task 5.7
# ---------------------------------------------------------------------------

from hypothesis import given, settings, assume
from hypothesis import strategies as st


# Feature: ethiopia-electricity-analysis, Property 6: Linear regression is returned if and only if at least 5 non-NaN data points exist
@settings(max_examples=100, deadline=None)
@given(
    non_nan_count=st.integers(min_value=0, max_value=20),
    nan_count=st.integers(min_value=0, max_value=10),
)
def test_property_6_regression_threshold(non_nan_count, nan_count):
    """
    Property 6: fit_linear_trend returns a non-None dict iff the column has >= 5
    non-NaN values; returns None iff the column has < 5 non-NaN values.

    Validates: Requirements 4.4, 4.5
    """
    total_rows = non_nan_count + nan_count
    assume(total_rows >= 1)  # need at least one row to build a DataFrame

    # Build year column (unique integers starting from 2000)
    years = list(range(2000, 2000 + total_rows))

    # Build the value column: first non_nan_count real values, then nan_count NaNs
    values = [float(i * 10 + 5) for i in range(non_nan_count)] + [float("nan")] * nan_count

    df = pd.DataFrame(
        {
            "year": years,
            "consumption_kwh_per_capita": pd.array(values, dtype=float),
        }
    )

    result = fit_linear_trend(df, "consumption_kwh_per_capita")

    if non_nan_count >= 5:
        # Must return a non-None dict with the three required keys
        assert result is not None, (
            f"Expected non-None dict for {non_nan_count} non-NaN rows, got None"
        )
        assert isinstance(result, dict), (
            f"Expected dict, got {type(result)}"
        )
        assert set(result.keys()) == {"slope", "intercept", "r_squared"}, (
            f"Dict keys mismatch: {set(result.keys())}"
        )
    else:
        # Fewer than 5 non-NaN rows — must return None
        assert result is None, (
            f"Expected None for {non_nan_count} non-NaN rows, got {result}"
        )


# ---------------------------------------------------------------------------
# Property-Based Tests — Task 5.6
# ---------------------------------------------------------------------------

# Feature: ethiopia-electricity-analysis, Property 5: Year-over-year percentage change satisfies the expected formula

from hypothesis import given, settings, assume
from hypothesis import strategies as st
import hypothesis.extra.pandas as hpd


@settings(max_examples=100)
@given(
    # Generate a list of at least 2 floats that may include NaN, but must contain
    # at least 2 finite (non-NaN, non-zero) values so the formula is checkable.
    values=st.lists(
        st.one_of(
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False).filter(lambda x: x != 0.0),
            st.just(float("nan")),
        ),
        min_size=2,
        max_size=20,
    ).filter(
        # Ensure at least 2 non-NaN values are present
        lambda lst: sum(1 for v in lst if not math.isnan(v)) >= 2
    )
)
def test_property_5_yoy_formula_correctness(values):
    """
    Property 5: Year-over-year percentage change satisfies the expected formula.

    For any cleaned DataFrame with at least 2 non-NaN consecutive rows for a
    given column, the YoY value at row i (where row i-1 is the preceding
    non-NaN row) SHALL equal (value[i] - value[i-1]) / abs(value[i-1]) * 100.

    Validates: Requirements 4.2
    """
    n = len(values)
    years = list(range(2000, 2000 + n))

    # Build a DataFrame using the consumption column; access column gets fixed values
    df = pd.DataFrame(
        {
            "year": years,
            "consumption_kwh_per_capita": pd.array(values, dtype=float),
            "access_pct_of_population": pd.array([50.0] * n, dtype=float),
        }
    )

    result = compute_yoy_change(df)

    assert "consumption_yoy_pct" in result.columns

    yoy = result["consumption_yoy_pct"]
    source = result["consumption_kwh_per_capita"]

    # Walk through the series the same way the implementation does:
    # track the last seen non-NaN value and verify the formula at each step.
    prev_value = None
    for idx in source.index:
        current = source[idx]

        if pd.isna(current):
            # NaN source → NaN yoy
            assert pd.isna(yoy[idx]), (
                f"Expected NaN yoy at index {idx} (source is NaN) but got {yoy[idx]}"
            )
            continue

        if prev_value is None:
            # First non-NaN value: no prior → NaN yoy
            assert pd.isna(yoy[idx]), (
                f"Expected NaN yoy at index {idx} (first non-NaN value) but got {yoy[idx]}"
            )
            prev_value = current
            continue

        abs_prev = abs(prev_value)
        if abs_prev == 0.0:
            # Division by zero → NaN (prev == 0 is excluded by the generator,
            # but guard anyway for safety)
            assert pd.isna(yoy[idx]), (
                f"Expected NaN yoy at index {idx} (prev==0) but got {yoy[idx]}"
            )
        else:
            expected = (current - prev_value) / abs_prev * 100.0
            assert math.isclose(yoy[idx], expected, rel_tol=1e-9, abs_tol=1e-9), (
                f"At index {idx}: expected {expected}, got {yoy[idx]} "
                f"(current={current}, prev={prev_value})"
            )

        prev_value = current


# ---------------------------------------------------------------------------
# Property-Based Tests — Task 5.5
# ---------------------------------------------------------------------------
# Feature: ethiopia-electricity-analysis, Property 4: Summary statistics match
# reference computations on non-NaN subsets

from hypothesis import given, settings, assume
from hypothesis import strategies as st
import numpy as np


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy for a single value that may be NaN
_maybe_nan_float = st.one_of(
    st.just(float("nan")),
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
)


@st.composite
def cleaned_dataframes(draw):
    """
    Generate a cleaned-style DataFrame with:
      - year: unique integers, sorted ascending
      - consumption_kwh_per_capita: mix of floats and NaN
      - access_pct_of_population: mix of floats and NaN
      - flagged: booleans
    At least one row must have a non-NaN value in at least one column
    (mirrors the clean_and_merge invariant).
    """
    n = draw(st.integers(min_value=1, max_value=50))

    # Generate n unique sorted years
    start_year = draw(st.integers(min_value=1960, max_value=2010))
    years = list(range(start_year, start_year + n))

    consumption = draw(st.lists(_maybe_nan_float, min_size=n, max_size=n))
    access = draw(st.lists(_maybe_nan_float, min_size=n, max_size=n))
    flagged = draw(st.lists(st.booleans(), min_size=n, max_size=n))

    # Ensure at least one non-NaN in at least one column (mirrors clean_and_merge)
    has_valid = any(not np.isnan(v) for v in consumption) or any(
        not np.isnan(v) for v in access
    )
    assume(has_valid)

    return pd.DataFrame(
        {
            "year": years,
            "consumption_kwh_per_capita": pd.array(consumption, dtype=float),
            "access_pct_of_population": pd.array(access, dtype=float),
            "flagged": flagged,
        }
    )


# ---------------------------------------------------------------------------
# Helper: reference stats for one column
# ---------------------------------------------------------------------------

def _ref_stats(df: pd.DataFrame, col: str) -> dict:
    """Compute reference statistics using pandas/numpy on non-NaN rows."""
    nan_count = int(df[col].isna().sum())
    valid = df.dropna(subset=[col])
    series = valid[col]

    if series.empty:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "std": float("nan"),
            "nan_count": nan_count,
            "peak_year": None,
            "low_year": None,
        }

    return {
        "mean": float(series.mean()),
        "median": float(series.median()),
        "min": float(series.min()),
        "max": float(series.max()),
        "std": float(series.std()),  # ddof=1 — pandas default
        "nan_count": nan_count,
        "peak_year": int(valid.loc[series.idxmax(), "year"]),
        "low_year": int(valid.loc[series.idxmin(), "year"]),
    }


# ---------------------------------------------------------------------------
# Property 4 test
# ---------------------------------------------------------------------------

@settings(max_examples=100, deadline=None)
@given(df=cleaned_dataframes())
def test_property_4_summary_stats_correctness(df):
    """
    Property 4: Summary statistics match reference computations on non-NaN subsets.

    For each value column, mean, median, min, max, std, peak_year, low_year and
    nan_count returned by compute_summary SHALL equal the corresponding reference
    values computed exclusively on the non-NaN rows of that column.

    Validates: Requirements 4.1, 4.3, 8.3
    """
    result = compute_summary(df)

    for col in ("consumption_kwh_per_capita", "access_pct_of_population"):
        assert col in result, f"compute_summary missing key '{col}'"

        actual = result[col]
        ref = _ref_stats(df, col)

        # --- nan_count ---
        assert actual["nan_count"] == ref["nan_count"], (
            f"[{col}] nan_count mismatch: got {actual['nan_count']}, "
            f"expected {ref['nan_count']}"
        )

        # --- stats that may be NaN when no valid rows ---
        for stat in ("mean", "median", "min", "max", "std"):
            act_val = actual[stat]
            ref_val = ref[stat]
            if math.isnan(ref_val):
                assert math.isnan(act_val), (
                    f"[{col}] {stat}: expected NaN, got {act_val}"
                )
            else:
                assert math.isclose(act_val, ref_val, rel_tol=1e-9, abs_tol=1e-12), (
                    f"[{col}] {stat} mismatch: got {act_val}, expected {ref_val}"
                )

        # --- peak_year and low_year ---
        assert actual["peak_year"] == ref["peak_year"], (
            f"[{col}] peak_year mismatch: got {actual['peak_year']}, "
            f"expected {ref['peak_year']}"
        )
        assert actual["low_year"] == ref["low_year"], (
            f"[{col}] low_year mismatch: got {actual['low_year']}, "
            f"expected {ref['low_year']}"
        )
