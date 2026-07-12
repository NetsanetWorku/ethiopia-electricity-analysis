"""
test_report_generator.py — Unit tests for src/report_generator.py

Covers:
- All six section headers are present in the report
- Numeric mean values appear in the report
- Trend slope values appear in the report
- Peak year values appear in the report
- Positive slope → "increasing" text
- Negative slope → "decreasing" text
- None slope → "no trend data" text
- Report file is saved to reports/ethiopia_electricity_report.md
- Function returns a string
"""

import os

import pandas as pd
import pytest

from src.report_generator import generate_report


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _FakeConfig:
    """Minimal stand-in for the real config module."""
    COUNTRY_CODE = "ET"
    INDICATORS = {
        "consumption": "EG.USE.ELEC.KH.PC",
        "access": "EG.ELC.ACCS.ZS",
    }
    PATHS = {
        "raw": "data/raw",
        "processed": "data/processed",
        "charts": "charts",
        "reports": "reports",
        "notebook": "notebooks",
    }


def _make_df():
    """Return a minimal cleaned DataFrame for testing."""
    return pd.DataFrame({
        "year": [2000, 2001, 2002, 2003, 2004],
        "consumption_kwh_per_capita": [100.0, 110.0, 120.0, 130.0, 140.0],
        "access_pct_of_population": [20.0, 25.0, 30.0, 35.0, 40.0],
    })


def _make_results_positive():
    """Results dict with positive slopes for both indicators."""
    return {
        "summary": {
            "consumption_kwh_per_capita": {
                "mean": 120.0,
                "median": 120.0,
                "min": 100.0,
                "max": 140.0,
                "std": 15.811,
                "nan_count": 0,
                "peak_year": 2004,
                "low_year": 2000,
            },
            "access_pct_of_population": {
                "mean": 30.0,
                "median": 30.0,
                "min": 20.0,
                "max": 40.0,
                "std": 7.906,
                "nan_count": 0,
                "peak_year": 2004,
                "low_year": 2000,
            },
        },
        "trends": {
            "consumption_kwh_per_capita": {"slope": 10.0, "intercept": -19880.0, "r_squared": 1.0},
            "access_pct_of_population": {"slope": 5.0, "intercept": -9980.0, "r_squared": 1.0},
        },
    }


def _make_results_negative():
    """Results dict with negative slopes for both indicators."""
    results = _make_results_positive()
    results["trends"]["consumption_kwh_per_capita"]["slope"] = -10.0
    results["trends"]["access_pct_of_population"]["slope"] = -5.0
    return results


def _make_results_none_trend():
    """Results dict where trend is None for both indicators."""
    results = _make_results_positive()
    results["trends"]["consumption_kwh_per_capita"] = None
    results["trends"]["access_pct_of_population"] = None
    return results


# ---------------------------------------------------------------------------
# Helper: call generate_report and return the markdown string
# ---------------------------------------------------------------------------

def _run(results):
    return generate_report(_make_df(), results, _FakeConfig())


# ---------------------------------------------------------------------------
# Tests: return type
# ---------------------------------------------------------------------------

class TestReturnType:
    def test_returns_string(self):
        report = _run(_make_results_positive())
        assert isinstance(report, str)


# ---------------------------------------------------------------------------
# Tests: section headers
# ---------------------------------------------------------------------------

class TestSectionHeaders:
    @pytest.fixture(autouse=True)
    def _report(self):
        self.report = _run(_make_results_positive())

    def test_introduction_header(self):
        assert "# Introduction" in self.report

    def test_data_sources_header(self):
        assert "# Data Sources" in self.report

    def test_methodology_header(self):
        assert "# Methodology" in self.report

    def test_key_findings_header(self):
        assert "# Key Findings" in self.report

    def test_visualizations_header(self):
        assert "# Visualizations" in self.report

    def test_conclusion_header(self):
        assert "# Conclusion" in self.report


# ---------------------------------------------------------------------------
# Tests: numeric content
# ---------------------------------------------------------------------------

class TestNumericContent:
    @pytest.fixture(autouse=True)
    def _report(self):
        self.report = _run(_make_results_positive())

    def test_contains_consumption_mean(self):
        # mean is 120.0 — formatted as 120.0000
        assert "120.0000" in self.report

    def test_contains_access_mean(self):
        # mean is 30.0 — formatted as 30.0000
        assert "30.0000" in self.report

    def test_contains_consumption_slope(self):
        # slope 10.0 → formatted as 10.000000
        assert "10.000000" in self.report

    def test_contains_access_slope(self):
        # slope 5.0 → formatted as 5.000000
        assert "5.000000" in self.report

    def test_contains_consumption_peak_year(self):
        assert "2004" in self.report

    def test_contains_access_peak_year(self):
        assert "2004" in self.report


# ---------------------------------------------------------------------------
# Tests: trend direction language
# ---------------------------------------------------------------------------

class TestTrendDirection:
    def test_positive_slope_produces_increasing(self):
        report = _run(_make_results_positive())
        assert "increasing" in report

    def test_negative_slope_produces_decreasing(self):
        report = _run(_make_results_negative())
        assert "decreasing" in report

    def test_none_slope_produces_no_trend_data(self):
        report = _run(_make_results_none_trend())
        assert "no trend data" in report

    def test_positive_slope_does_not_produce_decreasing(self):
        report = _run(_make_results_positive())
        assert "decreasing" not in report

    def test_negative_slope_does_not_produce_increasing(self):
        report = _run(_make_results_negative())
        assert "increasing" not in report


# ---------------------------------------------------------------------------
# Tests: file saved to reports/ethiopia_electricity_report.md
# ---------------------------------------------------------------------------

class TestFileSaved:
    def test_report_file_exists_after_generation(self):
        _run(_make_results_positive())

        # Resolve expected path relative to src/report_generator.py
        src_dir = os.path.dirname(os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "src", "report_generator.py")
        ))
        project_root = os.path.join(src_dir, "..")
        expected_path = os.path.abspath(
            os.path.join(project_root, "reports", "ethiopia_electricity_report.md")
        )
        assert os.path.isfile(expected_path), (
            f"Report file not found at expected path: {expected_path}"
        )

    def test_saved_file_matches_returned_string(self):
        report = _run(_make_results_positive())

        src_dir = os.path.dirname(os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "src", "report_generator.py")
        ))
        project_root = os.path.join(src_dir, "..")
        report_path = os.path.abspath(
            os.path.join(project_root, "reports", "ethiopia_electricity_report.md")
        )

        with open(report_path, encoding="utf-8") as fh:
            saved_content = fh.read()

        assert report == saved_content


# ---------------------------------------------------------------------------
# Tests: Data Sources section content
# ---------------------------------------------------------------------------

class TestDataSources:
    @pytest.fixture(autouse=True)
    def _report(self):
        self.report = _run(_make_results_positive())

    def test_contains_consumption_indicator_code(self):
        assert "EG.USE.ELEC.KH.PC" in self.report

    def test_contains_access_indicator_code(self):
        assert "EG.ELC.ACCS.ZS" in self.report

    def test_contains_ethiopia_country_code(self):
        assert "ET" in self.report


# ---------------------------------------------------------------------------
# Tests: Visualizations section image references
# ---------------------------------------------------------------------------

class TestVisualizationsSection:
    @pytest.fixture(autouse=True)
    def _report(self):
        self.report = _run(_make_results_positive())

    def test_contains_consumption_image_ref(self):
        assert "consumption_kwh_per_capita.png" in self.report

    def test_contains_access_image_ref(self):
        assert "access_pct_of_population.png" in self.report

    def test_image_refs_use_markdown_syntax(self):
        assert "![" in self.report
        assert "](../charts/" in self.report
