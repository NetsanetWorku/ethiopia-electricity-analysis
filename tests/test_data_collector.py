"""
Unit tests for src/data_collector.py — fetch_indicator function.

Covers:
- Successful API response → correct DataFrame shape/dtypes/values
- API returns empty data list → ValueError
- Network failure + fallback CSV present → loads fallback, logs WARNING
- Network failure + fallback CSV missing → RuntimeError
- Null (None) values in API response → stored as NaN
"""

import math
import os
import tempfile
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

# Make sure the src package is importable when running from the project root
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_collector import fetch_indicator, _records_to_dataframe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

INDICATOR = "EG.USE.ELEC.KH.PC"
COUNTRY = "ET"

_SAMPLE_RECORDS = [
    {"date": "2022", "value": 100.5},
    {"date": "2021", "value": 98.0},
    {"date": "2020", "value": None},  # null → NaN
]

_FAKE_API_RESPONSE = [
    {"page": 1, "pages": 1, "per_page": 100, "total": 3},
    _SAMPLE_RECORDS,
]

_EMPTY_API_RESPONSE = [
    {"page": 1, "pages": 0, "per_page": 100, "total": 0},
    [],
]


def _make_mock_response(json_data, status_code=200):
    """Return a mock requests.Response object."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


# ---------------------------------------------------------------------------
# _records_to_dataframe (private, but tested for correctness)
# ---------------------------------------------------------------------------

class TestRecordsToDataFrame:
    def test_columns_present(self):
        df = _records_to_dataframe(_SAMPLE_RECORDS, INDICATOR)
        assert list(df.columns) == ["year", INDICATOR]

    def test_year_dtype_is_int(self):
        df = _records_to_dataframe(_SAMPLE_RECORDS, INDICATOR)
        assert df["year"].dtype == int

    def test_value_dtype_is_float(self):
        df = _records_to_dataframe(_SAMPLE_RECORDS, INDICATOR)
        assert df[INDICATOR].dtype == float

    def test_null_value_becomes_nan(self):
        df = _records_to_dataframe(_SAMPLE_RECORDS, INDICATOR)
        nan_rows = df[df["year"] == 2020]
        assert len(nan_rows) == 1
        assert math.isnan(nan_rows[INDICATOR].iloc[0])

    def test_valid_values_parsed(self):
        df = _records_to_dataframe(_SAMPLE_RECORDS, INDICATOR)
        row_2022 = df[df["year"] == 2022]
        assert row_2022[INDICATOR].iloc[0] == pytest.approx(100.5)

    def test_row_count_matches_records(self):
        df = _records_to_dataframe(_SAMPLE_RECORDS, INDICATOR)
        assert len(df) == len(_SAMPLE_RECORDS)

    def test_empty_records_returns_empty_dataframe(self):
        df = _records_to_dataframe([], INDICATOR)
        assert df.empty
        assert list(df.columns) == ["year", INDICATOR]


# ---------------------------------------------------------------------------
# fetch_indicator — successful API call
# ---------------------------------------------------------------------------

class TestFetchIndicatorSuccess:
    @patch("src.data_collector.requests.get")
    def test_returns_dataframe(self, mock_get):
        mock_get.return_value = _make_mock_response(_FAKE_API_RESPONSE)
        df = fetch_indicator(INDICATOR, COUNTRY, 1990, 2023)
        assert isinstance(df, pd.DataFrame)

    @patch("src.data_collector.requests.get")
    def test_columns_are_year_and_indicator(self, mock_get):
        mock_get.return_value = _make_mock_response(_FAKE_API_RESPONSE)
        df = fetch_indicator(INDICATOR, COUNTRY, 1990, 2023)
        assert "year" in df.columns
        assert INDICATOR in df.columns

    @patch("src.data_collector.requests.get")
    def test_year_dtype_int(self, mock_get):
        mock_get.return_value = _make_mock_response(_FAKE_API_RESPONSE)
        df = fetch_indicator(INDICATOR, COUNTRY, 1990, 2023)
        assert df["year"].dtype == int

    @patch("src.data_collector.requests.get")
    def test_value_dtype_float(self, mock_get):
        mock_get.return_value = _make_mock_response(_FAKE_API_RESPONSE)
        df = fetch_indicator(INDICATOR, COUNTRY, 1990, 2023)
        assert df[INDICATOR].dtype == float

    @patch("src.data_collector.requests.get")
    def test_null_api_value_is_nan(self, mock_get):
        mock_get.return_value = _make_mock_response(_FAKE_API_RESPONSE)
        df = fetch_indicator(INDICATOR, COUNTRY, 1990, 2023)
        # 2020 has value=None in _SAMPLE_RECORDS
        row = df[df["year"] == 2020]
        assert math.isnan(row[INDICATOR].iloc[0])

    @patch("src.data_collector.requests.get")
    def test_row_count(self, mock_get):
        mock_get.return_value = _make_mock_response(_FAKE_API_RESPONSE)
        df = fetch_indicator(INDICATOR, COUNTRY, 1990, 2023)
        assert len(df) == len(_SAMPLE_RECORDS)


# ---------------------------------------------------------------------------
# fetch_indicator — empty API response → ValueError
# ---------------------------------------------------------------------------

class TestFetchIndicatorEmptyResponse:
    @patch("src.data_collector.requests.get")
    def test_raises_value_error_on_empty_data(self, mock_get):
        mock_get.return_value = _make_mock_response(_EMPTY_API_RESPONSE)
        with pytest.raises(ValueError, match=INDICATOR):
            fetch_indicator(INDICATOR, COUNTRY, 1990, 2023)

    @patch("src.data_collector.requests.get")
    def test_value_error_mentions_country(self, mock_get):
        mock_get.return_value = _make_mock_response(_EMPTY_API_RESPONSE)
        with pytest.raises(ValueError, match=COUNTRY):
            fetch_indicator(INDICATOR, COUNTRY, 1990, 2023)


# ---------------------------------------------------------------------------
# fetch_indicator — network failure + fallback CSV present
# ---------------------------------------------------------------------------

class TestFetchIndicatorFallback:
    def _write_fallback(self, tmp_dir: str) -> pd.DataFrame:
        """Write a minimal fallback CSV and return the expected DataFrame."""
        fallback_df = pd.DataFrame(
            {"year": [2020, 2021], INDICATOR: [90.0, 95.0]}
        )
        csv_path = os.path.join(tmp_dir, f"{INDICATOR}.csv")
        fallback_df.to_csv(csv_path, index=False)
        return fallback_df

    @patch("src.data_collector.requests.get",
           side_effect=requests.exceptions.ConnectionError("no network"))
    def test_loads_fallback_on_connection_error(self, mock_get):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._write_fallback(tmp_dir)
            # Patch _load_fallback to point at the temp directory
            raw_dir = tmp_dir
            fallback_path = os.path.join(raw_dir, f"{INDICATOR}.csv")
            with patch("src.data_collector._load_fallback") as mock_fallback:
                mock_fallback.return_value = pd.DataFrame(
                    {"year": [2020, 2021], INDICATOR: [90.0, 95.0]}
                )
                df = fetch_indicator(INDICATOR, COUNTRY, 1990, 2023)
                mock_fallback.assert_called_once_with(INDICATOR)
                assert isinstance(df, pd.DataFrame)

    @patch("src.data_collector.requests.get",
           side_effect=requests.exceptions.Timeout("timed out"))
    def test_loads_fallback_on_timeout(self, mock_get):
        with patch("src.data_collector._load_fallback") as mock_fallback:
            mock_fallback.return_value = pd.DataFrame(
                {"year": [2020], INDICATOR: [88.0]}
            )
            df = fetch_indicator(INDICATOR, COUNTRY, 1990, 2023)
            mock_fallback.assert_called_once_with(INDICATOR)
            assert not df.empty

    @patch("src.data_collector.requests.get",
           side_effect=requests.exceptions.ConnectionError("no network"))
    def test_logs_warning_on_fallback(self, mock_get, caplog):
        import logging
        with patch("src.data_collector._load_fallback") as mock_fallback:
            mock_fallback.return_value = pd.DataFrame(
                {"year": [2020], INDICATOR: [88.0]}
            )
            with caplog.at_level(logging.WARNING, logger="src.data_collector"):
                fetch_indicator(INDICATOR, COUNTRY, 1990, 2023)
            assert any("WARNING" in r.levelname or r.levelno >= logging.WARNING
                       for r in caplog.records)


# ---------------------------------------------------------------------------
# fetch_indicator — network failure + fallback CSV missing → RuntimeError
# ---------------------------------------------------------------------------

class TestFetchIndicatorNoFallback:
    @patch("src.data_collector.requests.get",
           side_effect=requests.exceptions.ConnectionError("no network"))
    def test_raises_runtime_error_when_fallback_missing(self, mock_get):
        # _load_fallback raises RuntimeError when file is absent
        with patch("src.data_collector._load_fallback",
                   side_effect=RuntimeError("fallback missing")):
            with pytest.raises(RuntimeError):
                fetch_indicator(INDICATOR, COUNTRY, 1990, 2023)


# ---------------------------------------------------------------------------
# _load_fallback — unit tests exercising the real file system
# ---------------------------------------------------------------------------

class TestLoadFallback:
    def test_raises_runtime_error_when_file_missing(self):
        from src.data_collector import _load_fallback
        # Use a non-existent indicator code so no CSV can exist
        with pytest.raises(RuntimeError, match="fallback CSV not found"):
            _load_fallback("NONEXISTENT.INDICATOR.CODE")

    def test_loads_csv_successfully(self, tmp_path):
        from src.data_collector import _load_fallback
        # Write a valid CSV to the real data/raw/ path relative to project root
        src_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(src_dir)
        raw_dir = os.path.join(project_root, "data", "raw")
        os.makedirs(raw_dir, exist_ok=True)

        test_indicator = "TEST.INDICATOR.UNIT"
        csv_path = os.path.join(raw_dir, f"{test_indicator}.csv")
        try:
            pd.DataFrame({"year": [2020], test_indicator: [1.0]}).to_csv(
                csv_path, index=False
            )
            df = _load_fallback(test_indicator)
            assert not df.empty
        finally:
            if os.path.exists(csv_path):
                os.remove(csv_path)


# ---------------------------------------------------------------------------
# collect_all tests
# ---------------------------------------------------------------------------

from src.data_collector import collect_all


class _FakeConfig:
    """Minimal config object for testing collect_all."""
    COUNTRY_CODE = "ET"
    START_YEAR = 1990
    END_YEAR = 2023
    INDICATORS = {
        "consumption": "EG.USE.ELEC.KH.PC",
        "access": "EG.ELC.ACCS.ZS",
    }
    PATHS = {"raw": "data/raw"}


_CONSUMPTION_INDICATOR = "EG.USE.ELEC.KH.PC"
_ACCESS_INDICATOR = "EG.ELC.ACCS.ZS"

_FAKE_CONSUMPTION_DF = pd.DataFrame({
    "year": [2021, 2022],
    _CONSUMPTION_INDICATOR: [100.0, 105.0],
})
_FAKE_ACCESS_DF = pd.DataFrame({
    "year": [2021, 2022],
    _ACCESS_INDICATOR: [45.0, 50.0],
})


class TestCollectAll:
    def _make_fetch_side_effect(self):
        """Return a side_effect function that returns the right DF per indicator."""
        def _side_effect(indicator_code, country_code, start_year, end_year):
            if indicator_code == _CONSUMPTION_INDICATOR:
                return _FAKE_CONSUMPTION_DF.copy()
            if indicator_code == _ACCESS_INDICATOR:
                return _FAKE_ACCESS_DF.copy()
            raise ValueError(f"Unexpected indicator: {indicator_code}")
        return _side_effect

    @patch("src.data_collector.fetch_indicator")
    def test_returns_dict_with_both_short_names(self, mock_fetch, tmp_path, monkeypatch):
        mock_fetch.side_effect = self._make_fetch_side_effect()
        monkeypatch.setattr("src.data_collector.os.makedirs", lambda *a, **kw: None)
        # Redirect CSV saves to tmp_path so we don't hit the real filesystem
        with patch("src.data_collector.pd.DataFrame.to_csv"):
            result = collect_all(_FakeConfig())
        assert set(result.keys()) == {"consumption", "access"}

    @patch("src.data_collector.fetch_indicator")
    def test_values_are_dataframes(self, mock_fetch, monkeypatch):
        mock_fetch.side_effect = self._make_fetch_side_effect()
        with patch("src.data_collector.pd.DataFrame.to_csv"):
            result = collect_all(_FakeConfig())
        assert isinstance(result["consumption"], pd.DataFrame)
        assert isinstance(result["access"], pd.DataFrame)

    @patch("src.data_collector.fetch_indicator")
    def test_saves_csv_for_each_indicator(self, mock_fetch, tmp_path):
        """collect_all must write one CSV per indicator to data/raw/."""
        mock_fetch.side_effect = self._make_fetch_side_effect()

        # Build a config whose raw path points at tmp_path
        class _TmpConfig(_FakeConfig):
            PATHS = {"raw": str(tmp_path)}

        # Patch project_root resolution so raw_dir resolves to tmp_path itself
        with patch("src.data_collector.os.path.dirname", return_value=str(tmp_path)):
            # os.path.dirname is called twice (src_dir, project_root), so we
            # instead patch os.path.join to steer the final raw_dir correctly.
            pass

        # Simpler approach: patch os.makedirs and capture to_csv calls
        saved_paths = []
        original_to_csv = pd.DataFrame.to_csv

        def _capture_to_csv(self_df, path, **kwargs):
            saved_paths.append(path)
            # Actually write so we can verify
            original_to_csv(self_df, path, **kwargs)

        with patch.object(pd.DataFrame, "to_csv", _capture_to_csv):
            # We need the real raw_dir to exist; build a config pointing at tmp
            import sys, types
            fake_cfg = types.SimpleNamespace(
                COUNTRY_CODE="ET",
                START_YEAR=1990,
                END_YEAR=2023,
                INDICATORS={
                    "consumption": _CONSUMPTION_INDICATOR,
                    "access": _ACCESS_INDICATOR,
                },
                PATHS={"raw": str(tmp_path)},
            )
            # Patch project_root so raw_dir = tmp_path
            with patch("src.data_collector.os.path.dirname",
                       side_effect=[str(tmp_path), str(tmp_path)]):
                result = collect_all(fake_cfg)

        assert len(saved_paths) == 2
        filenames = {os.path.basename(p) for p in saved_paths}
        assert f"{_CONSUMPTION_INDICATOR}.csv" in filenames
        assert f"{_ACCESS_INDICATOR}.csv" in filenames

    @patch("src.data_collector.fetch_indicator",
           side_effect=ValueError("API returned zero rows for EG.USE.ELEC.KH.PC"))
    def test_logs_error_and_reraises_on_fetch_failure(self, mock_fetch, caplog):
        import logging
        with caplog.at_level(logging.ERROR, logger="src.data_collector"):
            with patch("src.data_collector.pd.DataFrame.to_csv"):
                with pytest.raises(ValueError):
                    collect_all(_FakeConfig())
        assert any(r.levelno >= logging.ERROR for r in caplog.records)

    @patch("src.data_collector.fetch_indicator")
    def test_fetch_called_once_per_indicator(self, mock_fetch, monkeypatch):
        mock_fetch.side_effect = self._make_fetch_side_effect()
        with patch("src.data_collector.pd.DataFrame.to_csv"):
            collect_all(_FakeConfig())
        assert mock_fetch.call_count == len(_FakeConfig.INDICATORS)

    @patch("src.data_collector.fetch_indicator")
    def test_fetch_called_with_correct_args(self, mock_fetch, monkeypatch):
        mock_fetch.side_effect = self._make_fetch_side_effect()
        with patch("src.data_collector.pd.DataFrame.to_csv"):
            collect_all(_FakeConfig())
        calls = {call.args[0]: call.args for call in mock_fetch.call_args_list}
        assert calls[_CONSUMPTION_INDICATOR] == (
            _CONSUMPTION_INDICATOR, "ET", 1990, 2023
        )
        assert calls[_ACCESS_INDICATOR] == (
            _ACCESS_INDICATOR, "ET", 1990, 2023
        )
