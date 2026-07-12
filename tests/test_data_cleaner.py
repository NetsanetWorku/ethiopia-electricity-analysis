"""
Unit tests for src/data_cleaner.py — clean_and_merge function.

Covers:
- Happy path: two valid DataFrames → correct merged output with right columns/dtypes
- Both-NaN row dropping
- Single-NaN row retention
- Column renaming (raw indicator codes → human-readable names)
- Duplicate year detection → ValueError
- Negative value flagging → flagged=True and WARNING logged
- Sorting by year (ascending)
"""

import logging
import math
import os
import sys

import pandas as pd
import pytest

# Make src importable when running from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data_cleaner import clean_and_merge

# ---------------------------------------------------------------------------
# Constants matching what data_cleaner.py uses internally
# ---------------------------------------------------------------------------
_CONSUMPTION_CODE = "EG.USE.ELEC.KH.PC"
_ACCESS_CODE = "EG.ELC.ACCS.ZS"
_CONSUMPTION_COL = "consumption_kwh_per_capita"
_ACCESS_COL = "access_pct_of_population"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_frames(
    consumption_rows: list[dict],
    access_rows: list[dict],
) -> dict[str, pd.DataFrame]:
    """Build the raw_frames dict that clean_and_merge expects."""
    consumption_df = pd.DataFrame(consumption_rows, columns=["year", _CONSUMPTION_CODE])
    access_df = pd.DataFrame(access_rows, columns=["year", _ACCESS_CODE])
    return {"consumption": consumption_df, "access": access_df}


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestHappyPath:
    """Two clean DataFrames → correctly merged output."""

    def _raw_frames(self):
        return _make_frames(
            consumption_rows=[
                {"year": 2020, _CONSUMPTION_CODE: 100.0},
                {"year": 2021, _CONSUMPTION_CODE: 110.0},
                {"year": 2022, _CONSUMPTION_CODE: 120.0},
            ],
            access_rows=[
                {"year": 2020, _ACCESS_CODE: 45.0},
                {"year": 2021, _ACCESS_CODE: 50.0},
                {"year": 2022, _ACCESS_CODE: 55.0},
            ],
        )

    def test_returns_dataframe(self):
        result = clean_and_merge(self._raw_frames())
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns_present(self):
        result = clean_and_merge(self._raw_frames())
        assert set(result.columns) == {"year", _CONSUMPTION_COL, _ACCESS_COL, "flagged"}

    def test_year_dtype_is_int(self):
        result = clean_and_merge(self._raw_frames())
        assert result["year"].dtype == int

    def test_consumption_dtype_is_float(self):
        result = clean_and_merge(self._raw_frames())
        assert result[_CONSUMPTION_COL].dtype == float

    def test_access_dtype_is_float(self):
        result = clean_and_merge(self._raw_frames())
        assert result[_ACCESS_COL].dtype == float

    def test_flagged_dtype_is_bool(self):
        result = clean_and_merge(self._raw_frames())
        assert result["flagged"].dtype == bool

    def test_row_count_matches_input(self):
        result = clean_and_merge(self._raw_frames())
        assert len(result) == 3

    def test_values_are_correct(self):
        result = clean_and_merge(self._raw_frames())
        row = result[result["year"] == 2021].iloc[0]
        assert row[_CONSUMPTION_COL] == pytest.approx(110.0)
        assert row[_ACCESS_COL] == pytest.approx(50.0)

    def test_no_rows_flagged_for_clean_data(self):
        result = clean_and_merge(self._raw_frames())
        assert result["flagged"].sum() == 0


# ---------------------------------------------------------------------------
# Both-NaN row dropping
# ---------------------------------------------------------------------------

class TestBothNanRowDropping:
    """Rows where BOTH value columns are NaN must be removed."""

    def test_both_nan_row_is_dropped(self):
        # year=2019 exists in neither DataFrame → both columns NaN after outer join
        frames = _make_frames(
            consumption_rows=[
                {"year": 2020, _CONSUMPTION_CODE: 100.0},
            ],
            access_rows=[
                {"year": 2021, _ACCESS_CODE: 45.0},
            ],
        )
        result = clean_and_merge(frames)
        # Only years 2020 and 2021 should survive (each has one real value)
        assert set(result["year"].tolist()) == {2020, 2021}

    def test_explicit_nan_both_cols_dropped(self):
        """Inject a row where both values are explicitly NaN."""
        consumption_df = pd.DataFrame({
            "year": [2020, 2021],
            _CONSUMPTION_CODE: [100.0, float("nan")],
        })
        access_df = pd.DataFrame({
            "year": [2020, 2021],
            _ACCESS_CODE: [45.0, float("nan")],
        })
        frames = {"consumption": consumption_df, "access": access_df}
        result = clean_and_merge(frames)
        assert 2021 not in result["year"].tolist()
        assert len(result) == 1

    def test_remaining_rows_after_drop_are_correct(self):
        consumption_df = pd.DataFrame({
            "year": [2020, 2021],
            _CONSUMPTION_CODE: [100.0, float("nan")],
        })
        access_df = pd.DataFrame({
            "year": [2020, 2021],
            _ACCESS_CODE: [45.0, float("nan")],
        })
        frames = {"consumption": consumption_df, "access": access_df}
        result = clean_and_merge(frames)
        assert result.iloc[0]["year"] == 2020
        assert result.iloc[0][_CONSUMPTION_COL] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Single-NaN row retention
# ---------------------------------------------------------------------------

class TestSingleNanRetention:
    """Rows where only ONE value column is NaN must be KEPT."""

    def test_nan_consumption_kept_when_access_present(self):
        consumption_df = pd.DataFrame({
            "year": [2020, 2021],
            _CONSUMPTION_CODE: [100.0, float("nan")],
        })
        access_df = pd.DataFrame({
            "year": [2020, 2021],
            _ACCESS_CODE: [45.0, 50.0],  # 2021 has a real access value
        })
        frames = {"consumption": consumption_df, "access": access_df}
        result = clean_and_merge(frames)
        assert 2021 in result["year"].tolist()
        row = result[result["year"] == 2021].iloc[0]
        assert math.isnan(row[_CONSUMPTION_COL])
        assert row[_ACCESS_COL] == pytest.approx(50.0)

    def test_nan_access_kept_when_consumption_present(self):
        consumption_df = pd.DataFrame({
            "year": [2020, 2021],
            _CONSUMPTION_CODE: [100.0, 110.0],
        })
        access_df = pd.DataFrame({
            "year": [2020, 2021],
            _ACCESS_CODE: [45.0, float("nan")],
        })
        frames = {"consumption": consumption_df, "access": access_df}
        result = clean_and_merge(frames)
        assert 2021 in result["year"].tolist()
        row = result[result["year"] == 2021].iloc[0]
        assert row[_CONSUMPTION_COL] == pytest.approx(110.0)
        assert math.isnan(row[_ACCESS_COL])


# ---------------------------------------------------------------------------
# Column renaming
# ---------------------------------------------------------------------------

class TestColumnRenaming:
    """Raw indicator codes must be renamed to human-readable column names."""

    def test_consumption_code_not_in_output(self):
        frames = _make_frames(
            [{"year": 2020, _CONSUMPTION_CODE: 100.0}],
            [{"year": 2020, _ACCESS_CODE: 45.0}],
        )
        result = clean_and_merge(frames)
        assert _CONSUMPTION_CODE not in result.columns

    def test_access_code_not_in_output(self):
        frames = _make_frames(
            [{"year": 2020, _CONSUMPTION_CODE: 100.0}],
            [{"year": 2020, _ACCESS_CODE: 45.0}],
        )
        result = clean_and_merge(frames)
        assert _ACCESS_CODE not in result.columns

    def test_consumption_human_name_in_output(self):
        frames = _make_frames(
            [{"year": 2020, _CONSUMPTION_CODE: 100.0}],
            [{"year": 2020, _ACCESS_CODE: 45.0}],
        )
        result = clean_and_merge(frames)
        assert _CONSUMPTION_COL in result.columns

    def test_access_human_name_in_output(self):
        frames = _make_frames(
            [{"year": 2020, _CONSUMPTION_CODE: 100.0}],
            [{"year": 2020, _ACCESS_CODE: 45.0}],
        )
        result = clean_and_merge(frames)
        assert _ACCESS_COL in result.columns


# ---------------------------------------------------------------------------
# Duplicate year detection → ValueError
# ---------------------------------------------------------------------------

class TestDuplicateYearDetection:
    """Duplicate year values after merge must raise ValueError."""

    def test_raises_value_error_on_duplicate_years_in_consumption(self):
        consumption_df = pd.DataFrame({
            "year": [2020, 2020],
            _CONSUMPTION_CODE: [100.0, 105.0],
        })
        access_df = pd.DataFrame({
            "year": [2020],
            _ACCESS_CODE: [45.0],
        })
        frames = {"consumption": consumption_df, "access": access_df}
        with pytest.raises(ValueError):
            clean_and_merge(frames)

    def test_raises_value_error_on_duplicate_years_in_access(self):
        consumption_df = pd.DataFrame({
            "year": [2020],
            _CONSUMPTION_CODE: [100.0],
        })
        access_df = pd.DataFrame({
            "year": [2020, 2020],
            _ACCESS_CODE: [45.0, 50.0],
        })
        frames = {"consumption": consumption_df, "access": access_df}
        with pytest.raises(ValueError):
            clean_and_merge(frames)

    def test_error_message_mentions_duplicate_year(self):
        consumption_df = pd.DataFrame({
            "year": [2020, 2020],
            _CONSUMPTION_CODE: [100.0, 105.0],
        })
        access_df = pd.DataFrame({
            "year": [2020],
            _ACCESS_CODE: [45.0],
        })
        frames = {"consumption": consumption_df, "access": access_df}
        with pytest.raises(ValueError, match="2020"):
            clean_and_merge(frames)


# ---------------------------------------------------------------------------
# Negative value flagging → flagged=True and WARNING logged
# ---------------------------------------------------------------------------

class TestNegativeValueFlagging:
    """Rows with any negative value must have flagged=True and a WARNING logged."""

    def test_negative_consumption_sets_flagged_true(self):
        frames = _make_frames(
            [{"year": 2020, _CONSUMPTION_CODE: -5.0}],
            [{"year": 2020, _ACCESS_CODE: 45.0}],
        )
        result = clean_and_merge(frames)
        assert bool(result.loc[result["year"] == 2020, "flagged"].iloc[0]) is True

    def test_negative_access_sets_flagged_true(self):
        frames = _make_frames(
            [{"year": 2020, _CONSUMPTION_CODE: 100.0}],
            [{"year": 2020, _ACCESS_CODE: -1.0}],
        )
        result = clean_and_merge(frames)
        assert bool(result.loc[result["year"] == 2020, "flagged"].iloc[0]) is True

    def test_both_negative_sets_flagged_true(self):
        frames = _make_frames(
            [{"year": 2020, _CONSUMPTION_CODE: -5.0}],
            [{"year": 2020, _ACCESS_CODE: -1.0}],
        )
        result = clean_and_merge(frames)
        assert bool(result.loc[result["year"] == 2020, "flagged"].iloc[0]) is True

    def test_positive_values_not_flagged(self):
        frames = _make_frames(
            [{"year": 2020, _CONSUMPTION_CODE: 100.0}],
            [{"year": 2020, _ACCESS_CODE: 45.0}],
        )
        result = clean_and_merge(frames)
        assert bool(result.loc[result["year"] == 2020, "flagged"].iloc[0]) is False

    def test_negative_value_logs_warning(self, caplog):
        frames = _make_frames(
            [{"year": 2020, _CONSUMPTION_CODE: -5.0}],
            [{"year": 2020, _ACCESS_CODE: 45.0}],
        )
        with caplog.at_level(logging.WARNING, logger="src.data_cleaner"):
            clean_and_merge(frames)
        assert any(r.levelno >= logging.WARNING for r in caplog.records)

    def test_only_flagged_rows_are_negative(self):
        frames = _make_frames(
            [
                {"year": 2020, _CONSUMPTION_CODE: 100.0},
                {"year": 2021, _CONSUMPTION_CODE: -5.0},
                {"year": 2022, _CONSUMPTION_CODE: 120.0},
            ],
            [
                {"year": 2020, _ACCESS_CODE: 45.0},
                {"year": 2021, _ACCESS_CODE: 50.0},
                {"year": 2022, _ACCESS_CODE: 55.0},
            ],
        )
        result = clean_and_merge(frames)
        assert bool(result.loc[result["year"] == 2021, "flagged"].iloc[0]) is True
        assert bool(result.loc[result["year"] == 2020, "flagged"].iloc[0]) is False
        assert bool(result.loc[result["year"] == 2022, "flagged"].iloc[0]) is False


# ---------------------------------------------------------------------------
# Sorting by year
# ---------------------------------------------------------------------------

class TestSortingByYear:
    """Output must be sorted ascending by year."""

    def test_output_sorted_ascending(self):
        frames = _make_frames(
            consumption_rows=[
                {"year": 2022, _CONSUMPTION_CODE: 120.0},
                {"year": 2020, _CONSUMPTION_CODE: 100.0},
                {"year": 2021, _CONSUMPTION_CODE: 110.0},
            ],
            access_rows=[
                {"year": 2022, _ACCESS_CODE: 55.0},
                {"year": 2020, _ACCESS_CODE: 45.0},
                {"year": 2021, _ACCESS_CODE: 50.0},
            ],
        )
        result = clean_and_merge(frames)
        years = result["year"].tolist()
        assert years == sorted(years)

    def test_first_row_is_earliest_year(self):
        frames = _make_frames(
            consumption_rows=[
                {"year": 2022, _CONSUMPTION_CODE: 120.0},
                {"year": 2019, _CONSUMPTION_CODE: 90.0},
                {"year": 2021, _CONSUMPTION_CODE: 110.0},
            ],
            access_rows=[
                {"year": 2022, _ACCESS_CODE: 55.0},
                {"year": 2019, _ACCESS_CODE: 40.0},
                {"year": 2021, _ACCESS_CODE: 50.0},
            ],
        )
        result = clean_and_merge(frames)
        assert result.iloc[0]["year"] == 2019

    def test_last_row_is_latest_year(self):
        frames = _make_frames(
            consumption_rows=[
                {"year": 2022, _CONSUMPTION_CODE: 120.0},
                {"year": 2019, _CONSUMPTION_CODE: 90.0},
            ],
            access_rows=[
                {"year": 2022, _ACCESS_CODE: 55.0},
                {"year": 2019, _ACCESS_CODE: 40.0},
            ],
        )
        result = clean_and_merge(frames)
        assert result.iloc[-1]["year"] == 2022


# ---------------------------------------------------------------------------
# Property-Based Tests (Hypothesis)
# ---------------------------------------------------------------------------

# Feature: ethiopia-electricity-analysis, Property 2: Cleaned dataset is sorted strictly ascending by year with no duplicates
import random
from hypothesis import given, settings
from hypothesis import strategies as st


@settings(max_examples=100)
@given(
    years=st.lists(
        st.integers(min_value=1900, max_value=2100),
        min_size=2,
        max_size=50,
        unique=True,
    )
)
def test_property_2_sorted_ascending_no_duplicates(years):
    """
    Property 2: For any raw input data (regardless of original row order),
    after running clean_and_merge, the year column SHALL be strictly increasing.

    Validates: Requirements 3.5
    """
    # Shuffle so input is not already sorted
    shuffled = years[:]
    random.shuffle(shuffled)

    # Build raw frames — each year appears exactly once in each DataFrame
    consumption_rows = [{"year": y, _CONSUMPTION_CODE: float(y)} for y in shuffled]
    access_rows = [{"year": y, _ACCESS_CODE: float(y) * 0.5} for y in shuffled]
    frames = _make_frames(consumption_rows, access_rows)

    result = clean_and_merge(frames)

    year_list = result["year"].tolist()
    # Assert strictly ascending: each consecutive pair satisfies year[i] < year[i+1]
    for i in range(len(year_list) - 1):
        assert year_list[i] < year_list[i + 1], (
            f"Year order violated at index {i}: {year_list[i]} >= {year_list[i + 1]}"
        )


# ---------------------------------------------------------------------------
# Property-Based Tests (Hypothesis)
# ---------------------------------------------------------------------------

# Feature: ethiopia-electricity-analysis, Property 1: Cleaned dataset column types are correct
# Validates: Requirements 3.3, 3.4

from hypothesis import given, settings, assume
import hypothesis.strategies as st

# Strategy: generate a list of unique integer years in a realistic range
_year_strategy = st.lists(
    st.integers(min_value=1960, max_value=2030),
    min_size=1,
    max_size=60,
    unique=True,
)

# Strategy: float values that may include NaN (represented as None → converted later)
_value_strategy = st.one_of(
    st.floats(min_value=-1000.0, max_value=100000.0, allow_nan=False, allow_infinity=False),
    st.just(float("nan")),
)


@settings(max_examples=100)
@given(
    years=_year_strategy,
    consumption_values=st.lists(
        _value_strategy, min_size=1, max_size=60
    ),
    access_values=st.lists(
        _value_strategy, min_size=1, max_size=60
    ),
)
def test_property_1_column_types_are_correct(years, consumption_values, access_values):
    """
    Property 1: For any raw input, after clean_and_merge the resulting DataFrame
    SHALL have `year` as integer dtype and both value columns as float dtype.
    """
    import math

    n = len(years)

    # Align value lists to the number of years (pad or truncate)
    def align(values, length):
        if len(values) >= length:
            return values[:length]
        # Pad with NaN if too short
        return values + [float("nan")] * (length - len(values))

    cons_vals = align(consumption_values, n)
    acc_vals = align(access_values, n)

    # Build raw DataFrames
    consumption_df = pd.DataFrame({
        "year": years,
        _CONSUMPTION_CODE: cons_vals,
    })
    access_df = pd.DataFrame({
        "year": years,
        _ACCESS_CODE: acc_vals,
    })
    raw_frames = {"consumption": consumption_df, "access": access_df}

    # Skip inputs where both columns are NaN for every row (nothing survives after clean)
    all_both_nan = all(
        math.isnan(c) and math.isnan(a)
        for c, a in zip(cons_vals, acc_vals)
    )
    assume(not all_both_nan)

    result = clean_and_merge(raw_frames)

    # The result must be a non-empty DataFrame
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0

    # Property 1: year must be integer dtype
    assert result["year"].dtype == int, (
        f"Expected year dtype to be int, got {result['year'].dtype}"
    )

    # Property 1: consumption column must be float dtype
    assert result[_CONSUMPTION_COL].dtype == float, (
        f"Expected {_CONSUMPTION_COL} dtype to be float, got {result[_CONSUMPTION_COL].dtype}"
    )

    # Property 1: access column must be float dtype
    assert result[_ACCESS_COL].dtype == float, (
        f"Expected {_ACCESS_COL} dtype to be float, got {result[_ACCESS_COL].dtype}"
    )


# ---------------------------------------------------------------------------
# Property-Based Tests (Hypothesis)
# ---------------------------------------------------------------------------

# Feature: ethiopia-electricity-analysis, Property 3: Every row in the cleaned dataset has at least one non-NaN indicator value

from hypothesis import given, settings, assume
from hypothesis import strategies as st
import numpy as np


def _nan_or_float(draw):
    """Draw either a float or NaN."""
    return draw(st.one_of(st.just(float("nan")), st.floats(allow_nan=False, allow_infinity=False)))


@st.composite
def raw_frames_strategy(draw):
    """
    Generate a raw_frames dict with a mix of real values and NaN for both indicators.
    Produces DataFrames with unique years so no ValueError is raised for duplicates.
    """
    # Generate a set of unique years
    years = draw(
        st.lists(
            st.integers(min_value=1960, max_value=2030),
            min_size=1,
            max_size=20,
            unique=True,
        )
    )

    # For each year, independently decide whether each indicator has a real value or NaN
    consumption_values = [draw(st.one_of(st.just(float("nan")), st.floats(allow_nan=False, allow_infinity=False, min_value=-1e9, max_value=1e9))) for _ in years]
    access_values = [draw(st.one_of(st.just(float("nan")), st.floats(allow_nan=False, allow_infinity=False, min_value=-1e9, max_value=1e9))) for _ in years]

    consumption_df = pd.DataFrame({
        "year": years,
        _CONSUMPTION_CODE: consumption_values,
    })
    access_df = pd.DataFrame({
        "year": years,
        _ACCESS_CODE: access_values,
    })

    return {"consumption": consumption_df, "access": access_df}


@settings(max_examples=100, deadline=None)
@given(raw_frames_strategy())
def test_property_3_no_all_nan_rows(raw_frames):
    """
    Property 3: Every row in the cleaned dataset has at least one non-NaN indicator value.
    After clean_and_merge, no row shall have BOTH consumption_kwh_per_capita AND
    access_pct_of_population as NaN.

    Validates: Requirements 3.1, 3.2
    """
    result = clean_and_merge(raw_frames)

    both_nan_mask = result[_CONSUMPTION_COL].isna() & result[_ACCESS_COL].isna()
    assert not both_nan_mask.any(), (
        f"Found {both_nan_mask.sum()} row(s) where both indicator columns are NaN. "
        f"Years with both-NaN: {result.loc[both_nan_mask, 'year'].tolist()}"
    )


# ---------------------------------------------------------------------------
# Property 9: NaN values from null API responses are preserved;
#             negative values are flagged
# ---------------------------------------------------------------------------

# Feature: ethiopia-electricity-analysis, Property 9: NaN values from null API responses are preserved; negative values are flagged
import math
from hypothesis import given, settings
from hypothesis import strategies as st
import numpy as np


# Strategy: generate a single row's (consumption, access) values as
# one of: positive float, zero, negative float, or NaN
_value_strategy = st.one_of(
    st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False),
    st.floats(min_value=-1000.0, max_value=-0.01, allow_nan=False, allow_infinity=False),
    st.just(float("nan")),
)


@settings(max_examples=100, deadline=None)
@given(
    rows=st.lists(
        st.tuples(_value_strategy, _value_strategy),
        min_size=1,
        max_size=30,
    )
)
def test_property_9_nan_preserved_negative_flagged(rows):
    """
    Property 9: NaN values from null API responses are preserved;
    negative values are flagged.

    For any raw input row where an indicator value is null/NaN, the
    corresponding value in the cleaned DataFrame SHALL remain NaN
    (not dropped or replaced). For any raw input row where an indicator
    value is negative, the cleaned DataFrame SHALL set the `flagged`
    boolean column to True for that row.

    Validates: Requirements 8.1
    """
    # Assign unique years to avoid duplicate-year ValueError
    years = list(range(2000, 2000 + len(rows)))

    consumption_vals = [c for c, _a in rows]
    access_vals = [a for _c, a in rows]

    consumption_df = pd.DataFrame({
        "year": years,
        _CONSUMPTION_CODE: consumption_vals,
    })
    access_df = pd.DataFrame({
        "year": years,
        _ACCESS_CODE: access_vals,
    })
    frames = {"consumption": consumption_df, "access": access_df}

    result = clean_and_merge(frames)

    for year, c_val, a_val in zip(years, consumption_vals, access_vals):
        c_nan = math.isnan(c_val)
        a_nan = math.isnan(a_val)

        # Rows where BOTH are NaN are intentionally dropped by clean_and_merge
        if c_nan and a_nan:
            assert year not in result["year"].values, (
                f"Year {year} with both NaN values should have been dropped"
            )
            continue

        # Row must be present
        matching = result[result["year"] == year]
        assert len(matching) == 1, f"Expected exactly one row for year {year}"
        row = matching.iloc[0]

        # 1. NaN inputs must remain NaN in the output (not replaced)
        if c_nan:
            assert math.isnan(row[_CONSUMPTION_COL]), (
                f"Year {year}: NaN consumption should be preserved, got {row[_CONSUMPTION_COL]}"
            )
        if a_nan:
            assert math.isnan(row[_ACCESS_COL]), (
                f"Year {year}: NaN access should be preserved, got {row[_ACCESS_COL]}"
            )

        # 2. Rows with any negative value → flagged=True
        has_negative = (not c_nan and c_val < 0) or (not a_nan and a_val < 0)
        if has_negative:
            assert row["flagged"] is True or bool(row["flagged"]) is True, (
                f"Year {year}: expected flagged=True for negative value(s), "
                f"consumption={c_val}, access={a_val}"
            )
        # 3. Rows with all non-negative values → flagged=False
        else:
            assert row["flagged"] is False or bool(row["flagged"]) is False, (
                f"Year {year}: expected flagged=False for non-negative values, "
                f"consumption={c_val}, access={a_val}"
            )


# ---------------------------------------------------------------------------
# Property 8: Cleaned CSV round-trip preserves data
# ---------------------------------------------------------------------------
# Feature: ethiopia-electricity-analysis, Property 8: Cleaned CSV round-trip preserves data

import math
import os
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st


# Validates: Requirements 3.6

_CONSUMPTION_CODE_PBT = "EG.USE.ELEC.KH.PC"
_ACCESS_CODE_PBT = "EG.ELC.ACCS.ZS"


def _finite_or_nan():
    """Strategy: float that is either a finite number or NaN."""
    return st.one_of(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
        st.just(float("nan")),
    )


def _make_raw_frames_strategy():
    """
    Generate a dict[str, pd.DataFrame] matching what clean_and_merge expects.

    Constraints:
    - years are unique integers in [1960, 2030]
    - at least one row has at least one non-NaN value (so some rows survive NaN-drop)
    - no duplicate years in either frame
    """
    return st.integers(min_value=1, max_value=20).flatmap(
        lambda n: st.lists(
            st.integers(min_value=1960, max_value=2030),
            min_size=n,
            max_size=n,
            unique=True,
        ).flatmap(
            lambda years: st.tuples(
                st.lists(
                    _finite_or_nan(),
                    min_size=len(years),
                    max_size=len(years),
                ),
                st.lists(
                    _finite_or_nan(),
                    min_size=len(years),
                    max_size=len(years),
                ),
            ).map(
                lambda vals: {
                    "consumption": pd.DataFrame(
                        {"year": years, _CONSUMPTION_CODE_PBT: vals[0]}
                    ),
                    "access": pd.DataFrame(
                        {"year": years, _ACCESS_CODE_PBT: vals[1]}
                    ),
                }
            )
        )
    )


@settings(max_examples=100, deadline=None)
@given(raw_frames=_make_raw_frames_strategy())
def test_property_8_csv_roundtrip(raw_frames):
    """
    Property 8: Cleaned CSV round-trip preserves data.

    For any cleaned DataFrame produced by clean_and_merge, saving it to CSV
    and reading it back SHALL produce a DataFrame with identical values,
    column names, and row count (within standard CSV float precision).

    Validates: Requirements 3.6
    """
    # At least one row must have at least one non-NaN value so clean_and_merge
    # doesn't end up with an empty DataFrame (which is valid, but the round-trip
    # still holds; we just assume non-empty for a more meaningful test).
    consumption_vals = raw_frames["consumption"][_CONSUMPTION_CODE_PBT].tolist()
    access_vals = raw_frames["access"][_ACCESS_CODE_PBT].tolist()
    has_surviving_row = any(
        not (math.isnan(c) if isinstance(c, float) else False)
        or not (math.isnan(a) if isinstance(a, float) else False)
        for c, a in zip(consumption_vals, access_vals)
    )
    assume(has_surviving_row)

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Patch the processed directory inside data_cleaner so the CSV is
        # written to our temp directory instead of data/processed/.
        fake_output_path = os.path.join(tmp_dir, "ethiopia_electricity_cleaned.csv")

        import src.data_cleaner as dc_module

        original_makedirs = os.makedirs

        def _patched_to_csv(df_self, path, **kwargs):
            # Intercept only the cleaned-dataset write; redirect to tmp_dir.
            if "ethiopia_electricity_cleaned" in str(path):
                df_self.to_csv.__wrapped__(df_self, fake_output_path, **kwargs)
            else:
                _original_to_csv(df_self, path, **kwargs)

        # Simplest approach: patch os.makedirs to allow creation of tmp_dir
        # structure, and redirect the output path via patching open inside
        # data_cleaner. We actually patch the whole module-level os reference
        # so that the join producing the output path lands in our tmp_dir.

        # Compute what the module would use as project_root
        src_dir = os.path.dirname(os.path.abspath(dc_module.__file__))
        project_root = os.path.dirname(src_dir)
        real_output = os.path.join(project_root, "data", "processed",
                                   "ethiopia_electricity_cleaned.csv")

        # Use patch to swap out the final to_csv call target by monkey-patching
        # os.path.join inside the data_cleaner module to redirect the path.
        _join_calls = []

        original_join = os.path.join  # capture before patching

        def _redirect_join(*args):
            result = original_join(*args)
            if result == real_output:
                return fake_output_path
            return result

        with patch.object(dc_module.os.path, "join", side_effect=_redirect_join):
            cleaned_df = dc_module.clean_and_merge(raw_frames)

        # --- Round-trip check ---
        assert os.path.isfile(fake_output_path), "CSV file was not created"

        reloaded_df = pd.read_csv(fake_output_path)

        # 1. Column names must be identical (same set, same order)
        assert list(cleaned_df.columns) == list(reloaded_df.columns), (
            f"Column mismatch: original={list(cleaned_df.columns)}, "
            f"reloaded={list(reloaded_df.columns)}"
        )

        # 2. Row count must be identical
        assert len(cleaned_df) == len(reloaded_df), (
            f"Row count mismatch: original={len(cleaned_df)}, "
            f"reloaded={len(reloaded_df)}"
        )

        # 3. Values must be identical within CSV float precision
        for col in cleaned_df.columns:
            orig_col = cleaned_df[col]
            rel_col = reloaded_df[col]

            if orig_col.dtype == bool or col == "flagged":
                # CSV stores bool as True/False strings; read_csv may read as bool or object
                orig_bool = orig_col.astype(bool).tolist()
                # handle both bool and string representations
                rel_bool = rel_col.map(
                    lambda v: v if isinstance(v, (bool, np.bool_))
                    else str(v).strip().lower() == "true"
                ).tolist()
                assert orig_bool == rel_bool, (
                    f"Bool column '{col}' mismatch after round-trip"
                )
            elif pd.api.types.is_integer_dtype(orig_col):
                assert orig_col.tolist() == rel_col.tolist(), (
                    f"Integer column '{col}' mismatch after round-trip"
                )
            else:
                # Float columns: use allclose with NaN-equality
                for i, (ov, rv) in enumerate(zip(orig_col, rel_col)):
                    orig_nan = isinstance(ov, float) and math.isnan(ov)
                    rel_nan = isinstance(rv, float) and math.isnan(rv) or (
                        isinstance(rv, float) and math.isnan(rv)
                    )
                    if orig_nan:
                        assert math.isnan(float(rv)), (
                            f"Column '{col}' row {i}: expected NaN, got {rv}"
                        )
                    else:
                        assert math.isclose(float(ov), float(rv), rel_tol=1e-6, abs_tol=1e-9), (
                            f"Column '{col}' row {i}: {ov} != {rv} after round-trip"
                        )
