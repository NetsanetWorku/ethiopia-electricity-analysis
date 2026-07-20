"""
trend_analyzer.py — Computes summary statistics and trend analysis for the
Ethiopia electricity cleaned dataset.

Public API (implemented incrementally across tasks 5.1–5.4):
    compute_summary(df: pd.DataFrame) -> dict           [Task 5.1]
    compute_yoy_change(df: pd.DataFrame) -> pd.DataFrame  [Task 5.2]
    fit_linear_trend(df: pd.DataFrame, column: str) -> dict | None  [Task 5.3]
    analyze(df: pd.DataFrame, config) -> dict           [Task 5.4]
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

_STEP_NAME = "trend_analyzer.compute_summary"

# Value columns analysed by this module
_VALUE_COLUMNS = [
    "consumption_kwh_per_capita",
    "access_pct_of_population",
]


def compute_summary(df: pd.DataFrame) -> dict:
    """
    Compute descriptive statistics for each value column using only non-NaN rows.

    For each column in ``_VALUE_COLUMNS`` that is present in *df*, the following
    metrics are computed on the non-NaN subset:

    - ``mean``      — arithmetic mean
    - ``median``    — median value
    - ``min``       — minimum value
    - ``max``       — maximum value
    - ``std``       — standard deviation (ddof=1, i.e. the pandas / numpy default)
    - ``nan_count`` — number of rows excluded because the value was NaN
    - ``peak_year`` — the ``year`` value at the row with the maximum non-NaN value
    - ``low_year``  — the ``year`` value at the row with the minimum non-NaN value

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset with at minimum a ``year`` (int) column and one or both
        of the value columns listed in ``_VALUE_COLUMNS``.

    Returns
    -------
    dict
        A dict keyed by column name, e.g.::

            {
                "consumption_kwh_per_capita": {
                    "mean": ..., "median": ..., "min": ..., "max": ...,
                    "std": ..., "nan_count": ..., "peak_year": ..., "low_year": ...
                },
                "access_pct_of_population": { ... }
            }

        Only columns that exist in *df* are included in the output.
    """
    summary: dict = {}

    for col in _VALUE_COLUMNS:
        if col not in df.columns:
            logger.warning(
                "[%s] Column '%s' not found in DataFrame — skipping.",
                _STEP_NAME,
                col,
            )
            continue

        nan_count = int(df[col].isna().sum())
        valid_df = df.dropna(subset=[col])

        if valid_df.empty:
            logger.warning(
                "[%s] Column '%s' has no non-NaN values — all statistics will be NaN.",
                _STEP_NAME,
                col,
            )
            summary[col] = {
                "mean": float("nan"),
                "median": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
                "std": float("nan"),
                "nan_count": nan_count,
                "peak_year": None,
                "low_year": None,
            }
            continue

        series = valid_df[col]

        mean_val = float(series.mean())
        median_val = float(series.median())
        min_val = float(series.min())
        max_val = float(series.max())
        std_val = float(series.std())  # ddof=1 by default (pandas / numpy consistent)

        # peak_year: year at the row with the maximum value
        peak_year = int(valid_df.loc[series.idxmax(), "year"])
        # low_year: year at the row with the minimum value
        low_year = int(valid_df.loc[series.idxmin(), "year"])

        logger.info(
            "[%s] '%s' — mean=%.4f, median=%.4f, min=%.4f, max=%.4f, "
            "std=%.4f, nan_count=%d, peak_year=%d, low_year=%d",
            _STEP_NAME,
            col,
            mean_val,
            median_val,
            min_val,
            max_val,
            std_val,
            nan_count,
            peak_year,
            low_year,
        )

        summary[col] = {
            "mean": mean_val,
            "median": median_val,
            "min": min_val,
            "max": max_val,
            "std": std_val,
            "nan_count": nan_count,
            "peak_year": peak_year,
            "low_year": low_year,
        }

    return summary


def compute_yoy_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append year-over-year percentage change columns to a copy of the input DataFrame.

    For each column in ``_VALUE_COLUMNS`` that is present in *df*, a new column is
    added:

    - ``consumption_yoy_pct``  — YoY % change for ``consumption_kwh_per_capita``
    - ``access_yoy_pct``       — YoY % change for ``access_pct_of_population``

    Formula applied for consecutive non-NaN rows::

        yoy[i] = (value[i] - value[i-1]) / abs(value[i-1]) * 100

    The first row (and any row whose preceding non-NaN value does not exist)
    receives ``NaN``.  NaN values in the source column are skipped — they do
    *not* consume a "previous" slot, so the change is computed between the two
    nearest non-NaN values regardless of gap size.

    Division by zero (i.e. ``value[i-1] == 0``) yields ``NaN`` (not ±inf).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset containing a ``year`` column and one or both value
        columns listed in ``_VALUE_COLUMNS``.  The DataFrame is **not** modified
        in place.

    Returns
    -------
    pd.DataFrame
        A new DataFrame (copy of *df*) with the YoY columns appended.
        If a source column is absent, its corresponding YoY column is also
        absent (no error is raised).
    """
    _YOY_STEP = "trend_analyzer.compute_yoy_change"

    result = df.copy()

    # Mapping: source column → new YoY column name
    _YOY_MAP = {
        "consumption_kwh_per_capita": "consumption_yoy_pct",
        "access_pct_of_population": "access_yoy_pct",
    }

    for src_col, yoy_col in _YOY_MAP.items():
        if src_col not in result.columns:
            logger.warning(
                "[%s] Column '%s' not found — skipping YoY computation.",
                _YOY_STEP,
                src_col,
            )
            continue

        series = result[src_col]
        yoy_values = pd.Series([float("nan")] * len(series), index=series.index)

        # Iterate only over non-NaN positions, carrying the previous non-NaN value
        prev_value = None
        for idx in series.index:
            current = series[idx]
            if pd.isna(current):
                # NaN in source: leave yoy as NaN, do not update prev_value
                continue
            if prev_value is None:
                # First non-NaN value: no previous row, leave as NaN
                prev_value = current
                continue
            # Guard against division by zero
            abs_prev = abs(prev_value)
            if abs_prev == 0:
                yoy_values[idx] = float("nan")
                logger.warning(
                    "[%s] Division by zero at index %s for column '%s' — YoY set to NaN.",
                    _YOY_STEP,
                    idx,
                    src_col,
                )
            else:
                yoy_values[idx] = (current - prev_value) / abs_prev * 100.0
            prev_value = current

        result[yoy_col] = yoy_values
        logger.info(
            "[%s] Added '%s' column.",
            _YOY_STEP,
            yoy_col,
        )

    return result


def fit_linear_trend(df: pd.DataFrame, column: str) -> dict | None:
    """
    Fit a linear regression to non-NaN rows of the specified column.

    Uses NumPy to compute the ordinary least squares line with ``year`` as the
    independent variable and *column* values as the dependent variable.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset with at minimum a ``year`` (int) column and the
        target *column*.
    column : str
        Name of the value column to regress (e.g. ``'consumption_kwh_per_capita'``).

    Returns
    -------
    dict | None
        When at least 5 non-NaN rows exist::

            {'slope': float, 'intercept': float, 'r_squared': float}

        Returns ``None`` and logs a WARNING when fewer than 5 non-NaN rows
        are available.
    """
    import numpy as np

    _STEP = "trend_analyzer.fit_linear_trend"

    if column not in df.columns:
        logger.warning(
            "[%s] Column '%s' not found in DataFrame — returning None.",
            _STEP,
            column,
        )
        return None

    valid_df = df.dropna(subset=[column])
    n_valid = len(valid_df)

    if n_valid < 5:
        logger.warning(
            "[%s] Column '%s' has only %d non-NaN row(s); "
            "at least 5 are required for regression — returning None.",
            _STEP,
            column,
            n_valid,
        )
        return None

    x = valid_df["year"].to_numpy(dtype=float)
    y = valid_df[column].to_numpy(dtype=float)

    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_diff = x - x_mean
    y_diff = y - y_mean

    ss_xx = np.sum(x_diff * x_diff)
    ss_xy = np.sum(x_diff * y_diff)
    slope = float(ss_xy / ss_xx) if ss_xx != 0 else 0.0
    intercept = float(y_mean - slope * x_mean)

    y_pred = slope * x + intercept
    ss_tot = np.sum(y_diff * y_diff)
    ss_res = np.sum((y - y_pred) ** 2)

    if ss_tot == 0:
        r_squared = 1.0
    else:
        r_squared = float(1.0 - ss_res / ss_tot)

    logger.info(
        "[%s] '%s' — slope=%.6f, intercept=%.4f, r_squared=%.4f",
        _STEP,
        column,
        slope,
        intercept,
        r_squared,
    )

    return {"slope": slope, "intercept": intercept, "r_squared": r_squared}


def analyze(df: pd.DataFrame, config) -> dict:
    """
    Orchestrate all analysis steps and persist summary statistics.

    This function calls :func:`compute_summary`, :func:`compute_yoy_change`,
    and :func:`fit_linear_trend` for each value column, then saves a summary
    CSV to ``data/processed/ethiopia_electricity_summary.csv`` (resolved
    relative to this source file so the path is correct regardless of the
    current working directory).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset as produced by ``data_cleaner.clean_and_merge``.
    config : module
        The project ``config`` module (used for ``PATHS`` — kept as a
        parameter for future extensibility and testability).

    Returns
    -------
    dict
        A results dict with three top-level keys::

            {
                "summary": {
                    "consumption_kwh_per_capita": {
                        "mean": float, "median": float, "min": float,
                        "max": float, "std": float, "nan_count": int,
                        "peak_year": int | None, "low_year": int | None
                    },
                    "access_pct_of_population": { ... }
                },
                "trends": {
                    "consumption_kwh_per_capita": {"slope": float, "intercept": float, "r_squared": float} | None,
                    "access_pct_of_population":   { ... } | None
                },
                "df_with_yoy": pd.DataFrame  # augmented with YoY columns
            }

    Raises
    ------
    Exception
        Any exception from a sub-step is logged with the step name and
        then re-raised.
    """
    import os

    _STEP = "trend_analyzer.analyze"

    # ------------------------------------------------------------------
    # Step 1: compute summary statistics
    # ------------------------------------------------------------------
    try:
        summary = compute_summary(df)
    except Exception as exc:
        logger.error(
            "[%s] Error during compute_summary: %s",
            _STEP,
            exc,
        )
        raise

    # ------------------------------------------------------------------
    # Step 2: add year-over-year columns
    # ------------------------------------------------------------------
    try:
        df_with_yoy = compute_yoy_change(df)
    except Exception as exc:
        logger.error(
            "[%s] Error during compute_yoy_change: %s",
            _STEP,
            exc,
        )
        raise

    # ------------------------------------------------------------------
    # Step 3: fit linear trend for each value column
    # ------------------------------------------------------------------
    trends: dict = {}
    for col in _VALUE_COLUMNS:
        try:
            trends[col] = fit_linear_trend(df, col)
        except Exception as exc:
            logger.error(
                "[%s] Error during fit_linear_trend for column '%s': %s",
                _STEP,
                col,
                exc,
            )
            raise

    # ------------------------------------------------------------------
    # Step 4: save summary CSV
    # ------------------------------------------------------------------
    try:
        # Resolve path relative to this source file:
        # __file__ → src/trend_analyzer.py
        # dirname → src/
        # join(.., '..') → project root
        # join(.., 'data', 'processed') → data/processed/
        src_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(src_dir, "..")
        processed_dir = os.path.join(project_root, "data", "processed")
        os.makedirs(processed_dir, exist_ok=True)
        summary_path = os.path.join(processed_dir, "ethiopia_electricity_summary.csv")

        # Build a tidy DataFrame:
        # rows = metric names, columns = metric | consumption_kwh_per_capita | access_pct_of_population
        metric_names = ["mean", "median", "min", "max", "std", "nan_count", "peak_year", "low_year"]
        rows = []
        for metric in metric_names:
            row = {"metric": metric}
            for col in _VALUE_COLUMNS:
                row[col] = summary.get(col, {}).get(metric, float("nan"))
            rows.append(row)

        summary_df = pd.DataFrame(rows, columns=["metric"] + _VALUE_COLUMNS)
        summary_df.to_csv(summary_path, index=False)
        logger.info("[%s] Summary CSV saved to '%s'.", _STEP, summary_path)
    except Exception as exc:
        logger.error(
            "[%s] Error saving summary CSV: %s",
            _STEP,
            exc,
        )
        raise

    # ------------------------------------------------------------------
    # Step 5: assemble and return results dict
    # ------------------------------------------------------------------
    return {
        "summary": summary,
        "trends": trends,
        "df_with_yoy": df_with_yoy,
    }
