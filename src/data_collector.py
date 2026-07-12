"""
data_collector.py — Fetches raw indicator data from the World Bank API.

Public API (Task 2):
    fetch_indicator(indicator_code, country_code, start_year, end_year) -> pd.DataFrame
    collect_all(config) -> dict[str, pd.DataFrame]   # implemented in Task 2.2
"""

import logging
import os

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# World Bank JSON API template
# ---------------------------------------------------------------------------
_WB_URL_TEMPLATE = (
    "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
    "?format=json&per_page=100&mrv=60"
)

_STEP_NAME = "data_collector.fetch_indicator"


def fetch_indicator(
    indicator_code: str,
    country_code: str,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """
    Fetch one World Bank indicator via the JSON API.

    Parameters
    ----------
    indicator_code : str
        World Bank indicator code, e.g. ``"EG.USE.ELEC.KH.PC"``.
    country_code : str
        ISO-2 country code, e.g. ``"ET"``.
    start_year : int
        Inclusive start year (used only to document intent; the API ``mrv``
        parameter controls how many years are returned).
    end_year : int
        Inclusive end year (informational; filtering against this is not
        required by the spec — the API returns the most-recent values).

    Returns
    -------
    pd.DataFrame
        Columns: ``['year', indicator_code]``.
        ``year`` is ``int``; the indicator column is ``float``.

    Raises
    ------
    ValueError
        When the API response contains zero data rows.
    RuntimeError
        When the API is unreachable *and* the fallback CSV is missing or
        cannot be read.
    """
    url = _WB_URL_TEMPLATE.format(country=country_code, indicator=indicator_code)

    df = None  # will be populated from API or fallback

    # ------------------------------------------------------------------
    # 1. Try the live API
    # ------------------------------------------------------------------
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        payload = response.json()

        # payload[0] is metadata; payload[1] is the data list
        records = payload[1] if (isinstance(payload, list) and len(payload) >= 2) else []

        if not records:
            raise ValueError(
                f"[{_STEP_NAME}] World Bank API returned zero data rows "
                f"for indicator '{indicator_code}' and country '{country_code}'."
            )

        df = _records_to_dataframe(records, indicator_code)

    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
        logger.warning(
            "[%s] Network failure while fetching indicator '%s' for country '%s': %s. "
            "Attempting to load fallback CSV.",
            _STEP_NAME,
            indicator_code,
            country_code,
            exc,
        )
        df = _load_fallback(indicator_code)

    # ------------------------------------------------------------------
    # 2. Post-processing: validate we have rows
    # ------------------------------------------------------------------
    if df is not None and df.empty:
        raise ValueError(
            f"[{_STEP_NAME}] World Bank API returned zero data rows "
            f"for indicator '{indicator_code}' and country '{country_code}'."
        )

    return df


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _records_to_dataframe(records: list, indicator_code: str) -> pd.DataFrame:
    """
    Convert the list of dicts from the World Bank JSON response into a
    DataFrame with columns ``['year', indicator_code]``.

    The ``date`` field from each record becomes ``year`` (int), and the
    ``value`` field becomes the indicator column (float — NaN when null).
    """
    rows = []
    for rec in records:
        year_raw = rec.get("date")
        value_raw = rec.get("value")

        try:
            year = int(year_raw)
        except (TypeError, ValueError):
            # Skip records with unparseable dates
            logger.warning(
                "[%s] Skipping record with unparseable date: %r", _STEP_NAME, year_raw
            )
            continue

        # value may be None (null in JSON) → stored as NaN
        value = float(value_raw) if value_raw is not None else float("nan")

        rows.append({"year": year, indicator_code: value})

    df = pd.DataFrame(rows, columns=["year", indicator_code])
    df["year"] = df["year"].astype(int)
    df[indicator_code] = df[indicator_code].astype(float)
    return df


def _load_fallback(indicator_code: str) -> pd.DataFrame:
    """
    Load the raw fallback CSV for *indicator_code* from ``data/raw/``.

    The path is resolved relative to the project root (two levels above
    this file: ``src/data_collector.py`` → ``src/`` → project root).

    Raises
    ------
    RuntimeError
        When the fallback file is missing or cannot be read.
    """
    # Resolve project root relative to this source file so the module works
    # regardless of the current working directory.
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    fallback_path = os.path.join(project_root, "data", "raw", f"{indicator_code}.csv")

    if not os.path.isfile(fallback_path):
        raise RuntimeError(
            f"[{_STEP_NAME}] API unreachable and fallback CSV not found: "
            f"'{fallback_path}'. Cannot load data for indicator '{indicator_code}'."
        )

    try:
        df = pd.read_csv(fallback_path)
        logger.warning(
            "[%s] Loaded fallback CSV from '%s' for indicator '%s'.",
            _STEP_NAME,
            fallback_path,
            indicator_code,
        )
        return df
    except Exception as exc:
        raise RuntimeError(
            f"[{_STEP_NAME}] API unreachable and fallback CSV at '{fallback_path}' "
            f"could not be read: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# collect_all
# ---------------------------------------------------------------------------

_COLLECT_ALL_STEP = "data_collector.collect_all"


def collect_all(config) -> dict:
    """
    Fetch all indicators defined in *config* and save them as raw CSVs.

    Parameters
    ----------
    config : module or object
        Must expose:
        - ``INDICATORS``  : dict[str, str]  — e.g. ``{"consumption": "EG.USE.ELEC.KH.PC"}``
        - ``COUNTRY_CODE``: str             — ISO-2 code, e.g. ``"ET"``
        - ``START_YEAR``  : int
        - ``END_YEAR``    : int
        - ``PATHS``       : dict            — must contain key ``"raw"`` with the
                             path to the raw data directory (relative to project
                             root or absolute).

    Returns
    -------
    dict[str, pd.DataFrame]
        Keys are the short names from ``config.INDICATORS`` (e.g.
        ``'consumption'``, ``'access'``); values are the DataFrames returned by
        :func:`fetch_indicator`.

    Raises
    ------
    Exception
        Any exception raised by :func:`fetch_indicator` or by file I/O is
        logged with the step name before being re-raised.
    """
    # Resolve the raw-data directory relative to the project root so the
    # function works regardless of the current working directory.
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    raw_dir = os.path.join(project_root, config.PATHS["raw"])
    os.makedirs(raw_dir, exist_ok=True)

    results: dict = {}

    for short_name, indicator_code in config.INDICATORS.items():
        try:
            df = fetch_indicator(
                indicator_code,
                config.COUNTRY_CODE,
                config.START_YEAR,
                config.END_YEAR,
            )

            # Save the raw result to data/raw/<indicator_code>.csv
            csv_path = os.path.join(raw_dir, f"{indicator_code}.csv")
            df.to_csv(csv_path, index=False)
            logger.info(
                "[%s] Saved raw data for indicator '%s' to '%s'.",
                _COLLECT_ALL_STEP,
                indicator_code,
                csv_path,
            )

            results[short_name] = df

        except Exception as exc:
            logger.error(
                "[%s] Error while collecting indicator '%s' (%s): %s",
                _COLLECT_ALL_STEP,
                indicator_code,
                short_name,
                exc,
            )
            raise

    return results
