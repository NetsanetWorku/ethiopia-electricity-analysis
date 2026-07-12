"""
data_cleaner.py — Cleans and merges the two raw World Bank indicator DataFrames.

Public API (Task 3.1):
    clean_and_merge(raw_frames: dict[str, pd.DataFrame]) -> pd.DataFrame
"""

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

_STEP_NAME = "data_cleaner.clean_and_merge"

# Column name mapping from raw indicator codes to human-readable names
_CONSUMPTION_CODE = "EG.USE.ELEC.KH.PC"
_ACCESS_CODE = "EG.ELC.ACCS.ZS"

_CONSUMPTION_COL = "consumption_kwh_per_capita"
_ACCESS_COL = "access_pct_of_population"


def clean_and_merge(raw_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Clean and merge the two raw indicator DataFrames into a single analysis-ready dataset.

    Parameters
    ----------
    raw_frames : dict[str, pd.DataFrame]
        Dictionary with keys:
        - ``'consumption'``: DataFrame with columns ``['year', 'EG.USE.ELEC.KH.PC']``
        - ``'access'``:      DataFrame with columns ``['year', 'EG.ELC.ACCS.ZS']``

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns:
        - ``year`` (int): calendar year, unique, sorted ascending
        - ``consumption_kwh_per_capita`` (float): kWh consumed per capita (may be NaN)
        - ``access_pct_of_population`` (float): % of population with electricity access (may be NaN)
        - ``flagged`` (bool): True when any value column in that row is negative

    Raises
    ------
    ValueError
        When duplicate year values are detected after the merge.
    """
    consumption_df = raw_frames["consumption"].copy()
    access_df = raw_frames["access"].copy()

    # ------------------------------------------------------------------
    # 1. Outer-join on year
    # ------------------------------------------------------------------
    merged = pd.merge(consumption_df, access_df, on="year", how="outer")

    # ------------------------------------------------------------------
    # 2. Rename columns to human-readable names
    # ------------------------------------------------------------------
    rename_map = {}
    if _CONSUMPTION_CODE in merged.columns:
        rename_map[_CONSUMPTION_CODE] = _CONSUMPTION_COL
    if _ACCESS_CODE in merged.columns:
        rename_map[_ACCESS_CODE] = _ACCESS_COL

    merged = merged.rename(columns=rename_map)

    # ------------------------------------------------------------------
    # 3. Cast types: year → int, value columns → float
    # ------------------------------------------------------------------
    merged["year"] = merged["year"].astype(int)

    for col in (_CONSUMPTION_COL, _ACCESS_COL):
        if col in merged.columns:
            merged[col] = merged[col].astype(float)

    # ------------------------------------------------------------------
    # 4. Check for duplicate years (must happen AFTER type cast)
    # ------------------------------------------------------------------
    duplicate_years = merged.loc[merged["year"].duplicated(keep=False), "year"].unique().tolist()
    if duplicate_years:
        raise ValueError(
            f"[{_STEP_NAME}] Duplicate year values found after merge: {sorted(duplicate_years)}"
        )

    # ------------------------------------------------------------------
    # 5. Drop rows where BOTH value columns are NaN
    # ------------------------------------------------------------------
    both_nan_mask = merged[_CONSUMPTION_COL].isna() & merged[_ACCESS_COL].isna()
    merged = merged[~both_nan_mask].copy()

    # ------------------------------------------------------------------
    # 6. Add 'flagged' column for negative values, log warnings
    # ------------------------------------------------------------------
    consumption_negative = merged[_CONSUMPTION_COL].notna() & (merged[_CONSUMPTION_COL] < 0)
    access_negative = merged[_ACCESS_COL].notna() & (merged[_ACCESS_COL] < 0)
    flagged_mask = consumption_negative | access_negative

    merged["flagged"] = flagged_mask

    flagged_rows = merged[merged["flagged"]]
    if not flagged_rows.empty:
        for _, row in flagged_rows.iterrows():
            logger.warning(
                "[%s] Flagged row with negative value(s): year=%d, "
                "consumption_kwh_per_capita=%s, access_pct_of_population=%s",
                _STEP_NAME,
                row["year"],
                row[_CONSUMPTION_COL],
                row[_ACCESS_COL],
            )

    # ------------------------------------------------------------------
    # 7. Sort ascending by year
    # ------------------------------------------------------------------
    merged = merged.sort_values("year").reset_index(drop=True)

    # ------------------------------------------------------------------
    # 8. Save to data/processed/ethiopia_electricity_cleaned.csv
    # ------------------------------------------------------------------
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(src_dir)
    processed_dir = os.path.join(project_root, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    output_path = os.path.join(processed_dir, "ethiopia_electricity_cleaned.csv")

    merged.to_csv(output_path, index=False)
    logger.info(
        "[%s] Saved cleaned dataset (%d rows) to '%s'.",
        _STEP_NAME,
        len(merged),
        output_path,
    )

    return merged
