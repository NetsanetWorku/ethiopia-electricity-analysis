"""
report_generator.py — Generates a Markdown analytical report for the
Ethiopia Electricity Analysis project.

Public API:
    generate_report(df: pd.DataFrame, results: dict, config) -> str
"""

import logging
import os
from datetime import date

import pandas as pd

logger = logging.getLogger(__name__)

_STEP_NAME = "report_generator.generate_report"

# Human-readable labels for each indicator column
_INDICATOR_LABELS = {
    "consumption_kwh_per_capita": "Electric Power Consumption (kWh per capita)",
    "access_pct_of_population": "Access to Electricity (% of population)",
}

# Chart filenames produced by visualizer.py (matched by convention)
_CHART_FILES = {
    "consumption_kwh_per_capita": "consumption_kwh_per_capita.png",
    "access_pct_of_population": "access_pct_of_population.png",
}


def _trend_direction(slope) -> str:
    """Return a human-readable trend direction string given a slope value."""
    if slope is None:
        return "no trend data"
    if slope > 0:
        return "increasing"
    if slope < 0:
        return "decreasing"
    return "no trend data"


def generate_report(df: pd.DataFrame, results: dict, config) -> str:
    """
    Render a Markdown report from the analysis results and save it to disk.

    The report contains six sections:
        1. Introduction
        2. Data Sources
        3. Methodology
        4. Key Findings
        5. Visualizations
        6. Conclusion

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset (used for context; row count, year range, etc.).
    results : dict
        Analysis results dict as returned by ``trend_analyzer.analyze``.
        Expected structure::

            {
                "summary": {
                    "<col>": {
                        "mean": float,
                        "median": float,
                        "min": float,
                        "max": float,
                        "std": float,
                        "nan_count": int,
                        "peak_year": int | None,
                        "low_year": int | None,
                    },
                    ...
                },
                "trends": {
                    "<col>": {"slope": float, "intercept": float, "r_squared": float} | None,
                    ...
                },
            }
    config : module
        The project ``config`` module (used for indicator codes, country code,
        and paths — kept as a parameter for testability).

    Returns
    -------
    str
        The full Markdown string that was written to disk.

    Side Effects
    ------------
    Saves the report to ``reports/ethiopia_electricity_report.md``,
    resolved relative to this source file.
    """
    logger.info("[%s] Starting report generation.", _STEP_NAME)

    summary: dict = results.get("summary", {})
    trends: dict = results.get("trends", {})

    access_date = date.today().isoformat()

    # ------------------------------------------------------------------
    # Section 1: Introduction
    # ------------------------------------------------------------------
    intro = (
        "# Introduction\n\n"
        "This report presents an analysis of Ethiopia's electricity sector using publicly "
        "available data from the World Bank. Two key indicators are examined: electric power "
        "consumption per capita (kWh) and access to electricity as a percentage of the total "
        "population. The analysis covers all available years and aims to identify long-term "
        "trends that can inform policy discussion and academic research on Ethiopia's "
        "energy development trajectory.\n"
    )

    # ------------------------------------------------------------------
    # Section 2: Data Sources
    # ------------------------------------------------------------------
    consumption_code = config.INDICATORS.get("consumption", "EG.USE.ELEC.KH.PC")
    access_code = config.INDICATORS.get("access", "EG.ELC.ACCS.ZS")
    country_code = getattr(config, "COUNTRY_CODE", "ET")

    data_sources = (
        "# Data Sources\n\n"
        "All data were obtained from the **World Bank Open Data** platform via the "
        "World Bank JSON REST API (`https://api.worldbank.org/v2`).\n\n"
        "| Field | Detail |\n"
        "|-------|--------|\n"
        f"| Country | Ethiopia (country code: `{country_code}`) |\n"
        f"| Indicator 1 | Electric Power Consumption — `{consumption_code}` |\n"
        f"| Indicator 2 | Access to Electricity — `{access_code}` |\n"
        f"| Access Date | {access_date} |\n\n"
        "**Full citation:**\n\n"
        "> World Bank (various years). *Electric power consumption (kWh per capita)* "
        f"[{consumption_code}] and *Access to electricity (% of population)* [{access_code}]. "
        "Washington, D.C.: World Bank. "
        f"Retrieved {access_date} from https://data.worldbank.org/indicator/{consumption_code} "
        f"and https://data.worldbank.org/indicator/{access_code}.\n"
    )

    # ------------------------------------------------------------------
    # Section 3: Methodology
    # ------------------------------------------------------------------
    methodology = (
        "# Methodology\n\n"
        "The analysis pipeline consists of four sequential steps:\n\n"
        "1. **Data Collection** — Annual indicator values for Ethiopia were fetched from the "
        "World Bank API using the `data_collector` module. Raw responses were saved as CSV "
        "files in `data/raw/` for reproducibility. If the API was unreachable, local fallback "
        "files were used.\n\n"
        "2. **Data Cleaning** — The `data_cleaner` module merged the two indicator datasets "
        "on the `year` column (outer join), renamed columns to human-readable names, cast "
        "`year` to integer and value columns to float, dropped rows where both indicator "
        "values were missing, and sorted data in ascending year order. The cleaned dataset "
        "was saved to `data/processed/ethiopia_electricity_cleaned.csv`.\n\n"
        "3. **Trend Analysis** — The `trend_analyzer` module computed descriptive statistics "
        "(mean, median, min, max, standard deviation) and year-over-year percentage changes "
        "for each indicator. Where at least five non-NaN data points were available, a linear "
        "regression (using `scipy.stats.linregress`) was fitted to quantify the overall trend "
        "slope. Summary statistics were saved to `data/processed/ethiopia_electricity_summary.csv`.\n\n"
        "4. **Visualization** — The `visualizer` module produced line charts for each "
        "indicator over time, with trend lines overlaid where regression was available. "
        "Charts were saved as PNG files in the `charts/` directory.\n"
    )

    # ------------------------------------------------------------------
    # Section 4: Key Findings
    # ------------------------------------------------------------------
    findings_lines = ["# Key Findings\n"]

    for col, label in _INDICATOR_LABELS.items():
        col_summary = summary.get(col, {})
        col_trend = trends.get(col)

        mean_val = col_summary.get("mean", float("nan"))
        peak_year = col_summary.get("peak_year", None)

        slope = col_trend.get("slope") if col_trend else None
        direction = _trend_direction(slope)

        findings_lines.append(f"## {label}\n")

        # Mean
        if mean_val != mean_val:  # NaN check
            findings_lines.append("- **Mean:** N/A (no valid data)\n")
        else:
            findings_lines.append(f"- **Mean:** {mean_val:.4f}\n")

        # Trend slope
        if slope is None:
            findings_lines.append("- **Trend Slope:** no trend data\n")
            findings_lines.append("- **Trend Direction:** no trend data\n")
        else:
            findings_lines.append(f"- **Trend Slope:** {slope:.6f} per year\n")
            findings_lines.append(f"- **Trend Direction:** {direction}\n")

        # Peak year
        if peak_year is None:
            findings_lines.append("- **Peak Year:** N/A\n")
        else:
            findings_lines.append(f"- **Peak Year:** {peak_year}\n")

        findings_lines.append("\n")

    key_findings = "\n".join(findings_lines)

    # ------------------------------------------------------------------
    # Section 5: Visualizations
    # ------------------------------------------------------------------
    viz_lines = ["# Visualizations\n\n"]
    for col, label in _INDICATOR_LABELS.items():
        filename = _CHART_FILES.get(col, f"{col}.png")
        viz_lines.append(f"## {label}\n\n")
        viz_lines.append(f"![{label}](../charts/{filename})\n\n")

    visualizations = "\n".join(viz_lines)

    # ------------------------------------------------------------------
    # Section 6: Conclusion
    # ------------------------------------------------------------------
    conclusion = (
        "# Conclusion\n\n"
        "This analysis provides a data-driven overview of Ethiopia's electricity sector "
        "over the available historical period. The findings reveal the trajectory of both "
        "per capita electricity consumption and population-level access to electricity, "
        "highlighting long-term trends driven by economic growth, infrastructure expansion, "
        "and rural electrification efforts. The linear trend analysis offers a quantitative "
        "basis for projecting future developments. These insights can support evidence-based "
        "policy decisions aimed at achieving universal electricity access and sustainable "
        "energy consumption patterns in Ethiopia.\n"
    )

    # ------------------------------------------------------------------
    # Assemble full report
    # ------------------------------------------------------------------
    report_md = "\n\n".join([intro, data_sources, methodology, key_findings, visualizations, conclusion])

    # ------------------------------------------------------------------
    # Save report to disk
    # ------------------------------------------------------------------
    src_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(src_dir, "..")
    reports_dir = os.path.join(project_root, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    report_path = os.path.join(reports_dir, "ethiopia_electricity_report.md")

    try:
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write(report_md)
        logger.info("[%s] Report saved to '%s'.", _STEP_NAME, report_path)
    except OSError as exc:
        logger.error("[%s] Failed to save report: %s", _STEP_NAME, exc)
        raise

    return report_md
