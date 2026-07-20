"""
app.py — FastAPI application for the Ethiopia Electricity Analysis project.

Run with:
    uvicorn app:app --host 0.0.0.0 --port 8501
"""

import os
import sys
import logging
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

# Ensure project root is on sys.path so imports work regardless of the working directory.
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import config as _config
from src.data_collector import collect_all
from src.data_cleaner import clean_and_merge
from src.trend_analyzer import analyze
from src.report_generator import generate_report

logging.disable(logging.CRITICAL)

app = FastAPI(
    title="Ethiopia Electricity Analysis",
    description="FastAPI service exposing analysis results and generated charts for Ethiopia electricity data.",
)

# Make sure static asset directories exist and mount them for direct access.
for static_dir in ("charts", "reports"):
    os.makedirs(os.path.join(_APP_DIR, static_dir), exist_ok=True)

app.mount(
    "/charts",
    StaticFiles(directory=os.path.join(_APP_DIR, "charts")),
    name="charts",
)
app.mount(
    "/reports",
    StaticFiles(directory=os.path.join(_APP_DIR, "reports")),
    name="reports",
)

_pipeline_cache: dict[str, tuple[pd.DataFrame, dict, str]] = {}


def _build_config(viz_library: str) -> object:
    class _Cfg:
        COUNTRY_CODE = _config.COUNTRY_CODE
        INDICATORS = _config.INDICATORS
        START_YEAR = _config.START_YEAR
        END_YEAR = _config.END_YEAR
        VIZ_LIBRARY = viz_library
        PATHS = _config.PATHS

    return _Cfg


def _load_pipeline(viz_library: str = "matplotlib"):
    config = _build_config(viz_library)
    raw_frames = collect_all(config)
    df = clean_and_merge(raw_frames)
    results = analyze(df, config)

    if os.getenv("ENABLE_CHARTS", "false").lower() in ("1", "true", "yes"):
        from src.visualizer import generate_all_charts

        generate_all_charts(df, results, config)

    report_md = generate_report(df, results, config)
    return df, results, report_md


def get_pipeline_data(refresh: bool = False, viz_library: str = "matplotlib"):
    if refresh or "data" not in _pipeline_cache:
        _pipeline_cache["data"] = _load_pipeline(viz_library)
    return _pipeline_cache["data"]


@app.on_event("startup")
async def startup_event():
    get_pipeline_data()


def _format_metric(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _trend_direction(slope: Optional[float]) -> str:
    if slope is None:
        return "no trend data"
    if slope > 0:
        return "increasing"
    if slope < 0:
        return "decreasing"
    return "flat"


@app.get("/", response_class=HTMLResponse)
async def read_root():
    df, results, _ = get_pipeline_data()
    summary = results.get("summary", {})
    trends = results.get("trends", {})
    year_min = int(df["year"].min()) if not df.empty else "N/A"
    year_max = int(df["year"].max()) if not df.empty else "N/A"

    items = []
    for column, label in [
        ("consumption_kwh_per_capita", "Electric Power Consumption (kWh per capita)"),
        ("access_pct_of_population", "Access to Electricity (% population)"),
    ]:
        metrics = summary.get(column, {})
        trend = trends.get(column)
        items.append(
            f"<li><strong>{label}</strong><br>"
            f"Mean: {_format_metric(metrics.get('mean'))}<br>"
            f"Peak year: {_format_metric(metrics.get('peak_year'))}<br>"
            f"Trend: {_trend_direction(trend.get('slope') if trend else None)} "
            f"({_format_metric(trend.get('slope') if trend else None)} per year)</li>"
        )

    html = f"""
    <html>
      <head>
        <title>Ethiopia Electricity Analysis</title>
        <style>
          body {{ font-family: Arial, sans-serif; margin: 2rem; background: #f4f4f9; color: #1a1a1a; }}
          h1 {{ color: #1a6b2f; }}
          .card {{ background: white; border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: 1rem; box-shadow: 0 1px 6px rgba(0,0,0,0.08); }}
          a {{ color: #1a6b2f; text-decoration: none; }}
          a:hover {{ text-decoration: underline; }}
        </style>
      </head>
      <body>
        <h1>Ethiopia Electricity Analysis</h1>
        <div class="card">
          <p>Available years: {year_min}–{year_max}</p>
          <p>Endpoints:</p>
          <ul>
            <li><a href="/api/summary">/api/summary</a></li>
            <li><a href="/api/data">/api/data</a></li>
            <li><a href="/api/report">/api/report</a></li>
            <li><a href="/charts/consumption_kwh_per_capita.png">Consumption chart</a></li>
            <li><a href="/charts/access_pct_of_population.png">Access chart</a></li>
            <li><a href="/reports/ethiopia_electricity_report.md">Generated report</a></li>
          </ul>
        </div>
        <div class="card">
          <h2>Summary metrics</h2>
          <ul>{''.join(items)}</ul>
        </div>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/api/summary")
async def get_summary():
    _, results, _ = get_pipeline_data()
    return JSONResponse(content={"summary": results.get("summary"), "trends": results.get("trends")})


@app.get("/api/data")
async def get_data(
    year_min: Optional[int] = Query(None, description="Minimum year to filter"),
    year_max: Optional[int] = Query(None, description="Maximum year to filter"),
):
    df, _, _ = get_pipeline_data()
    filtered = df.copy()
    if year_min is not None:
        filtered = filtered[filtered["year"] >= year_min]
    if year_max is not None:
        filtered = filtered[filtered["year"] <= year_max]
    return JSONResponse(content={"data": filtered.to_dict(orient="records")})


@app.get("/api/report", response_class=PlainTextResponse)
async def get_report():
    report_path = os.path.join(_APP_DIR, "reports", "ethiopia_electricity_report.md")
    if not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report not found")
    with open(report_path, "r", encoding="utf-8") as fh:
        return PlainTextResponse(content=fh.read())


@app.post("/api/refresh")
async def refresh_data():
    get_pipeline_data(refresh=True)
    return JSONResponse(content={"status": "refreshed"})
