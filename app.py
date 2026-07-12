"""
app.py — Streamlit dashboard for the Ethiopia Electricity Analysis project.

Run with:
    streamlit run app.py
"""

import os
import sys
import logging

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
# Ensure project root is on sys.path so `import config` and `from src.*` work
# regardless of where streamlit is launched from.
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import config as _config
from src.data_collector import collect_all
from src.data_cleaner import clean_and_merge
from src.trend_analyzer import analyze
from src.report_generator import generate_report

logging.disable(logging.CRITICAL)  # suppress pipeline logs in the UI

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ethiopia Electricity Analysis",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .metric-card {
        background: #f0f4f8;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }
    .section-divider { margin-top: 2rem; margin-bottom: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/7/71/Flag_of_Ethiopia.svg/320px-Flag_of_Ethiopia.svg.png",
        width=120,
    )
    st.title("⚡ Ethiopia\nElectricity Analysis")
    st.markdown("---")

    st.subheader("Settings")
    viz_library = st.radio(
        "Chart library",
        options=["matplotlib", "plotly"],
        index=0,
        help="Plotly charts are interactive; matplotlib charts are static.",
    )

    st.markdown("---")
    st.subheader("Year Range Filter")
    year_min_global = 1960
    year_max_global = 2024
    year_range = st.slider(
        "Select years to display",
        min_value=year_min_global,
        max_value=year_max_global,
        value=(1990, 2023),
        step=1,
    )

    st.markdown("---")
    run_btn = st.button("🔄  Refresh Data", use_container_width=True)
    st.caption("Fetches latest data from the World Bank API.\nFalls back to local CSVs if offline.")


# ── Data loading (cached) ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_pipeline(viz_lib: str):
    """Run the full pipeline and return (df, results, report_md)."""
    # Build a lightweight config shim so we can override VIZ_LIBRARY at runtime
    class _Cfg:
        COUNTRY_CODE = _config.COUNTRY_CODE
        INDICATORS   = _config.INDICATORS
        START_YEAR   = _config.START_YEAR
        END_YEAR     = _config.END_YEAR
        VIZ_LIBRARY  = viz_lib
        PATHS        = _config.PATHS

    raw    = collect_all(_Cfg)
    df     = clean_and_merge(raw)
    results = analyze(df, _Cfg)
    report  = generate_report(df, results, _Cfg)
    return df, results, report


# Bust cache when user clicks Refresh
if run_btn:
    st.cache_data.clear()

with st.spinner("Loading data and running analysis…"):
    try:
        df_full, results, report_md = load_pipeline(viz_library)
    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")
        st.stop()

# Apply year-range filter for display (does not affect saved outputs)
df = df_full[
    (df_full["year"] >= year_range[0]) & (df_full["year"] <= year_range[1])
].copy()

summary = results["summary"]
trends  = results["trends"]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚡ Ethiopia Electricity Analysis")
st.markdown(
    "Interactive dashboard powered by **World Bank Open Data**. "
    "Use the sidebar to adjust the year range and chart style."
)
st.markdown("---")

# ── KPI cards ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

cons_summary = summary.get("consumption_kwh_per_capita", {})
acc_summary  = summary.get("access_pct_of_population",  {})

with col1:
    st.metric(
        label="⚡ Avg Consumption (kWh/capita)",
        value=f"{cons_summary.get('mean', float('nan')):.1f}",
        delta=f"Peak: {int(cons_summary.get('peak_year', 0))}",
    )
with col2:
    cons_trend = trends.get("consumption_kwh_per_capita")
    slope_cons = cons_trend["slope"] if cons_trend else None
    st.metric(
        label="📈 Consumption Trend (slope/yr)",
        value=f"{slope_cons:.3f}" if slope_cons is not None else "N/A",
        delta="increasing" if slope_cons and slope_cons > 0 else "decreasing" if slope_cons else "",
        delta_color="normal",
    )
with col3:
    st.metric(
        label="💡 Avg Access (% population)",
        value=f"{acc_summary.get('mean', float('nan')):.1f}%",
        delta=f"Peak: {int(acc_summary.get('peak_year', 0))}",
    )
with col4:
    acc_trend = trends.get("access_pct_of_population")
    slope_acc = acc_trend["slope"] if acc_trend else None
    st.metric(
        label="📈 Access Trend (slope/yr)",
        value=f"{slope_acc:.3f}" if slope_acc is not None else "N/A",
        delta="increasing" if slope_acc and slope_acc > 0 else "decreasing" if slope_acc else "",
        delta_color="normal",
    )

st.markdown("---")

# ── Charts ────────────────────────────────────────────────────────────────────
st.subheader("📊 Indicator Charts")

chart_tab1, chart_tab2 = st.tabs([
    "⚡ Electric Power Consumption",
    "💡 Access to Electricity",
])

def _render_chart(tab, df_plot, column, ylabel, title, trend):
    with tab:
        valid = df_plot.dropna(subset=[column])
        if len(valid) < 2:
            st.warning(f"Not enough data points in the selected year range to plot '{column}'.")
            return

        if viz_library == "plotly":
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=valid["year"], y=valid[column],
                mode="lines+markers", name=ylabel,
                line=dict(color="#1f77b4", width=2),
                marker=dict(size=5),
            ))
            if trend:
                x_arr = np.arange(int(valid["year"].min()), int(valid["year"].max()) + 1)
                y_arr = trend["slope"] * x_arr + trend["intercept"]
                fig.add_trace(go.Scatter(
                    x=x_arr, y=y_arr,
                    mode="lines", name="Linear trend",
                    line=dict(dash="dash", color="red", width=1.5),
                ))
            fig.update_layout(
                title=title,
                xaxis_title="Year",
                yaxis_title=ylabel,
                legend=dict(orientation="h", y=-0.2),
                height=420,
                margin=dict(t=50, b=60),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(valid["year"], valid[column],
                    marker="o", linewidth=1.8, markersize=4,
                    label=ylabel, color="#1f77b4")
            if trend:
                x_arr = np.arange(int(valid["year"].min()), int(valid["year"].max()) + 1)
                y_arr = trend["slope"] * x_arr + trend["intercept"]
                ax.plot(x_arr, y_arr, linestyle="--", linewidth=1.4,
                        color="red", label="Linear trend")
            ax.set_xlabel("Year")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

_render_chart(
    chart_tab1, df,
    column="consumption_kwh_per_capita",
    ylabel="kWh per Capita",
    title="Ethiopia: Electric Power Consumption per Capita",
    trend=trends.get("consumption_kwh_per_capita"),
)
_render_chart(
    chart_tab2, df,
    column="access_pct_of_population",
    ylabel="% of Population",
    title="Ethiopia: Access to Electricity (% of Population)",
    trend=trends.get("access_pct_of_population"),
)

st.markdown("---")

# ── Summary statistics table ──────────────────────────────────────────────────
st.subheader("📋 Summary Statistics")

stats_rows = []
for col, label in [
    ("consumption_kwh_per_capita", "Consumption (kWh/capita)"),
    ("access_pct_of_population",   "Access (% population)"),
]:
    s = summary.get(col, {})
    t = trends.get(col) or {}
    stats_rows.append({
        "Indicator":       label,
        "Mean":            f"{s.get('mean', float('nan')):.3f}",
        "Median":          f"{s.get('median', float('nan')):.3f}",
        "Min":             f"{s.get('min', float('nan')):.3f}",
        "Max":             f"{s.get('max', float('nan')):.3f}",
        "Std Dev":         f"{s.get('std', float('nan')):.3f}",
        "Peak Year":       str(int(s["peak_year"])) if s.get("peak_year") else "N/A",
        "Low Year":        str(int(s["low_year"])) if s.get("low_year") else "N/A",
        "Trend Slope/yr":  f"{t['slope']:.4f}" if t.get("slope") is not None else "N/A",
        "R²":              f"{t['r_squared']:.4f}" if t.get("r_squared") is not None else "N/A",
        "Missing Rows":    str(int(s.get("nan_count", 0))),
    })

st.dataframe(
    pd.DataFrame(stats_rows).set_index("Indicator"),
    use_container_width=True,
)

st.markdown("---")

# ── Raw data explorer ─────────────────────────────────────────────────────────
st.subheader("🗂️ Raw Data Explorer")

with st.expander("Show / hide data table", expanded=False):
    display_cols = ["year", "consumption_kwh_per_capita", "access_pct_of_population", "flagged"]
    st.dataframe(
        df[display_cols].rename(columns={
            "year":                       "Year",
            "consumption_kwh_per_capita": "Consumption (kWh/capita)",
            "access_pct_of_population":   "Access (% population)",
            "flagged":                    "Flagged",
        }),
        use_container_width=True,
        height=300,
    )

    csv_bytes = df[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️  Download CSV",
        data=csv_bytes,
        file_name="ethiopia_electricity_data.csv",
        mime="text/csv",
    )

st.markdown("---")

# ── Report viewer & download ──────────────────────────────────────────────────
st.subheader("📄 Generated Report")

with st.expander("Show / hide Markdown report", expanded=False):
    st.markdown(report_md)

st.download_button(
    label="⬇️  Download Full Report (.md)",
    data=report_md.encode("utf-8"),
    file_name="ethiopia_electricity_report.md",
    mime="text/markdown",
)

st.markdown("---")

# ── Footer ────────────────────────────────────────────────────────────────────
st.caption(
    "Data source: [World Bank Open Data](https://data.worldbank.org) · "
    "Indicators: EG.USE.ELEC.KH.PC, EG.ELC.ACCS.ZS · "
    "Author: Netsanet Worku, Madda Walabu University"
)
