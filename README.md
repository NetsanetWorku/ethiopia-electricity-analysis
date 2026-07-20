# Ethiopia Electricity Analysis

A Python-based data analysis of Ethiopia's electricity sector using World Bank open data.

**Author:** Netsanet Worku  
**Institution:** Madda Walabu University  
**Indicators:**
- Electric power consumption per capita — `EG.USE.ELEC.KH.PC`
- Access to electricity (% of population) — `EG.ELC.ACCS.ZS`

---

## Setup

1. **Clone / download** this repository and `cd` into the project root:
   ```bash
   cd ethiopia-electricity-analysis
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate
   ```

3. **Install dependencies** with pinned versions:
   ```bash
   pip install -r requirements.txt
   ```

---

## Data Download

Data is fetched automatically from the [World Bank API](https://api.worldbank.org/v2/) the first time the notebook or the data-collection script is run.

To download and cache the raw data manually:
```bash
python -c "import config; from src.data_collector import collect_all; collect_all(config)"
```

Raw CSV files are saved to `data/raw/`:
- `data/raw/EG.USE.ELEC.KH.PC.csv`
- `data/raw/EG.ELC.ACCS.ZS.csv`

If the API is unreachable, the pipeline falls back to these cached files automatically. If the files are also missing, a descriptive error is raised.

---

## Running the API

After installing dependencies, run the FastAPI application from the project root:
```bash
uvicorn app:app --host 0.0.0.0 --port 8501
```

Then open:
- http://localhost:8501 for the HTML landing page
- http://localhost:8501/docs for the interactive OpenAPI docs
- http://localhost:8501/api/summary for JSON summary results
- http://localhost:8501/api/data for the cleaned dataset
- http://localhost:8501/api/report for the generated Markdown report

## Vercel deployment

This repository includes a `vercel.json` file at the project root that routes all incoming requests to `app.py` and uses Vercel's Python runtime. After pushing to GitHub, deploy with the Vercel CLI:

```bash
vercel --prod
```

## Running the Notebook

Launch JupyterLab (or classic Jupyter Notebook) from the project root:
```bash
jupyter lab
# or
jupyter notebook
```

Open `notebooks/ethiopia_electricity.ipynb` and run **Kernel → Restart & Run All**.

The notebook executes the full pipeline in order:
1. Data collection
2. Data cleaning
3. Trend analysis
4. Visualization
5. Report generation

To execute the notebook non-interactively and produce an executed copy:
```bash
jupyter nbconvert --to notebook --execute \
    notebooks/ethiopia_electricity.ipynb \
    --output notebooks/ethiopia_electricity_executed.ipynb
```
---

## Running Tests

Run the full test suite from the project root:
```bash
pytest tests/ -v
```

Property-based tests (using [Hypothesis](https://hypothesis.readthedocs.io/)) are included alongside unit tests. To run only property tests:
```bash
pytest tests/ -v -k "property"
```

---

## Output Description

| Path | Description |
|------|-------------|
| `data/raw/EG.USE.ELEC.KH.PC.csv` | Raw API data — electricity consumption per capita |
| `data/raw/EG.ELC.ACCS.ZS.csv` | Raw API data — electricity access (% of population) |
| `data/processed/ethiopia_electricity_cleaned.csv` | Cleaned, merged, type-cast dataset sorted by year |
| `data/processed/ethiopia_electricity_summary.csv` | Summary statistics and trend results |
| `charts/consumption_kwh_per_capita.png` | Line chart — consumption over time (with trend line) |
| `charts/access_pct_of_population.png` | Line chart — access over time (with trend line) |
| `charts/*.html` | Interactive Plotly charts (only when `VIZ_LIBRARY = "plotly"`) |
| `reports/ethiopia_electricity_report.md` | Markdown analytical report with key findings |
| `notebooks/ethiopia_electricity.ipynb` | Source Jupyter Notebook |
| `notebooks/ethiopia_electricity_executed.ipynb` | Executed copy produced by nbconvert |

---

## Project Structure

```
ethiopia-electricity-analysis/
├── config.py                   # Central configuration (indicators, paths, options)
├── requirements.txt            # Pinned Python dependencies
├── README.md                   # This file
├── data/
│   ├── raw/                    # Raw CSVs downloaded from the World Bank API
│   └── processed/              # Cleaned dataset and summary statistics
├── charts/                     # Generated PNG (and optionally HTML) charts
├── notebooks/
│   └── ethiopia_electricity.ipynb
├── reports/
│   └── ethiopia_electricity_report.md
├── src/
│   ├── __init__.py
│   ├── data_collector.py
│   ├── data_cleaner.py
│   ├── trend_analyzer.py
│   ├── visualizer.py
│   └── report_generator.py
└── tests/
    ├── test_data_collector.py
    ├── test_data_cleaner.py
    ├── test_trend_analyzer.py
    ├── test_visualizer.py
    └── test_report_generator.py
```
