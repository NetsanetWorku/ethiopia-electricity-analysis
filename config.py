"""
Central configuration for the Ethiopia Electricity Analysis project.
All tunable parameters are defined here so that every module imports
from this single source of truth.
"""

COUNTRY_CODE = "ET"

INDICATORS = {
    "consumption": "EG.USE.ELEC.KH.PC",
    "access":      "EG.ELC.ACCS.ZS",
}

START_YEAR = 1990
END_YEAR   = 2023

# Switch between "matplotlib" (default) and "plotly"
VIZ_LIBRARY = "matplotlib"

PATHS = {
    "raw":       "data/raw",
    "processed": "data/processed",
    "charts":    "charts",
    "reports":   "reports",
    "notebook":  "notebooks",
}
