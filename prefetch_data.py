"""
Helper script: pre-fetch World Bank data and save raw CSVs.
Run from the project root before executing the notebook with nbconvert.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.data_collector import collect_all

try:
    raw_frames = collect_all(config)
    for name, df in raw_frames.items():
        print(f"{name}: shape={df.shape}, columns={list(df.columns)}")
    print("SUCCESS: raw CSVs saved to data/raw/")
except Exception as e:
    print(f"FAILED: {e}")
