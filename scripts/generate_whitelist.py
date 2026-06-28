#!/usr/bin/env python3
"""
generate_whitelist.py
---------------------
Generates models/whitelist.txt from Tranco top-1m.csv
Run once before starting Flask.
"""

import os
import pandas as pd

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRANCO_PATH    = os.path.join(BASE_DIR, "data", "raw", "top-1m.csv")
WHITELIST_PATH = os.path.join(BASE_DIR, "models", "whitelist.txt")

TOP_N = 50000  # top 50k domains

print(f"Loading Tranco list from {TRANCO_PATH}...")
df = pd.read_csv(TRANCO_PATH, header=None, names=["rank", "domain"], nrows=TOP_N)
df["domain"] = df["domain"].str.strip().str.lower()

print(f"Saving top {TOP_N:,} domains to {WHITELIST_PATH}...")
df["domain"].to_csv(WHITELIST_PATH, index=False, header=False)

print(f"Done — {len(df):,} domains saved to models/whitelist.txt")