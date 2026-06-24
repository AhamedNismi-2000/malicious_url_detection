#!/usr/bin/env python3
"""
split.py
--------
Splits cleaned_urls.csv into train / val / test BEFORE feature extraction.

No undersampling needed — dataset is already 1:1 balanced.

Split:
  Train : 70%
  Val   : 15%
  Test  : 15%

All splits are stratified to preserve 50/50 class ratio.

Output:
  data/splits/train_urls.csv
  data/splits/val_urls.csv
  data/splits/test_urls.csv
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------- PATHS ----------------
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
SPLITS_DIR   = os.path.join(BASE_DIR, "data", "splits")
os.makedirs(SPLITS_DIR, exist_ok=True)

INPUT_PATH = os.path.join(PROCESSED_DIR, "cleaned_urls.csv")
TRAIN_PATH = os.path.join(SPLITS_DIR, "train_urls.csv")
VAL_PATH   = os.path.join(SPLITS_DIR, "val_urls.csv")
TEST_PATH  = os.path.join(SPLITS_DIR, "test_urls.csv")

# ---------------- CONFIG ----------------
VAL_SIZE      = 0.15
TEST_SIZE     = 0.15
RANDOM_STATE  = 42


# ---------------- HELPERS ----------------
def print_distribution(name, df):
    counts    = df["label"].value_counts().sort_index()
    total     = len(df)
    benign    = counts.get(0, 0)
    malicious = counts.get(1, 0)
    print(f"\n   {name}:")
    print(f"     Total        : {total:>8,}")
    print(f"     Benign    (0): {benign:>8,}  ({100*benign/total:.1f}%)")
    print(f"     Malicious (1): {malicious:>8,}  ({100*malicious/total:.1f}%)")
    if malicious > 0:
        print(f"     Ratio  (B:M) : {benign/malicious:.2f}:1")


# ---------------- MAIN ----------------
def main():
    print("="*60)
    print("SPLIT PIPELINE")
    print("="*60)
    print("No undersampling — dataset is already 1:1 balanced")
    print("="*60)

    # Load cleaned data
    print("\nLoading cleaned_urls.csv...")
    if not os.path.exists(INPUT_PATH):
        print(f"ERROR: File not found: {INPUT_PATH}")
        print("Run preprocessing.py first.")
        return

    df = pd.read_csv(INPUT_PATH, dtype={"url": str, "label": int})
    df = df.dropna(subset=["url", "label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    print(f"Loaded: {len(df):,} URLs")
    print_distribution("Full dataset", df)

    # Verify labels
    unique_labels = set(df["label"].unique())
    if not unique_labels.issubset({0, 1}):
        print(f"ERROR: Unexpected labels: {unique_labels}")
        return

    # Step 1 — Separate test set
    print("\nStep 1: Separating test set (15%, stratified)...")
    df_trainval, df_test = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df["label"],
        random_state=RANDOM_STATE
    )
    print(f"   Train+Val: {len(df_trainval):,}  |  Test: {len(df_test):,}")

    # Step 2 — Separate val from train
    val_relative = VAL_SIZE / (1.0 - TEST_SIZE)
    print(f"\nStep 2: Separating val set ({val_relative:.1%} of train+val)...")
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=val_relative,
        stratify=df_trainval["label"],
        random_state=RANDOM_STATE
    )
    print(f"   Train: {len(df_train):,}  |  Val: {len(df_val):,}")

    # No undersampling needed — dataset already 1:1
    print("\nStep 3: No undersampling needed — dataset already 1:1 balanced")

    # Print final distributions
    print("\nFinal split distributions:")
    print_distribution("Train", df_train)
    print_distribution("Val  ", df_val)
    print_distribution("Test ", df_test)

    # Sanity checks
    print("\nSanity checks...")
    train_urls = set(df_train["url"])
    val_urls   = set(df_val["url"])
    test_urls  = set(df_test["url"])

    tv_overlap = train_urls & val_urls
    tt_overlap = train_urls & test_urls
    vt_overlap = val_urls   & test_urls

    if tv_overlap:
        print(f"   WARNING: Train/Val overlap: {len(tv_overlap):,} URLs")
    else:
        print(f"   No train/val overlap")

    if tt_overlap:
        print(f"   WARNING: Train/Test overlap: {len(tt_overlap):,} URLs")
    else:
        print(f"   No train/test overlap")

    if vt_overlap:
        print(f"   WARNING: Val/Test overlap: {len(vt_overlap):,} URLs")
    else:
        print(f"   No val/test overlap")

    print(f"   Total across splits: {len(df_train)+len(df_val)+len(df_test):,}")

    # Save splits
    print("\nSaving split files...")
    df_train.to_csv(TRAIN_PATH, index=False, encoding="utf-8")
    df_val.to_csv(VAL_PATH,     index=False, encoding="utf-8")
    df_test.to_csv(TEST_PATH,   index=False, encoding="utf-8")

    print(f"   {TRAIN_PATH}")
    print(f"   {VAL_PATH}")
    print(f"   {TEST_PATH}")

    print("\n"+"="*60)
    print("SPLIT COMPLETE")
    print("="*60)
    print("\nCorrect order:")
    print("  1. preprocessing.py    done")
    print("  2. split.py            done")
    print("  3. feature_extraction.py  <- next")
    print("  4. train_model.py")
    print("  5. test_model.py")


if __name__ == "__main__":
    main()