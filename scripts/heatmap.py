#!/usr/bin/env python3

"""
Correlation Heatmap — Malicious URL Detection

Creates a Pearson correlation heatmap for:
    57 heuristic features + binary target label

The 502 TF-IDF features are intentionally excluded because
a 559-feature correlation matrix would be difficult to interpret.

FIXES:
  1. Now calls load_tranco_whitelist() before feature extraction, so
     whitelist-dependent features (http_no_brand_no_age, leet_brand_score)
     match what was actually used during training instead of silently
     defaulting to an empty whitelist.
  2. All top-level execution wrapped in `if __name__ == "__main__":` so
     multiprocessing (used inside extract_heuristic_batch for >100k URLs)
     doesn't re-import and re-run this entire script in each worker
     process on spawn-based platforms (macOS/Windows, or Linux with the
     spawn start method).
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TRAIN_PATH = os.path.join(
    BASE_DIR,
    "data",
    "splits",
    "train_urls.csv"
)

# Your feature_extraction.py location
sys.path.insert(
    0,
    os.path.join(BASE_DIR, "scripts")
)

from feature_extraction import (
    extract_heuristic_batch,
    HEURISTIC_FEATURE_NAMES,
    load_tranco_whitelist,   # FIX 1: needed so whitelist-dependent features match training
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "correlation"
)

HEATMAP_PATH = os.path.join(
    OUTPUT_DIR,
    "heuristic_correlation_heatmap.png"
)

CORRELATION_CSV = os.path.join(
    OUTPUT_DIR,
    "heuristic_correlations.csv"
)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # LOAD TRAINING DATA
    # ---------------------------------------------------------

    print("=" * 60)
    print("MALICIOUS URL DETECTION — CORRELATION ANALYSIS")
    print("=" * 60)

    print("\nLoading training data...")

    df = pd.read_csv(
        TRAIN_PATH,
        dtype={
            "url": str,
            "label": int
        }
    )

    df = df.dropna(
        subset=["url", "label"]
    ).reset_index(drop=True)

    print(f"URLs loaded: {len(df):,}")

    # ---------------------------------------------------------
    # LOAD WHITELIST  (FIX 1)
    # ---------------------------------------------------------
    # extract_heuristic_batch() depends on features that call
    # is_whitelisted(), which reads a module-level set that is only
    # populated by this call. Without it, is_whitelisted() silently
    # falls back to an (almost) empty whitelist and the correlations
    # for http_no_brand_no_age / leet_brand_score won't reflect how
    # the model was actually trained.
    print("\nLoading Tranco whitelist...")
    load_tranco_whitelist()

    # ---------------------------------------------------------
    # EXTRACT 57 HEURISTIC FEATURES
    # ---------------------------------------------------------

    print("\nExtracting heuristic features...")

    X_heuristic = extract_heuristic_batch(
        df["url"].tolist()
    )

    print(
        f"Heuristic feature matrix: "
        f"{X_heuristic.shape}"
    )

    # Safety check
    assert X_heuristic.shape[1] == len(
        HEURISTIC_FEATURE_NAMES
    ), (
        f"Feature mismatch: "
        f"{X_heuristic.shape[1]} extracted, "
        f"{len(HEURISTIC_FEATURE_NAMES)} names"
    )

    # ---------------------------------------------------------
    # CREATE DATAFRAME
    # ---------------------------------------------------------

    feature_df = pd.DataFrame(
        X_heuristic,
        columns=HEURISTIC_FEATURE_NAMES
    )

    # Add target
    feature_df["label"] = df["label"].values

    print(
        f"Correlation dataframe shape: "
        f"{feature_df.shape}"
    )

    # ---------------------------------------------------------
    # PEARSON CORRELATION
    # ---------------------------------------------------------

    print("\nCalculating Pearson correlations...")

    corr = feature_df.corr(
        method="pearson"
    )

    # Save complete correlation matrix
    corr.to_csv(CORRELATION_CSV)

    print(
        f"Correlation matrix saved to:\n"
        f"{CORRELATION_CSV}"
    )

    # ---------------------------------------------------------
    # HEATMAP
    # ---------------------------------------------------------

    print("\nGenerating heatmap...")

    plt.figure(
        figsize=(24, 20)
    )

    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.2,
        cbar_kws={
            "label": "Pearson Correlation"
        },
        xticklabels=True,
        yticklabels=True
    )

    plt.title(
        "Pearson Correlation Heatmap of URL Heuristic Features",
        fontsize=18,
        pad=20
    )

    plt.xticks(
        rotation=90,
        fontsize=7
    )

    plt.yticks(
        rotation=0,
        fontsize=7
    )

    plt.tight_layout()

    plt.savefig(
        HEATMAP_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nHeatmap saved to:\n"
        f"{HEATMAP_PATH}"
    )

    # ---------------------------------------------------------
    # TARGET CORRELATIONS
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("FEATURE CORRELATION WITH TARGET")
    print("=" * 60)

    target_corr = (
        corr["label"]
        .drop("label")
        .sort_values(
            key=lambda x: x.abs(),
            ascending=False
        )
    )

    print("\nTop features associated with the target:\n")

    for feature, value in target_corr.head(15).items():
        print(
            f"{feature:<30} "
            f"{value:>8.4f}"
        )

    # ---------------------------------------------------------
    # HIGHLY CORRELATED FEATURE PAIRS
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("HIGHLY CORRELATED FEATURE PAIRS")
    print("=" * 60)

    # Only upper triangle to avoid duplicate pairs
    upper = corr.where(
        np.triu(
            np.ones(corr.shape),
            k=1
        ).astype(bool)
    )

    pairs = (
        upper
        .stack()
        .reset_index()
    )

    pairs.columns = [
        "Feature_1",
        "Feature_2",
        "Correlation"
    ]

    pairs["Abs_Correlation"] = (
        pairs["Correlation"].abs()
    )

    pairs = pairs.sort_values(
        "Abs_Correlation",
        ascending=False
    )

    high_corr = pairs[
        pairs["Abs_Correlation"] >= 0.80
    ]

    if len(high_corr) == 0:
        print(
            "\nNo feature pairs with "
            "|r| >= 0.80 were found."
        )
    else:
        print(
            f"\nFound {len(high_corr)} "
            f"highly correlated pairs:\n"
        )

        print(
            high_corr[
                [
                    "Feature_1",
                    "Feature_2",
                    "Correlation"
                ]
            ].to_string(
                index=False
            )
        )

    # Save pair analysis
    pairs.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "feature_correlation_pairs.csv"
        ),
        index=False
    )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()