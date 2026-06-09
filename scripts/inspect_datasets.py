#!/usr/bin/env python3
"""
inspect_datasets.py
-------------------
Inspects all datasets in data/raw/ and prints a quality report.

Run:
  python scripts/inspect_datasets.py

Then paste the output so dataset quality can be verified
before running the full pipeline.
"""

import os
import sys
import pandas as pd
import numpy as np
from urllib.parse import urlparse
import tldextract

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")

EXTRACTOR = tldextract.TLDExtract(cache_dir=None, suffix_list_urls=None)

# ---------------- HELPERS ----------------
def has_path(url: str) -> bool:
    """Check if URL has a real path beyond just /"""
    try:
        p = urlparse(str(url)).path
        return bool(p) and p != "/"
    except Exception:
        return False

def has_subdomain(url: str) -> bool:
    """Check if URL has a subdomain."""
    try:
        ext = EXTRACTOR(urlparse(str(url)).netloc)
        return bool(ext.subdomain)
    except Exception:
        return False

def has_scheme(url: str) -> bool:
    """Check if URL has http:// or https://"""
    try:
        return str(url).startswith(("http://", "https://"))
    except Exception:
        return False

def url_length_stats(urls: pd.Series) -> dict:
    lengths = urls.astype(str).apply(len)
    return {
        "min"   : int(lengths.min()),
        "max"   : int(lengths.max()),
        "mean"  : round(float(lengths.mean()), 1),
        "median": int(lengths.median())
    }

def inspect_file(filepath: str, filename: str):
    """Inspect a single dataset file and print quality report."""
    print(f"\n{'='*65}")
    print(f"FILE: {filename}")
    print(f"{'='*65}")

    # Try reading the file
    try:
        # Try comma separator first
        try:
            df = pd.read_csv(filepath, dtype=str, low_memory=False)
        except Exception:
            # Try tab separator
            df = pd.read_csv(filepath, dtype=str,
                             low_memory=False, sep="\t")
    except Exception as e:
        print(f"  ERROR reading file: {e}")
        return

    print(f"  Shape       : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Columns     : {list(df.columns)}")

    # Detect URL column
    url_col = None
    for candidate in ["url", "URL", "Url", "phish_url",
                       "domain", "Domain", "address"]:
        if candidate in df.columns:
            url_col = candidate
            break

    if url_col is None:
        # Try first column
        url_col = df.columns[0]
        print(f"  URL column  : guessing '{url_col}' (first column)")
    else:
        print(f"  URL column  : '{url_col}'")

    # Detect label column
    label_col = None
    for candidate in ["label", "Label", "type", "Type",
                       "class", "Class", "category", "status"]:
        if candidate in df.columns:
            label_col = candidate
            break

    if label_col is None and df.shape[1] > 1:
        label_col = df.columns[1]
        print(f"  Label column: guessing '{label_col}' (second column)")
    elif label_col:
        print(f"  Label column: '{label_col}'")
    else:
        print(f"  Label column: NOT FOUND — single column file")

    # Label distribution
    if label_col:
        label_counts = df[label_col].value_counts()
        print(f"\n  Label distribution:")
        for label, count in label_counts.items():
            pct = 100 * count / len(df)
            print(f"    {str(label):<20}: {count:>8,}  ({pct:.1f}%)")

    # URL quality analysis
    urls = df[url_col].dropna().astype(str)
    print(f"\n  URL quality ({len(urls):,} non-null URLs):")

    # Check URL format
    with_scheme    = urls.apply(has_scheme).sum()
    with_path      = urls.apply(has_path).sum()
    with_subdomain = urls.apply(has_subdomain).sum()

    print(f"    Has http/https scheme : "
          f"{with_scheme:>8,}  ({100*with_scheme/len(urls):.1f}%)")
    print(f"    Has path (not just /) : "
          f"{with_path:>8,}  ({100*with_path/len(urls):.1f}%)")
    print(f"    Has subdomain         : "
          f"{with_subdomain:>8,}  ({100*with_subdomain/len(urls):.1f}%)")

    # URL length stats
    length_stats = url_length_stats(urls)
    print(f"    URL length — "
          f"min:{length_stats['min']} "
          f"max:{length_stats['max']} "
          f"mean:{length_stats['mean']} "
          f"median:{length_stats['median']}")

    # Null/empty check
    null_count  = df[url_col].isna().sum()
    empty_count = (df[url_col].astype(str).str.strip() == "").sum()
    print(f"    Null URLs             : {null_count:>8,}")
    print(f"    Empty URLs            : {empty_count:>8,}")

    # Duplicate check
    dup_count = urls.duplicated().sum()
    print(f"    Duplicate URLs        : {dup_count:>8,}  "
          f"({100*dup_count/len(urls):.1f}%)")

    # Sample URLs per label
    print(f"\n  Sample URLs:")
    if label_col:
        for label in df[label_col].dropna().unique()[:4]:
            subset = df[df[label_col] == label][url_col].dropna()
            samples = subset.sample(min(3, len(subset)),
                                    random_state=42).tolist()
            print(f"\n    Label = '{label}':")
            for s in samples:
                display = s if len(s) <= 60 else s[:57] + "..."
                print(f"      {display}")
    else:
        samples = urls.sample(min(5, len(urls)), random_state=42).tolist()
        for s in samples:
            display = s if len(s) <= 60 else s[:57] + "..."
            print(f"    {display}")

    # Quality verdict
    print(f"\n  Quality Assessment:")

    issues = []
    good   = []

    if with_scheme / len(urls) < 0.5:
        issues.append("Less than 50% URLs have http/https scheme (bare domains)")
    else:
        good.append("Most URLs have proper scheme")

    if with_path / len(urls) < 0.3:
        issues.append("Less than 30% URLs have a path (too many bare domains)")
    else:
        good.append("Good path coverage")

    if dup_count / len(urls) > 0.2:
        issues.append(f"High duplicate rate ({100*dup_count/len(urls):.1f}%)")
    else:
        good.append("Low duplicate rate")

    if null_count > len(df) * 0.1:
        issues.append(f"High null rate ({100*null_count/len(df):.1f}%)")
    else:
        good.append("Low null rate")

    if label_col is None:
        issues.append("No label column found")
    elif len(df[label_col].unique()) < 2:
        issues.append("Only one label value — single class dataset")
    else:
        good.append("Has multiple label classes")

    for g in good:
        print(f"    GOOD   : {g}")
    for i in issues:
        print(f"    ISSUE  : {i}")

    if not issues:
        print(f"    VERDICT: Usable as-is")
    elif len(issues) == 1 and "bare domains" in issues[0]:
        print(f"    VERDICT: Usable with URL prefix fix")
    else:
        print(f"    VERDICT: Needs attention before use")


# ---------------- MAIN ----------------
def main():
    print("=" * 65)
    print("DATASET QUALITY INSPECTOR")
    print("=" * 65)
    print(f"Scanning: {RAW_DIR}")

    # Find all CSV and TXT files
    files = []
    for f in sorted(os.listdir(RAW_DIR)):
        if f.endswith((".csv", ".txt", ".tsv")):
            files.append(f)

    if not files:
        print(f"\nNo CSV/TXT files found in {RAW_DIR}")
        return

    print(f"Found {len(files)} files: {files}\n")

    # Inspect each file
    for filename in files:
        filepath = os.path.join(RAW_DIR, filename)
        inspect_file(filepath, filename)

    # Final summary
    print(f"\n{'='*65}")
    print(f"INSPECTION COMPLETE")
    print(f"{'='*65}")
    print(f"Paste this full output to identify which datasets are usable.")
    print(f"Total files inspected: {len(files)}")


if __name__ == "__main__":
    main()