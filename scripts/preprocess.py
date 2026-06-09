#!/usr/bin/env python3
"""
preprocessing.py
----------------
Loads verified research-grade datasets + data.csv for both classes.

Sources:
  Malicious:
    Phish.csv     - PhishTank verified phishing (verified=yes only)
    urlhaus.txt   - URLhaus malware (abuse.ch verified)
    openphish.txt - OpenPhish active phishing
    data.csv      - faizann24 bad URLs (75K malicious with paths)

  Benign:
    data.csv      - faizann24 good URLs (344K benign with real paths)
    top-1m.csv    - Alexa top domains (small sample for variety)

Why data.csv for both classes:
  Bad URLs  : malicious URLs with real paths — fills gap in training
  Good URLs : benign URLs with real paths, subdomains, query strings
  Together they teach the model path/subdomain patterns for both classes

Output:
  data/processed/cleaned_urls.csv
  data/processed/cleaned_urls_reference.csv
"""

import os
import re
import pandas as pd
import tldextract
from urllib.parse import urlparse

# ---------------- PATHS ----------------
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR       = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

PHISH_PATH     = os.path.join(RAW_DIR, "Phish.csv")
URLHAUS_PATH   = os.path.join(RAW_DIR, "urlhaus.txt")
OPENPHISH_PATH = os.path.join(RAW_DIR, "openphish.txt")
ALEXA_PATH     = os.path.join(RAW_DIR, "top-1m.csv")
DATA_PATH      = os.path.join(RAW_DIR, "data.csv")

OUTPUT_ML        = os.path.join(PROCESSED_DIR, "cleaned_urls.csv")
OUTPUT_REFERENCE = os.path.join(PROCESSED_DIR, "cleaned_urls_reference.csv")

# Infrastructure domains to skip from Alexa
SKIP_DOMAINS = {
    "googletagmanager.com", "doubleclick.net", "googleapis.com",
    "gstatic.com", "cloudflare.com", "akamaized.net",
    "fastly.net", "cloudfront.net", "amazonaws.com",
    "gtld-servers.net", "root-servers.net"
}

# ---------------- HELPERS ----------------
def normalize_url(u: str) -> str:
    try:
        u = str(u).strip().lower()
        u = re.sub(r"\s+", "", u)
        markdown_match = re.search(r'\(https?://[^\)]+\)', u)
        if markdown_match:
            u = markdown_match.group(0)[1:-1]
        if not re.match(r"https?://", u):
            u = "http://" + u
        if u.endswith("/") and len(u) > 8:
            u = u[:-1]
        return u
    except Exception:
        return None


def valid_domain(url: str) -> bool:
    try:
        parsed   = urlparse(url)
        if not parsed.netloc:
            return False
        hostname  = parsed.netloc.split(":")[0]
        extracted = tldextract.extract(hostname)
        return bool(extracted.domain) and bool(extracted.suffix)
    except Exception:
        return False


def resolve_label_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    diversity     = df.groupby("url")["label"].nunique()
    conflict_urls = diversity[diversity > 1].index
    n_conflicts   = len(conflict_urls)
    if n_conflicts > 0:
        print(f"  Removing {n_conflicts:,} URLs with conflicting labels")
        df = df[~df["url"].isin(conflict_urls)]
    return df.drop_duplicates(subset=["url"]).reset_index(drop=True)


def convert_labels(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "benign"    : 0,
        "legitimate": 0,
        "good"      : 0,
        "safe"      : 0,
        "malicious" : 1,
        "phishing"  : 1,
        "malware"   : 1,
        "defacement": 1,
        "bad"       : 1,
    }
    df          = df.copy()
    df["label"] = df["label"].str.lower().map(mapping)
    before      = len(df)
    df          = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    removed     = before - len(df)
    if removed > 0:
        print(f"  Removed {removed:,} URLs with unrecognized labels")
    return df


def clean_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
    print(f"  Cleaning {name}...")
    df["url"] = df["url"].apply(normalize_url)
    df        = df[df["url"].notnull()]
    df        = df[df["url"].apply(valid_domain)]
    df        = df.drop_duplicates(subset=["url"]).reset_index(drop=True)
    print(f"  After cleaning: {len(df):,}")
    return df


def add_scheme(url: str) -> str:
    """Add http:// if URL has no scheme."""
    u = str(url).strip()
    if not u.startswith(("http://", "https://")):
        return "http://" + u
    return u


# ---------------- LOADERS ----------------

def load_phishtank() -> pd.DataFrame:
    print("Loading Phish.csv (PhishTank)...")
    try:
        df = pd.read_csv(PHISH_PATH, dtype=str, low_memory=False)
        if "verified" in df.columns:
            df = df[df["verified"].str.lower() == "yes"]
        url_col = "url" if "url" in df.columns else df.columns[1]
        out = pd.DataFrame({
            "url"  : df[url_col].astype(str),
            "label": "phishing"
        })
        out = out.dropna(subset=["url"])
        print(f"  Loaded: {len(out):,}")
        return out[["url", "label"]]
    except Exception as e:
        print(f"  ERROR: {e}")
        return pd.DataFrame(columns=["url", "label"])


def load_urlhaus() -> pd.DataFrame:
    print("Loading urlhaus.txt...")
    try:
        df   = pd.read_csv(
            URLHAUS_PATH, comment="#",
            header=None, dtype=str,
            on_bad_lines="skip"
        )
        col  = 2 if df.shape[1] >= 3 else 0
        urls = df.iloc[:, col].dropna().astype(str)
        urls = urls[urls.str.startswith(("http://", "https://"))]
        out  = pd.DataFrame({"url": urls, "label": "malware"})
        print(f"  Loaded: {len(out):,}")
        return out[["url", "label"]]
    except Exception as e:
        print(f"  ERROR: {e}")
        return pd.DataFrame(columns=["url", "label"])


def load_openphish() -> pd.DataFrame:
    print("Loading openphish.txt...")
    try:
        with open(OPENPHISH_PATH, encoding="utf-8", errors="ignore") as f:
            lines = [
                l.strip() for l in f
                if l.strip().startswith(("http://", "https://"))
            ]
        out = pd.DataFrame({"url": lines, "label": "phishing"})
        print(f"  Loaded: {len(out):,}")
        return out[["url", "label"]]
    except Exception as e:
        print(f"  ERROR: {e}")
        return pd.DataFrame(columns=["url", "label"])


def load_data_csv_malicious() -> pd.DataFrame:
    """
    Load malicious URLs from data.csv (faizann24 dataset).
    Contains 75K bad URLs with real paths.
    These fill the gap — malicious URLs with path patterns
    that PhishTank/URLhaus may not cover.
    """
    print("Loading data.csv (malicious only)...")
    try:
        df = pd.read_csv(
            DATA_PATH,
            dtype=str,
            header=None,
            names=["url", "label"],
            skiprows=1       # skip duplicate header row
        )
        bad = df[df["label"].str.lower() == "bad"].copy()
        bad["label"] = "malicious"
        bad["url"]   = bad["url"].apply(add_scheme)
        bad          = bad.dropna(subset=["url"])
        print(f"  Loaded: {len(bad):,} malicious URLs with paths")
        return bad[["url", "label"]]
    except Exception as e:
        print(f"  ERROR: {e}")
        return pd.DataFrame(columns=["url", "label"])


def load_data_csv_benign() -> pd.DataFrame:
    """
    Load benign URLs from data.csv (faizann24 dataset).
    Contains 344K good URLs with real paths and subdomains.
    This is the primary benign source — has structural variety.
    """
    print("Loading data.csv (benign only)...")
    try:
        df = pd.read_csv(
            DATA_PATH,
            dtype=str,
            header=None,
            names=["url", "label"],
            skiprows=1
        )
        good = df[df["label"].str.lower() == "good"].copy()
        good["label"] = "benign"
        good["url"]   = good["url"].apply(add_scheme)
        good          = good.dropna(subset=["url"])
        print(f"  Loaded: {len(good):,} benign URLs with paths")
        return good[["url", "label"]]
    except Exception as e:
        print(f"  ERROR: {e}")
        return pd.DataFrame(columns=["url", "label"])


def load_alexa(n: int) -> pd.DataFrame:
    """
    Load top N Alexa domains as benign.
    Small sample for variety — bare domain format.
    """
    print(f"Loading top-1m.csv (top {n:,} for variety)...")
    try:
        df = pd.read_csv(
            ALEXA_PATH,
            dtype=str,
            header=None,
            names=["rank", "domain"],
            nrows=n
        )
        df["domain"] = df["domain"].str.strip().str.lower()
        df = df[~df["domain"].isin(SKIP_DOMAINS)]
        df["url"]    = "https://www." + df["domain"]
        df["label"]  = "benign"
        print(f"  Loaded: {len(df):,}")
        return df[["url", "label"]]
    except Exception as e:
        print(f"  ERROR: {e}")
        return pd.DataFrame(columns=["url", "label"])


# ---------------- MAIN ----------------
def main():
    print("=" * 60)
    print("PREPROCESSING — FULL VERIFIED PIPELINE")
    print("=" * 60)
    print("Malicious: PhishTank + URLhaus + OpenPhish + data.csv bad")
    print("Benign   : data.csv good (real paths) + Alexa (variety)")
    print("=" * 60)

    # Step 1 — Load all malicious sources
    print("\nStep 1: Loading malicious datasets...")
    malicious = pd.concat(
        [
            load_phishtank(),
            load_urlhaus(),
            load_openphish(),
            load_data_csv_malicious()   # NEW — adds path variety
        ],
        ignore_index=True
    )
    malicious   = clean_df(malicious, "malicious")
    n_malicious = len(malicious)
    print(f"\n  Total malicious: {n_malicious:,}")

    # Step 2 — Load benign sources
    print(f"\nStep 2: Loading benign datasets...")
    data_benign = load_data_csv_benign()
    data_benign = clean_df(data_benign, "data.csv benign")

    # Small Alexa sample for variety (bare domains)
    alexa = load_alexa(n=20000)
    alexa = clean_df(alexa, "alexa benign")

    # Combine benign
    benign = pd.concat([data_benign, alexa], ignore_index=True)
    benign = benign.drop_duplicates(
        subset=["url"]
    ).reset_index(drop=True)
    print(f"\n  Total benign available: {len(benign):,}")

    # Sample benign to match malicious for 1:1 balance
    if len(benign) > n_malicious:
        benign = benign.sample(
            n=n_malicious, random_state=42
        ).reset_index(drop=True)
        print(f"  Sampled benign to: {len(benign):,} (1:1 balance)")
    else:
        print(f"  Using all {len(benign):,} benign URLs")

    print(f"\n  Final malicious: {n_malicious:,}")
    print(f"  Final benign   : {len(benign):,}")

    # Step 3 — Combine
    print("\nStep 3: Combining all datasets...")
    combined = pd.concat([malicious, benign], ignore_index=True)
    print(f"  Combined total: {len(combined):,}")

    # Step 4 — Resolve label conflicts
    print("\nStep 4: Resolving label conflicts...")
    combined = resolve_label_conflicts(combined)
    print(f"  After conflicts: {len(combined):,}")

    # Step 5 — Save reference (string labels)
    reference_df = combined.copy()

    # Step 6 — Convert labels to numeric
    print("\nStep 5: Converting labels to numeric...")
    ml_df = convert_labels(combined)

    # Step 7 — Shuffle both consistently
    ml_df = ml_df.sample(
        frac=1, random_state=42
    ).reset_index(drop=True)
    reference_df = reference_df.sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    # Step 8 — Save
    print("\nStep 6: Saving files...")
    ml_df.to_csv(OUTPUT_ML, index=False, encoding="utf-8")
    reference_df.to_csv(OUTPUT_REFERENCE, index=False, encoding="utf-8")

    # Step 9 — Report
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)

    counts       = ml_df["label"].value_counts().sort_index()
    benign_count = counts.get(0, 0)
    mal_count    = counts.get(1, 0)
    total        = len(ml_df)

    print(f"\nFinal dataset  : {total:,} URLs")
    print(f"Benign    (0)  : {benign_count:>8,}  ({100*benign_count/total:.1f}%)")
    print(f"Malicious (1)  : {mal_count:>8,}  ({100*mal_count/total:.1f}%)")
    print(f"Ratio          : {benign_count/max(mal_count,1):.2f}:1")

    print(f"\nMalicious breakdown:")
    ref_counts = reference_df["label"].value_counts()
    for lbl in ["phishing", "malware", "malicious"]:
        n = ref_counts.get(lbl, 0)
        if n > 0:
            print(f"  {lbl:<12}: {n:>8,}")

    print(f"\nBenign breakdown:")
    print(f"  data.csv good: {len(data_benign):>8,}  (real paths)")
    print(f"  Alexa        : {len(alexa):>8,}  (bare domains)")

    print(f"\nOutput: {OUTPUT_ML}")
    print("Next  : python scripts/split.py")


if __name__ == "__main__":
    main()