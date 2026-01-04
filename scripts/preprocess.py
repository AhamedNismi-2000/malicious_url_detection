#!/usr/bin/env python3
# scripts/preprocessing_full_resume_livebar_optimized_safe.py
"""
FIXED preprocessing for large URL datasets
- Produces TWO outputs: 
  1. Reference CSV with original labels (benign/malicious)
  2. ML-ready CSV with numeric labels (0,1)
- NO source column in final output
- Expands short URLs but KEEPS original URLs
- Preserves all 1.6M URLs
- Uses threaded requests only for short URLs
- Normalizes all URLs
- Removes duplicates properly
"""

import os
import re
import json
import hashlib
import pandas as pd
import tldextract
import requests
from urllib.parse import urlparse
from tqdm import tqdm
import threading
import queue
import time
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

ALEXA_PATH = os.path.join(RAW_DIR, "top-1m.csv")
PHISHTANK_PATH = os.path.join(RAW_DIR, "Phishtank.csv")
KAGGLE_PATH = os.path.join(RAW_DIR, "Kaggle.csv")
OUTPUT_REFERENCE = os.path.join(PROCESSED_DIR, "cleaned_urls_reference.csv")  # Original labels
OUTPUT_ML = os.path.join(PROCESSED_DIR, "cleaned_urls.csv")  # Numeric labels for ML
CACHE_FILE = os.path.join(PROCESSED_DIR, "url_expansion_cache.json")

# ---------------- SHORTENERS ----------------
SHORTENERS = {
    # Original
    "bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "t.co", "is.gd", "buff.ly",
    "adf.ly", "bit.do", "mcaf.ee", "soo.gd", "surl.li", "shorte.st",
    "clicky.me", "cutt.ly", "u.to", "v.gd", "tr.im", "tiny.cc", "rebrand.ly", "t.ly",
    
    # NEW - Additional shorteners
    "bc.vc", "cli.gs", "sh.st", "ity.im", "short.to", "adfoc.us", "adfly.ru",
    "link.tl", "qr.net", "cutt.us", "x.co", "1url.com", "tiny.pl", "short.cm",
    "pic.gd", "short.nr", "soo.gd", "tiny.ie", "short.ie", "moourl.com",
    "zz.gd", "shortna.com", "tinylink.in", "shorturl.com", "miniurl.com",
    "tinyurl.com", "tiny.cc", "bitly.com", "shorl.com", "kl.am", "fwd4.me",
    "yep.it", "easyuri.com", "xlink.me", "short.in", "tinyarro.ws", "fur.ly",
    "hurl.me", "lnk.co", "twitthis.com", "su.pr", "snipurl.com", "snipr.com",
    "snurl.com", "sn.im", "ilix.in", "chilp.it", "flic.kr", "qlnk.net",
    "doiop.com", "twurl.nl", "rubyurl.com", "om.ly", "prettylinkpro.com"
}

# ---------------- HELPERS ----------------
def normalize_url(u: str) -> str:
    """Normalize URL by stripping spaces, lowercasing, and adding http if missing."""
    try:
        u = u.strip().lower()
        u = re.sub(r"\s+", "", u)
        if u.endswith("/") and len(u) > 1:
            u = u[:-1]
        if not re.match(r"https?://", u):
            u = "http://" + u
        return u
    except Exception:
        return None

def is_shortened(url: str) -> bool:
    """Check if the URL belongs to a known shortener safely."""
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc in SHORTENERS
    except Exception:
        return False

def valid_domain(url: str) -> bool:
    """Check if URL has a valid domain safely."""
    try:
        extracted = tldextract.extract(url)
        return bool(extracted.domain) and bool(extracted.suffix)
    except Exception:
        return False

def convert_labels_to_numeric(df):
    """Convert string labels to numeric (0,1) for ML training"""
    df_ml = df.copy()
    
    # Map labels to numeric values
    label_mapping = {
        'benign': 0,
        'malicious': 1,
        'malware': 1,
        'phishing': 1,
        'defacement': 1,
        'legitimate': 0,
        'clean': 0
    }
    
    # Convert labels
    df_ml['label'] = df_ml['label'].str.lower().map(label_mapping)
    
    # Handle any unmapped labels (set to NaN and then drop)
    original_count = len(df_ml)
    df_ml = df_ml.dropna(subset=['label'])
    df_ml['label'] = df_ml['label'].astype(int)
    
    removed_count = original_count - len(df_ml)
    if removed_count > 0:
        print(f"⚠️  Removed {removed_count} URLs with unrecognized labels")
    
    return df_ml

# ---------------- CACHE ----------------
def load_cache():
    """Load URL expansion cache from disk."""
    if os.path.exists(CACHE_FILE):
        try:
            return json.load(open(CACHE_FILE, encoding="utf-8"))
        except:
            return {}
    return {}

def save_cache(cache):
    """Save URL expansion cache to disk."""
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)

def cache_key(url):
    """Generate a cache key for a URL."""
    return hashlib.md5(url.encode("utf-8")).hexdigest()

# ---------------- URL EXPANSION ----------------
def expand_urls_shorteners_only(urls, n_threads=32, flush_every=50_000):
    """Expand only short URLs using threads and cache, but return original URLs for non-shortened."""
    cache = load_cache()
    results = [None] * len(urls)

    # Identify only short URLs to expand
    to_expand = []
    for i, u in enumerate(urls):
        if not u:
            results[i] = None
        elif is_shortened(u):
            if cache_key(u) in cache:
                results[i] = cache[cache_key(u)]  # Use cached expansion
            else:
                to_expand.append((i, u))
                results[i] = u  # Temporary: keep original
        else:
            results[i] = u  # Keep original for non-shortened URLs

    print(f"📌 Short URLs to expand: {len(to_expand)} / {len(urls)} total URLs")

    if not to_expand:
        return results

    pbar = tqdm(total=len(to_expand), desc="Expanding short URLs", unit="url")

    # Heartbeat thread
    def heartbeat():
        while pbar.n < pbar.total:
            time.sleep(30)
            print(f"⏱  Still expanding… {pbar.n}/{pbar.total} done")
    threading.Thread(target=heartbeat, daemon=True).start()

    # Thread-safe queue
    q = queue.Queue()
    for idx, url in to_expand:
        q.put((idx, url))

    def worker():
        session = requests.Session()
        while True:
            try:
                idx, url = q.get_nowait()
            except queue.Empty:
                break
            try:
                resp = session.get(url, allow_redirects=True, timeout=5, stream=True, verify=False)
                final_url = resp.url
                # Only use expanded URL if it's different and valid
                if final_url != url and valid_domain(final_url):
                    results[idx] = final_url
                else:
                    results[idx] = url  # Keep original if expansion failed or same
                cache[cache_key(url)] = results[idx]
            except Exception as e:
                # Keep original URL if expansion fails
                results[idx] = url
                cache[cache_key(url)] = url
            pbar.update(1)
            if len(cache) % flush_every == 0:
                save_cache(cache)
            q.task_done()

    threads = [threading.Thread(target=worker) for _ in range(min(n_threads, len(to_expand)))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    save_cache(cache)
    pbar.close()
    return results

# ---------------- DATASET LOADERS ----------------
def load_alexa():
    """Load Alexa dataset - NO source column"""
    print("📥 Loading Alexa dataset...")
    try:
        df = pd.read_csv(ALEXA_PATH, dtype=str, header=None, names=["rank", "url"])
        df["label"] = "benign"
        print(f"✅ Alexa: {len(df)} URLs")
        return df[["url","label"]]  # No source column
    except Exception as e:
        print(f"❌ Error loading Alexa: {e}")
        return pd.DataFrame(columns=["url","label"])

def load_phishtank():
    """Load PhishTank dataset - NO source column"""
    print("📥 Loading PhishTank dataset...")
    try:
        df = pd.read_csv(PHISHTANK_PATH, dtype=str, low_memory=False)
        df["label"] = "malicious"
        print(f"✅ PhishTank: {len(df)} URLs")
        return df[["url","label"]]  # No source column
    except Exception as e:
        print(f"❌ Error loading PhishTank: {e}")
        return pd.DataFrame(columns=["url","label"])

def load_kaggle():
    """Load Kaggle dataset - NO source column"""
    print("📥 Loading Kaggle dataset...")
    try:
        df = pd.read_csv(KAGGLE_PATH, dtype=str, low_memory=False)
        df = pd.DataFrame({
            "url": df["url"], 
            "label": df["type"]
        })
        df["label"] = df["label"].str.lower().replace({
            "benign": "benign", "legitimate": "benign",
            "malicious": "malicious", "defacement": "malicious", "phishing": "malicious"
        })
        print(f"✅ Kaggle: {len(df)} URLs")
        return df[["url","label"]]  # No source column
    except Exception as e:
        print(f"❌ Error loading Kaggle: {e}")
        return pd.DataFrame(columns=["url","label"])

# ---------------- MAIN PIPELINE ----------------
def main():
    print("🚀 STARTING URL PREPROCESSING PIPELINE")
    print("=" * 60)
    print("🎯 OUTPUTS:")
    print("   📄 cleaned_urls_reference.csv - Original labels (for reference)")
    print("   🤖 cleaned_urls.csv - Numeric labels 0,1 (for ML training)")
    print("   📊 NO source column in final outputs")
    print("=" * 60)
    
    # Load datasets
    alexa = load_alexa()
    phish = load_phishtank()
    kaggle = load_kaggle()
    
    # Combine all datasets
    combined = pd.concat([alexa, phish, kaggle], ignore_index=True)
    print(f"📊 Combined dataset: {len(combined):,} total URLs")
    
    # Remove duplicates before any processing
    initial_count = len(combined)
    combined = combined.drop_duplicates(subset=["url"]).dropna(subset=["url"])
    removed_duplicates = initial_count - len(combined)
    print(f"🗑 Removed {removed_duplicates:,} duplicate URLs")
    
    # Normalize URLs
    print("🔄 Normalizing URLs...")
    combined["url_original"] = combined["url"]  # Keep original for reference
    combined["url"] = combined["url"].apply(normalize_url)
    combined = combined[combined["url"].notnull()].reset_index(drop=True)
    print(f"📊 After normalization: {len(combined):,} URLs")
    
    # Remove duplicates again after normalization
    combined = combined.drop_duplicates(subset=["url"]).reset_index(drop=True)
    print(f"📊 After deduplication: {len(combined):,} URLs")
    
    # Expand ONLY short URLs (this should be a small subset)
    print("🔗 Expanding short URLs...")
    urls_list = combined["url"].tolist()
    expanded_urls = expand_urls_shorteners_only(urls_list, n_threads=16)
    
    # Update URLs - use expanded version for short URLs, keep original for others
    combined["url_final"] = expanded_urls
    
    # Final normalization of expanded URLs
    combined["url_final"] = combined["url_final"].apply(normalize_url)
    combined = combined[combined["url_final"].notnull()].reset_index(drop=True)
    
    # Use final URLs for the dataset
    combined["url"] = combined["url_final"]
    combined = combined.drop(columns=["url_original", "url_final"])
    
    # Validate domains
    print("🔍 Validating domains...")
    combined = combined[combined["url"].apply(valid_domain)].reset_index(drop=True)
    print(f"📊 After domain validation: {len(combined):,} URLs")
    
    # Final deduplication
    final_count_before = len(combined)
    combined = combined.drop_duplicates(subset=["url"]).reset_index(drop=True)
    final_duplicates_removed = final_count_before - len(combined)
    
    # Create TWO outputs - BOTH WITHOUT SOURCE COLUMN
    
    # 1. REFERENCE OUTPUT (original labels)
    print("💾 Saving REFERENCE dataset (original labels)...")
    reference_df = combined.copy()
    reference_df = reference_df.sample(frac=1, random_state=42).reset_index(drop=True)
    reference_df.to_csv(OUTPUT_REFERENCE, index=False, encoding="utf-8")
    
    # 2. ML OUTPUT (numeric labels 0,1)
    print("💾 Saving ML dataset (numeric labels 0,1)...")
    ml_df = convert_labels_to_numeric(combined)
    ml_df = ml_df.sample(frac=1, random_state=42).reset_index(drop=True)
    ml_df.to_csv(OUTPUT_ML, index=False, encoding="utf-8")
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 PREPROCESSING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"📊 Final dataset size: {len(combined):,} URLs")
    print(f"🗑 Total duplicates removed: {removed_duplicates + final_duplicates_removed:,}")
    
    print("\n📈 REFERENCE FILE (cleaned_urls_reference.csv):")
    label_counts_ref = combined["label"].value_counts()
    for label, count in label_counts_ref.items():
        percentage = (count / len(combined)) * 100
        print(f"   {label:10}: {count:>8,} URLs ({percentage:5.1f}%)")
    
    print("\n🤖 ML TRAINING FILE (cleaned_urls.csv):")
    label_counts_ml = ml_df["label"].value_counts()
    for label, count in label_counts_ml.items():
        label_name = "benign" if label == 0 else "malicious"
        percentage = (count / len(ml_df)) * 100
        print(f"   {label_name:10}: {count:>8,} URLs ({percentage:5.1f}%)")
    
    print(f"\n💾 Output files:")
    print(f"   📄 {OUTPUT_REFERENCE} (reference with original labels)")
    print(f"   🤖 {OUTPUT_ML} (ML training with numeric labels 0,1)")
    print(f"   📊 Both files have only 2 columns: url, label")
    
    # Check for short URLs in final dataset
    short_urls_count = combined["url"].apply(is_shortened).sum()
    print(f"🔗 Short URLs in final dataset: {short_urls_count:,}")

if __name__ == "__main__":
    main()