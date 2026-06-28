#!/usr/bin/env python3
"""
test_model.py
-------------
Sanity check script to verify model predictions before production.

Two-layer prediction:
  Layer 1: Whitelist  — known trusted domains bypass ML model
  Layer 2: ML model   — unknown domains run through full pipeline

Feature count: 559
  57 heuristic + obfuscation + rule-based + domain + new structural features
 300 char n-gram TF-IDF
 202 word n-gram TF-IDF

Run:
  python scripts/test_model.py
"""

import os
import sys
import json
import re
import warnings

import numpy as np
import joblib
import scipy.sparse as sp
import tldextract
from urllib.parse import urlparse

warnings.filterwarnings("ignore")

# ---------------- PATHS ----------------
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR    = os.path.join(BASE_DIR, "models")
SCRIPTS_DIR   = os.path.join(BASE_DIR, "scripts")

MODEL_PATH     = os.path.join(MODELS_DIR, "rf_model_latest.joblib")
CHAR_VEC_PATH  = os.path.join(MODELS_DIR, "vectorizer_char.joblib")
WORD_VEC_PATH  = os.path.join(MODELS_DIR, "vectorizer_word.joblib")
SCALER_PATH    = os.path.join(MODELS_DIR, "scaler.joblib")
THRESHOLD_PATH = os.path.join(MODELS_DIR, "threshold.json")

# ---------------- IMPORT FROM feature_extraction.py ----------------
# Single source of truth — no duplicate feature code
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from feature_extraction import (
    extract_heuristic_features,
    segment_url,
    N_HEURISTIC,
    HEURISTIC_FEATURE_NAMES,
)

# ---------------- EXPECTED FEATURE COUNT ----------------
EXPECTED_HEURISTIC = 57   # 56 heuristic features
EXPECTED_CHAR      = 300  # char n-gram TF-IDF
EXPECTED_WORD      = 202  # word n-gram TF-IDF
EXPECTED_TOTAL     = EXPECTED_HEURISTIC + EXPECTED_CHAR + EXPECTED_WORD # 559

# ---------------- WHITELIST ----------------
TRUSTED_DOMAINS = {
    # Search and productivity
    "google.com", "gmail.com", "youtube.com", "googleapis.com",
    "github.com", "gitlab.com", "stackoverflow.com",
    "wikipedia.org", "wikimedia.org",
    # Microsoft
    "microsoft.com", "microsoftonline.com", "live.com",
    "outlook.com", "office.com", "azure.com", "bing.com",
    # Apple
    "apple.com", "icloud.com",
    # Amazon
    "amazon.com", "amazon.co.uk", "amazon.de", "amazon.fr",
    "amazonaws.com", "aws.amazon.com",
    # Social media
    "facebook.com", "instagram.com", "twitter.com",
    "linkedin.com", "reddit.com", "pinterest.com",
    "whatsapp.com", "telegram.org",
    # Finance
    "paypal.com", "bankofamerica.com", "chase.com",
    "wellsfargo.com", "citibank.com", "visa.com",
    "mastercard.com", "stripe.com",
    # Entertainment
    "netflix.com", "spotify.com", "twitch.tv",
    "discord.com", "slack.com", "zoom.us",
    # Shopping
    "ebay.com", "etsy.com", "shopify.com",
    # Tools
    "dropbox.com", "adobe.com", "salesforce.com",
    "wordpress.com", "medium.com",
    # Developer sites
    "roadmap.sh", "dev.to", "freecodecamp.org",
    "scrimba.com", "codecademy.com", "hashnode.com",
    # Job boards
    "dailyremote.com", "jobspresso.co", "remoteok.com",
    # Media
    "4kwallpapers.com", "unsplash.com", "pexels.com",
}

EXTRACTOR_GLOBAL = tldextract.TLDExtract(cache_dir=None, suffix_list_urls=None)


def is_whitelisted(url: str) -> bool:
    try:
        parsed  = urlparse(url.lower())
        ext     = EXTRACTOR_GLOBAL(parsed.netloc)
        reg_dom = ext.registered_domain or ""
        return reg_dom in TRUSTED_DOMAINS
    except Exception:
        return False


# ---------------- TEST URLs ----------------
TEST_URLS = [
    # Clearly Benign
    ("https://www.google.com",                             "Clearly Benign"),
    ("https://github.com/user/repo",                       "Clearly Benign"),
    ("https://www.wikipedia.org/wiki/Machine_learning",    "Clearly Benign"),
    ("https://stackoverflow.com/questions/12345",          "Clearly Benign"),
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ",       "Clearly Benign"),
    # Clearly Malicious
    ("http://paypal-security-alert.com/verify",            "Clearly Malicious"),
    ("http://192.168.1.1/login/secure/account/update",     "Clearly Malicious"),
    ("http://xn--pypal-4ve.com/signin",                    "Clearly Malicious"),
    ("http://secure-banking-update.tk/account/verify.php", "Clearly Malicious"),
    ("http://microsoft-alert.support/windows/virus-detected.exe",
                                                           "Clearly Malicious"),
    # Tricky Benign
    ("https://www.paypal.com/signin",                      "Tricky Benign"),
    ("https://accounts.google.com/login",                  "Tricky Benign"),
    ("https://secure.bankofamerica.com/login",             "Tricky Benign"),
    ("https://signin.aws.amazon.com/console",              "Tricky Benign"),
    ("https://login.microsoftonline.com/account",          "Tricky Benign"),
    # Tricky Malicious
    ("http://paypa1.com/secure/login",                     "Tricky Malicious"),
    ("http://google.com.phishing-site.ru/login",           "Tricky Malicious"),
    ("http://amaz0n-prime.com/verify/account",             "Tricky Malicious"),
    ("http://apple.com.id-verify.net/signin",              "Tricky Malicious"),
    ("http://faceb00k-login.com/checkpoint",               "Tricky Malicious"),
    # Edge Cases
    ("http://185.220.101.45/malware/payload.exe",          "Edge Case"),
    ("http://bit.ly/3xK9mNp",                             "Edge Case"),
    ("https://legitimate-very-long-domain-name.com/page",  "Edge Case"),
    ("http://free-iphone-winner.xyz/claim/prize/now",      "Edge Case"),
    ("https://www.amazon.co.uk/dp/B08N5WRWNW",             "Edge Case"),
]


# ---------------- PREDICTION ----------------

def predict_with_model(urls, model, char_vec, word_vec, scaler, threshold):
    """
    Full ML prediction pipeline.
    Uses segment_url() for NLP — matches training pipeline exactly.
    Imports extract_heuristic_features from feature_extraction.py
    — single source of truth, no duplicate code.
    """
    # Heuristic features — imported from feature_extraction.py
    heuristic = np.array(
        [extract_heuristic_features(u) for u in urls],
        dtype=np.float32
    )
    heuristic        = np.nan_to_num(heuristic, nan=0.0,
                                     posinf=0.0, neginf=0.0)
    heuristic_scaled = scaler.transform(heuristic).astype(np.float32)

    # NLP features — use segment_url() to match training
    segmented = [segment_url(u) for u in urls]
    X_char    = char_vec.transform(segmented)
    X_word    = word_vec.transform(segmented)
    X_nlp     = sp.hstack([X_char, X_word], format="csr").astype(np.float32)

    # Combine heuristic + NLP
    X = sp.hstack(
        [sp.csr_matrix(heuristic_scaled), X_nlp],
        format="csr"
    )

    mal_col = list(model.classes_).index(1)
    y_proba = model.predict_proba(X)[:, mal_col]
    y_pred  = (y_proba >= threshold).astype(int)

    return [
        {
            "prediction": "MALICIOUS" if y_pred[i] == 1 else "BENIGN",
            "confidence": round(float(y_proba[i]) * 100, 2),
            "source"    : "model"
        }
        for i in range(len(urls))
    ]


def predict(urls, model, char_vec, word_vec, scaler, threshold):
    """Two-layer: whitelist first, then ML model."""
    results          = [None] * len(urls)
    model_input_idx  = []
    model_input_urls = []

    for i, url in enumerate(urls):
        if is_whitelisted(url):
            results[i] = {
                "prediction": "BENIGN",
                "confidence": 0.0,
                "source"    : "whitelist"
            }
        else:
            model_input_idx.append(i)
            model_input_urls.append(url)

    if model_input_urls:
        model_results = predict_with_model(
            model_input_urls, model, char_vec,
            word_vec, scaler, threshold
        )
        for idx, result in zip(model_input_idx, model_results):
            results[idx] = result

    return results


# ---------------- MAIN ----------------
def main():
    print("=" * 70)
    print("MODEL SANITY CHECK — 25 URL TEST")
    print("=" * 70)
    print(f"Pipeline: Whitelist + ML Model ({EXPECTED_TOTAL} features)")
    print(f"  Heuristic : {EXPECTED_HEURISTIC} features")
    print(f"  Char TF-IDF: {EXPECTED_CHAR} features")
    print(f"  Word TF-IDF: {EXPECTED_WORD} features")
    print("=" * 70)

    # Check all model files exist
    required = {
        "Model"    : MODEL_PATH,
        "Char vec" : CHAR_VEC_PATH,
        "Word vec" : WORD_VEC_PATH,
        "Scaler"   : SCALER_PATH,
        "Threshold": THRESHOLD_PATH
    }
    all_exist = True
    for name, path in required.items():
        status = "OK" if os.path.exists(path) else "MISSING"
        print(f"  {name:12}: {status}")
        if status == "MISSING":
            all_exist = False

    if not all_exist:
        print("\nERROR: Missing model files.")
        print("Run: feature_extraction.py → train_model.py")
        sys.exit(1)

    print("\nLoading model artifacts...")
    model    = joblib.load(MODEL_PATH)
    char_vec = joblib.load(CHAR_VEC_PATH)
    word_vec = joblib.load(WORD_VEC_PATH)
    scaler   = joblib.load(SCALER_PATH)

    with open(THRESHOLD_PATH) as f:
        threshold = json.load(f)["threshold"]

    print(f"  Model     : {model.n_estimators} trees, "
          f"{model.n_features_in_} features")
    print(f"  Threshold : {threshold}")
    print(f"  Whitelist : {len(TRUSTED_DOMAINS)} trusted domains")
    print(f"  N_HEURISTIC from feature_extraction: {N_HEURISTIC}")

    # Verify feature count matches
    if model.n_features_in_ != EXPECTED_TOTAL:
        print(f"\n  WARNING: Model has {model.n_features_in_} features "
              f"but expected {EXPECTED_TOTAL}.")
        print("  Re-run feature_extraction.py and train_model.py first.")
    else:
        print(f"  Feature count: {model.n_features_in_} ✓")

    print(f"\nRunning predictions on {len(TEST_URLS)} URLs...\n")
    urls       = [u for u, _ in TEST_URLS]
    categories = [c for _, c in TEST_URLS]
    results    = predict(urls, model, char_vec, word_vec,
                         scaler, threshold)

    expected_map = {
        "Clearly Benign"   : "BENIGN",
        "Clearly Malicious": "MALICIOUS",
        "Tricky Benign"    : "BENIGN",
        "Tricky Malicious" : "MALICIOUS",
        "Edge Case"        : None
    }

    categories_seen = []
    correct = 0
    total   = 0

    for result, category, url in zip(results, categories, urls):
        if category not in categories_seen:
            categories_seen.append(category)
            print(f"\n{'─'*70}")
            print(f"  CATEGORY: {category}")
            print(f"{'─'*70}")
            print(f"  {'URL':<45} {'PRED':<12} {'CONF':>8} {'SRC'}")
            print(f"  {'-'*45} {'-'*11} {'-'*8} {'-'*9}")

        expected    = expected_map.get(category)
        pred        = result["prediction"]
        conf        = result["confidence"]
        source      = result["source"]
        display_url = url if len(url) <= 45 else url[:42] + "..."

        if expected:
            total += 1
            is_correct = pred == expected
            if is_correct:
                correct += 1
            marker = "✓ OK" if is_correct else "✗ WRONG"
        else:
            marker = "?"

        conf_str = f"{conf:.1f}%" if source == "model" else "whitelist"
        print(f"  {display_url:<45} {pred:<12} "
              f"{conf_str:>8}  [{marker}]")

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Total URLs tested   : {len(TEST_URLS)}")
    print(f"  Correct predictions : {correct}/{total}  "
          f"({100*correct/total:.1f}%)")
    print(f"  Threshold used      : {threshold}")

    print(f"\n  Category breakdown:")
    for cat in ["Clearly Benign", "Clearly Malicious",
                "Tricky Benign", "Tricky Malicious"]:
        cat_indices = [i for i, c in enumerate(categories) if c == cat]
        cat_correct = sum(
            1 for i in cat_indices
            if results[i]["prediction"] == expected_map[cat]
        )
        print(f"    {cat:<22}: {cat_correct}/{len(cat_indices)} correct")

    print(f"\n  Wrong predictions:")
    found_wrong = False
    for result, category, url in zip(results, categories, urls):
        expected = expected_map.get(category)
        if expected and result["prediction"] != expected:
            found_wrong = True
            print(f"    ✗ {url}")
            print(f"      Expected {expected}, "
                  f"got {result['prediction']} "
                  f"(conf: {result['confidence']}%, "
                  f"src: {result['source']})")

    if not found_wrong:
        print("    None — all expected predictions correct ✓")

    accuracy = correct / total
    print(f"\n{'='*70}")
    if accuracy >= 0.90:
        print(f"  VERDICT: ✓ Model ready for production ({accuracy:.0%})")
    elif accuracy >= 0.80:
        print(f"  VERDICT: ⚠ Acceptable — review wrong predictions ({accuracy:.0%})")
    else:
        print(f"  VERDICT: ✗ Needs improvement ({accuracy:.0%})")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()