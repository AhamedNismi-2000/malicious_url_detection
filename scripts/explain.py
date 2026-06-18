"""
explain.py — Standalone LIME explanation script for malicious URL detector.
Usage: python scripts/explain.py <url>
       python scripts/explain.py  (uses built-in test URLs)
"""

import sys
import os
import json
import numpy as np

# Add project root to path so app/ imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.model_loader import classifier

FEATURE_NAMES = (
    [
        "url_len", "path_len", "num_dots", "path_dots", "num_hyphens",
        "num_underscores", "num_at", "num_qmark", "num_equal", "num_amp",
        "num_percent", "num_digits", "num_letters", "num_subdirs", "num_frag",
        "num_special", "num_repeating", "num_upper", "num_non_ascii",
        "num_slashes", "num_params", "ratio_digits", "ratio_letters",
        "url_entropy", "ip_flag", "subdomain_parts", "has_multi_subdomain",
        "tld_len", "risky_tld", "https_flag", "shortened", "sus_words",
        "brand_mismatch", "puny", "susp_ext", "suspicious_port",
        "max_consonants", "max_vowels", "max_digits",
        "leet_speak_score", "homoglyph_suspicious", "encoding_ratio",
        "punycode_suspicious", "subdomain_spam_score", "visual_brand_similarity",
        "brand_in_domain", "leet_in_domain", "brand_hyphen_suspicious",
    ]
    + [f"char_{i}" for i in range(300)]
    + [f"word_{i}" for i in range(200)]
)

TEST_URLS = [
    "https://paypal-security.com/verify/account",
    "https://amaz0n.com/login",
    "https://google.com",
    "https://github.com/user/repo",
    "http://192.168.1.1/admin",
]


def explain_url(url: str, num_features: int = 10) -> dict:
    """Return prediction + LIME explanation for a single URL."""
    result = classifier.explain_url(url, num_features=num_features)
    return result


def pretty_print(result: dict) -> None:
    label = result.get("prediction", "UNKNOWN")
    confidence = result.get("confidence", 0)
    source = result.get("source", "model")

    color = "\033[91m" if label == "MALICIOUS" else "\033[92m"
    reset = "\033[0m"

    print(f"\nURL       : {result['url']}")
    print(f"Prediction: {color}{label}{reset}  ({confidence:.1f}% confidence)")
    print(f"Source    : {source}")

    explanation = result.get("explanation", [])
    if explanation:
        print(f"\nTop {len(explanation)} contributing features:")
        print(f"  {'Feature':<30} {'Weight':>8}  {'Value':>8}")
        print("  " + "-" * 52)
        for feat in explanation:
            weight = feat["weight"]
            value = feat["value"]
            sign = "+" if weight > 0 else ""
            direction = "→ MALICIOUS" if weight > 0 else "→ BENIGN   "
            print(f"  {feat['feature']:<30} {sign}{weight:>7.4f}  {value:>8.4f}  {direction}")
    else:
        print("  (No explanation — whitelist hit or invalid URL)")

    print()


def main():
    urls = sys.argv[1:] if len(sys.argv) > 1 else TEST_URLS

    print("=" * 60)
    print("  Malicious URL Detector — LIME Explanation")
    print("=" * 60)

    for url in urls:
        try:
            result = explain_url(url)
            pretty_print(result)
        except Exception as exc:
            print(f"\nERROR on {url}: {exc}\n")


if __name__ == "__main__":
    main()
