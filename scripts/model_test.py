#!/usr/bin/env python3
"""
test_model.py
-------------
Sanity check script to verify model predictions before production.

Two-layer prediction:
  Layer 1: Whitelist  — known trusted domains bypass ML model
  Layer 2: ML model   — unknown domains run through full pipeline

Feature count: 548
  45 heuristic + obfuscation
   3 new rule-based (brand_in_domain, leet_in_domain, brand_hyphen)
 300 char n-gram TF-IDF
 200 word n-gram TF-IDF

Run:
  python scripts/test_model.py
"""

import os
import sys
import json
import re
import math
import ipaddress
import warnings
from collections import Counter
from urllib.parse import urlparse

import numpy as np
import joblib
import scipy.sparse as sp
import tldextract

warnings.filterwarnings("ignore")

# ---------------- PATHS ----------------
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR    = os.path.join(BASE_DIR, "models")

MODEL_PATH     = os.path.join(MODELS_DIR, "rf_model_latest.joblib")
CHAR_VEC_PATH  = os.path.join(MODELS_DIR, "vectorizer_char.joblib")
WORD_VEC_PATH  = os.path.join(MODELS_DIR, "vectorizer_word.joblib")
SCALER_PATH    = os.path.join(MODELS_DIR, "scaler.joblib")
THRESHOLD_PATH = os.path.join(MODELS_DIR, "threshold.json")

# ---------------- WHITELIST ----------------
TRUSTED_DOMAINS = {
    "google.com", "gmail.com", "youtube.com", "googleapis.com",
    "github.com", "gitlab.com", "stackoverflow.com",
    "wikipedia.org", "wikimedia.org",
    "microsoft.com", "microsoftonline.com", "live.com",
    "outlook.com", "office.com", "azure.com", "bing.com",
    "apple.com", "icloud.com",
    "amazon.com", "amazon.co.uk", "amazon.de", "amazon.fr",
    "amazonaws.com", "aws.amazon.com",
    "facebook.com", "instagram.com", "twitter.com",
    "linkedin.com", "reddit.com", "pinterest.com",
    "whatsapp.com", "telegram.org",
    "paypal.com", "bankofamerica.com", "chase.com",
    "wellsfargo.com", "citibank.com", "visa.com",
    "mastercard.com", "stripe.com",
    "netflix.com", "spotify.com", "twitch.tv",
    "discord.com", "slack.com", "zoom.us",
    "ebay.com", "etsy.com", "shopify.com",
    "dropbox.com", "adobe.com", "salesforce.com",
    "wordpress.com", "medium.com"
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
    ("https://www.google.com",                          "Clearly Benign"),
    ("https://github.com/user/repo",                    "Clearly Benign"),
    ("https://www.wikipedia.org/wiki/Machine_learning", "Clearly Benign"),
    ("https://stackoverflow.com/questions/12345",       "Clearly Benign"),
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ",    "Clearly Benign"),
    ("http://paypal-security-alert.com/verify",         "Clearly Malicious"),
    ("http://192.168.1.1/login/secure/account/update",  "Clearly Malicious"),
    ("http://xn--pypal-4ve.com/signin",                 "Clearly Malicious"),
    ("http://secure-banking-update.tk/account/verify.php", "Clearly Malicious"),
    ("http://microsoft-alert.support/windows/virus-detected.exe",
     "Clearly Malicious"),
    ("https://www.paypal.com/signin",                   "Tricky Benign"),
    ("https://accounts.google.com/login",               "Tricky Benign"),
    ("https://secure.bankofamerica.com/login",          "Tricky Benign"),
    ("https://signin.aws.amazon.com/console",           "Tricky Benign"),
    ("https://login.microsoftonline.com/account",       "Tricky Benign"),
    ("http://paypa1.com/secure/login",                  "Tricky Malicious"),
    ("http://google.com.phishing-site.ru/login",        "Tricky Malicious"),
    ("http://amaz0n-prime.com/verify/account",          "Tricky Malicious"),
    ("http://apple.com.id-verify.net/signin",           "Tricky Malicious"),
    ("http://faceb00k-login.com/checkpoint",            "Tricky Malicious"),
    ("http://185.220.101.45/malware/payload.exe",       "Edge Case"),
    ("http://bit.ly/3xK9mNp",                          "Edge Case"),
    ("https://legitimate-very-long-domain-name.com/page", "Edge Case"),
    ("http://free-iphone-winner.xyz/claim/prize/now",   "Edge Case"),
    ("https://www.amazon.co.uk/dp/B08N5WRWNW",          "Edge Case"),
]

# ---------------- CONSTANTS ----------------
SHORTENERS = {
    "bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "t.co", "is.gd",
    "buff.ly", "adf.ly", "bit.do", "mcaf.ee", "surl.li", "shorte.st",
    "clicky.me", "cutt.ly", "u.to", "v.gd", "tr.im", "tiny.cc",
    "rebrand.ly", "t.ly", "bc.vc", "cli.gs", "sh.st", "ity.im",
    "short.to", "adfoc.us", "link.tl", "qr.net", "cutt.us", "x.co",
    "1url.com", "tiny.pl", "short.cm", "pic.gd", "short.nr", "tiny.ie",
    "short.ie", "moourl.com", "zz.gd", "tinylink.in", "shorturl.com",
    "miniurl.com", "bitly.com", "shorl.com", "kl.am", "fwd4.me",
    "yep.it", "xlink.me", "fur.ly", "hurl.me", "lnk.co",
    "snipurl.com", "snipr.com", "snurl.com", "sn.im", "flic.kr",
    "qlnk.net", "doiop.com", "twurl.nl", "rubyurl.com", "om.ly"
}

SUSPICIOUS_WORDS = {
    "suspend", "urgent", "prize", "winner",
    "congratulations", "free-iphone", "limited-offer",
    "click-here", "verify-now", "act-now",
    "account-suspended", "password-reset-required"
}

RISKY_TLDS = {
    "zip", "review", "country", "gq", "tk", "ml", "cf", "ga", "top",
    "xyz", "click", "link", "pw", "club", "work", "site", "online",
    "space", "webcam", "stream", "download", "gdn", "racing", "loan",
    "win", "bid", "trade", "science", "party", "cricket", "date",
    "faith", "accountant", "men", "biz", "info", "su", "cc", "icu",
    "cyou", "rest", "bar", "buzz", "live", "xxx", "dating"
}

BRANDS = {
    "paypal", "amazon", "microsoft", "apple", "google", "facebook",
    "netflix", "bankofamerica", "wellsfargo", "whatsapp", "instagram",
    "twitter", "linkedin", "ebay", "visa", "mastercard", "chase",
    "citi", "bank", "pay", "secure"
}

REAL_BRAND_DOMAINS = {
    "paypal.com", "amazon.com", "microsoft.com", "apple.com",
    "google.com", "facebook.com", "netflix.com", "bankofamerica.com",
    "wellsfargo.com", "whatsapp.com", "instagram.com", "twitter.com",
    "linkedin.com", "ebay.com", "visa.com", "mastercard.com",
    "chase.com", "citibank.com", "amazon.co.uk", "amazon.de",
    "amazon.fr", "amazon.in", "microsoftonline.com", "live.com",
    "outlook.com", "office.com", "icloud.com", "amazonaws.com"
}

BRAND_SUSPICIOUS_WORDS = {
    "security", "alert", "verify", "update", "login",
    "signin", "secure", "confirm", "account", "banking",
    "support", "help", "service", "center", "care",
    "warning", "suspend", "locked", "unlock", "recover"
}

COMMON_PORTS = {80, 443, 8080, 8443, 3000, 5000, 8000, 9000}
EXTRACTOR    = tldextract.TLDExtract(cache_dir=None, suffix_list_urls=None)
N_HEURISTIC  = 48   # updated from 45 to 48


# ---------------- FEATURE FUNCTIONS ----------------

def has_ip_address(hostname):
    try:
        ipaddress.IPv4Address(hostname)
        return True
    except Exception:
        pass
    try:
        ipaddress.IPv6Address(hostname)
        return True
    except Exception:
        return False

def is_shortened(hostname, registered_domain):
    try:
        h = hostname.lower()
        if h.startswith("www."):
            h = h[4:]
        rd = (registered_domain or "").lower()
        return (h in SHORTENERS) or (rd in SHORTENERS)
    except Exception:
        return False

def count_suspicious_words(url):
    return sum(1 for w in SUSPICIOUS_WORDS if w in url.lower())

def simple_entropy(s):
    if not s or len(s) <= 1:
        return 0.0
    cnt = Counter(s)
    l   = len(s)
    return -sum((v/l)*math.log2(v/l) for v in cnt.values() if v > 0)

def max_consecutive(s, char_type):
    max_count = current = 0
    for char in s.lower():
        if char_type == "digit" and char.isdigit():
            current += 1
        elif char_type == "consonant" and char in "bcdfghjklmnpqrstvwxyz":
            current += 1
        elif char_type == "vowel" and char in "aeiou":
            current += 1
        else:
            max_count = max(max_count, current)
            current   = 0
    return max(max_count, current)

def max_repeating(s):
    if len(s) <= 1:
        return 0
    mx = cur = 1
    for i in range(1, len(s)):
        cur = cur + 1 if s[i] == s[i-1] else 1
        mx  = max(mx, cur)
    return mx

def detect_leet_speak(url):
    url_lower = url.lower()
    try:
        domain_part = urlparse(url_lower).netloc
    except Exception:
        domain_part = url_lower
    leet_map = {"4":"a","3":"e","1":"i","0":"o","5":"s","7":"t"}
    score = 0.0
    for digit in leet_map:
        score += len(re.findall(rf"[a-z]{re.escape(digit)}[a-z]",
                                domain_part)) * 0.2
    return min(score, 1.0)

def detect_homoglyph(url):
    for char in "аеіосурхјѕѡ":
        if char in url:
            return 1.0
    non_latin = len(re.findall(r"[^\x00-\x7F]", url))
    if non_latin > 0 and len(url) > 0 and non_latin/len(url) > 0.1:
        return 0.7
    return 0.0

def calc_encoding_ratio(url):
    encoded = len(re.findall(r"%[0-9A-Fa-f]{2}", url))
    total   = len(url)
    if total == 0:
        return 0.0
    r = encoded / total
    return 1.0 if r > 0.2 else (0.5 if r > 0.05 else 0.0)

def detect_punycode(url):
    matches = re.findall(r"xn--[a-z0-9]+", url.lower())
    if not matches:
        return 0.0
    for m in matches:
        if len(m) > 12:
            return 1.0
        if any(c.isdigit() for c in m):
            return 0.8
    return 0.5

def detect_subdomain_spam(url):
    try:
        parts = [p for p in urlparse(url).netloc.split(".") if p]
        sc    = max(0, len(parts) - 2)
        return 1.0 if sc >= 4 else (0.7 if sc >= 3 else (0.3 if sc >= 2 else 0.0))
    except Exception:
        return 0.0

def calc_visual_similarity(url, hostname):
    url_lower  = url.lower()
    host_lower = hostname.lower()
    max_sim = 0.0
    for brand in BRANDS:
        if brand in url_lower and brand not in host_lower:
            max_sim = max(max_sim, 0.9)
    return max_sim

# NEW rule-based features
def brand_in_registered_domain(registered_domain):
    rd = (registered_domain or "").lower()
    for brand in BRANDS:
        if brand in rd:
            if rd in REAL_BRAND_DOMAINS:
                return 0.0
            else:
                return 1.0
    return 0.0

def leet_in_domain_only(domain):
    domain_lower = (domain or "").lower()
    leet_map = {"4":"a","3":"e","1":"i","0":"o","5":"s","7":"t"}
    for digit in leet_map:
        if re.search(rf"[a-z]{re.escape(digit)}[a-z]", domain_lower):
            return 1.0
    return 0.0

def brand_hyphen_suspicious_word(url):
    url_lower = url.lower()
    for brand in BRANDS:
        for word in BRAND_SUSPICIOUS_WORDS:
            if f"{brand}-{word}" in url_lower:
                return 1.0
            if f"{word}-{brand}" in url_lower:
                return 1.0
    return 0.0


def extract_features(url: str) -> list:
    """Extract all 48 features — identical to feature_extraction.py."""
    try:
        if not isinstance(url, str) or len(url) < 5:
            return [0.0] * N_HEURISTIC

        url_to_parse = url if url.startswith(("http://","https://")) \
            else "http://" + url
        parsed   = urlparse(url_to_parse)
        hostname = parsed.netloc.split("@")[-1].split(":")[0] \
            if parsed.netloc else ""
        if not hostname:
            return [0.0] * N_HEURISTIC

        ext       = EXTRACTOR(hostname)
        domain    = ext.registered_domain or hostname
        subdomain = ext.subdomain or ""
        tld       = ext.suffix or ""
        url_lower = url.lower()
        url_len   = len(url)

        num_dots        = url.count(".")
        num_hyphens     = url.count("-")
        num_underscores = url.count("_")
        num_at          = url.count("@")
        num_qmark       = url.count("?")
        num_equal       = url.count("=")
        num_amp         = url.count("&")
        num_percent     = url.count("%")
        num_slashes     = url.count("/")
        num_digits      = sum(c.isdigit() for c in url)
        num_letters     = sum(c.isalpha() for c in url)
        num_upper       = sum(c.isupper() for c in url)
        num_non_ascii   = sum(ord(c) > 127 for c in url)

        path        = parsed.path or ""
        num_subdirs = max(0, path.count("/")-(1 if path.startswith("/") else 0))
        path_length = len(path)
        num_frag    = 1 if parsed.fragment else 0
        num_special = sum(c in "!$*,;()[]{}+~|" for c in url)
        num_params  = parsed.query.count("&")+1 if parsed.query else 0

        ratio_digits  = num_digits/url_len  if url_len else 0.0
        ratio_letters = num_letters/url_len if url_len else 0.0
        url_entropy   = simple_entropy(url)

        ip_flag    = 1.0 if has_ip_address(hostname) else 0.0
        risky_tld  = 1.0 if tld.lower() in RISKY_TLDS else 0.0
        https_flag = 1.0 if url.startswith("https") else 0.0
        shortened  = 1.0 if is_shortened(hostname, domain) else 0.0
        sus_words  = float(count_suspicious_words(url))

        brand_mismatch = 0.0
        for brand in BRANDS:
            if brand in url_lower and brand not in hostname.lower():
                brand_mismatch = 1.0
                break

        puny     = 1.0 if "xn--" in url_lower else 0.0
        susp_ext = 1.0 if any(url_lower.endswith(e)
                               for e in [".exe",".zip",".scr",
                                         ".jar",".msi"]) else 0.0

        spc = len([p for p in subdomain.split(".") if p]) if subdomain else 0
        has_multi_subdomain = 1.0 if spc >= 2 else 0.0
        tld_len = len(tld)

        max_digs = float(max_consecutive(url, "digit"))
        max_cons = float(max_consecutive(url, "consonant"))
        max_vows = float(max_consecutive(url, "vowel"))
        num_rep  = float(max_repeating(url))

        suspicious_port = 0.0
        try:
            port = parsed.port
            if port and port not in COMMON_PORTS:
                suspicious_port = 1.0
        except Exception:
            pass

        leet       = detect_leet_speak(url)
        homoglyph  = detect_homoglyph(url)
        enc_ratio  = calc_encoding_ratio(url)
        punycode   = detect_punycode(url)
        sub_spam   = detect_subdomain_spam(url)
        visual_sim = calc_visual_similarity(url, hostname)

        # NEW 3 rule-based features
        brand_in_dom   = brand_in_registered_domain(domain)
        leet_dom       = leet_in_domain_only(ext.domain or "")
        brand_hyp_susp = brand_hyphen_suspicious_word(url)

        return [
            float(url_len), float(path_length), float(num_dots),
            float(path.count(".")), float(num_hyphens),
            float(num_underscores), float(num_at), float(num_qmark),
            float(num_equal), float(num_amp), float(num_percent),
            float(num_digits), float(num_letters), float(num_subdirs),
            float(num_frag), float(num_special), float(num_rep),
            float(num_upper), float(num_non_ascii), float(num_slashes),
            float(num_params), ratio_digits, ratio_letters,
            url_entropy, ip_flag, float(spc),
            has_multi_subdomain, float(tld_len), risky_tld, https_flag,
            shortened, sus_words, brand_mismatch, puny, susp_ext,
            suspicious_port, max_cons, max_vows, max_digs,
            leet, homoglyph, enc_ratio, punycode, sub_spam, visual_sim,
            brand_in_dom, leet_dom, brand_hyp_susp
        ]

    except Exception:
        return [0.0] * N_HEURISTIC


def preprocess_url_for_nlp(url):
    url = str(url).strip().lower()
    url = re.sub(r"^https?://(www\.)?", "", url)
    url = url.rstrip("/")
    url = re.sub(r"/+", "/", url)
    return url


# ---------------- PREDICTION ----------------

def predict_with_model(urls, model, char_vec, word_vec,
                       scaler, threshold):
    heuristic = np.array(
        [extract_features(u) for u in urls], dtype=np.float32
    )
    heuristic        = np.nan_to_num(heuristic, nan=0.0,
                                     posinf=0.0, neginf=0.0)
    heuristic_scaled = scaler.transform(heuristic).astype(np.float32)

    processed = [preprocess_url_for_nlp(u) for u in urls]
    X_char    = char_vec.transform(processed)
    X_word    = word_vec.transform(processed)
    X_nlp     = sp.hstack([X_char, X_word], format="csr").astype(np.float32)
    X         = sp.hstack([sp.csr_matrix(heuristic_scaled), X_nlp],
                           format="csr")

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
    print("Two-layer: Whitelist + ML Model (548 features)")
    print("=" * 70)

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

    # Verify feature count matches
    expected_features = 548
    if model.n_features_in_ != expected_features:
        print(f"\n  WARNING: Model has {model.n_features_in_} features "
              f"but extraction produces {expected_features}.")
        print("  Re-run feature_extraction.py and train_model.py first.")

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
            marker = "OK" if is_correct else "WRONG"
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

    print(f"\n  Unexpected predictions:")
    found_unexpected = False
    for result, category, url in zip(results, categories, urls):
        expected = expected_map.get(category)
        if expected and result["prediction"] != expected:
            found_unexpected = True
            print(f"    WRONG: {url}")
            print(f"           Expected {expected}, "
                  f"got {result['prediction']} "
                  f"(conf: {result['confidence']}%, "
                  f"src: {result['source']})")

    if not found_unexpected:
        print("    None — all expected predictions correct")

    accuracy = correct / total
    print(f"\n{'='*70}")
    if accuracy >= 0.90:
        print(f"  VERDICT: Model is ready for production "
              f"({accuracy:.0%})")
    elif accuracy >= 0.80:
        print(f"  VERDICT: Acceptable — review wrong predictions "
              f"({accuracy:.0%})")
    else:
        print(f"  VERDICT: Needs improvement ({accuracy:.0%})")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()