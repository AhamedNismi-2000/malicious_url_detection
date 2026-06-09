#!/usr/bin/env python3
"""
feature_extraction.py
---------------------
Single unified feature extraction script.

Split-aware pipeline:
  - TF-IDF vectorizer fitted on TRAIN only, transforms val+test
  - StandardScaler fitted on TRAIN only, transforms val+test
  - No data leakage of any kind

Feature breakdown (total: 545):
  - Heuristic + Obfuscation : 45  (39 structural + 6 obfuscation)
  - Char n-gram TF-IDF      : 300
  - Word n-gram TF-IDF      : 200
  Total                     : 545

Output:
  features/features_train.npz
  features/features_val.npz
  features/features_test.npz
  models/vectorizer_char.joblib
  models/vectorizer_word.joblib
  models/scaler.joblib
"""

import os
import re
import math
import gc
import warnings
import logging
import ipaddress
import sys
from collections import Counter
from urllib.parse import urlparse
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
import tldextract
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore")

# ---------------- PATHS ----------------
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS_DIR   = os.path.join(BASE_DIR, "data", "splits")
FEATURES_DIR = os.path.join(BASE_DIR, "features")
MODELS_DIR   = os.path.join(BASE_DIR, "models")

os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(SPLITS_DIR, "train_urls.csv")
VAL_PATH   = os.path.join(SPLITS_DIR, "val_urls.csv")
TEST_PATH  = os.path.join(SPLITS_DIR, "test_urls.csv")

TRAIN_OUT = os.path.join(FEATURES_DIR, "features_train")
VAL_OUT   = os.path.join(FEATURES_DIR, "features_val")
TEST_OUT  = os.path.join(FEATURES_DIR, "features_test")

CHAR_VEC_PATH = os.path.join(MODELS_DIR, "vectorizer_char.joblib")
WORD_VEC_PATH = os.path.join(MODELS_DIR, "vectorizer_word.joblib")
SCALER_PATH   = os.path.join(MODELS_DIR, "scaler.joblib")

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(FEATURES_DIR, "feature_extraction.log"),
            encoding="utf-8",
            mode="w"
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------- CONFIG ----------------
MAX_FEAT_CHAR = 300
MAX_FEAT_WORD = 200
NGRAM_CHAR    = (2, 4)
NGRAM_WORD    = (1, 2)
MIN_DF        = 3        # lowered from 5 — smaller dataset now
CHUNK_SIZE    = 50_000

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

# Reduced set — only words almost never on legitimate sites
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

COMMON_PORTS = {80, 443, 8080, 8443, 3000, 5000, 8000, 9000}

EXTRACTOR = tldextract.TLDExtract(cache_dir=None, suffix_list_urls=None)

# 45 feature names — must stay in sync with extract_heuristic_features()
HEURISTIC_FEATURE_NAMES = [
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
    "punycode_suspicious", "subdomain_spam_score", "visual_brand_similarity"
]
N_HEURISTIC = len(HEURISTIC_FEATURE_NAMES)  # 45


# ================================================================
# HELPER FUNCTIONS
# ================================================================

def has_ip_address(hostname: str) -> bool:
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


def is_shortened(hostname: str, registered_domain: str) -> bool:
    try:
        h = hostname.lower()
        if h.startswith("www."):
            h = h[4:]
        rd = (registered_domain or "").lower()
        return (h in SHORTENERS) or (rd in SHORTENERS)
    except Exception:
        return False


def count_suspicious_words(url: str) -> int:
    url_lower = url.lower()
    return sum(1 for w in SUSPICIOUS_WORDS if w in url_lower)


def simple_entropy(s: str) -> float:
    if not s or len(s) <= 1:
        return 0.0
    cnt    = Counter(s)
    length = len(s)
    return -sum((v / length) * math.log2(v / length)
                for v in cnt.values() if v > 0)


def max_consecutive(s: str, char_type: str) -> int:
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


def max_repeating(s: str) -> int:
    if len(s) <= 1:
        return 0
    max_count = current = 1
    for i in range(1, len(s)):
        current   = current + 1 if s[i] == s[i - 1] else 1
        max_count = max(max_count, current)
    return max_count


# ================================================================
# OBFUSCATION DETECTORS
# ================================================================

def detect_leet_speak(url: str) -> float:
    """
    Flag leet speak only when digits appear inside alphabetic word
    boundaries on the domain. Avoids false positives on paths like
    /404 or filenames like setup64.exe.
    """
    url_lower = url.lower()
    try:
        domain_part = urlparse(url_lower).netloc
    except Exception:
        domain_part = url_lower

    leet_map = {
        "4": "a", "3": "e", "1": "i",
        "0": "o", "5": "s", "7": "t"
    }
    score = 0.0
    for digit in leet_map:
        pattern = rf"[a-z]{re.escape(digit)}[a-z]"
        matches = re.findall(pattern, domain_part)
        score  += len(matches) * 0.2
    return min(score, 1.0)


def detect_homoglyph(url: str) -> float:
    cyrillic = "аеіосурхјѕѡ"
    for char in cyrillic:
        if char in url:
            return 1.0
    non_latin = len(re.findall(r"[^\x00-\x7F]", url))
    if non_latin > 0 and len(url) > 0:
        if (non_latin / len(url)) > 0.1:
            return 0.7
    return 0.0


def calc_encoding_ratio(url: str) -> float:
    encoded = len(re.findall(r"%[0-9A-Fa-f]{2}", url))
    total   = len(url)
    if total == 0:
        return 0.0
    ratio = encoded / total
    if ratio > 0.2:
        return 1.0
    elif ratio > 0.05:
        return 0.5
    return 0.0


def detect_punycode(url: str) -> float:
    matches = re.findall(r"xn--[a-z0-9]+", url.lower())
    if not matches:
        return 0.0
    for m in matches:
        if len(m) > 12:
            return 1.0
        if any(c.isdigit() for c in m):
            return 0.8
    return 0.5


def detect_subdomain_spam(url: str) -> float:
    try:
        netloc = urlparse(url).netloc
        parts  = [p for p in netloc.split(".") if p]
        subdomain_count = max(0, len(parts) - 2)
        if subdomain_count >= 4:
            return 1.0
        elif subdomain_count >= 3:
            return 0.7
        elif subdomain_count >= 2:
            return 0.3
        return 0.0
    except Exception:
        return 0.0


def calc_visual_similarity(url: str, hostname: str) -> float:
    """
    FIX: Brand appears in URL path/query but NOT in hostname.
    Legitimate brand domains (paypal.com, google.com) score 0.0.
    Impersonation in path (evil.com/paypal/login) scores 0.9.
    """
    url_lower  = url.lower()
    host_lower = hostname.lower()
    max_sim    = 0.0
    for brand in BRANDS:
        if brand in url_lower and brand not in host_lower:
            max_sim = max(max_sim, 0.9)
    return max_sim


# ================================================================
# HEURISTIC FEATURE EXTRACTION
# ================================================================

def extract_heuristic_features(url: str) -> list:
    """Extract all 45 heuristic + obfuscation features for one URL."""
    try:
        if not isinstance(url, str) or len(url) < 5:
            return [0.0] * N_HEURISTIC

        url_to_parse = url if url.startswith(("http://", "https://")) \
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
        num_subdirs = max(0, path.count("/") - (1 if path.startswith("/") else 0))
        path_length = len(path)
        num_frag    = 1 if parsed.fragment else 0
        num_special = sum(c in "!$*,;()[]{}+~|" for c in url)
        num_params  = parsed.query.count("&") + 1 if parsed.query else 0

        ratio_digits  = num_digits  / url_len if url_len else 0.0
        ratio_letters = num_letters / url_len if url_len else 0.0
        url_entropy   = simple_entropy(url)

        ip_flag    = 1.0 if has_ip_address(hostname) else 0.0
        risky_tld  = 1.0 if tld.lower() in RISKY_TLDS else 0.0
        https_flag = 1.0 if url.startswith("https") else 0.0
        shortened  = 1.0 if is_shortened(hostname, domain) else 0.0
        sus_words  = float(count_suspicious_words(url))

        # FIX: check hostname not just registered domain
        brand_mismatch = 0.0
        for brand in BRANDS:
            if brand in url_lower and brand not in hostname.lower():
                brand_mismatch = 1.0
                break

        puny     = 1.0 if "xn--" in url_lower else 0.0
        susp_ext = 1.0 if any(url_lower.endswith(e)
                               for e in [".exe", ".zip", ".scr",
                                         ".jar", ".msi"]) else 0.0

        subdomain_parts_count = len([p for p in subdomain.split(".")
                                     if p]) if subdomain else 0
        has_multi_subdomain   = 1.0 if subdomain_parts_count >= 2 else 0.0
        tld_len               = len(tld)

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

        # Obfuscation features — computed ONCE here only
        leet       = detect_leet_speak(url)
        homoglyph  = detect_homoglyph(url)
        enc_ratio  = calc_encoding_ratio(url)
        punycode   = detect_punycode(url)
        sub_spam   = detect_subdomain_spam(url)
        visual_sim = calc_visual_similarity(url, hostname)

        return [
            float(url_len), float(path_length), float(num_dots),
            float(path.count(".")), float(num_hyphens), float(num_underscores),
            float(num_at), float(num_qmark), float(num_equal), float(num_amp),
            float(num_percent), float(num_digits), float(num_letters),
            float(num_subdirs), float(num_frag), float(num_special),
            float(num_rep), float(num_upper), float(num_non_ascii),
            float(num_slashes), float(num_params), ratio_digits, ratio_letters,
            url_entropy, ip_flag, float(subdomain_parts_count),
            has_multi_subdomain, float(tld_len), risky_tld, https_flag,
            shortened, sus_words, brand_mismatch, puny, susp_ext,
            suspicious_port, max_cons, max_vows, max_digs,
            leet, homoglyph, enc_ratio, punycode, sub_spam, visual_sim
        ]

    except Exception:
        return [0.0] * N_HEURISTIC


def extract_heuristic_chunk(urls_chunk: list) -> list:
    """Wrapper for multiprocessing pool."""
    return [extract_heuristic_features(u) for u in urls_chunk]


def extract_heuristic_batch(urls: list) -> np.ndarray:
    n_workers = min(cpu_count(), 8)
    chunks    = [urls[i:i + CHUNK_SIZE]
                 for i in range(0, len(urls), CHUNK_SIZE)]

    all_features = []
    if len(urls) > 100_000:
        logger.info(f"   Using {n_workers} workers for heuristic extraction")
        with Pool(n_workers) as pool:
            results = list(tqdm(
                pool.imap(extract_heuristic_chunk, chunks),
                total=len(chunks), desc="Heuristic chunks"
            ))
        for r in results:
            all_features.extend(r)
    else:
        for url in tqdm(urls, desc="Heuristic features"):
            all_features.append(extract_heuristic_features(url))

    arr = np.array(all_features, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


# ================================================================
# NLP FEATURE EXTRACTION
# ================================================================

def preprocess_url_for_nlp(url: str) -> str:
    """Minimal cleanup for NLP tokenization."""
    url = str(url).strip().lower()
    url = re.sub(r"^https?://(www\.)?", "", url)
    url = url.rstrip("/")
    url = re.sub(r"/+", "/", url)
    return url


def fit_vectorizers(train_urls: list):
    """
    Fit TF-IDF vectorizers on TRAINING URLs ONLY.
    Returns fitted char_vec and word_vec.
    """
    logger.info("   Preprocessing URLs for NLP...")
    processed = [preprocess_url_for_nlp(u)
                 for u in tqdm(train_urls, desc="NLP preprocess")]

    logger.info("   Fitting char n-gram TF-IDF on train only...")
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=NGRAM_CHAR,
        max_features=MAX_FEAT_CHAR,
        min_df=MIN_DF,
        lowercase=False,
        dtype=np.float32
    )
    char_vec.fit(processed)

    logger.info("   Fitting word n-gram TF-IDF on train only...")
    word_vec = TfidfVectorizer(
        analyzer="word",
        ngram_range=NGRAM_WORD,
        max_features=MAX_FEAT_WORD,
        min_df=MIN_DF,
        lowercase=False,
        token_pattern=r"[a-zA-Z0-9@\-\.]+",
        dtype=np.float32
    )
    word_vec.fit(processed)

    return char_vec, word_vec


def transform_nlp(urls: list,
                  char_vec: TfidfVectorizer,
                  word_vec: TfidfVectorizer) -> sp.csr_matrix:
    """Transform URLs using already-fitted vectorizers."""
    processed = [preprocess_url_for_nlp(u)
                 for u in tqdm(urls, desc="NLP transform")]
    X_char = char_vec.transform(processed)
    X_word = word_vec.transform(processed)
    return sp.hstack([X_char, X_word], format="csr").astype(np.float32)


# ================================================================
# COMBINE AND SAVE
# ================================================================

def combine_features(heuristic_arr: np.ndarray,
                     nlp_sparse: sp.csr_matrix) -> sp.csr_matrix:
    """Combine heuristic (dense) and NLP (sparse) into one sparse matrix."""
    heuristic_sparse = sp.csr_matrix(heuristic_arr)
    return sp.hstack([heuristic_sparse, nlp_sparse],
                     format="csr").astype(np.float32)


def save_features(path: str, X: sp.csr_matrix,
                  y: np.ndarray, feature_names: list):
    """Save sparse feature matrix + labels to NPZ."""
    X_csr = X.tocsr()
    np.savez_compressed(
        path,
        data=X_csr.data,
        indices=X_csr.indices,
        indptr=X_csr.indptr,
        shape=np.array(X_csr.shape),
        labels=y.astype(np.int8),
        feature_names=np.array(feature_names, dtype=object)
    )
    saved_path = path + ".npz"
    size_mb    = os.path.getsize(saved_path) / 1e6 \
        if os.path.exists(saved_path) else 0
    logger.info(
        f"   Saved: {saved_path}  "
        f"({X_csr.shape[0]:,} x {X_csr.shape[1]:,} features, "
        f"{size_mb:.1f} MB)"
    )


def load_features(path: str):
    """Load feature NPZ saved by save_features."""
    if not path.endswith(".npz"):
        path = path + ".npz"
    data = np.load(path, allow_pickle=True)
    X    = sp.csr_matrix(
        (data["data"], data["indices"], data["indptr"]),
        shape=tuple(data["shape"])
    )
    y             = data["labels"].astype(int)
    feature_names = list(data["feature_names"])
    return X, y, feature_names


# ================================================================
# MAIN
# ================================================================

def process_split(name: str, path: str,
                  char_vec: TfidfVectorizer,
                  word_vec: TfidfVectorizer,
                  scaler: StandardScaler,
                  feature_names: list,
                  out_path: str,
                  fit_scaler: bool = False):
    """
    Full feature extraction pipeline for one split.
    fit_scaler=True ONLY for train split.
    Val and test use transform only — no fitting.
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {name} split: {path}")
    logger.info(f"{'='*60}")

    df   = pd.read_csv(path, dtype={"url": str, "label": int})
    df   = df.dropna(subset=["url", "label"]).reset_index(drop=True)
    urls = df["url"].tolist()
    y    = df["label"].values.astype(np.int8)

    logger.info(f"   {name}: {len(urls):,} URLs")
    label_counts = np.bincount(y.astype(int))
    logger.info(f"   Labels — Benign: {label_counts[0]:,} | "
                f"Malicious: {label_counts[1]:,}")

    # 1. Heuristic features
    logger.info("   Extracting heuristic features...")
    heuristic_arr = extract_heuristic_batch(urls)
    logger.info(f"   Heuristic shape: {heuristic_arr.shape}")

    # 2. Scale — fit on train only
    if fit_scaler:
        logger.info("   Fitting scaler on TRAIN heuristic features only...")
        scaler.fit(heuristic_arr)
        joblib.dump(scaler, SCALER_PATH)
        logger.info(f"   Scaler saved: {SCALER_PATH}")

    heuristic_scaled = scaler.transform(heuristic_arr).astype(np.float32)
    del heuristic_arr
    gc.collect()

    # 3. NLP features
    logger.info("   Extracting NLP features...")
    nlp_sparse = transform_nlp(urls, char_vec, word_vec)
    logger.info(f"   NLP shape: {nlp_sparse.shape}")

    # 4. Combine
    logger.info("   Combining heuristic + NLP features...")
    X = combine_features(heuristic_scaled, nlp_sparse)
    del heuristic_scaled, nlp_sparse
    gc.collect()
    logger.info(f"   Combined shape: {X.shape}")

    # 5. Save
    save_features(out_path, X, y, feature_names)
    del X
    gc.collect()


def main():
    logger.info("FEATURE EXTRACTION PIPELINE")
    logger.info("=" * 60)
    logger.info("Split-aware: vectorizer and scaler fitted on TRAIN only")
    logger.info("=" * 60)

    # Verify split files exist
    for p in [TRAIN_PATH, VAL_PATH, TEST_PATH]:
        if not os.path.exists(p):
            logger.error(f"Split file not found: {p}")
            logger.error("Run split.py first.")
            return

    # Build feature name list
    char_feature_names = [f"char_{i}" for i in range(MAX_FEAT_CHAR)]
    word_feature_names = [f"word_{i}" for i in range(MAX_FEAT_WORD)]
    feature_names      = (HEURISTIC_FEATURE_NAMES
                          + char_feature_names
                          + word_feature_names)
    logger.info(f"Total features: {len(feature_names):,}")

    # Initialize scaler
    scaler = StandardScaler()

    # Load train URLs to fit vectorizers
    logger.info("\nLoading train URLs to fit vectorizers...")
    df_train   = pd.read_csv(TRAIN_PATH, dtype={"url": str, "label": int})
    df_train   = df_train.dropna(subset=["url"]).reset_index(drop=True)
    train_urls = df_train["url"].tolist()
    logger.info(f"   Train URLs: {len(train_urls):,}")

    # Fit vectorizers on TRAIN ONLY
    logger.info("\nFitting TF-IDF vectorizers on TRAIN only...")
    char_vec, word_vec = fit_vectorizers(train_urls)
    joblib.dump(char_vec, CHAR_VEC_PATH)
    joblib.dump(word_vec, WORD_VEC_PATH)
    logger.info(f"   Char vectorizer saved: {CHAR_VEC_PATH}")
    logger.info(f"   Word vectorizer saved: {WORD_VEC_PATH}")

    del train_urls, df_train
    gc.collect()

    # Process TRAIN — fit scaler here
    process_split(
        name="TRAIN", path=TRAIN_PATH,
        char_vec=char_vec, word_vec=word_vec,
        scaler=scaler, feature_names=feature_names,
        out_path=TRAIN_OUT, fit_scaler=True
    )

    # Process VAL — transform only
    process_split(
        name="VAL", path=VAL_PATH,
        char_vec=char_vec, word_vec=word_vec,
        scaler=scaler, feature_names=feature_names,
        out_path=VAL_OUT, fit_scaler=False
    )

    # Process TEST — transform only
    process_split(
        name="TEST", path=TEST_PATH,
        char_vec=char_vec, word_vec=word_vec,
        scaler=scaler, feature_names=feature_names,
        out_path=TEST_OUT, fit_scaler=False
    )

    logger.info("\n" + "=" * 60)
    logger.info("FEATURE EXTRACTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\nSaved files:")
    logger.info(f"  {TRAIN_OUT}.npz")
    logger.info(f"  {VAL_OUT}.npz")
    logger.info(f"  {TEST_OUT}.npz")
    logger.info(f"  {CHAR_VEC_PATH}")
    logger.info(f"  {WORD_VEC_PATH}")
    logger.info(f"  {SCALER_PATH}")
    logger.info(f"\nNext step: run train_model.py")


if __name__ == "__main__":
    main()