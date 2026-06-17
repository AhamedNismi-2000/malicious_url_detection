#!/usr/bin/env python3
"""
model_loader.py
---------------
Loads all model artifacts once at startup and exposes a single
predict_url() function used by routes.py.

Two-layer prediction:
  Layer 1: Whitelist  — known trusted domains -> BENIGN immediately
  Layer 2: ML model   — 548 features -> RandomForest -> threshold
"""

import os
import re
import json
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
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

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

# ---------------- CONSTANTS (must match feature_extraction.py) ----------------
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

EXTRACTOR   = tldextract.TLDExtract(cache_dir=None, suffix_list_urls=None)
N_HEURISTIC = 48


# ================================================================
# HELPER FUNCTIONS  (identical to feature_extraction.py)
# ================================================================

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
    length = len(s)
    return -sum((v / length) * math.log2(v / length)
                for v in cnt.values() if v > 0)


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
            current = 0
    return max(max_count, current)


def max_repeating(s):
    if len(s) <= 1:
        return 0
    max_count = current = 1
    for i in range(1, len(s)):
        current = current + 1 if s[i] == s[i - 1] else 1
        max_count = max(max_count, current)
    return max_count


# ================================================================
# OBFUSCATION DETECTORS  (identical to feature_extraction.py)
# ================================================================

def detect_leet_speak(url):
    url_lower = url.lower()
    try:
        domain_part = urlparse(url_lower).netloc
    except Exception:
        domain_part = url_lower
    leet_map = {"4": "a", "3": "e", "1": "i", "0": "o", "5": "s", "7": "t"}
    score = 0.0
    for digit in leet_map:
        pattern = rf"[a-z]{re.escape(digit)}[a-z]"
        score += len(re.findall(pattern, domain_part)) * 0.2
    return min(score, 1.0)


def detect_homoglyph(url):
    for char in "аеіосурхјѕѡ":
        if char in url:
            return 1.0
    non_latin = len(re.findall(r"[^\x00-\x7F]", url))
    if non_latin > 0 and len(url) > 0 and (non_latin / len(url)) > 0.1:
        return 0.7
    return 0.0


def calc_encoding_ratio(url):
    encoded = len(re.findall(r"%[0-9A-Fa-f]{2}", url))
    total = len(url)
    if total == 0:
        return 0.0
    ratio = encoded / total
    if ratio > 0.2:
        return 1.0
    elif ratio > 0.05:
        return 0.5
    return 0.0


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
        sc = max(0, len(parts) - 2)
        if sc >= 4:
            return 1.0
        elif sc >= 3:
            return 0.7
        elif sc >= 2:
            return 0.3
        return 0.0
    except Exception:
        return 0.0


def calc_visual_similarity(url, hostname):
    url_lower = url.lower()
    host_lower = hostname.lower()
    max_sim = 0.0
    for brand in BRANDS:
        if brand in url_lower and brand not in host_lower:
            max_sim = max(max_sim, 0.9)
    return max_sim


# ================================================================
# RULE-BASED FEATURES  (identical to feature_extraction.py)
# ================================================================

def brand_in_registered_domain(registered_domain):
    rd = (registered_domain or "").lower()
    for brand in BRANDS:
        if brand in rd:
            return 0.0 if rd in REAL_BRAND_DOMAINS else 1.0
    return 0.0


def leet_in_domain_only(domain):
    domain_lower = (domain or "").lower()
    leet_map = {"4": "a", "3": "e", "1": "i", "0": "o", "5": "s", "7": "t"}
    for digit in leet_map:
        if re.search(rf"[a-z]{re.escape(digit)}[a-z]", domain_lower):
            return 1.0
    return 0.0


def brand_hyphen_suspicious_word(url):
    url_lower = url.lower()
    for brand in BRANDS:
        for word in BRAND_SUSPICIOUS_WORDS:
            if f"{brand}-{word}" in url_lower or f"{word}-{brand}" in url_lower:
                return 1.0
    return 0.0


# ================================================================
# MAIN FEATURE EXTRACTION  (identical order to feature_extraction.py)
# ================================================================

def extract_features(url: str) -> list:
    """Extract all 48 heuristic features for one URL."""
    try:
        if not isinstance(url, str) or len(url) < 5:
            return [0.0] * N_HEURISTIC

        url_to_parse = url if url.startswith(("http://", "https://")) \
            else "http://" + url
        parsed = urlparse(url_to_parse)
        hostname = parsed.netloc.split("@")[-1].split(":")[0] \
            if parsed.netloc else ""

        if not hostname:
            return [0.0] * N_HEURISTIC

        ext = EXTRACTOR(hostname)
        domain = ext.registered_domain or hostname
        subdomain = ext.subdomain or ""
        tld = ext.suffix or ""
        url_lower = url.lower()
        url_len = len(url)

        num_dots = url.count(".")
        num_hyphens = url.count("-")
        num_underscores = url.count("_")
        num_at = url.count("@")
        num_qmark = url.count("?")
        num_equal = url.count("=")
        num_amp = url.count("&")
        num_percent = url.count("%")
        num_slashes = url.count("/")
        num_digits = sum(c.isdigit() for c in url)
        num_letters = sum(c.isalpha() for c in url)
        num_upper = sum(c.isupper() for c in url)
        num_non_ascii = sum(ord(c) > 127 for c in url)

        path = parsed.path or ""
        num_subdirs = max(0, path.count("/") - (1 if path.startswith("/") else 0))
        path_length = len(path)
        num_frag = 1 if parsed.fragment else 0
        num_special = sum(c in "!$*,;()[]{}+~|" for c in url)
        num_params = parsed.query.count("&") + 1 if parsed.query else 0

        ratio_digits = num_digits / url_len if url_len else 0.0
        ratio_letters = num_letters / url_len if url_len else 0.0
        url_entropy = simple_entropy(url)

        ip_flag = 1.0 if has_ip_address(hostname) else 0.0
        risky_tld = 1.0 if tld.lower() in RISKY_TLDS else 0.0
        https_flag = 1.0 if url.startswith("https") else 0.0
        shortened = 1.0 if is_shortened(hostname, domain) else 0.0
        sus_words = float(count_suspicious_words(url))

        brand_mismatch = 0.0
        for brand in BRANDS:
            if brand in url_lower and brand not in hostname.lower():
                brand_mismatch = 1.0
                break

        puny = 1.0 if "xn--" in url_lower else 0.0
        susp_ext = 1.0 if any(url_lower.endswith(e)
                               for e in [".exe", ".zip", ".scr",
                                         ".jar", ".msi"]) else 0.0

        spc = len([p for p in subdomain.split(".") if p]) if subdomain else 0
        has_multi_subdomain = 1.0 if spc >= 2 else 0.0
        tld_len = len(tld)

        max_digs = float(max_consecutive(url, "digit"))
        max_cons = float(max_consecutive(url, "consonant"))
        max_vows = float(max_consecutive(url, "vowel"))
        num_rep = float(max_repeating(url))

        suspicious_port = 0.0
        try:
            port = parsed.port
            if port and port not in COMMON_PORTS:
                suspicious_port = 1.0
        except Exception:
            pass

        leet = detect_leet_speak(url)
        homoglyph = detect_homoglyph(url)
        enc_ratio = calc_encoding_ratio(url)
        punycode = detect_punycode(url)
        sub_spam = detect_subdomain_spam(url)
        visual_sim = calc_visual_similarity(url, hostname)

        brand_in_dom = brand_in_registered_domain(domain)
        leet_dom = leet_in_domain_only(ext.domain or "")
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


def preprocess_url_for_nlp(url: str) -> str:
    url = str(url).strip().lower()
    url = re.sub(r"^https?://", "", url)
    url = re.sub(r"[.\-_/=?&@#+:]", " ", url)
    url = re.sub(r"\s+", " ", url).strip()
    return url


# ================================================================
# WHITELIST CHECK
# ================================================================

def is_whitelisted(url: str) -> bool:
    try:
        parsed = urlparse(url.lower())
        ext = EXTRACTOR(parsed.netloc)
        reg_dom = ext.registered_domain or ""
        return reg_dom in TRUSTED_DOMAINS
    except Exception:
        return False


# ================================================================
# MODEL LOADER (singleton)
# ================================================================

class URLClassifier:
    """
    Loads all model artifacts once and provides predict_url().
    Instantiate once at app startup, reuse for all requests.
    """

    def __init__(self):
        for name, path in [
            ("Model", MODEL_PATH),
            ("Char vectorizer", CHAR_VEC_PATH),
            ("Word vectorizer", WORD_VEC_PATH),
            ("Scaler", SCALER_PATH),
            ("Threshold", THRESHOLD_PATH),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{name} not found at {path}. "
                    f"Run the training pipeline first."
                )

        self.model = joblib.load(MODEL_PATH)
        self.char_vec = joblib.load(CHAR_VEC_PATH)
        self.word_vec = joblib.load(WORD_VEC_PATH)
        self.scaler = joblib.load(SCALER_PATH)

        with open(THRESHOLD_PATH) as f:
            self.threshold = json.load(f)["threshold"]

        self.malicious_col = list(self.model.classes_).index(1)

    def info(self) -> dict:
        return {
            "n_estimators": self.model.n_estimators,
            "n_features": self.model.n_features_in_,
            "threshold": self.threshold,
            "whitelist_size": len(TRUSTED_DOMAINS),
        }

    def _predict_with_model(self, url: str) -> dict:
        heuristic = np.array([extract_features(url)], dtype=np.float32)
        heuristic = np.nan_to_num(heuristic, nan=0.0, posinf=0.0, neginf=0.0)
        heuristic_scaled = self.scaler.transform(heuristic).astype(np.float32)

        processed = [preprocess_url_for_nlp(url)]
        X_char = self.char_vec.transform(processed)
        X_word = self.word_vec.transform(processed)
        X_nlp = sp.hstack([X_char, X_word], format="csr").astype(np.float32)

        X = sp.hstack([sp.csr_matrix(heuristic_scaled), X_nlp], format="csr")

        y_proba = self.model.predict_proba(X)[:, self.malicious_col]
        confidence = float(y_proba[0])
        prediction = "MALICIOUS" if confidence >= self.threshold else "BENIGN"

        return {
            "url": url,
            "prediction": prediction,
            "confidence": round(confidence * 100, 2),
            "threshold": round(self.threshold * 100, 2),
            "source": "model",
        }

    def predict_url(self, url: str) -> dict:
        """
        Two-layer prediction for a single URL.
        Returns a dict with url, prediction, confidence, threshold, source.
        """
        url = (url or "").strip()
        if not url:
            return {
                "url": url,
                "prediction": "UNKNOWN",
                "confidence": 0.0,
                "threshold": round(self.threshold * 100, 2),
                "source": "invalid",
                "error": "Empty URL",
            }

        if is_whitelisted(url):
            return {
                "url": url,
                "prediction": "BENIGN",
                "confidence": 0.0,
                "threshold": round(self.threshold * 100, 2),
                "source": "whitelist",
            }

        return self._predict_with_model(url)

    def predict_batch(self, urls: list) -> list:
        """Predict multiple URLs. Whitelisted ones skip the model."""
        results = [None] * len(urls)
        model_idx, model_urls = [], []

        for i, url in enumerate(urls):
            url = (url or "").strip()
            if not url:
                results[i] = {
                    "url": url, "prediction": "UNKNOWN",
                    "confidence": 0.0,
                    "threshold": round(self.threshold * 100, 2),
                    "source": "invalid", "error": "Empty URL"
                }
            elif is_whitelisted(url):
                results[i] = {
                    "url": url, "prediction": "BENIGN",
                    "confidence": 0.0,
                    "threshold": round(self.threshold * 100, 2),
                    "source": "whitelist"
                }
            else:
                model_idx.append(i)
                model_urls.append(url)

        if model_urls:
            heuristic = np.array(
                [extract_features(u) for u in model_urls], dtype=np.float32
            )
            heuristic = np.nan_to_num(heuristic, nan=0.0, posinf=0.0, neginf=0.0)
            heuristic_scaled = self.scaler.transform(heuristic).astype(np.float32)

            processed = [preprocess_url_for_nlp(u) for u in model_urls]
            X_char = self.char_vec.transform(processed)
            X_word = self.word_vec.transform(processed)
            X_nlp = sp.hstack([X_char, X_word], format="csr").astype(np.float32)
            X = sp.hstack([sp.csr_matrix(heuristic_scaled), X_nlp], format="csr")

            y_proba = self.model.predict_proba(X)[:, self.malicious_col]

            for j, idx in enumerate(model_idx):
                confidence = float(y_proba[j])
                prediction = "MALICIOUS" if confidence >= self.threshold else "BENIGN"
                results[idx] = {
                    "url": model_urls[j],
                    "prediction": prediction,
                    "confidence": round(confidence * 100, 2),
                    "threshold": round(self.threshold * 100, 2),
                    "source": "model",
                }

        return results


# ---------------- SINGLETON INSTANCE ----------------
# routes.py imports this directly: from model_loader import classifier
classifier = URLClassifier()