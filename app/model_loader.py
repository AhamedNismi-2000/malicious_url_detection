"""
model_loader.py
---------------
Loads model artefacts and exposes three public methods:

  predict_url(url)                   -> dict
  predict_batch(urls)                -> list[dict]
  explain_url(url, num_features=30)  -> dict

Prediction pipeline:
  Layer 0: Whitelist check
  Layer 1: Google Safe Browsing API (known threats)
  Layer 2: ML Model (Random Forest 558 features)
  Layer 3: Post-prediction confidence adjustment

Fixes applied:
  - FIX 1: https_flag backup threshold -0.1
  - FIX 2: explain_url skips https_flag when site has HTTPS
  - FIX 3: feature_to_natural_language validates flag values
  - FIX 4: _classify reduces confidence for clean HTTP sites
  - FIX 5: brand reason skipped when no brand detected
  - FIX 6: Google Safe Browsing API layer added
  - FIX 7: WHOIS removed — uses http_no_brand_no_age feature instead
  - FIX 8: Updated for 558 features (new feature_extraction.py)
"""

import json
import os
import re
import sys
import socket
import threading
import warnings
from typing import Optional

import joblib
import numpy as np
import requests

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Locate project root ───────────────────────────────────────────────────────
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.abspath(os.path.join(_APP_DIR, ".."))
_SCRIPTS = os.path.join(_ROOT, "scripts")

if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from feature_extraction import (
    extract_heuristic_features,
    segment_url,
    SHORTENERS,
    HEURISTIC_FEATURE_NAMES,
    N_HEURISTIC,
)

# ── Path constants ────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(_ROOT, "models")
DATA_DIR   = os.path.join(_ROOT, "data")

# ── Google Safe Browsing API ──────────────────────────────────────────────────
GSB_API_KEY = os.environ.get("GOOGLE_SAFE_BROWSING_API_KEY", "")
GSB_URL     = "https://safebrowsing.googleapis.com/v4/threatMatches:find"

# ── Feature names — must match feature_extraction.py exactly ─────────────────
HEURISTIC_FEATURES: list[str] = HEURISTIC_FEATURE_NAMES  # 56 features

FEATURE_NAMES: list[str] = (
    HEURISTIC_FEATURES
    + [f"char_{i}" for i in range(300)]
    + [f"word_{i}" for i in range(202)]
)

_FEAT_IDX = {name: i for i, name in enumerate(HEURISTIC_FEATURES)}

_CATEGORICAL_FEATURE_NAMES: list[str] = [
    "ip_flag", "has_multi_subdomain", "risky_tld", "https_flag",
    "shortened", "sus_words", "brand_mismatch", "puny", "susp_ext",
    "suspicious_port", "brand_in_domain", "leet_in_domain",
    "brand_hyphen_suspicious", "has_redirect", "double_slash_in_path",
    "abnormal_subdomain", "http_no_brand_no_age",
]

_PRIVATE_IP_RE = re.compile(
    r"^(10\.|172\.(1[6-9]|2\d|3[01])\.|192\.168\.|127\.|0\.0\.0\.0|::1)"
)

# ── Brand map ─────────────────────────────────────────────────────────────────
BRAND_MAP = {
    "paypal"       : ("PayPal",         "paypal.com"),
    "amazon"       : ("Amazon",         "amazon.com"),
    "microsoft"    : ("Microsoft",      "microsoft.com"),
    "apple"        : ("Apple",          "apple.com"),
    "google"       : ("Google",         "google.com"),
    "facebook"     : ("Facebook",       "facebook.com"),
    "netflix"      : ("Netflix",        "netflix.com"),
    "bankofamerica": ("Bank of America","bankofamerica.com"),
    "wellsfargo"   : ("Wells Fargo",    "wellsfargo.com"),
    "whatsapp"     : ("WhatsApp",       "whatsapp.com"),
    "instagram"    : ("Instagram",      "instagram.com"),
    "twitter"      : ("Twitter",        "twitter.com"),
    "linkedin"     : ("LinkedIn",       "linkedin.com"),
    "ebay"         : ("eBay",           "ebay.com"),
    "visa"         : ("Visa",           "visa.com"),
    "mastercard"   : ("Mastercard",     "mastercard.com"),
    "chase"        : ("Chase Bank",     "chase.com"),
    "citi"         : ("Citibank",       "citibank.com"),
    "dropbox"      : ("Dropbox",        "dropbox.com"),
    "steam"        : ("Steam",          "steampowered.com"),
    "dhl"          : ("DHL",            "dhl.com"),
    "fedex"        : ("FedEx",          "fedex.com"),
    "ups"          : ("UPS",            "ups.com"),
}

# ── Natural language templates ────────────────────────────────────────────────
_NL_TEMPLATES = {
    "brand_in_domain"        : {
        "mal": "This site is pretending to be {brand} — the real website is {real_domain}",
        "ben": "No brand impersonation detected",
    },
    "brand_hyphen_suspicious": {
        "mal": "The domain uses a fake {brand} pattern (e.g. {brand}-security.com)",
        "ben": "No suspicious brand-hyphen pattern found",
    },
    "brand_mismatch"         : {
        "mal": "{brand} name appears in the URL but this is not the real {brand} website",
        "ben": "Brand name matches the actual domain",
    },
    "leet_in_domain"         : {
        "mal": "The domain disguises a brand name using look-alike characters (e.g. amaz0n, paypa1)",
        "ben": "No character substitution tricks detected",
    },
    "visual_brand_similarity": {
        "mal": "This domain looks visually similar to a well-known brand website",
        "ben": "Domain does not visually resemble known brands",
    },
    "homoglyph_suspicious"   : {
        "mal": "The URL contains look-alike characters designed to deceive (e.g. Cyrillic letters)",
        "ben": "No deceptive look-alike characters found",
    },
    "leet_speak_score"       : {
        "mal": "The URL uses digit substitutions to disguise words (leet speak)",
        "ben": "No leet speak detected",
    },
    "risky_tld"              : {
        "mal": "This site uses a high-risk domain ending commonly used for phishing",
        "ben": "Domain ending appears legitimate",
    },
    "ip_flag"                : {
        "mal": "The site uses a raw IP address instead of a proper domain name — a common phishing trick",
        "ben": "Site uses a proper domain name",
    },
    "shortened"              : {
        "mal": "This is a shortened URL hiding the real destination",
        "ben": "URL is not shortened",
    },
    "suspicious_port"        : {
        "mal": "The site runs on an unusual port number which legitimate sites rarely use",
        "ben": "Site uses a standard port",
    },
    "has_multi_subdomain"    : {
        "mal": "The URL has an unusual number of subdomains — a common phishing tactic",
        "ben": "Normal subdomain structure",
    },
    "subdomain_spam_score"   : {
        "mal": "The domain has excessive subdomains designed to confuse users",
        "ben": "Subdomain structure looks normal",
    },
    "puny"                   : {
        "mal": "The domain uses international character encoding to disguise its true identity",
        "ben": "No punycode tricks detected",
    },
    "punycode_suspicious"    : {
        "mal": "The domain uses punycode encoding to impersonate a legitimate website",
        "ben": "Punycode usage looks normal",
    },
    "sus_words"              : {
        "mal": "The URL contains phishing keywords such as 'security', 'alert', or 'verify'",
        "ben": "No phishing keywords found",
    },
    "url_entropy"            : {
        "mal": "The domain name appears randomly generated — a sign of automated phishing",
        "ben": "Domain name entropy looks normal",
    },
    "url_len"                : {
        "mal": "The URL is unusually long — often used to hide the real destination",
        "ben": "URL length looks normal",
    },
    "num_hyphens"            : {
        "mal": "The domain contains excessive hyphens which is uncommon in legitimate sites",
        "ben": "Normal use of hyphens",
    },
    "num_at"                 : {
        "mal": "The URL contains an @ symbol which can be used to disguise the real destination",
        "ben": "No @ symbol tricks detected",
    },
    "num_percent"            : {
        "mal": "The URL uses heavy percent-encoding which may be hiding malicious content",
        "ben": "URL encoding looks normal",
    },
    "encoding_ratio"         : {
        "mal": "An unusually high proportion of the URL is percent-encoded — possible obfuscation",
        "ben": "URL encoding ratio is normal",
    },
    "susp_ext"               : {
        "mal": "The URL points to a suspicious file type (e.g. .exe, .zip, .scr)",
        "ben": "File extension looks safe",
    },
    "num_non_ascii"          : {
        "mal": "The URL contains non-standard characters that may be used to deceive",
        "ben": "URL uses standard characters only",
    },
    "ratio_digits"           : {
        "mal": "The URL contains an unusually high number of digits",
        "ben": "Digit ratio looks normal",
    },
    "https_flag"             : {
        "mal": "This site does not use HTTPS — your connection may not be secure",
        "ben": "Site uses HTTPS encryption",
    },
    "http_no_brand_no_age"   : {
        "mal": "This site uses HTTP without HTTPS and shows other suspicious signals",
        "ben": "Site connection appears normal",
    },
    "domain_len"             : {
        "mal": "The domain name is unusually short — often seen in newly registered phishing domains",
        "ben": "Domain name length looks normal",
    },
    "domain_digit_ratio"     : {
        "mal": "The domain name contains an unusually high proportion of digits",
        "ben": "Domain digit ratio looks normal",
    },
    "max_domain_digits"      : {
        "mal": "The domain name contains a long sequence of digits — a common sign of generated domains",
        "ben": "No suspicious digit sequences in domain",
    },
    "path_depth"             : {
        "mal": "The URL has an unusually deep path structure — often used to mimic legitimate sites",
        "ben": "URL path depth looks normal",
    },
    "has_redirect"           : {
        "mal": "The URL contains a redirect parameter — often used to send users to malicious sites",
        "ben": "No suspicious redirect parameters found",
    },
    "double_slash_in_path"   : {
        "mal": "The URL path contains double slashes — a common obfuscation technique",
        "ben": "URL path structure looks normal",
    },
    "abnormal_subdomain"     : {
        "mal": "The subdomain contains suspicious patterns such as random digits or hex strings",
        "ben": "Subdomain looks normal",
    },
}

# ── Backup rule-based checks ──────────────────────────────────────────────────
_BACKUP_CHECKS = [
    ("brand_in_domain",          0.5, "This site is pretending to be {brand} — the real website is {real_domain}"),
    ("brand_hyphen_suspicious",  0.5, "The domain uses a fake {brand} pattern (e.g. {brand}-security.com)"),
    ("sus_words",                0.5, "The URL contains phishing keywords such as 'security', 'alert', or 'verify'"),
    ("brand_mismatch",           0.5, "{brand} name appears in the URL but this is not the real {brand} website"),
    ("risky_tld",                0.5, "This site uses a high-risk domain ending commonly used for phishing"),
    ("leet_in_domain",           0.5, "The domain disguises a brand name using look-alike characters (e.g. amaz0n)"),
    ("ip_flag",                  0.5, "The site uses a raw IP address instead of a proper domain name"),
    ("shortened",                0.5, "This is a shortened URL hiding the real destination"),
    ("puny",                     0.5, "The domain uses international character encoding to disguise its identity"),
    ("susp_ext",                 0.5, "The URL points to a suspicious file type (e.g. .exe, .zip, .scr)"),
    ("has_redirect",             0.5, "The URL contains a redirect parameter — often used in phishing attacks"),
    ("double_slash_in_path",     0.5, "The URL path contains double slashes — a common obfuscation technique"),
    ("abnormal_subdomain",       0.5, "The subdomain contains suspicious patterns such as random digits"),
    ("http_no_brand_no_age",     0.5, "This site uses HTTP without HTTPS and shows other suspicious signals"),
    # FIX 1: threshold -0.1 so only fires when https_flag truly 0
    ("https_flag",              -0.1, "This site does not use HTTPS — your connection may not be secure"),
    ("num_hyphens",              2.0, "The domain contains excessive hyphens which is uncommon in legitimate sites"),
    ("path_depth",               3.0, "The URL has an unusually deep path — often used to mimic legitimate sites"),
]


# ── Google Safe Browsing ──────────────────────────────────────────────────────

def check_google_safe_browsing(url: str) -> tuple[bool, str]:
    """
    Returns (is_malicious, threat_type)
    Checks URL against Google Safe Browsing API.
    Returns False if API key not configured.
    """
    if not GSB_API_KEY:
        return False, ""
    try:
        payload = {
            "client": {
                "clientId":      "malicious-url-detector",
                "clientVersion": "1.0.0"
            },
            "threatInfo": {
                "threatTypes": [
                    "MALWARE",
                    "SOCIAL_ENGINEERING",
                    "UNWANTED_SOFTWARE",
                    "POTENTIALLY_HARMFUL_APPLICATION"
                ],
                "platformTypes":    ["ANY_PLATFORM"],
                "threatEntryTypes": ["URL"],
                "threatEntries":    [{"url": url}]
            }
        }
        response = requests.post(
            f"{GSB_URL}?key={GSB_API_KEY}",
            json=payload,
            timeout=5
        )
        data = response.json()
        if data.get("matches"):
            threat_type = data["matches"][0].get("threatType", "MALICIOUS")
            return True, threat_type
        return False, ""
    except Exception:
        return False, ""


# ── Brand detection ───────────────────────────────────────────────────────────

def detect_brand(url: str) -> tuple[Optional[str], Optional[str]]:
    url_lower = url.lower()
    for keyword, (display_name, real_domain) in BRAND_MAP.items():
        if keyword in url_lower:
            host = re.sub(r"^https?://", "", url_lower).split("/")[0]
            reg  = ".".join(host.split(".")[-2:]) if "." in host else host
            if reg != real_domain:
                return display_name, real_domain
    return None, None


# ── Natural language explanation ──────────────────────────────────────────────

def feature_to_natural_language(
    feature: str,
    weight: float,
    value: float,
    brand_name: Optional[str] = None,
    real_domain: Optional[str] = None,
) -> Optional[str]:
    if feature.startswith("char_") or feature.startswith("word_"):
        return None

    # FIX 5: skip brand reasons when no brand detected
    if brand_name is None and weight > 0 and any(x in feature for x in
        ["brand", "visual_brand"]):
        return None

    # FIX 3: never show malicious reason when flag is actually clean
    _flag_sanity = {
        "https_flag"             : lambda v: v != 1.0,
        "ip_flag"                : lambda v: v != 0.0,
        "risky_tld"              : lambda v: v != 0.0,
        "shortened"              : lambda v: v != 0.0,
        "brand_in_domain"        : lambda v: v != 0.0,
        "leet_in_domain"         : lambda v: v != 0.0,
        "brand_hyphen_suspicious": lambda v: v != 0.0,
        "brand_mismatch"         : lambda v: v != 0.0,
        "puny"                   : lambda v: v != 0.0,
        "susp_ext"               : lambda v: v != 0.0,
        "suspicious_port"        : lambda v: v != 0.0,
        "sus_words"              : lambda v: v != 0.0,
        "has_redirect"           : lambda v: v != 0.0,
        "double_slash_in_path"   : lambda v: v != 0.0,
        "abnormal_subdomain"     : lambda v: v != 0.0,
        "http_no_brand_no_age"   : lambda v: v != 0.0,
    }

    if weight > 0:
        check = _flag_sanity.get(feature)
        if check and not check(value):
            return None

    template = _NL_TEMPLATES.get(feature)
    if not template:
        return None
    direction = "mal" if weight > 0 else "ben"
    sentence  = template[direction]
    bn = brand_name or "a known brand"
    rd = real_domain or "the official website"
    return sentence.replace("{brand}", bn).replace("{real_domain}", rd)


def _build_backup_reasons(
    heuristic: list,
    brand_name: Optional[str],
    real_domain: Optional[str],
    existing: set,
    needed: int,
) -> list[str]:
    bn      = brand_name or "a known brand"
    rd      = real_domain or "the official website"
    reasons = []

    for feat_name, threshold, template in _BACKUP_CHECKS:
        if len(reasons) >= needed:
            break
        idx = _FEAT_IDX.get(feat_name, -1)
        if idx < 0:
            continue
        val = heuristic[idx]

        if threshold < 0:
            triggered = val < abs(threshold)
        else:
            triggered = val >= threshold
        if not triggered:
            continue

        # FIX 5: skip brand-related reasons when no brand detected
        if brand_name is None and any(x in feat_name for x in
            ["brand", "leet", "visual"]):
            continue

        sentence = template.replace("{brand}", bn).replace("{real_domain}", rd)
        if sentence not in existing:
            existing.add(sentence)
            reasons.append(sentence)

    return reasons


# ── Reverse DNS ───────────────────────────────────────────────────────────────

def reverse_dns(ip: str, timeout: int = 3) -> Optional[str]:
    if _PRIVATE_IP_RE.match(ip):
        return None
    try:
        socket.setdefaulttimeout(timeout)
        hostname = socket.gethostbyaddr(ip)[0]
        return hostname.lower().rstrip(".")
    except (socket.herror, socket.gaierror, OSError):
        return None


# ── URL Unshortening ──────────────────────────────────────────────────────────

def unshorten_url(url: str, timeout: int = 5) -> tuple[str, bool]:
    try:
        resp = requests.head(
            url, allow_redirects=True, timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                     "AppleWebKit/537.36 Chrome/120.0 Safari/537.36"},
        )
        final          = resp.url
        was_redirected = final.rstrip("/") != url.rstrip("/")
        return final, was_redirected
    except Exception:
        return url, False


def _is_shortener(url: str) -> bool:
    try:
        cleaned = re.sub(r"^https?://", "", url, flags=re.IGNORECASE)
        host    = cleaned.split("/")[0].split(":")[0].lower()
        parts   = host.split(".")
        rd      = ".".join(parts[-2:]) if len(parts) >= 2 else host
        return rd in SHORTENERS or host in SHORTENERS
    except Exception:
        return False


def _extract_ip(url: str) -> Optional[str]:
    try:
        cleaned = re.sub(r"^https?://", "", url, flags=re.IGNORECASE)
        host    = cleaned.split("/")[0].split(":")[0].strip("[]")
        parts   = host.split(".")
        if len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255
                                    for p in parts):
            return host
        if ":" in host:
            return host
        return None
    except Exception:
        return None


# ── Whitelist ─────────────────────────────────────────────────────────────────

def _load_whitelist() -> set:
    whitelist = set()

    # Manual additions — known FP sites
    WHITELIST_MANUAL = {
        "dailyremote.com", "jobspresso.co", "remoteok.com",
        "weworkremotely.com", "flexjobs.com", "remote.co",
        "linkedin.com", "indeed.com", "glassdoor.com",
        "wellfound.com", "monster.com", "ziprecruiter.com",
        "neverssl.com", "httpforever.com", "roadmap.sh",
        "scrimba.com", "dev.to", "hashnode.com",
        "4kwallpapers.com", "freecodecamp.org",
        "codecademy.com", "pluralsight.com", "egghead.io",
        "frontendmasters.com", "udemy.com", "coursera.org",
        "edx.org", "khanacademy.org", "brilliant.org",
    }
    whitelist.update(WHITELIST_MANUAL)

    # Tranco list (if downloaded)
    whitelist_path = os.path.join(MODELS_DIR, "whitelist.txt")
    if os.path.exists(whitelist_path):
        with open(whitelist_path) as f:
            for line in f:
                domain = line.strip().lower()
                if domain:
                    whitelist.add(domain)
        print(f"[Whitelist] Loaded {len(whitelist):,} domains")
    else:
        print(f"[Whitelist] whitelist.txt not found — using manual list "
              f"({len(whitelist)} domains)")

    return whitelist

WHITELIST = _load_whitelist()


# ── URLClassifier (singleton) ─────────────────────────────────────────────────

class URLClassifier:
    _instance: Optional["URLClassifier"] = None
    _init_lock = threading.Lock()

    def __new__(cls):
        with cls._init_lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instance = instance
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.model    = joblib.load(os.path.join(MODELS_DIR, "rf_model_latest.joblib"))
        self.vec_char = joblib.load(os.path.join(MODELS_DIR, "vectorizer_char.joblib"))
        self.vec_word = joblib.load(os.path.join(MODELS_DIR, "vectorizer_word.joblib"))
        self.scaler   = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))

        with open(os.path.join(MODELS_DIR, "threshold.json")) as fh:
            self.threshold = float(json.load(fh).get("threshold", 0.45))

        self._explainer: Optional[object] = None
        self._explainer_lock = threading.Lock()

        if GSB_API_KEY:
            print("[GSB] Google Safe Browsing API loaded")
        else:
            print("[GSB] WARNING: No API key found — GSB layer disabled")

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _registered_domain(url: str) -> str:
        cleaned = re.sub(r"^https?://", "", url, flags=re.IGNORECASE)
        host    = cleaned.split("/")[0].split(":")[0].split("?")[0].lower()
        parts   = host.split(".")
        return ".".join(parts[-2:]) if len(parts) >= 2 else host

    def _feature_vector(self, url: str) -> np.ndarray:
        heuristic = np.array(
            extract_heuristic_features(url), dtype=np.float32
        ).reshape(1, -1)
        heuristic_scaled = self.scaler.transform(heuristic).flatten()
        # Use segment_url for NLP — matches training
        segmented  = segment_url(url)
        char_dense = self.vec_char.transform([segmented]).toarray().flatten()
        word_dense = self.vec_word.transform([segmented]).toarray().flatten()
        return np.concatenate([heuristic_scaled, char_dense, word_dense])

    def _classify(self, url: str) -> dict:
        try:
            fv    = self._feature_vector(url)
            proba = float(self.model.predict_proba(fv.reshape(1, -1))[0][1])

            raw_heuristic = extract_heuristic_features(url)
            https_val     = raw_heuristic[_FEAT_IDX.get("https_flag", -1)]

            # FIX 4: reduce confidence for clean HTTP-only sites
            if https_val == 0.0 and proba < 0.80:
                other_flags = [
                    "brand_mismatch", "sus_words", "brand_in_domain",
                    "risky_tld", "shortened", "ip_flag", "puny",
                    "leet_in_domain", "brand_hyphen_suspicious",
                    "susp_ext", "suspicious_port", "has_redirect",
                    "double_slash_in_path", "abnormal_subdomain",
                    "http_no_brand_no_age",
                ]
                other_scores = [
                    raw_heuristic[_FEAT_IDX.get(f, -1)]
                    for f in other_flags
                ]
                if all(s == 0.0 for s in other_scores):
                    proba = proba * 0.55

            label = "MALICIOUS" if proba >= self.threshold else "BENIGN"
            return {
                "prediction": label,
                "confidence": round(proba * 100, 2),
                "threshold" : round(self.threshold * 100, 2),
                "source"    : "model",
            }
        except Exception as exc:
            return {
                "prediction": "BENIGN",
                "confidence": 0.0,
                "threshold" : round(self.threshold * 100, 2),
                "source"    : "invalid",
                "error"     : str(exc),
            }

    # ── Public: prediction ────────────────────────────────────────────────────

    def predict_url(self, url: str) -> dict:
        if not url or not isinstance(url, str):
            return {
                "url": url, "prediction": "BENIGN",
                "confidence": 0.0,
                "threshold": round(self.threshold * 100, 2),
                "source": "invalid",
            }

        original_url = url
        resolved_url = None
        unshortened  = None

        # Layer 0: Whitelist
        if self._registered_domain(url) in WHITELIST:
            return {
                "url": original_url, "prediction": "BENIGN",
                "confidence": 100.0,
                "threshold": round(self.threshold * 100, 2),
                "source": "whitelist",
            }

        # Layer 1: Google Safe Browsing
        is_malicious, threat_type = check_google_safe_browsing(url)
        if is_malicious:
            return {
                "url": original_url, "prediction": "MALICIOUS",
                "confidence": 100.0,
                "threshold": round(self.threshold * 100, 2),
                "source": "google_safe_browsing",
                "threat_type": threat_type,
            }

        # Reverse DNS
        ip = _extract_ip(url)
        if ip:
            hostname = reverse_dns(ip)
            if hostname:
                resolved_url = url.replace(ip, hostname)
                if self._registered_domain(resolved_url) in WHITELIST:
                    return {
                        "url": original_url, "prediction": "BENIGN",
                        "confidence": 100.0,
                        "threshold": round(self.threshold * 100, 2),
                        "source": "whitelist",
                        "resolved_ip": hostname,
                    }
                url = resolved_url
        elif _is_shortener(url):
            final_url, was_redirected = unshorten_url(url)
            if was_redirected:
                unshortened = final_url
                if self._registered_domain(final_url) in WHITELIST:
                    return {
                        "url": original_url, "prediction": "BENIGN",
                        "confidence": 100.0,
                        "threshold": round(self.threshold * 100, 2),
                        "source": "whitelist",
                        "unshortened": final_url,
                    }
                url = final_url

        brand_name, real_domain = detect_brand(url)
        result        = self._classify(url)
        result["url"] = original_url

        if brand_name:
            result["brand_detected"] = brand_name
            result["real_domain"]    = real_domain
        if resolved_url:
            result["resolved_ip"]    = hostname
        if unshortened:
            result["unshortened"]    = unshortened

        return result

    def predict_batch(self, urls: list[str]) -> list[dict]:
        return [self.predict_url(u) for u in urls]

    # ── Public: LIME explanation ──────────────────────────────────────────────

    def explain_url(self, url: str, num_features: int = 30) -> dict:
        base = self.predict_url(url)

        # Handle non-model sources
        if base["source"] == "google_safe_browsing":
            threat = base.get("threat_type", "malicious content")
            return {
                **base,
                "explanation": [],
                "reasons": [
                    f"This URL was flagged by Google Safe Browsing as {threat}",
                    "Google's database of known malicious URLs identified this site",
                    "This site has been reported and verified as dangerous",
                ]
            }

        if base["source"] in ("whitelist", "invalid"):
            return {**base, "explanation": [], "reasons": []}

        brand_name    = base.get("brand_detected")
        real_domain   = base.get("real_domain")
        classify_url  = base.get("unshortened") or url
        raw_heuristic = extract_heuristic_features(classify_url)

        try:
            explainer = self._get_explainer()
            fv        = self._feature_vector(classify_url)

            exp = explainer.explain_instance(
                data_row     = fv,
                predict_fn   = self._lime_predict_fn,
                num_features = num_features,
                top_labels   = 1,
            )

            raw_list    = exp.as_list(label=1)
            explanation = []
            reasons     = []
            seen        = set()

            for condition_str, weight in raw_list:
                feat_name = _parse_lime_feature(condition_str)
                if not feat_name:
                    continue
                feat_idx = FEATURE_NAMES.index(feat_name) \
                           if feat_name in FEATURE_NAMES else -1
                feat_val = float(fv[feat_idx]) if feat_idx >= 0 else 0.0

                explanation.append({
                    "feature": feat_name,
                    "weight" : round(float(weight), 6),
                    "value"  : round(feat_val, 6),
                })

                if weight > 0:
                    # FIX 2: skip https_flag reason if site actually has HTTPS
                    if feat_name == "https_flag":
                        raw_https = raw_heuristic[_FEAT_IDX.get("https_flag", -1)]
                        if raw_https == 1.0:
                            continue

                    nl = feature_to_natural_language(
                        feat_name, weight, feat_val,
                        brand_name, real_domain
                    )
                    if nl and nl not in seen:
                        seen.add(nl)
                        reasons.append(nl)

            explanation.sort(key=lambda x: abs(x["weight"]), reverse=True)
            top_reasons = reasons[:3]

            # Fill with backup reasons if needed
            if len(top_reasons) < 3:
                backup = _build_backup_reasons(
                    raw_heuristic, brand_name, real_domain,
                    set(top_reasons), 3 - len(top_reasons)
                )
                top_reasons.extend(backup)

            return {**base, "explanation": explanation, "reasons": top_reasons}

        except Exception as exc:
            backup = _build_backup_reasons(
                raw_heuristic, brand_name, real_domain, set(), 3
            )
            return {**base, "explanation": [], "reasons": backup,
                    "explain_error": str(exc)}

    # ── LIME internals ────────────────────────────────────────────────────────

    def _get_explainer(self):
        if self._explainer is not None:
            return self._explainer
        with self._explainer_lock:
            if self._explainer is not None:
                return self._explainer
            try:
                from lime.lime_tabular import LimeTabularExplainer
            except ImportError as exc:
                raise ImportError("pip install lime") from exc

            bg = self._load_background()
            cat_indices = [
                FEATURE_NAMES.index(n)
                for n in _CATEGORICAL_FEATURE_NAMES
                if n in FEATURE_NAMES
            ]
            self._explainer = LimeTabularExplainer(
                training_data        = bg,
                feature_names        = FEATURE_NAMES,
                class_names          = ["BENIGN", "MALICIOUS"],
                categorical_features = cat_indices,
                mode                 = "classification",
                discretize_continuous= True,
                random_state         = 42,
            )
            return self._explainer

    def _load_background(self, n_samples: int = 500) -> np.ndarray:
        bg_path = os.path.join(MODELS_DIR, "lime_background.npz")
        if os.path.exists(bg_path):
            return np.load(bg_path)["X"]

        warnings.warn(
            f"{bg_path} not found — building LIME background.",
            RuntimeWarning, stacklevel=3,
        )
        import csv, random
        csv_path = os.path.join(_ROOT, "data", "splits", "train_urls.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Cannot find {bg_path} or {csv_path}.")

        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader   = csv.DictReader(fh)
            all_urls = [row["url"] for row in reader if row.get("url")]

        sample = random.sample(all_urls, min(n_samples, len(all_urls)))
        rows   = []
        for u in sample:
            try:
                rows.append(self._feature_vector(u))
            except Exception:
                pass

        bg = np.array(rows, dtype=np.float32)
        np.savez_compressed(bg_path, X=bg)
        return bg

    def _lime_predict_fn(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)


# ── Helper ────────────────────────────────────────────────────────────────────

def _parse_lime_feature(condition_str: str) -> str:
    for name in sorted(FEATURE_NAMES, key=len, reverse=True):
        if condition_str.startswith(name):
            return name
    first = re.split(r"[\s<>=!]", condition_str)[0]
    if first and (first[0].isdigit() or first[0] in "-+."):
        return ""
    return first


# ── Module-level singleton ────────────────────────────────────────────────────
classifier = URLClassifier()