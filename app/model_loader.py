"""
model_loader.py
---------------
Loads model artefacts and exposes three public methods:

  predict_url(url)                   -> dict
  predict_batch(urls)                -> list[dict]
  explain_url(url, num_features=30)  -> dict

Prediction pipeline:
  Layer 0: Smart Whitelist check (Tranco top 50k + manual list)
  Layer 1: Google Safe Browsing API (known threats)
  Layer 2: ML Model (Random Forest 559 features)
  Layer 3: Domain Age post-prediction adjustment (WHOIS, cached)

Fixes in this version:
  - FIX 11: Smart whitelist — replaces simple registered_domain check
      Problem 1: Legit subdomains (mail.google.com, docs.python.org)
                 were going to ML because only "google.com" is whitelisted
                 not "mail.google.com". Now trusted subdomains pass through.
      Problem 2: Attacker uses google.com.evil.tk — old code extracted
                 evil.tk (not whitelisted, correct) but mail.google.com
                 extracted google.com (whitelisted, bypassed ML — wrong).
                 New code handles both correctly via _is_safe_whitelist_url().
  - FIX 9: Early return for leet domains — skips domain age adjustment
      so 0.85 override is not crushed to 0.255 by age multiplier.

All previous fixes retained:
  - FIX 1:  https_flag backup threshold -0.1
  - FIX 2:  explain_url skips https_flag when site has HTTPS
  - FIX 3:  feature_to_natural_language validates flag values
  - FIX 4:  _classify reduces confidence for clean HTTP sites
  - FIX 5:  brand reason skipped when no brand detected
  - FIX 6:  Google Safe Browsing API layer
  - FIX 7:  Domain age WHOIS post-prediction adjustment (cached)
  - FIX 8:  Dynamic leet brand detection (no hardcoded map)
  - FIX 10: Updated for 559 features
"""

import json
import os
import re
import sys
import socket
import threading
import warnings
from typing import Optional
from datetime import datetime, timezone

import joblib
import numpy as np
import requests
import tldextract as _tldextract

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
    BRANDS,
    REAL_BRAND_DOMAINS,
    LEET_DECODE_MAP,
    decode_leet,
)

# ── Path constants ────────────────────────────────────────────────────────────
MODELS_DIR            = os.path.join(_ROOT, "models")
DATA_DIR              = os.path.join(_ROOT, "data")
DOMAIN_AGE_CACHE_PATH = os.path.join(MODELS_DIR, "domain_age_cache.json")

# ── Google Safe Browsing ──────────────────────────────────────────────────────
GSB_API_KEY = os.environ.get("GOOGLE_SAFE_BROWSING_API_KEY", "")
GSB_URL     = "https://safebrowsing.googleapis.com/v4/threatMatches:find"

# ── Feature names ─────────────────────────────────────────────────────────────
HEURISTIC_FEATURES: list[str] = HEURISTIC_FEATURE_NAMES   # 57

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
    "leet_brand_score",
]

_PRIVATE_IP_RE = re.compile(
    r"^(10\.|172\.(1[6-9]|2\d|3[01])\.|192\.168\.|127\.|0\.0\.0\.0|::1)"
)

# ── TLDExtract instance ───────────────────────────────────────────────────────
_EXTRACTOR = _tldextract.TLDExtract(cache_dir=None, suffix_list_urls=None)

# ── Brand map ─────────────────────────────────────────────────────────────────
BRAND_MAP = {
    "paypal"       : ("PayPal",          "paypal.com"),
    "amazon"       : ("Amazon",          "amazon.com"),
    "microsoft"    : ("Microsoft",       "microsoft.com"),
    "apple"        : ("Apple",           "apple.com"),
    "google"       : ("Google",          "google.com"),
    "facebook"     : ("Facebook",        "facebook.com"),
    "netflix"      : ("Netflix",         "netflix.com"),
    "bankofamerica": ("Bank of America", "bankofamerica.com"),
    "wellsfargo"   : ("Wells Fargo",     "wellsfargo.com"),
    "whatsapp"     : ("WhatsApp",        "whatsapp.com"),
    "instagram"    : ("Instagram",       "instagram.com"),
    "twitter"      : ("Twitter",         "twitter.com"),
    "linkedin"     : ("LinkedIn",        "linkedin.com"),
    "ebay"         : ("eBay",            "ebay.com"),
    "visa"         : ("Visa",            "visa.com"),
    "mastercard"   : ("Mastercard",      "mastercard.com"),
    "chase"        : ("Chase Bank",      "chase.com"),
    "citi"         : ("Citibank",        "citibank.com"),
    "dropbox"      : ("Dropbox",         "dropbox.com"),
    "steam"        : ("Steam",           "steampowered.com"),
    "dhl"          : ("DHL",             "dhl.com"),
    "fedex"        : ("FedEx",           "fedex.com"),
    "ups"          : ("UPS",             "ups.com"),
}

# ── FIX 11: Trusted subdomain prefixes ───────────────────────────────────────
# Subdomains on these prefixes of a whitelisted domain are considered safe.
# If a subdomain is NOT in this set it goes to the ML model instead.
TRUSTED_SUBDOMAIN_PREFIXES: set[str] = {
    # Common functional subdomains
    "www", "mail", "email", "webmail",
    "docs", "doc", "help", "support", "status",
    "api", "apis", "dev", "developer", "developers",
    "app", "apps", "portal", "dashboard", "console",
    "login", "signin", "auth", "accounts", "account",
    "secure", "security", "safe",
    "shop", "store", "pay", "checkout", "billing",
    "blog", "news", "press", "media",
    "cdn", "static", "assets", "img", "images",
    "video", "videos", "stream",
    "search", "maps", "translate",
    "careers", "jobs", "about", "corporate",
    # Regional / mobile
    "m", "mobile", "wap",
    "en", "us", "uk", "au", "ca", "in", "de", "fr",
    # Cloud / infra
    "cloud", "aws", "azure",
    # Numeric / versioned (e.g. v2.api.example.com)
    "v1", "v2", "v3",
}


# ════════════════════════════════════════════════════════════════
# WHITELIST — LOAD
# ════════════════════════════════════════════════════════════════

def _load_whitelist() -> set:
    whitelist = set()

    WHITELIST_MANUAL = {
        # Major search and tech
        "google.com", "gmail.com", "youtube.com", "googleapis.com",
        "google.co.uk", "google.com.au", "google.ca", "google.in",
        "github.com", "gitlab.com", "stackoverflow.com",
        "wikipedia.org", "wikimedia.org",
        # Microsoft
        "microsoft.com", "microsoftonline.com", "live.com",
        "outlook.com", "office.com", "azure.com", "bing.com",
        # Apple
        "apple.com", "icloud.com",
        # Amazon
        "amazon.com", "amazon.co.uk", "amazon.de", "amazon.fr",
        "amazonaws.com",
        # Social
        "facebook.com", "instagram.com", "twitter.com", "x.com",
        "linkedin.com", "reddit.com", "pinterest.com",
        "whatsapp.com", "telegram.org", "discord.com",
        # Finance
        "paypal.com", "bankofamerica.com", "chase.com",
        "wellsfargo.com", "citibank.com", "visa.com",
        "mastercard.com", "stripe.com",
        # Entertainment
        "netflix.com", "spotify.com", "twitch.tv",
        "slack.com", "zoom.us",
        # Shopping
        "ebay.com", "etsy.com", "shopify.com",
        # Tools
        "dropbox.com", "adobe.com", "salesforce.com",
        "wordpress.com", "medium.com",
        # Developer sites
        "roadmap.sh", "dev.to", "freecodecamp.org",
        "scrimba.com", "codecademy.com", "hashnode.com",
        "hackerearth.com", "hackerrank.com", "leetcode.com",
        "codechef.com", "codeforces.com", "kaggle.com",
        "replit.com", "codepen.io", "codesandbox.io",
        "exercism.org", "theodinproject.com",
        # Programming language / framework sites
        "python.org", "nodejs.org", "rust-lang.org", "golang.org",
        "reactjs.org", "vuejs.org", "angular.io", "svelte.dev",
        "nextjs.org", "djangoproject.com", "flask.palletsprojects.com",
        "npmjs.com", "pypi.org", "crates.io",
        # Job boards
        "dailyremote.com", "jobspresso.co", "remoteok.com",
        "weworkremotely.com", "flexjobs.com", "remote.co",
        "wellfound.com", "monster.com", "ziprecruiter.com",
        "indeed.com", "glassdoor.com",
        # Media
        "4kwallpapers.com", "unsplash.com", "pexels.com",
        # Other legit
        "neverssl.com", "httpforever.com",
        # Cloud / hosting
        "vercel.com", "netlify.com", "heroku.com",
        "digitalocean.com", "cloudflare.com",
        # News
        "bbc.com", "bbc.co.uk", "cnn.com", "reuters.com",
        "nytimes.com", "theguardian.com", "bloomberg.com",
    }
    whitelist.update(WHITELIST_MANUAL)

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


# ════════════════════════════════════════════════════════════════
# FIX 11: SMART WHITELIST CHECK
# ════════════════════════════════════════════════════════════════

def _is_safe_whitelist_url(url: str) -> bool:
    """
    FIX 11: Smart whitelist check that fixes two problems:

    PROBLEM 1 — Legit subdomains flagged as malicious:
      https://mail.google.com     → google.com in whitelist
                                  → subdomain "mail" is trusted → BENIGN ✓
      https://docs.python.org     → python.org in whitelist
                                  → subdomain "docs" is trusted → BENIGN ✓
      https://api.github.com      → github.com in whitelist
                                  → subdomain "api" is trusted  → BENIGN ✓

    PROBLEM 2 — Attacker spoofs whitelisted domain in subdomain:
      http://google.com.evil.tk   → registered = evil.tk
                                  → NOT in whitelist → goes to ML ✓
      http://paypal.verify-now.net→ registered = verify-now.net
                                  → NOT in whitelist → goes to ML ✓

    PROBLEM 3 — Suspicious subdomain on whitelisted domain:
      http://randomhex8f3a.google.com → google.com whitelisted
                                      → subdomain "randomhex8f3a" not trusted
                                      → goes to ML ✓

    Logic:
      1. Extract registered domain using tldextract
      2. If NOT in whitelist → return False (goes to ML)
      3. If no subdomain → return True (clean root domain)
      4. If subdomain contains brand name → spoofing attempt → False
      5. If all subdomain parts are trusted prefixes → True
      6. Otherwise → False (goes to ML to decide)
    """
    try:
        cleaned = re.sub(r"^https?://", "", url, flags=re.IGNORECASE)
        host    = cleaned.split("/")[0].split(":")[0].lower()
        ext     = _EXTRACTOR(host)

        registered = ext.registered_domain or ""
        subdomain  = (ext.subdomain or "").strip()

        # Step 1: registered domain must be whitelisted
        if registered not in WHITELIST:
            return False

        # Step 2: no subdomain at all → clean root domain → safe
        if not subdomain:
            return True

        # Step 3: remove leading "www" — www.google.com is always safe
        sub_parts = [p for p in subdomain.split(".") if p and p != "www"]
        if not sub_parts:
            return True

        # Step 4: check if any subdomain part looks like a brand domain
        # e.g. subdomain = "paypal.com" in "paypal.com.evil.tk"
        # This is the spoofing attack — brand name used as subdomain
        for part in sub_parts:
            # Check if part itself is a whitelisted domain label
            # e.g. "google" in "google.evil.tk" → spoofing
            for wl_domain in WHITELIST:
                wl_label = wl_domain.split(".")[0]  # e.g. "google" from "google.com"
                if part == wl_label and registered != wl_domain:
                    # Brand label used as subdomain on different domain → spoof
                    return False
            # Also check brand map keywords directly
            for brand_key in BRAND_MAP:
                if part == brand_key and registered not in REAL_BRAND_DOMAINS:
                    return False

        # Step 5: all remaining subdomain parts must be trusted prefixes
        all_trusted = all(p in TRUSTED_SUBDOMAIN_PREFIXES for p in sub_parts)
        if all_trusted:
            return True

        # Step 6: unknown subdomain → let ML decide
        # e.g. "randomhex8f3a.google.com" or "xyz123.github.com"
        return False

    except Exception:
        return False


# ════════════════════════════════════════════════════════════════
# NATURAL LANGUAGE TEMPLATES
# ════════════════════════════════════════════════════════════════

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
    "leet_brand_score"       : {
        "mal": "The domain uses digit substitutions to impersonate {brand} (e.g. g00gle, paypa1)",
        "ben": "No leet brand impersonation detected",
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
    ("brand_in_domain",         0.5, "This site is pretending to be {brand} — the real website is {real_domain}"),
    ("brand_hyphen_suspicious", 0.5, "The domain uses a fake {brand} pattern (e.g. {brand}-security.com)"),
    ("sus_words",               0.5, "The URL contains phishing keywords such as 'security', 'alert', or 'verify'"),
    ("brand_mismatch",          0.5, "{brand} name appears in the URL but this is not the real {brand} website"),
    ("risky_tld",               0.5, "This site uses a high-risk domain ending commonly used for phishing"),
    ("leet_in_domain",          0.5, "The domain disguises a brand name using look-alike characters (e.g. amaz0n)"),
    ("leet_brand_score",        0.5, "The domain uses digit substitutions to impersonate {brand} (e.g. g00gle, paypa1)"),
    ("ip_flag",                 0.5, "The site uses a raw IP address instead of a proper domain name"),
    ("shortened",               0.5, "This is a shortened URL hiding the real destination"),
    ("puny",                    0.5, "The domain uses international character encoding to disguise its identity"),
    ("susp_ext",                0.5, "The URL points to a suspicious file type (e.g. .exe, .zip, .scr)"),
    ("has_redirect",            0.5, "The URL contains a redirect parameter — often used in phishing attacks"),
    ("double_slash_in_path",    0.5, "The URL path contains double slashes — a common obfuscation technique"),
    ("abnormal_subdomain",      0.5, "The subdomain contains suspicious patterns such as random digits"),
    ("http_no_brand_no_age",    0.5, "This site uses HTTP without HTTPS and shows other suspicious signals"),
    ("https_flag",             -0.1, "This site does not use HTTPS — your connection may not be secure"),
    ("num_hyphens",             2.0, "The domain contains excessive hyphens which is uncommon in legitimate sites"),
    ("path_depth",              3.0, "The URL has an unusually deep path — often used to mimic legitimate sites"),
]


# ════════════════════════════════════════════════════════════════
# DOMAIN AGE — WHOIS WITH PERSISTENT CACHE
# ════════════════════════════════════════════════════════════════

_domain_age_cache: dict = {}
_cache_lock = threading.Lock()


def _load_domain_age_cache():
    global _domain_age_cache
    if os.path.exists(DOMAIN_AGE_CACHE_PATH):
        try:
            with open(DOMAIN_AGE_CACHE_PATH, "r") as f:
                _domain_age_cache = json.load(f)
            print(f"[DomainAge] Cache loaded: {len(_domain_age_cache):,} domains")
        except Exception:
            _domain_age_cache = {}
    else:
        print("[DomainAge] No cache found — will build on first requests")


def _save_domain_age_cache():
    try:
        with open(DOMAIN_AGE_CACHE_PATH, "w") as f:
            json.dump(_domain_age_cache, f)
    except Exception:
        pass


def get_domain_age_days(domain: str) -> int:
    """
    Returns domain age in days via WHOIS.
    Uses persistent disk cache — each domain looked up only ONCE ever.
    Returns -1 if unknown.
    """
    if not domain:
        return -1
    if domain in _domain_age_cache:
        return _domain_age_cache[domain]
    try:
        import whois
        w             = whois.whois(domain)
        creation_date = w.creation_date
        if creation_date is None:
            age = -1
        else:
            if isinstance(creation_date, list):
                creation_date = creation_date[0]
            if hasattr(creation_date, "tzinfo") and creation_date.tzinfo is None:
                creation_date = creation_date.replace(tzinfo=timezone.utc)
            age = max(0, (datetime.now(timezone.utc) - creation_date).days)
    except Exception:
        age = -1
    with _cache_lock:
        _domain_age_cache[domain] = age
        _save_domain_age_cache()
    return age


def adjust_confidence_by_domain_age(proba: float, domain: str) -> tuple[float, str]:
    """
    Adjusts ML confidence score based on domain age.

    Multipliers:
      Unknown      → no change
      < 30 days    → × 1.4
      30-180 days  → × 1.15
      180-365 days → × 1.0
      1-2 years    → × 0.7
      2-5 years    → × 0.5
      > 5 years    → × 0.3
    """
    age = get_domain_age_days(domain)
    if age < 0:
        return proba, "unknown"
    if age < 30:
        multiplier, age_label = 1.4,  f"very_new ({age}d)"
    elif age < 180:
        multiplier, age_label = 1.15, f"new ({age}d)"
    elif age < 365:
        multiplier, age_label = 1.0,  f"recent ({age}d)"
    elif age < 730:
        multiplier, age_label = 0.7,  f"established ({age}d)"
    elif age < 1825:
        multiplier, age_label = 0.5,  f"old ({age}d)"
    else:
        multiplier, age_label = 0.3,  f"very_old ({age}d)"
    return min(1.0, proba * multiplier), age_label


# ════════════════════════════════════════════════════════════════
# GOOGLE SAFE BROWSING
# ════════════════════════════════════════════════════════════════

def check_google_safe_browsing(url: str) -> tuple[bool, str]:
    if not GSB_API_KEY:
        return False, ""
    try:
        payload = {
            "client": {"clientId": "malicious-url-detector", "clientVersion": "1.0.0"},
            "threatInfo": {
                "threatTypes": [
                    "MALWARE", "SOCIAL_ENGINEERING",
                    "UNWANTED_SOFTWARE", "POTENTIALLY_HARMFUL_APPLICATION",
                ],
                "platformTypes":    ["ANY_PLATFORM"],
                "threatEntryTypes": ["URL"],
                "threatEntries":    [{"url": url}],
            },
        }
        response = requests.post(f"{GSB_URL}?key={GSB_API_KEY}", json=payload, timeout=5)
        data     = response.json()
        if data.get("matches"):
            return True, data["matches"][0].get("threatType", "MALICIOUS")
        return False, ""
    except Exception:
        return False, ""


# ════════════════════════════════════════════════════════════════
# BRAND DETECTION  —  FIX 8 REWRITE (dynamic leet decode)
# ════════════════════════════════════════════════════════════════

def detect_brand(url: str) -> tuple[Optional[str], Optional[str]]:
    """
    Detects brand impersonation in URL.

    Order:
      1. Plain-text brand match
      2. Dynamic leet decode → brand match (catches ANY leet variant)
    """
    url_lower = url.lower()

    # Standard detection
    for keyword, (display_name, real_domain) in BRAND_MAP.items():
        if keyword in url_lower:
            host = re.sub(r"^https?://", "", url_lower).split("/")[0]
            reg  = ".".join(host.split(".")[-2:]) if "." in host else host
            if reg != real_domain and reg not in REAL_BRAND_DOMAINS:
                return display_name, real_domain

    # Dynamic leet decode
    try:
        host  = re.sub(r"^https?://", "", url_lower).split("/")[0].split(":")[0]
        parts = host.split(".")
        for part in parts[:-1]:   # exclude TLD
            if not part:
                continue
            for candidate in decode_leet(part):
                if candidate == part:
                    continue
                for brand_key, (display_name, real_domain) in BRAND_MAP.items():
                    if brand_key in candidate:
                        reg = ".".join(parts[-2:]) if len(parts) >= 2 else host
                        if reg not in REAL_BRAND_DOMAINS:
                            return display_name, real_domain
    except Exception:
        pass

    return None, None


# ════════════════════════════════════════════════════════════════
# NATURAL LANGUAGE EXPLANATION HELPERS
# ════════════════════════════════════════════════════════════════

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
    if brand_name is None and weight > 0 and any(
        x in feature for x in ["brand", "visual_brand", "leet_brand"]
    ):
        return None

    # FIX 3: never show malicious reason when flag is actually clean
    _flag_sanity = {
        "https_flag"             : lambda v: v != 1.0,
        "ip_flag"                : lambda v: v != 0.0,
        "risky_tld"              : lambda v: v != 0.0,
        "shortened"              : lambda v: v != 0.0,
        "brand_in_domain"        : lambda v: v != 0.0,
        "leet_in_domain"         : lambda v: v != 0.0,
        "leet_brand_score"       : lambda v: v != 0.0,
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
    sentence = template["mal" if weight > 0 else "ben"]
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
    bn, rd  = brand_name or "a known brand", real_domain or "the official website"
    reasons = []
    for feat_name, threshold, template in _BACKUP_CHECKS:
        if len(reasons) >= needed:
            break
        idx = _FEAT_IDX.get(feat_name, -1)
        if idx < 0:
            continue
        val       = heuristic[idx]
        triggered = val < abs(threshold) if threshold < 0 else val >= threshold
        if not triggered:
            continue
        if brand_name is None and any(
            x in feat_name for x in ["brand", "leet_brand", "visual"]
        ):
            continue
        sentence = template.replace("{brand}", bn).replace("{real_domain}", rd)
        if sentence not in existing:
            existing.add(sentence)
            reasons.append(sentence)
    return reasons


# ════════════════════════════════════════════════════════════════
# NETWORK HELPERS
# ════════════════════════════════════════════════════════════════

def reverse_dns(ip: str, timeout: int = 3) -> Optional[str]:
    if _PRIVATE_IP_RE.match(ip):
        return None
    try:
        socket.setdefaulttimeout(timeout)
        return socket.gethostbyaddr(ip)[0].lower().rstrip(".")
    except (socket.herror, socket.gaierror, OSError):
        return None


def unshorten_url(url: str, timeout: int = 5) -> tuple[str, bool]:
    try:
        resp = requests.head(
            url, allow_redirects=True, timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                     "AppleWebKit/537.36 Chrome/120.0 Safari/537.36"},
        )
        final = resp.url
        return final, final.rstrip("/") != url.rstrip("/")
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
        if len(parts) == 4 and all(p.isdigit() and 0 <= int(p) <= 255 for p in parts):
            return host
        if ":" in host:
            return host
        return None
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════
# URL CLASSIFIER SINGLETON
# ════════════════════════════════════════════════════════════════

class URLClassifier:
    _instance:   Optional["URLClassifier"] = None
    _init_lock = threading.Lock()

    def __new__(cls):
        with cls._init_lock:
            if cls._instance is None:
                inst = super().__new__(cls)
                inst._initialized = False
                cls._instance     = inst
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

        self._explainer      = None
        self._explainer_lock = threading.Lock()

        _load_domain_age_cache()

        if GSB_API_KEY:
            print("[GSB] Google Safe Browsing API loaded")
        else:
            print("[GSB] WARNING: No API key — GSB layer disabled")

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _registered_domain(url: str) -> str:
        cleaned = re.sub(r"^https?://", "", url, flags=re.IGNORECASE)
        host    = cleaned.split("/")[0].split(":")[0].split("?")[0].lower()
        if host.startswith("www."):
            host = host[4:]
        parts = host.split(".")
        return ".".join(parts[-2:]) if len(parts) >= 2 else host

    def _feature_vector(self, url: str) -> np.ndarray:
        h  = np.array(extract_heuristic_features(url), dtype=np.float32).reshape(1, -1)
        hs = self.scaler.transform(h).flatten()
        s  = segment_url(url)
        c  = self.vec_char.transform([s]).toarray().flatten()
        w  = self.vec_word.transform([s]).toarray().flatten()
        return np.concatenate([hs, c, w])

    # ── _classify ─────────────────────────────────────────────────────────────

    def _classify(self, url: str) -> dict:
        try:
            fv    = self._feature_vector(url)
            proba = float(self.model.predict_proba(fv.reshape(1, -1))[0][1])

            raw       = extract_heuristic_features(url)
            https_val = raw[_FEAT_IDX.get("https_flag",      -1)]
            leet_dom  = raw[_FEAT_IDX.get("leet_in_domain",  -1)]
            leet_br   = raw[_FEAT_IDX.get("leet_brand_score",-1)]

            # FIX 9: leet override — EARLY RETURN skips domain age
            # Without early return: 0.85 × 0.3 (old domain) = 0.255 → BENIGN (bug)
            # With early return: 0.85 → MALICIOUS always ✓
            leet_detected = (leet_dom == 1.0) or (leet_br == 1.0)
            if leet_detected and https_val == 0.0:
                proba = max(proba, 0.85)
                label = "MALICIOUS" if proba >= self.threshold else "BENIGN"
                return {
                    "prediction": label,
                    "confidence": round(proba * 100, 2),
                    "threshold" : round(self.threshold * 100, 2),
                    "source"    : "model",
                    "domain_age": "skipped_leet",
                }

            # FIX 4: reduce confidence for clean HTTP sites
            if https_val == 0.0 and proba < 0.80:
                other_flags = [
                    "brand_mismatch","sus_words","brand_in_domain",
                    "risky_tld","shortened","ip_flag","puny",
                    "leet_in_domain","leet_brand_score",
                    "brand_hyphen_suspicious","susp_ext",
                    "suspicious_port","has_redirect",
                    "double_slash_in_path","abnormal_subdomain",
                    "http_no_brand_no_age",
                ]
                if all(raw[_FEAT_IDX.get(f, -1)] == 0.0 for f in other_flags):
                    proba *= 0.55

            # FIX 7: domain age adjustment (non-leet URLs only)
            domain = self._registered_domain(url)
            proba, age_note = adjust_confidence_by_domain_age(proba, domain)

            label = "MALICIOUS" if proba >= self.threshold else "BENIGN"
            return {
                "prediction": label,
                "confidence": round(proba * 100, 2),
                "threshold" : round(self.threshold * 100, 2),
                "source"    : "model",
                "domain_age": age_note,
            }
        except Exception as exc:
            return {
                "prediction": "BENIGN",
                "confidence": 0.0,
                "threshold" : round(self.threshold * 100, 2),
                "source"    : "invalid",
                "error"     : str(exc),
            }

    # ── predict_url ───────────────────────────────────────────────────────────

    def predict_url(self, url: str) -> dict:
        if not url or not isinstance(url, str):
            return {
                "url": url, "prediction": "BENIGN",
                "confidence": 0.0,
                "threshold" : round(self.threshold * 100, 2),
                "source"    : "invalid",
            }

        original_url = url
        resolved_url = None
        unshortened  = None

        # ── Layer 0: FIX 11 Smart Whitelist ──────────────────────────────────
        # Old: self._registered_domain(url) in WHITELIST
        #   → missed legit subdomains (mail.google.com)
        # New: _is_safe_whitelist_url() handles subdomains correctly
        if _is_safe_whitelist_url(url):
            return {
                "url"       : original_url,
                "prediction": "BENIGN",
                "confidence": 100.0,
                "threshold" : round(self.threshold * 100, 2),
                "source"    : "whitelist",
            }

        # ── Layer 1: Google Safe Browsing ─────────────────────────────────────
        is_mal, threat_type = check_google_safe_browsing(url)
        if is_mal:
            return {
                "url"        : original_url,
                "prediction" : "MALICIOUS",
                "confidence" : 100.0,
                "threshold"  : round(self.threshold * 100, 2),
                "source"     : "google_safe_browsing",
                "threat_type": threat_type,
            }

        # Reverse DNS for IP URLs
        ip = _extract_ip(url)
        if ip:
            hostname = reverse_dns(ip)
            if hostname:
                resolved_url = url.replace(ip, hostname)
                if _is_safe_whitelist_url(resolved_url):
                    return {
                        "url"        : original_url,
                        "prediction" : "BENIGN",
                        "confidence" : 100.0,
                        "threshold"  : round(self.threshold * 100, 2),
                        "source"     : "whitelist",
                        "resolved_ip": hostname,
                    }
                url = resolved_url
        elif _is_shortener(url):
            final_url, was_redirected = unshorten_url(url)
            if was_redirected:
                unshortened = final_url
                if _is_safe_whitelist_url(final_url):
                    return {
                        "url"        : original_url,
                        "prediction" : "BENIGN",
                        "confidence" : 100.0,
                        "threshold"  : round(self.threshold * 100, 2),
                        "source"     : "whitelist",
                        "unshortened": final_url,
                    }
                url = final_url

        # ── Layer 2: ML Model ─────────────────────────────────────────────────
        brand_name, real_domain = detect_brand(url)
        result        = self._classify(url)
        result["url"] = original_url

        if brand_name:
            result["brand_detected"] = brand_name
            result["real_domain"]    = real_domain
        if resolved_url and ip:
            result["resolved_ip"] = hostname
        if unshortened:
            result["unshortened"] = unshortened

        return result

    def predict_batch(self, urls: list[str]) -> list[dict]:
        return [self.predict_url(u) for u in urls]

    # ── explain_url ───────────────────────────────────────────────────────────

    def explain_url(self, url: str, num_features: int = 30) -> dict:
        base = self.predict_url(url)

        if base["source"] == "google_safe_browsing":
            threat = base.get("threat_type", "malicious content")
            return {
                **base,
                "explanation": [],
                "reasons": [
                    f"This URL was flagged by Google Safe Browsing as {threat}",
                    "Google's database of known malicious URLs identified this site",
                    "This site has been reported and verified as dangerous",
                ],
            }

        if base["source"] in ("whitelist", "invalid"):
            return {**base, "explanation": [], "reasons": []}

        brand_name    = base.get("brand_detected")
        real_domain   = base.get("real_domain")
        classify_url  = base.get("unshortened") or url
        raw_heuristic = extract_heuristic_features(classify_url)
        domain        = self._registered_domain(classify_url)
        age_days      = get_domain_age_days(domain)

        try:
            explainer = self._get_explainer()
            fv        = self._feature_vector(classify_url)

            exp = explainer.explain_instance(
                data_row   = fv,
                predict_fn = self._lime_predict_fn,
                num_features = num_features,
                top_labels   = 1,
            )

            explanation, reasons, seen = [], [], set()

            for condition_str, weight in exp.as_list(label=1):
                feat_name = _parse_lime_feature(condition_str)
                if not feat_name:
                    continue
                feat_idx = FEATURE_NAMES.index(feat_name) if feat_name in FEATURE_NAMES else -1
                feat_val = float(fv[feat_idx]) if feat_idx >= 0 else 0.0

                explanation.append({
                    "feature": feat_name,
                    "weight" : round(float(weight), 6),
                    "value"  : round(feat_val, 6),
                })

                if weight > 0:
                    # FIX 2: skip https_flag if site actually has HTTPS
                    if feat_name == "https_flag":
                        if raw_heuristic[_FEAT_IDX.get("https_flag", -1)] == 1.0:
                            continue
                    nl = feature_to_natural_language(
                        feat_name, weight, feat_val, brand_name, real_domain
                    )
                    if nl and nl not in seen:
                        seen.add(nl)
                        reasons.append(nl)

            explanation.sort(key=lambda x: abs(x["weight"]), reverse=True)
            top_reasons = reasons[:3]

            if len(top_reasons) < 3:
                top_reasons.extend(
                    _build_backup_reasons(
                        raw_heuristic, brand_name, real_domain,
                        set(top_reasons), 3 - len(top_reasons),
                    )
                )

            # Add domain age reason if very new domain
            if 0 <= age_days < 30:
                age_reason = (
                    f"This domain was registered only {age_days} days ago — "
                    "newly registered domains are a strong phishing indicator"
                )
                if age_reason not in top_reasons:
                    if len(top_reasons) >= 3:
                        top_reasons[2] = age_reason
                    else:
                        top_reasons.append(age_reason)

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

            bg          = self._load_background()
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

        warnings.warn(f"{bg_path} not found — building LIME background.",
                      RuntimeWarning, stacklevel=3)
        import csv, random
        csv_path = os.path.join(_ROOT, "data", "splits", "train_urls.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Cannot find {bg_path} or {csv_path}.")

        with open(csv_path, newline="", encoding="utf-8") as fh:
            all_urls = [row["url"] for row in csv.DictReader(fh) if row.get("url")]

        rows = []
        for u in random.sample(all_urls, min(n_samples, len(all_urls))):
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