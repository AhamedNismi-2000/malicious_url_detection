"""
model_loader.py  (v2)
---------------------
Loads model artefacts and exposes three public methods:

  predict_url(url)                   -> dict
  predict_batch(urls)                -> list[dict]
  explain_url(url, num_features=30)  -> dict

Run `python app/model_loader.py` (no arguments) to execute the built-in self-test.

Prediction pipeline (v2)
  Layer 1: Google Safe Browsing (known threats). Runs for EVERY url, whitelisted or not.
  Layer 2: Trust assessment. The whitelist is now a TRUST SIGNAL, not a bypass:
           a trusted host gets a lower score, but it is still analysed. Trust is revoked
           when there is evidence of badness (see assess_trust).
  Layer 3: Rule layer (typosquat.py): lookalike / combosquat / brand-in-subdomain rules.
  Layer 4: ML model (Random Forest, features listed in models/feature_config.json).
  Layer 5: Overrides, clean-site dampening, trust multiplier, domain-age adjustment.

How a trusted URL can still be MALICIOUS
  - Google Safe Browsing says so,
  - a rule hit or hard evidence appears (e.g. user@host trick, lookalike of another brand),
  - the redirect target in the query string is malicious,
  - the raw model probability is above TRUST_FLIP_PROBA (very high).
  Trust only ever LOWERS the score; it never raises it and never skips the checks above.

Response format is unchanged (url, prediction, confidence, threshold, source, ...).
New optional keys: whitelisted (bool), trust ("strong"/"weak"), risk_score, rules, evidence.

Kept from the previous version: FIX 4 (clean-site dampening), FIX 6 (GSB), FIX 7 (domain
age, cached), FIX 9 (leet), FIX 12 (suspicious words in the domain label), FIX 13 (brand +
no HTTPS), FIX 14 (raw IP + no HTTPS). What changed in them (marked "# v2:"):
  - "no HTTPS" now comes from the REAL scheme of the URL. The model features are
    scheme-independent (feature_extraction v2), so https_flag is always 0 there.
  - FIX 9/13/14 are not applied to strongly trusted hosts.
  - Domain age is skipped for strongly trusted hosts (saves WHOIS calls).
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
from urllib.parse import urlsplit, parse_qsl, unquote

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
import typosquat as ts                                            # v2:

# ── Path constants ────────────────────────────────────────────────────────────
MODELS_DIR            = os.path.join(_ROOT, "models")
DATA_DIR              = os.path.join(_ROOT, "data")
DOMAIN_AGE_CACHE_PATH = os.path.join(MODELS_DIR, "domain_age_cache.json")
FEATURE_CONFIG_PATH   = os.path.join(MODELS_DIR, "feature_config.json")   # v2:

# ── Google Safe Browsing ──────────────────────────────────────────────────────
GSB_API_KEY = os.environ.get("GOOGLE_SAFE_BROWSING_API_KEY", "")
GSB_URL     = "https://safebrowsing.googleapis.com/v4/threatMatches:find"

# ═════════════════════════ v2 SETTINGS (edit here, then restart the API) ═════════════════════════
WHOIS_ENABLED   = True      # False = never look up domain age (faster, fewer dependencies)
GSB_FOR_TRUSTED = True      # run Google Safe Browsing also for whitelisted hosts

# Trust only LOWERS the model score. The multiplier is applied to the probability.
TRUST_MULTIPLIER = {"strong": 0.50, "weak": 0.80}
# ...unless the raw model probability is this high: then trust is ignored (very high = flip).
TRUST_FLIP_PROBA = {"strong": 0.95, "weak": 0.90}

# Rule layer (typosquat.py). A hard hit forces MALICIOUS, a soft hit raises the score.
RULE_HARD_SCORE  = 0.85     # homoglyph, hyphen/www tricks, combosquat+keyword, brand in subdomain...
RULE_SOFT_SCORE  = 0.60     # edit distance 2 and similar
RULE_HARD_FLOOR  = 0.90     # probability a hard hit is raised to
RULE_SOFT_BUMP   = 0.20

# Evidence of badness found in the URL itself. Soft evidence adds to the score.
EVIDENCE_BUMP = {"idn_punycode": 0.25, "offsite_redirect": 0.15, "sub_spoof": 0.30}

# FIX 4: dampening for URLs where every heuristic red flag is 0 (1.0 = off).
# v2: OFF by default. It was tuned for the old model. With the new model most real phishing
# looks "clean" (no red flag), so 0.35-0.55 would turn almost every model detection below
# the threshold, on top of a threshold that already limits false positives (FPR < 5%).
# Re-enable (old values 0.55 / 0.35) only if a regression set shows too many false positives.
CLEAN_DAMPEN_HTTP  = 1.0
CLEAN_DAMPEN_HTTPS = 1.0

# Academic / government suffixes: weak trust only (they used to bypass the model).
ACADEMIC_GOV_SUFFIXES = {
    "ac.lk", "edu.lk", "gov.lk", "ac.uk", "gov.uk", "edu.au", "gov.au", "ac.nz",
    "edu", "gov", "mil", "ac.in", "edu.in", "ac.jp", "ac.kr",
}
# Hosts where the subdomain ("tenant") is chosen by whoever signs up: never trusted.
SHARED_PLATFORMS = set(ts.SHARED_PLATFORMS) | {
    "medium.com", "substack.com", "github.com.cn", "vercel.com", "netlify.com",
    "herokuapp.com", "pythonanywhere.com", "ngrok.io", "ngrok-free.app", "bubbleapps.io",
}
REDIRECT_KEYS = {"url", "redirect", "next", "goto", "return", "returnurl", "returnto",
                 "continue", "dest", "destination", "redir", "redirect_uri", "callback",
                 "q", "target", "link", "u", "r", "to", "out", "forward"}

# ── Feature names (from feature_config.json so they always match the trained model) ──
def _load_feature_names() -> list:
    if os.path.exists(FEATURE_CONFIG_PATH):
        with open(FEATURE_CONFIG_PATH, encoding="utf-8") as fh:
            cfg = json.load(fh)
        return (list(cfg["heuristic_feature_names"])
                + [f"char_{i}" for i in range(cfg["n_char"])]
                + [f"word_{i}" for i in range(cfg["n_word"])])
    warnings.warn(f"{FEATURE_CONFIG_PATH} not found: assuming 300 char + 202 word features",
                  RuntimeWarning)
    return (list(HEURISTIC_FEATURE_NAMES)
            + [f"char_{i}" for i in range(300)] + [f"word_{i}" for i in range(202)])


HEURISTIC_FEATURES: list = list(HEURISTIC_FEATURE_NAMES)          # v2: 62
FEATURE_NAMES: list = _load_feature_names()                       # v2: dynamic

_FEAT_IDX = {name: i for i, name in enumerate(HEURISTIC_FEATURES)}

_CATEGORICAL_FEATURE_NAMES: list = [
    "ip_flag", "has_multi_subdomain", "risky_tld", "https_flag",
    "shortened", "sus_words", "brand_mismatch", "puny", "susp_ext",
    "suspicious_port", "brand_in_domain", "leet_in_domain",
    "brand_hyphen_suspicious", "has_redirect", "double_slash_in_path",
    "abnormal_subdomain", "http_no_brand_no_age",
    "leet_brand_score",
]

# FIX 12: words that should never appear in a legitimate domain label
_DOMAIN_SUSPICIOUS_WORDS: set = {
    "phishing", "malware", "trojan", "virus", "hack",
    "steal", "crack", "cheat", "fraud", "scam",
    "evil", "danger", "threat", "attack", "exploit",
    "payload", "botnet", "ransomware", "spyware",
    "keylogger", "rootkit", "worm", "phish", "pwn",
    "shell", "backdoor", "owned",
}

_PRIVATE_IP_RE = re.compile(
    r"^(10\.|172\.(1[6-9]|2\d|3[01])\.|192\.168\.|127\.|0\.0\.0\.0|::1)"
)

# ── TLDExtract instance ───────────────────────────────────────────────────────
_EXTRACTOR = _tldextract.TLDExtract(cache_dir=None, suffix_list_urls=None)

# ── Brand map (display names) ─────────────────────────────────────────────────
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

# ── Trusted subdomain prefixes ────────────────────────────────────────────────
TRUSTED_SUBDOMAIN_PREFIXES: set = {
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
    "m", "mobile", "wap",
    "en", "us", "uk", "au", "ca", "in", "de", "fr",
    "cloud", "aws", "azure",
    "v1", "v2", "v3",
}


# ════════════════════════════════════════════════════════════════
# WHITELIST — LOAD  (now a trust signal, see assess_trust)
# ════════════════════════════════════════════════════════════════

def _load_whitelist() -> set:
    whitelist = set()

    WHITELIST_MANUAL = {
        "google.com", "gmail.com", "youtube.com", "googleapis.com",
        "google.co.uk", "google.com.au", "google.ca", "google.in",
        "github.com", "gitlab.com", "stackoverflow.com",
        "wikipedia.org", "wikimedia.org",
        "microsoft.com", "microsoftonline.com", "live.com",
        "outlook.com", "office.com", "azure.com", "bing.com",
        "apple.com", "icloud.com",
        "amazon.com", "amazon.co.uk", "amazon.de", "amazon.fr",
        "amazonaws.com",
        "facebook.com", "instagram.com", "twitter.com", "x.com",
        "linkedin.com", "reddit.com", "pinterest.com",
        "whatsapp.com", "telegram.org", "discord.com",
        "paypal.com", "bankofamerica.com", "chase.com",
        "wellsfargo.com", "citibank.com", "visa.com",
        "mastercard.com", "stripe.com",
        "netflix.com", "spotify.com", "twitch.tv",
        "slack.com", "zoom.us",
        "ebay.com", "etsy.com", "shopify.com",
        "dropbox.com", "adobe.com", "salesforce.com",
        "wordpress.com", "medium.com",
        "roadmap.sh", "dev.to", "freecodecamp.org",
        "scrimba.com", "codecademy.com", "hashnode.com",
        "hackerearth.com", "hackerrank.com", "leetcode.com",
        "codechef.com", "codeforces.com", "kaggle.com",
        "replit.com", "codepen.io", "codesandbox.io",
        "exercism.org", "theodinproject.com",
        "python.org", "nodejs.org", "rust-lang.org", "golang.org",
        "reactjs.org", "vuejs.org", "angular.io", "svelte.dev",
        "nextjs.org", "djangoproject.com", "flask.palletsprojects.com",
        "npmjs.com", "pypi.org", "crates.io",
        "dailyremote.com", "jobspresso.co", "remoteok.com",
        "weworkremotely.com", "flexjobs.com", "remote.co",
        "wellfound.com", "monster.com", "ziprecruiter.com",
        "indeed.com", "glassdoor.com",
        "4kwallpapers.com", "unsplash.com", "pexels.com",
        "neverssl.com", "httpforever.com",
        "vercel.com", "netlify.com", "heroku.com",
        "digitalocean.com", "cloudflare.com",
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

    # URL shorteners must go to the model, never be trusted
    whitelist = whitelist - SHORTENERS
    return whitelist


WHITELIST = _load_whitelist()


# ════════════════════════════════════════════════════════════════
# v2: URL PARSING AND TRUST ASSESSMENT
# ════════════════════════════════════════════════════════════════

_IPV4_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def parse_url(url: str) -> dict:
    """Parse once, correctly (userinfo, port, www), and reuse everywhere."""
    s = str(url).strip()
    has_scheme = "://" in s
    sp = urlsplit(s if has_scheme else "//" + s)
    try:
        host = (sp.hostname or "").lower().rstrip(".")
    except ValueError:
        host = ""
    ext = _EXTRACTOR(host) if host else _EXTRACTOR("")
    registered = ext.registered_domain or host
    sub_parts = [p for p in (ext.subdomain or "").split(".") if p and p != "www"]
    userinfo = sp.netloc.rsplit("@", 1)[0] if "@" in sp.netloc else ""
    return dict(url=s, scheme=sp.scheme.lower() if has_scheme else "", host=host,
                registered=registered, suffix=ext.suffix or "", label=ext.domain or "",
                sub_parts=sub_parts, userinfo=userinfo, path=sp.path or "",
                query=sp.query or "", is_ip=bool(_IPV4_RE.match(host)) or ":" in host)


def _redirect_target(p: dict) -> Optional[str]:
    """First query value that is a URL (or host/path) pointing to ANOTHER registered domain."""
    for key, val in parse_qsl(p["query"], keep_blank_values=False):
        if key.lower() not in REDIRECT_KEYS or not val:
            continue
        v = unquote(val).strip()
        cand = v if "://" in v else ("http://" + v if re.match(r"^[\w.-]+\.[a-z]{2,}(/|$)", v, re.I) else "")
        if not cand:
            continue
        tp = parse_url(cand)
        if tp["host"] and tp["registered"] != p["registered"]:
            return cand
    return None


def assess_trust(p: dict) -> dict:
    """Return {"level": "strong"|"weak"|"none", "evidence": [codes], "redirect_target": url|None}.

    The whitelist grants trust only when nothing in the URL contradicts it:
      userinfo_domain  http://google.com@evil.tk  (the part before @ looks like a domain)
      idn_punycode     xn-- or non-ASCII host
      offsite_redirect a redirect parameter pointing to a different registered domain
      sub_spoof        a different brand's name used as a subdomain (paypal.google.com)
      shared_platform_tenant  x.github.io, x.weebly.com ...  (the tenant is not the owner)
      ip_host          raw IP address
    """
    evidence: list = []
    host, reg = p["host"], p["registered"]
    if p["userinfo"] and "." in p["userinfo"]:
        evidence.append("userinfo_domain")
    if "xn--" in host or any(ord(c) > 127 for c in host):
        evidence.append("idn_punycode")
    target = _redirect_target(p)
    if target:
        evidence.append("offsite_redirect")
    if p["is_ip"]:
        evidence.append("ip_host")
    if reg in SHARED_PLATFORMS and p["sub_parts"]:
        evidence.append("shared_platform_tenant")

    trusted_reg = reg in WHITELIST
    if trusted_reg and p["sub_parts"]:
        idx = ts.get_index()
        for part in p["sub_parts"]:
            b = idx.brands.get(part)
            if b is not None and b.risk and b.domain != reg:
                evidence.append("sub_spoof")
                break

    level = "none"
    if not evidence:
        if trusted_reg:
            level = "strong" if all(x in TRUSTED_SUBDOMAIN_PREFIXES for x in p["sub_parts"]) else "weak"
        elif p["suffix"] in ACADEMIC_GOV_SUFFIXES:
            level = "weak"
    return dict(level=level, evidence=evidence, redirect_target=target)


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

# v2: sentences for rule hits and evidence (shown first in "reasons")
_TECHNIQUE_TEXT = {
    "homoglyph": "This address copies {brand} using look-alike characters (the real site is {real})",
    "hyphen_insert": "This address copies {brand} by splitting its name with a hyphen (the real site is {real})",
    "dot_split": "This address copies {brand} by splitting its name with a dot (the real site is {real})",
    "www_prefix": "This address copies {brand} with a 'www' glued to the name (the real site is {real})",
    "combosquat_keyword": "This address combines the name {brand} with words like 'secure' or 'login', a common phishing trick (the real site is {real})",
    "combosquat": "This address combines the name {brand} with extra words (the real site is {real})",
    "brand_domain_in_subdomain": "The address starts with {real}, but the site itself belongs to a different domain",
    "brand_in_subdomain": "The name {brand} is only a prefix: the site itself belongs to a different domain (the real site is {real})",
    "tld_swap": "This address uses the name {brand} with a different domain ending (the real site is {real})",
    "brand_in_path": "The page path mentions {brand}, but this site is not {real}",
    "brand_substring": "The domain name hides the brand name {brand}",
}
_MISSPELLING = {"transposition", "keyboard_adjacent", "bitsquat", "doubled_letter", "dropped_letter",
                "substitution", "insertion", "plural", "edit_distance_2"}
_EVIDENCE_TEXT = {
    "userinfo_domain": "The address contains an '@' that makes it look like another website",
    "idn_punycode": "The address uses international characters that can disguise its real name",
    "offsite_redirect": "The link redirects to a different website",
    "sub_spoof": "A well-known brand name is used as a prefix on a different website",
    "redirect_to_malicious": "The link redirects to a website that looks malicious",
}


def rule_reason(m: dict) -> str:
    brand = BRAND_MAP.get(m["brand"], (m["brand"].title(), m["brand_domain"]))[0]
    tpl = _TECHNIQUE_TEXT.get(m["technique"])
    if tpl is None and m["technique"] in _MISSPELLING:
        tpl = "This address is a misspelling of {brand} (the real site is {real})"
    return (tpl or "This address imitates {brand}").format(brand=brand, real=m["brand_domain"])


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
    if not domain or not WHOIS_ENABLED:                              # v2:
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


def adjust_confidence_by_domain_age(proba: float, domain: str) -> tuple:
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

def check_google_safe_browsing(url: str) -> tuple:
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
# BRAND DETECTION  (v2: token based, so "groups.google.com" no longer matches "ups")
# ════════════════════════════════════════════════════════════════

def detect_brand(url: str, ts_result: Optional[dict] = None) -> tuple:
    """Return (display name, real domain) of the brand the URL imitates, or (None, None)."""
    if ts_result and ts_result.get("brand"):
        label = ts_result["brand"]
        disp = BRAND_MAP.get(label, (label.title(), ts_result["brand_domain"]))[0]
        return disp, ts_result["brand_domain"]

    p = parse_url(url)
    if p["registered"] in REAL_BRAND_DOMAINS:
        return None, None
    tokens = [t for t in re.split(r"[^a-z0-9]+", (p["host"] + " " + p["path"]).lower()) if t]
    for tok in tokens:
        if tok in BRAND_MAP and p["registered"] != BRAND_MAP[tok][1]:
            return BRAND_MAP[tok]
    try:
        for tok in tokens:
            for candidate in decode_leet(tok):
                if candidate != tok and candidate in BRAND_MAP and p["registered"] not in REAL_BRAND_DOMAINS:
                    return BRAND_MAP[candidate]
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

    if brand_name is None and weight > 0 and any(
        x in feature for x in ["brand", "visual_brand", "leet_brand"]
    ):
        return None

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
) -> list:
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
        if feat_name == "https_flag":                              # v2: features have no scheme
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


def unshorten_url(url: str, timeout: int = 5) -> tuple:
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
        p = parse_url(url)
        return p["registered"] in SHORTENERS or p["host"] in SHORTENERS
    except Exception:
        return False


def _extract_ip(url: str) -> Optional[str]:
    try:
        p = parse_url(url)
        return p["host"] if p["is_ip"] else None
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

        # v2: the model, the scaler and feature_config.json must come from the same run
        if self.model.n_features_in_ != len(FEATURE_NAMES):
            raise RuntimeError(
                f"model expects {self.model.n_features_in_} features but feature_config.json "
                f"describes {len(FEATURE_NAMES)}: re-run feature_extraction.py and train_model.py")
        ts.get_index()                                              # v2: fail early, warm the index

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
        """v2: public-suffix aware (co.uk, com.au ...), no longer the last two labels."""
        return parse_url(url)["registered"]

    def _feature_vector(self, url: str) -> np.ndarray:
        h  = np.array(extract_heuristic_features(url), dtype=np.float32).reshape(1, -1)
        hs = self.scaler.transform(h).flatten()
        s  = segment_url(url)
        c  = self.vec_char.transform([s]).toarray().flatten()
        w  = self.vec_word.transform([s]).toarray().flatten()
        return np.concatenate([hs, c, w])

    # ── scoring ───────────────────────────────────────────────────────────────

    def _score(self, url: str, p: dict, trust: dict, rules: dict) -> dict:
        """Model probability plus rules, overrides, dampening, trust and domain age."""
        fv        = self._feature_vector(url)
        raw_proba = float(self.model.predict_proba(fv.reshape(1, -1))[0][1])
        raw       = extract_heuristic_features(url)
        is_https  = p["scheme"] == "https"                         # v2: the REAL scheme
        level     = trust["level"]
        proba     = raw_proba
        flags     = lambda name: raw[_FEAT_IDX.get(name, -1)]

        ts_score = rules["score"] if rules["where"] != "path" else 0.0
        hard_rule     = ts_score >= RULE_HARD_SCORE
        soft_rule     = RULE_SOFT_SCORE <= ts_score < RULE_HARD_SCORE
        hard_evidence = "userinfo_domain" in trust["evidence"]

        forced = None                       # name of the override that fired (skips age + trust)
        if hard_rule or hard_evidence:
            forced = "rule" if hard_rule else "evidence"
            proba = max(proba, RULE_HARD_FLOOR)
        elif level != "strong":             # v2: substring-based overrides never touch strong trust
            domain_label = p["label"].lower()
            leet_detected = (flags("leet_in_domain") == 1.0) or (flags("leet_brand_score") == 1.0)
            if leet_detected and not is_https:                                          # FIX 9
                forced, proba = "leet", max(proba, 0.85)
            elif any(w in domain_label for w in _DOMAIN_SUSPICIOUS_WORDS):              # FIX 12
                forced, proba = "suspicious_domain", max(proba, 0.90)
            elif (flags("brand_in_domain") == 1.0 or flags("brand_mismatch") == 1.0) \
                    and not is_https:                                                   # FIX 13
                forced, proba = "brand_impersonation", max(proba, 0.85)
            elif flags("ip_flag") == 1.0 and not is_https:                              # FIX 14
                forced, proba = "ip", max(proba, 0.85)

        if forced:
            return dict(proba=proba, raw_proba=raw_proba, age=f"skipped_{forced}",
                        trust_applied=False, forced=forced)

        # FIX 4: clean sites (no heuristic red flag, no rule hit) are dampened
        other_flags = [
            "brand_mismatch", "sus_words", "brand_in_domain", "risky_tld", "shortened",
            "ip_flag", "puny", "leet_in_domain", "leet_brand_score",
            "brand_hyphen_suspicious", "susp_ext", "suspicious_port", "has_redirect",
            "double_slash_in_path", "abnormal_subdomain", "http_no_brand_no_age",
        ]
        is_clean = all(flags(f) == 0.0 for f in other_flags) and rules["score"] == 0.0
        if is_clean and proba < 0.90:
            proba *= CLEAN_DAMPEN_HTTPS if is_https else CLEAN_DAMPEN_HTTP

        if soft_rule:                                               # v2: soft rule hit
            proba = min(1.0, proba + RULE_SOFT_BUMP)
        for code in trust["evidence"]:                              # v2: soft evidence
            proba = min(1.0, proba + EVIDENCE_BUMP.get(code, 0.0))

        trust_applied = False
        if level != "none":                                         # v2: trust only lowers
            if raw_proba >= TRUST_FLIP_PROBA[level]:
                proba = max(proba, raw_proba)                       # very high: trust ignored
            else:
                proba *= TRUST_MULTIPLIER[level]
                trust_applied = True

        age = "skipped_trusted"
        if level != "strong":                                       # v2: no WHOIS for strong trust
            proba, age = adjust_confidence_by_domain_age(proba, p["registered"])
        return dict(proba=proba, raw_proba=raw_proba, age=age, trust_applied=trust_applied,
                    forced=None)

    # ── predict_url ───────────────────────────────────────────────────────────

    def predict_url(self, url: str) -> dict:
        return self._predict(url, depth=0)

    def _threshold_pct(self) -> float:
        return round(self.threshold * 100, 2)

    def _predict(self, url: str, depth: int) -> dict:
        if not url or not isinstance(url, str):
            return {"url": url, "prediction": "BENIGN", "confidence": 0.0,
                    "threshold": self._threshold_pct(), "source": "invalid"}

        original_url = url
        resolved_url = None
        unshortened  = None
        hostname     = None
        ip           = None

        # Layer 1: Google Safe Browsing (every URL, trusted or not)
        p0 = parse_url(url)
        quick_trust = p0["registered"] in WHITELIST and not p0["userinfo"]
        if GSB_FOR_TRUSTED or not quick_trust:
            is_mal, threat_type = check_google_safe_browsing(url)
            if is_mal:
                return {"url": original_url, "prediction": "MALICIOUS", "confidence": 100.0,
                        "threshold": self._threshold_pct(), "source": "google_safe_browsing",
                        "threat_type": threat_type}

        # Raw IP + no HTTPS -> MALICIOUS before reverse DNS (reverse DNS would hide the IP)
        if _extract_ip(url) and p0["scheme"] != "https":
            return {"url": original_url, "prediction": "MALICIOUS", "confidence": 85.0,
                    "threshold": self._threshold_pct(), "source": "model",
                    "domain_age": "skipped_ip"}

        # Reverse DNS for IP URLs / unshorten short links
        ip = _extract_ip(url)
        if ip:
            hostname = reverse_dns(ip)
            if hostname:
                resolved_url = url.replace(ip, hostname)
                url = resolved_url
        elif _is_shortener(url):
            final_url, was_redirected = unshorten_url(url)
            if was_redirected:
                unshortened = final_url
                is_mal, threat_type = check_google_safe_browsing(final_url)
                if is_mal:
                    return {"url": original_url, "prediction": "MALICIOUS", "confidence": 100.0,
                            "threshold": self._threshold_pct(), "source": "google_safe_browsing",
                            "threat_type": threat_type, "unshortened": final_url}
                url = final_url

        # Layer 2 + 3: trust assessment and rule layer
        p     = parse_url(url)
        trust = assess_trust(p)
        rules = ts.analyze(url)

        # Redirect target in the query string: judge where the link really goes
        if trust["redirect_target"] and depth == 0:
            target = self._predict(trust["redirect_target"], depth=1)
            if target["prediction"] == "MALICIOUS":
                trust["evidence"].append("redirect_to_malicious")
                return {"url": original_url, "prediction": "MALICIOUS",
                        "confidence": max(target["confidence"], 90.0),
                        "threshold": self._threshold_pct(), "source": "model",
                        "domain_age": "skipped_redirect", "evidence": trust["evidence"],
                        "redirect_target": trust["redirect_target"]}

        hard_hit = (rules["score"] >= RULE_HARD_SCORE and rules["where"] != "path") \
            or "userinfo_domain" in trust["evidence"]
        if hard_hit:
            trust = dict(trust, level="none")                      # trust never survives a hard hit

        # Layer 4 + 5: model, overrides, trust, domain age
        try:
            sc = self._score(url, p, trust, rules)
        except Exception as exc:
            return {"url": original_url, "prediction": "BENIGN", "confidence": 0.0,
                    "threshold": self._threshold_pct(), "source": "invalid", "error": str(exc)}

        if sc["forced"]:                                           # an override fired: not "trusted"
            trust = dict(trust, level="none")
        proba = sc["proba"]
        label = "MALICIOUS" if proba >= self.threshold else "BENIGN"
        trusted = trust["level"] != "none"

        if trusted and label == "BENIGN":
            # externally identical to the old whitelist answer, but it was analysed
            result = {"url": original_url, "prediction": "BENIGN", "confidence": 100.0,
                      "threshold": self._threshold_pct(), "source": "whitelist",
                      "whitelisted": True, "trust": trust["level"],
                      "risk_score": round(proba * 100, 2)}
        else:
            result = {"url": original_url, "prediction": label,
                      "confidence": round(proba * 100, 2),
                      "threshold": self._threshold_pct(), "source": "model",
                      "domain_age": sc["age"]}
            if trusted:
                result.update(whitelisted=True, trust=trust["level"])

        brand_name, real_domain = detect_brand(url, rules)
        if brand_name and not (trusted and label == "BENIGN"):
            result["brand_detected"] = brand_name
            result["real_domain"]    = real_domain
        if rules["score"] >= RULE_SOFT_SCORE and rules["where"] != "path":
            result["rules"] = [f"{m['technique']}:{m['brand']}" for m in rules["matches"][:3]]
        if trust["evidence"]:
            result["evidence"] = trust["evidence"]
        if resolved_url and ip:
            result["resolved_ip"] = hostname
        if unshortened:
            result["unshortened"] = unshortened
        return result

    def predict_batch(self, urls: list) -> list:
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

        # v2: rule and evidence reasons come first: they are the most specific explanation
        lead: list = []
        rules = ts.analyze(classify_url)
        if rules["score"] >= RULE_SOFT_SCORE and rules["where"] != "path":
            lead.append(rule_reason(rules["matches"][0]))
        for code in base.get("evidence", []):
            if code in _EVIDENCE_TEXT and _EVIDENCE_TEXT[code] not in lead:
                lead.append(_EVIDENCE_TEXT[code])

        try:
            explainer = self._get_explainer()
            fv        = self._feature_vector(classify_url)

            exp = explainer.explain_instance(
                data_row     = fv,
                predict_fn   = self._lime_predict_fn,
                num_features = num_features,
                labels       = (1,),
            )

            explanation, reasons, seen = [], [], set(lead)

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
                    if feat_name == "https_flag":                   # v2: scheme is not a feature
                        continue
                    nl = feature_to_natural_language(
                        feat_name, weight, feat_val, brand_name, real_domain
                    )
                    if nl and nl not in seen:
                        seen.add(nl)
                        reasons.append(nl)

            explanation.sort(key=lambda x: abs(x["weight"]), reverse=True)
            top_reasons = (lead + reasons)[:3]

            if len(top_reasons) < 3:
                top_reasons.extend(
                    _build_backup_reasons(
                        raw_heuristic, brand_name, real_domain,
                        set(top_reasons), 3 - len(top_reasons),
                    )
                )

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
                raw_heuristic, brand_name, real_domain, set(lead), max(0, 3 - len(lead))
            )
            return {**base, "explanation": [], "reasons": (lead + backup)[:3],
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
            bg = np.load(bg_path)["X"]
            if bg.shape[1] == len(FEATURE_NAMES):                   # v2: ignore a stale background
                return bg
            warnings.warn("lime_background.npz has a different feature count: rebuilding it.",
                          RuntimeWarning, stacklevel=3)

        warnings.warn(f"{bg_path} not found or stale — building LIME background.",
                      RuntimeWarning, stacklevel=3)
        import csv, random
        csv_path = os.path.join(_ROOT, "data", "splits_v2", "train_urls.csv")   # v2: new split
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


# ════════════════════════════════════════════════════════════════
# v2: SELF-TEST  (python app/model_loader.py)
# ════════════════════════════════════════════════════════════════
# (url, expected prediction or None = just report, note)
_SELFTEST_CASES = [
    # lookalikes and tricks: must be MALICIOUS whatever the model says
    ("https://www.google.com.evil.tk/",               "MALICIOUS", "brand domain inside the subdomain"),
    ("https://gooogle.com/",                          "MALICIOUS", "doubled letter"),
    ("https://paypal-secure.com/",                    "MALICIOUS", "combosquat"),
    ("https://paypa1.com/login",                      "MALICIOUS", "homoglyph"),
    ("https://arnazon.com/",                          "MALICIOUS", "rn -> m"),
    ("https://wwwgoogle.com/",                        "MALICIOUS", "www glued to the name"),
    ("https://paypal.weebly.com/",                    "MALICIOUS", "brand as shared-platform tenant"),
    ("https://google.com@evil.tk/login",              "MALICIOUS", "user@host trick"),
    ("https://www.google.com/url?q=https://paypa1-login.com/", "MALICIOUS", "trusted host redirecting to a lookalike"),
    ("https://paypal-verify.ac.lk/",                  "MALICIOUS", "phishing name on an academic suffix"),
    ("https://phishing-login.edu.lk/",                "MALICIOUS", "suspicious word on an academic suffix"),
    # trusted hosts: must stay BENIGN (trusted, but analysed)
    ("https://www.google.com/search?q=cats+and+dogs&hl=en&source=hp", "BENIGN", "long Google search URL"),
    ("https://accounts.google.com/signin/v2/identifier", "BENIGN", "trusted subdomain"),
    ("https://github.com/google/guava",               "BENIGN", "brand in path of a trusted host"),
    ("https://docs.python.org/3/library/re.html",     "BENIGN", "trusted subdomain"),
    ("https://www.amazon.co.uk/",                     "BENIGN", "multi-part suffix"),
    # informational: depends on the model, shown for tuning
    ("https://staal.github.com/",                     None, "unknown subdomain of a trusted domain (weak trust)"),
    ("https://uom.ac.lk/",                            None, "academic suffix (weak trust)"),
    ("https://someone.github.io/",                    None, "shared platform tenant (no trust)"),
]


def _selftest() -> int:
    global WHOIS_ENABLED, GSB_API_KEY
    WHOIS_ENABLED, GSB_API_KEY = False, ""          # offline and deterministic
    print("\nMODEL LOADER SELF-TEST (WHOIS and Safe Browsing off)")
    print(f"model features: {classifier.model.n_features_in_} | threshold: {classifier.threshold:.2f}\n")
    failed = 0
    for url, expected, note in _SELFTEST_CASES:
        r = classifier.predict_url(url)
        ok = expected is None or r["prediction"] == expected
        failed += 0 if ok else 1
        tag = "info" if expected is None else ("PASS" if ok else "FAIL")
        extra = []
        if r.get("trust"):
            extra.append(f"trust={r['trust']}")
        if r.get("rules"):
            extra.append("rules=" + ",".join(r["rules"][:2]))
        if r.get("evidence"):
            extra.append("evidence=" + ",".join(r["evidence"]))
        if "risk_score" in r:
            extra.append(f"risk={r['risk_score']}")
        print(f"[{tag}] {r['prediction']:<9} {r['confidence']:>6.1f}  {url}")
        print(f"        {note}  {' '.join(extra)}")
    print(f"\n{'ALL REQUIRED CASES PASSED' if not failed else str(failed) + ' CASE(S) FAILED'}")
    print("A FAIL on a trusted URL usually means the model probability for that URL is above "
          "TRUST_FLIP_PROBA: look at the risk/confidence value and tune the settings at the top.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_selftest())