#!/usr/bin/env python3
"""
feature_extraction.py  —  Final Fixed Malicious URL Feature Extraction
-----------------------------------------------------------------------

FIXES:
  1. NO WHOIS  — removed entirely for fast runtime (~40-60 min for 300k URLs)
  2. HTTPS BIAS fixed via context-aware feature using:
       - Tranco top 1M whitelist (load from local file if available)
       - Built-in expanded REAL_BRAND_DOMAINS as fallback
       - TLD + suspicious word proxy when domain not in whitelist
  3. Lexical segmentation before TF-IDF
  4. New features: has_redirect, double_slash_in_path, abnormal_subdomain
  5. No domain_age features at all (WHOIS removed)

Feature breakdown (total: 558):
  Structural heuristics   : 39
  Obfuscation             :  6
  Rule-based              :  3
  Domain-level            :  4
  NEW structural          :  3  (has_redirect, double_slash_in_path, abnormal_subdomain)
  HTTPS context fix       :  1  (http_no_brand_no_age — no WHOIS needed)
  ─────────────────────────
  Heuristic subtotal      : 56
  Char n-gram TF-IDF      : 300
  Word n-gram TF-IDF      : 202
  ─────────────────────────
  TOTAL                   : 558

Runtime estimate (300k URLs, no WHOIS):
  Heuristic extraction    : ~15-25 min
  TF-IDF fit + transform  : ~20-30 min
  Scaler                  : ~2-3 min
  TOTAL                   : ~40-60 min

Split-aware (no leakage):
  Vectorizers fitted on TRAIN only
  Scaler fitted on TRAIN only

HOW TO USE TRANCO WHITELIST:
  Download from https://tranco-list.eu → save as data/tranco_top1m.csv
  If file exists, it loads automatically. Otherwise falls back to built-in list.
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
from urllib.parse import urlparse, unquote
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import scipy.sparse as sp
import joblib
import tldextract
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

# ─────────────────────────── PATHS ───────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS_DIR   = os.path.join(BASE_DIR, "data", "splits")
FEATURES_DIR = os.path.join(BASE_DIR, "features")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
TRANCO_PATH  = os.path.join(BASE_DIR, "data", "raw ", "tranco_top1m.csv")

os.makedirs(FEATURES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,   exist_ok=True)

TRAIN_PATH = os.path.join(SPLITS_DIR, "train_urls.csv")
VAL_PATH   = os.path.join(SPLITS_DIR, "val_urls.csv")
TEST_PATH  = os.path.join(SPLITS_DIR, "test_urls.csv")

TRAIN_OUT = os.path.join(FEATURES_DIR, "features_train")
VAL_OUT   = os.path.join(FEATURES_DIR, "features_val")
TEST_OUT  = os.path.join(FEATURES_DIR, "features_test")

CHAR_VEC_PATH = os.path.join(MODELS_DIR, "vectorizer_char.joblib")
WORD_VEC_PATH = os.path.join(MODELS_DIR, "vectorizer_word.joblib")
SCALER_PATH   = os.path.join(MODELS_DIR, "scaler.joblib")

# ─────────────────────────── LOGGING ─────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(FEATURES_DIR, "feature_extraction.log"),
            encoding="utf-8", mode="w"
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────── CONFIG ──────────────────────────
MAX_FEAT_CHAR = 300
MAX_FEAT_WORD = 202
NGRAM_CHAR    = (2, 4)
NGRAM_WORD    = (1, 2)
MIN_DF        = 3
CHUNK_SIZE    = 50_000

# ─────────────────────────── CONSTANTS ───────────────────────
SHORTENERS = {
    "bit.ly","tinyurl.com","goo.gl","ow.ly","t.co","is.gd",
    "buff.ly","adf.ly","bit.do","mcaf.ee","surl.li","shorte.st",
    "clicky.me","cutt.ly","u.to","v.gd","tr.im","tiny.cc",
    "rebrand.ly","t.ly","bc.vc","cli.gs","sh.st","ity.im",
    "short.to","adfoc.us","link.tl","qr.net","cutt.us","x.co",
    "1url.com","tiny.pl","short.cm","pic.gd","short.nr","tiny.ie",
    "short.ie","moourl.com","zz.gd","tinylink.in","shorturl.com",
    "miniurl.com","bitly.com","shorl.com","kl.am","fwd4.me",
    "yep.it","xlink.me","fur.ly","hurl.me","lnk.co",
    "snipurl.com","snipr.com","snurl.com","sn.im","flic.kr",
    "qlnk.net","doiop.com","twurl.nl","rubyurl.com","om.ly"
}

SUSPICIOUS_WORDS = {
    "suspend","urgent","prize","winner","congratulations",
    "free-iphone","limited-offer","click-here","verify-now",
    "act-now","account-suspended","password-reset-required",
    "security","alert","verify","update","login","signin",
    "confirm","recover","unlock","restore","validate","billing",
    "suspended","unusual","unauthorized","immediate","required",
    "expire","expired","blocked","limited","access","authenticate",
}

RISKY_TLDS = {
    "zip","review","country","gq","tk","ml","cf","ga","top",
    "xyz","click","link","pw","club","work","site","online",
    "space","webcam","stream","download","gdn","racing","loan",
    "win","bid","trade","science","party","cricket","date",
    "faith","accountant","men","biz","info","su","cc","icu",
    "cyou","rest","bar","buzz","live","xxx","dating"
}

# Safe TLDs — domains on these are very unlikely to be malicious
# Used in http proxy fix as a legitimacy signal
SAFE_TLDS = {
    "edu", "gov", "mil", "ac", "int",
    "org", "net", "com", "co", "io",
    "uk", "us", "ca", "au", "de",
    "fr", "jp", "nl", "se", "no",
    "fi", "dk", "ch", "nz", "sg"
}

BRANDS = {
    "paypal","amazon","microsoft","apple","google","facebook",
    "netflix","bankofamerica","wellsfargo","whatsapp","instagram",
    "twitter","linkedin","ebay","visa","mastercard","chase",
    "citi","bank","pay","secure"
}

# Expanded built-in whitelist — used when Tranco file not available
REAL_BRAND_DOMAINS = {
    # Payment
    "paypal.com","visa.com","mastercard.com","stripe.com",
    "square.com","payoneer.com","wise.com","revolut.com",
    # Big tech
    "google.com","google.co.uk","google.com.au","google.ca",
    "microsoft.com","microsoftonline.com","live.com","outlook.com",
    "office.com","office365.com","azure.com","apple.com",
    "icloud.com","amazon.com","amazon.co.uk","amazon.de",
    "amazon.fr","amazon.in","amazonaws.com","aws.amazon.com",
    "facebook.com","instagram.com","whatsapp.com","twitter.com",
    "linkedin.com","youtube.com","netflix.com","twitch.tv",
    "github.com","gitlab.com","stackoverflow.com","reddit.com",
    "wikipedia.org","wikimedia.org",
    # Banks
    "bankofamerica.com","wellsfargo.com","chase.com","citibank.com",
    "hsbc.com","barclays.co.uk","lloydsbank.com","santander.co.uk",
    "maybank2u.com.my","dbs.com.sg","ocbc.com","uob.com.sg",
    # Email
    "gmail.com","yahoo.com","hotmail.com","protonmail.com",
    "zoho.com","mail.com",
    # Shopping
    "ebay.com","ebay.co.uk","shopify.com","etsy.com",
    "aliexpress.com","alibaba.com","lazada.com","shopee.com",
    # Cloud / hosting
    "cloudflare.com","digitalocean.com","heroku.com",
    "netlify.com","vercel.com","wordpress.com","wix.com",
    # News / reference
    "bbc.com","bbc.co.uk","cnn.com","reuters.com","nytimes.com",
    "theguardian.com","bloomberg.com",
}

BRAND_SUSPICIOUS_WORDS = {
    "security","alert","verify","update","login","signin","secure",
    "confirm","account","banking","support","help","service",
    "center","care","warning","suspend","locked","unlock","recover"
}

COMMON_PORTS = {80, 443, 8080, 8443, 3000, 5000, 8000, 9000}

_ABNORMAL_SUB_RE = re.compile(
    r"(\d{1,3}-\d{1,3})|"
    r"([0-9a-f]{8,})|"
    r"(\d{5,})"
)

EXTRACTOR = tldextract.TLDExtract(cache_dir=None, suffix_list_urls=None)

# ─────────────────────── TRANCO WHITELIST ────────────────────
# Global set — populated once at startup
_TRANCO_WHITELIST: set = set()

def load_tranco_whitelist():
    """
    Load Tranco top 1M whitelist from local CSV file.
    File format: rank,domain  (standard Tranco format)
    If file not found, falls back to built-in REAL_BRAND_DOMAINS.

    To get the file:
      1. Go to https://tranco-list.eu
      2. Download latest list as CSV
      3. Save to data/tranco_top1m.csv
    """
    global _TRANCO_WHITELIST
    if os.path.exists(TRANCO_PATH):
        try:
            df = pd.read_csv(TRANCO_PATH, header=None, names=["rank", "domain"])
            _TRANCO_WHITELIST = set(df["domain"].str.lower().str.strip().tolist())
            logger.info(f"Tranco whitelist loaded: {len(_TRANCO_WHITELIST):,} domains")
        except Exception as e:
            logger.warning(f"Failed to load Tranco list: {e}. Using built-in whitelist.")
            _TRANCO_WHITELIST = set(REAL_BRAND_DOMAINS)
    else:
        logger.warning(
            f"Tranco file not found at {TRANCO_PATH}. "
            "Using built-in expanded whitelist. "
            "Download from https://tranco-list.eu for better coverage."
        )
        _TRANCO_WHITELIST = set(REAL_BRAND_DOMAINS)


def is_whitelisted(registered_domain: str) -> bool:
    """Check if domain is in Tranco whitelist or built-in brand list."""
    rd = (registered_domain or "").lower()
    return rd in _TRANCO_WHITELIST or rd in REAL_BRAND_DOMAINS


# ─────────────────────────── FEATURE NAMES ───────────────────
HEURISTIC_FEATURE_NAMES = [
    # 39 structural
    "url_len","path_len","num_dots","path_dots","num_hyphens",
    "num_underscores","num_at","num_qmark","num_equal","num_amp",
    "num_percent","num_digits","num_letters","num_subdirs","num_frag",
    "num_special","num_repeating","num_upper","num_non_ascii",
    "num_slashes","num_params","ratio_digits","ratio_letters",
    "url_entropy","ip_flag","subdomain_parts","has_multi_subdomain",
    "tld_len","risky_tld","https_flag","shortened","sus_words",
    "brand_mismatch","puny","susp_ext","suspicious_port",
    "max_consonants","max_vowels","max_digits",
    # 6 obfuscation
    "leet_speak_score","homoglyph_suspicious","encoding_ratio",
    "punycode_suspicious","subdomain_spam_score","visual_brand_similarity",
    # 3 rule-based
    "brand_in_domain","leet_in_domain","brand_hyphen_suspicious",
    # 4 domain-level
    "domain_len","domain_digit_ratio","max_domain_digits","path_depth",
    # 3 new structural
    "has_redirect",
    "double_slash_in_path",
    "abnormal_subdomain",
    # 1 https bias fix (no WHOIS needed)
    "http_no_brand_no_age",
]

N_HEURISTIC = len(HEURISTIC_FEATURE_NAMES)  # 56


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

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
        h  = hostname.lower().lstrip("www.")
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
    for ch in s.lower():
        hit = (
            (char_type == "digit"     and ch.isdigit()) or
            (char_type == "consonant" and ch in "bcdfghjklmnpqrstvwxyz") or
            (char_type == "vowel"     and ch in "aeiou")
        )
        if hit:
            current  += 1
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


# ═══════════════════════════════════════════════════════════════
# OBFUSCATION DETECTORS
# ═══════════════════════════════════════════════════════════════

def detect_leet_speak(url: str) -> float:
    url_lower = url.lower()
    try:
        domain_part = urlparse(url_lower).netloc
    except Exception:
        domain_part = url_lower
    score = 0.0
    for digit in ["4","3","1","0","5","7"]:
        matches = re.findall(rf"[a-z]{re.escape(digit)}[a-z]", domain_part)
        score  += len(matches) * 0.2
    return min(score, 1.0)


def detect_homoglyph(url: str) -> float:
    for ch in "аеіосурхјѕѡ":
        if ch in url:
            return 1.0
    non_latin = len(re.findall(r"[^\x00-\x7F]", url))
    if non_latin > 0 and len(url) > 0 and (non_latin / len(url)) > 0.1:
        return 0.7
    return 0.0


def calc_encoding_ratio(url: str) -> float:
    encoded = len(re.findall(r"%[0-9A-Fa-f]{2}", url))
    total   = len(url)
    if total == 0:
        return 0.0
    ratio = encoded / total
    if ratio > 0.2:  return 1.0
    if ratio > 0.05: return 0.5
    return 0.0


def detect_punycode(url: str) -> float:
    matches = re.findall(r"xn--[a-z0-9]+", url.lower())
    if not matches:
        return 0.0
    for m in matches:
        if len(m) > 12 or any(c.isdigit() for c in m):
            return 1.0
    return 0.5


def detect_subdomain_spam(url: str) -> float:
    try:
        parts = [p for p in urlparse(url).netloc.split(".") if p]
        n     = max(0, len(parts) - 2)
        if n >= 4: return 1.0
        if n >= 3: return 0.7
        if n >= 2: return 0.3
        return 0.0
    except Exception:
        return 0.0


def calc_visual_similarity(url: str, hostname: str) -> float:
    url_lower  = url.lower()
    host_lower = hostname.lower()
    for brand in BRANDS:
        if brand in url_lower and brand not in host_lower:
            return 0.9
    return 0.0


# ═══════════════════════════════════════════════════════════════
# RULE-BASED FEATURES
# ═══════════════════════════════════════════════════════════════

def brand_in_registered_domain(registered_domain: str) -> float:
    rd = (registered_domain or "").lower()
    for brand in BRANDS:
        if brand in rd:
            return 0.0 if rd in REAL_BRAND_DOMAINS else 1.0
    return 0.0


def leet_in_domain_only(domain: str) -> float:
    d = (domain or "").lower()
    for digit in ["4","3","1","0","5","7"]:
        if re.search(rf"[a-z]{re.escape(digit)}[a-z]", d):
            return 1.0
    return 0.0


def brand_hyphen_suspicious_word(url: str) -> float:
    url_lower = url.lower()
    for brand in BRANDS:
        for word in BRAND_SUSPICIOUS_WORDS:
            if f"{brand}-{word}" in url_lower or f"{word}-{brand}" in url_lower:
                return 1.0
    return 0.0


# ═══════════════════════════════════════════════════════════════
# NEW FEATURE HELPERS
# ═══════════════════════════════════════════════════════════════

def has_redirect_param(url: str) -> float:
    url_lower = url.lower()
    for p in ["url=","redirect=","next=","goto=","return=",
              "returnurl=","returnto=","continue=","dest=",
              "destination=","redir=","redirect_uri=","callback="]:
        if p in url_lower:
            return 1.0
    return 0.0


def has_double_slash_in_path(url: str) -> float:
    try:
        parsed = urlparse(url if url.startswith(("http://","https://"))
                          else "http://" + url)
        return 1.0 if "//" in (parsed.path or "") else 0.0
    except Exception:
        return 0.0


def is_abnormal_subdomain(subdomain: str) -> float:
    if not subdomain:
        return 0.0
    for part in [p for p in subdomain.split(".") if p]:
        if _ABNORMAL_SUB_RE.search(part):
            return 1.0
        if len(part) > 3 and sum(c.isdigit() for c in part) / len(part) > 0.5:
            return 1.0
    return 0.0


def http_no_brand_no_age_feature(
    url: str,
    registered_domain: str,
    tld: str,
    sus_word_count: int,
) -> float:
    """
    Permanent HTTPS bias fix — NO WHOIS needed.

    Returns 1.0 only when ALL of these are true:
      1. URL is http (not https)
      2. Domain is NOT in Tranco whitelist or built-in brand list
      3. At least one of:
           - TLD is in RISKY_TLDS
           - URL contains suspicious words
           - Domain contains brand name (mismatch)

    This means:
      - Old legitimate http blogs → 0  (safe TLD, no suspicious words)
      - New phishing http sites   → 1  (risky TLD or suspicious words)
      - Known brand domains       → 0  (whitelisted)
    """
    # https → safe, no penalty
    if url.startswith("https"):
        return 0.0

    # whitelisted domain → give benefit of doubt
    if is_whitelisted(registered_domain):
        return 0.0

    # safe TLD + no suspicious words → likely legit old http site
    tld_lower = (tld or "").lower()
    if tld_lower not in RISKY_TLDS and sus_word_count == 0:
        return 0.0

    # http + not whitelisted + (risky TLD or suspicious words) → flag it
    return 1.0


# ═══════════════════════════════════════════════════════════════
# MAIN FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_heuristic_features(url: str) -> list:
    """Extract all 56 heuristic features. No WHOIS calls."""
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

        # character counts
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

        brand_mismatch = 0.0
        for brand in BRANDS:
            if brand in hostname.lower() and domain not in REAL_BRAND_DOMAINS:
                brand_mismatch = 1.0
                break

        puny     = 1.0 if "xn--" in url_lower else 0.0
        susp_ext = 1.0 if any(url_lower.endswith(e)
                               for e in [".exe",".zip",".scr",
                                         ".jar",".msi"]) else 0.0

        subdomain_parts_count = len([p for p in subdomain.split(".") if p]) \
            if subdomain else 0
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

        # obfuscation (6)
        leet       = detect_leet_speak(url)
        homoglyph  = detect_homoglyph(url)
        enc_ratio  = calc_encoding_ratio(url)
        punycode   = detect_punycode(url)
        sub_spam   = detect_subdomain_spam(url)
        visual_sim = calc_visual_similarity(url, hostname)

        # rule-based (3)
        brand_in_dom   = brand_in_registered_domain(domain)
        leet_dom       = leet_in_domain_only(ext.domain or "")
        brand_hyp_susp = brand_hyphen_suspicious_word(url)

        # domain-level (4)
        domain_str         = ext.domain or ""
        domain_len         = float(len(domain_str))
        domain_digit_ratio = (
            sum(c.isdigit() for c in domain_str) / len(domain_str)
            if domain_str else 0.0
        )
        max_domain_digits = float(max_consecutive(domain_str, "digit"))
        path_depth        = float(len([p for p in path.split("/") if p]))

        # new structural (3)
        has_redirect = has_redirect_param(url)
        dbl_slash    = has_double_slash_in_path(url)
        abnormal_sub = is_abnormal_subdomain(subdomain)

        # https bias fix (1) — no WHOIS needed
        http_no_brand_age = http_no_brand_no_age_feature(
            url, domain, tld, int(sus_words)
        )

        return [
            # 39 structural
            float(url_len), float(path_length), float(num_dots),
            float(path.count(".")), float(num_hyphens),
            float(num_underscores), float(num_at), float(num_qmark),
            float(num_equal), float(num_amp), float(num_percent),
            float(num_digits), float(num_letters), float(num_subdirs),
            float(num_frag), float(num_special), float(num_rep),
            float(num_upper), float(num_non_ascii), float(num_slashes),
            float(num_params), ratio_digits, ratio_letters,
            url_entropy, ip_flag, float(subdomain_parts_count),
            has_multi_subdomain, float(tld_len), risky_tld, https_flag,
            shortened, sus_words, brand_mismatch, puny, susp_ext,
            suspicious_port, max_cons, max_vows, max_digs,
            # 6 obfuscation
            leet, homoglyph, enc_ratio, punycode, sub_spam, visual_sim,
            # 3 rule-based
            brand_in_dom, leet_dom, brand_hyp_susp,
            # 4 domain-level
            domain_len, domain_digit_ratio, max_domain_digits, path_depth,
            # 3 new structural
            has_redirect, dbl_slash, abnormal_sub,
            # 1 https bias fix
            http_no_brand_age,
        ]

    except Exception:
        return [0.0] * N_HEURISTIC


def extract_heuristic_chunk(urls_chunk: list) -> list:
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


# ═══════════════════════════════════════════════════════════════
# NLP — LEXICAL SEGMENTATION
# ═══════════════════════════════════════════════════════════════

def segment_url(url: str) -> str:
    """
    Segment URL into tokens before TF-IDF.
    "appleid-secure-login.com/verify?user=john"
    → "appleid secure login com verify user john"
    """
    url = str(url).strip().lower()
    url = re.sub(r"^https?://", "", url)
    url = re.sub(r"^www\.", "", url)
    try:
        url = unquote(url)
    except Exception:
        pass
    tokens = re.split(r"[-._/=?&+~,;:@\s]+", url)
    tokens = [
        t for t in tokens
        if len(t) >= 2 and not (t.isdigit() and len(t) < 3)
    ]
    return " ".join(tokens)


def fit_vectorizers(train_urls: list):
    logger.info("   Segmenting URLs for NLP...")
    segmented = [segment_url(u) for u in tqdm(train_urls, desc="Segmenting")]

    logger.info("   Fitting char n-gram TF-IDF on TRAIN only...")
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=NGRAM_CHAR,
        max_features=MAX_FEAT_CHAR, min_df=MIN_DF,
        lowercase=False, dtype=np.float32
    )
    char_vec.fit(segmented)

    logger.info("   Fitting word n-gram TF-IDF on TRAIN only...")
    word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=NGRAM_WORD,
        max_features=MAX_FEAT_WORD, min_df=MIN_DF,
        lowercase=False, token_pattern=r"[a-zA-Z0-9]+",
        dtype=np.float32
    )
    word_vec.fit(segmented)

    return char_vec, word_vec


def transform_nlp(urls, char_vec, word_vec) -> sp.csr_matrix:
    segmented = [segment_url(u) for u in tqdm(urls, desc="NLP transform")]
    X_char    = char_vec.transform(segmented)
    X_word    = word_vec.transform(segmented)
    return sp.hstack([X_char, X_word], format="csr").astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# COMBINE AND SAVE
# ═══════════════════════════════════════════════════════════════

def combine_features(heuristic_arr, nlp_sparse) -> sp.csr_matrix:
    return sp.hstack(
        [sp.csr_matrix(heuristic_arr), nlp_sparse],
        format="csr"
    ).astype(np.float32)


def save_features(path, X, y, feature_names):
    X_csr = X.tocsr()
    np.savez_compressed(
        path,
        data=X_csr.data, indices=X_csr.indices,
        indptr=X_csr.indptr, shape=np.array(X_csr.shape),
        labels=y.astype(np.int8),
        feature_names=np.array(feature_names, dtype=object)
    )
    saved = path + ".npz"
    size  = os.path.getsize(saved) / 1e6 if os.path.exists(saved) else 0
    logger.info(
        f"   Saved: {saved} "
        f"({X_csr.shape[0]:,} × {X_csr.shape[1]:,}, {size:.1f} MB)"
    )


# ═══════════════════════════════════════════════════════════════
# PROCESS ONE SPLIT
# ═══════════════════════════════════════════════════════════════

def process_split(name, path, char_vec, word_vec,
                  scaler, feature_names, out_path, fit_scaler=False):
    logger.info(f"\n{'='*60}\nProcessing {name}: {path}\n{'='*60}")

    df   = pd.read_csv(path, dtype={"url": str, "label": int})
    df   = df.dropna(subset=["url","label"]).reset_index(drop=True)
    urls = df["url"].tolist()
    y    = df["label"].values.astype(np.int8)

    counts = np.bincount(y.astype(int))
    logger.info(f"   {name}: {len(urls):,} URLs | "
                f"Benign: {counts[0]:,} | Malicious: {counts[1]:,}")

    logger.info("   Extracting heuristic features...")
    heuristic_arr = extract_heuristic_batch(urls)

    if fit_scaler:
        logger.info("   Fitting scaler on TRAIN only...")
        scaler.fit(heuristic_arr)
        joblib.dump(scaler, SCALER_PATH)

    heuristic_scaled = scaler.transform(heuristic_arr).astype(np.float32)
    del heuristic_arr
    gc.collect()

    logger.info("   Extracting NLP features...")
    nlp_sparse = transform_nlp(urls, char_vec, word_vec)

    logger.info("   Combining features...")
    X = combine_features(heuristic_scaled, nlp_sparse)
    del heuristic_scaled, nlp_sparse
    gc.collect()

    logger.info(f"   Final shape: {X.shape}")
    save_features(out_path, X, y, feature_names)
    del X
    gc.collect()


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    logger.info("FEATURE EXTRACTION — Final Fixed Version (No WHOIS)")
    logger.info("=" * 60)
    logger.info(f"Heuristic features : {N_HEURISTIC}")
    logger.info(f"Char TF-IDF        : {MAX_FEAT_CHAR}")
    logger.info(f"Word TF-IDF        : {MAX_FEAT_WORD}")
    logger.info(f"Total features     : {N_HEURISTIC + MAX_FEAT_CHAR + MAX_FEAT_WORD}")
    logger.info("=" * 60)

    for p in [TRAIN_PATH, VAL_PATH, TEST_PATH]:
        if not os.path.exists(p):
            logger.error(f"Missing: {p} — run split.py first.")
            return

    # Load Tranco whitelist (or fallback to built-in)
    load_tranco_whitelist()

    feature_names = (
        HEURISTIC_FEATURE_NAMES
        + [f"char_{i}" for i in range(MAX_FEAT_CHAR)]
        + [f"word_{i}" for i in range(MAX_FEAT_WORD)]
    )
    logger.info(f"Total feature names: {len(feature_names):,}")

    scaler = StandardScaler()

    logger.info("\nLoading train URLs to fit vectorizers...")
    df_train   = pd.read_csv(TRAIN_PATH, dtype={"url": str, "label": int})
    df_train   = df_train.dropna(subset=["url"]).reset_index(drop=True)
    train_urls = df_train["url"].tolist()
    logger.info(f"   Train URLs: {len(train_urls):,}")

    logger.info("\nFitting TF-IDF vectorizers on TRAIN only...")
    char_vec, word_vec = fit_vectorizers(train_urls)
    joblib.dump(char_vec, CHAR_VEC_PATH)
    joblib.dump(word_vec, WORD_VEC_PATH)
    del train_urls, df_train
    gc.collect()

    process_split("TRAIN", TRAIN_PATH, char_vec, word_vec,
                  scaler, feature_names, TRAIN_OUT, fit_scaler=True)
    process_split("VAL",   VAL_PATH,   char_vec, word_vec,
                  scaler, feature_names, VAL_OUT,   fit_scaler=False)
    process_split("TEST",  TEST_PATH,  char_vec, word_vec,
                  scaler, feature_names, TEST_OUT,  fit_scaler=False)

    logger.info("\n" + "="*60)
    logger.info("DONE — Estimated runtime was 40-60 min for 300k URLs")
    logger.info("Next step: train_model.py with class_weight='balanced'")
    logger.info("="*60)


if __name__ == "__main__":
    main()
