#!/usr/bin/env python3
"""
generate_typosquatting.py
-------------------------
Generates synthetic malicious URLs covering typosquatting patterns
that are underrepresented in real datasets.

This is data augmentation — a standard ML technique used in research.
Label these URLs as 'synthetic_malicious' in reference file
and 'malicious' (label=1) in ML training file.

Patterns covered:
  1. brand-suspicious_word.tld   (paypal-security.com)
  2. suspicious_word-brand.tld   (security-paypal.com)
  3. leet speak brand            (paypa1.com, amaz0n.com)
  4. double letter brand         (paypall.com, amazzon.com)
  5. brand + risky TLD           (paypal.tk, amazon.xyz)
  6. brand.suspicious_word.com   (paypal.security.com)
  7. brand-word-word.tld         (paypal-security-alert.com)
  8. missing letter brand        (payal.com, amazn.com)
  9. extra word brand            (mypaypal.com, mymicrosoft.com)
 10. brand with common suffix    (paypallogin.com, amazonverify.com)

Output:
  data/raw/synthetic_malicious.csv
  Columns: url, label
"""

import os
import re
import random
import itertools
import pandas as pd
import tldextract
from urllib.parse import urlparse

# ---------------- PATHS ----------------
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR    = os.path.join(BASE_DIR, "data", "raw")
OUTPUT_PATH = os.path.join(RAW_DIR, "synthetic_malicious.csv")
os.makedirs(RAW_DIR, exist_ok=True)

# ---------------- BRAND TARGETS ----------------
# Real brand names that phishers commonly impersonate
BRANDS = [
    "paypal", "amazon", "microsoft", "apple", "google",
    "facebook", "netflix", "instagram", "twitter", "linkedin",
    "ebay", "visa", "mastercard", "chase", "wellsfargo",
    "bankofamerica", "whatsapp", "dropbox", "adobe", "spotify"
]

# Real brand domains — these will NOT be generated
REAL_BRAND_DOMAINS = {
    "paypal.com", "amazon.com", "microsoft.com", "apple.com",
    "google.com", "facebook.com", "netflix.com", "instagram.com",
    "twitter.com", "linkedin.com", "ebay.com", "visa.com",
    "mastercard.com", "chase.com", "wellsfargo.com",
    "bankofamerica.com", "whatsapp.com", "dropbox.com",
    "adobe.com", "spotify.com"
}

# Suspicious words commonly combined with brands in phishing
SUSPICIOUS_WORDS = [
    "security", "alert", "verify", "update", "login",
    "signin", "secure", "confirm", "account", "banking",
    "support", "help", "service", "center", "care",
    "warning", "suspend", "locked", "unlock", "recover",
    "access", "auth", "portal", "manage", "validation",
    "notification", "billing", "payment", "refund", "reward"
]

# Prefixes commonly added before brand names
PREFIXES = [
    "my", "the", "official", "safe", "secure",
    "login", "signin", "verify", "access", "get",
    "new", "real", "best", "free", "online"
]

# Suffixes commonly added after brand names
SUFFIXES = [
    "login", "signin", "verify", "access", "account",
    "secure", "safe", "help", "support", "service",
    "online", "web", "portal", "app", "site"
]

# Risky TLDs used by phishers
RISKY_TLDS = [
    "tk", "ml", "ga", "cf", "gq",   # free TLDs
    "xyz", "top", "club", "site",    # cheap TLDs
    "online", "info", "biz",         # cheap TLDs
    "net", "org"                     # legitimate but misused
]

# Common paths used in phishing URLs
PHISHING_PATHS = [
    "/login", "/signin", "/verify", "/account",
    "/secure", "/update", "/confirm", "/auth",
    "/login.php", "/signin.php", "/verify.php",
    "/account/verify", "/secure/login",
    "/update/account", "/confirm/identity",
    "/login?redirect=true", "/verify?token=abc123",
    "/account/suspended", "/security/check",
    "/billing/update", "/payment/verify"
]

# Leet speak substitutions
LEET_MAP = {
    "a": ["4", "@"],
    "e": ["3"],
    "i": ["1", "!"],
    "o": ["0"],
    "s": ["5", "$"],
    "t": ["7"],
    "g": ["9"],
    "l": ["1"]
}


# ---------------- GENERATORS ----------------

def is_real_brand(domain: str) -> bool:
    """Check if domain is a real brand domain — skip these."""
    return domain.lower() in REAL_BRAND_DOMAINS


def add_path(domain: str, scheme: str = "http") -> str:
    """Add a random phishing path to domain."""
    path = random.choice(PHISHING_PATHS)
    return f"{scheme}://{domain}{path}"


def gen_brand_hyphen_word(brands, words, tlds) -> list:
    """
    Pattern: brand-word.tld
    Example: paypal-security.com, amazon-verify.net
    """
    urls = []
    for brand in brands:
        for word in words:
            for tld in random.sample(tlds, min(3, len(tlds))):
                domain = f"{brand}-{word}.{tld}"
                if not is_real_brand(domain):
                    urls.append(add_path(domain))
    return urls


def gen_word_hyphen_brand(brands, words, tlds) -> list:
    """
    Pattern: word-brand.tld
    Example: security-paypal.com, verify-amazon.net
    """
    urls = []
    for brand in brands:
        for word in random.sample(words, min(10, len(words))):
            for tld in random.sample(tlds, min(2, len(tlds))):
                domain = f"{word}-{brand}.{tld}"
                if not is_real_brand(domain):
                    urls.append(add_path(domain))
    return urls


def gen_brand_hyphen_word_hyphen_word(brands, words, tlds) -> list:
    """
    Pattern: brand-word-word.tld
    Example: paypal-security-alert.com
    This is the EXACT failing pattern
    """
    urls = []
    word_pairs = list(itertools.combinations(random.sample(words, 10), 2))
    for brand in brands:
        for w1, w2 in word_pairs[:8]:
            for tld in random.sample(tlds, min(2, len(tlds))):
                domain = f"{brand}-{w1}-{w2}.{tld}"
                if not is_real_brand(domain):
                    urls.append(add_path(domain))
    return urls


def gen_leet_speak(brands, tlds) -> list:
    """
    Pattern: leet speak brand
    Example: paypa1.com, amaz0n.com, micros0ft.com
    This is the EXACT failing pattern for amaz0n-prime.com
    """
    urls = []
    for brand in brands:
        for char, replacements in LEET_MAP.items():
            if char in brand:
                for replacement in replacements:
                    leet_brand = brand.replace(char, replacement, 1)
                    if leet_brand != brand:
                        for tld in random.sample(tlds, min(3, len(tlds))):
                            domain = f"{leet_brand}.{tld}"
                            if not is_real_brand(domain):
                                urls.append(add_path(domain))
                            # Also with path variations
                            domain2 = f"{leet_brand}-login.{tld}"
                            if not is_real_brand(domain2):
                                urls.append(add_path(domain2))
    return urls


def gen_double_letter(brands, tlds) -> list:
    """
    Pattern: doubled letter in brand
    Example: paypall.com, amazzon.com, microsofft.com
    """
    urls = []
    for brand in brands:
        for i, char in enumerate(brand):
            if char.isalpha():
                doubled = brand[:i] + char + brand[i:]
                for tld in random.sample(tlds, min(2, len(tlds))):
                    domain = f"{doubled}.{tld}"
                    if not is_real_brand(domain):
                        urls.append(add_path(domain))
    return urls


def gen_risky_tld(brands, tlds) -> list:
    """
    Pattern: real brand name but risky TLD
    Example: paypal.tk, amazon.xyz, google.gq
    """
    urls = []
    risky_only = ["tk", "ml", "ga", "cf", "gq", "xyz", "top"]
    for brand in brands:
        for tld in risky_only:
            domain = f"{brand}.{tld}"
            if not is_real_brand(domain):
                urls.append(add_path(domain))
    return urls


def gen_prefix_brand(brands, prefixes, tlds) -> list:
    """
    Pattern: prefix+brand.tld
    Example: mypaypal.com, theamazon.net, securemicrosoft.com
    """
    urls = []
    for brand in brands:
        for prefix in random.sample(prefixes, min(5, len(prefixes))):
            for tld in random.sample(tlds, min(2, len(tlds))):
                domain = f"{prefix}{brand}.{tld}"
                if not is_real_brand(domain):
                    urls.append(add_path(domain))
                # hyphenated version
                domain2 = f"{prefix}-{brand}.{tld}"
                if not is_real_brand(domain2):
                    urls.append(add_path(domain2))
    return urls


def gen_brand_suffix(brands, suffixes, tlds) -> list:
    """
    Pattern: brand+suffix.tld
    Example: paypallogin.com, amazonverify.net
    """
    urls = []
    for brand in brands:
        for suffix in random.sample(suffixes, min(5, len(suffixes))):
            for tld in random.sample(tlds, min(2, len(tlds))):
                domain = f"{brand}{suffix}.{tld}"
                if not is_real_brand(domain):
                    urls.append(add_path(domain))
                # hyphenated
                domain2 = f"{brand}-{suffix}.{tld}"
                if not is_real_brand(domain2):
                    urls.append(add_path(domain2))
    return urls


def gen_missing_letter(brands, tlds) -> list:
    """
    Pattern: brand with one letter removed
    Example: payal.com (paypal), amazn.com (amazon)
    """
    urls = []
    for brand in brands:
        if len(brand) > 4:
            for i in range(1, len(brand) - 1):
                shortened = brand[:i] + brand[i+1:]
                if shortened != brand and len(shortened) > 3:
                    for tld in random.sample(tlds, min(2, len(tlds))):
                        domain = f"{shortened}.{tld}"
                        if not is_real_brand(domain):
                            urls.append(add_path(domain))
    return urls


def gen_subdomain_brand(brands, words, tlds) -> list:
    """
    Pattern: brand.suspicious-domain.tld
    Example: paypal.phishing-site.ru
             amazon.verify-now.com
    """
    fake_domains = [
        "phishing-site", "verify-now", "secure-login",
        "account-update", "id-verify", "security-check",
        "login-verify", "suspended-account"
    ]
    urls = []
    for brand in brands:
        for fake in random.sample(fake_domains, min(4, len(fake_domains))):
            for tld in random.sample(tlds, min(2, len(tlds))):
                domain = f"{brand}.{fake}.{tld}"
                urls.append(add_path(domain))
    return urls


def gen_ip_based() -> list:
    """
    Pattern: IP address URLs with phishing paths
    Example: http://192.168.1.1/paypal/login
    """
    urls = []
    # Common IP ranges used in phishing
    ip_ranges = [
        "192.168.{}.{}",
        "10.0.{}.{}",
        "172.16.{}.{}",
        "185.220.{}.{}",
        "91.239.{}.{}"
    ]
    brand_paths = [
        "/paypal/login", "/amazon/verify", "/microsoft/update",
        "/apple/id", "/google/signin", "/facebook/login",
        "/banking/secure", "/account/verify"
    ]
    for ip_template in ip_ranges:
        for _ in range(5):
            ip = ip_template.format(
                random.randint(1, 254),
                random.randint(1, 254)
            )
            path = random.choice(brand_paths)
            urls.append(f"http://{ip}{path}")
    return urls


# ---------------- MAIN ----------------
def main():
    print("=" * 60)
    print("SYNTHETIC TYPOSQUATTING URL GENERATOR")
    print("=" * 60)
    print("Generating malicious URL patterns missing from training data")
    print("=" * 60)

    random.seed(42)
    all_urls = []

    print("\nGenerating patterns...")

    # Pattern 1: brand-word.tld
    p1 = gen_brand_hyphen_word(BRANDS, SUSPICIOUS_WORDS, RISKY_TLDS)
    print(f"  Pattern 1 (brand-word.tld)         : {len(p1):>5,} URLs")
    all_urls.extend(p1)

    # Pattern 2: word-brand.tld
    p2 = gen_word_hyphen_brand(BRANDS, SUSPICIOUS_WORDS, RISKY_TLDS)
    print(f"  Pattern 2 (word-brand.tld)         : {len(p2):>5,} URLs")
    all_urls.extend(p2)

    # Pattern 3: brand-word-word.tld (EXACT failing pattern)
    p3 = gen_brand_hyphen_word_hyphen_word(BRANDS, SUSPICIOUS_WORDS,
                                            RISKY_TLDS)
    print(f"  Pattern 3 (brand-word-word.tld)    : {len(p3):>5,} URLs")
    all_urls.extend(p3)

    # Pattern 4: leet speak (EXACT failing pattern)
    p4 = gen_leet_speak(BRANDS, RISKY_TLDS)
    print(f"  Pattern 4 (leet speak)             : {len(p4):>5,} URLs")
    all_urls.extend(p4)

    # Pattern 5: double letter
    p5 = gen_double_letter(BRANDS, RISKY_TLDS)
    print(f"  Pattern 5 (double letter)          : {len(p5):>5,} URLs")
    all_urls.extend(p5)

    # Pattern 6: risky TLD
    p6 = gen_risky_tld(BRANDS, RISKY_TLDS)
    print(f"  Pattern 6 (risky TLD)              : {len(p6):>5,} URLs")
    all_urls.extend(p6)

    # Pattern 7: prefix+brand
    p7 = gen_prefix_brand(BRANDS, PREFIXES, RISKY_TLDS)
    print(f"  Pattern 7 (prefix+brand)           : {len(p7):>5,} URLs")
    all_urls.extend(p7)

    # Pattern 8: brand+suffix
    p8 = gen_brand_suffix(BRANDS, SUFFIXES, RISKY_TLDS)
    print(f"  Pattern 8 (brand+suffix)           : {len(p8):>5,} URLs")
    all_urls.extend(p8)

    # Pattern 9: missing letter
    p9 = gen_missing_letter(BRANDS, RISKY_TLDS)
    print(f"  Pattern 9 (missing letter)         : {len(p9):>5,} URLs")
    all_urls.extend(p9)

    # Pattern 10: brand as subdomain
    p10 = gen_subdomain_brand(BRANDS, SUSPICIOUS_WORDS, RISKY_TLDS)
    print(f"  Pattern 10 (brand as subdomain)    : {len(p10):>5,} URLs")
    all_urls.extend(p10)

    # Pattern 11: IP based
    p11 = gen_ip_based()
    print(f"  Pattern 11 (IP based)              : {len(p11):>5,} URLs")
    all_urls.extend(p11)

    # Deduplicate
    all_urls = list(set(all_urls))
    print(f"\n  Total before dedup : {len(all_urls):,}")

    # Remove any that match real brand domains
    def is_safe_to_include(url):
        try:
            parsed = urlparse(url)
            ext    = tldextract.extract(parsed.netloc)
            reg    = ext.registered_domain or ""
            return reg not in REAL_BRAND_DOMAINS
        except Exception:
            return True

    all_urls = [u for u in all_urls if is_safe_to_include(u)]
    print(f"  After safety check : {len(all_urls):,}")

    # Shuffle
    random.shuffle(all_urls)

    # Create DataFrame
    df = pd.DataFrame({
        "url"  : all_urls,
        "label": "malicious"
    })

    # Save
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total URLs generated : {len(df):,}")
    print(f"Label                : malicious")
    print(f"Output               : {OUTPUT_PATH}")

    # Show samples of key patterns
    print(f"\nSample URLs by pattern:")
    print(f"\n  brand-word-word.tld (was failing):")
    for url in [u for u in all_urls if
                re.search(r'paypal-\w+-\w+\.', u)][:3]:
        print(f"    {url}")

    print(f"\n  leet speak (was failing):")
    for url in [u for u in all_urls if
                re.search(r'[a-z][0-9][a-z]', urlparse(u).netloc)][:3]:
        print(f"    {url}")

    print(f"\n  brand as subdomain:")
    for url in [u for u in all_urls if
                urlparse(u).netloc.count('.') >= 2 and
                any(b in urlparse(u).netloc.split('.')[0]
                    for b in BRANDS)][:3]:
        print(f"    {url}")

    print(f"\nNext step:")
    print(f"  Add load_synthetic_malicious() to preprocessing.py")
    print(f"  Re-run full pipeline")


if __name__ == "__main__":
    main()