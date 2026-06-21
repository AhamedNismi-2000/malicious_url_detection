"""
model_loader.py
---------------
Loads model artefacts and exposes three public methods:

  predict_url(url)                   -> dict
  predict_batch(urls)                -> list[dict]
  explain_url(url, num_features=30)  -> dict   (prediction + LIME explanation)

Extra capabilities:
  - Reverse DNS      : IP-based URLs resolved to domain name before classification
  - Unshortening     : Short URLs (bit.ly etc.) followed to real destination first
  - Brand detection  : Detects which brand is being impersonated
  - Natural language : explain_url() returns user-friendly reason sentences
  - Backup reasons   : Rule-based backup ensures at least 3 reasons always shown
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

# ── Locate project root and add scripts/ to path ─────────────────────────────
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.abspath(os.path.join(_APP_DIR, ".."))
_SCRIPTS = os.path.join(_ROOT, "scripts")

if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from feature_extraction import (
    extract_heuristic_features,
    preprocess_url_for_nlp,
    SHORTENERS,
)

# ── Path constants ────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(_ROOT, "models")
DATA_DIR   = os.path.join(_ROOT, "data")

# ── Feature names (must match training order) ─────────────────────────────────
HEURISTIC_FEATURES: list[str] = [
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
    "domain_len", "domain_digit_ratio", "max_domain_digits", "path_depth",
]

FEATURE_NAMES: list[str] = (
    HEURISTIC_FEATURES
    + [f"char_{i}" for i in range(300)]
    + [f"word_{i}" for i in range(200)]
)

# Feature index lookup for backup reasons
_FEAT_IDX = {name: i for i, name in enumerate(HEURISTIC_FEATURES)}

# Boolean/flag features — LIME treats these as categorical
_CATEGORICAL_FEATURE_NAMES: list[str] = [
    "ip_flag", "has_multi_subdomain", "risky_tld", "https_flag",
    "shortened", "sus_words", "brand_mismatch", "puny", "susp_ext",
    "suspicious_port", "brand_in_domain", "leet_in_domain",
    "brand_hyphen_suspicious",
]

# Private IP ranges
_PRIVATE_IP_RE = re.compile(
    r"^(10\.|172\.(1[6-9]|2\d|3[01])\.|192\.168\.|127\.|0\.0\.0\.0|::1)"
)

# Brand map
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

# Natural language templates
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
}

# Backup rule-based checks — ordered by importance
# (feature_name, min_value_to_trigger, sentence_template)
_BACKUP_CHECKS = [
    ("brand_in_domain",         0.5, "This site is pretending to be {brand} — the real website is {real_domain}"),
    ("brand_hyphen_suspicious",  0.5, "The domain uses a fake {brand} pattern (e.g. {brand}-security.com)"),
    ("sus_words",                0.5, "The URL contains phishing keywords such as 'security', 'alert', or 'verify'"),
    ("brand_mismatch",           0.5, "{brand} name appears in the URL but this is not the real {brand} website"),
    ("risky_tld",                0.5, "This site uses a high-risk domain ending commonly used for phishing"),
    ("leet_in_domain",           0.5, "The domain disguises a brand name using look-alike characters (e.g. amaz0n)"),
    ("ip_flag",                  0.5, "The site uses a raw IP address instead of a proper domain name"),
    ("shortened",                0.5, "This is a shortened URL hiding the real destination"),
    ("puny",                     0.5, "The domain uses international character encoding to disguise its identity"),
    ("susp_ext",                 0.5, "The URL points to a suspicious file type (e.g. .exe, .zip, .scr)"),
    ("https_flag",              -0.5, "This site does not use HTTPS — your connection may not be secure"),
    ("num_hyphens",              2.0, "The domain contains excessive hyphens which is uncommon in legitimate sites"),
    ("path_depth",               3.0, "The URL has an unusually deep path — often used to mimic legitimate sites"),
]


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


def feature_to_natural_language(
    feature: str,
    weight: float,
    value: float,
    brand_name: Optional[str] = None,
    real_domain: Optional[str] = None,
) -> Optional[str]:
    if feature.startswith("char_") or feature.startswith("word_"):
        return None
    template = _NL_TEMPLATES.get(feature)
    if not template:
        return None   # skip unmapped features entirely
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
    """
    Generate rule-based backup reasons from raw heuristic feature values.
    Used when LIME doesn't surface enough interpretable reasons.
    """
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
        # For https_flag — negative threshold means fire when value < threshold
        if threshold < 0:
            triggered = val < abs(threshold)
        else:
            triggered = val >= threshold
        if not triggered:
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

WHITELIST: frozenset[str] = frozenset({
    "163.com",
    "1rx.io",
    "2gis.com",
    "2mdn.net",
    "360.cn",
    "360safe.com",
    "360yield.com",
    "3gppnetwork.org",
    "3lift.com",
    "4dex.io",
    "a-mo.net",
    "a-msedge.net",
    "a2z.com",
    "aaplimg.com",
    "aboutads.info",
    "abovedomains.com",
    "academia.edu",
    "accuweather.com",
    "addtoany.com",
    "adform.net",
    "adgrx.com",
    "adguard-vpn.online",
    "adjust.com",
    "adobe.com",
    "adobe.io",
    "adobe.net",
    "adobedc.net",
    "adobedtm.com",
    "adriver.ru",
    "adsrvr.org",
    "adtrafficquality.google",
    "afafb.com",
    "afilias-nst.info",
    "afilias-nst.org",
    "afternic.com",
    "agkn.com",
    "agoda.com",
    "agora.io",
    "airbnb.com",
    "aiv-delivery.net",
    "akahost.net",
    "akam.net",
    "akamaiedge.net",
    "akamaihd.net",
    "akamaitech.net",
    "alibaba.com",
    "alibabadns.com",
    "alicdn.com",
    "alidns.com",
    "aliexpress.com",
    "alipaydns.com",
    "aliyun.com",
    "aliyuncs.com",
    "aliyuncsslbintl.com",
    "allaboutcookies.org",
    "allawnos.com",
    "allegro.pl",
    "amagi.tv",
    "amazon.ca",
    "amazon.co.jp",
    "amazon.co.uk",
    "amazon.com",
    "amazon.com.au",
    "amazon.com.br",
    "amazon.com.mx",
    "amazon.de",
    "amazon.dev",
    "amazon.es",
    "amazon.fr",
    "amazon.in",
    "amazon.it",
    "amazonalexa.com",
    "amazonaws.com",
    "amazontrust.com",
    "amazonvideo.com",
    "ameblo.jp",
    "amemv.com",
    "ampproject.org",
    "amzn.to",
    "ancestry.com",
    "android.com",
    "anthropic.com",
    "anydesk.com",
    "aol.com",
    "apache.org",
    "apnews.com",
    "app-analytics-services-att.com",
    "app-analytics-services.com",
    "app-measurement.com",
    "appcenter.ms",
    "apple-dns.net",
    "apple.com",
    "applovin.com",
    "appsflyer.com",
    "appsflyersdk.com",
    "appspot.com",
    "arcgis.com",
    "archive.org",
    "arubanetworks.com",
    "arxiv.org",
    "asus.com",
    "atlassian.com",
    "atlassian.net",
    "att.net",
    "autodesk.com",
    "avast.com",
    "avcdn.net",
    "avito.ru",
    "avsxappcaptiveportal.com",
    "aws.amazon.com",
    "aws.dev",
    "awsglobalaccelerator.com",
    "awswaf.com",
    "ax-msedge.net",
    "azure-devices.net",
    "azure-dns.com",
    "azure.com",
    "azure.microsoft.com",
    "azureedge.net",
    "azurefd.net",
    "azurewebsites.net",
    "b-cdn.net",
    "b-msedge.net",
    "baidu.com",
    "bamgrid.com",
    "bandcamp.com",
    "bankofamerica.com",
    "bbc.co.uk",
    "bbc.com",
    "bdydns.com",
    "behance.net",
    "beian.gov.cn",
    "berkeley.edu",
    "bestbuy.com",
    "beyondwickedmapping.org",
    "biblegateway.com",
    "bidmachine.io",
    "bidr.io",
    "bidswitch.net",
    "bild.de",
    "bilibili.com",
    "binance.com",
    "bing.com",
    "bitrix24.ru",
    "blackberry.com",
    "blackhub.team",
    "blogger.com",
    "blogspot.com",
    "bloomberg.com",
    "bluehost.com",
    "bol.com",
    "booking.com",
    "box.com",
    "branch.io",
    "brave.com",
    "braze.com",
    "britannica.com",
    "browser-intake-datadoghq.com",
    "bsky.app",
    "btloader.com",
    "bugsnag.com",
    "bunnyinfra.net",
    "businessinsider.com",
    "bx-msedge.net",
    "bytedns1.com",
    "bytefcdn-oversea.com",
    "bytefcdn-ttpus.com",
    "bytefcdn.com",
    "byteglb.com",
    "byteoversea.net",
    "ca.gov",
    "caixa.gov.br",
    "calendly.com",
    "cambridge.org",
    "canva.com",
    "capcut.com",
    "capcutapi.com",
    "cbsnews.com",
    "cdc.gov",
    "cdn-apple.com",
    "cdn-vk.ru",
    "cdn20.com",
    "cdn77.org",
    "cdnbuild.net",
    "cdngslb.com",
    "cdnhwc1.com",
    "cdnhwc2.com",
    "cdninstagram.com",
    "cdnvideo.ru",
    "change.org",
    "character.ai",
    "chatgpt.com",
    "chaturbate.com",
    "checkpoint.com",
    "chess.com",
    "chinamobile.com",
    "ci-servers.net",
    "ci-servers.org",
    "cisco.com",
    "clarity.ms",
    "claude.ai",
    "clever.com",
    "cloud.google.com",
    "cloud.microsoft",
    "cloudflare-dns.com",
    "cloudflare.com",
    "cloudflare.net",
    "cloudflareinsights.com",
    "cloudinary.com",
    "cloudns.net",
    "cloudsink.net",
    "cmediahub.ru",
    "cnbc.com",
    "cnet.com",
    "cnn.com",
    "columbia.edu",
    "comcast.com",
    "comcast.net",
    "consultant.ru",
    "contentsquare.net",
    "conviva.com",
    "cookiedatabase.org",
    "cookielaw.org",
    "cornell.edu",
    "corriere.it",
    "costco.com",
    "coupang.com",
    "coursera.org",
    "cpanel.net",
    "crashlytics.com",
    "creativecdn.com",
    "creativecommons.org",
    "criteo.net",
    "crpt.ru",
    "crwdcntrl.net",
    "cursor.sh",
    "dailymail.co.uk",
    "dailymotion.com",
    "datadoghq.com",
    "daum.net",
    "dbankcloud.com",
    "dbankcloud.ru",
    "ddnss.de",
    "debian.org",
    "deepl.com",
    "deere.com",
    "dell.com",
    "deloitte.com",
    "demdex.net",
    "deviantart.com",
    "digicert.com",
    "digitalocean.com",
    "digitaloceanspaces.com",
    "discogs.com",
    "discord.com",
    "discord.gg",
    "discord.media",
    "discordapp.com",
    "disneyplus.com",
    "disqus.com",
    "dns-parking.com",
    "dns.google",
    "dnsmadeeasy.com",
    "dnsowl.com",
    "dnspod.net",
    "docker.com",
    "docker.io",
    "docomo.ne.jp",
    "doi.org",
    "domaincontrol.com",
    "dotaplabs.net",
    "dotomi.com",
    "doubleverify.com",
    "douyincdn.com",
    "dreamhost.com",
    "drom.ru",
    "dropbox.com",
    "dropcatch.com",
    "dtkn.ru",
    "dual-s-msedge.net",
    "duckdns.org",
    "duckduckgo.com",
    "duolingo.com",
    "dv.tech",
    "dynatrace.com",
    "dyndns.org",
    "dzen.ru",
    "e2ro.com",
    "ea.com",
    "easebar.com",
    "ebay.co.uk",
    "ebay.com",
    "ebay.de",
    "ecosia.org",
    "edgcdn.net",
    "edgecdn.ru",
    "eeroup.com",
    "elasticbeanstalk.com",
    "elmundo.es",
    "elpais.com",
    "enacdn.net",
    "epa.gov",
    "epicgames.com",
    "eporner.com",
    "erome.com",
    "eset.com",
    "espn.com",
    "etsy.com",
    "eu-1-id5-sync.com",
    "eu.com",
    "europa.eu",
    "eventbrite.com",
    "everesttech.net",
    "example.com",
    "exp-tas.com",
    "expireddomains.com",
    "eye4.cn",
    "ezviz7.com",
    "ezvizlife.com",
    "f5.com",
    "facebook.com",
    "facebook.net",
    "fandom.com",
    "faphouse.com",
    "fast.com",
    "fastly-edge.com",
    "fb.com",
    "fbcdn.net",
    "fbpigeon.com",
    "fbsbx.com",
    "fda.gov",
    "featureassets.org",
    "fidelity.com",
    "figma.com",
    "firefox.com",
    "firetvcaptiveportal.com",
    "fiverr.com",
    "flashtalking.com",
    "flickr.com",
    "flipkart.com",
    "focus.de",
    "fontawesome.com",
    "forbes.com",
    "force.com",
    "forms.gle",
    "forter.com",
    "foxnews.com",
    "free.fr",
    "freepik.com",
    "frontiersin.org",
    "ft.com",
    "fwmrm.net",
    "g.co",
    "g.page",
    "gamepass.com",
    "gandi-ns.fr",
    "gandi.net",
    "garmin.com",
    "gartner.com",
    "gcdn.co",
    "genius.com",
    "geobasket.ru",
    "ggpht.com",
    "giphy.com",
    "github.com",
    "github.io",
    "githubusercontent.com",
    "gitlab.com",
    "globalsign.com",
    "globo.com",
    "gmail.com",
    "gnu.org",
    "go-mpulse.net",
    "go.com",
    "godaddy.com",
    "goodreads.com",
    "google-analytics.com",
    "google.cn",
    "google.co.uk",
    "google.com",
    "google.com.br",
    "google.com.hk",
    "google.de",
    "googleapis.com",
    "googleblog.com",
    "googledomains.com",
    "googletagmanager.com",
    "googletagservices.com",
    "googleusercontent.com",
    "googlevideo.com",
    "googlezip.net",
    "goskope.com",
    "gosuslugi.ru",
    "grammarly.com",
    "grammarly.io",
    "gravatar.com",
    "gstatic.com",
    "gtld-servers.net",
    "gumgum.com",
    "gvt1.com",
    "gvt2.com",
    "gwfb.net",
    "harvard.edu",
    "hath.network",
    "hbr.org",
    "hcaptcha.com",
    "healthline.com",
    "herokuapp.com",
    "herokudns.com",
    "heytapdl.com",
    "heytapmobi.com",
    "heytapmobile.com",
    "hichina.com",
    "hicloud.com",
    "hicloudcam.com",
    "hihonorcloud.com",
    "hilton.com",
    "hm.com",
    "homedepot.com",
    "hostgator.com",
    "hostgator.com.br",
    "hp.com",
    "hstgr.net",
    "huawei.com",
    "hubspot.com",
    "huffpost.com",
    "hugedomains.com",
    "ibm.com",
    "ibyteimg.com",
    "icloud-content.com",
    "icloud.com",
    "id5-sync.com",
    "ieee.org",
    "ietf.org",
    "iiko.it",
    "ikea.com",
    "imcmdb.net",
    "imdb.com",
    "imgsmail.ru",
    "imgur.com",
    "immedia-semi.com",
    "impervadns.net",
    "imrworldwide.com",
    "indeed.com",
    "independent.co.uk",
    "indiatimes.com",
    "infobae.com",
    "inmobi.com",
    "inner-active.mobi",
    "instagram.com",
    "intel.com",
    "intercom.io",
    "internetwarriors.net",
    "intuit.com",
    "investopedia.com",
    "ioref.io",
    "ip-api.com",
    "ipify.org",
    "ipv4only.arpa",
    "irs.gov",
    "iso.org",
    "issuu.com",
    "it.com",
    "ivi.ru",
    "jd.com",
    "jetbrains.com",
    "jimdo.com",
    "jomodns.com",
    "jotform.com",
    "jquery.com",
    "jsdelivr.net",
    "kaspersky-labs.com",
    "kaspersky.com",
    "keenetic.io",
    "kick.com",
    "kickstarter.com",
    "klaviyo.com",
    "kleinanzeigen.de",
    "kontur.ru",
    "ks-cdn.com",
    "kslawin.com",
    "ksyuncdn.com",
    "kubernetes.io",
    "kueezrtb.com",
    "kunluncan.com",
    "kwai-pro.com",
    "kwai.com",
    "kwai.net",
    "kwaipros.com",
    "kwcdn.com",
    "latimes.com",
    "launchdarkly.com",
    "launchpad.net",
    "lefigaro.fr",
    "leiniao.com",
    "lemonde.fr",
    "lencr.org",
    "lenovo.com",
    "lgtvcommon.com",
    "liadm.com",
    "libp2p.direct",
    "licdn.com",
    "life360.com",
    "liftoff.io",
    "lijit.com",
    "line.me",
    "linkedin.com",
    "linktr.ee",
    "linode.com",
    "list-manage.com",
    "live-video.net",
    "live.com",
    "live.net",
    "livejournal.com",
    "ln-msedge.net",
    "loc.gov",
    "lowes.com",
    "lsrelayaccess.com",
    "macromedia.com",
    "mail.ru",
    "mailchi.mp",
    "mailchimp.com",
    "mailinabox.email",
    "mangosip.ru",
    "markmonitor.com",
    "marriott.com",
    "mayoclinic.org",
    "mcafee.com",
    "mckinsey.com",
    "mdpi.com",
    "me.com",
    "media-amazon.com",
    "mediafire.com",
    "mediatek.com",
    "medium.com",
    "mega.co.nz",
    "meraki.com",
    "mercadolibre.com.ar",
    "mercadolivre.com.br",
    "merriam-webster.com",
    "mhverifier.ru",
    "mi.com",
    "microsoft.com",
    "microsoftonline.com",
    "miit.gov.cn",
    "mikrotik.com",
    "mit.edu",
    "miui.com",
    "miwifi.com",
    "mlb.com",
    "moe.video",
    "moloco.com",
    "moneycontrol.com",
    "mozilla.com",
    "mozilla.net",
    "mozilla.org",
    "msedge.net",
    "msftauth.net",
    "msftconnecttest.com",
    "msftncsi.com",
    "msidentity.com",
    "msn.com",
    "mtgglobals.com",
    "mts.ru",
    "my.com",
    "mybluehost.me",
    "myfritz.net",
    "myhuaweicloud.com",
    "mynetname.net",
    "myqcloud.com",
    "myshopify.com",
    "myspace.com",
    "mysql.com",
    "mzstatic.com",
    "name-services.com",
    "name.com",
    "namebrightdns.com",
    "nasa.gov",
    "nationalgeographic.com",
    "nature.com",
    "naver.com",
    "nbcnews.com",
    "ndtv.com",
    "nease.net",
    "nel.goog",
    "nelreports.net",
    "netangels.ru",
    "netcraze.io",
    "netease.com",
    "netflix.com",
    "netflix.net",
    "netgear.com",
    "networkadvertising.org",
    "nextcloud.com",
    "nextlgsdp.com",
    "nexusmods.com",
    "nflximg.com",
    "nflxso.net",
    "nflxvideo.net",
    "ngenix.net",
    "nginx.com",
    "nginx.org",
    "nic.direct",
    "nic.io",
    "nic.network",
    "nic.ru",
    "nih.gov",
    "nike.com",
    "nikkei.com",
    "nintendo.com",
    "nintendo.net",
    "nist.gov",
    "nmrodam.com",
    "no-ip.com",
    "noaa.gov",
    "nominetdns.uk",
    "note.com",
    "npmjs.com",
    "npr.org",
    "nstld.com",
    "ntp.org",
    "nvidia.com",
    "nypost.com",
    "nytimes.com",
    "odoo.com",
    "office.com",
    "office.net",
    "office365.com",
    "ok.ru",
    "okcdn.ru",
    "okta.com",
    "omtrdc.net",
    "on.aws",
    "one.one",
    "onelink.me",
    "onesignal.com",
    "onet.pl",
    "onetag-sys.com",
    "onetrust.com",
    "online-metrix.net",
    "onlyfans.com",
    "openai.com",
    "opendns.com",
    "openstreetmap.org",
    "opera-api.com",
    "opera.com",
    "optimizely.com",
    "oracle.com",
    "oraclecloud.com",
    "orderbox-dns.com",
    "otto.de",
    "oup.com",
    "outlook.com",
    "ovh.net",
    "ovscdns.com",
    "ox.ac.uk",
    "oxylabs.io",
    "ozon.ru",
    "ozone.ru",
    "pages.dev",
    "palmplaystore.com",
    "paloaltonetworks.com",
    "pangle.io",
    "patreon.com",
    "paypal.com",
    "pbs.org",
    "pccc.com",
    "people.com",
    "perplexity.ai",
    "pexels.com",
    "php.net",
    "pinimg.com",
    "pinterest.com",
    "pixabay.com",
    "pixiv.net",
    "pki.goog",
    "playfabapi.com",
    "playrix.com",
    "playstation.com",
    "playstation.net",
    "plesk.com",
    "poki.com",
    "pornhub.com",
    "presage.io",
    "primevideo.com",
    "princeton.edu",
    "privacy-mgmt.com",
    "prnewswire.com",
    "prodregistryv2.org",
    "pushy.io",
    "pv-cdn.net",
    "px-cloud.net",
    "pypi.org",
    "python.org",
    "qlivecdn.com",
    "qq.com",
    "qualtrics.com",
    "quickconnect.to",
    "quizlet.com",
    "quora.com",
    "rackspace.com",
    "rackspace.net",
    "rakuten.co.jp",
    "rakuten.com",
    "rambler.ru",
    "rbxcdn.com",
    "readthedocs.io",
    "recaptcha.net",
    "reddit.com",
    "redhat.com",
    "reg.ru",
    "registrar-servers.com",
    "repubblica.it",
    "researchgate.net",
    "resolver.arpa",
    "reuters.com",
    "richaudience.com",
    "ring.com",
    "ripn.net",
    "rlcdn.com",
    "roblox.com",
    "rocket-cdn.com",
    "roku.com",
    "root-servers.net",
    "rt.ru",
    "run.app",
    "rutube.ru",
    "ryanair.com",
    "rzone.de",
    "safebrowsing.apple",
    "sagepub.com",
    "salesforce.com",
    "samsung.com",
    "samsungacr.com",
    "samsungapps.com",
    "samsungcloud.com",
    "samsungcloudsolution.com",
    "samsungcloudsolution.net",
    "samsungosp.com",
    "samsungqbe.com",
    "sberbank.ru",
    "sc-cdn.net",
    "sc-gw.com",
    "scdn.co",
    "sciencedirect.com",
    "scribd.com",
    "sedo.com",
    "seedtag.com",
    "segment.io",
    "selectel.ru",
    "sendgrid.com",
    "sentry.io",
    "service.gov.uk",
    "seznam.cz",
    "sfx.ms",
    "shalltry.com",
    "share-dns.com",
    "share.google",
    "sharepoint.com",
    "shein.com",
    "shifen.com",
    "shopee.co.id",
    "shopee.com.br",
    "shopeemobile.com",
    "shopify.com",
    "shopifysvc.com",
    "sina.com.cn",
    "skyhigh.cloud",
    "skype.com",
    "slack.com",
    "slideshare.net",
    "smaato.net",
    "smartthings.com",
    "smilewanted.com",
    "snapchat.com",
    "snapkit.com",
    "sohu.com",
    "sophos.com",
    "soundcloud.com",
    "sourceforge.net",
    "spaceweb.pro",
    "speedtest.net",
    "spiegel.de",
    "spo-msedge.net",
    "spotify.com",
    "spotifycdn.com",
    "spov-msedge.net",
    "springer.com",
    "squarespace.com",
    "squarespacedns.com",
    "ssl-images-amazon.com",
    "stackadapt.com",
    "stackoverflow.com",
    "stanford.edu",
    "starlink.com",
    "state.gov",
    "static.microsoft",
    "statista.com",
    "stbid.ru",
    "steamcommunity.com",
    "steampowered.com",
    "steamserver.net",
    "steamstatic.com",
    "stripchat.com",
    "stripe.com",
    "substack.com",
    "supercell.com",
    "supertms.com",
    "surveymonkey.com",
    "svc.ms",
    "synology.com",
    "t-mobile.com",
    "t-msedge.net",
    "t-online.de",
    "t.me",
    "tandfonline.com",
    "taobao.com",
    "tapad.com",
    "target.com",
    "tawk.to",
    "tbcache.com",
    "teads.tv",
    "teamviewer.com",
    "techcrunch.com",
    "ted.com",
    "telecid.ru",
    "telegram.me",
    "telegram.org",
    "telegraph.co.uk",
    "telekom.de",
    "telekom.net",
    "telephony.goog",
    "temu.com",
    "tencent-cloud.net",
    "tencent.com",
    "theatlantic.com",
    "theconversation.com",
    "theguardian.com",
    "themeforest.net",
    "thenai.org",
    "theverge.com",
    "threads.com",
    "tiktok.com",
    "tiktokcdn-eu.com",
    "tiktokcdn-us.com",
    "tiktokcdn.com",
    "tiktokpangle.us",
    "tiktokrow-cdn.com",
    "tiktokv.com",
    "tiktokv.eu",
    "tiktokv.us",
    "tiktokw.us",
    "time.com",
    "timeweb.ru",
    "tm-azurefd.net",
    "tp-link.com",
    "tplinkcloud.com",
    "tplinknbu.com",
    "tradingview.com",
    "trafficmanager.net",
    "trbcdn.net",
    "trendmicro.com",
    "tripadvisor.com",
    "triplinkintl.com",
    "trueconf.net",
    "trustpilot.com",
    "tsyndicate.com",
    "ttdns2.com",
    "ttvnw.net",
    "tumblr.com",
    "turn.com",
    "twilio.com",
    "twimg.com",
    "twitch.tv",
    "twitter.com",
    "typeform.com",
    "typekit.net",
    "uber.com",
    "ubi.com",
    "ubnt.com",
    "ubuntu.com",
    "udemy.com",
    "ui-dns.com",
    "ui.com",
    "uk.com",
    "umich.edu",
    "un.org",
    "unesco.org",
    "unity3d.com",
    "unpkg.com",
    "unsplash.com",
    "uol.com.br",
    "ups.com",
    "usatoday.com",
    "usda.gov",
    "userapi.com",
    "usercontent.goog",
    "usgovcloudapi.net",
    "usps.com",
    "vecdnlb.com",
    "vedcdnlb.com",
    "vedsalb.com",
    "vercel-dns-016.com",
    "vercel-dns-017.com",
    "vercel.app",
    "verisign.com",
    "viber.com",
    "vidaahub.com",
    "vimeo.com",
    "virginm.net",
    "visualstudio.com",
    "vivo.com.cn",
    "vivoglobal.com",
    "vk-analytics.ru",
    "vk.com",
    "vk.ru",
    "vkontakte.ru",
    "vkuser.net",
    "vkuserphoto.ru",
    "volcfcdndvs.com",
    "vungle.com",
    "w3.org",
    "wa.me",
    "wac-msedge.net",
    "walmart.com",
    "washington.edu",
    "washingtonpost.com",
    "wattpad.com",
    "wb.ru",
    "wbbasket.ru",
    "wbx2.com",
    "weather.com",
    "webempresa.eu",
    "webex.com",
    "webmd.com",
    "weebly.com",
    "weforum.org",
    "weibo.com",
    "welt.de",
    "whatsapp.com",
    "whatsapp.net",
    "whecloud.com",
    "whitehouse.gov",
    "who.int",
    "wikimedia.org",
    "wikipedia.org",
    "wildberries.ru",
    "wiley.com",
    "windows.com",
    "windows.net",
    "windowsupdate.com",
    "wired.com",
    "withgoogle.com",
    "wix.com",
    "wixsite.com",
    "wordpress.com",
    "wordpress.org",
    "workers.dev",
    "worldbank.org",
    "worldnic.com",
    "wp.com",
    "wp.pl",
    "wpguardian.com",
    "wpguardian.io",
    "wps.com",
    "wsdvs.com",
    "wsj.com",
    "wswebcdn.com",
    "www.gov.br",
    "www.gov.uk",
    "wyzecam.com",
    "x.com",
    "xboxlive.com",
    "xcal.tv",
    "xerox.com",
    "xhcdn.com",
    "xiaomi.com",
    "xiaomi.net",
    "ya.ru",
    "yahoo.co.jp",
    "yahoo.com",
    "yahoodns.net",
    "yandex.com",
    "yandex.com.tr",
    "yandex.net",
    "yandex.ru",
    "yandexcloud.net",
    "yccdn.ru",
    "yellowblue.io",
    "yelp.com",
    "yieldmo.com",
    "youku.com",
    "youronlinechoices.com",
    "youtu.be",
    "youtube-nocookie.com",
    "youtube.com",
    "ys7.com",
    "ytimg.com",
    "yximgs.com",
    "zdnscloud.cn",
    "zendesk.com",
    "zenecn.net",
    "zhihu.com",
    "zillow.com",
    "zoho.com",
    "zoom.com",
    "zoom.us",
})


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
        processed  = preprocess_url_for_nlp(url)
        char_dense = self.vec_char.transform([processed]).toarray().flatten()
        word_dense = self.vec_word.transform([processed]).toarray().flatten()
        return np.concatenate([heuristic_scaled, char_dense, word_dense])

    def _classify(self, url: str) -> dict:
        try:
            fv    = self._feature_vector(url)
            proba = float(self.model.predict_proba(fv.reshape(1, -1))[0][1])
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

        if self._registered_domain(url) in WHITELIST:
            return {
                "url": original_url, "prediction": "BENIGN",
                "confidence": 100.0,
                "threshold": round(self.threshold * 100, 2),
                "source": "whitelist",
            }

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
        if base["source"] in ("whitelist", "invalid"):
            return {**base, "explanation": [], "reasons": []}

        brand_name   = base.get("brand_detected")
        real_domain  = base.get("real_domain")
        classify_url = base.get("unshortened") or url

        # Always extract raw heuristic features for backup reasons
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
                    nl = feature_to_natural_language(
                        feat_name, weight, feat_val,
                        brand_name, real_domain
                    )
                    if nl and nl not in seen:
                        seen.add(nl)
                        reasons.append(nl)

            explanation.sort(key=lambda x: abs(x["weight"]), reverse=True)

            # Keep top 3 from LIME
            top_reasons = reasons[:3]

            # Fill remaining slots with rule-based backup reasons
            if len(top_reasons) < 3:
                backup = _build_backup_reasons(
                    raw_heuristic, brand_name, real_domain,
                    set(top_reasons), 3 - len(top_reasons)
                )
                top_reasons.extend(backup)

            return {**base, "explanation": explanation, "reasons": top_reasons}

        except Exception as exc:
            # On LIME failure — use pure rule-based reasons
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
    # If starts with digit or operator — unrecognised condition
    if first and (first[0].isdigit() or first[0] in "-+."):
        return ""
    return first


# ── Module-level singleton ────────────────────────────────────────────────────
classifier = URLClassifier()