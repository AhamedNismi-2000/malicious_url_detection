#!/usr/bin/env python3
"""
update_whitelist.py
-------------------
Downloads the Tranco top 1M list and extracts the top 1000 domains.
Then updates the WHITELIST in app/model_loader.py automatically.

Run from project root:
  python scripts/update_whitelist.py

Requirements:
  pip install requests
"""

import os
import io
import re
import sys
import zipfile
import requests

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_LOADER     = os.path.join(BASE_DIR, "app", "model_loader.py")
WHITELIST_CACHE  = os.path.join(BASE_DIR, "data", "tranco_top1000.txt")

TRANCO_URL       = "https://tranco-list.eu/top-1m.csv.zip"
TOP_N            = 1000

# ── Domains to always exclude from whitelist (even if in Tranco top 1000) ────
# Ad networks, trackers, known bad actors
EXCLUDE = {
    # Ad networks
    "doubleclick.net", "googlesyndication.com", "googleadservices.com",
    "adnxs.com", "rubiconproject.com", "openx.net", "pubmatic.com",
    "casalemedia.com", "criteo.com", "taboola.com", "outbrain.com",
    "revcontent.com", "mgid.com", "adsafeprotected.com", "moatads.com",
    "amazon-adsystem.com", "media.net", "advertising.com", "adroll.com",
    "sharethrough.com", "33across.com", "smartadserver.com",
    # Trackers
    "scorecardresearch.com", "quantserve.com", "comscore.com",
    "clicktale.net", "hotjar.com", "fullstory.com", "loggly.com",
    "newrelic.com", "nr-data.net", "segment.com", "mixpanel.com",
    "amplitude.com", "heap.io", "mouseflow.com", "luckyorange.com",
    # CDN/infrastructure only — not user-facing
    "fastly.net", "akamaized.net", "cloudfront.net", "edgekey.net",
    "akadns.net", "akamai.net", "edgesuite.net",
    # Known risky despite popularity
    "t.co",        # Twitter shortener
    "bit.ly",      # URL shortener
    "tinyurl.com", # URL shortener
    "goo.gl",      # URL shortener (deprecated)
    "ow.ly",       # URL shortener
}

# ── Domains to always include (your original 54 trusted domains) ──────────────
ALWAYS_INCLUDE = {
    "google.com", "youtube.com", "facebook.com", "twitter.com", "instagram.com",
    "linkedin.com", "wikipedia.org", "amazon.com", "apple.com", "microsoft.com",
    "github.com", "stackoverflow.com", "reddit.com", "netflix.com", "spotify.com",
    "dropbox.com", "slack.com", "zoom.us", "adobe.com", "salesforce.com",
    "paypal.com", "ebay.com", "walmart.com", "target.com", "bestbuy.com",
    "nytimes.com", "bbc.com", "cnn.com", "theguardian.com", "reuters.com",
    "harvard.edu", "mit.edu", "stanford.edu", "coursera.org", "udemy.com",
    "python.org", "npmjs.com", "pypi.org", "docker.com", "kubernetes.io",
    "cloudflare.com", "aws.amazon.com", "azure.microsoft.com", "cloud.google.com",
    "stripe.com", "twilio.com", "sendgrid.com", "mailchimp.com", "hubspot.com",
    "wordpress.com", "shopify.com", "squarespace.com", "wix.com", "weebly.com",
    "anthropic.com", "openai.com",
}


# ── Download Tranco list ───────────────────────────────────────────────────────

def download_tranco(top_n: int) -> list[str]:
    """Download Tranco top-1M ZIP and extract top N domains."""
    print(f"Downloading Tranco top-1M list from {TRANCO_URL}...")
    try:
        resp = requests.get(TRANCO_URL, timeout=60, stream=True)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  ERROR: Could not download Tranco list: {e}")
        print("  Check your internet connection and try again.")
        sys.exit(1)

    total = int(resp.headers.get("content-length", 0))
    downloaded = b""
    for chunk in resp.iter_content(chunk_size=65536):
        downloaded += chunk
        if total:
            pct = len(downloaded) / total * 100
            print(f"  Downloading... {pct:.0f}%", end="\r")
    print(f"  Downloaded {len(downloaded)/1e6:.1f} MB          ")

    print("  Extracting...")
    with zipfile.ZipFile(io.BytesIO(downloaded)) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            lines = f.read().decode("utf-8").splitlines()

    domains = []
    for line in lines[:top_n * 2]:   # read extra to account for exclusions
        parts = line.strip().split(",")
        if len(parts) >= 2:
            domains.append(parts[1].strip().lower())

    print(f"  Read {len(domains)} domains from Tranco list")
    return domains


# ── Filter and build final whitelist ──────────────────────────────────────────

def build_whitelist(tranco_domains: list[str], top_n: int) -> set[str]:
    """Filter Tranco domains and combine with always-include list."""
    filtered = []
    excluded_log = []

    for domain in tranco_domains:
        if domain in EXCLUDE:
            excluded_log.append(domain)
            continue
        if not domain or "." not in domain:
            continue
        # Skip domains with risky TLDs even in top 1000
        tld = domain.rsplit(".", 1)[-1]
        if tld in {"tk", "ml", "ga", "cf", "gq", "xyz", "top", "pw"}:
            excluded_log.append(f"{domain} (risky TLD)")
            continue
        filtered.append(domain)
        if len(filtered) >= top_n:
            break

    print(f"\n  Excluded {len(excluded_log)} domains:")
    for d in excluded_log[:10]:
        print(f"    - {d}")
    if len(excluded_log) > 10:
        print(f"    ... and {len(excluded_log)-10} more")

    # Combine with always-include
    final = set(filtered) | ALWAYS_INCLUDE
    print(f"\n  Tranco top-{top_n} (filtered) : {len(filtered)}")
    print(f"  Always-include              : {len(ALWAYS_INCLUDE)}")
    print(f"  Final whitelist size        : {len(final)}")
    return final


# ── Update model_loader.py ────────────────────────────────────────────────────

def update_model_loader(whitelist: set[str]):
    """Replace the WHITELIST frozenset in model_loader.py."""
    with open(MODEL_LOADER, "r", encoding="utf-8") as f:
        content = f.read()

    # Build new whitelist block
    sorted_domains = sorted(whitelist)
    lines = [f'    "{d}",' for d in sorted_domains]
    domains_str = "\n".join(lines)

    new_block = (
        "WHITELIST: frozenset[str] = frozenset({\n"
        + domains_str + "\n})"
    )

    # Replace existing WHITELIST block using regex
    pattern = r"WHITELIST:\s*frozenset\[str\]\s*=\s*frozenset\(\{.*?\}\)"
    if not re.search(pattern, content, flags=re.DOTALL):
        print("\n  ERROR: Could not find WHITELIST block in model_loader.py")
        print("  Please update it manually.")
        return False

    new_content = re.sub(pattern, new_block, content, flags=re.DOTALL)

    with open(MODEL_LOADER, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"\n  model_loader.py updated successfully")
    return True


# ── Save whitelist cache ──────────────────────────────────────────────────────

def save_cache(whitelist: set[str]):
    os.makedirs(os.path.dirname(WHITELIST_CACHE), exist_ok=True)
    with open(WHITELIST_CACHE, "w", encoding="utf-8") as f:
        for domain in sorted(whitelist):
            f.write(domain + "\n")
    print(f"  Whitelist cached: {WHITELIST_CACHE}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 55)
    print("  Whitelist Updater — Tranco Top 1000")
    print("=" * 55 + "\n")

    # Download
    tranco_domains = download_tranco(TOP_N)

    # Filter and build
    whitelist = build_whitelist(tranco_domains, TOP_N)

    # Preview top 20
    print("\n  Sample of whitelist (first 20):")
    for d in sorted(whitelist)[:20]:
        print(f"    {d}")

    # Confirm before updating
    print(f"\n  This will update WHITELIST in app/model_loader.py")
    confirm = input("  Proceed? (y/n): ").strip().lower()
    if confirm != "y":
        print("  Aborted.")
        return

    # Update model_loader.py
    success = update_model_loader(whitelist)
    if not success:
        return

    # Save cache
    save_cache(whitelist)

    print("\n" + "=" * 55)
    print("  Done!")
    print(f"  Whitelist updated: {len(whitelist)} trusted domains")
    print("  Restart app/app.py for changes to take effect")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()