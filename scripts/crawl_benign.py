#!/usr/bin/env python3
"""
crawl_benign.py
---------------
Crawls top Alexa domains to collect real full benign URLs.

Why this is needed:
  Alexa dataset only has bare domains (google.com)
  Real benign URLs look like: https://accounts.google.com/login
  Model needs to see benign URLs with subdomains, paths, query params

What this script does:
  1. Reads top 10,000 domains from Alexa top-1m.csv
  2. For each domain visits https://www.domain.com
  3. Extracts all internal links (up to MAX_URLS_PER_DOMAIN)
  4. Saves only URLs that return HTTP 200
  5. Outputs data/raw/benign_crawled.csv

Output format:
  url,label
  https://accounts.google.com/login,benign
  https://github.com/login,benign
  ...

Expected results:
  ~100,000 - 200,000 real benign URLs
  Runtime: 2-3 hours with 32 threads
"""

import os
import re
import time
import random
import logging
import threading
import queue
from urllib.parse import urlparse, urljoin, quote
from collections import defaultdict

import requests
import pandas as pd
import tldextract
from bs4 import BeautifulSoup
from tqdm import tqdm
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")
ALEXA_PATH  = os.path.join(RAW_DIR, "top-1m.csv")
OUTPUT_PATH = os.path.join(RAW_DIR, "benign_crawled.csv")
LOG_PATH    = os.path.join(BASE_DIR, "data", "raw", "crawl.log")

# ---------------- CONFIG ----------------
TOP_N_DOMAINS       = 10_000   # crawl top 10K domains
MAX_URLS_PER_DOMAIN = 20       # max URLs to collect per domain
N_THREADS           = 32       # parallel threads
REQUEST_TIMEOUT     = 8        # seconds per request
MAX_RETRIES         = 2        # retries per domain
SAVE_EVERY          = 500      # save progress every N domains
MIN_URL_LENGTH      = 10       # ignore very short URLs
MAX_URL_LENGTH      = 300      # ignore very long URLs

# Domains to skip — these cause issues or are not useful
SKIP_DOMAINS = {
    "google.com", "facebook.com", "youtube.com", "twitter.com",
    "instagram.com", "tiktok.com", "whatsapp.com", "t.me",
    "doubleclick.net", "googletagmanager.com", "googleapis.com",
    "gstatic.com", "cloudflare.com", "akamai.com", "fastly.com",
    "analytics.google.com"
}

# Extensions to skip — not useful as benign URL examples
SKIP_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".ico", ".webp",
    ".mp4", ".mp3", ".avi", ".mov", ".pdf", ".zip", ".exe",
    ".css", ".js", ".woff", ".woff2", ".ttf", ".eot",
    ".xml", ".json", ".csv", ".txt"
}

# User agents to rotate — avoid being blocked
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) "
    "Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
]

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8", mode="w"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------- HELPERS ----------------
EXTRACTOR = tldextract.TLDExtract(cache_dir=None, suffix_list_urls=None)


def get_registered_domain(url: str) -> str:
    try:
        ext = EXTRACTOR(urlparse(url).netloc)
        return ext.registered_domain or ""
    except Exception:
        return ""


def is_internal_url(url: str, base_domain: str) -> bool:
    """Check if URL belongs to the same registered domain."""
    try:
        url_domain = get_registered_domain(url)
        return url_domain == base_domain
    except Exception:
        return False


def is_valid_url(url: str) -> bool:
    """Check if URL is worth collecting."""
    try:
        if not url or not isinstance(url, str):
            return False
        if len(url) < MIN_URL_LENGTH or len(url) > MAX_URL_LENGTH:
            return False
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False
        if not parsed.netloc:
            return False
        # Skip URLs with unwanted extensions
        path_lower = parsed.path.lower()
        if any(path_lower.endswith(ext) for ext in SKIP_EXTENSIONS):
            return False
        # Skip URLs with fragments only
        if not parsed.path and not parsed.query and parsed.fragment:
            return False
        return True
    except Exception:
        return False


def normalize_url(url: str) -> str:
    """Normalize URL for consistency."""
    try:
        url = url.strip()
        # Remove fragment
        if "#" in url:
            url = url[:url.index("#")]
        # Remove trailing slash for non-root URLs
        parsed = urlparse(url)
        if parsed.path and parsed.path != "/" and url.endswith("/"):
            url = url[:-1]
        return url
    except Exception:
        return url


def extract_links(html: str, base_url: str, base_domain: str) -> list:
    """Extract all internal links from HTML page."""
    links = set()
    try:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup.find_all("a", href=True):
            href = tag["href"].strip()
            if not href or href.startswith(("javascript:", "mailto:",
                                             "tel:", "#")):
                continue
            # Convert relative to absolute URL
            full_url = urljoin(base_url, href)
            full_url = normalize_url(full_url)

            if (is_valid_url(full_url) and
                    is_internal_url(full_url, base_domain)):
                links.add(full_url)
    except Exception:
        pass
    return list(links)


# ---------------- CRAWLER ----------------
class DomainCrawler:
    def __init__(self):
        self.session = requests.Session()
        self.session.max_redirects = 5

    def get_page(self, url: str, timeout: int = REQUEST_TIMEOUT):
        """Fetch a page and return response."""
        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Connection": "keep-alive"
        }
        response = self.session.get(
            url,
            headers=headers,
            timeout=timeout,
            verify=False,
            allow_redirects=True,
            stream=False
        )
        return response

    def crawl_domain(self, domain: str) -> list:
        """
        Crawl a single domain and return list of valid URLs found.
        Returns empty list if domain is unreachable.
        """
        collected_urls = []
        base_url = f"https://www.{domain}"

        # Try https://www. first, then https://, then http://www.
        base_urls_to_try = [
            f"https://www.{domain}",
            f"https://{domain}",
            f"http://www.{domain}"
        ]

        html = None
        final_base_url = None

        for attempt_url in base_urls_to_try:
            try:
                response = self.get_page(attempt_url)
                if response.status_code == 200:
                    html = response.text
                    final_base_url = response.url
                    # Add the landing URL itself
                    landing = normalize_url(response.url)
                    if is_valid_url(landing):
                        collected_urls.append(landing)
                    break
                elif response.status_code in (301, 302, 303, 307, 308):
                    # Redirect handled by requests
                    html = response.text
                    final_base_url = response.url
                    break
            except Exception:
                continue

        if not html or not final_base_url:
            return []

        # Extract internal links from homepage
        registered = get_registered_domain(final_base_url) or domain
        links = extract_links(html, final_base_url, registered)

        # Shuffle to get variety
        random.shuffle(links)

        # Visit up to MAX_URLS_PER_DOMAIN links
        for link in links[:MAX_URLS_PER_DOMAIN * 2]:
            if len(collected_urls) >= MAX_URLS_PER_DOMAIN:
                break
            try:
                resp = self.get_page(link, timeout=5)
                if resp.status_code == 200:
                    final_link = normalize_url(resp.url)
                    if (is_valid_url(final_link) and
                            is_internal_url(final_link, registered) and
                            final_link not in collected_urls):
                        collected_urls.append(final_link)
            except Exception:
                continue

        return collected_urls[:MAX_URLS_PER_DOMAIN]


# ---------------- THREAD WORKER ----------------
def worker(domain_queue: queue.Queue,
           results: list,
           results_lock: threading.Lock,
           pbar: tqdm,
           stats: dict,
           stats_lock: threading.Lock):
    """Worker thread — crawls domains from queue."""
    crawler = DomainCrawler()

    while True:
        try:
            domain = domain_queue.get_nowait()
        except queue.Empty:
            break

        try:
            urls = crawler.crawl_domain(domain)
            if urls:
                with results_lock:
                    results.extend([(u, "benign") for u in urls])
                with stats_lock:
                    stats["successful"] += 1
                    stats["total_urls"] += len(urls)
            else:
                with stats_lock:
                    stats["failed"] += 1
        except Exception as e:
            with stats_lock:
                stats["failed"] += 1

        pbar.update(1)
        domain_queue.task_done()


# ---------------- SAVE PROGRESS ----------------
def save_results(results: list, output_path: str, mode: str = "w"):
    """Save crawled URLs to CSV."""
    if not results:
        return
    df = pd.DataFrame(results, columns=["url", "label"])
    df = df.drop_duplicates(subset=["url"])
    df.to_csv(output_path, index=False, encoding="utf-8", mode=mode,
              header=(mode == "w"))
    return len(df)


# ---------------- MAIN ----------------
def main():
    logger.info("BENIGN URL CRAWLER")
    logger.info("=" * 60)
    logger.info(f"Top domains to crawl  : {TOP_N_DOMAINS:,}")
    logger.info(f"Max URLs per domain   : {MAX_URLS_PER_DOMAIN}")
    logger.info(f"Threads               : {N_THREADS}")
    logger.info(f"Expected output       : "
                f"{TOP_N_DOMAINS * MAX_URLS_PER_DOMAIN // 4:,} - "
                f"{TOP_N_DOMAINS * MAX_URLS_PER_DOMAIN // 2:,} URLs")
    logger.info(f"Expected runtime      : 2-3 hours")
    logger.info("=" * 60)

    # Check BeautifulSoup is installed
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logger.error("BeautifulSoup not installed.")
        logger.error("Run: pip install beautifulsoup4")
        return

    # Load Alexa domains
    if not os.path.exists(ALEXA_PATH):
        logger.error(f"Alexa file not found: {ALEXA_PATH}")
        return

    logger.info(f"\nLoading top {TOP_N_DOMAINS:,} domains from Alexa...")
    alexa_df = pd.read_csv(ALEXA_PATH, header=None,
                           names=["rank", "domain"],
                           dtype=str, nrows=TOP_N_DOMAINS)
    alexa_df["domain"] = alexa_df["domain"].str.strip().str.lower()

    # Filter out skip domains
    domains = [
        d for d in alexa_df["domain"].tolist()
        if d and d not in SKIP_DOMAINS
        and "." in d
    ]
    domains = domains[:TOP_N_DOMAINS]
    logger.info(f"Domains to crawl: {len(domains):,}")

    # Check if partial results exist — resume from where we left off
    already_crawled = set()
    file_mode = "w"
    if os.path.exists(OUTPUT_PATH):
        try:
            existing = pd.read_csv(OUTPUT_PATH)
            already_crawled = set(
                get_registered_domain(u)
                for u in existing["url"].tolist()
                if u
            )
            if already_crawled:
                logger.info(f"Resuming — already have URLs from "
                            f"{len(already_crawled):,} domains")
                domains = [d for d in domains
                           if d not in already_crawled]
                file_mode = "a"  # append to existing file
                logger.info(f"Remaining domains: {len(domains):,}")
        except Exception:
            pass

    if not domains:
        logger.info("All domains already crawled.")
        return

    # Set up queue and shared state
    domain_queue = queue.Queue()
    for d in domains:
        domain_queue.put(d)

    results      = []
    results_lock = threading.Lock()
    stats        = {"successful": 0, "failed": 0, "total_urls": 0}
    stats_lock   = threading.Lock()

    # Progress bar
    pbar = tqdm(total=len(domains), desc="Crawling domains", unit="domain")

    # Heartbeat — logs progress every 5 minutes
    def heartbeat():
        while not domain_queue.empty():
            time.sleep(300)
            with stats_lock:
                s = stats.copy()
            with results_lock:
                n = len(results)
            logger.info(f"Progress: {s['successful']} done, "
                        f"{s['failed']} failed, "
                        f"{n:,} URLs collected so far")
            # Save intermediate results
            if results:
                with results_lock:
                    snap = results.copy()
                save_results(snap, OUTPUT_PATH, mode=file_mode)
                logger.info(f"Intermediate save: {len(snap):,} URLs")

    threading.Thread(target=heartbeat, daemon=True).start()

    # Launch worker threads
    threads = []
    for _ in range(min(N_THREADS, len(domains))):
        t = threading.Thread(
            target=worker,
            args=(domain_queue, results, results_lock,
                  pbar, stats, stats_lock)
        )
        t.start()
        threads.append(t)

    # Wait for all threads to finish
    for t in threads:
        t.join()
    pbar.close()

    # Final save
    logger.info(f"\nSaving final results to {OUTPUT_PATH}...")
    if results:
        n_saved = save_results(results, OUTPUT_PATH, mode=file_mode)
        logger.info(f"Saved {n_saved:,} unique URLs")
    else:
        logger.warning("No URLs collected.")
        return

    # Final stats
    logger.info("\n" + "=" * 60)
    logger.info("CRAWL COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Domains attempted : {len(domains):,}")
    logger.info(f"Domains succeeded : {stats['successful']:,}")
    logger.info(f"Domains failed    : {stats['failed']:,}")
    logger.info(f"Total URLs saved  : {stats['total_urls']:,}")
    logger.info(f"Output file       : {OUTPUT_PATH}")

    # Verify output
    final_df = pd.read_csv(OUTPUT_PATH)
    logger.info(f"\nFinal CSV stats:")
    logger.info(f"  Total rows  : {len(final_df):,}")
    logger.info(f"  Unique URLs : {final_df['url'].nunique():,}")
    logger.info(f"  Label       : {final_df['label'].value_counts().to_dict()}")

    if len(final_df) < 50_000:
        logger.warning(f"\nWARNING: Only {len(final_df):,} URLs collected.")
        logger.warning("Consider increasing TOP_N_DOMAINS or "
                       "MAX_URLS_PER_DOMAIN.")
    else:
        logger.info(f"\nReady for pipeline.")
        logger.info("Next step: update preprocessing.py to include "
                    "benign_crawled.csv")
        logger.info("Then run the full pipeline again.")


if __name__ == "__main__":
    main()