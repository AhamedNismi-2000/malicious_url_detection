#!/usr/bin/env python3
"""
crawl_benign.py  -  build data/raw/benign_crawled.csv (benign URLs with REAL scheme + paths)
-----------------------------------------------------------------------------------------
Why: data.csv (the old source of benign URLs with paths) has no scheme and is disabled.
Tranco gives only bare domains and PhiUSIIL-legit has no paths, so benign URLs with
paths/queries must come from somewhere real. This script visits sampled Tranco domains
and records the URLs a real visitor would see:

    home     the homepage URL after redirects (real scheme, e.g. https://www.x.com/en/)
    link     same-site links found on the homepage
    sitemap  same-site URLs listed in the site's sitemap.xml

It only FETCHES robots.txt, the homepage and (optionally) one sitemap per domain.
The collected links are recorded, never fetched, so the load on each site is tiny.

Run from the project root (venv active):
    python scripts/crawl_benign.py --dry-sample            # show the domain sample, no network
    python scripts/crawl_benign.py --limit 200             # small trial run first
    python scripts/crawl_benign.py                         # full run (hours; Ctrl+C is safe)
    python scripts/crawl_benign.py --selftest              # offline test with a fake web

Resumable: re-run the same command and finished domains are skipped.
Output (appended while running):
    data/raw/benign_crawled.csv          columns: url, domain, rank, kind   (preprocessing reads "url")
    data/raw/benign_crawl_progress.jsonl one line per finished domain (status, rows)
    reports/crawl_report.json            written at the end of every run

Politeness: robots.txt is obeyed (RFC 9309), a descriptive User-Agent is sent, each domain is
visited by one worker at a time with a delay between requests, and bodies are size-limited.
EDIT the USER_AGENT contact below before the full run.

Needs: requests, pandas, scripts/url_utils.py (tldextract).
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter, namedtuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from url_utils import SHORTENERS, canonicalize                    # noqa: E402

# ═════════════════════════ SETTINGS (edit here) ═════════════════════════
CONFIG = dict(
    SEED=42,
    # (rank_from, rank_to, domains_to_crawl): long tail on purpose
    BUCKETS=[(1, 1_000, 500), (1_001, 10_000, 3_000),
             (10_001, 100_000, 12_000), (100_001, 1_000_000, 20_000)],
    # extra domains by TLD, because country-code sites are under-represented in old data
    BOOST_TLDS={"lk": 500, "in": 1_500, "pk": 500, "bd": 300, "ng": 300, "za": 300},
    OVERSAMPLE=1.4,           # sample more, then drop unusable domains (shared platforms...)
    MAX_LINKS_PER_DOMAIN=12,  # URLs recorded per domain besides the homepage
    MAX_QUERY_LINKS=5,        # of those, at most this many with a ?query (malicious data has ~16% queries)
    MAX_URL_LEN=300,
    WORKERS=48,               # domains crawled in parallel (each domain: one request at a time)
    DELAY_SECONDS=1.0,        # pause between two requests to the SAME domain
    MAX_CRAWL_DELAY=5.0,      # honour robots Crawl-delay up to this many seconds
    CONNECT_TIMEOUT=4, READ_TIMEOUT=8,
    MAX_BODY_BYTES=1_500_000, MAX_SITEMAP_BYTES=1_000_000, MAX_SITEMAP_LOCS=300,
    USE_SITEMAP=True,
)
USER_AGENT = ("ThesisURLResearchBot/1.0 (+academic research on URL classification; "
              "contact: mnanismi2000@gmail.com)")                 # <- put a real contact here
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SKIP_DOMAINS = {   # infrastructure hosts without a normal website (same list as preprocessing v1)
    "googletagmanager.com", "doubleclick.net", "googleapis.com", "gstatic.com",
    "cloudflare.com", "akamaized.net", "fastly.net", "cloudfront.net", "amazonaws.com",
    "gtld-servers.net", "root-servers.net",
}
BAD_EXT = re.compile(r"\.(zip|rar|7z|gz|tar|tgz|exe|msi|dmg|pkg|apk|iso|bin|deb|rpm|jar|"
                     r"mp3|mp4|avi|mov|mkv|wav|flac|jpg|jpeg|png|gif|webp|svg|ico|"
                     r"css|js|json|xml|woff2?|ttf|eot|pdf|docx?|xlsx?|pptx?)$", re.I)
BAD_WORDS = re.compile(r"logout|log-out|signout|sign-out|unsubscribe|delete|remove|"
                       r"cart/add|add-to-cart|\bsession", re.I)
PRIVATE_HOST = re.compile(r"^(localhost|127\.|10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.|0\.0\.0\.0|\[?::1)")

FetchResult = namedtuple("FetchResult", "status final_url ctype text error")


def classify_error(exc: Exception) -> str:
    """Short, countable reason for a failed request (shown in the report and --diagnose)."""
    t = f"{type(exc).__name__} {exc}".lower()
    for needle, label in (("certificate_verify_failed", "ssl_cert"), ("name or service not known", "dns"),
                          ("getaddrinfo failed", "dns"), ("nameresolution", "dns"),
                          ("nodename nor servname", "dns"), ("connecttimeout", "connect_timeout"),
                          ("readtimeout", "read_timeout"), ("timed out", "timeout"),
                          ("refused", "refused"), ("reset", "reset"), ("remotedisconnected", "reset"),
                          ("ssl", "ssl"), ("too many redirects", "redirect_loop"),
                          ("proxy", "proxy")):
        if needle in t:
            return label
    return type(exc).__name__


def say(msg: str = "") -> None:
    print(msg, flush=True)


# ═════════════════════════ domain sampling (offline, deterministic) ═════════════════════════
def _usable(domain: str) -> bool:
    """Apex domains of normal websites only."""
    if domain in SKIP_DOMAINS or domain in SHORTENERS:
        return False
    c, _ = canonicalize("https://" + domain)
    if c is None or c["is_ip"] or c["shared_platform"]:
        return False
    return c["host"] == c["reg_domain"]


def sample_domains(tr: pd.DataFrame, cfg: dict) -> list:
    """tr has columns rank (int) and domain. Returns [(rank, domain)] in shuffled order, so a
    run that is stopped early is still a representative sample."""
    seed = cfg["SEED"]
    chosen: dict = {}

    def take(frame: pd.DataFrame, n: int, salt: int) -> None:
        if n <= 0 or frame.empty:
            return
        frame = frame[~frame["domain"].isin(chosen)]
        cand = frame.sample(min(len(frame), int(n * cfg["OVERSAMPLE"]) + 5),
                            random_state=seed + salt)
        got = 0
        for r, d in zip(cand["rank"], cand["domain"]):
            if got >= n:
                break
            if _usable(d):
                chosen[d] = int(r)
                got += 1

    for i, (lo, hi, n) in enumerate(cfg["BUCKETS"]):
        take(tr[(tr["rank"] >= lo) & (tr["rank"] <= hi)], n, i)
    tld = tr["domain"].str.rsplit(".", n=1).str[-1]
    for j, (suffix, n) in enumerate(cfg["BOOST_TLDS"].items()):
        take(tr[tld == suffix], n, 100 + j)
    out = sorted(chosen.items(), key=lambda kv: kv[1])
    random.Random(seed).shuffle(out)
    return [(r, d) for d, r in out]


def load_tranco(path: str) -> pd.DataFrame:
    tr = pd.read_csv(path, dtype=str, header=None, names=["rank", "domain"])
    tr["rank"] = pd.to_numeric(tr["rank"], errors="coerce")
    tr["domain"] = tr["domain"].str.strip().str.lower()
    return tr.dropna().astype({"rank": int}).reset_index(drop=True)


# ═════════════════════════ HTML / sitemap parsing ═════════════════════════
class _LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hrefs: list = []
        self.base = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "a" and a.get("href"):
            self.hrefs.append(a["href"].strip())
        elif tag == "base" and a.get("href") and self.base is None:
            self.base = a["href"].strip()


def extract_links(html: str, page_url: str) -> list:
    p = _LinkParser()
    try:
        p.feed(html[:1_500_000])
    except Exception:
        pass
    base = urljoin(page_url, p.base) if p.base else page_url
    return [urljoin(base, h) for h in p.hrefs]


def sitemap_locs(xml: str) -> tuple:
    """Return (is_index, [urls])."""
    locs = [m.strip() for m in re.findall(r"<loc>\s*([^<\s][^<]*)</loc>", xml, flags=re.I)]
    return ("<sitemapindex" in xml[:2000].lower()), locs


def _reg(url: str) -> str:
    c, _ = canonicalize(url)
    return c["reg_domain"] if c else ""


def usable_link(url: str, site_reg: str, cfg: dict) -> bool:
    """Same-site, plain http(s), no downloads, no tracking/logout-style URLs."""
    try:
        sp = urlsplit(url)
    except ValueError:
        return False
    if sp.scheme not in ("http", "https") or not sp.hostname or "@" in sp.netloc:
        return False
    if PRIVATE_HOST.match(sp.hostname):
        return False
    if len(url) > cfg["MAX_URL_LEN"] or BAD_EXT.search(sp.path) or BAD_WORDS.search(url):
        return False
    return _reg(url) == site_reg


def same_site_http(url: str, site_reg: str) -> bool:
    """Plain http(s) URL on the same registered domain (no file-type filtering)."""
    try:
        sp = urlsplit(url)
    except ValueError:
        return False
    return (sp.scheme in ("http", "https") and bool(sp.hostname) and "@" not in sp.netloc
            and not PRIVATE_HOST.match(sp.hostname) and _reg(url) == site_reg)


def pick_links(candidates: list, home_url: str, rng: random.Random, cfg: dict) -> list:
    """Diverse sample: round-robin over first path segments, limited number of ?query links."""
    seen, uniq = {home_url.rstrip("/")}, []
    for u, kind in candidates:
        u = u.split("#", 1)[0]
        key = u.rstrip("/")
        if u and key not in seen:
            seen.add(key)
            uniq.append((u, kind))
    rng.shuffle(uniq)
    groups: dict = {}
    for u, kind in uniq:
        seg = (urlsplit(u).path.strip("/").split("/") + [""])[0].lower()
        groups.setdefault(seg, []).append((u, kind))
    picked, n_query = [], 0
    while groups and len(picked) < cfg["MAX_LINKS_PER_DOMAIN"]:
        for seg in list(groups):
            if len(picked) >= cfg["MAX_LINKS_PER_DOMAIN"]:
                break
            while groups[seg]:
                u, kind = groups[seg].pop()
                has_q = "?" in u
                if has_q and n_query >= cfg["MAX_QUERY_LINKS"]:
                    continue
                n_query += has_q
                picked.append((u, kind))
                break
            if not groups[seg]:
                del groups[seg]
    return picked


# ═════════════════════════ one domain ═════════════════════════
def crawl_domain(domain: str, rank: int, cfg: dict, fetch, sleep=time.sleep) -> dict:
    """Return dict(status, rows). `fetch(url) -> FetchResult` is injectable (tests use a fake web).
    status: ok | no_response | robots_blocked | robots_error | redirected_off_site |
            non_html | bad_homepage"""
    rng = random.Random(hashlib.sha256(f"{cfg['SEED']}:{domain}".encode()).hexdigest())
    gap = cfg["DELAY_SECONDS"]

    def get(url):
        r = fetch(url)
        if r.error is None:                            # v2: delay only after a real answer
            sleep(gap)
        return r

    # 1) find a reachable base + read robots.txt (before touching any page)
    base, robots, errs = None, None, []
    for cand in (f"https://{domain}", f"https://www.{domain}", f"http://{domain}"):
        r = get(cand + "/robots.txt")
        if r.error is None:                       # any HTTP answer = host reachable
            base, robots = cand, r
            break
        errs.append(r.error)
    if base is None:
        return dict(status="no_response", rows=[], detail=Counter(errs).most_common(1)[0][0])

    rp = RobotFileParser()
    if robots.status is not None and 500 <= robots.status < 600:
        return dict(status="robots_error", rows=[], detail=f"http_{robots.status}")      # RFC 9309: 5xx = assume disallow
    if robots.status == 200 and robots.text is not None:
        rp.parse(robots.text.splitlines())
    else:
        rp.parse([])                                      # 4xx = no robots rules
    allowed = lambda u: rp.can_fetch(USER_AGENT, u)       # noqa: E731
    cd = rp.crawl_delay(USER_AGENT)
    if cd:
        gap = min(max(gap, float(cd)), cfg["MAX_CRAWL_DELAY"])
    if not allowed(base + "/"):
        return dict(status="robots_blocked", rows=[], detail="disallow_root")

    # 2) homepage
    h = get(base + "/")
    if h.error is not None or h.status is None or h.status >= 400:
        return dict(status="bad_homepage", rows=[], detail=h.error or f"http_{h.status}")
    site_reg = _reg("https://" + domain)
    final = h.final_url or (base + "/")
    if _reg(final) != site_reg:
        return dict(status="redirected_off_site", rows=[], detail=_reg(final))
    if "html" not in (h.ctype or "").lower() or not h.text:
        return dict(status="non_html", rows=[], detail=(h.ctype or "")[:30])
    final_canon, _ = canonicalize(final)
    if final_canon is None or PRIVATE_HOST.match(final_canon["host"]):
        return dict(status="bad_homepage", rows=[], detail="invalid_or_private_final_url")

    rows = [(final, "home")]
    cands = [(u, "link") for u in extract_links(h.text, final)
             if usable_link(u, site_reg, cfg)]

    # 3) one sitemap (optional)
    if cfg["USE_SITEMAP"]:
        sm_url = None
        if robots.text:
            m = re.search(r"(?im)^\s*sitemap:\s*(\S+)", robots.text)
            sm_url = m.group(1) if m else None
        sm_url = sm_url or (urlsplit(final)._replace(path="/sitemap.xml", query="").geturl())
        if sm_url.lower().endswith(".gz") or not allowed(sm_url):
            sm_url = None
        if sm_url and same_site_http(sm_url, site_reg):
            s = get(sm_url)
            if s.error is None and s.status == 200 and s.text:
                is_index, locs = sitemap_locs(s.text)
                if is_index and locs and allowed(locs[0]) and same_site_http(locs[0], site_reg) \
                        and not locs[0].lower().endswith(".gz"):
                    s2 = get(locs[0])
                    if s2.error is None and s2.status == 200 and s2.text:
                        _, locs = sitemap_locs(s2.text)
                    else:
                        locs = []
                for u in locs[:cfg["MAX_SITEMAP_LOCS"]]:
                    if usable_link(u, site_reg, cfg):
                        cands.append((u, "sitemap"))

    cands = [(u, k) for u, k in cands if allowed(u)]
    rows += pick_links(cands, final, rng, cfg)
    return dict(status="ok", rows=rows)


# ═════════════════════════ real HTTP fetcher ═════════════════════════
def make_fetcher(cfg: dict):
    import requests
    local = threading.local()

    def session():
        if not hasattr(local, "s"):
            s = requests.Session()
            s.headers.update({"User-Agent": USER_AGENT, "Accept": "text/html,application/xml;q=0.9,*/*;q=0.5"})
            s.max_redirects = 5
            local.s = s
        return local.s

    def fetch(url: str) -> FetchResult:
        try:
            r = session().get(url, timeout=(cfg["CONNECT_TIMEOUT"], cfg["READ_TIMEOUT"]),
                              stream=True, allow_redirects=True)
        except Exception as exc:
            return FetchResult(None, None, None, None, classify_error(exc))
        try:
            for hop in list(r.history) + [r]:                         # never follow redirects to private hosts
                host = urlsplit(hop.url).hostname or ""
                if PRIVATE_HOST.match(host):
                    return FetchResult(None, None, None, None, "private_host")
            limit = cfg["MAX_SITEMAP_BYTES"] if ("sitemap" in url or "robots" in url) else cfg["MAX_BODY_BYTES"]
            ctype = r.headers.get("Content-Type", "")
            text = None
            if any(t in ctype.lower() for t in ("html", "xml", "text/plain")) or "robots" in url:
                buf, size = [], 0
                for chunk in r.iter_content(65_536):
                    buf.append(chunk)
                    size += len(chunk)
                    if size >= limit:
                        break
                text = b"".join(buf).decode(r.encoding or "utf-8", errors="replace")
            return FetchResult(r.status_code, r.url, ctype, text, None)
        except Exception as exc:
            return FetchResult(None, None, None, None, classify_error(exc))
        finally:
            r.close()

    return fetch


# ═════════════════════════ driver ═════════════════════════
class Store:
    """Append-only CSV + progress log; thread-safe; makes the run resumable."""

    def __init__(self, out_csv: str, progress: str) -> None:
        self.out_csv, self.progress = out_csv, progress
        self.lock = threading.Lock()
        self.done: dict = {}
        if os.path.exists(progress):
            with open(progress, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        j = json.loads(line)
                        self.done[j["domain"]] = j["status"]
                    except (ValueError, KeyError):
                        pass
        new = not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0
        self.csv = open(out_csv, "a", encoding="utf-8", newline="")
        self.prog = open(progress, "a", encoding="utf-8")
        if new:
            self.csv.write("url,domain,rank,kind\n")
            self.csv.flush()

    def add(self, domain: str, rank: int, res: dict) -> None:
        with self.lock:
            for url, kind in res["rows"]:
                u = url.replace('"', "%22")
                self.csv.write(f'"{u}",{domain},{rank},{kind}\n')
            self.csv.flush()
            self.prog.write(json.dumps(dict(domain=domain, status=res["status"],
                                            detail=res.get("detail", ""),
                                            n=len(res["rows"]))) + "\n")
            self.prog.flush()
            self.done[domain] = res["status"]

    def close(self) -> None:
        self.csv.close()
        self.prog.close()


def run_crawl(todo: list, cfg: dict, store: Store, fetch, workers: int, sleep=time.sleep) -> bool:
    """Crawl `todo` = [(rank, domain)]. Returns True if finished, False if interrupted."""
    t0, n_done = time.time(), 0
    counts: Counter = Counter()
    batch = max(workers * 4, 8)
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i in range(0, len(todo), batch):
                futs = {ex.submit(crawl_domain, d, r, cfg, fetch, sleep): (r, d)
                        for r, d in todo[i:i + batch]}
                for f in as_completed(futs):
                    r, d = futs[f]
                    try:
                        res = f.result()
                    except Exception as exc:                       # one bad domain never stops the run
                        res = dict(status=f"error:{type(exc).__name__}", rows=[], detail=str(exc)[:80])
                    store.add(d, r, res)
                    counts[res["status"]] += 1
                    n_done += 1
                    if n_done % 100 == 0:
                        rate = n_done / max(time.time() - t0, 1e-9)
                        left = (len(todo) - n_done) / max(rate, 1e-9) / 60
                        say(f"  {n_done:>6,}/{len(todo):,} domains | ok {counts['ok']:,} | "
                            f"{rate:.1f}/s | ~{left:.0f} min left")
        return True
    except KeyboardInterrupt:
        say("\nInterrupted. Progress is saved: run the same command again to resume.")
        return False


def build_report(out_csv: str, progress: str, report_path: str, finished: bool) -> dict:
    status: Counter = Counter()
    detail: Counter = Counter()
    if os.path.exists(progress):
        for line in open(progress, encoding="utf-8"):
            try:
                j = json.loads(line)
                status[j["status"]] += 1
                if j.get("detail"):
                    detail[f"{j['status']}: {j['detail']}"] += 1
            except (ValueError, KeyError):
                pass
    rep = dict(finished=finished, domain_status=dict(status),
               failure_reasons=dict(detail.most_common(15)))
    if os.path.exists(out_csv):
        df = pd.read_csv(out_csv, dtype=str, keep_default_na=False)
        c = [canonicalize(u)[0] for u in df["url"]]
        ok = [x for x in c if x]
        rep.update(rows=len(df), unique_urls=int(df["url"].nunique()),
                   kinds=df["kind"].value_counts().to_dict(),
                   https_pct=round(100 * sum(x["scheme"] == "https" for x in ok) / max(len(ok), 1), 1),
                   has_path_pct=round(100 * sum(x["has_path"] for x in ok) / max(len(ok), 1), 1),
                   has_query_pct=round(100 * sum(x["has_query"] for x in ok) / max(len(ok), 1), 1),
                   top_suffixes=Counter(x["suffix"] for x in ok).most_common(10))
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2)
    return rep


def diagnose(sample: list, cfg: dict, n: int) -> None:
    """Connectivity check + first n sampled domains, one at a time, with reasons."""
    fetch = make_fetcher(cfg)
    say("Connectivity check (these should all work from a normal connection):")
    for u in ("https://example.com/", "https://www.wikipedia.org/", "https://www.python.org/",
              "http://neverssl.com/"):
        t = time.time()
        r = fetch(u)
        say(f"   {u:<32} status={r.status} error={r.error} ({time.time() - t:.1f}s)")
    say(f"\nFirst {n} sampled domains, sequential:")
    for rank, d in sample[:n]:
        t = time.time()
        res = crawl_domain(d, rank, cfg, fetch)
        say(f"   {d:<28} {res['status']:<20} {res.get('detail', ''):<18} "
            f"rows={len(res['rows'])} ({time.time() - t:.1f}s)")


# ═════════════════════════ self-test (offline fake web) ═════════════════════════
def _selftest() -> int:
    import tempfile

    page = ('<html><body><a href="/news/2024/story-1">a</a><a href="/news/2024/story-2">b</a>'
            '<a href="/about">c</a><a href="/shop?item=5">d</a><a href="/shop?item=6">e</a>'
            '<a href="/shop?item=7">f</a><a href="https://other.org/x">off</a>'
            '<a href="mailto:a@b.c">m</a><a href="/files/setup.exe">x</a><a href="/logout">lo</a>'
            '<a href="/blog/post-1#top">g</a><a href="javascript:void(0)">j</a></body></html>')
    web = {
        "https://example-a.com/robots.txt": (200, "text/plain", "User-agent: *\nDisallow: /private\n"
                                                "Sitemap: https://example-a.com/sitemap.xml"),
        "https://example-a.com/": (200, "text/html", page),
        "https://example-a.com/sitemap.xml": (200, "application/xml",
            "<urlset><loc>https://example-a.com/docs/a</loc><loc>https://example-a.com/private/x</loc>"
            "<loc>https://example-a.com/docs/b</loc></urlset>"),
        "https://blocked.com/robots.txt": (200, "text/plain", "User-agent: *\nDisallow: /\n"),
        "https://offsite.com/robots.txt": (404, "text/html", ""),
        "https://offsite.com/": (200, "text/html", "<html></html>", "https://other.org/"),
        "https://pdfonly.com/robots.txt": (404, "text/html", ""),
        "https://pdfonly.com/": (200, "application/pdf", ""),
        "https://srv5xx.com/robots.txt": (503, "text/html", ""),
        "https://shop.lk/robots.txt": None, "https://www.shop.lk/robots.txt": None,
        "http://shop.lk/robots.txt": (404, "text/html", ""),
        "http://shop.lk/": (200, "text/html", '<a href="/products/phone-5">p</a>'),
    }

    def fake_fetch(url):
        v = web.get(url)
        if v is None:
            return FetchResult(None, None, None, None, "ConnectionError")
        status, ctype, text = v[0], v[1], v[2]
        final = v[3] if len(v) > 3 else url
        return FetchResult(status, final, ctype, text, None)

    cfg = dict(CONFIG)
    cfg.update(DELAY_SECONDS=0.0)
    ok_all: list = []

    def check(name, cond):
        ok_all.append(bool(cond))
        say(f"[{'PASS' if cond else 'FAIL'}] {name}")

    nosleep = lambda s: None                                                  # noqa: E731
    R = {d: crawl_domain(d, 1, cfg, fake_fetch, nosleep)
         for d in ("example-a.com", "blocked.com", "offsite.com", "pdfonly.com",
                   "srv5xx.com", "down.com", "shop.lk")}
    a = R["example-a.com"]
    urls = [u for u, _ in a["rows"]]
    check("ok domain: status ok, homepage recorded with real https scheme",
          a["status"] == "ok" and urls[0] == "https://example-a.com/")
    check("same-site only (no other.org, mailto, javascript)",
          all("example-a.com" in u for u in urls))
    check("no .exe / logout / robots-disallowed /private", not any(
        ("setup.exe" in u or "logout" in u or "/private" in u) for u in urls))
    check("fragment stripped", not any("#" in u for u in urls))
    check("max links respected", len(urls) <= cfg["MAX_LINKS_PER_DOMAIN"] + 1)
    check("at most MAX_QUERY_LINKS query links", sum("?" in u for u in urls) <= cfg["MAX_QUERY_LINKS"])
    check("sitemap URLs are used (and robots-disallowed sitemap entry is not)",
          any(k == "sitemap" for _, k in a["rows"]) and not any("/private" in u for u in urls))
    check("robots Disallow: / -> robots_blocked", R["blocked.com"]["status"] == "robots_blocked")
    check("redirect to another site -> redirected_off_site", R["offsite.com"]["status"] == "redirected_off_site")
    check("pdf homepage -> non_html", R["pdfonly.com"]["status"] == "non_html")
    check("robots 5xx -> robots_error (skipped, not crawled)", R["srv5xx.com"]["status"] == "robots_error")
    check("unreachable -> no_response", R["down.com"]["status"] == "no_response")
    check("http-only site keeps real http scheme", R["shop.lk"]["status"] == "ok"
          and R["shop.lk"]["rows"][0][0] == "http://shop.lk/")
    check("failure reasons are recorded (detail)", R["down.com"].get("detail") == "ConnectionError"
          and R["srv5xx.com"].get("detail") == "http_503" and R["offsite.com"].get("detail") == "other.org")
    check("error classifier", classify_error(Exception("getaddrinfo failed")) == "dns"
          and classify_error(Exception("CERTIFICATE_VERIFY_FAILED")) == "ssl_cert"
          and classify_error(TimeoutError("timed out")) == "timeout")
    check("deterministic for the same seed",
          crawl_domain("example-a.com", 1, cfg, fake_fetch, nosleep)["rows"] == a["rows"])

    # sampling
    mock = pd.DataFrame({"rank": range(1, 3001),
                         "domain": [f"site{i}.{['com', 'lk', 'in', 'org'][i % 4]}" for i in range(3000)]})
    mock.loc[5, "domain"] = "bit.ly"
    mock.loc[6, "domain"] = "foo.github.io"
    c2 = dict(cfg)
    c2.update(BUCKETS=[(1, 100, 20), (101, 3000, 200)], BOOST_TLDS={"lk": 40})
    s = sample_domains(mock, c2)
    doms = [d for _, d in s]
    check("sample: no shorteners / shared-platform roots", "bit.ly" not in doms and "foo.github.io" not in doms)
    check("sample: .lk boost applied", sum(d.endswith(".lk") for d in doms) >= 40)
    check("sample: unique + deterministic", len(set(doms)) == len(doms) and s == sample_domains(mock, c2))

    # full driver + resume
    tmp = tempfile.mkdtemp(prefix="crawl_selftest_")
    out, prog = os.path.join(tmp, "b.csv"), os.path.join(tmp, "p.jsonl")
    todo = [(1, "example-a.com"), (2, "blocked.com"), (3, "shop.lk"), (4, "down.com")]
    st = Store(out, prog)
    done = run_crawl(todo, cfg, st, fake_fetch, workers=3, sleep=nosleep)
    st.close()
    df = pd.read_csv(out, keep_default_na=False)
    check("driver: finished and wrote csv with url column", done and "url" in df.columns and len(df) > 3)
    st2 = Store(out, prog)
    todo2 = [(r, d) for r, d in todo if d not in st2.done]
    st2.close()
    check("resume: finished domains are skipped", todo2 == [])
    rep = build_report(out, prog, os.path.join(tmp, "r.json"), True)
    check("report: statuses counted", rep["domain_status"].get("ok") == 2
          and rep["domain_status"].get("robots_blocked") == 1)

    n = sum(ok_all)
    say(f"\nself-test: {n}/{len(ok_all)} passed")
    return 0 if all(ok_all) else 1


# ═════════════════════════ entry point ═════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tranco", default=os.path.join(BASE_DIR, "data", "raw", "top-1m.csv"))
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "data", "raw", "benign_crawled.csv"))
    ap.add_argument("--progress", default=os.path.join(BASE_DIR, "data", "raw", "benign_crawl_progress.jsonl"))
    ap.add_argument("--report", default=os.path.join(BASE_DIR, "reports", "crawl_report.json"))
    ap.add_argument("--limit", type=int, help="crawl only the first N domains of the shuffled sample (trial run)")
    ap.add_argument("--workers", type=int)
    ap.add_argument("--delay", type=float, help="seconds between requests to the same domain")
    ap.add_argument("--max-links", type=int)
    ap.add_argument("--no-sitemap", action="store_true")
    ap.add_argument("--dry-sample", action="store_true", help="print the domain sample, no network")
    ap.add_argument("--diagnose", type=int, metavar="N",
                    help="connectivity check + crawl N domains sequentially, printing the reason for each failure")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(_selftest())

    cfg = dict(CONFIG)
    if args.workers:
        cfg["WORKERS"] = args.workers
    if args.delay is not None:
        cfg["DELAY_SECONDS"] = args.delay
    if args.max_links:
        cfg["MAX_LINKS_PER_DOMAIN"] = args.max_links
    if args.no_sitemap:
        cfg["USE_SITEMAP"] = False

    if not os.path.exists(args.tranco):
        sys.exit(f"Tranco list not found: {args.tranco}")
    sample = sample_domains(load_tranco(args.tranco), cfg)
    if args.limit:
        sample = sample[:args.limit]
    tld = Counter(d.rsplit(".", 1)[-1] for _, d in sample)
    say(f"Domain sample: {len(sample):,} | top TLDs: {tld.most_common(8)} | "
        f".lk {tld.get('lk', 0)}, .in {tld.get('in', 0)}")
    if args.dry_sample:
        for r, d in sample[:15]:
            say(f"   rank {r:>7}  {d}")
        return
    if args.diagnose:
        diagnose(sample, cfg, args.diagnose)
        return
    if "YOUR_EMAIL" in USER_AGENT:
        say("WARNING: edit USER_AGENT in this file and put a real contact address before the full run.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    store = Store(args.out, args.progress)
    todo = [(r, d) for r, d in sample if d not in store.done]
    say(f"Already done: {len(sample) - len(todo):,} | to crawl now: {len(todo):,}")
    finished = run_crawl(todo, cfg, store, make_fetcher(cfg), cfg["WORKERS"])
    store.close()
    rep = build_report(args.out, args.progress, args.report, finished)
    say("\nCrawl report: " + json.dumps(rep, indent=2))
    say(f"\nWrote {args.out}\nNext: python scripts/preprocessing.py")


if __name__ == "__main__":
    main()