#!/usr/bin/env python3
"""
typosquat.py  (v2)  -  lookalike-domain detection, shared by training AND serving
-------------------------------------------------------------------------------------
ONE implementation used by feature_extraction.py (training features), model_loader.py
(rule layer) and generate_typosquatting.py (checking the synthetic data). Using the same
code everywhere is what keeps training and serving consistent.

Run from the project root (venv active). No flags needed:

    python scripts/typosquat.py          # self-test + false-positive report

Needs data/reference/popular_domains.csv (written by preprocess.py, Tranco top-N).

What it detects (technique name -> example)
    homoglyph          paypa1.com, arnazon.com (rn->m), g00gle.com, IDN look-alikes
    transposition      paypla.com
    keyboard_adjacent  paypak.com
    bitsquat           paypal -> pcypal (one flipped bit)
    doubled_letter     gooogle.com
    dropped_letter     paypl.com
    substitution/insertion  other single-character typos
    plural             paypals.com
    edit_distance_2    two typos in a long brand (microsooft -> only for 8+ letters)
    hyphen_insert      pay-pal.com
    dot_split          pay.pal.com
    www_prefix         wwwgoogle.com
    tld_swap           paypal.tk  (brand label on a foreign registered domain)
    combosquat         paypal-secure.com, securepaypal.com, paypal-xyz.com
    brand_domain_in_subdomain   www.paypal.com.evil.tk
    brand_in_subdomain          paypal.secure-login.evil.tk
    brand_in_path               evil.com/paypal/login
    tenant:   the same rules are applied to the tenant name on shared platforms
              (paypa1-login.weebly.com), because a tenant name is chosen by the attacker

False-positive guards (the part that makes this usable)
    - a registered domain that is itself popular (top POPULAR_SKIP_N) is never flagged
    - a token that is itself the label of a popular site is never fuzzy-matched
    - fuzzy rules need a minimum brand length (longer for edit distance 2)
    - subdomain / path / substring / keyword rules only protect HIGH_RISK_BRANDS
      (generic words such as "mail" or "live" would otherwise fire everywhere)
    - ALLOWLIST_DOMAINS and NOISY_BRANDS are small, editable exception lists

Public API
    analyze(url)         -> dict with score (0-1), technique, brand, where, matches ...
    feature_vector(url)  -> list of 5 floats (FEATURE_NAMES) for the ML model
    get_index()          -> the lazily built BrandIndex (safe with multiprocessing/spawn)
"""

import csv
import os
import re
import sys
import unicodedata
from collections import namedtuple
from urllib.parse import urlsplit

import tldextract

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ═════════════════════════ SETTINGS (edit here) ═════════════════════════
POPULAR_CSV       = os.path.join(BASE_DIR, "data", "reference", "popular_domains.csv")
BRAND_TOP_N       = 1500     # top-N popular domains protected by the fuzzy rules
POPULAR_SKIP_N    = 10_000   # registered domains in the top-N are never flagged
MIN_LEN_DIST1     = (7, 6)   # (generic brand, high-risk brand): min length for 1 typo
MIN_LEN_DIST2     = (99, 8)  # same, for 2 typos (generic brands: never, too noisy)
MIN_LEN_SKELETON  = (6, 4)   # same, for homoglyph / leet matching
MIN_LEN_SUBSTRING = 6        # brand hidden inside a longer label (high-risk brands only)

# Exception lists: extend these when you find a false positive.
ALLOWLIST_DOMAINS = set()    # registered domains that must never be flagged
# seeded from the false-positive report below: popular sites that are not impersonation targets
NOISY_BRANDS      = {"nhentai", "xvideos", "faphouse", "xhamster", "xhamster2", "xnxx", "pornhub"}

# Brands phishers impersonate. Applied everywhere, even when not in the top-N list
# (e.g. wellsfargo.com ranks 2,872, chase.com 3,031, hsbc.com 7,422 in Tranco).
HIGH_RISK_BRANDS = {
    # tech / social
    "paypal": "paypal.com", "amazon": "amazon.com", "microsoft": "microsoft.com",
    "apple": "apple.com", "google": "google.com", "facebook": "facebook.com",
    "instagram": "instagram.com", "twitter": "twitter.com", "linkedin": "linkedin.com",
    "whatsapp": "whatsapp.com", "telegram": "telegram.org", "tiktok": "tiktok.com",
    "snapchat": "snapchat.com", "pinterest": "pinterest.com", "discord": "discord.com",
    "github": "github.com", "dropbox": "dropbox.com", "adobe": "adobe.com",
    "icloud": "icloud.com", "yahoo": "yahoo.com", "gmail": "gmail.com",
    "outlook": "outlook.com", "office365": "office365.com", "onedrive": "onedrive.com",
    "docusign": "docusign.com", "zoom": "zoom.us", "slack": "slack.com",
    "salesforce": "salesforce.com", "netflix": "netflix.com", "spotify": "spotify.com",
    "steampowered": "steampowered.com", "roblox": "roblox.com", "epicgames": "epicgames.com",
    "norton": "norton.com", "mcafee": "mcafee.com", "kaspersky": "kaspersky.com",
    # payments / banks
    "ebay": "ebay.com", "visa": "visa.com", "mastercard": "mastercard.com",
    "chase": "chase.com", "wellsfargo": "wellsfargo.com",
    "bankofamerica": "bankofamerica.com", "citibank": "citibank.com", "hsbc": "hsbc.com",
    "barclays": "barclays.co.uk", "lloydsbank": "lloydsbank.com",
    "santander": "santander.com", "americanexpress": "americanexpress.com",
    "capitalone": "capitalone.com", "usbank": "usbank.com", "westernunion": "westernunion.com",
    "moneygram": "moneygram.com", "stripe": "stripe.com", "venmo": "venmo.com",
    "zellepay": "zellepay.com", "cashapp": "cash.app", "intuit": "intuit.com",
    "quickbooks": "quickbooks.com", "turbotax": "turbotax.com",
    # crypto
    "coinbase": "coinbase.com", "binance": "binance.com", "blockchain": "blockchain.com",
    "metamask": "metamask.io", "kraken": "kraken.com", "robinhood": "robinhood.com",
    "trustwallet": "trustwallet.com", "bitcoin": "bitcoin.org",
    # shipping / retail / travel / telecom
    "dhl": "dhl.com", "fedex": "fedex.com", "usps": "usps.com", "royalmail": "royalmail.com",
    "walmart": "walmart.com", "costco": "costco.com", "bestbuy": "bestbuy.com",
    "alibaba": "alibaba.com", "aliexpress": "aliexpress.com", "rakuten": "rakuten.com",
    "shopify": "shopify.com", "booking": "booking.com", "airbnb": "airbnb.com",
    "verizon": "verizon.com", "vodafone": "vodafone.com",
    "comcast": "comcast.com", "xfinity": "xfinity.com",
}

# Shared platforms: the tenant name (x in x.weebly.com) is attacker-chosen, so the
# lookalike rules are applied to it. Same list as audit_data.py.
SHARED_PLATFORMS = {
    "github.io", "githubusercontent.com", "gitlab.io", "blogspot.com", "herokuapp.com",
    "netlify.app", "vercel.app", "pages.dev", "workers.dev", "web.app", "firebaseapp.com",
    "azurewebsites.net", "amazonaws.com", "cloudfront.net", "wixsite.com", "weebly.com",
    "wordpress.com", "notion.site", "000webhostapp.com", "glitch.me", "repl.co",
    "onrender.com", "sites.google.com", "storage.googleapis.com", "tumblr.com",
}

# Words attackers glue to a brand. Used for combosquatting.
KEYWORDS = {
    "secure", "security", "login", "signin", "verify", "verification", "update", "account",
    "accounts", "support", "help", "service", "services", "billing", "payment", "payments",
    "wallet", "auth", "alert", "alerts", "confirm", "recovery", "recover", "unlock",
    "refund", "online", "portal", "customer", "official", "mail", "id", "pay", "bank",
    "banking", "my", "the", "app", "web", "get", "free", "new", "access", "center",
    "care", "team", "prime", "store", "shop", "claim", "reward", "rewards", "invoice",
    "www", "ww",
}

# Technique -> confidence. Higher = stronger evidence of impersonation.
SCORE = {
    "homoglyph": 1.0, "brand_domain_in_subdomain": 1.0,
    "hyphen_insert": 0.9, "dot_split": 0.9, "www_prefix": 0.9, "transposition": 0.9,
    "combosquat_keyword": 0.9, "brand_in_subdomain": 0.9,
    "keyboard_adjacent": 0.85, "bitsquat": 0.85, "doubled_letter": 0.85,
    "dropped_letter": 0.85, "substitution": 0.8, "insertion": 0.8,
    "combosquat": 0.7, "edit_distance_2": 0.6,
    "plural": 0.5, "tld_swap": 0.5, "brand_substring": 0.5, "brand_in_path": 0.4,
}
EDIT_TECHNIQUES = {"transposition", "keyboard_adjacent", "bitsquat", "doubled_letter",
                   "dropped_letter", "substitution", "insertion", "plural"}

EXTRACTOR = tldextract.TLDExtract(cache_dir=None, suffix_list_urls=None)

Brand = namedtuple("Brand", "label domain rank risk")

_IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
_LABEL_RE = re.compile(r"^[a-z0-9]+$")
_INFRA_HINTS = ("akam", "gstatic", "googleapis", "gtld", "fbcdn", "doubleclick",
                "googlevideo", "googleusercontent", "windowsupdate", "msedge", "msft",
                "edgekey", "edgesuite", "cloudfront", "amazonaws", "cdn", "analytics",
                "adservice", "telemetry", "nameserver", "rootservers", "akadns", "tracking",
                "dns", "static", "usercontent", "content", "edge", "glb")


# ═════════════════════════ string helpers ═════════════════════════
_CONFUSABLE = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y", "і": "i",
    "ј": "j", "ѕ": "s", "ԁ": "d", "ɡ": "g", "ӏ": "l", "ɑ": "a", "ο": "o", "α": "a",
    "ν": "v", "ι": "i", "ı": "i", "ɩ": "i", "ᴏ": "o", "ᴀ": "a", "ѵ": "v", "ԛ": "q",
}
_SKEL_CHAR = str.maketrans({"1": "l", "i": "l", "!": "l", "|": "l", "0": "o", "3": "e",
                            "4": "a", "@": "a", "5": "s", "$": "s", "7": "t", "8": "b",
                            "9": "g", "6": "g"})
_SKEL_MULTI = (("rn", "m"), ("vv", "w"), ("cl", "d"))


def ascii_fold(s: str) -> str:
    s = "".join(_CONFUSABLE.get(c, c) for c in s)
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def skeleton(s: str) -> str:
    """Collapse look-alike characters so that paypa1 / paypai / paypal share one skeleton.
    Applied identically to brands and candidates."""
    s = ascii_fold(s.lower()).translate(_SKEL_CHAR)
    for a, b in _SKEL_MULTI:
        s = s.replace(a, b)
    return s


def osa(a: str, b: str) -> int:
    """Optimal string alignment distance (insert, delete, substitute, adjacent swap)."""
    la, lb = len(a), len(b)
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            c = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[la][lb]


def _deletions(s: str, k: int) -> set:
    out, level = {s}, {s}
    for _ in range(k):
        level = {w[:i] + w[i + 1:] for w in level for i in range(len(w))}
        out |= level
    return out


def _keyboard_neighbors() -> dict:
    rows = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
    nb: dict = {}
    for r, row in enumerate(rows):
        for c, ch in enumerate(row):
            s = set()
            for cc in (c - 1, c + 1):
                if 0 <= cc < len(row):
                    s.add(row[cc])
            if r > 0:                                  # row above sits half a key to the left
                for cc in (c, c + 1):
                    if 0 <= cc < len(rows[r - 1]):
                        s.add(rows[r - 1][cc])
            if r < 2:                                  # row below sits half a key to the right
                for cc in (c - 1, c):
                    if 0 <= cc < len(rows[r + 1]):
                        s.add(rows[r + 1][cc])
            nb[ch] = "".join(sorted(s))
    return nb


KEYBOARD = _keyboard_neighbors()


def classify_dist1(t: str, b: str) -> str:
    """Name the single edit that turns brand b into token t (len(t)-len(b) in -1..1)."""
    if len(t) == len(b):
        i = next(k for k in range(len(t)) if t[k] != b[k])
        x = ord(t[i]) ^ ord(b[i])
        if x and (x & (x - 1)) == 0:
            return "bitsquat"
        if b[i] in KEYBOARD.get(t[i], "") or t[i] in KEYBOARD.get(b[i], ""):
            return "keyboard_adjacent"
        return "substitution"
    if len(t) == len(b) + 1:
        if t == b + "s":
            return "plural"
        for j in range(len(t)):
            if t[:j] + t[j + 1:] == b:
                c = t[j]
                if (j > 0 and t[j - 1] == c) or (j + 1 < len(t) and t[j + 1] == c):
                    return "doubled_letter"
                return "insertion"
        return "insertion"
    return "dropped_letter"


# ═════════════════════════ brand index ═════════════════════════
def _read_popular(path: str) -> list:
    rows = []
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.reader(fh):
            if len(r) >= 2 and r[0].strip().isdigit():
                rows.append((int(r[0]), r[1].strip().lower()))
    rows.sort()
    return rows


def _registered(ext) -> str:
    if hasattr(ext, "top_domain_under_public_suffix"):
        return ext.top_domain_under_public_suffix
    return ext.registered_domain


class BrandIndex:
    """Everything needed to match a token against brands quickly (no per-URL scans of
    the whole brand list): exact dict, skeleton dict, deletion indexes (SymSpell style)."""

    def __init__(self, popular: list, brand_top_n: int = BRAND_TOP_N,
                 skip_top_n: int = POPULAR_SKIP_N) -> None:
        self.popular: set = set()            # registered domains never flagged
        self.popular_labels: set = set()     # labels of popular sites (never fuzzy-matched)
        self.brands: dict = {}               # label -> Brand
        self.label_domains: dict = {}        # label -> set of registered domains (google.co.uk ...)
        self.brand_domains: dict = {}        # registered domain -> label (high-risk + top-N)
        for rank, dom in popular:
            ext = EXTRACTOR(dom)
            label = ext.domain
            if rank <= skip_top_n:
                self.popular.add(dom)
                if len(label) >= 4:
                    self.popular_labels.add(label)
            if rank <= brand_top_n and _LABEL_RE.match(label) and len(label) >= 4 \
                    and not any(h in label for h in _INFRA_HINTS):
                self.label_domains.setdefault(label, set()).add(dom)
                self.brand_domains[dom] = label
                self.brands.setdefault(label, Brand(label, dom, rank, False))
        for label, dom in HIGH_RISK_BRANDS.items():       # curated list wins over rank
            old = self.brands.get(label)
            self.brands[label] = Brand(label, dom, old.rank if old else 0, True)
            self.label_domains.setdefault(label, set()).add(dom)
            self.brand_domains[dom] = label
            self.popular.add(dom)
        for label in NOISY_BRANDS:
            self.brands.pop(label, None)

        self.skel: dict = {}
        self.del1: dict = {}
        self.del2: dict = {}
        self.sub_lengths: set = set()
        self.risk_labels: set = set()
        for label, b in sorted(self.brands.items(), key=lambda kv: (not kv[1].risk, kv[1].rank)):
            risk = int(b.risk)
            if b.risk:
                self.risk_labels.add(label)
                if len(label) >= MIN_LEN_SUBSTRING:
                    self.sub_lengths.add(len(label))
            if len(label) >= MIN_LEN_SKELETON[risk]:
                self.skel.setdefault(skeleton(label), b)
            if len(label) >= MIN_LEN_DIST1[risk]:
                for v in _deletions(label, 1):
                    self.del1.setdefault(v, set()).add(label)
            if len(label) >= MIN_LEN_DIST2[risk]:
                for v in _deletions(label, 2):
                    self.del2.setdefault(v, set()).add(label)

    # ───────────── token level ─────────────
    def exact(self, tok: str):
        return self.brands.get(tok)

    def fuzzy(self, tok: str) -> list:
        """Typo-style matches of ONE token (not the exact brand). -> [(technique, Brand, dist)]"""
        out: list = []
        L = len(tok)
        if L < 4 or tok in self.brands or tok in self.popular_labels:
            return out
        b = self.skel.get(skeleton(tok))
        if b and b.label != tok and L >= MIN_LEN_SKELETON[int(b.risk)]:
            out.append(("homoglyph", b, 0))
        for i in range(L - 1):                                   # transposition
            if tok[i] != tok[i + 1]:
                s = tok[:i] + tok[i + 1] + tok[i] + tok[i + 2:]
                b = self.brands.get(s)
                if b and len(b.label) >= MIN_LEN_DIST1[int(b.risk)]:
                    out.append(("transposition", b, 1))
        if L >= 5:
            seen = set()
            for v in _deletions(tok, 1):                         # one substitution/insert/delete
                for lab in self.del1.get(v, ()):
                    if lab in seen or lab == tok or abs(len(lab) - L) > 1:
                        continue
                    seen.add(lab)
                    if osa(tok, lab) == 1:
                        b = self.brands[lab]
                        out.append((classify_dist1(tok, lab), b, 1))
        if L >= 8 and not out:
            seen = set()
            for v in _deletions(tok, 2):                         # two typos, long brands only
                for lab in self.del2.get(v, ()):
                    if lab in seen or lab == tok or abs(len(lab) - L) > 2:
                        continue
                    seen.add(lab)
                    if osa(tok, lab) == 2:
                        out.append(("edit_distance_2", self.brands[lab], 2))
        return out


_INDEX = None


def get_index() -> BrandIndex:
    """Built once per process, on first use (works with multiprocessing 'spawn')."""
    global _INDEX
    if _INDEX is None:
        if not os.path.exists(POPULAR_CSV):
            raise FileNotFoundError(f"{POPULAR_CSV} not found: run preprocess.py first")
        _INDEX = BrandIndex(_read_popular(POPULAR_CSV))
    return _INDEX


# ═════════════════════════ analysis ═════════════════════════
def _match(technique, brand, distance, where, combo=False, score=None) -> dict:
    return dict(technique=technique, brand=brand.label, brand_domain=brand.domain,
                distance=distance, where=where, combo=combo, rank=brand.rank,
                score=SCORE[technique] if score is None else score)


def _parse(url: str) -> tuple:
    s = str(url).strip()
    sp = urlsplit(s if "://" in s else "//" + s)
    try:
        host = (sp.hostname or "").lower().rstrip(".")
    except ValueError:
        host = ""
    return host, sp.path or ""


def _tokens_with_keywords(label: str) -> list:
    """The label plus its hyphen-separated tokens; every candidate is also tried with a
    leading/trailing keyword removed (securepaypal -> paypal, paypallogin -> paypal)."""
    toks = [t for t in label.split("-") if t]
    cands = set(toks) | {label}
    kws = sorted(KEYWORDS, key=len, reverse=True)
    for t in list(cands):
        for kw in kws:
            if t.startswith(kw) and len(t) - len(kw) >= 4:
                cands.add(t[len(kw):])
            if t.endswith(kw) and len(t) - len(kw) >= 4:
                cands.add(t[:-len(kw)])
    return list(cands)


def analyze(url: str) -> dict:
    """Return the strongest lookalike evidence for this URL (all rules, all positions)."""
    idx = get_index()
    res = dict(score=0.0, technique="", brand="", brand_domain="", where="", distance=0,
               combo=False, matches=[])
    host, path = _parse(url)
    if not host or _IP_RE.match(host) or ":" in host or "." not in host:
        return res
    ext = EXTRACTOR(host)
    suffix, label, sub = ext.suffix, ext.domain, ext.subdomain
    reg = f"{label}.{suffix}" if suffix else label
    if reg in ALLOWLIST_DOMAINS:
        return res
    if label.startswith("xn--"):                                    # IDN: look at the real characters
        try:
            label = label.encode("ascii").decode("idna")
        except Exception:
            pass

    platform = reg in SHARED_PLATFORMS
    popular = reg in idx.popular and not platform
    matches: list = []

    # which label is attacker-controlled?
    subparts = [p for p in sub.split(".") if p]
    if platform:
        cand_label, rest_sub, where_label = (subparts[-1] if subparts else ""), subparts[:-1], "tenant"
    else:
        cand_label, rest_sub, where_label = label, subparts, "domain"

    if not popular:
        # 1) a brand domain written inside the subdomain: www.paypal.com.evil.tk
        parts = rest_sub
        for n in (2, 3):
            for i in range(len(parts) - n + 1):
                dom = ".".join(parts[i:i + n])
                if dom in idx.brand_domains and idx.brand_domains[dom] in idx.risk_labels:
                    matches.append(_match("brand_domain_in_subdomain", idx.brands[idx.brand_domains[dom]], 0, "subdomain"))
        # 2) brand name (or a typo of it) as a subdomain token
        for p in parts:
            for tok in p.split("-"):
                b = idx.exact(tok)
                if b and b.risk and len(tok) >= 4:
                    matches.append(_match("brand_in_subdomain", b, 0, "subdomain"))
                else:
                    for tech, bb, d in idx.fuzzy(tok):
                        if bb.risk:
                            matches.append(_match(tech, bb, d, "subdomain", score=SCORE[tech] * 0.9))

    if cand_label and not popular:
        # 3) the registered label (or tenant) itself
        b = idx.exact(cand_label)
        if b:
            if not platform and reg not in idx.label_domains.get(b.label, ()):
                if len(b.label) >= 5 or b.risk:
                    matches.append(_match("tld_swap", b, 0, "domain"))
            elif platform:
                matches.append(_match("brand_in_subdomain", b, 0, "tenant"))
        else:
            for tech, bb, d in idx.fuzzy(cand_label):
                matches.append(_match(tech, bb, d, where_label))
        compact = cand_label.replace("-", "")
        if compact != cand_label:
            b = idx.exact(compact)
            if b:
                matches.append(_match("hyphen_insert", b, 0, where_label))
        glued = "".join(subparts + [cand_label]) if not platform and subparts else ""
        if glued and idx.exact(glued) and len(subparts) >= 1:
            matches.append(_match("dot_split", idx.exact(glued), 0, where_label))
        toks = [t for t in cand_label.split("-") if t]
        for c in _tokens_with_keywords(cand_label):
            if c == cand_label and len(toks) == 1:
                continue                                    # whole label already handled above
            b = idx.exact(c)
            if b and (b.risk or len(toks) > 1) and len(c) >= 4:
                kw_in = len(toks) > 1 and any(t in KEYWORDS for t in toks if t != c)
                stripped = cand_label.replace(c, "", 1)
                if stripped in ("www", "ww") or cand_label in ("www" + c, "www-" + c):
                    matches.append(_match("www_prefix", b, 0, where_label))
                elif b.risk or kw_in:
                    kw = kw_in or stripped in KEYWORDS or any(
                        k and (cand_label.startswith(k) or cand_label.endswith(k)) for k in KEYWORDS
                        if k in stripped and len(k) >= 2)
                    matches.append(_match("combosquat_keyword" if kw else "combosquat", b, 0, where_label, combo=True))
            elif c not in toks or len(toks) > 1:
                for tech, bb, d in idx.fuzzy(c):
                    if c != cand_label or len(toks) > 1:
                        matches.append(_match(tech, bb, d, where_label, combo=True))
        # 4) brand hidden inside a longer label (weak evidence)
        if not any(m["where"] in ("domain", "tenant") for m in matches):
            best = None
            for L in idx.sub_lengths:
                for i in range(len(cand_label) - L + 1):
                    s = cand_label[i:i + L]
                    if s in idx.risk_labels and s != cand_label and (best is None or L > len(best)):
                        best = s
            if best:
                matches.append(_match("brand_substring", idx.brands[best], 0, where_label))

    # 5) brand in the path of a foreign domain
    if path and not popular and not any(m["technique"] == "brand_domain_in_subdomain" for m in matches):
        for tok in re.split(r"[^a-z0-9.]+", path.lower()):
            tok = tok.strip(".")
            if not tok:
                continue
            base = tok.split(".")[0]
            b = idx.exact(base)
            if b and b.risk and len(base) >= 4 and b.domain not in (reg,):
                matches.append(_match("brand_in_path", b, 0, "path"))
                break

    if matches:
        matches.sort(key=lambda m: (-m["score"], m["rank"] or 10**9))
        uniq, seen = [], set()
        for m in matches:
            k = (m["technique"], m["brand"], m["where"])
            if k not in seen:
                seen.add(k)
                uniq.append(m)
        top = uniq[0]
        res.update(score=top["score"], technique=top["technique"], brand=top["brand"],
                   brand_domain=top["brand_domain"], where=top["where"],
                   distance=top["distance"], combo=top["combo"], matches=uniq[:5])
    return res


# ═════════════════════════ ML features ═════════════════════════
FEATURE_NAMES = ["ts_lookalike_score", "ts_edit_distance", "ts_combosquat",
                 "ts_brand_in_subdomain", "ts_brand_in_path"]


def feature_vector(url: str) -> list:
    """5 numeric features for the model. Same function in training and serving."""
    try:
        m = analyze(url)["matches"]
    except Exception:
        return [0.0] * len(FEATURE_NAMES)
    dom = [x for x in m if x["where"] in ("domain", "tenant")]
    score = max((x["score"] for x in dom), default=0.0)
    edit = 0.0
    for x in dom:
        if x["technique"] in EDIT_TECHNIQUES or x["technique"] == "edit_distance_2":
            edit = max(edit, float(x["distance"]) or 1.0)
    combo = 1.0 if any(x["combo"] or x["technique"].startswith("combosquat") for x in dom) else 0.0
    sub = 1.0 if any(x["where"] == "subdomain" and x["score"] >= 0.8 for x in m) else 0.0
    path = 1.0 if any(x["where"] == "path" for x in m) else 0.0
    return [score, edit, combo, sub, path]


# ═════════════════════════ self-test ═════════════════════════
def _selftest() -> int:
    ok_all = True

    def check(name, cond):
        nonlocal ok_all
        ok_all &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name}")

    print("Positive cases (must be flagged, technique shown):")
    positives = [
        ("https://paypa1.com/login", "homoglyph"), ("amaz0n-prime.com", None),
        ("arnazon.com", "homoglyph"), ("rnicrosoft.com", "homoglyph"),
        ("g00gle.com", "homoglyph"), ("gooogle.com", "doubled_letter"),
        ("paypall.com", "doubled_letter"), ("paypl.com", "dropped_letter"),
        ("paypla.com", "transposition"), ("wwwgoogle.com", "www_prefix"),
        ("pay-pal.com", "hyphen_insert"), ("paypal.tk", "tld_swap"),
        ("paypal-secure.com", "combosquat_keyword"), ("securepaypal.com", "combosquat_keyword"),
        ("paypal-account-verify.com", "combosquat_keyword"),
        ("https://www.paypal.com.evil.tk/", "brand_domain_in_subdomain"),
        ("paypal.secure-login.tk", "brand_in_subdomain"),
        ("evil-site.com/paypal/login", "brand_in_path"),
        ("paypa1-login.weebly.com", None), ("paypal.weebly.com", "brand_in_subdomain"),
        ("wellsfargo-secure.top", "combosquat_keyword"), ("wellsfarg0.com", "homoglyph"),
        ("microsooft.com", None), ("faceb00k-login.xyz", None),
    ]
    for url, tech in positives:
        r = analyze(url)
        good = r["score"] >= 0.4 and (tech is None or r["technique"] == tech)
        check(f"{url:<42} -> {r['technique'] or '-':<26} {r['brand'] or '-':<10} {r['where'] or '-':<10} {r['score']:.2f}", good)

    print("\nNegative cases (must NOT be flagged above 0.5):")
    negatives = ["google.com", "https://accounts.google.com/signin", "www.paypal.com/us/home",
                 "paypal.com", "support.apple.com/en-us", "amazon.co.uk", "login.microsoftonline.com",
                 "github.com/google/guava", "en.wikipedia.org/wiki/PayPal", "pineapple.com",
                 "mail.example.com", "docs.google.com/document", "aws.amazon.com", "apple.com.au"]
    for url in negatives:
        r = analyze(url)
        check(f"{url:<42} -> score {r['score']:.2f} {r['technique']}", r["score"] <= 0.5)

    print("\nFeature vector example:", dict(zip(FEATURE_NAMES, feature_vector("paypa1-secure.xyz"))))
    print("\nself-test:", "ALL PASSED" if ok_all else "SOME FAILED")
    return 0 if ok_all else 1


def _fp_report() -> None:
    """False-positive estimate: legitimate sites ranked 1,501-10,000 checked against an
    index that only knows the top 1,500 (as if they were unknown sites)."""
    rows = _read_popular(POPULAR_CSV)
    global _INDEX
    saved = _INDEX
    _INDEX = BrandIndex(rows, brand_top_n=BRAND_TOP_N, skip_top_n=BRAND_TOP_N)
    test = [d for r, d in rows if BRAND_TOP_N < r <= 10_000]
    hits: dict = {}
    for d in test:
        r = analyze(d)
        if r["score"] > 0.5:
            hits[d] = r
    _INDEX = saved
    print(f"\nFalse-positive check on {len(test):,} legitimate sites (ranks {BRAND_TOP_N + 1}-10,000):")
    print(f"   flagged with score > 0.5: {len(hits)} ({100 * len(hits) / max(len(test), 1):.2f}%)")
    by_brand: dict = {}
    for d, r in hits.items():
        by_brand.setdefault(r["brand"], []).append((d, r["technique"]))
    for brand, items in sorted(by_brand.items(), key=lambda kv: -len(kv[1]))[:12]:
        print(f"   {brand:<14} {len(items):>3}  e.g. " + ", ".join(f"{d} ({t})" for d, t in items[:3]))
    print("   -> a brand that keeps showing up here is a generic word: add it to NOISY_BRANDS")


if __name__ == "__main__":
    code = _selftest()
    if os.path.exists(POPULAR_CSV):
        _fp_report()
    sys.exit(code)