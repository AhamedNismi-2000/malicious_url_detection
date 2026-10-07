#!/usr/bin/env python3
"""
generate_typosquatting.py  (v2)  -  synthetic lookalike-domain URLs
-----------------------------------------------------------------------
Run from the project root (venv active). No flags needed, edit SETTINGS below:

    python scripts/generate_typosquatting.py

Needs scripts/typosquat.py and data/reference/popular_domains.csv (from preprocess.py).

Writes
    data/raw/synthetic_malicious.csv   TRAINING techniques  (read by preprocess.py)
    data/raw/synthetic_heldout.csv     HELD-OUT techniques  (NEVER training data:
                                       a challenge set to measure generalisation)
    columns: url, label, technique, brand

Why two files
    The model is trained on some techniques only. The held-out techniques were never
    seen, so detection on them shows whether the model learned "looks like a brand"
    or only memorised the generator's patterns. Evaluate it in train_model/test_model.

What is different from v1
    - ~500 brands from your popular_domains.csv + the curated high-risk brand list
    - 17 techniques (v1: 11, mostly brand+keyword and one TLD list)
    - realistic TLD mix (about half .com), mixed schemes, bare domains, some queries
    - no "$", no private IP addresses, no ftp, no real brand domains
    - a "technique" column for per-technique evaluation
    - preprocess.py keeps this at <= 2% of the real malicious rows and train-only

Honest limit: these are artificial. They are a small supplement (2%) for patterns that
are rare in PhishTank/PhiUSIIL, not a replacement for real phishing data. Real lookalikes
in the real test set are the fair measure.
"""

import os
import random
import re
import sys
from collections import Counter

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import typosquat as ts                                            # noqa: E402

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR    = os.path.join(BASE_DIR, "data", "raw")
OUT_TRAIN  = os.path.join(RAW_DIR, "synthetic_malicious.csv")
OUT_HELD   = os.path.join(RAW_DIR, "synthetic_heldout.csv")
os.makedirs(RAW_DIR, exist_ok=True)

# ═════════════════════════ SETTINGS (edit here) ═════════════════════════
SEED             = 42
N_TRAIN_ROWS     = 6_000     # preprocess.py will cap this at 2% of the real malicious rows
N_HELDOUT_ROWS   = 1_500
N_GENERIC_BRANDS = 500       # top-ranked popular brands (besides the curated high-risk list)
HIGH_RISK_SHARE  = 0.60      # share of rows aimed at the curated high-risk brands

# techniques that go ONLY to the held-out file (never trained on)
HOLDOUT_TECHNIQUES = ["keyboard_adjacent", "bitsquat", "hyphen_insert"]

# relative frequency of each technique
TECHNIQUE_WEIGHTS = {
    "transposition": 5, "keyboard_adjacent": 6, "doubled_letter": 6, "dropped_letter": 6,
    "plural": 2, "bitsquat": 3, "homoglyph_multichar": 6, "homoglyph_leet": 8,
    "idn_homoglyph": 3, "hyphen_insert": 4, "dot_split": 3, "www_prefix": 3,
    "tld_swap": 3, "edit_distance_2": 4, "combosquat_domain": 18,
    "combosquat_subdomain": 12, "combosquat_path": 8,
}

SCHEME_MIX = [("https", 0.55), ("http", 0.35), ("", 0.10)]          # "" = bare domain
PATH_MIX   = [("none", 0.55), ("path", 0.35), ("query", 0.10)]

TLD_WEIGHTS = [
    ("com", 50), ("net", 6), ("org", 3), ("xyz", 5), ("top", 5), ("online", 3), ("site", 3),
    ("info", 3), ("co", 2), ("io", 2), ("app", 2), ("shop", 2), ("club", 2), ("live", 2),
    ("click", 1.5), ("link", 1.5), ("tk", 1.5), ("ml", 1), ("ga", 1), ("cf", 0.5),
    ("gq", 0.5), ("ru", 1.5), ("us", 1.5), ("biz", 1.5), ("cc", 1), ("icu", 1),
    ("store", 1), ("dev", 1), ("website", 1), ("support", 0.5), ("page", 0.5),
]

PATHS = [
    "/login", "/signin", "/verify", "/account/update", "/secure/login.php", "/webscr",
    "/auth/session", "/index.php", "/wp-content/login.html", "/app/signin", "/confirm",
    "/security/check", "/billing/update", "/recover/password", "/id/verify",
    "/customer/login.html", "/update/payment", "/account/suspended", "/signin.php",
]
EVIL_WORDS = ["secure", "login", "verify", "account", "update", "support", "alert", "service",
              "center", "portal", "online", "web", "mail", "docs", "cloud", "pay", "wallet",
              "help", "auth", "check", "safe", "view", "notice", "review", "claim"]
NEUTRAL_WORDS = ["blue", "north", "delta", "prime", "nova", "sky", "smart", "global", "bright",
                 "rapid", "metro", "urban", "alpha", "omega", "river", "stone", "green", "swift"]
GEN_KEYWORDS = sorted(k for k in ts.KEYWORDS if k not in ("www", "ww", "id", "the", "my") or k == "my")

LEET = {"o": "0", "i": "1", "l": "1", "e": "3", "a": "4", "s": "5", "t": "7"}
CYRILLIC = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "y": "у", "i": "і"}
MULTI = {"m": "rn", "w": "vv", "d": "cl"}
RISK_ONLY = {"combosquat_domain", "combosquat_subdomain", "combosquat_path", "www_prefix",
             "edit_distance_2", "tld_swap"}
SWAP_TLDS = ["xyz", "top", "online", "site", "tk", "ml", "ga", "cf", "gq", "icu", "click",
             "link", "club", "live", "shop", "support"]
_VALID_LABEL = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$")


# ═════════════════════════ label-level mutations ═════════════════════════
def m_transposition(b, rng):
    pos = [i for i in range(len(b) - 1) if b[i] != b[i + 1]]
    if not pos:
        return None
    i = rng.choice(pos)
    return b[:i] + b[i + 1] + b[i] + b[i + 2:]


def m_keyboard(b, rng):
    pos = [i for i in range(len(b)) if b[i] in ts.KEYBOARD]
    if not pos:
        return None
    i = rng.choice(pos)
    return b[:i] + rng.choice(ts.KEYBOARD[b[i]]) + b[i + 1:]


def m_doubled(b, rng):
    i = rng.randrange(len(b))
    return b[:i] + b[i] + b[i:]


def m_dropped(b, rng):
    if len(b) < 5:
        return None
    i = rng.randrange(1, len(b))
    return b[:i] + b[i + 1:]


def m_plural(b, rng):
    return b + "s"


def m_bitsquat(b, rng):
    idx = list(range(len(b)))
    rng.shuffle(idx)
    for i in idx:
        bits = list(range(7))
        rng.shuffle(bits)
        for k in bits:
            n = chr(ord(b[i]) ^ (1 << k))
            if n != b[i] and re.match(r"^[a-z0-9]$", n):
                return b[:i] + n + b[i + 1:]
    return None


def m_multichar(b, rng):
    pos = [i for i in range(len(b)) if b[i] in MULTI]
    if not pos:
        return None
    i = rng.choice(pos)
    return b[:i] + MULTI[b[i]] + b[i + 1:]


def m_leet(b, rng):
    pos = [i for i in range(len(b)) if b[i] in LEET]
    if not pos:
        return None
    k = 1 if rng.random() < 0.7 or len(pos) == 1 else 2
    chosen = set(rng.sample(pos, k))
    return "".join(LEET[c] if i in chosen else c for i, c in enumerate(b))


def m_idn(b, rng):
    pos = [i for i in range(len(b)) if b[i] in CYRILLIC]
    if not pos:
        return None
    i = rng.choice(pos)
    uni = b[:i] + CYRILLIC[b[i]] + b[i + 1:]
    try:
        return uni.encode("idna").decode("ascii")
    except Exception:
        return None


def m_hyphen(b, rng):
    if len(b) < 5:
        return None
    i = rng.randrange(2, len(b) - 1)
    return b[:i] + "-" + b[i:]


def m_two_typos(b, rng):
    if len(b) < 8:
        return None
    out = b
    for fn in rng.sample([m_dropped, m_doubled, m_keyboard, m_transposition], 2):
        nxt = fn(out, rng)
        if nxt is None:
            return None
        out = nxt
    return out if ts.osa(out, b) == 2 else None


# ═════════════════════════ host / URL assembly ═════════════════════════
class Gen:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)
        idx = ts.get_index()
        self.idx = idx
        risk = sorted(l for l, b in idx.brands.items() if b.risk and len(l) >= 5)
        generic = sorted((b for b in idx.brands.values()
                          if not b.risk and 5 <= len(b.label) and 0 < b.rank <= N_GENERIC_BRANDS),
                         key=lambda b: b.rank)
        self.risk_brands = risk
        self.generic_brands = [b.label for b in generic]
        self.generic_typo_brands = [l for l in self.generic_brands if len(l) >= 7]
        self.tlds, self.tld_w = zip(*TLD_WEIGHTS)

    def brand(self, technique: str) -> str:
        """Keyword / subdomain / path / www / 2-typo rules protect the curated high-risk
        brands only (generic words would fire everywhere), so those techniques are
        generated for high-risk brands only. Typo and homoglyph techniques also use
        generic brands, but only long ones (short words collide with ordinary words)."""
        if technique in RISK_ONLY or self.rng.random() < HIGH_RISK_SHARE:
            return self.rng.choice(self.risk_brands)
        return self.rng.choice(self.generic_typo_brands or self.risk_brands)

    def tld(self) -> str:
        return self.rng.choices(self.tlds, weights=self.tld_w)[0]

    def evil_domain(self) -> str:
        r = self.rng
        a = r.choice(EVIL_WORDS + NEUTRAL_WORDS)
        b = r.choice(EVIL_WORDS + NEUTRAL_WORDS)
        form = r.random()
        if form < 0.45:
            name = f"{a}-{b}"
        elif form < 0.80:
            name = f"{a}{b}"
        else:
            name = f"{a}-{b}{r.randint(10, 9999)}"
        return f"{name}.{self.tld()}"

    def keyword(self) -> str:
        return self.rng.choice(GEN_KEYWORDS)

    def make(self, technique: str):
        """Return (host, forced_path_or_None, brand) or None."""
        r, b = self.rng, self.brand(technique)
        fn = {"transposition": m_transposition, "keyboard_adjacent": m_keyboard,
              "doubled_letter": m_doubled, "dropped_letter": m_dropped, "plural": m_plural,
              "bitsquat": m_bitsquat, "homoglyph_multichar": m_multichar,
              "homoglyph_leet": m_leet, "idn_homoglyph": m_idn, "hyphen_insert": m_hyphen,
              "edit_distance_2": m_two_typos}.get(technique)
        if fn:
            lab = fn(b, r)
            if not lab or lab == b:
                return None
            if technique == "homoglyph_leet" and r.random() < 0.35:     # leet inside a combo
                lab = f"{lab}-{self.keyword()}"
            return f"{lab}.{self.tld()}", None, b
        if technique == "dot_split":
            if len(b) < 5:
                return None
            i = r.randrange(2, len(b) - 1)
            return f"{b[:i]}.{b[i:]}.{self.tld()}", None, b
        if technique == "www_prefix":
            return f"{'www' if r.random() < 0.7 else 'www-'}{b}.{self.tld()}", None, b
        if technique == "tld_swap":
            return f"{b}.{r.choice(SWAP_TLDS)}", None, b
        if technique == "combosquat_domain":
            k, k2 = self.keyword(), self.keyword()
            form = r.choice(["{b}-{k}", "{k}-{b}", "{b}-{k}-{k2}", "{b}{k}", "{k}{b}",
                             "{k}-{b}-{k2}", "{b}-{k}{n}"])
            lab = form.format(b=b, k=k, k2=k2, n=r.randint(1, 99))
            return f"{lab}.{self.tld()}", None, b
        if technique == "combosquat_subdomain":
            ev, k = self.evil_domain(), self.keyword()
            form = r.choice(["{b}.com.{ev}", "www.{b}.com.{ev}", "{b}.{k}.{ev}",
                             "{k}.{b}.{ev}", "{b}-{k}.{ev}.", "{k}.{b}.com.{ev}"])
            host = form.format(b=b, k=k, ev=ev).replace("..", ".").rstrip(".")
            return host, None, b
        if technique == "combosquat_path":
            ev, k = self.evil_domain(), self.keyword()
            path = r.choice(["/{b}/{k}", "/{b}.com/{k}", "/{k}/{b}", "/{b}/{k}.php",
                             "/{b}-{k}/index.html"]).format(b=b, k=k)
            return ev, path, b
        return None

    def url(self, host: str, forced_path) -> str:
        r = self.rng
        scheme = r.choices([s for s, _ in SCHEME_MIX], weights=[w for _, w in SCHEME_MIX])[0]
        style = "path" if forced_path else r.choices([s for s, _ in PATH_MIX],
                                                     weights=[w for _, w in PATH_MIX])[0]
        path = forced_path or ""
        if style in ("path", "query") and not forced_path:
            path = r.choice(PATHS)
        if style == "query" or (forced_path and r.random() < 0.10):
            path += "?id=%x&token=%x" % (r.getrandbits(32), r.getrandbits(48))
        if style == "none" and r.random() < 0.3:
            path = "/"
        return (f"{scheme}://" if scheme else "") + host + path


def host_ok(host: str, idx, brand: str, technique: str = "") -> bool:
    if len(host) > 200 or any(c in host for c in "$ _@"):
        return False
    for lab in host.split("."):
        if not _VALID_LABEL.match(lab) or len(lab) > 63:
            return False
    ext = ts.EXTRACTOR(host)
    reg = f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain
    if not ext.suffix or reg in idx.popular or (
            ext.domain in idx.popular_labels and technique != "tld_swap"):
        return False                                  # a real, popular site: never "malicious"
    return reg not in idx.label_domains.get(brand, ())


def build(g: Gen, techniques: list, n_rows: int, seen: set) -> list:
    weights = [TECHNIQUE_WEIGHTS[t] for t in techniques]
    rows, tries = [], 0
    while len(rows) < n_rows and tries < n_rows * 60:
        tries += 1
        tech = g.rng.choices(techniques, weights=weights)[0]
        made = g.make(tech)
        if not made:
            continue
        host, path, brand = made
        if not host_ok(host, g.idx, brand, tech):
            continue
        url = g.url(host, path)
        key = url.lower()
        if key in seen:
            continue
        seen.add(key)
        rows.append((url, "malicious", tech, brand))
    return rows


def report(name: str, rows: list) -> None:
    df = pd.DataFrame(rows, columns=["url", "label", "technique", "brand"])
    print(f"\n{name}: {len(df):,} rows, {df['brand'].nunique()} brands")
    t = df["technique"].value_counts()
    flagged = {}
    for tech, sub in df.groupby("technique"):
        hit = sum(ts.analyze(u)["score"] >= 0.4 for u in sub["url"])
        flagged[tech] = 100 * hit / len(sub)
    print("   technique                 rows   rule-layer detection (informational)")
    for tech, n in t.items():
        print(f"   {tech:<24} {n:>5}   {flagged[tech]:5.1f}%")
    sch = df["url"].str.extract(r"^(https?)://", expand=False).fillna("(bare)").value_counts(normalize=True)
    print("   scheme mix: " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in sch.items()))
    from urllib.parse import urlsplit
    sp = df["url"].map(lambda u: urlsplit(u if "://" in u else "//" + u))
    has_path = sp.map(lambda x: bool(x.path.strip("/"))).mean()
    has_q = sp.map(lambda x: bool(x.query)).mean()
    print(f"   has path: {100 * has_path:.0f}% | has query: {100 * has_q:.0f}%")


def main() -> None:
    print("=" * 70)
    print("SYNTHETIC LOOKALIKE URL GENERATOR v2")
    print("=" * 70)
    g = Gen(SEED)
    print(f"brands: {len(g.risk_brands)} high-risk + {len(g.generic_brands)} generic")
    held = [t for t in TECHNIQUE_WEIGHTS if t in HOLDOUT_TECHNIQUES]
    train = [t for t in TECHNIQUE_WEIGHTS if t not in HOLDOUT_TECHNIQUES]
    seen: set = set()
    tr = build(g, train, N_TRAIN_ROWS, seen)
    ho = build(g, held, N_HELDOUT_ROWS, seen)

    for rows, path in ((tr, OUT_TRAIN), (ho, OUT_HELD)):
        df = pd.DataFrame(rows, columns=["url", "label", "technique", "brand"])
        assert not df["url"].str.contains(r"[$]").any(), "found a $"
        assert df["url"].map(lambda u: u.isascii()).all(), "non-ascii URL"
        df.to_csv(path, index=False, encoding="utf-8")
    report("TRAIN file", tr)
    report("HELD-OUT file", ho)
    print("\nSamples:")
    for tech in ("homoglyph_leet", "combosquat_subdomain", "combosquat_path", "idn_homoglyph"):
        ex = [r[0] for r in tr if r[2] == tech][:3]
        print(f"   {tech:<22} " + "  ".join(ex))
    print(f"\nWrote {OUT_TRAIN}\nWrote {OUT_HELD}")
    print("Next: python scripts/preprocess.py   (loads only synthetic_malicious.csv)")


if __name__ == "__main__":
    main()