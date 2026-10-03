#!/usr/bin/env python3
"""
inspect_raw.py  -  profile every RAW source file (read-only)
------------------------------------------------------------
Run from the project root (venv active):

    python scripts/inspect_raw.py              # 10 samples per source
    python scripts/inspect_raw.py --n 20

Reads  data/raw/: Phish.csv, urlhaus.txt, openphish.txt, data.csv,
                  top-1m.csv, synthetic_malicious.csv,
                  PhiUSIIL_Phishing_URL_Dataset.csv,
                  phishing-links-ACTIVE.txt           (missing ones are skipped)
Writes reports/raw_inspection.txt  (same text as the console; paste it to me)

Sections
    1. Per-source profile (as collected: scheme, case, path, TLDs, concentration)
    2. Random raw samples per source
    3. Cleaning funnel (reproduces preprocessing.py: what each step drops)
    4. Cross-source: malicious domains that are in Tranco, benign/malicious overlap
    5. data.csv internal label conflicts
"""

import argparse
import os
import re
import sys
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import tldextract

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **_):
        return x

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

EXTRACTOR = tldextract.TLDExtract(cache_dir=None, suffix_list_urls=None)
_IP_RE    = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
LINES: list = []


def out(text: str = "") -> None:
    print(text)
    LINES.append(text)


def header(title: str) -> None:
    out("\n" + "=" * 78)
    out(title)
    out("=" * 78)


def _registered(ext) -> str:
    if hasattr(ext, "top_domain_under_public_suffix"):
        return ext.top_domain_under_public_suffix
    return ext.registered_domain


def trunc(s: str, w: int = 110) -> str:
    s = str(s)
    return s if len(s) <= w else s[: w - 1] + "…"


# ───────────────────── loaders (same formats as preprocessing.py) ─────────────────────
def load_sources(raw_dir: str, tranco_n: int) -> tuple:
    src, notes = {}, []

    def p(name):
        return os.path.join(raw_dir, name)

    if os.path.exists(p("Phish.csv")):
        df = pd.read_csv(p("Phish.csv"), dtype=str, low_memory=False)
        total = len(df)
        if "verified" in df.columns:
            df = df[df["verified"].str.lower() == "yes"]
        col = "url" if "url" in df.columns else df.columns[1]
        src["PhishTank (Phish.csv)"] = (df[col].dropna().astype(str), "malicious")
        notes.append(f"Phish.csv: {total:,} rows, {len(df):,} after verified==yes")
    if os.path.exists(p("urlhaus.txt")):
        df = pd.read_csv(p("urlhaus.txt"), comment="#", header=None, dtype=str,
                         on_bad_lines="skip")
        s = df.iloc[:, 2 if df.shape[1] >= 3 else 0].dropna().astype(str)
        src["URLhaus"] = (s[s.str.startswith(("http://", "https://"))], "malicious")
    if os.path.exists(p("openphish.txt")):
        with open(p("openphish.txt"), encoding="utf-8", errors="ignore") as fh:
            s = pd.Series([l.strip() for l in fh if l.strip().startswith(("http://", "https://"))])
        src["OpenPhish"] = (s, "malicious")
    if os.path.exists(p("data.csv")):
        df = pd.read_csv(p("data.csv"), dtype=str, header=None, names=["url", "label"],
                         skiprows=1)
        lab = df["label"].str.lower()
        src["data.csv (bad)"]  = (df.loc[lab == "bad", "url"].dropna(), "malicious")
        src["data.csv (good)"] = (df.loc[lab == "good", "url"].dropna(), "benign")
        other = lab[~lab.isin(["good", "bad"])].value_counts()
        if len(other):
            notes.append(f"data.csv has unexpected labels: {other.to_dict()}")
    if os.path.exists(p("synthetic_malicious.csv")):
        df = pd.read_csv(p("synthetic_malicious.csv"), dtype=str)
        src["Synthetic typosquats"] = (df["url"].dropna(), "malicious")


    # PhiUSIIL (UCI id 967): label 1 = legitimate, 0 = phishing (REVERSED vs this project)
    phi = p("PhiUSIIL_Phishing_URL_Dataset.csv")
    if os.path.exists(phi):
        df = pd.read_csv(phi, dtype=str, low_memory=False,
                         usecols=lambda c: c.strip().lower() in ("url", "label"))
        cols = {c.strip().lower(): c for c in df.columns}
        if "url" in cols and "label" in cols:
            lab  = pd.to_numeric(df[cols["label"]], errors="coerce")
            urls = df[cols["url"]]
            src["PhiUSIIL (legitimate)"] = (urls[lab == 1].dropna().astype(str), "benign")
            src["PhiUSIIL (phishing)"]   = (urls[lab == 0].dropna().astype(str), "malicious")
            notes.append("PhiUSIIL: label 1 = legitimate, 0 = phishing "
                         f"(counts: {lab.value_counts().to_dict()})")
        else:
            notes.append(f"PhiUSIIL file found but no URL/label columns: {list(df.columns)}")

    # Phishing.Database active links: plain text, one URL per line
    pdb = p("phishing-links-ACTIVE.txt")
    if os.path.exists(pdb):
        with open(pdb, encoding="utf-8", errors="ignore") as fh:
            s = pd.Series([l.strip() for l in fh if l.strip() and not l.startswith("#")])
        src["Phishing.Database (active links)"] = (s, "malicious")

    tranco = None
    if os.path.exists(p("top-1m.csv")):
        tranco = pd.read_csv(p("top-1m.csv"), dtype=str, header=None,
                             names=["rank", "domain"])
        tranco["rank"] = pd.to_numeric(tranco["rank"], errors="coerce")
        tranco["domain"] = tranco["domain"].str.strip().str.lower()
        src[f"Tranco top {tranco_n:,} (bare domains)"] = (
            tranco["domain"].head(tranco_n), "benign")
    return src, tranco, notes


# ───────────────────── parsing ─────────────────────
def parse_one(u: str) -> dict:
    s = str(u).strip()
    try:
        has_scheme = "://" in s
        scheme = s.split("://", 1)[0].lower() if has_scheme else ""
        p = urlparse(s if has_scheme else "http://" + s)
        host = p.hostname or ""
        e = EXTRACTOR(host)
        return dict(has_scheme=has_scheme, scheme=scheme, host=host,
                    reg=_registered(e) or host, suffix=e.suffix or "(none)",
                    has_path=bool((p.path or "").strip("/")), has_query=bool(p.query),
                    upper=any(c.isupper() for c in s), non_ascii=any(ord(c) > 127 for c in s),
                    is_ip=bool(_IP_RE.match(host)), length=len(s))
    except Exception:
        return dict(has_scheme=False, scheme="", host="", reg="", suffix="(none)",
                    has_path=False, has_query=False, upper=False, non_ascii=False,
                    is_ip=False, length=len(s))


# ───────────────────── preprocessing.py logic (copied) ─────────────────────
def normalize_url(u):
    try:
        u = str(u).strip().lower()
        u = re.sub(r"\s+", "", u)
        m = re.search(r'\(https?://[^\)]+\)', u)
        if m:
            u = m.group(0)[1:-1]
        if not re.match(r"https?://", u):
            u = "http://" + u
        if u.endswith("/") and len(u) > 8:
            u = u[:-1]
        return u
    except Exception:
        return None


def valid_domain(url: str) -> bool:
    try:
        pr = urlparse(url)
        if not pr.netloc:
            return False
        e = EXTRACTOR(pr.netloc.split(":")[0])
        return bool(e.domain) and bool(e.suffix)
    except Exception:
        return False


# ───────────────────── main ─────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="samples per source")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--raw-dir", default=os.path.join(BASE_DIR, "data", "raw"))
    ap.add_argument("--tranco-n", type=int, default=100_000,
                    help="how many Tranco rows preprocessing.py uses")
    args = ap.parse_args()

    if not os.path.isdir(args.raw_dir):
        sys.exit(f"Raw directory not found: {args.raw_dir}")
    sources, tranco, notes = load_sources(args.raw_dir, args.tranco_n)
    if not sources:
        sys.exit(f"No known raw files found in {args.raw_dir}")

    header("0. FILES FOUND")
    for name, (s, cls) in sources.items():
        out(f"  {name:<40} {len(s):>10,} rows   [{cls}]")
    for n in notes:
        out("  note: " + n)

    # parse every source once
    parsed = {}
    for name, (s, cls) in sources.items():
        out(f"\nParsing {name}...")
        parsed[name] = pd.DataFrame([parse_one(u) for u in tqdm(s, desc="parse")])
        parsed[name]["url"] = s.reset_index(drop=True)

    # ── 1. profile
    header("1. PER-SOURCE PROFILE (as collected, before any cleaning)")
    rows = {}
    for name, d in parsed.items():
        vc = d["reg"].value_counts()
        with_scheme = d[d["has_scheme"]]
        rows[name] = {
            "class": sources[name][1],
            "rows": len(d),
            "unique_url_%": round(100 * d["url"].nunique() / len(d), 1),
            "has_scheme_%": round(100 * d["has_scheme"].mean(), 1),
            "https_%_of_with_scheme": round(100 * (with_scheme["scheme"] == "https").mean(), 1)
                                      if len(with_scheme) else None,
            "has_path_%": round(100 * d["has_path"].mean(), 1),
            "has_query_%": round(100 * d["has_query"].mean(), 1),
            "has_uppercase_%": round(100 * d["upper"].mean(), 1),
            "non_ascii_%": round(100 * d["non_ascii"].mean(), 2),
            "ip_host_%": round(100 * d["is_ip"].mean(), 2),
            "median_len": float(d["length"].median()),
            "distinct_domains": len(vc),
            "top1_domain_%": round(100 * vc.iloc[0] / len(d), 1),
            "top10_domains_%": round(100 * vc.head(10).sum() / len(d), 1),
        }
    out(pd.DataFrame(rows).to_string())
    out("\nTop 6 suffixes and top 6 domains per source:")
    for name, d in parsed.items():
        suf = d["suffix"].value_counts(normalize=True).head(6)
        dom = d["reg"].value_counts().head(6)
        out(f"\n[{name}]")
        out("  suffixes: " + ", ".join(f"{k} {100 * v:.1f}%" for k, v in suf.items()))
        out("  domains : " + ", ".join(f"{k} ({v:,})" for k, v in dom.items()))

    # ── 2. samples
    header(f"2. RANDOM RAW SAMPLES ({args.n} per source, seed {args.seed})")
    for name, d in parsed.items():
        out(f"\n--- {name} ---")
        for u in d["url"].sample(min(args.n, len(d)), random_state=args.seed):
            out("  " + trunc(u))

    # ── 3. cleaning funnel
    header("3. CLEANING FUNNEL (reproduces preprocessing.py steps per source)")
    funnel = {}
    for name, (s, cls) in sources.items():
        n0 = len(s)
        norm = s.map(normalize_url)
        norm = norm[norm.notnull()]
        valid = norm[[valid_domain(u) for u in tqdm(norm, desc=f"validate {name[:18]}")]]
        dedup = valid.drop_duplicates()
        funnel[name] = {
            "raw": n0, "after_normalize": len(norm),
            "after_valid_domain": len(valid), "after_dedupe": len(dedup),
            "lost_%": round(100 * (1 - len(dedup) / n0), 1),
            "scheme_ADDED_by_pipeline_%": round(100 * (1 - parsed[name]["has_scheme"].mean()), 1),
        }
    out(pd.DataFrame(funnel).T.to_string())
    out("\nscheme_ADDED_by_pipeline_% = share of rows whose http:// was invented "
        "by normalize_url / add_scheme. Tranco rows additionally get https:// "
        "in preprocessing.load_alexa.")

    # ── 4. cross-source overlap
    header("4. CROSS-SOURCE: TRUSTED DOMAINS INSIDE THE MALICIOUS DATA")
    if tranco is not None:
        rank = dict(zip(tranco["domain"], tranco["rank"]))
        buckets = [(1_000, "top 1k"), (10_000, "top 10k"), (100_000, "top 100k"),
                   (1_000_000, "top 1M")]
        res = {}
        for name, d in parsed.items():
            if sources[name][1] != "malicious":
                continue
            r = d["reg"].map(rank)
            res[name] = {label: round(100 * (r <= lim).mean(), 2) for lim, label in buckets}
        if res:
            out("% of MALICIOUS rows whose registered domain is in Tranco:")
            out(pd.DataFrame(res).T.to_string())
        allmal = pd.concat([d.assign(src=n) for n, d in parsed.items()
                            if sources[n][1] == "malicious"])
        allmal["rank"] = allmal["reg"].map(rank)
        top = (allmal[allmal["rank"] <= 10_000].groupby("reg")
               .agg(rows=("url", "size"), tranco_rank=("rank", "first"))
               .sort_values("rows", ascending=False).head(20))
        out("\nMost frequent Tranco-top-10k domains inside malicious sources:")
        out(top.to_string())
    else:
        out("top-1m.csv not found; skipped.")

    mal_dom = set(pd.concat([d["reg"] for n, d in parsed.items()
                             if sources[n][1] == "malicious"]))
    ben_dom = set(pd.concat([d["reg"] for n, d in parsed.items()
                             if sources[n][1] == "benign"]))
    out(f"\nRegistered domains in BOTH benign and malicious sources: "
        f"{len(mal_dom & ben_dom):,} "
        f"(of {len(mal_dom):,} malicious / {len(ben_dom):,} benign domains)")

    # ── 5. data.csv conflicts
    header("5. data.csv INTERNAL CONFLICTS")
    if "data.csv (good)" in parsed and "data.csv (bad)" in parsed:
        g, b = parsed["data.csv (good)"], parsed["data.csv (bad)"]
        same_url = set(g["url"].str.lower()) & set(b["url"].str.lower())
        both = set(g["reg"]) & set(b["reg"])
        out(f"Exact URLs labelled both good and bad : {len(same_url):,}")
        out(f"Domains with both good and bad rows   : {len(both):,}")
        top = pd.DataFrame({"good": g["reg"].value_counts(), "bad": b["reg"].value_counts()}) \
                .loc[list(both)].fillna(0).astype(int)
        top["total"] = top.sum(axis=1)
        out(top.sort_values("total", ascending=False).head(15).to_string())
    else:
        out("data.csv not found or lacks both labels; skipped.")

    path = os.path.join(REPORT_DIR, "raw_inspection.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES))
    out(f"\nSaved: {path}")


if __name__ == "__main__":
    main()