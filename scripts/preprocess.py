#!/usr/bin/env python3
"""
preprocessing.py  (v2)  -  build ONE clean, documented dataset from all raw sources
------------------------------------------------------------------------------------
Run from the project root (venv active):

    python scripts/preprocessing.py
    python scripts/preprocessing.py --max-per-group 15 --out-dir data/processed

Needs scripts/url_utils.py next to this file.

Raw files looked for in data/raw/ (missing ones are skipped with a message):
    Phish.csv                          PhishTank           malicious (phishing)
    urlhaus.txt                        URLhaus             malicious (malware)
    openphish.txt                      OpenPhish           malicious (phishing)
    data.csv                           faizann24 (old)     good / bad
    PhiUSIIL_Phishing_URL_Dataset.csv  PhiUSIIL            label 1=legit, 0=phish
    phishing-links-ACTIVE.txt          Phishing.Database   malicious (phishing)
    synthetic_malicious.csv            typosquats          malicious (synthetic)
    top-1m.csv                         Tranco              benign (rank-bucket sample)
    benign_crawled.csv                 your crawler        benign, column "url"

Outputs
    data/processed/cleaned_urls_v2.csv       the dataset (schema below)
    reports/preprocessing_report.json        every count, step by step
    reports/preprocessing_conflicts.csv      groups that carry both labels
    data/reference/popular_domains.csv       Tranco top-N (for typosquat.py)

Schema of cleaned_urls_v2.csv
    url         canonical, scheme-less, no "www."  <- the model input
    label       0 benign / 1 malicious
    source      which raw file / feed
    label_type  benign, phishing, malware, malicious, synthetic_typosquat
    scheme      scheme observed in the source ("" = unknown; never invented)
    host, reg_domain, suffix, group_key
    has_path, has_query, is_ip, is_synthetic
    url_raw     exactly as collected (debugging only)

Pipeline (each step is counted per source in the report)
    load -> canonicalize -> dedupe -> label conflicts -> per-group cap ->
    source caps -> synthetic cap -> class matching -> final checks -> write
"""

import argparse
import json
import os
import sys
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from url_utils import canonicalize                      # noqa: E402

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **_):
        return x

# ═════════════════════════ SETTINGS (edit here) ═════════════════════════
CONFIG = dict(
    SEED=42,
    MAX_PER_GROUP=25,        # max rows per (label, group_key): kills domain dominance
    MAX_PER_PLATFORM=400,    # max rows per (label, registered domain) for shared platforms
                             # and big multi-owner sites (weebly, web.app, google.com...)
    MAX_PER_IP=3,            # max rows per IP host
    SYNTHETIC_FRACTION=0.02, # synthetic rows <= this share of the REAL malicious rows
    IP_MAX_FRACTION=0.05,    # IP-host rows <= this share of final malicious rows
    MATCH_COLS=["has_path", "has_query"],   # classes are matched on these
    SOURCE_CAPS={},          # e.g. {"PhiUSIIL (legitimate)": 60000}
    # Tranco sampling: (rank_from, rank_to, rows_to_sample). Long tail on purpose.
    TRANCO_BUCKETS=[(1, 1_000, 1_000), (1_001, 10_000, 6_000),
                    (10_001, 100_000, 25_000), (100_001, 1_000_000, 60_000)],
    DROP_TRANCO_SHARED_ROOTS=True,   # drop roots like web.app / github.io from benign
    POPULAR_REF_N=10_000,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ═════════════════════════ funnel bookkeeping ═════════════════════════
class Funnel:
    """Counts rows per source after every step, plus drop reasons."""

    def __init__(self) -> None:
        self.steps: OrderedDict = OrderedDict()
        self.reasons: Counter = Counter()

    def snap(self, step: str, df: pd.DataFrame) -> None:
        self.steps[step] = df["source"].value_counts().to_dict()

    def snap_raw(self, step: str, counts: dict) -> None:
        self.steps[step] = dict(counts)

    def table(self) -> pd.DataFrame:
        t = pd.DataFrame(self.steps).fillna(0).astype(int)
        first, last = t.columns[0], t.columns[-1]
        t["kept_%"] = (100 * t[last] / t[first].replace(0, np.nan)).round(1)
        return t


def say(msg: str = "") -> None:
    print(msg, flush=True)


# ═════════════════════════ loaders ═════════════════════════
def load_sources(raw_dir: str, cfg: dict) -> tuple:
    """Return (list of source dicts, tranco_frame_or_None)."""
    out: list = []
    rng_seed = cfg["SEED"]

    def p(name):
        return os.path.join(raw_dir, name)

    def add(name, label, ltype, urls, synthetic=False):
        urls = pd.Series(urls).dropna().astype(str).reset_index(drop=True)
        out.append(dict(name=name, label=label, label_type=ltype,
                        synthetic=synthetic, urls=urls))
        say(f"  loaded {name:<38} {len(urls):>9,} rows")

    def attempt(fname, fn):
        if not os.path.exists(p(fname)):
            say(f"  (skip) {fname} not found")
            return
        try:
            fn()
        except Exception as exc:                                  # keep going
            say(f"  [ERROR] {fname}: {exc}")

    def phishtank():
        df = pd.read_csv(p("Phish.csv"), dtype=str, low_memory=False)
        if "verified" in df.columns:
            df = df[df["verified"].str.lower() == "yes"]
        col = "url" if "url" in df.columns else df.columns[1]
        add("PhishTank", 1, "phishing", df[col])

    def urlhaus():
        df = pd.read_csv(p("urlhaus.txt"), comment="#", header=None, dtype=str,
                         on_bad_lines="skip")
        s = df.iloc[:, 2 if df.shape[1] >= 3 else 0].dropna().astype(str)
        add("URLhaus", 1, "malware", s[s.str.startswith(("http://", "https://"))])

    def openphish():
        with open(p("openphish.txt"), encoding="utf-8", errors="ignore") as fh:
            add("OpenPhish", 1, "phishing",
                [l.strip() for l in fh if l.strip().startswith(("http://", "https://"))])

    def data_csv():
        df = pd.read_csv(p("data.csv"), dtype=str, header=None,
                         names=["url", "label"], skiprows=1)
        lab = df["label"].str.lower()
        add("data.csv (bad)", 1, "malicious", df.loc[lab == "bad", "url"])
        add("data.csv (good)", 0, "benign", df.loc[lab == "good", "url"])

    def phiusiil():
        df = pd.read_csv(p("PhiUSIIL_Phishing_URL_Dataset.csv"), dtype=str,
                         low_memory=False,
                         usecols=lambda c: c.strip().lower() in ("url", "label"))
        cols = {c.strip().lower(): c for c in df.columns}
        lab = pd.to_numeric(df[cols["label"]], errors="coerce")
        add("PhiUSIIL (legitimate)", 0, "benign", df.loc[lab == 1, cols["url"]])
        add("PhiUSIIL (phishing)", 1, "phishing", df.loc[lab == 0, cols["url"]])

    def phishing_db():
        with open(p("phishing-links-ACTIVE.txt"), encoding="utf-8", errors="ignore") as fh:
            add("Phishing.Database", 1, "phishing",
                [l.strip() for l in fh if l.strip() and not l.startswith("#")])

    def synthetic():
        df = pd.read_csv(p("synthetic_malicious.csv"), dtype=str)
        add("Synthetic typosquats", 1, "synthetic_typosquat", df["url"], synthetic=True)

    def crawled():
        df = pd.read_csv(p("benign_crawled.csv"), dtype=str)
        add("Crawled benign", 0, "benign", df["url"])

    for fname, fn in [("Phish.csv", phishtank), ("urlhaus.txt", urlhaus),
                      ("openphish.txt", openphish), ("data.csv", data_csv),
                      ("PhiUSIIL_Phishing_URL_Dataset.csv", phiusiil),
                      ("phishing-links-ACTIVE.txt", phishing_db),
                      ("synthetic_malicious.csv", synthetic),
                      ("benign_crawled.csv", crawled)]:
        attempt(fname, fn)

    tranco = None
    if os.path.exists(p("top-1m.csv")):
        try:
            tr = pd.read_csv(p("top-1m.csv"), dtype=str, header=None,
                             names=["rank", "domain"])
            tr["rank"] = pd.to_numeric(tr["rank"], errors="coerce")
            tr["domain"] = tr["domain"].str.strip().str.lower()
            tr = tr.dropna()
            tranco = tr
            parts = []
            for lo, hi, n in cfg["TRANCO_BUCKETS"]:
                b = tr[(tr["rank"] >= lo) & (tr["rank"] <= hi)]
                parts.append(b.sample(min(n, len(b)), random_state=rng_seed))
            sample = pd.concat(parts)
            add("Tranco (rank-bucket sample)", 0, "benign", sample["domain"])
        except Exception as exc:
            say(f"  [ERROR] top-1m.csv: {exc}")
    else:
        say("  (skip) top-1m.csv not found")
    return out, tranco


# ═════════════════════════ steps ═════════════════════════
def canonicalize_all(sources: list, funnel: Funnel) -> pd.DataFrame:
    rows: list = []
    for src in sources:
        for raw in tqdm(src["urls"], desc=f"canon {src['name'][:22]}", leave=False):
            c, reason = canonicalize(raw)
            if c is None:
                funnel.reasons[(src["name"], reason)] += 1
                continue
            c.update(source=src["name"], label=src["label"],
                     label_type=src["label_type"], is_synthetic=src["synthetic"],
                     url_raw=str(raw).replace("\n", " ").replace("\r", " "))
            rows.append(c)
    df = pd.DataFrame(rows)
    df["label"] = df["label"].astype(int)
    return df


def dedupe_and_conflicts(df: pd.DataFrame, funnel: Funnel, rng_seed: int) -> tuple:
    df = df.sample(frac=1, random_state=rng_seed).reset_index(drop=True)
    df["_k"] = df["url"].str.lower()
    n_lab = df.groupby("_k")["label"].transform("nunique")
    n_conflict = int((n_lab > 1).sum())
    df = df[n_lab == 1]
    funnel.snap("no_url_label_conflict", df)
    n_before = len(df)
    df = df.drop_duplicates(subset="_k").drop(columns="_k")
    funnel.snap("deduplicated", df)
    return df.reset_index(drop=True), n_conflict, n_before - len(df)


def group_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    c = (df.groupby(["group_key", "label"]).size().unstack(fill_value=0)
           .rename(columns={0: "n_benign", 1: "n_malicious"}))
    for col in ("n_benign", "n_malicious"):
        if col not in c.columns:
            c[col] = 0
    c = c[(c["n_benign"] > 0) & (c["n_malicious"] > 0)].copy()
    c["total"] = c["n_benign"] + c["n_malicious"]
    return c.sort_values("total", ascending=False)


def cap_per_group(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    df = df.sample(frac=1, random_state=cfg["SEED"]).reset_index(drop=True)
    pos = df.groupby(["label", "group_key"]).cumcount()
    cap = np.where(df["is_ip"], cfg["MAX_PER_IP"], cfg["MAX_PER_GROUP"])
    df = df[pos < cap]
    # second pass: platforms whose tenants/hostnames are separate groups but which
    # would still dominate a class (e.g. 4,000 different x.weebly.com tenants)
    platform = (df["group_key"] != df["reg_domain"]) & (~df["is_ip"])
    pos2 = df[platform].groupby(["label", "reg_domain"]).cumcount()
    drop = pos2.index[pos2 >= cfg["MAX_PER_PLATFORM"]]
    return df.drop(index=drop).reset_index(drop=True)


def cap_sources(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    for name, cap in cfg["SOURCE_CAPS"].items():
        idx = df.index[df["source"] == name]
        if len(idx) > cap:
            drop = np.random.RandomState(cfg["SEED"]).choice(idx, len(idx) - cap, replace=False)
            df = df.drop(index=drop)
    return df.reset_index(drop=True)


def cap_synthetic(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    real_mal = int(((df["label"] == 1) & (~df["is_synthetic"])).sum())
    limit = int(cfg["SYNTHETIC_FRACTION"] * real_mal)
    idx = df.index[df["is_synthetic"]]
    if len(idx) > limit:
        drop = np.random.RandomState(cfg["SEED"]).choice(idx, len(idx) - limit, replace=False)
        df = df.drop(index=drop)
    return df.reset_index(drop=True)


def match_classes(df: pd.DataFrame, cfg: dict) -> tuple:
    """Make benign and malicious share the same (has_path, has_query) mix, so those
    format features cannot separate the classes. IP-host rows are exempt (benign
    IP URLs barely exist) and limited to IP_MAX_FRACTION of malicious."""
    cols, seed = cfg["MATCH_COLS"], cfg["SEED"]
    non_ip, ip = df[~df["is_ip"]], df[df["is_ip"]]

    before = (non_ip.groupby(cols + ["label"]).size().unstack(fill_value=0)
              .rename(columns={0: "benign", 1: "malicious"}))
    kept = []
    for _, g in non_ip.groupby(cols):
        b, m = g[g["label"] == 0], g[g["label"] == 1]
        n = min(len(b), len(m))
        if n:
            kept += [b.sample(n, random_state=seed), m.sample(n, random_state=seed)]
    matched = pd.concat(kept) if kept else non_ip.iloc[0:0]
    after = (matched.groupby(cols + ["label"]).size().unstack(fill_value=0)
             .rename(columns={0: "benign", 1: "malicious"}))

    scarce = before[(before.min(axis=1) == 0) & (before.max(axis=1) > 0)]
    if len(scarce):
        lost = int(scarce.sum().sum())
        say(f"   [WARN] {len(scarce)} stratum(s) contain only ONE class, so {lost:,} rows "
            "are dropped by matching:")
        say(scarce.to_string())
        say("   -> add data for the missing class in that stratum, or remove that column "
            "from MATCH_COLS")

    n_mal = int((matched["label"] == 1).sum())
    ip_mal = ip[ip["label"] == 1]
    ip_limit = int(cfg["IP_MAX_FRACTION"] / (1 - cfg["IP_MAX_FRACTION"]) * n_mal)
    if len(ip_mal) > ip_limit:
        ip_mal = ip_mal.sample(ip_limit, random_state=seed)
    final = pd.concat([matched, ip_mal, ip[ip["label"] == 0]])
    final = final.sample(frac=1, random_state=seed).reset_index(drop=True)
    return final, before, after


# ═════════════════════════ main ═════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=os.path.join(BASE_DIR, "data", "raw"))
    ap.add_argument("--out-dir", default=os.path.join(BASE_DIR, "data", "processed"))
    ap.add_argument("--report-dir", default=os.path.join(BASE_DIR, "reports"))
    ap.add_argument("--ref-dir", default=os.path.join(BASE_DIR, "data", "reference"))
    ap.add_argument("--max-per-group", type=int)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()

    cfg = dict(CONFIG)
    if args.max_per_group:
        cfg["MAX_PER_GROUP"] = args.max_per_group
    if args.seed:
        cfg["SEED"] = args.seed
    for d in (args.out_dir, args.report_dir, args.ref_dir):
        os.makedirs(d, exist_ok=True)

    say("=" * 70)
    say("PREPROCESSING v2")
    say("=" * 70)
    say("\n1. Loading raw sources")
    sources, tranco = load_sources(args.raw_dir, cfg)
    if not sources:
        sys.exit(f"No raw files found in {args.raw_dir}")

    funnel = Funnel()
    funnel.snap_raw("raw", {s["name"]: len(s["urls"]) for s in sources})

    say("\n2. Canonicalizing (scheme removed, www removed, &amp; decoded, validated)")
    df = canonicalize_all(sources, funnel)
    funnel.snap("canonicalized", df)

    if cfg["DROP_TRANCO_SHARED_ROOTS"]:
        is_tranco = df["source"].str.startswith("Tranco")
        shared_root = is_tranco & df["shared_platform"]      # github.io, web.app ...
        say(f"   dropping {int(shared_root.sum())} Tranco rows that are shared-platform roots")
        df = df[~shared_root].reset_index(drop=True)
        funnel.snap("no_shared_roots", df)

    say("\n3. Removing duplicates and URLs labelled both ways")
    df, n_conflict, n_dup = dedupe_and_conflicts(df, funnel, cfg["SEED"])
    say(f"   URL-level label conflicts removed: {n_conflict:,} | duplicates: {n_dup:,}")

    conf = group_conflicts(df)
    conf.to_csv(os.path.join(args.report_dir, "preprocessing_conflicts.csv"))
    say(f"   groups carrying both labels (kept, reported): {len(conf):,}")

    say(f"\n4. Capping rows: per group <= {cfg['MAX_PER_GROUP']}, per IP <= {cfg['MAX_PER_IP']}, "
        f"per platform <= {cfg['MAX_PER_PLATFORM']}")
    df = cap_per_group(df, cfg)
    funnel.snap("group_capped", df)

    if cfg["SOURCE_CAPS"]:
        df = cap_sources(df, cfg)
        funnel.snap("source_capped", df)

    say(f"\n5. Limiting synthetic rows to {cfg['SYNTHETIC_FRACTION']:.0%} of real malicious")
    df = cap_synthetic(df, cfg)
    funnel.snap("synthetic_capped", df)

    say(f"\n6. Matching classes on {cfg['MATCH_COLS']} (IP hosts exempt)")
    pool_stats = {c: df.groupby("label")[c].mean().round(3).to_dict()
                  for c in ("has_path", "has_query", "is_ip")}
    df, before, after = match_classes(df, cfg)
    funnel.snap("class_matched", df)
    say("   strata before matching:\n" + before.to_string())
    say("   strata after matching:\n" + after.to_string())

    # ───────── final checks (fail loudly) ─────────
    say("\n7. Final checks")
    assert df["url"].notna().all() and df["label"].isin([0, 1]).all(), "null url or bad label"
    assert not df["url"].str.lower().duplicated().any(), "duplicate URLs survived"
    assert df["group_key"].notna().all(), "missing group_key"
    n_b, n_m = int((df.label == 0).sum()), int((df.label == 1).sum())
    ratio = n_m / max(n_b, 1)
    say(f"   benign {n_b:,} | malicious {n_m:,} | ratio {ratio:.3f}")
    if not 0.85 <= ratio <= 1.15:
        say("   [WARN] classes differ by more than 15%: check the strata tables above")

    cols = ["url", "label", "source", "label_type", "scheme", "host", "reg_domain",
            "suffix", "group_key", "has_path", "has_query", "is_ip", "is_synthetic",
            "url_raw"]
    out_path = os.path.join(args.out_dir, "cleaned_urls_v2.csv")
    df[cols].to_csv(out_path, index=False, encoding="utf-8")

    if tranco is not None:
        ref = tranco[tranco["rank"] <= cfg["POPULAR_REF_N"]].sort_values("rank")
        ref.assign(rank=ref["rank"].astype(int)).to_csv(
            os.path.join(args.ref_dir, "popular_domains.csv"), index=False, header=False)

    # ───────── report ─────────
    tbl = funnel.table()
    say("\n" + "=" * 70)
    say("FUNNEL: rows per source after each step")
    say("=" * 70)
    say(tbl.to_string())
    if funnel.reasons:
        say("\nDrop reasons during canonicalization (source, reason): count")
        for (s, r), n in sorted(funnel.reasons.items(), key=lambda x: -x[1])[:15]:
            say(f"   {s:<34} {r:<18} {n:>8,}")

    comp = pd.crosstab(df["source"], df["label"].map({0: "benign", 1: "malicious"}))
    say("\nFinal composition by source:")
    say(comp.to_string())
    final_stats = {c: df.groupby("label")[c].mean().round(3).to_dict()
                   for c in ("has_path", "has_query", "is_ip")}
    say(f"\nhas_path / has_query / is_ip per class (label 0, 1):\n   {final_stats}")
    say(f"\nWrote {out_path}")

    report = dict(config={k: (v if not isinstance(v, dict) else v) for k, v in cfg.items()},
                  funnel=tbl.reset_index().to_dict(orient="records"),
                  drop_reasons={f"{s}|{r}": n for (s, r), n in funnel.reasons.items()},
                  url_label_conflicts=n_conflict, duplicates=n_dup,
                  groups_with_both_labels=int(len(conf)),
                  pool_stats_before_matching=pool_stats,
                  final_stats=final_stats,
                  final_counts=dict(benign=n_b, malicious=n_m),
                  composition=comp.to_dict())
    with open(os.path.join(args.report_dir, "preprocessing_report.json"), "w",
              encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    say(f"Wrote {os.path.join(args.report_dir, 'preprocessing_report.json')}")
    say("\nNext: python scripts/split.py")


if __name__ == "__main__":
    main()