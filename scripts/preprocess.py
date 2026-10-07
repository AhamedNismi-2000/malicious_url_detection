#!/usr/bin/env python3
"""
preprocess.py  (v2)  -  build ONE clean, documented dataset from all raw sources
------------------------------------------------------------------------------------
Run from the project root (venv active). No flags needed, all options are in CONFIG:

    python scripts/preprocess.py

Optional (rarely needed):
    python scripts/preprocess.py --no-strict      # diagnose: still writes the data
    python scripts/preprocess.py --selftest       # run on tiny mock data

Needs scripts/url_utils.py next to this file.

Raw files looked for in data/raw/ (missing ones are skipped with a message):
    Phish.csv                          PhishTank           malicious (phishing)
    urlhaus.txt                        URLhaus             malicious (malware)
    openphish.txt                      OpenPhish           malicious (phishing)
    data.csv                           faizann24 (old)     DISABLED (no scheme)
    PhiUSIIL_Phishing_URL_Dataset.csv  PhiUSIIL            label 1=legit, 0=phish
    phishing-links-ACTIVE.txt          Phishing.Database   malicious (phishing)
    synthetic_malicious.csv            typosquats          malicious (synthetic),
                                                           optional column "technique"
    top-1m.csv                         Tranco              benign (rank-bucket sample)
    benign_crawled.csv                 your crawler        benign, column "url"
                                                           (real scheme + paths)

Outputs
    data/processed/cleaned_urls_v2.csv       the dataset (written ONLY if the run passed)
    reports/preprocessing_report.json        counts, gates, status (ALWAYS written)
    reports/preprocessing_conflicts.csv      groups that carry both labels
    data/reference/popular_domains.csv       Tranco top-N (for typosquat.py)

Exit codes
    0  passed (or "forced" with --no-strict)     1  a hard gate failed     2  preflight failed

Schema of cleaned_urls_v2.csv
    url         canonical, scheme-less, no "www."  <- the model input
    label       0 benign / 1 malicious
    source      which raw file / feed
    label_type  benign, phishing, malware, malicious, synthetic_typosquat
    scheme      scheme observed in the source ("" = unknown; never invented)
    host, reg_domain, suffix, group_key
    has_path, has_query, is_ip, is_synthetic
    technique   typosquat technique for synthetic rows ("" otherwise)
    url_raw     exactly as collected (debugging only)

Pipeline (each step is counted per source in the report)
    preflight -> load -> canonicalize -> dedupe -> label conflicts -> per-group cap ->
    source caps -> synthetic cap -> class matching -> TLD balance -> GATES -> atomic write

Every change compared with the previous v2 file is marked  "# v2:".
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
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

    # v2: PhiUSIIL phishing was 69% of all malicious rows in the first real run.
    # The cap is applied BEFORE class matching, so the benign side is trimmed to follow.
    SOURCE_CAPS={"PhiUSIIL (phishing)": 35_000},

    # Tranco sampling: (rank_from, rank_to, rows_to_sample). Long tail on purpose.
    TRANCO_BUCKETS=[(1, 1_000, 1_000), (1_001, 10_000, 6_000),
                    (10_001, 100_000, 25_000), (100_001, 1_000_000, 60_000)],
    DROP_TRANCO_SHARED_ROOTS=True,   # drop roots like web.app / github.io from benign
    POPULAR_REF_N=10_000,

    # raw files that are never loaded (they stay on disk). data.csv has no scheme:
    # the old pipeline invented "http://" for it, which created a scheme shortcut.
    DISABLED_SOURCES=["data.csv"],

    # v2: TLD balance check (a suffix that mostly belongs to one class is a shortcut)
    TLD_TOP_N=15,            # suffixes shown in the table
    TLD_GAP_WARN_PP=10.0,    # soft: |malicious% - benign%| of one suffix above this
    TLD_MIN_ROWS=50,         # suffixes with fewer rows count as coin flips in the shortcut score
    TLD_SHORTCUT_MAX=0.65,   # soft: accuracy of "guess by suffix alone" must stay below this

    # GATES. HARD gates stop the run (no dataset written), SOFT gates only warn.
    # Tune the numbers to your data, but do not remove a gate to "make it pass".
    MIN_SOURCE_ROWS=100,          # soft: a loaded source smaller than this is suspicious
    MIN_CANON_KEEP=0.50,          # soft: share of a source surviving canonicalization
    MIN_ROWS_PER_CLASS=20_000,    # hard: final rows per class
    MIN_BENIGN_WITH_PATH=20_000,  # hard: benign rows with a path/query BEFORE matching
    RATIO_RANGE=(0.85, 1.15),     # hard: malicious / benign after matching
    MAX_MATCH_LOSS=0.50,          # hard: share of malicious rows lost to class matching
    MAX_SOURCE_SHARE=0.60,        # soft: one source supplying more than this of a class
    MIN_SCHEME_SHARE=0.50,        # soft: share of rows with an observed scheme per source
    SCHEMELESS_OK_PREFIXES=("Tranco",),   # sources that are bare domains by design
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KNOWN_RAW_FILES = ["Phish.csv", "urlhaus.txt", "openphish.txt", "data.csv",
                   "PhiUSIIL_Phishing_URL_Dataset.csv", "phishing-links-ACTIVE.txt",
                   "synthetic_malicious.csv", "top-1m.csv", "benign_crawled.csv"]


# ═════════════════════════ funnel bookkeeping ═════════════════════════
class Funnel:
    """Counts rows per source after every step, plus drop reasons."""

    def __init__(self) -> None:
        self.steps: OrderedDict = OrderedDict()
        self.reasons: Counter = Counter()

    def snap(self, step: str, df: pd.DataFrame) -> None:
        self.steps[step] = df["source"].value_counts().to_dict() if len(df) else {}

    def snap_raw(self, step: str, counts: dict) -> None:
        self.steps[step] = dict(counts)

    def table(self) -> pd.DataFrame:
        t = pd.DataFrame(self.steps).fillna(0).astype(int)
        first, last = t.columns[0], t.columns[-1]
        t["kept_%"] = (100 * t[last] / t[first].replace(0, np.nan)).round(1)
        return t


def say(msg: str = "") -> None:
    print(msg, flush=True)


# ───────────────────────── gates ─────────────────────────
class Gates:
    """Collects PASS / FAIL / WARN results. hard=True gates stop the run."""

    def __init__(self) -> None:
        self.results: list = []

    def check(self, name: str, ok: bool, detail: str, hard: bool = True, fix: str = "") -> None:
        self.results.append(dict(name=name, ok=bool(ok), hard=hard, detail=detail, fix=fix))

    def failed_hard(self) -> list:
        return [r for r in self.results if not r["ok"] and r["hard"]]

    def warnings(self) -> list:
        return [r for r in self.results if not r["ok"] and not r["hard"]]

    def print(self) -> None:
        for r in self.results:
            tag = "PASS" if r["ok"] else ("FAIL" if r["hard"] else "WARN")
            say(f"   [{tag}] {r['name']}: {r['detail']}")
            if not r["ok"] and r["fix"]:
                say(f"          -> {r['fix']}")


def atomic_write(path: str, writer) -> None:
    """Write to <path>.tmp, then rename: a crash never leaves a half-written file."""
    tmp = path + ".tmp"
    try:
        writer(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _col(t: pd.DataFrame, name: str) -> int:
    return int(t[name].sum()) if (len(t) and name in t.columns) else 0


# ═════════════════════════ loaders ═════════════════════════
def load_sources(raw_dir: str, cfg: dict, include_disabled: bool = False) -> tuple:
    """Return (list of source dicts, tranco_frame_or_None, problems, disabled_files).
    problems = errors that must stop the run (a file exists but could not be used)."""
    out: list = []
    problems: list = []
    disabled_seen: list = []
    disabled = set() if include_disabled else set(cfg.get("DISABLED_SOURCES", []))
    rng_seed = cfg["SEED"]

    def p(name):
        return os.path.join(raw_dir, name)

    def add(name, label, ltype, urls, synthetic=False, technique=None):
        s = pd.Series(urls).reset_index(drop=True)
        keep = s.notna()
        s = s[keep].astype(str).reset_index(drop=True)
        t = None
        if technique is not None:
            t = pd.Series(technique).reset_index(drop=True)[keep]
            t = t.fillna("").astype(str).reset_index(drop=True)
        out.append(dict(name=name, label=label, label_type=ltype,
                        synthetic=synthetic, urls=s, technique=t))
        say(f"  loaded {name:<38} {len(s):>9,} rows")
        if len(s) == 0:
            problems.append(f"{name}: file was read but contains 0 usable URLs")

    def attempt(fname, fn):
        if fname in disabled:
            if os.path.exists(p(fname)):
                say(f"  (DISABLED on purpose) {fname}: not loaded, file left untouched")
            disabled_seen.append(fname)
            return
        if not os.path.exists(p(fname)):
            say(f"  (skip) {fname} not found")
            return
        try:
            fn()
        except Exception as exc:                          # record instead of hiding
            say(f"  [ERROR] {fname}: {exc}")
            problems.append(f"{fname}: could not be read ({exc})")

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

    def data_csv():          # only reached with --include-disabled
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
        tech = df["technique"] if "technique" in df.columns else None
        add("Synthetic typosquats", 1, "synthetic_typosquat", df["url"],
            synthetic=True, technique=tech)

    def crawled():
        # real scheme, path and query are kept by canonicalize() (stored in "scheme")
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
            problems.append(f"top-1m.csv: could not be read ({exc})")
    else:
        say("  (skip) top-1m.csv not found")
    return out, tranco, problems, disabled_seen


# ═════════════════════════ steps ═════════════════════════
def canonicalize_all(sources: list, funnel: Funnel) -> pd.DataFrame:
    rows: list = []
    for src in sources:
        techs = src["technique"].tolist() if src.get("technique") is not None else None
        for i, raw in enumerate(tqdm(src["urls"], desc=f"canon {src['name'][:22]}",
                                     leave=False)):
            c, reason = canonicalize(raw)
            if c is None:
                funnel.reasons[(src["name"], reason)] += 1
                continue
            c.update(source=src["name"], label=src["label"],
                     label_type=src["label_type"], is_synthetic=src["synthetic"],
                     technique=(techs[i] if techs is not None else ""),
                     url_raw=str(raw).replace("\n", " ").replace("\r", " "))
            rows.append(c)
    df = pd.DataFrame(rows)
    if len(df):
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
            say(f"   capped '{name}': {len(idx):,} -> {cap:,} rows")             # v2:
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
    if len(non_ip) == 0:
        empty = pd.DataFrame(columns=["benign", "malicious"])
        return df.iloc[0:0], empty, empty

    before = (non_ip.groupby(cols + ["label"]).size().unstack(fill_value=0)
              .rename(columns={0: "benign", 1: "malicious"}))
    for c in ("benign", "malicious"):
        if c not in before.columns:
            before[c] = 0
    kept = []
    for _, g in non_ip.groupby(cols):
        b, m = g[g["label"] == 0], g[g["label"] == 1]
        n = min(len(b), len(m))
        if n:
            kept += [b.sample(n, random_state=seed), m.sample(n, random_state=seed)]
    matched = pd.concat(kept) if kept else non_ip.iloc[0:0]
    after = (matched.groupby(cols + ["label"]).size().unstack(fill_value=0)
             .rename(columns={0: "benign", 1: "malicious"})) if len(matched) \
        else pd.DataFrame(columns=["benign", "malicious"])

    scarce = before[(before.min(axis=1) == 0) & (before.max(axis=1) > 0)]
    if len(scarce):
        lost = int(scarce.sum().sum())
        say(f"   [WARN] {len(scarce)} stratum(s) contain only ONE class, so {lost:,} rows "
            "are dropped by matching:")
        say(scarce.to_string())
        say("   -> add data for the missing class in that stratum, or remove that column "
            "from MATCH_COLS")

    n_mal = int((matched["label"] == 1).sum()) if len(matched) else 0
    ip_mal = ip[ip["label"] == 1]
    ip_limit = int(cfg["IP_MAX_FRACTION"] / (1 - cfg["IP_MAX_FRACTION"]) * n_mal)
    if len(ip_mal) > ip_limit:
        ip_mal = ip_mal.sample(ip_limit, random_state=seed)
    final = pd.concat([matched, ip_mal, ip[ip["label"] == 0]])
    final = final.sample(frac=1, random_state=seed).reset_index(drop=True)
    return final, before, after


# v2: ───────────────────────── TLD class balance ─────────────────────────
def tld_balance(df: pd.DataFrame, cfg: dict) -> tuple:
    """Per-suffix share of each class (IP hosts excluded: they have no TLD), plus the
    accuracy of guessing the class from the suffix alone (0.5 = no shortcut)."""
    d = df[~df["is_ip"]]
    suf = d["suffix"].fillna("").replace("", "(none)")
    ct = pd.crosstab(suf, d["label"].map({0: "benign", 1: "malicious"}))
    for c in ("benign", "malicious"):
        if c not in ct.columns:
            ct[c] = 0
    ct["total"] = ct["benign"] + ct["malicious"]
    n_b, n_m = max(int(ct["benign"].sum()), 1), max(int(ct["malicious"].sum()), 1)
    ct["benign_%"] = (100 * ct["benign"] / n_b).round(1)
    ct["malicious_%"] = (100 * ct["malicious"] / n_m).round(1)
    ct["gap_pp"] = (ct["malicious_%"] - ct["benign_%"]).round(1)
    # rare suffixes would look perfectly predictive by chance: count them as coin flips
    best = np.where(ct["total"] >= cfg["TLD_MIN_ROWS"],
                    np.maximum(ct["benign"], ct["malicious"]), ct["total"] / 2)
    acc = float(best.sum() / max(ct["total"].sum(), 1))
    top = ct.sort_values("total", ascending=False).head(cfg["TLD_TOP_N"])
    return top[["benign", "malicious", "benign_%", "malicious_%", "gap_pp"]], acc


# ───────────────────────── preflight + gate evaluation ─────────────────────────
def preflight(raw_dir: str, sources: list, problems: list) -> list:
    """Return a list of fatal messages. Empty list = OK to continue."""
    fatal = list(problems)
    if not os.path.isdir(raw_dir):
        fatal.append(f"raw directory not found: {raw_dir}")
    if not any(s["label"] == 1 and len(s["urls"]) for s in sources):
        fatal.append("no MALICIOUS source loaded (expected e.g. Phish.csv, urlhaus.txt, "
                     "PhiUSIIL_Phishing_URL_Dataset.csv)")
    if not any(s["label"] == 0 and len(s["urls"]) for s in sources):
        fatal.append("no BENIGN source loaded (expected e.g. top-1m.csv, "
                     "benign_crawled.csv, PhiUSIIL_Phishing_URL_Dataset.csv)")
    return fatal


def evaluate_gates(cfg: dict, funnel: Funnel, sources: list, df: pd.DataFrame,
                   pool_benign_path: int, before: pd.DataFrame, after: pd.DataFrame,
                   n_conflict_free_dups: bool) -> Gates:
    g = Gates()
    n_b, n_m = int((df["label"] == 0).sum()), int((df["label"] == 1).sum())

    # per-source health (soft)
    t = funnel.table()
    raw_col, canon_col = t.columns[0], "canonicalized" if "canonicalized" in t.columns else None
    for s in sources:
        name, n_raw = s["name"], len(s["urls"])
        if n_raw < cfg["MIN_SOURCE_ROWS"]:
            g.check(f"source_size[{name}]", False, f"only {n_raw} rows loaded", hard=False,
                    fix="check the raw file is complete")
        if canon_col and name in t.index and n_raw:
            keep = t.loc[name, canon_col] / n_raw
            g.check(f"canonicalization_keep[{name}]", keep >= cfg["MIN_CANON_KEEP"],
                    f"{keep:.0%} of rows survive canonicalization", hard=False,
                    fix="see 'Drop reasons' below; the file may use another format")

    # hard gates
    g.check("rows_per_class", min(n_b, n_m) >= cfg["MIN_ROWS_PER_CLASS"],
            f"benign {n_b:,} / malicious {n_m:,} (minimum {cfg['MIN_ROWS_PER_CLASS']:,} each)",
            fix="load more sources, relax the caps, or lower MIN_ROWS_PER_CLASS knowingly")
    lo, hi = cfg["RATIO_RANGE"]
    ratio = n_m / max(n_b, 1)
    g.check("class_ratio", lo <= ratio <= hi, f"malicious/benign = {ratio:.3f} (allowed {lo}-{hi})",
            fix="look at the strata tables: one class is missing in some stratum")
    g.check("benign_with_path_or_query", pool_benign_path >= cfg["MIN_BENIGN_WITH_PATH"],
            f"{pool_benign_path:,} benign rows with a path/query before matching "
            f"(minimum {cfg['MIN_BENIGN_WITH_PATH']:,})",
            fix="run the benign crawler (data/raw/benign_crawled.csv) or add a benign "
                "dataset that has real paths and schemes")
    mal_before, mal_after = _col(before, "malicious"), _col(after, "malicious")
    loss = 1 - mal_after / mal_before if mal_before else 1.0
    g.check("malicious_lost_to_matching", loss <= cfg["MAX_MATCH_LOSS"],
            f"{loss:.0%} of non-IP malicious rows dropped by class matching "
            f"(allowed {cfg['MAX_MATCH_LOSS']:.0%})",
            fix="the benign side lacks rows in the strata shown in the [WARN] table above")
    g.check("no_duplicate_urls", not df["url"].str.lower().duplicated().any(),
            "no duplicate URLs in the final set")
    g.check("no_missing_fields", bool(df["url"].notna().all() and df["group_key"].notna().all()
                                      and df["label"].isin([0, 1]).all()),
            "url, label and group_key are present on every row")

    # soft gates
    if len(df):
        for lab, lname in ((0, "benign"), (1, "malicious")):
            sub = df[df["label"] == lab]["source"].value_counts(normalize=True)
            if len(sub):
                g.check(f"source_share[{lname}]", sub.iloc[0] <= cfg["MAX_SOURCE_SHARE"],
                        f"largest {lname} source '{sub.index[0]}' = {sub.iloc[0]:.0%} "
                        f"(limit {cfg['MAX_SOURCE_SHARE']:.0%})", hard=False,
                        fix="cap that source (SOURCE_CAPS) or add more sources")
        for src, sub in df.groupby("source"):
            if src.startswith(tuple(cfg["SCHEMELESS_OK_PREFIXES"])):
                continue
            share = float((sub["scheme"] != "").mean())
            g.check(f"scheme_observed[{src}]", share >= cfg["MIN_SCHEME_SHARE"],
                    f"{share:.0%} of rows carry an observed scheme", hard=False,
                    fix="schemes missing in the source: scheme stays unknown, never invented")
        syn = float(df["is_synthetic"].sum()) / max(n_m, 1)
        g.check("synthetic_share", syn <= cfg["SYNTHETIC_FRACTION"] + 0.005,
                f"synthetic = {syn:.2%} of malicious (limit {cfg['SYNTHETIC_FRACTION']:.0%})",
                hard=False)
        ipf = float(df[df["label"] == 1]["is_ip"].mean()) if n_m else 0.0
        g.check("ip_share", ipf <= cfg["IP_MAX_FRACTION"] + 0.01,
                f"IP-host rows = {ipf:.2%} of malicious (limit {cfg['IP_MAX_FRACTION']:.0%})",
                hard=False)
    return g


# ═════════════════════════ main pipeline ═════════════════════════
def run(cfg: dict, raw_dir: str, out_dir: str, report_dir: str, ref_dir: str,
        strict: bool = True, include_disabled: bool = False) -> int:
    for d in (out_dir, report_dir, ref_dir):
        os.makedirs(d, exist_ok=True)
    out_path = os.path.join(out_dir, "cleaned_urls_v2.csv")
    report_path = os.path.join(report_dir, "preprocessing_report.json")

    say("=" * 70)
    say("PREPROCESSING v2" + ("   (strict)" if strict else "   (--no-strict: diagnose mode)"))
    say("=" * 70)
    say("\n1. Loading raw sources")
    sources, tranco, problems, disabled_seen = load_sources(raw_dir, cfg, include_disabled)

    # preflight. A source that exists but cannot be read must never be skipped silently.
    fatal = preflight(raw_dir, sources, problems)
    if fatal:
        say("\n" + "!" * 70)
        say("PREFLIGHT FAILED: nothing was written")
        for m in fatal:
            say(f"   - {m}")
        say("!" * 70)
        _write_report(report_path, dict(status="preflight_failed", problems=fatal,
                                        disabled_sources=disabled_seen))
        return 2

    funnel = Funnel()
    funnel.snap_raw("raw", {s["name"]: len(s["urls"]) for s in sources})

    say("\n2. Canonicalizing (scheme removed, www removed, &amp; decoded, validated)")
    df = canonicalize_all(sources, funnel)
    if len(df) == 0:
        say("PREFLIGHT FAILED: no URL survived canonicalization")
        _write_report(report_path, dict(status="preflight_failed",
                                        problems=["no URL survived canonicalization"]))
        return 2
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
    atomic_write(os.path.join(report_dir, "preprocessing_conflicts.csv"),
                 lambda t: conf.to_csv(t))
    say(f"   groups carrying both labels (kept, reported): {len(conf):,}")

    say(f"\n4. Capping rows: per group <= {cfg['MAX_PER_GROUP']}, per IP <= {cfg['MAX_PER_IP']}, "
        f"per platform <= {cfg['MAX_PER_PLATFORM']}")
    df = cap_per_group(df, cfg)
    funnel.snap("group_capped", df)

    if cfg["SOURCE_CAPS"]:
        say(f"   source caps: {cfg['SOURCE_CAPS']}")                               # v2:
        df = cap_sources(df, cfg)
        funnel.snap("source_capped", df)

    say(f"\n5. Limiting synthetic rows to {cfg['SYNTHETIC_FRACTION']:.0%} of real malicious")
    df = cap_synthetic(df, cfg)
    funnel.snap("synthetic_capped", df)

    say(f"\n6. Matching classes on {cfg['MATCH_COLS']} (IP hosts exempt)")
    pool_stats = {c: df.groupby("label")[c].mean().round(3).to_dict()
                  for c in ("has_path", "has_query", "is_ip")}
    pool_benign_path = int(((df["label"] == 0) & (~df["is_ip"])
                            & (df["has_path"] | df["has_query"])).sum())
    df, before, after = match_classes(df, cfg)
    funnel.snap("class_matched", df)
    say("   strata before matching:\n" + before.to_string())
    say("   strata after matching:\n" + after.to_string())

    # ───────── report tables ─────────
    tbl = funnel.table()
    say("\n" + "=" * 70)
    say("FUNNEL: rows per source after each step")
    say("=" * 70)
    say(tbl.to_string())
    if funnel.reasons:
        say("\nDrop reasons during canonicalization (source, reason): count")
        for (s, r), n in sorted(funnel.reasons.items(), key=lambda x: -x[1])[:15]:
            say(f"   {s:<34} {r:<18} {n:>8,}")

    comp = pd.crosstab(df["source"], df["label"].map({0: "benign", 1: "malicious"})) \
        if len(df) else pd.DataFrame()
    say("\nFinal composition by source:")
    say(comp.to_string())

    scheme_tbl = pd.DataFrame()
    if len(df):
        scheme_tbl = df.groupby("source").agg(
            rows=("url", "size"),
            scheme_known_pct=("scheme", lambda s: round(100 * (s != "").mean(), 1)),
            https_pct_of_known=("scheme", lambda s: round(
                100 * (s == "https").sum() / max((s != "").sum(), 1), 1)),
            has_path_pct=("has_path", lambda s: round(100 * s.mean(), 1)),
            non_com_pct=("suffix", lambda s: round(100 * (s != "com").mean(), 1)))
        say("\nScheme / path / TLD profile per source (final rows):")
        say(scheme_tbl.to_string())
    final_stats = {c: df.groupby("label")[c].mean().round(3).to_dict()
                   for c in ("has_path", "has_query", "is_ip")} if len(df) else {}
    say(f"\nhas_path / has_query / is_ip per class (label 0, 1):\n   {final_stats}")

    # ───────── gates (fail loudly) ─────────
    say("\n7. Gates")
    gates = evaluate_gates(cfg, funnel, sources, df, pool_benign_path, before, after, True)

    # v2: TLD balance table + soft gates
    tld_top, tld_acc = pd.DataFrame(), float("nan")
    if len(df):
        tld_top, tld_acc = tld_balance(df, cfg)
        say("\nTLD balance (non-IP rows; share of each class per suffix):")
        say(tld_top.to_string())
        say(f"   accuracy of guessing by suffix alone: {tld_acc:.1%} (50% = no shortcut)\n")
        lop = tld_top[tld_top["gap_pp"].abs() >= cfg["TLD_GAP_WARN_PP"]]
        gates.check("tld_balance", len(lop) == 0,
                    ("no top suffix differs by >= %.0f points between classes"
                     % cfg["TLD_GAP_WARN_PP"]) if len(lop) == 0 else
                    "lopsided suffixes: " + ", ".join(
                        f"{i} ({r.gap_pp:+.1f} pp)" for i, r in lop.iterrows()),
                    hard=False,
                    fix="add data for the under-represented class on those suffixes "
                        "(e.g. crawl more benign sites there) or state it as a limitation")
        gates.check("tld_shortcut", tld_acc <= cfg["TLD_SHORTCUT_MAX"],
                    f"suffix alone predicts the class with {tld_acc:.1%} accuracy "
                    f"(limit {cfg['TLD_SHORTCUT_MAX']:.0%})", hard=False,
                    fix="the model could learn the TLD instead of the URL structure")

    gates.print()
    hard = gates.failed_hard()
    status = "passed" if not hard else ("forced" if not strict else "failed")

    inputs = {f: os.path.getsize(os.path.join(raw_dir, f)) for f in KNOWN_RAW_FILES
              if os.path.exists(os.path.join(raw_dir, f))}
    n_b, n_m = int((df["label"] == 0).sum()), int((df["label"] == 1).sum())
    report = dict(status=status, config={k: v for k, v in cfg.items()},
                  disabled_sources=disabled_seen, input_file_bytes=inputs,
                  python=sys.version.split()[0], pandas=pd.__version__,
                  gates=gates.results,
                  funnel=tbl.reset_index().to_dict(orient="records"),
                  drop_reasons={f"{s}|{r}": n for (s, r), n in funnel.reasons.items()},
                  url_label_conflicts=n_conflict, duplicates=n_dup,
                  groups_with_both_labels=int(len(conf)),
                  pool_stats_before_matching=pool_stats,
                  benign_with_path_before_matching=pool_benign_path,
                  scheme_profile=scheme_tbl.reset_index().to_dict(orient="records"),
                  final_stats=final_stats, final_counts=dict(benign=n_b, malicious=n_m),
                  composition=comp.to_dict(),
                  tld_balance=tld_top.reset_index().to_dict(orient="records"),     # v2:
                  tld_only_accuracy=(None if np.isnan(tld_acc)
                                     else round(float(tld_acc), 4)))               # v2:
    _write_report(report_path, report)

    if tranco is not None:
        ref = tranco[tranco["rank"] <= cfg["POPULAR_REF_N"]].sort_values("rank")
        atomic_write(os.path.join(ref_dir, "popular_domains.csv"),
                     lambda t: ref.assign(rank=ref["rank"].astype(int)).to_csv(
                         t, index=False, header=False))

    cols = ["url", "label", "source", "label_type", "scheme", "host", "reg_domain",
            "suffix", "group_key", "has_path", "has_query", "is_ip", "is_synthetic",
            "technique", "url_raw"]

    say("\n" + "=" * 70)
    if status == "failed":
        say("PREPROCESSING FAILED: dataset NOT written")
        for r in hard:
            say(f"   - {r['name']}: {r['detail']}")
        if os.path.exists(out_path):
            say(f"   WARNING: {out_path} is an OLD file from an earlier run. Do not use it.")
        say(f"   Details: {report_path}")
        say("   To look at the data anyway:  python scripts/preprocess.py --no-strict")
        say("=" * 70)
        return 1

    atomic_write(out_path, lambda t: df[cols].to_csv(t, index=False, encoding="utf-8"))
    if status == "forced":
        say("PREPROCESSING FORCED (--no-strict): dataset written although these gates failed:")
        for r in hard:
            say(f"   - {r['name']}: {r['detail']}")
        say("   Do NOT train a final model on this file. Report status = 'forced'.")
    else:
        say("PREPROCESSING PASSED")
        if gates.warnings():
            say(f"   ({len(gates.warnings())} warning(s), see the gates above)")
    say(f"Wrote {out_path}")
    say(f"Wrote {report_path}")
    say("=" * 70)
    say("\nNext: python scripts/split.py")
    return 0


def _write_report(path: str, report: dict) -> None:
    atomic_write(path, lambda t: json.dump(report, open(t, "w", encoding="utf-8"),
                                           indent=2, default=str))


# ───────────────────────── self-test on tiny mock data ─────────────────────────
def _selftest() -> int:
    import contextlib
    import io
    import random

    rng = random.Random(1)
    tmp = tempfile.mkdtemp(prefix="prep_selftest_")
    results: list = []

    def build(raw: str, crawled: bool = True, corrupt_phish: bool = False) -> None:
        os.makedirs(raw, exist_ok=True)
        # PhishTank
        rows = ["phish_id,url,verified"]
        for i in range(800):
            sch = "https" if rng.random() < 0.9 else "http"
            path = f"/login/{i}.php" if rng.random() < 0.65 else ""
            q = "?id=%d" % i if path and rng.random() < 0.2 else ""
            rows.append(f"{i},{sch}://phishsite{i}.{rng.choice(['com','net','ru'])}{path}{q},yes")
        rows.append("9999,ftp://bad.example.com/x,yes")
        with open(os.path.join(raw, "Phish.csv"), "w") as fh:
            fh.write("\n".join(rows))
        if corrupt_phish:
            open(os.path.join(raw, "Phish.csv"), "w").close()          # empty file
        # URLhaus (comment header, url in column 3)
        lines = ["# URLhaus mock"]
        for i in range(200):
            lines.append(f'{i},2024-01-01,"http://45.{i % 250}.{i // 250}.9:8080/bin.sh",online,x')
        with open(os.path.join(raw, "urlhaus.txt"), "w") as fh:
            fh.write("\n".join(lines))
        with open(os.path.join(raw, "openphish.txt"), "w") as fh:
            fh.write("\n".join(f"https://openph{i}.com/p/{i}" for i in range(60)))
        # old data.csv (must be ignored)
        with open(os.path.join(raw, "data.csv"), "w") as fh:
            fh.write("url,label\n" + "\n".join(
                f"old{i}.com/page{i}.html,{'good' if i % 2 else 'bad'}" for i in range(300)))
        # PhiUSIIL
        pl = ["URL,Label"]
        pl += [f"https://www.legit{i}.org,1" for i in range(700)]
        pl += [f"http://www.phi{i}.co{'/a/' + str(i) + '.html' if i % 4 == 0 else ''},0"
               for i in range(400)]
        with open(os.path.join(raw, "PhiUSIIL_Phishing_URL_Dataset.csv"), "w") as fh:
            fh.write("\n".join(pl))
        with open(os.path.join(raw, "synthetic_malicious.csv"), "w") as fh:
            fh.write("url,technique\n" + "\n".join(
                f"http://paypa1-{i}.com/login,leet" for i in range(40)))
        with open(os.path.join(raw, "top-1m.csv"), "w") as fh:
            fh.write("\n".join(f"{i + 1},tr{i}site.{rng.choice(['com', 'lk', 'in', 'org'])}"
                               for i in range(4000)))
        if crawled:
            cl = ["url"]
            for i in range(1500):
                sch = "https" if rng.random() < 0.8 else "http"
                path = f"/Articles/{i}" if rng.random() < 0.6 else ""
                q = "?page=%d" % i if path and rng.random() < 0.2 else ""
                cl.append(f"{sch}://crawl{i}.com{path}{q}")
            with open(os.path.join(raw, "benign_crawled.csv"), "w") as fh:
                fh.write("\n".join(cl))

    def scenario(name, want_code, want_output, want_in_text=(), **kw):
        raw = os.path.join(tmp, name, "raw")
        build(raw, crawled=kw.get("crawled", True), corrupt_phish=kw.get("corrupt", False))
        cfg = dict(CONFIG)
        cfg.update(MIN_ROWS_PER_CLASS=100, MIN_BENIGN_WITH_PATH=100, MIN_SOURCE_ROWS=10)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = run(cfg, raw, os.path.join(tmp, name, "proc"),
                       os.path.join(tmp, name, "rep"), os.path.join(tmp, name, "ref"),
                       strict=kw.get("strict", True))
        text = buf.getvalue()
        out = os.path.join(tmp, name, "proc", "cleaned_urls_v2.csv")
        ok = (code == want_code and os.path.exists(out) == want_output
              and all(s in text for s in want_in_text))
        results.append(ok)
        say(f"[{'PASS' if ok else 'FAIL'}] {name}: exit={code} (want {want_code}), "
            f"dataset written={os.path.exists(out)} (want {want_output})")
        if not ok:
            say(text[-2500:])
        return out, text

    out, text = scenario("A_good_data", 0, True,
                         want_in_text=("DISABLED on purpose", "PREPROCESSING PASSED",
                                       "TLD balance"))                             # v2:
    if os.path.exists(out):
        d = pd.read_csv(out, keep_default_na=False)
        checks = {
            "data.csv rows absent": not d["source"].str.contains("data.csv").any(),
            "crawled keeps real schemes": set(d[d.source == "Crawled benign"]["scheme"]) >= {"http", "https"},
            "no invented scheme on Tranco": (d[d.source.str.startswith("Tranco")]["scheme"] == "").all(),
            "technique column carried": (d[d.is_synthetic == True]["technique"] == "leet").all()  # noqa: E712
                                        if d.is_synthetic.any() else True,
            "ftp row dropped": not d["url_raw"].str.startswith("ftp").any(),
            "class sizes equal-ish": 0.85 <= (d.label == 1).sum() / max((d.label == 0).sum(), 1) <= 1.15,
        }
        for k, v in checks.items():
            results.append(bool(v))
            say(f"[{'PASS' if v else 'FAIL'}] A: {k}")
    scenario("B_no_benign_paths", 1, False,
             want_in_text=("PREPROCESSING FAILED", "benign_with_path_or_query"), crawled=False)
    scenario("C_corrupt_source", 2, False, want_in_text=("PREFLIGHT FAILED",), corrupt=True)
    scenario("D_forced", 0, True, want_in_text=("FORCED",), crawled=False, strict=False)

    # v2: unit test of the TLD table on a deliberately lopsided frame
    lop = pd.DataFrame({"suffix": ["com"] * 100 + ["xyz"] * 100,
                        "label": [0] * 100 + [1] * 100, "is_ip": [False] * 200})
    top, acc = tld_balance(lop, CONFIG)
    ok = acc > 0.95 and abs(float(top.loc["xyz", "gap_pp"])) >= CONFIG["TLD_GAP_WARN_PP"]
    results.append(ok)
    say(f"[{'PASS' if ok else 'FAIL'}] tld_balance flags a lopsided suffix (acc={acc:.2f})")

    shutil.rmtree(tmp, ignore_errors=True)
    n_ok = sum(results)
    say(f"\nself-test: {n_ok}/{len(results)} passed")
    return 0 if all(results) else 1


# ═════════════════════════ entry point ═════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default=os.path.join(BASE_DIR, "data", "raw"))
    ap.add_argument("--out-dir", default=os.path.join(BASE_DIR, "data", "processed"))
    ap.add_argument("--report-dir", default=os.path.join(BASE_DIR, "reports"))
    ap.add_argument("--ref-dir", default=os.path.join(BASE_DIR, "data", "reference"))
    ap.add_argument("--max-per-group", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--no-strict", action="store_true",
                    help="write the dataset even if hard gates fail (diagnosis only)")
    ap.add_argument("--include-disabled", action="store_true",
                    help="also load DISABLED_SOURCES (e.g. data.csv)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(_selftest())

    cfg = dict(CONFIG)
    if args.max_per_group:
        cfg["MAX_PER_GROUP"] = args.max_per_group
    if args.seed:
        cfg["SEED"] = args.seed
    sys.exit(run(cfg, args.raw_dir, args.out_dir, args.report_dir, args.ref_dir,
                 strict=not args.no_strict, include_disabled=args.include_disabled))


if __name__ == "__main__":
    main()