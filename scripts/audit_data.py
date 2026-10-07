#!/usr/bin/env python3
"""
audit_data.py  (v2)  -  read-only data audit for the malicious-URL project
--------------------------------------------------------------------------
Run from the project root (venv active). No flags needed, edit SETTINGS below:

    python scripts/audit_data.py

Reads (never modifies):
    data/processed/cleaned_urls_v2.csv
    data/splits_v2/train_urls.csv, val_urls.csv, test_urls.csv

Writes:
    reports/data_audit.json
    reports/domain_conflicts.csv

What it measures (each maps to a known risk):
    1. Schema / duplicates / balance      -> broken or dirty data
    2. Scheme, path, query, case per class -> dataset-construction shortcuts
    3. TLD distribution per class          -> regional bias (eseat.lk case)
    4. Domain overlap between splits       -> leakage that inflates metrics
    5. Domain-level label conflicts        -> label noise (shared hosting)
    6. Near-duplicate URLs across splits   -> leakage
    7. Trivial-feature baselines           -> how much a model can "cheat"

Every change compared with the previous file is marked  "# v2:".
"""

import json
import os
import re
import sys
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import tldextract
from sklearn.tree import DecisionTreeClassifier

try:
    from tqdm import tqdm
except ImportError:                      # tqdm is optional
    def tqdm(x, **_):
        return x

# ═════════════════════════ SETTINGS (edit here) ═════════════════════════
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED    = os.path.join(BASE_DIR, "data", "processed", "cleaned_urls_v2.csv")
SPLITS_DIR = os.path.join(BASE_DIR, "data", "splits_v2")                      # v2: fixed path
SPLIT_PATHS = {                                                               # v2:
    "train": os.path.join(SPLITS_DIR, "train_urls.csv"),
    "val"  : os.path.join(SPLITS_DIR, "val_urls.csv"),
    "test" : os.path.join(SPLITS_DIR, "test_urls.csv"),
}
HOLDOUT_FILE = os.path.join(SPLITS_DIR, "test_holdout_source.csv")            # v2: optional
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

# ───────────────────────── thresholds ─────────────────────────
GAP_WARN            = 0.30   # |rate_benign - rate_malicious| for scheme / path
OVERLAP_WARN        = 0.05   # share of test rows whose domain is in train
SHORTCUT_ACC_WARN   = 0.65   # accuracy of a trivial-feature model (chance ~0.50)
CONFLICT_WARN       = 1      # domains that appear with both labels

SHARED_HOSTING = {
    "github.io", "githubusercontent.com", "gitlab.io", "blogspot.com",
    "herokuapp.com", "netlify.app", "vercel.app", "pages.dev", "workers.dev",
    "web.app", "firebaseapp.com", "azurewebsites.net", "amazonaws.com",
    "cloudfront.net", "wixsite.com", "weebly.com", "wordpress.com",
    "notion.site", "000webhostapp.com", "glitch.me", "repl.co", "onrender.com",
    "sites.google.com", "storage.googleapis.com", "tumblr.com",
}

EXTRACTOR = tldextract.TLDExtract(cache_dir=None, suffix_list_urls=None)
_IP_RE    = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")

FINDINGS: list = []      # (level, message)
REPORT:   dict = {}


def flag(level: str, msg: str) -> None:
    FINDINGS.append((level, msg))


def header(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


# ───────────────────────── loading ─────────────────────────
def load() -> tuple:
    needed  = [CLEANED] + list(SPLIT_PATHS.values())
    missing = [p for p in needed if not os.path.exists(p)]
    if missing:
        sys.exit("Missing files (run preprocess.py and split.py first):\n  "
                 + "\n  ".join(missing))

    # v2: keep_default_na=False so a domain such as "na" or "null" is never turned into NaN
    full = pd.read_csv(CLEANED, dtype={"url": str}, keep_default_na=False)
    parts = []
    for name, path in SPLIT_PATHS.items():
        d = pd.read_csv(path, dtype={"url": str}, keep_default_na=False)
        d["split"] = name
        parts.append(d)
    return full, pd.concat(parts, ignore_index=True)


def _registered(ext) -> str:
    """tldextract renamed registered_domain in newer versions; support both."""
    if hasattr(ext, "top_domain_under_public_suffix"):
        return ext.top_domain_under_public_suffix
    return ext.registered_domain


def parse_one(u: str) -> tuple:
    """Return (scheme, host, registered_domain, suffix, has_path, has_query,
    has_upper, is_ip, url_len). Never raises."""
    try:
        s = str(u)
        scheme = s.split("://", 1)[0].lower() if "://" in s else ""
        p = urlparse(s if "://" in s else "http://" + s)
        host = p.hostname or ""
        ext  = EXTRACTOR(host)
        reg  = _registered(ext) or host
        return (scheme, host, reg, ext.suffix or "(none)",
                bool((p.path or "").strip("/")), bool(p.query),
                any(c.isupper() for c in s), bool(_IP_RE.match(host)), len(s))
    except Exception:
        return ("", "", "", "(none)", False, False, False, False, len(str(u)))


def near_dup_key(u: str) -> str:
    k = str(u).lower()
    k = re.sub(r"^[a-z]+://", "", k)
    k = re.sub(r"^www\.", "", k)
    return k.rstrip("/")


# ───────────────────────── sections ─────────────────────────
def section_schema(full: pd.DataFrame, df: pd.DataFrame) -> None:
    header("1. SCHEMA, DUPLICATES, BALANCE")
    for col in ("url", "label"):
        if col not in df.columns:
            flag("FAIL", f"column '{col}' missing from split files")
            return

    bad_labels = set(df["label"].dropna().unique()) - {0, 1}
    nulls      = int(df["url"].isna().sum() + df["label"].isna().sum())
    dup_in     = int(df.duplicated(subset=["url"]).sum())
    print(f"rows in cleaned file     : {len(full):,}")
    print(f"rows across splits       : {len(df):,}")
    print(f"null url/label values    : {nulls}")
    print(f"duplicate URLs (all)     : {dup_in:,}")

    # v2: the split legitimately drops synthetic rows whose group landed in val/test
    # (and near-duplicates, and held-out sources), so a small difference is expected.
    n_syn = 0
    if "is_synthetic" in full.columns:
        n_syn = int(full["is_synthetic"].astype(str).str.lower().isin(["true", "1"]).sum())
    diff = len(full) - len(df)
    if os.path.exists(HOLDOUT_FILE):
        print(f"(a hold-out file exists: {diff:,} fewer rows in splits than in the cleaned file is expected)")
    elif diff < 0 or diff > n_syn:
        flag("WARN", f"cleaned file has {len(full):,} rows but splits have {len(df):,}: "
                     "splits are stale or incomplete (re-run split.py)")
    elif diff > 0:
        print(f"({diff:,} synthetic row(s) were dropped by split.py because their group "
              "fell into val/test: expected)")
    if bad_labels:
        flag("FAIL", f"unexpected label values: {bad_labels}")
    if nulls:
        flag("FAIL", f"{nulls} null url/label values")
    if dup_in:
        flag("WARN", f"{dup_in:,} URLs appear more than once across splits")

    bal = df.groupby(["split", "label"]).size().unstack(fill_value=0)
    bal.columns = ["benign", "malicious"]
    print("\nClass balance per split:")
    print(bal.to_string())
    REPORT["balance"] = bal.to_dict()


def section_construction(df: pd.DataFrame) -> None:
    header("2. SCHEME / PATH / QUERY / CASE PER CLASS (construction shortcuts)")
    g = df.groupby("label").agg(
        https_rate=("is_https", "mean"),
        path_rate =("has_path", "mean"),
        query_rate=("has_query", "mean"),
        upper_rate=("has_upper", "mean"),
        ip_rate   =("is_ip", "mean"),
        median_len=("url_len", "median"),
    )
    g.index = ["benign", "malicious"]
    print(g.round(3).to_string())
    print("(note: rows without an observed scheme, e.g. Tranco, count as not-https here; "
          "scheme is not a model feature)")                                    # v2:
    REPORT["construction"] = g.round(4).to_dict()

    https_gap = abs(g.loc["benign", "https_rate"] - g.loc["malicious", "https_rate"])
    path_gap  = abs(g.loc["benign", "path_rate"]  - g.loc["malicious", "path_rate"])
    if https_gap > GAP_WARN:
        flag("WARN", f"HTTPS rate differs by {pct(https_gap)} between classes: "
                     "the scheme is a shortcut the model can learn")
    if path_gap > GAP_WARN:
        flag("WARN", f"Path presence differs by {pct(path_gap)} between classes: "
                     "bare-domain benign vs path-bearing malicious shortcut")
    if g["upper_rate"].max() == 0:
        flag("WARN", "no URL contains an uppercase letter: URLs were lowercased "
                     "in preprocessing, so num_upper carries no information")

    if "label_type" in df.columns and df["label_type"].notna().any():
        print("\nBy label_type:")
        t = df.groupby("label_type").agg(
            n=("url", "size"), https=("is_https", "mean"),
            path=("has_path", "mean"), med_len=("url_len", "median"))
        print(t.round(3).to_string())
        REPORT["by_label_type"] = t.round(4).to_dict()

    if "source" in df.columns:                                                 # v2: per source
        print("\nBy source:")
        t = df.groupby("source").agg(
            n=("url", "size"), label=("label", "mean"), https=("is_https", "mean"),
            path=("has_path", "mean"), query=("has_query", "mean"),
            med_len=("url_len", "median"))
        print(t.round(3).to_string())
        REPORT["by_source"] = t.round(4).to_dict()


def section_tld(df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame) -> None:
    header("3. TLD DISTRIBUTION PER CLASS (regional / vocabulary bias)")
    df["cc_tld"] = df["suffix"].apply(
        lambda s: len(s.split(".")[-1]) == 2 and s != "(none)")
    for lab, name in ((0, "benign"), (1, "malicious")):
        sub = df[df["label"] == lab]
        top = sub["suffix"].value_counts(normalize=True).head(8)
        print(f"\n{name}: {sub['suffix'].nunique():,} distinct suffixes, "
              f"country-code share {pct(sub['cc_tld'].mean())}")
        print("  top: " + ", ".join(f"{k} {pct(v)}" for k, v in top.items()))
        REPORT[f"tld_top_{name}"] = {k: round(float(v), 4) for k, v in top.items()}

    com_b = (df[df["label"] == 0]["suffix"] == "com").mean()
    if com_b > 0.60:
        flag("WARN", f"{pct(com_b)} of benign URLs are .com: regional sites "
                     "(.lk, .in, ...) are under-represented")

    # suffix-only baseline: how far does the TLD alone get you? (train -> test, honest)
    maj = train.groupby("suffix")["label"].mean().ge(0.5).astype(int)
    pred = test["suffix"].map(maj)
    cov  = pred.notna().mean()
    acc  = float((pred[pred.notna()] == test.loc[pred.notna(), "label"]).mean())
    print(f"\nSuffix-only lookup (train -> test): covers {pct(cov)} of test rows, "
          f"accuracy on covered rows {pct(acc)}")
    REPORT["suffix_only_baseline"] = {"coverage": round(float(cov), 4),
                                      "accuracy": round(acc, 4)}
    if acc > SHORTCUT_ACC_WARN:                                                # v2:
        flag("WARN", f"the TLD alone predicts the class with {pct(acc)} accuracy on test: "
                     "check with the ablation whether the model leans on TLD features")
    unseen = test[~test["suffix"].isin(train["suffix"])]
    print(f"Test rows with a suffix never seen in train: {len(unseen):,}")


def section_domains(df: pd.DataFrame, train: pd.DataFrame, val: pd.DataFrame,
                    test: pd.DataFrame) -> None:
    header("4. DOMAIN OVERLAP BETWEEN SPLITS (leakage)")
    key = "group_key" if "group_key" in train.columns else "registered_domain"
    print(f"(overlap is judged on '{key}')")
    train_dom = set(train[key])
    for name, part in (("val", val), ("test", test)):
        seen = part[key].isin(train_dom)
        share = float(seen.mean())
        print(f"{name}: {pct(share)} of rows share a '{key}' with train "
              f"({int(seen.sum()):,} rows)")
        REPORT[f"{name}_domain_overlap"] = round(share, 4)
        if share > 0 and key == "group_key":                                   # v2: must be 0
            flag("FAIL", f"{name} shares {pct(share)} of its group_keys with train: "
                         "the split is leaking")
        elif name == "test" and share > OVERLAP_WARN:
            flag("WARN", f"{pct(share)} of test rows come from domains seen in "
                         "training: metrics are inflated. Split by registered_domain")

    if "group_key" in train.columns:
        ro = float(test["registered_domain"].isin(set(train["registered_domain"])).mean())
        print(f"(informational) test rows sharing a registered domain with train: {pct(ro)}"
              " - tenants of one platform are separate groups by design")
        gk = set(train["group_key"])
        for name, part in (("val", val), ("test", test)):
            print(f"{name}: {pct(float(part['group_key'].isin(gk).mean()))} of rows share a "
                  "group_key with train (must be 0.0% for a clean split)")
            REPORT[f"{name}_group_key_overlap"] = round(float(part["group_key"].isin(gk).mean()), 4)

    # domain-memorisation baseline
    maj  = train.groupby(key)["label"].mean().ge(0.5).astype(int)
    pred = test[key].map(maj)
    cov  = pred.notna()
    if cov.any():
        acc = float((pred[cov] == test.loc[cov, "label"]).mean())
        print(f"Domain-lookup baseline (no features at all): covers {pct(cov.mean())} "
              f"of test, accuracy {pct(acc)} on covered rows")
        REPORT["domain_lookup_baseline"] = {"coverage": round(float(cov.mean()), 4),
                                            "accuracy": round(acc, 4)}
        if acc > 0.90 and cov.mean() > OVERLAP_WARN:
            flag("WARN", f"a plain domain lookup scores {pct(acc)} on {pct(cov.mean())} "
                         "of test: part of the score is memorisation, not generalisation")

    header("5. DOMAIN-LEVEL LABEL CONFLICTS (label noise)")
    c = (df.groupby(["registered_domain", "label"]).size()
           .unstack(fill_value=0).rename(columns={0: "n_benign", 1: "n_malicious"}))
    for col in ("n_benign", "n_malicious"):
        if col not in c.columns:
            c[col] = 0
    conf = c[(c["n_benign"] > 0) & (c["n_malicious"] > 0)].copy()
    conf["total"] = conf["n_benign"] + conf["n_malicious"]
    conf = conf.sort_values("total", ascending=False)
    conf_path = os.path.join(REPORT_DIR, "domain_conflicts.csv")
    conf.to_csv(conf_path)
    print(f"Domains appearing with BOTH labels: {len(conf):,} (saved to {conf_path})")
    print(conf.head(12).to_string())
    REPORT["domain_conflicts"] = int(len(conf))
    if len(conf) >= CONFLICT_WARN:
        flag("WARN", f"{len(conf):,} domains carry both labels; many are shared-hosting "
                     "or compromised sites. Decide per-domain vs per-URL labelling")

    shared = c[c.index.isin(SHARED_HOSTING)]
    if len(shared):
        print("\nShared-hosting domains present in the data:")
        print(shared.sort_values("n_malicious", ascending=False).to_string())
        REPORT["shared_hosting_rows"] = shared.to_dict()


def section_near_dups(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    header("6. NEAR-DUPLICATES ACROSS SPLITS (scheme / www / trailing slash ignored)")
    keys = set(train["url"].map(near_dup_key))
    for name, part in (("val", val), ("test", test)):
        share = float(part["url"].map(near_dup_key).isin(keys).mean())
        print(f"{name}: {pct(share)} of rows are near-duplicates of a train URL")
        REPORT[f"{name}_near_dup"] = round(share, 4)
        if share > 0:
            flag("WARN", f"{pct(share)} of {name} rows duplicate a train URL "
                         "after normalisation")


def section_baselines(train: pd.DataFrame, test: pd.DataFrame) -> None:
    header("7. TRIVIAL-FEATURE BASELINES (chance is about 50%)")
    sets = {
        "https only"          : ["is_https"],
        "has_path only"       : ["has_path"],
        "https + path + query" : ["is_https", "has_path", "has_query"],
        "url length only"     : ["url_len"],
        "all five trivial"    : ["is_https", "has_path", "has_query", "url_len", "is_ip"],
    }
    REPORT["trivial_baselines"] = {}
    for name, cols in sets.items():
        clf = DecisionTreeClassifier(max_depth=4, random_state=42)
        clf.fit(train[cols].astype(float), train["label"])
        acc = float((clf.predict(test[cols].astype(float)) == test["label"]).mean())
        print(f"  {name:<22}: {pct(acc)}")
        REPORT["trivial_baselines"][name] = round(acc, 4)
        if acc > SHORTCUT_ACC_WARN and len(cols) <= 3:
            flag("WARN", f"'{name}' alone reaches {pct(acc)}: the dataset has a "
                         "construction shortcut the real model can exploit")


# ───────────────────────── main ─────────────────────────
def main() -> None:
    print("DATA AUDIT (read-only)")
    full, df = load()

    section_schema(full, df)
    if any(level == "FAIL" for level, _ in FINDINGS):
        print("\nStopping: fix the FAIL items above first.")
        _finish()
        return

    print("\nParsing URLs (tldextract)...")
    cols = ["scheme", "host", "registered_domain", "suffix", "has_path",
            "has_query", "has_upper", "is_ip", "url_len"]
    parsed = pd.DataFrame([parse_one(u) for u in tqdm(df["url"], desc="parse")],
                          columns=cols, index=df.index)
    keep_scheme = "scheme" in df.columns          # v2 data stores the OBSERVED scheme
    df = df.drop(columns=[c for c in cols if c in df.columns and not (c == "scheme" and keep_scheme)])
    if keep_scheme:
        parsed = parsed.drop(columns=["scheme"])
    df = pd.concat([df, parsed], axis=1)
    df["is_https"] = (df["scheme"] == "https").astype(int)   # before slicing

    train = df[df["split"] == "train"]
    val   = df[df["split"] == "val"]
    test  = df[df["split"] == "test"]

    section_construction(df)
    section_tld(df, train, test)
    section_domains(df, train, val, test)
    section_near_dups(train, val, test)
    section_baselines(train, test)

    if "label_type" in df.columns:                                             # v2: new label name
        syn = df[df["label_type"].isin(["synthetic_typosquat", "synthetic_malicious"])]
        if len(syn):
            header("8. SYNTHETIC URLS PER SPLIT")
            s = syn.groupby("split").size()
            print(s.to_string())
            n_syn_test = int(test["label_type"].isin(
                ["synthetic_typosquat", "synthetic_malicious"]).sum())
            share = float(n_syn_test / max(1, (test["label"] == 1).sum()))
            print(f"Synthetic share of malicious TEST rows: {pct(share)}")
            REPORT["synthetic_share_of_malicious_test"] = round(share, 4)
            if share > 0:
                flag("WARN", f"{pct(share)} of malicious test rows are synthetic: "
                             "they come from the same rules as the features (circular)")

    _finish()


def _finish() -> None:
    header("VERDICT")
    if not FINDINGS:
        print("OK: no issues flagged.")
    for level, msg in FINDINGS:
        print(f"[{level}] {msg}")
    REPORT["findings"] = [{"level": l, "message": m} for l, m in FINDINGS]
    out = os.path.join(REPORT_DIR, "data_audit.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(REPORT, fh, indent=2, default=lambda o: o.item()
                  if isinstance(o, np.generic) else str(o))
    print(f"\nReport saved: {out}")


if __name__ == "__main__":
    main()