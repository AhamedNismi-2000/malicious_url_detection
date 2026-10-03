#!/usr/bin/env python3
"""
split.py  (v2)  -  leakage-free train / val / test split
---------------------------------------------------------
Run from the project root (venv active):

    python scripts/split.py
    python scripts/split.py --holdout-sources "PhiUSIIL (legitimate),PhiUSIIL (phishing)"

Input   data/processed/cleaned_urls_v2.csv      (from preprocessing.py)
Output  data/splits_v2/train_urls.csv, val_urls.csv, test_urls.csv
        data/splits_v2/test_holdout_source.csv  (only with --holdout-sources)
        reports/split_report.json

Rules enforced (and asserted at the end)
    1. Rows are split by group_key, never by row: one domain (or one shared-platform
       tenant / one big-site hostname) lives in exactly ONE split.
    2. Class balance is kept in every split (StratifiedGroupKFold).
    3. Synthetic rows go to train only, and only if their group is not in val/test.
    4. A URL that is a near-duplicate of a train URL is removed from val/test.
    5. Optional: whole sources can be held out as a separate out-of-distribution
       test set (train on some feeds, test on a feed the model never saw).

7 folds: 1 fold -> test, 1 fold -> val, 5 folds -> train  (about 71 / 14 / 14 %).
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from url_utils import near_dup_key                       # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED     = 42
N_FOLDS  = 7
REQUIRED = ["url", "label", "source", "group_key", "reg_domain", "is_synthetic"]


def say(msg: str = "") -> None:
    print(msg, flush=True)


def summarize(name: str, d: pd.DataFrame) -> dict:
    n = len(d)
    ben, mal = int((d["label"] == 0).sum()), int((d["label"] == 1).sum())
    say(f"   {name:<8} {n:>9,} rows | benign {ben:>8,} | malicious {mal:>8,} "
        f"| groups {d['group_key'].nunique():>8,}")
    return dict(rows=n, benign=ben, malicious=mal, groups=int(d["group_key"].nunique()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(BASE_DIR, "data", "processed",
                                                    "cleaned_urls_v2.csv"))
    ap.add_argument("--out-dir", default=os.path.join(BASE_DIR, "data", "splits_v2"))
    ap.add_argument("--report-dir", default=os.path.join(BASE_DIR, "reports"))
    ap.add_argument("--holdout-sources", default="",
                    help='comma-separated source names, e.g. "PhiUSIIL (legitimate),PhiUSIIL (phishing)"')
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)

    say("=" * 70)
    say("SPLIT v2 (group-aware)")
    say("=" * 70)
    if not os.path.exists(args.input):
        sys.exit(f"Missing {args.input}. Run preprocessing.py first.")
    df = pd.read_csv(args.input, dtype={"url": str, "group_key": str,
                                        "reg_domain": str}, keep_default_na=False)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f"Input is missing columns: {missing}")
    df["is_synthetic"] = df["is_synthetic"].astype(str).str.lower().isin(["true", "1"])
    df["label"] = df["label"].astype(int)
    assert (df["group_key"] != "").all(), "empty group_key in input"
    say(f"Loaded {len(df):,} rows, {df['group_key'].nunique():,} groups")

    # ── optional held-out sources
    holdout = df.iloc[0:0]
    hold_names = [s.strip() for s in args.holdout_sources.split(",") if s.strip()]
    if hold_names:
        unknown = [s for s in hold_names if s not in set(df["source"])]
        if unknown:
            sys.exit(f"Unknown source names: {unknown}\nAvailable: {sorted(df['source'].unique())}")
        mask = df["source"].isin(hold_names)
        holdout, df = df[mask].copy(), df[~mask].copy()
        say(f"Held out sources {hold_names}: {len(holdout):,} rows")

    # ── group split on real (non-synthetic) rows
    synth = df[df["is_synthetic"]]
    pool = df[~df["is_synthetic"]].reset_index(drop=True)
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=args.seed)
    fold = np.full(len(pool), -1)
    for i, (_, idx) in enumerate(sgkf.split(pool, pool["label"], pool["group_key"])):
        fold[idx] = i
    pool["split"] = np.where(fold == 0, "test", np.where(fold == 1, "val", "train"))

    # ── synthetic -> train only, and only if the group is not in val/test
    held_groups = set(pool.loc[pool["split"] != "train", "group_key"])
    synth_ok = synth[~synth["group_key"].isin(held_groups)].copy()
    synth_ok["split"] = "train"
    say(f"Synthetic rows: {len(synth):,} -> train only, kept {len(synth_ok):,}")
    all_rows = pd.concat([pool, synth_ok], ignore_index=True)

    # ── near-duplicate removal (val/test lose rows that resemble train)
    all_rows["_k"] = all_rows["url"].map(near_dup_key)
    train_keys = set(all_rows.loc[all_rows["split"] == "train", "_k"])
    test_keys = set(all_rows.loc[all_rows["split"] == "test", "_k"])
    drop_val = (all_rows["split"] == "val") & all_rows["_k"].isin(train_keys | test_keys)
    drop_test = (all_rows["split"] == "test") & all_rows["_k"].isin(train_keys)
    n_nd = int(drop_val.sum() + drop_test.sum())
    all_rows = all_rows[~(drop_val | drop_test)].drop(columns="_k")
    say(f"Near-duplicates removed from val/test: {n_nd:,}")

    tr, va, te = (all_rows[all_rows["split"] == s] for s in ("train", "val", "test"))

    # ── hold-out set must not share groups with train
    if len(holdout):
        shared = holdout["group_key"].isin(set(tr["group_key"]))
        say(f"Hold-out rows sharing a group with train (removed): {int(shared.sum()):,}")
        holdout = holdout[~shared].copy()
        holdout["split"] = "test_holdout_source"

    # ── assertions
    say("\nLeakage checks")
    g_tr, g_va, g_te = set(tr["group_key"]), set(va["group_key"]), set(te["group_key"])
    assert not (g_tr & g_va), f"group overlap train/val: {len(g_tr & g_va)}"
    assert not (g_tr & g_te), f"group overlap train/test: {len(g_tr & g_te)}"
    assert not (g_va & g_te), f"group overlap val/test: {len(g_va & g_te)}"
    if len(holdout):
        assert not (set(holdout["group_key"]) & g_tr), "hold-out shares groups with train"
    say("   group_key overlap between splits: 0  [OK]")
    reg_overlap = {
        "val": float(va["reg_domain"].isin(set(tr["reg_domain"])).mean()) if len(va) else 0.0,
        "test": float(te["reg_domain"].isin(set(tr["reg_domain"])).mean()) if len(te) else 0.0}
    say(f"   registered-domain overlap with train (informational): "
        f"val {reg_overlap['val']:.1%}, test {reg_overlap['test']:.1%}")
    say("   (non-zero is expected: different tenants of one platform, e.g. a.web.app / b.web.app,"
        " are different groups but share a registered domain)")
    assert not te["is_synthetic"].any() and not va["is_synthetic"].any(), "synthetic in val/test"
    say("   no synthetic rows in val/test  [OK]")

    # ── write
    say("\nSplit sizes")
    rep = {"train": summarize("train", tr), "val": summarize("val", va),
           "test": summarize("test", te)}
    for name, part in (("train", tr), ("val", va), ("test", te)):
        part.sample(frac=1, random_state=args.seed).to_csv(
            os.path.join(args.out_dir, f"{name}_urls.csv"), index=False, encoding="utf-8")
    if len(holdout):
        rep["test_holdout_source"] = summarize("holdout", holdout)
        holdout.to_csv(os.path.join(args.out_dir, "test_holdout_source.csv"),
                       index=False, encoding="utf-8")

    comp = pd.crosstab(all_rows["source"], all_rows["split"])
    say("\nSource composition per split:")
    say(comp.to_string())

    report = dict(seed=args.seed, folds=N_FOLDS, splits=rep,
                  near_duplicates_removed=n_nd, registered_domain_overlap=reg_overlap,
                  holdout_sources=hold_names, source_composition=comp.to_dict())
    path = os.path.join(args.report_dir, "split_report.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)
    say(f"\nWrote splits to {args.out_dir}\nWrote {path}")
    say("Next: python scripts/audit_data.py --cleaned <cleaned_urls_v2.csv> --splits-dir <splits_v2>")


if __name__ == "__main__":
    main()