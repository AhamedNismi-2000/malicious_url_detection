#!/usr/bin/env python3
"""
evaluate_external.py
--------------------
Evaluate the trained model on an EXTERNAL URL set (different source / later date
than the training data) to check real-world generalization.

Put this file in your project's scripts/ folder (next to feature_extraction.py).

Input options (use either or both):
  --csv FILE            CSV with columns: url,label   (label: 0/1 or benign/malicious)
  --malicious FILE      text file, one malicious URL per line
  --benign FILE         text file, one benign URL per line

Examples:
  python scripts/evaluate_external.py --malicious fresh_phish.txt --benign fresh_benign.txt
  python scripts/evaluate_external.py --csv external.csv

Outputs (results/external/):
  external_metrics.json, false_positives.csv, false_negatives.csv
"""

import os
import sys
import json
import argparse
import warnings

import joblib
import numpy as np
import pandas as pd
import tldextract
from urllib.parse import urlparse
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

warnings.filterwarnings("ignore")

# Reuse the project's own code so preprocessing is identical to training
import feature_extraction as fe
from preprocess import normalize_url, valid_domain

BASE_DIR   = fe.BASE_DIR
MODELS_DIR = fe.MODELS_DIR
OUT_DIR    = os.path.join(BASE_DIR, "results", "external")
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH     = os.path.join(MODELS_DIR, "rf_model_latest.joblib")
THRESHOLD_PATH = os.path.join(MODELS_DIR, "threshold.json")
TRAIN_CSV      = fe.TRAIN_PATH

LABEL_MAP = {
    "0": 0, "benign": 0, "good": 0, "legitimate": 0, "safe": 0,
    "1": 1, "malicious": 1, "bad": 1, "phishing": 1, "malware": 1,
}


# ───────────────────────── LOADING ─────────────────────────
def read_lines(path: str) -> list:
    with open(path, encoding="utf-8", errors="ignore") as f:
        return [l.strip() for l in f if l.strip() and not l.startswith("#")]


def load_external(args) -> pd.DataFrame:
    frames = []
    if args.csv:
        df = pd.read_csv(args.csv, dtype=str)
        df.columns = [c.lower().strip() for c in df.columns]
        df = df[["url", "label"]].copy()
        df["label"] = df["label"].str.lower().str.strip().map(LABEL_MAP)
        frames.append(df)
    if args.malicious:
        frames.append(pd.DataFrame({"url": read_lines(args.malicious), "label": 1}))
    if args.benign:
        frames.append(pd.DataFrame({"url": read_lines(args.benign), "label": 0}))
    if not frames:
        sys.exit("Provide --csv and/or --malicious/--benign. See --help.")

    df = pd.concat(frames, ignore_index=True).dropna(subset=["url", "label"])
    df["label"] = df["label"].astype(int)
    print(f"Loaded {len(df):,} external URLs")

    # Same cleaning as training
    df["url"] = df["url"].apply(normalize_url)
    df = df[df["url"].notnull()]
    df = df[df["url"].apply(valid_domain)]
    df = df.drop_duplicates(subset=["url"]).reset_index(drop=True)

    # Drop URLs that carry both labels
    conflict = df.groupby("url")["label"].nunique()
    df = df[~df["url"].isin(conflict[conflict > 1].index)]
    print(f"After cleaning: {len(df):,}")
    return df.reset_index(drop=True)


def remove_train_overlap(df: pd.DataFrame) -> pd.DataFrame:
    """Drop external URLs that also appear in the training set (would inflate scores)."""
    if not os.path.exists(TRAIN_CSV):
        print("Train split not found; skipping overlap check.")
        return df
    train = pd.read_csv(TRAIN_CSV, dtype={"url": str})
    overlap = df["url"].isin(set(train["url"]))
    print(f"Removed {overlap.sum():,} URLs that overlap with the training set")
    return df[~overlap].reset_index(drop=True)


# ───────────────────────── FEATURES ─────────────────────────
def build_features(urls: list, char_vec, word_vec, scaler) -> np.ndarray:
    heur   = fe.extract_heuristic_batch(urls)
    heur_s = scaler.transform(heur).astype(np.float32)
    nlp    = fe.transform_nlp(urls, char_vec, word_vec)
    X      = fe.combine_features(heur_s, nlp)
    return X.toarray().astype(np.float32, copy=False)


# ───────────────────────── METRICS ─────────────────────────
def compute_metrics(y, proba, threshold) -> dict:
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {
        "n": int(len(y)),
        "threshold": round(float(threshold), 3),
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred, zero_division=0), 4),
        "f1": round(f1_score(y, pred, zero_division=0), 4),
        "fpr": round(fp / (fp + tn), 4) if (fp + tn) else None,
        "fnr": round(fn / (fn + tp), 4) if (fn + tp) else None,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }
    out["auc"] = round(roc_auc_score(y, proba), 4) if len(set(y)) == 2 else None
    return out


def print_metrics(title: str, m: dict):
    print(f"\n{title}  (n={m['n']:,}, threshold={m['threshold']})")
    print(f"  Acc {m['accuracy']}  Prec {m['precision']}  Rec {m['recall']}  "
          f"F1 {m['f1']}  AUC {m['auc']}")
    print(f"  FPR {m['fpr']}  FNR {m['fnr']}   "
          f"(TN={m['tn']:,} FP={m['fp']:,} FN={m['fn']:,} TP={m['tp']:,})")


def slice_report(df: pd.DataFrame, threshold: float):
    """Break results down by URL shape to expose shortcuts."""
    def has_path(u):
        p = urlparse(u).path
        return bool(p and p != "/")

    df = df.copy()
    df["scheme"]   = df["url"].str.extract(r"^(https?)://")[0]
    df["has_path"] = df["url"].apply(has_path)
    df["tld"]      = df["url"].apply(lambda u: tldextract.extract(u).suffix.split(".")[-1])
    df["pred"]     = (df["proba"] >= threshold).astype(int)

    def summarize(name, col, min_n=30):
        print(f"\nBreakdown by {name}:")
        print(f"  {'group':<14}{'n':>8}{'%malicious':>12}{'pred%mal':>10}{'acc':>8}{'FPR':>8}{'FNR':>8}")
        for g, sub in df.groupby(col):
            if len(sub) < min_n:
                continue
            neg, pos = sub[sub.label == 0], sub[sub.label == 1]
            fpr = (neg.pred == 1).mean() if len(neg) else float("nan")
            fnr = (pos.pred == 0).mean() if len(pos) else float("nan")
            print(f"  {str(g):<14}{len(sub):>8,}{sub.label.mean():>12.2f}"
                  f"{sub.pred.mean():>10.2f}{(sub.pred == sub.label).mean():>8.3f}"
                  f"{fpr:>8.3f}{fnr:>8.3f}")

    summarize("scheme", "scheme")
    summarize("has path", "has_path")
    top_tlds = df["tld"].value_counts().head(8).index
    summarize("TLD (top 8)", df[df.tld.isin(top_tlds)].assign(_t=lambda d: d.tld)["_t"]
              .reindex(df.index).fillna("other"))


# ───────────────────────── MAIN ─────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv")
    ap.add_argument("--malicious")
    ap.add_argument("--benign")
    ap.add_argument("--keep-overlap", action="store_true",
                    help="do not remove URLs that appear in the training set")
    args = ap.parse_args()

    # Load artifacts
    for p in [MODEL_PATH, THRESHOLD_PATH, fe.CHAR_VEC_PATH, fe.WORD_VEC_PATH, fe.SCALER_PATH]:
        if not os.path.exists(p):
            sys.exit(f"Missing artifact: {p}")
    model    = joblib.load(MODEL_PATH)
    char_vec = joblib.load(fe.CHAR_VEC_PATH)
    word_vec = joblib.load(fe.WORD_VEC_PATH)
    scaler   = joblib.load(fe.SCALER_PATH)
    with open(THRESHOLD_PATH) as f:
        threshold = float(json.load(f)["threshold"])

    fe.load_tranco_whitelist()   # same whitelist state as training

    df = load_external(args)
    if not args.keep_overlap:
        df = remove_train_overlap(df)
    if df.empty or df.label.nunique() < 1:
        sys.exit("No URLs left to evaluate.")

    print(f"\nEvaluating {len(df):,} URLs "
          f"(benign={int((df.label == 0).sum()):,}, malicious={int((df.label == 1).sum()):,})")

    X = build_features(df["url"].tolist(), char_vec, word_vec, scaler)
    if X.shape[1] != model.n_features_in_:
        sys.exit(f"Feature mismatch: got {X.shape[1]}, model expects {model.n_features_in_}. "
                 "Re-run feature_extraction.py + train_model.py together.")

    mal_col = list(model.classes_).index(1)
    df["proba"] = model.predict_proba(X)[:, mal_col]
    y = df["label"].values

    # Results at saved threshold and at default 0.5
    m_saved = compute_metrics(y, df["proba"].values, threshold)
    m_half  = compute_metrics(y, df["proba"].values, 0.5)
    print_metrics("EXTERNAL @ saved threshold", m_saved)
    print_metrics("EXTERNAL @ 0.50", m_half)

    slice_report(df, threshold)

    # Save error files
    df["pred"] = (df["proba"] >= threshold).astype(int)
    fp_df = df[(df.label == 0) & (df.pred == 1)].sort_values("proba", ascending=False)
    fn_df = df[(df.label == 1) & (df.pred == 0)].sort_values("proba")
    fp_df.to_csv(os.path.join(OUT_DIR, "false_positives.csv"), index=False)
    fn_df.to_csv(os.path.join(OUT_DIR, "false_negatives.csv"), index=False)
    with open(os.path.join(OUT_DIR, "external_metrics.json"), "w") as f:
        json.dump({"saved_threshold": m_saved, "threshold_0.5": m_half}, f, indent=2)

    print(f"\nSaved to {OUT_DIR}")
    print("\nWorst false positives (benign flagged malicious):")
    for _, r in fp_df.head(10).iterrows():
        print(f"  {r.proba:.2f}  {r.url[:100]}")
    print("\nWorst false negatives (malicious missed):")
    for _, r in fn_df.head(10).iterrows():
        print(f"  {r.proba:.2f}  {r.url[:100]}")

    # Interpretation hint
    print("\nHOW TO READ THIS:")
    print("  - Compare with your TEST scores. A big drop (e.g. F1 0.99 -> 0.85) means the")
    print("    test set was too easy / had dataset shortcuts.")
    print("  - Check the 'has path' breakdown: if benign URLs WITH paths have a high FPR,")
    print("    the model learned 'path = malicious' from the data sources.")
    print("  - Read false_positives.csv: repeated patterns show what to add to training.")


if __name__ == "__main__":
    main()