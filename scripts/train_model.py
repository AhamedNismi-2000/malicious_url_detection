#!/usr/bin/env python3
"""
train_model.py  (v2)  -  Random Forest training with an honest evaluation report
------------------------------------------------------------------------------------
Run from the project root (venv active). No flags needed, all options are in SETTINGS:

    python scripts/train_model.py

Reads   features/features_{train,val,test}.npz         (feature_extraction.py)
        features/features_heldout.npz                  (optional challenge set)
        models/feature_config.json                     (feature_extraction.py)
        data/splits_v2/{train,val,test}_urls.csv       (row metadata: source, suffix)
        reports/split_report.json                      (must be a verified split)
Writes  models/rf_model_<time>.joblib, rf_model_latest.joblib, threshold.json
        results/metrics/rf_metrics_<time>.json, training.log

What is new compared with the previous train_model.py   (marked "# v2:")
  1. Refuses to train on an unverified split (split_report.json) or when the feature
     matrix does not match models/feature_config.json.
  2. Threshold: the best F1 with FPR < MAX_ACCEPTABLE_FPR on VAL. If no threshold meets
     the limit it says so and takes the lowest-FPR one (the old code silently used 0.5).
  3. Calibration report (Brier score, ECE, reliability table) on VAL and TEST.
  4. TEST results per SOURCE and per TLD (is the model right everywhere, or only on average?).
  5. Challenge set: detection of typosquat techniques the model NEVER trained on.
  6. Ablation: retrain without TLD features / typosquat features / https proxies, and
     with heuristic-only / text-only features, to show what the score really depends on.
  7. Feature-importance share per feature group.

How to read the numbers
  The old scores (F1 0.957, AUC 0.991) came from the old, leakier split and data, so
  they are NOT comparable with these. A lower number on the new split is expected and
  is the honest one. VAL picks the threshold, TEST is looked at once.
"""

import gc
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

warnings.filterwarnings("ignore")

# ─────────────────────────── PATHS ───────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DIR = os.path.join(BASE_DIR, "features")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
RESULTS_DIR  = os.path.join(BASE_DIR, "results", "metrics")
SPLITS_DIR   = os.path.join(BASE_DIR, "data", "splits_v2")                  # v2:
REPORT_DIR   = os.path.join(BASE_DIR, "reports")                            # v2:
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

TRAIN_NPZ   = os.path.join(FEATURES_DIR, "features_train.npz")
VAL_NPZ     = os.path.join(FEATURES_DIR, "features_val.npz")
TEST_NPZ    = os.path.join(FEATURES_DIR, "features_test.npz")
HELDOUT_NPZ = os.path.join(FEATURES_DIR, "features_heldout.npz")            # v2:
HELDOUT_CSV = os.path.join(BASE_DIR, "data", "raw", "synthetic_heldout.csv")  # v2:
CONFIG_PATH = os.path.join(MODELS_DIR, "feature_config.json")               # v2:

# ═════════════════════════ SETTINGS (edit here) ═════════════════════════
RF_PARAMS = {
    "n_estimators"     : 200,
    "max_depth"        : 25,
    "min_samples_split": 10,
    "min_samples_leaf" : 4,
    "max_features"     : "sqrt",
    "max_samples"      : 0.8,
    "bootstrap"        : True,
    "oob_score"        : True,
    "class_weight"     : None,   # classes are close to 1:1 after preprocessing
    "n_jobs"           : -1,
    "random_state"     : 42,
    "verbose"          : 1,
}
MAX_ACCEPTABLE_FPR = 0.05          # a threshold is only allowed if val FPR is below this
THRESHOLD_GRID     = (0.20, 0.80, 0.01)

REQUIRE_VERIFIED_SPLIT = True      # v2: stop if split_report.json is not a passed/ok split
CALIB_BINS   = 10                  # v2:
TOP_TLDS     = 10                  # v2: rows in the per-TLD table
RUN_ABLATION = True                # v2: ~5 extra trainings; set False for a quick run
ABLATION_TREES = 100               # v2: trees per ablation model (full model uses n_estimators)

# v2: feature groups to remove, by name. "@nlp", "@heuristic", "@typosquat" are shortcuts.
ABLATIONS = {
    "no_tld_features":         ["risky_tld", "tld_len"],
    "no_typosquat_features":   ["@typosquat"],
    "no_https_proxy_features": ["https_flag", "http_no_brand_no_age"],
    "heuristics_only":         ["@nlp"],
    "text_ngrams_only":        ["@heuristic"],
}

# ─────────────────────────── LOGGING ─────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(RESULTS_DIR, "training.log"),
                            encoding="utf-8", mode="w"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# LOADING AND VERIFICATION
# ═══════════════════════════════════════════════════════════════

def load_npz(path: str):
    """Load a feature file saved by feature_extraction.py."""
    logger.info(f"   Loading {os.path.basename(path)}...")
    data = np.load(path, allow_pickle=True)
    X = sp.csr_matrix((data["data"], data["indices"], data["indptr"]),
                      shape=tuple(data["shape"]))
    y = data["labels"].astype(int)
    names = [str(n) for n in data["feature_names"]]
    logger.info(f"   Shape: {X.shape}  |  Benign: {(y == 0).sum():,}  Malicious: {(y == 1).sum():,}")
    return X, y, names


def to_dense(X: sp.csr_matrix, name: str) -> np.ndarray:
    """Convert once, up front (sparse input makes forests slow and memory-hungry)."""
    mb = X.shape[0] * X.shape[1] * 4 / 1e6
    logger.info(f"   Converting {name} to dense (about {mb:.0f} MB)...")
    return X.toarray().astype(np.float32)


def check_split_report() -> None:
    """v2: the split must have been produced from a dataset that passed preprocessing."""
    path = os.path.join(REPORT_DIR, "split_report.json")
    problems = []
    if not os.path.exists(path):
        problems.append(f"{path} not found")
    else:
        with open(path, encoding="utf-8") as fh:
            rep = json.load(fh)
        if rep.get("status") != "ok" or rep.get("preprocessing_status") != "passed":
            problems.append(f"split_report says status='{rep.get('status')}', "
                            f"preprocessing_status='{rep.get('preprocessing_status')}' "
                            "(need 'ok' and 'passed')")
    if problems:
        for m in problems:
            logger.error(f"   {m}")
        if REQUIRE_VERIFIED_SPLIT:
            logger.error("   Refusing to train. Re-run preprocess.py and split.py until they pass, "
                         "or set REQUIRE_VERIFIED_SPLIT = False for a diagnosis run only.")
            sys.exit(1)
        logger.warning("   REQUIRE_VERIFIED_SPLIT is False: continuing, do NOT report these results.")
    else:
        logger.info("   Split report: ok / preprocessing passed  [OK]")


def load_config(n_features: int, names: list) -> dict:
    """v2: the matrix must match what feature_extraction.py declared."""
    if not os.path.exists(CONFIG_PATH):
        logger.error(f"   {CONFIG_PATH} not found: run feature_extraction.py (v2) first.")
        sys.exit(1)
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        cfg = json.load(fh)
    if cfg["n_total"] != n_features:
        logger.error(f"   feature_config.json says {cfg['n_total']} features but the matrix has "
                     f"{n_features}: the .npz files and the config come from different runs.")
        sys.exit(1)
    if names[:cfg["n_heuristic"]] != cfg["heuristic_feature_names"]:
        logger.error("   Heuristic feature names in the .npz differ from feature_config.json.")
        sys.exit(1)
    logger.info(f"   Feature layout matches feature_config.json: {n_features} features "
                f"({cfg['n_heuristic']} heuristic + {cfg['n_char']} char + {cfg['n_word']} word)  [OK]")
    return cfg


def load_meta(split: str, y: np.ndarray) -> pd.DataFrame:
    """v2: row metadata for a split, aligned with the feature rows (verified by labels)."""
    df = pd.read_csv(os.path.join(SPLITS_DIR, f"{split}_urls.csv"),
                     dtype={"url": str, "label": int})
    df = df.dropna(subset=["url", "label"]).reset_index(drop=True)
    if len(df) != len(y) or not (df["label"].values == y).all():
        logger.error(f"   {split}_urls.csv does not line up with features_{split}.npz "
                     "(row count or labels differ): re-run split.py and feature_extraction.py.")
        sys.exit(1)
    return df


# ═══════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════

def proba_of(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, list(model.classes_).index(1)]


def metrics_from_proba(y: np.ndarray, p: np.ndarray, th: float) -> dict:
    pred = (p >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(
        threshold=round(float(th), 4),
        accuracy=round(accuracy_score(y, pred), 4),
        precision=round(precision_score(y, pred, pos_label=1, zero_division=0), 4),
        recall=round(recall_score(y, pred, pos_label=1, zero_division=0), 4),
        f1=round(f1_score(y, pred, pos_label=1, zero_division=0), 4),
        auc=round(roc_auc_score(y, p), 4),
        tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
        fpr=round(fp / (fp + tn), 4) if (fp + tn) else 0.0,
        fnr=round(fn / (fn + tp), 4) if (fn + tp) else 0.0,
    )


def tune_threshold(y: np.ndarray, p: np.ndarray, verbose: bool = True) -> float:
    """Best F1 among thresholds whose FPR is below MAX_ACCEPTABLE_FPR (on VAL)."""
    lo, hi, step = THRESHOLD_GRID
    rows = []
    for th in np.arange(lo, hi + 1e-9, step):
        m = metrics_from_proba(y, p, th)
        rows.append((float(th), m["f1"], m["recall"], m["precision"], m["fpr"]))
    allowed = [r for r in rows if r[4] < MAX_ACCEPTABLE_FPR]
    if allowed:
        best = max(allowed, key=lambda r: (r[1], -r[4]))
    else:                                                       # v2: say so, do not guess
        best = min(rows, key=lambda r: r[4])
        if verbose:
            logger.warning(f"   No threshold in {lo}-{hi} reaches FPR < {MAX_ACCEPTABLE_FPR}. "
                           f"Using the lowest-FPR one ({best[0]:.2f}, FPR {best[4]:.4f}).")
    if verbose:
        logger.info(f"   Chosen threshold: {best[0]:.2f}  (val F1 {best[1]:.4f}, FPR {best[4]:.4f})")
        logger.info(f"   {'Threshold':>10} {'F1':>8} {'Recall':>8} {'Precision':>10} {'FPR':>8}")
        for r in rows:
            if abs(r[0] - best[0]) <= 0.05 + 1e-9:
                tag = " <-- chosen" if abs(r[0] - best[0]) < 1e-9 else ""
                ok = "" if r[4] < MAX_ACCEPTABLE_FPR else "  (FPR too high)"
                logger.info(f"   {r[0]:>10.2f} {r[1]:>8.4f} {r[2]:>8.4f} {r[3]:>10.4f} {r[4]:>8.4f}{tag}{ok}")
    return best[0]


def log_metrics(m: dict, name: str) -> None:
    logger.info(f"\n   {name} RESULTS (threshold={m['threshold']:.2f})")
    logger.info("   " + "=" * 45)
    for k in ("accuracy", "precision", "recall", "f1", "auc"):
        logger.info(f"   {k.capitalize():<10}: {m[k]:.4f}")
    logger.info(f"   TN={m['tn']:,}  FP={m['fp']:,}   FN={m['fn']:,}  TP={m['tp']:,}")
    logger.info(f"   FPR : {m['fpr']:.4f}  ({m['fp']:,} legit URLs wrongly flagged)")
    logger.info(f"   FNR : {m['fnr']:.4f}  ({m['fn']:,} malicious URLs missed)")
    if m["fpr"] > MAX_ACCEPTABLE_FPR:
        logger.warning(f"   WARNING: FPR {m['fpr']:.4f} exceeds the {MAX_ACCEPTABLE_FPR} target")


# v2: ───────────────────────── calibration ─────────────────────────
def calibration(y: np.ndarray, p: np.ndarray, name: str) -> dict:
    """Do predicted probabilities mean what they say? (Random forests are usually
    under-confident near 0 and 1.) Matters because the extension shows a risk score."""
    brier = float(np.mean((p - y) ** 2))
    idx = np.minimum((p * CALIB_BINS).astype(int), CALIB_BINS - 1)
    ece, rows = 0.0, []
    for b in range(CALIB_BINS):
        m = idx == b
        if not m.any():
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        ece += float(m.mean()) * abs(conf - acc)
        rows.append(dict(bin=f"{b / CALIB_BINS:.1f}-{(b + 1) / CALIB_BINS:.1f}", n=int(m.sum()),
                         mean_predicted=round(conf, 3), actual_malicious=round(acc, 3)))
    logger.info(f"\n   CALIBRATION {name}: Brier {brier:.4f} | ECE {ece:.4f} (0 = perfect)")
    logger.info(f"   {'bin':>9} {'n':>7} {'predicted':>10} {'actual':>8}")
    for r in rows:
        logger.info(f"   {r['bin']:>9} {r['n']:>7,} {r['mean_predicted']:>10.3f} {r['actual_malicious']:>8.3f}")
    return dict(brier=round(brier, 4), ece=round(ece, 4), bins=rows)


# v2: ───────────────────────── breakdown tables (TEST) ─────────────────────────
def per_source(meta: pd.DataFrame, y: np.ndarray, p: np.ndarray, th: float) -> list:
    pred = (p >= th).astype(int)
    logger.info("\n   TEST per source (each source holds one class: 'correct' = recall for "
                "malicious sources, specificity for benign ones)")
    logger.info(f"   {'source':<30} {'class':<10} {'n':>7} {'correct':>9} {'mean p(mal)':>12}")
    out = []
    for src, idx in meta.groupby("source").indices.items():
        yy, pp = y[idx], pred[idx]
        cls = "malicious" if yy.mean() > 0.5 else "benign"
        row = dict(source=src, cls=cls, n=int(len(idx)), correct=round(float((pp == yy).mean()), 4),
                   mean_p=round(float(p[idx].mean()), 3))
        out.append(row)
        logger.info(f"   {src:<30} {cls:<10} {row['n']:>7,} {row['correct']:>9.3f} {row['mean_p']:>12.3f}")
    return out


def per_tld(meta: pd.DataFrame, y: np.ndarray, p: np.ndarray, th: float) -> list:
    if "suffix" not in meta.columns:
        return []
    pred = (p >= th).astype(int)
    logger.info(f"\n   TEST per TLD (top {TOP_TLDS} suffixes): is the model right on every TLD, "
                "or does it lean on the TLD?")
    logger.info(f"   {'suffix':<10} {'n':>7} {'benign':>7} {'malic.':>7} {'FPR':>8} {'FNR':>8}")
    out = []
    for suf, idx in sorted(meta.groupby("suffix").indices.items(), key=lambda kv: -len(kv[1]))[:TOP_TLDS]:
        yy, pp = y[idx], pred[idx]
        nb, nm = int((yy == 0).sum()), int((yy == 1).sum())
        fpr = float(((pp == 1) & (yy == 0)).sum() / nb) if nb else float("nan")
        fnr = float(((pp == 0) & (yy == 1)).sum() / nm) if nm else float("nan")
        out.append(dict(suffix=suf or "(none)", n=int(len(idx)), benign=nb, malicious=nm,
                        fpr=None if nb == 0 else round(fpr, 4), fnr=None if nm == 0 else round(fnr, 4)))
        f1s = f"{fpr:>8.3f}" if nb else f"{'-':>8}"
        f2s = f"{fnr:>8.3f}" if nm else f"{'-':>8}"
        logger.info(f"   {suf or '(none)':<10} {len(idx):>7,} {nb:>7,} {nm:>7,} {f1s} {f2s}")
    return out


def challenge_set(model, th: float) -> dict:
    """v2: techniques the model never saw in training (keyboard typos, bitsquats, hyphen tricks)."""
    if not (os.path.exists(HELDOUT_NPZ) and os.path.exists(HELDOUT_CSV)):
        logger.info("\n   Challenge set: features_heldout.npz or synthetic_heldout.csv not found, skipped.")
        return {}
    Xh, yh, _ = load_npz(HELDOUT_NPZ)
    df = pd.read_csv(HELDOUT_CSV, dtype=str, keep_default_na=False)
    df = df[df["url"].str.len() > 0].reset_index(drop=True)
    if len(df) != Xh.shape[0]:
        logger.warning("   Challenge set rows do not match the CSV: skipped (re-run feature_extraction.py).")
        return {}
    p = proba_of(model, Xh.toarray().astype(np.float32))
    hit = p >= th
    logger.info(f"\n   CHALLENGE SET: held-out typosquat techniques (never trained on), threshold {th:.2f}")
    logger.info(f"   {'technique':<22} {'n':>6} {'detected':>9} {'mean p(mal)':>12}")
    out = {}
    for tech, idx in df.groupby("technique").indices.items():
        out[tech] = dict(n=int(len(idx)), detected=round(float(hit[idx].mean()), 4),
                         mean_p=round(float(p[idx].mean()), 3))
        logger.info(f"   {tech:<22} {len(idx):>6,} {out[tech]['detected']:>9.3f} {out[tech]['mean_p']:>12.3f}")
    out["ALL"] = dict(n=int(len(p)), detected=round(float(hit.mean()), 4), mean_p=round(float(p.mean()), 3))
    logger.info(f"   {'ALL':<22} {len(p):>6,} {out['ALL']['detected']:>9.3f} {out['ALL']['mean_p']:>12.3f}")
    return out


# ═══════════════════════════════════════════════════════════════
# TRAINING, IMPORTANCE, ABLATION
# ═══════════════════════════════════════════════════════════════

def train(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    logger.info("\nTraining Random Forest...")
    logger.info(f"   Samples: {X.shape[0]:,} | Features: {X.shape[1]:,} | "
                f"Trees: {RF_PARAMS['n_estimators']} | Max depth: {RF_PARAMS['max_depth']}")
    model = RandomForestClassifier(**RF_PARAMS)
    t0 = time.time()
    model.fit(X, y)
    logger.info(f"   Training time: {(time.time() - t0) / 60:.1f} minutes")
    if RF_PARAMS.get("oob_score"):
        logger.info(f"   OOB score    : {model.oob_score_:.4f}")
    return model


def group_of(name: str, cfg: dict) -> str:
    if name.startswith(("char_", "word_")):
        return "text n-grams"
    if name in cfg["typosquat_feature_names"]:
        return "typosquat features"
    return "other heuristics"


def log_importance(model, names: list, cfg: dict, top_n: int = 20) -> dict:
    imp = model.feature_importances_
    order = np.argsort(imp)[::-1][:top_n]
    logger.info(f"\n   Top {top_n} features:")
    for rank, i in enumerate(order, 1):
        logger.info(f"   {rank:>4} {names[i]:>34} {imp[i]:>10.4f}")
    shares: dict = {}
    for n, v in zip(names, imp):
        g = group_of(n, cfg)
        shares[g] = shares.get(g, 0.0) + float(v)
    logger.info("   Importance share per group: " + ", ".join(f"{g} {100 * v:.1f}%" for g, v in shares.items()))
    return {g: round(v, 4) for g, v in shares.items()}


def resolve_group(spec: list, names: list, cfg: dict) -> list:
    drop = set()
    for item in spec:
        if item == "@nlp":
            drop |= {i for i, n in enumerate(names) if n.startswith(("char_", "word_"))}
        elif item == "@heuristic":
            drop |= {i for i, n in enumerate(names) if not n.startswith(("char_", "word_"))}
        elif item == "@typosquat":
            drop |= {i for i, n in enumerate(names) if n in cfg["typosquat_feature_names"]}
        elif item in names:
            drop.add(names.index(item))
        else:
            raise KeyError(f"ablation feature '{item}' not found in the feature names")
    return sorted(drop)


def run_ablation(Xtr, ytr, Xva, yva, Xte, yte, names, cfg) -> list:
    """v2: how much does each feature group matter? Same data, fewer trees, threshold
    re-tuned on VAL under the same FPR limit for every variant."""
    logger.info("\n" + "=" * 60)
    logger.info(f"ABLATION ({ABLATION_TREES} trees per model, threshold re-tuned on VAL each time)")
    logger.info("=" * 60)
    variants = [("all_features", [])] + [(k, resolve_group(v, names, cfg)) for k, v in ABLATIONS.items()]
    out = []
    for label, drop in variants:
        keep = np.array([i for i in range(len(names)) if i not in set(drop)])
        params = dict(RF_PARAMS, n_estimators=ABLATION_TREES, verbose=0, oob_score=False)
        t0 = time.time()
        m = RandomForestClassifier(**params).fit(Xtr[:, keep], ytr)
        th = tune_threshold(yva, proba_of(m, Xva[:, keep]), verbose=False)
        res = metrics_from_proba(yte, proba_of(m, Xte[:, keep]), th)
        out.append(dict(variant=label, n_features=int(len(keep)), **{k: res[k] for k in
                        ("threshold", "f1", "auc", "fpr", "fnr")}))
        logger.info(f"   {label:<26} {len(keep):>4} features | {(time.time() - t0) / 60:4.1f} min")
        del m
        gc.collect()
    logger.info(f"\n   {'variant':<26} {'features':>8} {'thr':>5} {'TEST F1':>8} {'AUC':>7} {'FPR':>7} {'FNR':>7}")
    base = out[0]
    for r in out:
        d = f"  (F1 {r['f1'] - base['f1']:+.4f})" if r is not base else ""
        logger.info(f"   {r['variant']:<26} {r['n_features']:>8} {r['threshold']:>5.2f} {r['f1']:>8.4f} "
                    f"{r['auc']:>7.4f} {r['fpr']:>7.4f} {r['fnr']:>7.4f}{d}")
    logger.info("   Reading it: a big drop without a group means the model leans on that group. "
                "heuristics_only / text_ngrams_only show how much each half carries alone.")
    return out


# ═══════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════

def save(model, threshold, report: dict) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODELS_DIR, f"rf_model_{ts}.joblib")
    joblib.dump(model, model_path, compress=3)
    joblib.dump(model, os.path.join(MODELS_DIR, "rf_model_latest.joblib"), compress=3)
    logger.info(f"\n   Model saved : {model_path} ({os.path.getsize(model_path) / 1e6:.1f} MB)")
    with open(os.path.join(MODELS_DIR, "threshold.json"), "w") as fh:
        json.dump({"threshold": threshold}, fh, indent=2)
    report.update(training_date=ts, model_path=model_path, threshold=threshold,
                  n_features=int(model.n_features_in_), n_estimators=int(model.n_estimators),
                  oob_score=round(float(model.oob_score_), 4) if RF_PARAMS.get("oob_score") else None,
                  rf_params={k: v for k, v in RF_PARAMS.items() if k != "verbose"})
    path = os.path.join(RESULTS_DIR, f"rf_metrics_{ts}.json")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    logger.info(f"   Metrics     : {path}")
    return model_path


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    logger.info("RANDOM FOREST TRAINING  (v2)")
    logger.info("=" * 60)
    for p in (TRAIN_NPZ, VAL_NPZ, TEST_NPZ):
        if not os.path.exists(p):
            logger.error(f"File not found: {p}: run feature_extraction.py first.")
            return

    logger.info("\nChecking inputs...")
    check_split_report()

    logger.info("\nLoading feature files...")
    Xtr_s, ytr, names = load_npz(TRAIN_NPZ)
    Xva_s, yva, nva = load_npz(VAL_NPZ)
    Xte_s, yte, nte = load_npz(TEST_NPZ)
    assert Xtr_s.shape[1] == Xva_s.shape[1] == Xte_s.shape[1], "feature count differs between splits"
    assert names == nva == nte, "feature names differ between splits"
    cfg = load_config(Xtr_s.shape[1], names)
    meta_te = load_meta("test", yte)

    Xtr, Xva, Xte = to_dense(Xtr_s, "train"), to_dense(Xva_s, "val"), to_dense(Xte_s, "test")
    del Xtr_s, Xva_s, Xte_s
    gc.collect()

    model = train(Xtr, ytr)
    importance = log_importance(model, names, cfg)

    logger.info("\nTuning threshold on VAL (FPR constrained)...")
    pva = proba_of(model, Xva)
    threshold = tune_threshold(yva, pva)

    val_m = metrics_from_proba(yva, pva, threshold)
    log_metrics(val_m, "VAL")
    pte = proba_of(model, Xte)
    test_m = metrics_from_proba(yte, pte, threshold)     # TEST is looked at once
    log_metrics(test_m, "TEST")

    report = dict(val_metrics=val_m, test_metrics=test_m, importance_share=importance,
                  train_samples=int(len(ytr)), val_samples=int(len(yva)), test_samples=int(len(yte)),
                  calibration=dict(val=calibration(yva, pva, "VAL"), test=calibration(yte, pte, "TEST")),
                  test_per_source=per_source(meta_te, yte, pte, threshold),
                  test_per_tld=per_tld(meta_te, yte, pte, threshold),
                  challenge_heldout_techniques=challenge_set(model, threshold))

    if RUN_ABLATION:
        report["ablation"] = run_ablation(Xtr, ytr, Xva, yva, Xte, yte, names, cfg)

    model_path = save(model, threshold, report)

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"   Threshold {threshold:.2f} | VAL F1 {val_m['f1']:.4f} FPR {val_m['fpr']:.4f} | "
                f"TEST F1 {test_m['f1']:.4f} AUC {test_m['auc']:.4f} FPR {test_m['fpr']:.4f} FNR {test_m['fnr']:.4f}")
    gap = abs(val_m["f1"] - test_m["f1"])
    if gap > 0.03:
        logger.warning(f"   Val/Test F1 gap = {gap:.4f}: possible overfitting to the validation split.")
    else:
        logger.info(f"   Val/Test F1 gap = {gap:.4f}  healthy")
    logger.info(f"   Model: {model_path}")
    logger.info("   Next : model_loader.py --selftest, then the regression URLs")


if __name__ == "__main__":
    main()