#!/usr/bin/env python3
"""
train_compare.py  (v2)  -  train candidates, compare them with the OLD model, keep the best
----------------------------------------------------------------------------------------------
Run from the project root (venv active). No flags needed, all options are in SETTINGS:

    python scripts/train_compare.py

Order of work (see the reply for the whole procedure):
    python scripts/snapshot_baseline.py       # once, BEFORE changing feature_extraction.py
    python scripts/feature_extraction.py      # with TLD_BLIND_TEXT = True
    python scripts/train_compare.py           # this file

Reads   features/features_{train,val,test}.npz, features/features_heldout.npz (optional)
        models/feature_config.json, data/splits_v2/{val,test}_urls.csv, reports/split_report.json
        results/baseline/*  (written by snapshot_baseline.py)
        data/reference/regression_urls.csv (url, expected, note)
Writes  models/<winner>_<time>.joblib, models/rf_model_latest.joblib (legacy name the
        loader reads), models/threshold.json, results/metrics/compare_<time>.json

Candidates (all trained on the SAME new TLD-blind features and the SAME train split)
    baseline_rf   the model you had before (probabilities frozen by snapshot_baseline.py)
    rf_tldblind   Random Forest, same settings as before          -> effect of the features
    hgb_tldblind  HistGradientBoosting                            -> effect of the model type

Rules that keep the comparison honest
  * Every threshold is chosen on VAL only. TEST is looked at once.
  * Models are compared at EQUAL false-positive rates (1 / 2 / 5 %), not at whatever
    threshold each one happened to get, because F1 at different FPRs is not comparable.
  * The winner is picked on VAL (recall at SELECT_AT_FPR), never on TEST.
  * The difference to the baseline gets a bootstrap interval (resampled by domain group
    when the split file has group_key). "Better" requires the interval to exclude zero.
  * The regression URLs (sites you saw falsely flagged) are scored by every candidate.
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
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (average_precision_score, log_loss, precision_recall_curve,
                             roc_auc_score)

warnings.filterwarnings("ignore")

# ─────────────────────────── PATHS ───────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR  = os.path.join(BASE_DIR, "scripts")
FEATURES_DIR = os.path.join(BASE_DIR, "features")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
RESULTS_DIR  = os.path.join(BASE_DIR, "results", "metrics")
BASELINE_DIR = os.path.join(BASE_DIR, "results", "baseline")
SPLITS_DIR   = os.path.join(BASE_DIR, "data", "splits_v2")
REPORT_DIR   = os.path.join(BASE_DIR, "reports")
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_NPZ   = os.path.join(FEATURES_DIR, "features_train.npz")
VAL_NPZ     = os.path.join(FEATURES_DIR, "features_val.npz")
TEST_NPZ    = os.path.join(FEATURES_DIR, "features_test.npz")
HELDOUT_NPZ = os.path.join(FEATURES_DIR, "features_heldout.npz")
HELDOUT_CSV = os.path.join(BASE_DIR, "data", "raw", "synthetic_heldout.csv")
CONFIG_PATH = os.path.join(MODELS_DIR, "feature_config.json")
REGRESSION_CSV = os.path.join(BASE_DIR, "data", "reference", "regression_urls.csv")

# ═════════════════════════ SETTINGS (edit here) ═════════════════════════
RUN_RF  = True
RUN_HGB = True

RF_PARAMS = {                       # unchanged from train_model.py
    "n_estimators": 200, "max_depth": 25, "min_samples_split": 10, "min_samples_leaf": 4,
    "max_features": "sqrt", "max_samples": 0.8, "bootstrap": True, "oob_score": False,
    "class_weight": None, "n_jobs": -1, "random_state": 42, "verbose": 0,
}
HGB_PARAMS = {
    "learning_rate": 0.08, "max_leaf_nodes": 63, "min_samples_leaf": 30,
    "l2_regularization": 1.0, "max_features": 0.5, "random_state": 42,
    "early_stopping": False,        # the number of rounds is chosen on VAL below
}
HGB_MAX_ITER = 600                  # upper limit; the best number is found on VAL log-loss

# Operating point that is SAVED for the extension (best F1 under this FPR limit, on VAL).
MAX_ACCEPTABLE_FPR = 0.03
THRESHOLD_GRID     = (0.05, 0.95, 0.01)

# How candidates are compared
FPR_POINTS     = (0.01, 0.02, 0.05)   # recall is reported at these false-positive rates
SELECT_AT_FPR  = 0.02                 # the winner = best VAL recall at this FPR
COMPARE_AT_FPR = 0.05                 # bootstrap difference to the baseline at this FPR
PRECISION_TARGET = 0.95               # research objective: what recall do we get at 95% precision?
RECALL_TARGET    = 0.95               # ...and what precision at 95% recall?
BOOTSTRAP_N   = 1000
WATCH_SUFFIXES = ("com", "org", "lk", "in", "co.uk", "top", "io")

REQUIRE_VERIFIED_SPLIT = True
EXPECT_TLD_BLIND       = True         # feature_config.json must say the text features are TLD-blind
LOADER_MODEL_FILE      = "rf_model_latest.joblib"   # legacy name: model_loader.py loads this file

# ─────────────────────────── LOGGING ─────────────────────────
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(os.path.join(RESULTS_DIR, "train_compare.log"),
                                  encoding="utf-8", mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
log = logger.info


# ═══════════════════════════════════════════════════════════════
# LOADING AND CHECKS
# ═══════════════════════════════════════════════════════════════

def load_npz(path: str):
    d = np.load(path, allow_pickle=True)
    X = sp.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"]))
    return X, d["labels"].astype(int), [str(n) for n in d["feature_names"]]


def dense(X: sp.csr_matrix) -> np.ndarray:
    return X.toarray().astype(np.float32)


def check_split_report() -> None:
    path = os.path.join(REPORT_DIR, "split_report.json")
    problem = None
    if not os.path.exists(path):
        problem = f"{path} not found"
    else:
        with open(path, encoding="utf-8") as fh:
            rep = json.load(fh)
        if rep.get("status") != "ok" or rep.get("preprocessing_status") != "passed":
            problem = (f"split_report says status='{rep.get('status')}', "
                       f"preprocessing_status='{rep.get('preprocessing_status')}'")
    if problem:
        logger.error(f"   {problem}")
        if REQUIRE_VERIFIED_SPLIT:
            logger.error("   Refusing to train on an unverified split.")
            sys.exit(1)
        logger.warning("   Continuing without a verified split: do NOT report these results.")
    else:
        log("   Split report ok / preprocessing passed  [OK]")


def check_config(n_features: int, names: list) -> dict:
    if not os.path.exists(CONFIG_PATH):
        logger.error(f"   {CONFIG_PATH} not found: run feature_extraction.py first.")
        sys.exit(1)
    with open(CONFIG_PATH, encoding="utf-8") as fh:
        cfg = json.load(fh)
    if cfg["n_total"] != n_features or names[:cfg["n_heuristic"]] != cfg["heuristic_feature_names"]:
        logger.error("   feature_config.json does not match the .npz files (different runs).")
        sys.exit(1)
    if EXPECT_TLD_BLIND and not cfg.get("tld_blind_text", False):
        logger.error("   feature_config.json has no tld_blind_text=true: the features were extracted "
                     "WITHOUT the TLD-blind patch. Apply the patch and re-run feature_extraction.py.")
        sys.exit(1)
    log(f"   Feature layout matches feature_config.json ({n_features} features, "
        f"tld_blind_text={cfg.get('tld_blind_text', False)})  [OK]")
    return cfg


def load_meta(split: str, y: np.ndarray) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(SPLITS_DIR, f"{split}_urls.csv"), dtype={"url": str, "label": int})
    df = df.dropna(subset=["url", "label"]).reset_index(drop=True)
    if len(df) != len(y) or not (df["label"].values == y).all():
        logger.error(f"   {split}_urls.csv does not line up with features_{split}.npz: re-run "
                     "split.py and feature_extraction.py.")
        sys.exit(1)
    return df


def load_baseline(yva: np.ndarray, yte: np.ndarray):
    path = os.path.join(BASELINE_DIR, "baseline_probs.npz")
    if not os.path.exists(path):
        logger.warning("   No baseline snapshot (results/baseline/baseline_probs.npz): the comparison "
                       "with the old model is skipped. Run snapshot_baseline.py BEFORE re-extracting features.")
        return None
    d = np.load(path)
    if not ((d["val_y"] == yva).all() and (d["test_y"] == yte).all()):
        logger.error("   The baseline snapshot was made on different rows (labels differ): the split "
                     "changed since the snapshot. Take a new snapshot or restore the old split.")
        sys.exit(1)
    return dict(pva=d["val_proba"], pte=d["test_proba"], th=float(d["threshold"]))


# ═══════════════════════════════════════════════════════════════
# METRICS
# ═══════════════════════════════════════════════════════════════

def proba_of(model, X) -> np.ndarray:
    return model.predict_proba(X)[:, list(model.classes_).index(1)]


def metrics_at(y: np.ndarray, p: np.ndarray, th: float) -> dict:
    pred = p >= th
    pos, neg = y == 1, y == 0
    tp, fp = int((pred & pos).sum()), int((pred & neg).sum())
    fn, tn = int((~pred & pos).sum()), int((~pred & neg).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return dict(threshold=float(th), accuracy=(tp + tn) / len(y), precision=prec, recall=rec,
                f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                fpr=fp / (fp + tn) if fp + tn else 0.0, fnr=1 - rec, tp=tp, fp=fp, fn=fn, tn=tn)


def tune_threshold(y: np.ndarray, p: np.ndarray, name: str) -> float:
    """Best F1 among thresholds with VAL FPR below MAX_ACCEPTABLE_FPR."""
    lo, hi, step = THRESHOLD_GRID
    rows = [metrics_at(y, p, t) for t in np.arange(lo, hi + 1e-9, step)]
    ok = [r for r in rows if r["fpr"] < MAX_ACCEPTABLE_FPR]
    if ok:
        best = max(ok, key=lambda r: (r["f1"], -r["fpr"]))
    else:
        best = min(rows, key=lambda r: r["fpr"])
        logger.warning(f"   {name}: no threshold reaches VAL FPR < {MAX_ACCEPTABLE_FPR}; "
                       f"using the lowest-FPR one ({best['threshold']:.2f}).")
    log(f"   {name}: threshold {best['threshold']:.2f}  (VAL F1 {best['f1']:.4f}, "
        f"recall {best['recall']:.4f}, FPR {best['fpr']:.4f})")
    return best["threshold"]


def th_for_fpr(y: np.ndarray, p: np.ndarray, target: float) -> float:
    """Smallest threshold whose FPR on these rows is <= target (used with p >= th)."""
    return float(np.nextafter(np.quantile(p[y == 0], 1 - target), np.inf))


def th_for_precision(y: np.ndarray, p: np.ndarray, target: float):
    pr, rc, th = precision_recall_curve(y, p)
    ok = np.where(pr[:-1] >= target)[0]
    return float(th[ok[np.argmax(rc[:-1][ok])]]) if len(ok) else None


def th_for_recall(y: np.ndarray, p: np.ndarray, target: float):
    pr, rc, th = precision_recall_curve(y, p)
    ok = np.where(rc[:-1] >= target)[0]
    return float(th[ok[np.argmax(pr[:-1][ok])]]) if len(ok) else None


def ece_of(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum((idx == b).mean() * abs(p[idx == b].mean() - y[idx == b].mean())
                     for b in range(bins) if (idx == b).any()))


def evaluate(name: str, pva, pte, yva, yte, th: float) -> dict:
    """All numbers for one candidate. Every threshold comes from VAL."""
    res = dict(name=name, threshold=float(th), val=metrics_at(yva, pva, th),
               test=metrics_at(yte, pte, th))
    res["test"].update(auc=float(roc_auc_score(yte, pte)), pr_auc=float(average_precision_score(yte, pte)),
                       ece=ece_of(yte, pte))
    res["val_pr_auc"] = float(average_precision_score(yva, pva))
    res["val_recall_at_select_fpr"] = metrics_at(yva, pva, th_for_fpr(yva, pva, SELECT_AT_FPR))["recall"]
    ops = {}
    for f in FPR_POINTS:                                    # equal-FPR comparison
        t = th_for_fpr(yva, pva, f)
        ops[f"fpr_{f}"] = metrics_at(yte, pte, t)
    for key, fn_, target in (("precision_target", th_for_precision, PRECISION_TARGET),
                             ("recall_target", th_for_recall, RECALL_TARGET)):
        t = fn_(yva, pva, target)
        ops[key] = metrics_at(yte, pte, t) if t is not None else None
    res["ops"] = ops
    return res


def segment_rates(meta: pd.DataFrame, y, p, th: float) -> dict:
    out = {}
    pred = p >= th
    if "suffix" in meta.columns:
        suf = meta["suffix"].astype(str).str.lstrip(".").str.lower().values
        for s in WATCH_SUFFIXES:
            m = suf == s
            nb, nm = int((m & (y == 0)).sum()), int((m & (y == 1)).sum())
            out[s] = dict(n_benign=nb, n_malicious=nm,
                          fpr=float((pred & m & (y == 0)).sum() / nb) if nb else None,
                          fnr=float((~pred & m & (y == 1)).sum() / nm) if nm else None)
    return out


def source_rates(meta: pd.DataFrame, y, p, th: float) -> dict:
    if "source" not in meta.columns:
        return {}
    pred = (p >= th).astype(int)
    return {src: dict(n=int(len(i)), cls="malicious" if y[i].mean() > 0.5 else "benign",
                      correct=float((pred[i] == y[i]).mean()))
            for src, i in meta.groupby("source").indices.items()}


def paired_bootstrap(y, pred_a, pred_b, groups, n_boot: int, seed: int = 42) -> dict:
    """Difference (b - a) in recall, FPR and F1 with a 95% interval. Resamples whole groups
    (domains) so near-duplicate rows are not counted as independent evidence."""
    G = int(groups.max()) + 1

    def counts(pred):
        pos, neg = y == 1, y == 0
        mk = lambda mask: np.bincount(groups, weights=mask.astype(float), minlength=G)
        return mk(pred & pos), mk(pred & neg), mk(~pred & pos), mk(~pred & neg)   # tp fp fn tn

    A, B = counts(pred_a), counts(pred_b)

    def rates(c):
        tp, fp, fn, tn = c
        rec = tp / (tp + fn) if tp + fn else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        return rec, fpr, (2 * tp / (2 * tp + fp + fn) if tp else 0.0)

    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, G, G), minlength=G).astype(float)
        ra, rb = rates([w @ v for v in A]), rates([w @ v for v in B])
        diffs.append([b - a for a, b in zip(ra, rb)])
    diffs = np.array(diffs)
    point = [b - a for a, b in zip(rates([v.sum() for v in A]), rates([v.sum() for v in B]))]
    out = {}
    for k, name in enumerate(("recall", "fpr", "f1")):
        lo, hi = np.percentile(diffs[:, k], [2.5, 97.5])
        out[name] = dict(diff=float(point[k]), lo=float(lo), hi=float(hi))
    return out


# ═══════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════

def train_rf(Xtr, ytr):
    log("\nTraining Random Forest (TLD-blind features)...")
    t0 = time.time()
    m = RandomForestClassifier(**RF_PARAMS).fit(Xtr, ytr)
    log(f"   done in {(time.time() - t0) / 60:.1f} min")
    return m


def train_hgb(Xtr, ytr, Xva, yva):
    """The number of boosting rounds is chosen on VAL log-loss (not on a random slice of
    train: train rows of one domain family would leak into that slice)."""
    log("\nTraining HistGradientBoosting (TLD-blind features)...")
    t0 = time.time()
    m = HistGradientBoostingClassifier(max_iter=HGB_MAX_ITER, **HGB_PARAMS).fit(Xtr, ytr)
    losses = [log_loss(yva, p[:, 1]) for p in m.staged_predict_proba(Xva)]
    best = int(np.argmin(losses)) + 1
    log(f"   best number of rounds on VAL log-loss: {best} of {HGB_MAX_ITER} "
        f"(loss {losses[best - 1]:.4f}; at the last round {losses[-1]:.4f})")
    if best >= HGB_MAX_ITER - 5:
        logger.warning("   still improving at the last round: raise HGB_MAX_ITER.")
    if best < HGB_MAX_ITER:
        m = HistGradientBoostingClassifier(max_iter=best, **HGB_PARAMS).fit(Xtr, ytr)   # same seed: same trees
    log(f"   done in {(time.time() - t0) / 60:.1f} min")
    return m


# ═══════════════════════════════════════════════════════════════
# REGRESSION URLS AND CHALLENGE SET
# ═══════════════════════════════════════════════════════════════

def score_regression(models: dict):
    """Score data/reference/regression_urls.csv with the new models (model only: no whitelist,
    rules or Safe Browsing). Uses the vectorizers/scaler that feature_extraction.py just saved."""
    if not os.path.exists(REGRESSION_CSV):
        logger.warning(f"   {REGRESSION_CSV} not found: regression check skipped.")
        return None, {}
    sys.path.insert(0, SCRIPTS_DIR)
    from feature_extraction import extract_heuristic_features, segment_url
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    vc = joblib.load(os.path.join(MODELS_DIR, "vectorizer_char.joblib"))
    vw = joblib.load(os.path.join(MODELS_DIR, "vectorizer_word.joblib"))
    df = pd.read_csv(REGRESSION_CSV, dtype=str, keep_default_na=False)
    rows = []
    for u in df["url"]:
        h = scaler.transform(np.array(extract_heuristic_features(u), dtype=np.float32).reshape(1, -1)).ravel()
        s = segment_url(u)
        rows.append(np.concatenate([h, vc.transform([s]).toarray().ravel(), vw.transform([s]).toarray().ravel()]))
    X = np.array(rows, dtype=np.float32)
    return df, {n: proba_of(m, X) for n, m in models.items()}


def regression_summary(name: str, df: pd.DataFrame, p: np.ndarray, th: float) -> dict:
    flag = p >= th
    ben, mal = (df["expected"] == "BENIGN").values, (df["expected"] == "MALICIOUS").values
    return dict(benign_total=int(ben.sum()), benign_flagged=int((flag & ben).sum()),
                malicious_total=int(mal.sum()), malicious_detected=int((flag & mal).sum()),
                flagged_benign_urls=[(u, round(float(x), 3)) for u, x, f, b in zip(df["url"], p, flag, ben) if f and b])


def challenge_detection(models: dict, thresholds: dict) -> dict:
    if not (os.path.exists(HELDOUT_NPZ) and os.path.exists(HELDOUT_CSV)):
        return {}
    Xh, _, _ = load_npz(HELDOUT_NPZ)
    df = pd.read_csv(HELDOUT_CSV, dtype=str, keep_default_na=False)
    df = df[df["url"].str.len() > 0].reset_index(drop=True)
    if len(df) != Xh.shape[0] or "technique" not in df.columns:
        return {}
    Xh = dense(Xh)
    return {n: round(float((proba_of(m, Xh) >= thresholds[n]).mean()), 4) for n, m in models.items()}


# ═══════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════

def f4(v, w=7):
    return f"{'-':>{w}}" if v is None else f"{v:>{w}.4f}"


def print_tables(results: list) -> None:
    log("\n" + "=" * 100)
    log("1. MAIN TABLE: TEST results (thresholds chosen on VAL)")
    log("=" * 100)
    log(f"{'candidate':<14}{'thr':>6}{'prec':>8}{'recall':>8}{'F1':>8}{'FPR':>8}{'AUC':>8}{'PR-AUC':>8}{'ECE':>8}  |"
        + "".join(f"{'rec@' + str(int(f * 100)) + '%':>9}" for f in FPR_POINTS))
    for r in results:
        t = r["test"]
        at = "".join(f4(r["ops"][f"fpr_{f}"]["recall"], 9) for f in FPR_POINTS)
        log(f"{r['name']:<14}{r['threshold']:>6.2f}{t['precision']:>8.4f}{t['recall']:>8.4f}{t['f1']:>8.4f}"
            f"{t['fpr']:>8.4f}{t['auc']:>8.4f}{t['pr_auc']:>8.4f}{t['ece']:>8.4f}  |{at}")
    log("   'rec@x%' = recall at FPR x%: the threshold is set on VAL so that VAL has FPR x%, then recall is read on TEST.")
    log("   Only these columns compare models fairly; precision/recall/F1 above sit at different FPRs.")

    log("\n" + "=" * 100)
    log(f"2. RESEARCH OBJECTIVE: precision >= {PRECISION_TARGET:.0%} and recall >= {RECALL_TARGET:.0%} (TEST, threshold from VAL)")
    log("=" * 100)
    log(f"{'candidate':<14}{'--- threshold for precision>=95% ---':>40}{'--- threshold for recall>=95% ---':>40}")
    log(f"{'':<14}{'precision':>12}{'recall':>10}{'FPR':>8}{'':>10}{'precision':>12}{'recall':>10}{'FPR':>8}")
    for r in results:
        a, b = r["ops"]["precision_target"], r["ops"]["recall_target"]
        sa = f"{f4(a['precision'], 12)}{f4(a['recall'], 10)}{f4(a['fpr'], 8)}" if a else f"{'not reachable on VAL':>30}"
        sb = f"{f4(b['precision'], 12)}{f4(b['recall'], 10)}{f4(b['fpr'], 8)}" if b else f"{'not reachable on VAL':>30}"
        log(f"{r['name']:<14}{sa}{'':>10}{sb}")


def print_segments(results: list, seg: dict, src: dict) -> None:
    names = [r["name"] for r in results]
    log("\n" + "=" * 100)
    log("3. WHERE THE ERRORS ARE (TEST, deployed thresholds): does the model lean on the TLD?")
    log("=" * 100)
    log(f"{'suffix':<8}{'#benign':>9}{'#malic.':>9}   " + "".join(f"{n[:13]:>14}" for n in names) + "      (cells: FPR / FNR)")
    for s in WATCH_SUFFIXES:
        row0 = seg[names[0]].get(s)
        if not row0:
            continue
        cells = ""
        for n in names:
            c = seg[n][s]
            fp = "-" if c["fpr"] is None else f"{c['fpr']:.2f}"
            fn = "-" if c["fnr"] is None else f"{c['fnr']:.2f}"
            cells += f"{fp + ' / ' + fn:>14}"
        log(f"{s:<8}{row0['n_benign']:>9,}{row0['n_malicious']:>9,}   {cells}")
    if src and src.get(names[0]):
        log(f"\n{'source (correct rate)':<30}{'class':<11}{'n':>7}   " + "".join(f"{n[:13]:>14}" for n in names))
        for s, v in src[names[0]].items():
            log(f"{s:<30}{v['cls']:<11}{v['n']:>7,}   " + "".join(f"{src[n][s]['correct']:>14.3f}" for n in names))


def save_winner(name: str, model, threshold: float, report: dict) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(MODELS_DIR, f"{name}_{stamp}.joblib")
    joblib.dump(model, path, compress=3)
    joblib.dump(model, os.path.join(MODELS_DIR, LOADER_MODEL_FILE), compress=3)
    with open(os.path.join(MODELS_DIR, "threshold.json"), "w", encoding="utf-8") as fh:
        json.dump({"threshold": threshold, "model": name}, fh, indent=2)
    out = os.path.join(RESULTS_DIR, f"compare_{stamp}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=lambda o: float(o) if isinstance(o, np.floating)
                  else int(o) if isinstance(o, np.integer) else str(o))
    log(f"\nSaved {path}\n      {os.path.join(MODELS_DIR, LOADER_MODEL_FILE)} (what model_loader.py reads)\n      {out}")
    return path


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main() -> None:
    log("TRAIN + COMPARE  (v2)")
    log("=" * 60)
    for p in (TRAIN_NPZ, VAL_NPZ, TEST_NPZ):
        if not os.path.exists(p):
            logger.error(f"Missing {p}: run feature_extraction.py first.")
            return
    check_split_report()
    Xtr_s, ytr, names = load_npz(TRAIN_NPZ)
    Xva_s, yva, nva = load_npz(VAL_NPZ)
    Xte_s, yte, nte = load_npz(TEST_NPZ)
    assert names == nva == nte, "feature names differ between splits"
    check_config(Xtr_s.shape[1], names)
    meta_te = load_meta("test", yte)
    if "group_key" in meta_te.columns:
        groups = pd.factorize(meta_te["group_key"])[0]
        log(f"   Bootstrap resamples {groups.max() + 1:,} domain groups (group_key)")
    else:
        groups = np.arange(len(yte))
        logger.warning("   test_urls.csv has no group_key: bootstrap resamples single rows (intervals too narrow).")
    Xtr, Xva, Xte = dense(Xtr_s), dense(Xva_s), dense(Xte_s)
    del Xtr_s, Xva_s, Xte_s
    gc.collect()

    cands, models = {}, {}
    base = load_baseline(yva, yte)
    if base:
        cands["baseline_rf"] = dict(pva=base["pva"], pte=base["pte"], th=base["th"])
    if RUN_RF:
        m = train_rf(Xtr, ytr)
        models["rf_tldblind"] = m
    if RUN_HGB:
        m = train_hgb(Xtr, ytr, Xva, yva)
        models["hgb_tldblind"] = m
    if not models:
        logger.error("Nothing to train (RUN_RF and RUN_HGB are both False).")
        return
    log("\nChoosing thresholds on VAL...")
    for n, m in models.items():
        pva, pte = proba_of(m, Xva), proba_of(m, Xte)
        cands[n] = dict(pva=pva, pte=pte, th=tune_threshold(yva, pva, n))

    results = [evaluate(n, c["pva"], c["pte"], yva, yte, c["th"]) for n, c in cands.items()]
    by_name = {r["name"]: r for r in results}
    seg = {n: segment_rates(meta_te, yte, c["pte"], c["th"]) for n, c in cands.items()}
    src = {n: source_rates(meta_te, yte, c["pte"], c["th"]) for n, c in cands.items()}
    print_tables(results)
    print_segments(results, seg, src)

    # winner among the NEW models, chosen on VAL only
    new = [r for r in results if r["name"] in models]
    winner = max(new, key=lambda r: (r["val_recall_at_select_fpr"], r["val_pr_auc"]))
    log("\n" + "=" * 100)
    log(f"4. WINNER among the new models (VAL recall at FPR {SELECT_AT_FPR:.0%}): {winner['name']}")
    for r in new:
        log(f"   {r['name']:<14} VAL recall at FPR {SELECT_AT_FPR:.0%}: {r['val_recall_at_select_fpr']:.4f} | VAL PR-AUC {r['val_pr_auc']:.4f}")

    boot, verdict_lines, verdict = None, [], "NO BASELINE"
    if base:
        log("\n" + "=" * 100)
        log(f"5. IS THE WINNER REALLY BETTER THAN THE OLD MODEL?  (equal FPR {COMPARE_AT_FPR:.0%}, "
            f"{BOOTSTRAP_N} bootstrap resamples)")
        log("=" * 100)
        ta = th_for_fpr(yva, cands["baseline_rf"]["pva"], COMPARE_AT_FPR)
        tb = th_for_fpr(yva, cands[winner["name"]]["pva"], COMPARE_AT_FPR)
        boot = paired_bootstrap(yte, cands["baseline_rf"]["pte"] >= ta, cands[winner["name"]]["pte"] >= tb,
                                groups, BOOTSTRAP_N)
        for k, label in (("recall", "recall"), ("fpr", "FPR   "), ("f1", "F1    ")):
            d = boot[k]
            log(f"   {label} new - old: {d['diff']:+.4f}   95% interval [{d['lo']:+.4f}, {d['hi']:+.4f}]")
        c1 = boot["recall"]["lo"] > 0
        c2 = winner["test"]["pr_auc"] >= by_name["baseline_rf"]["test"]["pr_auc"] - 0.002
        log(f"   recall gain is significant (interval above 0): {'YES' if c1 else 'NO'}")
        log(f"   PR-AUC not lower than the old model          : {'YES' if c2 else 'NO'} "
            f"({winner['test']['pr_auc']:.4f} vs {by_name['baseline_rf']['test']['pr_auc']:.4f})")
    else:
        c1 = c2 = False

    # regression URLs
    try:
        df_reg, reg_p = score_regression({n: models[n] for n in models})
    except Exception as exc:                                   # the models must still be saved
        logger.warning(f"   regression check skipped ({type(exc).__name__}: {exc})")
        df_reg, reg_p = None, {}
    c3 = True
    if df_reg is not None:
        log("\n" + "=" * 100)
        log("6. REGRESSION URLS (model only, no whitelist / rules / Safe Browsing)")
        log("=" * 100)
        reg = {}
        if base and os.path.exists(os.path.join(BASELINE_DIR, "baseline_regression.json")):
            with open(os.path.join(BASELINE_DIR, "baseline_regression.json"), encoding="utf-8") as fh:
                bj = json.load(fh)
            if bj.get("proba"):
                pb = np.array([bj["proba"].get(u, np.nan) for u in df_reg["url"]])
                reg["baseline_rf"] = regression_summary("baseline_rf", df_reg, np.nan_to_num(pb), bj["threshold"])
        for n, p in reg_p.items():
            reg[n] = regression_summary(n, df_reg, p, cands[n]["th"])
        for n, s in reg.items():
            log(f"   {n:<14} benign flagged {s['benign_flagged']}/{s['benign_total']} | "
                f"malicious detected {s['malicious_detected']}/{s['malicious_total']}")
            for u, x in s["flagged_benign_urls"]:
                log(f"        FALSE POSITIVE  p={x:.3f}  {u}")
        if "baseline_rf" in reg:
            c3 = reg[winner["name"]]["benign_flagged"] <= reg["baseline_rf"]["benign_flagged"]
            log(f"   winner flags no more benign URLs than the old model: {'YES' if c3 else 'NO'}")
    else:
        reg = {}

    try:
        chall = challenge_detection(models, {n: cands[n]["th"] for n in models})
    except Exception as exc:
        logger.warning(f"   challenge set skipped ({type(exc).__name__}: {exc})")
        chall = {}
    if chall:
        log("\n7. HELD-OUT TYPOSQUAT TECHNIQUES detected at the deployed thresholds: "
            + ", ".join(f"{n} {v:.3f}" for n, v in chall.items()) + "   (old model: 0.977, same generator: sanity check only)")

    log("\n" + "=" * 100)
    if base:
        if c1 and c2 and c3:
            verdict = "BETTER"
        elif c1:
            verdict = "MIXED"
        else:
            verdict = "NOT SHOWN TO BE BETTER"
        log(f"VERDICT: {verdict}")
        if verdict == "BETTER":
            log("   The winner recovers more malicious URLs at the same false-positive rate, the gain is outside "
                "the bootstrap noise, and it does not add regression false positives. Keep it.")
        elif verdict == "MIXED":
            log("   Recall improved, but PR-AUC dropped or the winner flags more of the regression sites. "
                "Read tables 5 and 6 before deciding.")
        else:
            log("   The gain is not distinguishable from noise. Do NOT switch to the new model.\n"
                "   The new vectorizers/scaler already replaced the old ones, so put the old set back (PowerShell):\n"
                "      Copy-Item models_backup_baseline\\* models\\ -Recurse -Force")
    else:
        log("VERDICT: no baseline snapshot, so 'better than before' cannot be shown.")

    report = dict(results=results, segments=seg, sources=src, winner=winner["name"], verdict=verdict,
                  bootstrap=boot, regression=reg, challenge=chall,
                  settings=dict(max_acceptable_fpr=MAX_ACCEPTABLE_FPR, select_at_fpr=SELECT_AT_FPR,
                                compare_at_fpr=COMPARE_AT_FPR, bootstrap_n=BOOTSTRAP_N))
    save_winner(winner["name"], models[winner["name"]], cands[winner["name"]]["th"], report)
    log("\nNEXT: delete models/lime_background.npz (it holds old feature values), restart the API, run the "
        "loader self-test and re-tune TRUST_FLIP_PROBA / RULE_* if the probability scale changed.")


if __name__ == "__main__":
    main()