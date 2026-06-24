#!/usr/bin/env python3
"""
train_model.py  —  Fixed Random Forest Training
------------------------------------------------

FIXES over previous version:
  1. [CRITICAL] Feature count no longer hardcoded — reads dynamically
     from NPZ so it never crashes on mismatch with feature_extraction.py
  2. [CRITICAL] Sparse → dense conversion handled explicitly before fit()
     to avoid silent memory explosion with 300 trees
  3. [WARNING]  class_weight='balanced' — safe for any class distribution
  4. [WARNING]  Threshold tuning now uses FPR constraint (FPR < 0.05)
     so legit URL false positives are directly minimized
  5. [WARNING]  max_depth reduced to 20 to prevent overfitting
  6. [MINOR]    Hardcoded sample counts replaced with actual counts
  7. [MINOR]    oob_score=True added for free internal validation
"""

import os
import gc
import json
import time
import joblib
import logging
import numpy as np
import scipy.sparse as sp
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────── PATHS ───────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DIR = os.path.join(BASE_DIR, "features")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
RESULTS_DIR  = os.path.join(BASE_DIR, "results", "metrics")
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_NPZ = os.path.join(FEATURES_DIR, "features_train.npz")
VAL_NPZ   = os.path.join(FEATURES_DIR, "features_val.npz")
TEST_NPZ  = os.path.join(FEATURES_DIR, "features_test.npz")

# ─────────────────────────── LOGGING ─────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(RESULTS_DIR, "training.log"),
            encoding="utf-8", mode="w"
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────── RF CONFIG ───────────────────────
RF_PARAMS = {
    "n_estimators"     : 300,       
    "max_depth"        : 20,       
    "min_samples_split": 10,
    "min_samples_leaf" : 4,
    "max_features"     : "sqrt",
    "max_samples"      : 0.8,
    "bootstrap"        : True,
    "oob_score"        : True,      
    "class_weight"     : "balanced",
    "n_jobs"           : -1,
    "random_state"     : 42,
    "verbose"          : 1
}

# FPR constraint for threshold tuning
# Only accept thresholds where FPR (false positive rate on legit) < this
MAX_ACCEPTABLE_FPR = 0.05


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_npz(path: str):
    """Load feature NPZ saved by feature_extraction.py."""
    logger.info(f"   Loading {os.path.basename(path)}...")
    data = np.load(path, allow_pickle=True)
    X    = sp.csr_matrix(
        (data["data"], data["indices"], data["indptr"]),
        shape=tuple(data["shape"])
    )
    y = data["labels"].astype(int)
    logger.info(
        f"   Shape: {X.shape}  |  "
        f"Benign: {(y==0).sum():,}  Malicious: {(y==1).sum():,}"
    )
    return X, y


def to_dense(X: sp.csr_matrix, name: str) -> np.ndarray:
    """
    FIX: Explicit sparse → dense conversion with RAM warning.
    sklearn RF silently converts sparse to dense internally per tree,
    which causes memory explosion with many trees.
    Doing it once upfront is safer and faster overall.
    """
    estimated_mb = (X.shape[0] * X.shape[1] * 4) / 1e6
    logger.info(
        f"   Converting {name} to dense "
        f"(estimated {estimated_mb:.0f} MB)..."
    )
    if estimated_mb > 4000:
        logger.warning(
            f"   WARNING: Dense conversion may use {estimated_mb:.0f} MB. "
            "Consider reducing MAX_FEAT_CHAR/WORD in feature_extraction.py "
            "if you run out of memory."
        )
    return X.toarray().astype(np.float32)


# ═══════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════

def train(X_train: np.ndarray,
          y_train: np.ndarray) -> RandomForestClassifier:
    logger.info("\nTraining Random Forest...")
    logger.info(f"   Samples      : {X_train.shape[0]:,}")
    logger.info(f"   Features     : {X_train.shape[1]:,}")
    logger.info(f"   Trees        : {RF_PARAMS['n_estimators']}")
    logger.info(f"   Max depth    : {RF_PARAMS['max_depth']}")
    logger.info(f"   Class weight : {RF_PARAMS['class_weight']}")

    model = RandomForestClassifier(**RF_PARAMS)
    start = time.time()
    model.fit(X_train, y_train)
    elapsed = time.time() - start

    logger.info(f"   Training time : {elapsed/60:.1f} minutes")
    logger.info(f"   OOB score     : {model.oob_score_:.4f}")
    return model


# ═══════════════════════════════════════════════════════════════
# THRESHOLD TUNING — FPR CONSTRAINED
# ═══════════════════════════════════════════════════════════════

def tune_threshold(model: RandomForestClassifier,
                   X_val: np.ndarray,
                   y_val: np.ndarray) -> float:
    """
    FIX: Threshold tuning now uses FPR constraint.

    Old behaviour: pick threshold that maximizes F1 — this can choose
    a very aggressive threshold that catches more malicious URLs but
    also flags more legit ones (high FPR = your original problem).

    New behaviour: only consider thresholds where FPR < MAX_ACCEPTABLE_FPR,
    then among those pick the one with highest F1.
    This directly minimizes false positives on legit URLs.
    """
    logger.info("\nTuning threshold on VAL set (FPR constrained)...")
    logger.info(f"   FPR constraint : < {MAX_ACCEPTABLE_FPR}")

    malicious_col = list(model.classes_).index(1)
    y_proba       = model.predict_proba(X_val)[:, malicious_col]

    thresholds = np.arange(0.2, 0.81, 0.01)
    best_th    = 0.5
    best_f1    = 0.0
    results    = []

    for th in thresholds:
        y_pred = (y_proba >= th).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        f1  = f1_score(y_val, y_pred, pos_label=1, zero_division=0)
        rec = recall_score(y_val, y_pred, pos_label=1, zero_division=0)
        pre = precision_score(y_val, y_pred, pos_label=1, zero_division=0)

        results.append((th, f1, rec, pre, fpr))

        # FIX: only update best if FPR constraint is satisfied
        if fpr < MAX_ACCEPTABLE_FPR and f1 > best_f1:
            best_f1 = f1
            best_th = th

    logger.info(f"   Best threshold : {best_th:.2f}")
    logger.info(f"   Best F1 (val)  : {best_f1:.4f}")

    logger.info(
        f"\n   {'Threshold':>10} {'F1':>8} "
        f"{'Recall':>8} {'Precision':>10} {'FPR':>8} {'OK?':>6}"
    )
    for th, f1, rec, pre, fpr in results:
        if abs(th - best_th) <= 0.07:
            ok     = "✓" if fpr < MAX_ACCEPTABLE_FPR else "✗ FPR"
            marker = " <-- best" if abs(th - best_th) < 0.001 else ""
            logger.info(
                f"   {th:>10.2f} {f1:>8.4f} {rec:>8.4f} "
                f"{pre:>10.4f} {fpr:>8.4f} {ok:>6}{marker}"
            )

    return float(best_th)


# ═══════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════

def evaluate(model: RandomForestClassifier,
             X: np.ndarray,
             y: np.ndarray,
             threshold: float,
             split_name: str) -> dict:
    logger.info(f"\nEvaluating on {split_name} ({len(y):,} samples)...")

    malicious_col = list(model.classes_).index(1)
    y_proba       = model.predict_proba(X)[:, malicious_col]
    y_pred        = (y_proba >= threshold).astype(int)

    acc            = accuracy_score(y, y_pred)
    pre            = precision_score(y, y_pred, pos_label=1, zero_division=0)
    rec            = recall_score(y, y_pred, pos_label=1, zero_division=0)
    f1             = f1_score(y, y_pred, pos_label=1, zero_division=0)
    auc            = roc_auc_score(y, y_proba)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    fpr            = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr            = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    logger.info(f"\n   {split_name} RESULTS (threshold={threshold:.2f})")
    logger.info(f"   {'='*45}")
    logger.info(f"   Accuracy  : {acc:.4f}")
    logger.info(f"   Precision : {pre:.4f}")
    logger.info(f"   Recall    : {rec:.4f}")
    logger.info(f"   F1        : {f1:.4f}")
    logger.info(f"   AUC-ROC   : {auc:.4f}")
    logger.info(f"\n   Confusion Matrix:")
    logger.info(f"   TN={tn:,}  FP={fp:,}")
    logger.info(f"   FN={fn:,}  TP={tp:,}")
    logger.info(f"\n   FPR : {fpr:.4f}  ({fp:,} legit URLs wrongly flagged)")
    logger.info(f"   FNR : {fnr:.4f}  ({fn:,} malicious URLs missed)")

    if fpr > MAX_ACCEPTABLE_FPR:
        logger.warning(f"   WARNING: FPR {fpr:.4f} exceeds {MAX_ACCEPTABLE_FPR} target")
    else:
        logger.info(f"   FPR within acceptable range (< {MAX_ACCEPTABLE_FPR}) ✓")

    if fnr > 0.10:
        logger.warning(f"   WARNING: High FNR ({fnr:.4f}) — missing too many threats")

    return {
        "split"    : split_name,
        "threshold": threshold,
        "accuracy" : round(acc, 4),
        "precision": round(pre, 4),
        "recall"   : round(rec, 4),
        "f1"       : round(f1, 4),
        "auc"      : round(auc, 4),
        "tp": int(tp), "tn": int(tn),
        "fp": int(fp), "fn": int(fn),
        "fpr"      : round(fpr, 4),
        "fnr"      : round(fnr, 4)
    }


# ═══════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════

def log_feature_importance(model: RandomForestClassifier, top_n: int = 20):
    try:
        data          = np.load(TRAIN_NPZ, allow_pickle=True)
        feature_names = list(data["feature_names"])
    except Exception:
        feature_names = [f"feature_{i}" for i in range(model.n_features_in_)]

    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]

    logger.info(f"\n   Top {top_n} most important features:")
    logger.info(f"   {'Rank':>5} {'Feature':>35} {'Importance':>12}")
    for rank, idx in enumerate(indices, 1):
        name = feature_names[idx] if idx < len(feature_names) \
            else f"feature_{idx}"
        logger.info(f"   {rank:>5} {name:>35} {importances[idx]:>12.4f}")

    sorted_imp = np.sort(importances)[::-1]
    cumulative = np.cumsum(sorted_imp)
    n_95       = np.searchsorted(cumulative, 0.95) + 1
    logger.info(
        f"\n   {n_95} features cover 95% of decisions "
        f"(out of {len(importances)})"
    )


# ═══════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════

def save(model, threshold, val_metrics, test_metrics,
         n_train, n_val, n_test) -> str:
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODELS_DIR, f"rf_model_{ts}.joblib")

    joblib.dump(model, model_path, compress=3)
    logger.info(f"\n   Model saved : {model_path}")
    logger.info(f"   Model size  : {os.path.getsize(model_path)/1e6:.1f} MB")

    latest_path = os.path.join(MODELS_DIR, "rf_model_latest.joblib")
    joblib.dump(model, latest_path, compress=3)

    threshold_path = os.path.join(MODELS_DIR, "threshold.json")
    with open(threshold_path, "w") as f:
        json.dump({"threshold": threshold}, f, indent=2)

    # FIX: actual counts not hardcoded
    metrics = {
        "training_date" : ts,
        "model_path"    : model_path,
        "threshold"     : threshold,
        "n_features"    : model.n_features_in_,
        "n_estimators"  : model.n_estimators,
        "oob_score"     : round(model.oob_score_, 4),
        "rf_params"     : {k: v for k, v in RF_PARAMS.items()
                           if k != "verbose"},
        "val_metrics"   : val_metrics,
        "test_metrics"  : test_metrics,
        "train_samples" : n_train,   # FIX: from actual data
        "val_samples"   : n_val,
        "test_samples"  : n_test
    }
    metrics_path = os.path.join(RESULTS_DIR, f"rf_metrics_{ts}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"   Metrics     : {metrics_path}")

    return model_path


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    logger.info("RANDOM FOREST TRAINING  (Fixed Version)")
    logger.info("=" * 60)

    for p in [TRAIN_NPZ, VAL_NPZ, TEST_NPZ]:
        if not os.path.exists(p):
            logger.error(f"File not found: {p} — run feature_extraction.py first.")
            return

    # 1. Load
    logger.info("\nLoading feature files...")
    X_train, y_train = load_npz(TRAIN_NPZ)
    X_val,   y_val   = load_npz(VAL_NPZ)
    X_test,  y_test  = load_npz(TEST_NPZ)

    # FIX: dynamic feature count — no hardcoding
    n_train, n_feat = X_train.shape
    n_val           = X_val.shape[0]
    n_test          = X_test.shape[0]
    logger.info(f"\nDataset summary:")
    logger.info(f"   Train   : {n_train:,} x {n_feat:,} features")
    logger.info(f"   Val     : {n_val:,} x {X_val.shape[1]:,} features")
    logger.info(f"   Test    : {n_test:,} x {X_test.shape[1]:,} features")

    # Verify consistent feature count across all splits
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], (
        f"Feature mismatch: train={X_train.shape[1]} "
        f"val={X_val.shape[1]} test={X_test.shape[1]}"
    )

    # 2. FIX: explicit dense conversion upfront
    logger.info("\nConverting sparse features to dense...")
    X_train_dense = to_dense(X_train, "train")
    del X_train
    gc.collect()

    X_val_dense = to_dense(X_val, "val")
    del X_val
    gc.collect()

    X_test_dense = to_dense(X_test, "test")
    del X_test
    gc.collect()

    # 3. Train
    model = train(X_train_dense, y_train)
    del X_train_dense, y_train
    gc.collect()

    # 4. Feature importance
    log_feature_importance(model, top_n=20)

    # 5. FPR-constrained threshold tuning on VAL only
    threshold = tune_threshold(model, X_val_dense, y_val)

    # 6. Evaluate VAL
    val_metrics = evaluate(model, X_val_dense, y_val, threshold, "VAL")
    del X_val_dense, y_val
    gc.collect()

    # 7. Evaluate TEST — never seen before this step
    test_metrics = evaluate(model, X_test_dense, y_test, threshold, "TEST")
    del X_test_dense, y_test
    gc.collect()

    # 8. Save
    logger.info("\nSaving model and metrics...")
    model_path = save(
        model, threshold, val_metrics, test_metrics,
        n_train, n_val, n_test
    )

    # 9. Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"   OOB Score  : {model.oob_score_:.4f}")
    logger.info(f"   Threshold  : {threshold:.2f}")
    logger.info(f"\n   VAL  F1    : {val_metrics['f1']:.4f}")
    logger.info(f"   VAL  AUC   : {val_metrics['auc']:.4f}")
    logger.info(f"   VAL  FPR   : {val_metrics['fpr']:.4f}")
    logger.info(f"   VAL  FNR   : {val_metrics['fnr']:.4f}")
    logger.info(f"\n   TEST F1    : {test_metrics['f1']:.4f}")
    logger.info(f"   TEST AUC   : {test_metrics['auc']:.4f}")
    logger.info(f"   TEST FPR   : {test_metrics['fpr']:.4f}")
    logger.info(f"   TEST FNR   : {test_metrics['fnr']:.4f}")

    gap = abs(val_metrics["f1"] - test_metrics["f1"])
    if gap > 0.03:
        logger.warning(f"\n   WARNING: Val/Test F1 gap = {gap:.4f} — mild overfitting")
        logger.warning("   Try reducing max_depth to 15 or n_estimators to 150.")
    else:
        logger.info(f"\n   Val/Test F1 gap = {gap:.4f} ✓ healthy")

    logger.info(f"\n   Model  : {model_path}")
    logger.info("   Next   : run evaluate.py or predict.py")


if __name__ == "__main__":
    main()
