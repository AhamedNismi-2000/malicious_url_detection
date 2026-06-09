#!/usr/bin/env python3
"""
train_model.py
--------------
Corrected Random Forest training script.

Dataset: 137,470 verified URLs (50/50 balanced)
  Train : 96,228  x 545 features
  Val   : 20,621  x 545 features
  Test  : 20,621  x 545 features

Pipeline:
  1. Load train/val/test feature NPZ files
  2. Train Random Forest on full train set
  3. Tune threshold on VAL set only
  4. Final evaluation on TEST set only
  5. Save model + threshold + metrics
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

# ---------------- PATHS ----------------
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DIR = os.path.join(BASE_DIR, "features")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
RESULTS_DIR  = os.path.join(BASE_DIR, "results", "metrics")
os.makedirs(RESULTS_DIR, exist_ok=True)

TRAIN_NPZ = os.path.join(FEATURES_DIR, "features_train.npz")
VAL_NPZ   = os.path.join(FEATURES_DIR, "features_val.npz")
TEST_NPZ  = os.path.join(FEATURES_DIR, "features_test.npz")

# ---------------- LOGGING ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(RESULTS_DIR, "training.log"),
            encoding="utf-8",
            mode="w"
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ---------------- RF CONFIG ----------------
# Dataset is 1:1 balanced so class_weight=None
# 96K train samples x 545 features
RF_PARAMS = {
    "n_estimators"     : 300,
    "max_depth"        : 25,      # reduced from 30 for 96K dataset
    "min_samples_split": 10,      # reduced from 20 for 96K dataset
    "min_samples_leaf" : 4,       # reduced from 8 for 96K dataset
    "max_features"     : "sqrt",  # sqrt(545) ~ 23 features per split
    "max_samples"      : 0.8,     # 80% bootstrap sample per tree
    "bootstrap"        : True,
    "class_weight"     : None,    # not needed — dataset already 1:1 balanced
    "n_jobs"           : -1,
    "random_state"     : 42,
    "verbose"          : 1
}


# ================================================================
# DATA LOADING
# ================================================================

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


# ================================================================
# TRAINING
# ================================================================

def train(X_train: sp.csr_matrix,
          y_train: np.ndarray) -> RandomForestClassifier:
    """Train Random Forest on full training set."""
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

    logger.info(f"   Training time: {elapsed/60:.1f} minutes")
    logger.info(f"   Trees built  : {model.n_estimators}")
    return model


# ================================================================
# THRESHOLD TUNING (VAL ONLY)
# ================================================================

def tune_threshold(model: RandomForestClassifier,
                   X_val: sp.csr_matrix,
                   y_val: np.ndarray) -> float:
    """
    Find best classification threshold using VAL set ONLY.
    Optimizes pure F1 score for malicious class (label=1).
    Test set is never touched here.
    """
    logger.info("\nTuning threshold on VAL set...")

    malicious_col = list(model.classes_).index(1)
    y_proba       = model.predict_proba(X_val)[:, malicious_col]

    thresholds = np.arange(0.2, 0.81, 0.01)
    best_th    = 0.5
    best_f1    = 0.0
    results    = []

    for th in thresholds:
        y_pred = (y_proba >= th).astype(int)
        f1     = f1_score(y_val, y_pred, pos_label=1, zero_division=0)
        rec    = recall_score(y_val, y_pred, pos_label=1, zero_division=0)
        pre    = precision_score(y_val, y_pred, pos_label=1, zero_division=0)
        results.append((th, f1, rec, pre))

        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    logger.info(f"   Best threshold : {best_th:.2f}")
    logger.info(f"   Best F1 (val)  : {best_f1:.4f}")

    logger.info("\n   Threshold search (around best):")
    logger.info(f"   {'Threshold':>10} {'F1':>8} {'Recall':>8} {'Precision':>10}")
    for th, f1, rec, pre in results:
        if abs(th - best_th) <= 0.05:
            marker = " <-- best" if abs(th - best_th) < 0.001 else ""
            logger.info(
                f"   {th:>10.2f} {f1:>8.4f} {rec:>8.4f} "
                f"{pre:>10.4f}{marker}"
            )

    return float(best_th)


# ================================================================
# EVALUATION
# ================================================================

def evaluate(model: RandomForestClassifier,
             X: sp.csr_matrix,
             y: np.ndarray,
             threshold: float,
             split_name: str) -> dict:
    """
    Full evaluation on complete split — no subsampling.
    Uses tuned threshold for predictions.
    """
    logger.info(f"\nEvaluating on {split_name} ({len(y):,} samples)...")

    malicious_col = list(model.classes_).index(1)
    y_proba       = model.predict_proba(X)[:, malicious_col]
    y_pred        = (y_proba >= threshold).astype(int)

    acc = accuracy_score(y, y_pred)
    pre = precision_score(y, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y, y_pred, pos_label=1, zero_division=0)
    f1  = f1_score(y, y_pred, pos_label=1, zero_division=0)
    auc = roc_auc_score(y, y_proba)

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    logger.info(f"\n   {split_name} RESULTS (threshold={threshold:.2f})")
    logger.info(f"   {'='*45}")
    logger.info(f"   Accuracy  : {acc:.4f}")
    logger.info(
        f"   Precision : {pre:.4f}  "
        f"(of predicted malicious, how many are)"
    )
    logger.info(
        f"   Recall    : {rec:.4f}  "
        f"(of actual malicious, how many caught)"
    )
    logger.info(f"   F1        : {f1:.4f}")
    logger.info(f"   AUC-ROC   : {auc:.4f}")
    logger.info(f"\n   Confusion Matrix:")
    logger.info(f"   TN={tn:,}  FP={fp:,}")
    logger.info(f"   FN={fn:,}  TP={tp:,}")
    logger.info(
        f"\n   FPR: {fpr:.4f}  ({fp:,} legit URLs wrongly blocked)"
    )
    logger.info(
        f"   FNR: {fnr:.4f}  ({fn:,} malicious URLs missed)"
    )

    if fpr > 0.05:
        logger.warning(f"   WARNING: High FPR ({fpr:.4f})")
    if fnr > 0.10:
        logger.warning(f"   WARNING: High FNR ({fnr:.4f})")

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


# ================================================================
# FEATURE IMPORTANCE
# ================================================================

def log_feature_importance(model: RandomForestClassifier,
                            top_n: int = 20):
    """Log top N most important features."""
    try:
        data          = np.load(TRAIN_NPZ, allow_pickle=True)
        feature_names = list(data["feature_names"])
    except Exception:
        feature_names = [f"feature_{i}"
                         for i in range(model.n_features_in_)]

    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]

    logger.info(f"\n   Top {top_n} most important features:")
    logger.info(f"   {'Rank':>5} {'Feature':>35} {'Importance':>12}")
    for rank, idx in enumerate(indices, 1):
        name = (feature_names[idx]
                if idx < len(feature_names) else f"feature_{idx}")
        logger.info(
            f"   {rank:>5} {name:>35} {importances[idx]:>12.4f}"
        )

    # How many features cover 95% of model decisions
    sorted_imp = np.sort(importances)[::-1]
    cumulative = np.cumsum(sorted_imp)
    n_95       = np.searchsorted(cumulative, 0.95) + 1
    logger.info(
        f"\n   {n_95} features cover 95% of decisions "
        f"(out of {len(importances)})"
    )


# ================================================================
# SAVE
# ================================================================

def save(model: RandomForestClassifier,
         threshold: float,
         val_metrics: dict,
         test_metrics: dict) -> str:
    """Save model, threshold, and full metrics report."""
    ts         = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(MODELS_DIR, f"rf_model_{ts}.joblib")

    # Timestamped model
    joblib.dump(model, model_path, compress=3)
    logger.info(f"\n   Model saved : {model_path}")
    logger.info(
        f"   Model size  : {os.path.getsize(model_path)/1e6:.1f} MB"
    )

    # Stable latest reference for app.py
    latest_path = os.path.join(MODELS_DIR, "rf_model_latest.joblib")
    joblib.dump(model, latest_path, compress=3)
    logger.info(f"   Latest model: {latest_path}")

    # Threshold file for app.py and test_model.py
    threshold_path = os.path.join(MODELS_DIR, "threshold.json")
    with open(threshold_path, "w") as f:
        json.dump({"threshold": threshold}, f, indent=2)
    logger.info(f"   Threshold   : {threshold_path}")

    # Full metrics report
    metrics = {
        "training_date" : ts,
        "model_path"    : model_path,
        "threshold"     : threshold,
        "n_features"    : model.n_features_in_,
        "n_estimators"  : model.n_estimators,
        "rf_params"     : {k: v for k, v in RF_PARAMS.items()
                           if k != "verbose"},
        "val_metrics"   : val_metrics,
        "test_metrics"  : test_metrics,
        "train_samples" : 96228,
        "val_samples"   : 20621,
        "test_samples"  : 20621
    }
    metrics_path = os.path.join(
        RESULTS_DIR, f"rf_metrics_{ts}.json"
    )
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"   Metrics     : {metrics_path}")

    return model_path


# ================================================================
# MAIN
# ================================================================

def main():
    logger.info("RANDOM FOREST TRAINING")
    logger.info("=" * 60)
    logger.info("Dataset: 137,470 verified URLs (50/50 balanced)")
    logger.info("Train : 96,228  x 545 features")
    logger.info("Val   : 20,621  x 545 features")
    logger.info("Test  : 20,621  x 545 features")
    logger.info("=" * 60)

    # Verify feature files exist
    for p in [TRAIN_NPZ, VAL_NPZ, TEST_NPZ]:
        if not os.path.exists(p):
            logger.error(f"File not found: {p}")
            logger.error("Run feature_extraction.py first.")
            return

    # 1. Load data
    logger.info("\nLoading feature files...")
    X_train, y_train = load_npz(TRAIN_NPZ)
    X_val,   y_val   = load_npz(VAL_NPZ)
    X_test,  y_test  = load_npz(TEST_NPZ)

    # 2. Verify consistent feature count
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], (
        f"Feature mismatch: train={X_train.shape[1]} "
        f"val={X_val.shape[1]} test={X_test.shape[1]}"
    )
    logger.info(f"\nFeature count consistent: {X_train.shape[1]}")

    # 3. Train on full training set
    model = train(X_train, y_train)
    del X_train, y_train
    gc.collect()

    # 4. Feature importance
    logger.info("\nFeature importances:")
    log_feature_importance(model, top_n=20)

    # 5. Tune threshold on VAL only
    threshold = tune_threshold(model, X_val, y_val)

    # 6. Evaluate on VAL
    val_metrics = evaluate(
        model, X_val, y_val, threshold, "VAL"
    )
    del X_val, y_val
    gc.collect()

    # 7. Final evaluation on TEST — never seen before this step
    test_metrics = evaluate(
        model, X_test, y_test, threshold, "TEST"
    )
    del X_test, y_test
    gc.collect()

    # 8. Save everything
    logger.info("\nSaving model and metrics...")
    model_path = save(model, threshold, val_metrics, test_metrics)

    # 9. Final summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\n   Threshold  : {threshold:.2f}")
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
        logger.warning(f"\n   WARNING: Val/Test F1 gap = {gap:.4f}")
        logger.warning("   Mild overfitting detected.")
        logger.warning("   Consider reducing max_depth or n_estimators.")
    else:
        logger.info(
            f"\n   Val/Test F1 gap = {gap:.4f} (healthy, no overfitting)"
        )

    logger.info(f"\n   Model : {model_path}")
    logger.info("   Next  : run test_model.py")


if __name__ == "__main__":
    main()