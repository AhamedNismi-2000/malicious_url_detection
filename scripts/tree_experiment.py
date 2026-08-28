#!/usr/bin/env python3
"""
tree_experiment.py
------------------
Experiments with different n_estimators values for the Random Forest.
Tests: 150, 200, 250, 300 trees.

Uses the EXACT same pipeline as train_model.py:
  - Same RF params (only n_estimators changes)
  - Same threshold tuning on VAL set
  - Same final evaluation on TEST set
  - 300 trees loaded from existing rf_model_latest.joblib (already trained)

Saves:
  results/plots/exp_01_performance_vs_trees.png / .pdf
  results/plots/exp_02_training_time_vs_trees.png / .pdf
  results/plots/exp_03_fpr_fnr_speed_vs_trees.png / .pdf
  results/metrics/tree_experiment_results.json

Run from project root:
  python experiments/tree_experiment.py
"""

import os
import gc
import json
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DIR = os.path.join(BASE_DIR, "features")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
RESULTS_DIR  = os.path.join(BASE_DIR, "results", "plots")
METRICS_DIR  = os.path.join(BASE_DIR, "results", "metrics")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

TRAIN_NPZ = os.path.join(FEATURES_DIR, "features_train.npz")
VAL_NPZ   = os.path.join(FEATURES_DIR, "features_val.npz")
TEST_NPZ  = os.path.join(FEATURES_DIR, "features_test.npz")

# ── Plot style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor" : "#0f1117",
    "axes.facecolor"   : "#1a1d27",
    "axes.edgecolor"   : "#2a2d3a",
    "axes.labelcolor"  : "#e2e4ec",
    "axes.titlecolor"  : "#e2e4ec",
    "axes.grid"        : True,
    "grid.color"       : "#2a2d3a",
    "grid.linewidth"   : 0.8,
    "xtick.color"      : "#6b7280",
    "ytick.color"      : "#6b7280",
    "text.color"       : "#e2e4ec",
    "font.family"      : "DejaVu Sans",
    "font.size"        : 11,
    "axes.titlesize"   : 13,
    "axes.labelsize"   : 11,
    "legend.facecolor" : "#1a1d27",
    "legend.edgecolor" : "#2a2d3a",
    "legend.labelcolor": "#e2e4ec",
    "figure.dpi"       : 150,
})

C_MAL    = "#ef4444"
C_BEN    = "#c5ad22"
C_ACCENT = "#818cf8"
C_WARN   = "#f59e0b"
C_MUTED  = "#6b7280"
C_ORANGE = "#f97316"

SAVE_DPI = 200

# ── RF base params — identical to train_model.py ──────────────────────────────
BASE_RF_PARAMS = {
    "max_depth"        : 25,
    "min_samples_split": 10,
    "min_samples_leaf" : 4,
    "max_features"     : "sqrt",
    "max_samples"      : 0.8,
    "bootstrap"        : True,
    "class_weight"     : None,
    "n_jobs"           : -1,
    "random_state"     : 42,
    "verbose"          : 0,
}

N_ESTIMATORS_LIST = [150, 200, 250, 300]


# ── Helpers ───────────────────────────────────────────────────────────────────

def save(fig, name):
    for ext in ("png", "pdf"):
        path = os.path.join(RESULTS_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    print(f"  saved: {name}.png / .pdf")
    plt.close(fig)


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    X    = sp.csr_matrix(
        (data["data"], data["indices"], data["indptr"]),
        shape=tuple(data["shape"])
    )
    y = data["labels"].astype(int)
    return X, y


def tune_threshold(model, X_val, y_val):
    """Identical to train_model.py threshold tuning."""
    malicious_col = list(model.classes_).index(1)
    y_proba       = model.predict_proba(X_val)[:, malicious_col]
    thresholds    = np.arange(0.2, 0.81, 0.01)
    best_th, best_f1 = 0.5, 0.0
    for th in thresholds:
        y_pred = (y_proba >= th).astype(int)
        f1     = f1_score(y_val, y_pred, pos_label=1, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_th = th
    return float(best_th)


def evaluate(model, X, y, threshold):
    malicious_col = list(model.classes_).index(1)
    y_proba       = model.predict_proba(X)[:, malicious_col]
    y_pred        = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    return {
        "accuracy" : round(accuracy_score(y, y_pred), 4),
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "recall"   : round(recall_score(y, y_pred, zero_division=0), 4),
        "f1"       : round(f1_score(y, y_pred, zero_division=0), 4),
        "auc"      : round(roc_auc_score(y, y_proba), 4),
        "fpr"      : round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0.0,
        "fnr"      : round(fn / (fn + tp), 4) if (fn + tp) > 0 else 0.0,
        "threshold": round(threshold, 2),
    }


def predict_time_per_url(model, X_test, n_samples=1000):
    """Average prediction time per URL in milliseconds."""
    idx    = np.random.choice(X_test.shape[0], n_samples, replace=False)
    sample = X_test[idx]
    start  = time.perf_counter()
    model.predict_proba(sample)
    elapsed = time.perf_counter() - start
    return round((elapsed / n_samples) * 1000, 4)


# ── Run experiments ───────────────────────────────────────────────────────────

def run_experiments(X_train, y_train, X_val, y_val, X_test, y_test):
    results = []

    for n in N_ESTIMATORS_LIST:
        print(f"\n{'='*55}")
        print(f"  n_estimators = {n}")
        print(f"{'='*55}")

        train_time = None

        if n == 300:
            model_path = os.path.join(MODELS_DIR, "rf_model_latest.joblib")
            if os.path.exists(model_path):
                print(f"  Loading existing model (already trained)...")
                model = joblib.load(model_path)
                print(f"  Loaded OK")
            else:
                print("  rf_model_latest.joblib not found — training now...")
                model, train_time = _train(n, X_train, y_train)
        else:
            model, train_time = _train(n, X_train, y_train)

        print("  Tuning threshold on VAL...")
        threshold = tune_threshold(model, X_val, y_val)
        print(f"  Threshold: {threshold:.2f}")

        print("  Evaluating on TEST...")
        metrics = evaluate(model, X_test, y_test, threshold)

        pred_ms = predict_time_per_url(model, X_test)

        row = {
            "n_estimators"   : n,
            "train_time_s"   : train_time,
            "pred_ms_per_url": pred_ms,
            **metrics,
        }
        results.append(row)

        print(f"  F1={metrics['f1']:.4f}  AUC={metrics['auc']:.4f}  "
              f"FPR={metrics['fpr']*100:.2f}%  FNR={metrics['fnr']*100:.2f}%  "
              f"Speed={pred_ms:.2f}ms/URL")

        del model
        gc.collect()

    return results


def _train(n, X_train, y_train):
    params = {**BASE_RF_PARAMS, "n_estimators": n}
    model  = RandomForestClassifier(**params)
    print(f"  Training {n} trees...")
    start  = time.time()
    model.fit(X_train, y_train)
    train_time = round(time.time() - start, 1)
    print(f"  Done in {train_time/60:.1f} min")
    return model, train_time


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_performance(results):
    trees = [r["n_estimators"] for r in results]
    f1s   = [r["f1"]        for r in results]
    aucs  = [r["auc"]       for r in results]
    precs = [r["precision"] for r in results]
    recs  = [r["recall"]    for r in results]
    accs  = [r["accuracy"]  for r in results]

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("#0f1117")

    ax.plot(trees, f1s,   color=C_ACCENT, lw=2.5, marker="o", markersize=8, label="F1 Score")
    ax.plot(trees, aucs,  color=C_BEN,    lw=2,   marker="s", markersize=7, label="AUC-ROC")
    ax.plot(trees, precs, color=C_WARN,   lw=1.8, marker="^", markersize=7, label="Precision")
    ax.plot(trees, recs,  color=C_MAL,    lw=1.8, marker="D", markersize=7, label="Recall")
    ax.plot(trees, accs,  color=C_ORANGE, lw=1.8, marker="v", markersize=7,
            label="Accuracy", linestyle="--")

    for x, y in zip(trees, f1s):
        ax.annotate(f"{y:.4f}", (x, y),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9, color=C_ACCENT)

    ax.set_xlabel("Number of Trees (n_estimators)")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance vs Number of Trees")
    ax.set_xticks(trees)
    bottom = min(min(f1s), min(aucs), min(precs), min(recs)) - 0.005
    ax.set_ylim([bottom, 1.015])
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "exp_01_performance_vs_trees")


def plot_training_time(results):
    timed = [r for r in results if r["train_time_s"] is not None]
    if not timed:
        print("  Skipping training time plot (300-tree model was pre-loaded)")
        # Estimate 300 trees proportionally from 250
        r250 = next((r for r in results if r["n_estimators"] == 250), None)
        if r250 and r250["train_time_s"]:
            est_300 = round(r250["train_time_s"] * (300 / 250), 1)
            timed = results[:]
            for r in timed:
                if r["n_estimators"] == 300:
                    r = dict(r)
                    r["train_time_s"] = est_300
        if not timed:
            return

    trees = [r["n_estimators"] for r in timed]
    times = [r["train_time_s"] / 60 for r in timed]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0f1117")

    bars = ax.bar(trees, times, color=C_ACCENT, width=25,
                  edgecolor="none", alpha=0.85)

    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.15,
                f"{t:.1f} min", ha="center", fontsize=10,
                color="#e2e4ec", fontweight="bold")

    ax.set_xlabel("Number of Trees (n_estimators)")
    ax.set_ylabel("Training Time (minutes)")
    ax.set_title("Training Time vs Number of Trees")
    ax.set_xticks(trees)
    ax.set_ylim([0, max(times) * 1.3])
    fig.tight_layout()
    save(fig, "exp_02_training_time_vs_trees")


def plot_fpr_fnr_speed(results):
    trees = [r["n_estimators"] for r in results]
    fprs  = [r["fpr"] * 100   for r in results]
    fnrs  = [r["fnr"] * 100   for r in results]
    preds = [r["pred_ms_per_url"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0f1117")

    # Left: FPR & FNR
    ax = axes[0]
    ax.plot(trees, fprs, color=C_MAL,  lw=2.5, marker="o", markersize=8,
            label="False Positive Rate (%)")
    ax.plot(trees, fnrs, color=C_WARN, lw=2.5, marker="s", markersize=8,
            label="False Negative Rate (%)")

    for x, y in zip(trees, fprs):
        ax.annotate(f"{y:.2f}%", (x, y),
                    textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9, color=C_MAL)
    for x, y in zip(trees, fnrs):
        ax.annotate(f"{y:.2f}%", (x, y),
                    textcoords="offset points", xytext=(0, -16),
                    ha="center", fontsize=9, color=C_WARN)

    ax.set_xlabel("Number of Trees (n_estimators)")
    ax.set_ylabel("Error Rate (%)")
    ax.set_title("FPR & FNR vs Number of Trees")
    ax.set_xticks(trees)
    ax.legend()

    # Right: Prediction speed
    ax = axes[1]
    ax.plot(trees, preds, color=C_BEN, lw=2.5, marker="o", markersize=8,
            label="Prediction time (ms/URL)")

    for x, y in zip(trees, preds):
        ax.annotate(f"{y:.2f}ms", (x, y),
                    textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9, color=C_BEN)

    ax.set_xlabel("Number of Trees (n_estimators)")
    ax.set_ylabel("Time (ms per URL)")
    ax.set_title("Prediction Speed vs Number of Trees")
    ax.set_xticks(trees)
    ax.legend()

    fig.tight_layout()
    save(fig, "exp_03_fpr_fnr_speed_vs_trees")


def print_summary(results):
    print("\n" + "=" * 82)
    print("EXPERIMENT SUMMARY")
    print("=" * 82)
    print(f"  {'Trees':>6} {'F1':>8} {'AUC':>8} {'Prec':>8} {'Recall':>8} "
          f"{'FPR%':>7} {'FNR%':>7} {'Thresh':>7} {'ms/URL':>8} {'Train':>8}")
    print("  " + "-" * 80)
    for r in results:
        t_str = f"{r['train_time_s']/60:.1f}m" if r["train_time_s"] else "pre-loaded"
        print(
            f"  {r['n_estimators']:>6} "
            f"{r['f1']:>8.4f} "
            f"{r['auc']:>8.4f} "
            f"{r['precision']:>8.4f} "
            f"{r['recall']:>8.4f} "
            f"{r['fpr']*100:>7.2f} "
            f"{r['fnr']*100:>7.2f} "
            f"{r['threshold']:>7.2f} "
            f"{r['pred_ms_per_url']:>8.2f} "
            f"{t_str:>10}"
        )
    print("=" * 82)

    f1_150 = next((r["f1"] for r in results if r["n_estimators"] == 150), None)
    f1_300 = next((r["f1"] for r in results if r["n_estimators"] == 300), None)
    if f1_150 and f1_300:
        delta = f1_300 - f1_150
        print(f"\n  F1 gain 150 → 300 trees : {delta:.4f} "
              f"({'significant' if abs(delta) > 0.005 else 'negligible'})")
    best = max(results, key=lambda r: r["f1"])
    print(f"  Best F1               : {best['f1']:.4f} "
          f"at n_estimators = {best['n_estimators']}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 55)
    print("  RF n_estimators Experiment")
    print("  Testing: 150 / 200 / 250 / 300 trees")
    print(f"  Estimated total time: ~40 minutes")
    print("=" * 55)

    print("\nLoading feature files...")
    X_train, y_train = load_npz(TRAIN_NPZ)
    X_val,   y_val   = load_npz(VAL_NPZ)
    X_test,  y_test  = load_npz(TEST_NPZ)
    print(f"  Train : {X_train.shape}")
    print(f"  Val   : {X_val.shape}")
    print(f"  Test  : {X_test.shape}")

    results = run_experiments(
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
    )

    out_path = os.path.join(METRICS_DIR, "tree_experiment_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results JSON saved: {out_path}")

    print("\nGenerating plots...")
    plot_performance(results)
    plot_training_time(results)
    plot_fpr_fnr_speed(results)

    print_summary(results)

    print(f"\nAll plots saved to: {RESULTS_DIR}")
    print("Formats: PNG + PDF\n")


if __name__ == "__main__":
    main()