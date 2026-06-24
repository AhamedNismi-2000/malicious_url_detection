#!/usr/bin/env python3
"""
evaluate_model.py
-----------------
Comprehensive model evaluation script for the malicious URL detector.

Generates the following plots (saved as both PNG and PDF):
  1. Confusion Matrix
  2. ROC Curve + AUC
  3. Precision-Recall Curve + AUC
  4. Feature Importance (top 20)
  5. Confidence Score Distribution
  6. Metrics Summary Bar Chart
  7. Threshold Analysis (F1, Precision, Recall vs Threshold)

All plots saved to:
  results/plots/

Run from project root:
  python scripts/evaluate_model.py
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, accuracy_score,
    matthews_corrcoef, balanced_accuracy_score,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
FEATURES_DIR= os.path.join(BASE_DIR, "features")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "plots")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor"  : "#0f1117",
    "axes.facecolor"    : "#1a1d27",
    "axes.edgecolor"    : "#2a2d3a",
    "axes.labelcolor"   : "#e2e4ec",
    "axes.titlecolor"   : "#e2e4ec",
    "axes.grid"         : True,
    "grid.color"        : "#2a2d3a",
    "grid.linewidth"    : 0.8,
    "xtick.color"       : "#6b7280",
    "ytick.color"       : "#6b7280",
    "text.color"        : "#e2e4ec",
    "font.family"       : "DejaVu Sans",
    "font.size"         : 11,
    "axes.titlesize"    : 13,
    "axes.labelsize"    : 11,
    "legend.facecolor"  : "#1a1d27",
    "legend.edgecolor"  : "#2a2d3a",
    "legend.labelcolor" : "#e2e4ec",
    "figure.dpi"        : 150,
})

C_MAL    = "#ef4444"   # red   — malicious
C_BEN    = "#22c55e"   # green — benign
C_ACCENT = "#818cf8"   # indigo
C_WARN   = "#f59e0b"   # amber
C_MUTED  = "#6b7280"   # gray

SAVE_DPI = 200


def save(fig, name):
    """Save figure as PNG only."""
    path = os.path.join(RESULTS_DIR, f"{name}.png")
    fig.savefig(
        path,
        dpi=SAVE_DPI,
        bbox_inches="tight",
        facecolor=fig.get_facecolor()
    )
    print(f"  ✓  {name}.png")
    plt.close(fig)

# ── Load artefacts ────────────────────────────────────────────────────────────

def load_artefacts():
    print("Loading model artefacts...")
    model    = joblib.load(os.path.join(MODELS_DIR, "rf_model_latest.joblib"))
    with open(os.path.join(MODELS_DIR, "threshold.json")) as f:
        threshold = float(json.load(f).get("threshold", 0.44))
    print(f"  threshold = {threshold:.2f}")
    return model, threshold


def load_test_features():
    print("Loading test features...")
    path = os.path.join(FEATURES_DIR, "features_test.npz")
    data = np.load(path, allow_pickle=True)

    X = sp.csr_matrix(
        (data["data"], data["indices"], data["indptr"]),
        shape=tuple(data["shape"])
    ).toarray().astype(np.float32)

    y = data["labels"].astype(int)

    feature_names = (
        list(data["feature_names"])
        if "feature_names" in data
        else [f"f{i}" for i in range(X.shape[1])]
    )

    print(f"  Test set: {X.shape[0]:,} samples × {X.shape[1]:,} features")
    print(f"  Malicious: {y.sum():,}  |  Benign: {(y==0).sum():,}")
    return X, y, feature_names


# ── 1. Confusion Matrix ───────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0f1117")

    cmap = LinearSegmentedColormap.from_list(
        "custom", ["#1a1d27", C_ACCENT], N=256
    )
    im = ax.imshow(cm, cmap=cmap, aspect="auto")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign\n(Predicted)", "Malicious\n(Predicted)"],
                       color="#e2e4ec")
    ax.set_yticklabels(["Benign\n(Actual)", "Malicious\n(Actual)"],
                       color="#e2e4ec")

    labels = [
        [f"TN\n{tn:,}", f"FP\n{fp:,}"],
        [f"FN\n{fn:,}", f"TP\n{tp:,}"],
    ]
    colors = [
        [C_BEN, C_MAL],
        [C_WARN, C_BEN],
    ]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, labels[i][j], ha="center", va="center",
                    fontsize=14, fontweight="bold", color=colors[i][j])

    ax.set_title("Confusion Matrix", pad=14)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save(fig, "01_confusion_matrix")
    return tn, fp, fn, tp


# ── 2. ROC Curve ─────────────────────────────────────────────────────────────

def plot_roc(y_true, y_prob, threshold):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    # Find point closest to our threshold
    idx = np.argmin(np.abs(thresholds - threshold))

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0f1117")

    ax.plot(fpr, tpr, color=C_ACCENT, lw=2.5,
            label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color=C_MUTED, lw=1.2,
            linestyle="--", label="Random Classifier")
    ax.scatter(fpr[idx], tpr[idx], color=C_WARN, s=120, zorder=5,
               label=f"Threshold = {threshold:.2f}")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    fig.tight_layout()
    save(fig, "02_roc_curve")
    return roc_auc


# ── 3. Precision-Recall Curve ─────────────────────────────────────────────────

def plot_pr_curve(y_true, y_prob, threshold):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    # Find point closest to our threshold
    idx = np.argmin(np.abs(thresholds - threshold))

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("#0f1117")

    ax.plot(recall, precision, color=C_MAL, lw=2.5,
            label=f"PR Curve (AP = {ap:.4f})")
    ax.axhline(y=y_true.mean(), color=C_MUTED, lw=1.2,
               linestyle="--", label=f"Baseline (prevalence = {y_true.mean():.2f})")
    ax.scatter(recall[idx], precision[idx], color=C_WARN, s=120, zorder=5,
               label=f"Threshold = {threshold:.2f}")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="upper right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    fig.tight_layout()
    save(fig, "03_precision_recall_curve")
    return ap


# ── 4. Feature Importance ─────────────────────────────────────────────────────

def plot_feature_importance(model, feature_names, top_n=20):
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]
    top_names   = [str(feature_names[i]) for i in indices]
    top_vals    = importances[indices]

    # Shorten n-gram feature names for display
    display_names = []
    for n in top_names:
        if n.startswith("char_") or n.startswith("word_"):
            display_names.append(n)
        else:
            display_names.append(n.replace("_", " ").title())

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor("#0f1117")

    colors = [C_ACCENT if not (n.startswith("char_") or n.startswith("word_"))
              else C_MUTED for n in top_names]

    bars = ax.barh(range(top_n), top_vals[::-1], color=colors[::-1],
                   edgecolor="none", height=0.7)

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(display_names[::-1], fontsize=9)
    ax.set_xlabel("Feature Importance (Gini)")
    ax.set_title(f"Top {top_n} Feature Importances")

    # Value labels
    for bar, val in zip(bars, top_vals[::-1]):
        ax.text(bar.get_width() + 0.0002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8, color=C_MUTED)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C_ACCENT, label="Heuristic features"),
        Patch(facecolor=C_MUTED,  label="N-gram TF-IDF features"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "04_feature_importance")


# ── 5. Confidence Distribution ────────────────────────────────────────────────

def plot_confidence_distribution(y_true, y_prob, threshold):
    benign_probs    = y_prob[y_true == 0]
    malicious_probs = y_prob[y_true == 1]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0f1117")

    bins = np.linspace(0, 1, 50)
    ax.hist(benign_probs,    bins=bins, alpha=0.7, color=C_BEN,
            label="Benign",    edgecolor="none")
    ax.hist(malicious_probs, bins=bins, alpha=0.7, color=C_MAL,
            label="Malicious", edgecolor="none")
    ax.axvline(x=threshold, color=C_WARN, lw=2, linestyle="--",
               label=f"Threshold = {threshold:.2f}")

    ax.set_xlabel("Predicted Probability (Malicious)")
    ax.set_ylabel("Number of URLs")
    ax.set_title("Confidence Score Distribution")
    ax.legend()
    fig.tight_layout()
    save(fig, "05_confidence_distribution")


# ── 6. Metrics Summary Bar Chart ──────────────────────────────────────────────

def plot_metrics_summary(y_true, y_pred, y_prob, roc_auc, ap):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    metrics = {
        "Accuracy"          : accuracy_score(y_true, y_pred),
        "Precision"         : precision_score(y_true, y_pred),
        "Recall"            : recall_score(y_true, y_pred),
        "F1 Score"          : f1_score(y_true, y_pred),
        "ROC-AUC"           : roc_auc,
        "Avg Precision"     : ap,
        "Balanced Accuracy" : balanced_accuracy_score(y_true, y_pred),
        "MCC"               : (matthews_corrcoef(y_true, y_pred) + 1) / 2,  # normalised 0-1
        "FPR"               : fp / (fp + tn),
        "FNR"               : fn / (fn + tp),
    }

    names  = list(metrics.keys())
    values = list(metrics.values())

    colors = []
    for v in values:
        if v >= 0.95:   colors.append(C_BEN)
        elif v >= 0.85: colors.append(C_ACCENT)
        elif v >= 0.70: colors.append(C_WARN)
        else:           colors.append(C_MAL)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0f1117")

    bars = ax.bar(names, values, color=colors, edgecolor="none", width=0.6)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Metrics")
    ax.tick_params(axis="x", rotation=30)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{val:.4f}", ha="center", va="bottom",
                fontsize=9, color="#e2e4ec")

    # Note for MCC
    ax.text(0.99, 0.02, "* MCC normalised to 0–1 for display",
            transform=ax.transAxes, ha="right", fontsize=8, color=C_MUTED)

    fig.tight_layout()
    save(fig, "06_metrics_summary")
    return metrics


# ── 7. Threshold Analysis ────────────────────────────────────────────────────

def plot_threshold_analysis(y_true, y_prob, best_threshold):
    thresholds = np.linspace(0.01, 0.99, 200)
    f1s, precs, recs, fprs, fnrs = [], [], [], [], []

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        f1s.append(f1_score(y_true, preds, zero_division=0))
        precs.append(precision_score(y_true, preds, zero_division=0))
        recs.append(recall_score(y_true, preds, zero_division=0))
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
        fnrs.append(fn / (fn + tp) if (fn + tp) > 0 else 0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0f1117")

    # Left: F1, Precision, Recall
    ax = axes[0]
    ax.plot(thresholds, f1s,   color=C_ACCENT, lw=2,   label="F1 Score")
    ax.plot(thresholds, precs, color=C_BEN,    lw=1.8, label="Precision")
    ax.plot(thresholds, recs,  color=C_MAL,    lw=1.8, label="Recall")
    ax.axvline(x=best_threshold, color=C_WARN, lw=1.5,
               linestyle="--", label=f"Chosen threshold ({best_threshold:.2f})")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("F1 / Precision / Recall vs Threshold")
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    # Right: FPR, FNR
    ax = axes[1]
    ax.plot(thresholds, fprs, color=C_MAL,  lw=2, label="False Positive Rate")
    ax.plot(thresholds, fnrs, color=C_WARN, lw=2, label="False Negative Rate")
    ax.axvline(x=best_threshold, color=C_ACCENT, lw=1.5,
               linestyle="--", label=f"Chosen threshold ({best_threshold:.2f})")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Rate")
    ax.set_title("FPR / FNR vs Threshold")
    ax.legend()
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    fig.tight_layout()
    save(fig, "07_threshold_analysis")


# ── Print classification report ───────────────────────────────────────────────

def print_report(y_true, y_pred, metrics):
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred,
                                target_names=["Benign", "Malicious"]))
    print("=" * 60)
    print("SUMMARY METRICS")
    print("=" * 60)
    for name, val in metrics.items():
        bar = "█" * int(val * 30)
        print(f"  {name:<20} {val:.4f}  {bar}")
    print("=" * 60)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  Malicious URL Detector — Model Evaluation")
    print("=" * 60 + "\n")

    model, threshold = load_artefacts()
    X, y, feature_names = load_test_features()

    print("\nRunning predictions on test set...")
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    print(f"  Done — {len(y_pred):,} predictions")

    print("\nGenerating plots → results/plots/\n")

    tn, fp, fn, tp = plot_confusion_matrix(y, y_pred)
    roc_auc        = plot_roc(y, y_prob, threshold)
    ap             = plot_pr_curve(y, y_prob, threshold)
    plot_feature_importance(model, feature_names)
    plot_confidence_distribution(y, y_prob, threshold)
    metrics        = plot_metrics_summary(y, y_pred, y_prob, roc_auc, ap)
    plot_threshold_analysis(y, y_prob, threshold)

    print_report(y, y_pred, metrics)

    print(f"\nAll plots saved to: {RESULTS_DIR}")
    print("Formats: PNG only\n")


if __name__ == "__main__":
    main()