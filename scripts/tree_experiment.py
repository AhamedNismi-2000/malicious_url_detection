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
  results/plots/exp_04_confusion_matrices.png / .pdf         <-- NEW
  results/plots/exp_05_roc_pr_curves.png / .pdf               <-- NEW
  results/metrics/tree_experiment_results.json
  results/metrics/tree_experiment_results.csv                 <-- NEW
  results/metrics/tree_experiment_results.tex                 <-- NEW (LaTeX table for thesis)

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
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score
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

# palette for the "vs trees" line plots in the ROC/PR overlay
TREE_COLORS = {150: "#818cf8", 200: "#c5ad22", 250: "#f97316", 300: "#ef4444"}

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

# The model actually shipped to production (no class_weight balancing applied,
# i.e. class_weight=None in BASE_RF_PARAMS above). Called out explicitly in
# every plot/table below so it's clear this is the selected model, not just
# whichever happens to have the best single metric.
DEFAULT_N = 200


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
    metrics = {
        "accuracy" : round(accuracy_score(y, y_pred), 4),
        "precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "recall"   : round(recall_score(y, y_pred, zero_division=0), 4),
        "f1"       : round(f1_score(y, y_pred, zero_division=0), 4),
        "auc"      : round(roc_auc_score(y, y_proba), 4),
        "avg_precision": round(average_precision_score(y, y_proba), 4),
        "fpr"      : round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0.0,
        "fnr"      : round(fn / (fn + tp), 4) if (fn + tp) > 0 else 0.0,
        "threshold": round(threshold, 2),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    return metrics, y_proba


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
    results  = []
    roc_data = {}   # n_estimators -> y_proba on TEST set (for ROC/PR curves)

    for n in N_ESTIMATORS_LIST:
        print(f"\n{'='*55}")
        print(f"  n_estimators = {n}")
        print(f"{'='*55}")

        train_time = None

        if n == 200:
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
        metrics, y_proba = evaluate(model, X_test, y_test, threshold)
        roc_data[n] = y_proba   # keep for ROC/PR overlay, cheap to store (1 float per URL)

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
        print(f"  Confusion matrix -> TN={metrics['tn']}  FP={metrics['fp']}  "
              f"FN={metrics['fn']}  TP={metrics['tp']}")

        del model
        gc.collect()

    return results, roc_data


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

    _mark_default(ax, results)

    ax.set_xlabel("Number of Trees (n_estimators)")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance vs Number of Trees")
    ax.set_xticks(trees)
    bottom = min(min(f1s), min(aucs), min(precs), min(recs)) - 0.005
    ax.set_ylim([bottom, 1.015])
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "exp_01_performance_vs_trees")


def _mark_default(ax, results):
    """Draws a dashed vertical line + label at DEFAULT_N, used across the line plots."""
    if DEFAULT_N not in [r["n_estimators"] for r in results]:
        return
    ax.axvline(DEFAULT_N, color="#e2e4ec", lw=1.2, linestyle=":", alpha=0.6, zorder=0)
    ylim = ax.get_ylim()
    ax.text(DEFAULT_N, ylim[1], " Selected\n default",
            ha="left", va="top", fontsize=8.5, color="#e2e4ec", alpha=0.85,
            fontstyle="italic")


def plot_training_time(results):
    timed = [r for r in results if r["train_time_s"] is not None]
    if not timed:
        print("  Skipping training time plot (300-tree model was pre-loaded)")
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

    _mark_default(ax, results)
    ax.set_xlabel("Number of Trees (n_estimators)")
    ax.set_ylabel("Error Rate (%)")
    ax.set_title("FPR & FNR vs Number of Trees")
    ax.set_xticks(trees)
    ax.legend()

    ax = axes[1]
    ax.plot(trees, preds, color=C_BEN, lw=2.5, marker="o", markersize=8,
            label="Prediction time (ms/URL)")

    for x, y in zip(trees, preds):
        ax.annotate(f"{y:.2f}ms", (x, y),
                    textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=9, color=C_BEN)

    _mark_default(ax, results)
    ax.set_xlabel("Number of Trees (n_estimators)")
    ax.set_ylabel("Time (ms per URL)")
    ax.set_title("Prediction Speed vs Number of Trees")
    ax.set_xticks(trees)
    ax.legend()

    fig.tight_layout()
    save(fig, "exp_03_fpr_fnr_speed_vs_trees")


def plot_confusion_matrices(results):
    """One heatmap per n_estimators, arranged in a grid. Shows counts + row-normalized %."""
    n_models = len(results)
    ncols = 2
    nrows = int(np.ceil(n_models / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 5 * nrows))
    fig.patch.set_facecolor("#0f1117")
    axes = np.array(axes).reshape(-1)

    labels = ["Benign", "Malicious"]

    for i, r in enumerate(results):
        ax = axes[i]
        cm = np.array([[r["tn"], r["fp"]],
                       [r["fn"], r["tp"]]])
        cm_pct = cm / cm.sum(axis=1, keepdims=True) * 100

        im = ax.imshow(cm, cmap="magma", aspect="equal")

        for row in range(2):
            for col in range(2):
                count = cm[row, col]
                pct = cm_pct[row, col]
                ax.text(col, row, f"{count}\n({pct:.2f}%)",
                        ha="center", va="center",
                        color="white" if cm[row, col] < cm.max() * 0.6 else "black",
                        fontsize=11, fontweight="bold")

        ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
        ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        is_default = r["n_estimators"] == DEFAULT_N
        title = f"n_estimators = {r['n_estimators']}  (F1={r['f1']:.4f}, thr={r['threshold']:.2f})"
        if is_default:
            title += "  \u2605 selected default"
            for spine in ax.spines.values():
                spine.set_edgecolor("#e2e4ec")
                spine.set_linewidth(2.5)
        ax.set_title(title, color="#e2e4ec" if not is_default else C_WARN)
        ax.grid(False)

    # hide unused axes if odd number of models
    for j in range(n_models, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Confusion Matrices Across Tree Counts", fontsize=15, y=1.02)
    fig.tight_layout()
    save(fig, "exp_04_confusion_matrices")


def plot_roc_pr_curves(roc_data, y_test, results):
    """Overlaid ROC curve and Precision-Recall curve for all n_estimators."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor("#0f1117")

    # ROC
    ax = axes[0]
    for r in results:
        n = r["n_estimators"]
        y_proba = roc_data[n]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        is_default = n == DEFAULT_N
        label = f"{n} trees (AUC={r['auc']:.4f})" + (" \u2605 default" if is_default else "")
        ax.plot(fpr, tpr, color=TREE_COLORS.get(n, C_ACCENT),
                lw=3.2 if is_default else 1.6,
                zorder=5 if is_default else 3,
                label=label)
    ax.plot([0, 1], [0, 1], color=C_MUTED, lw=1, linestyle="--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right", fontsize=9)

    # PR
    ax = axes[1]
    for r in results:
        n = r["n_estimators"]
        y_proba = roc_data[n]
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        is_default = n == DEFAULT_N
        label = f"{n} trees (AP={r['avg_precision']:.4f})" + (" \u2605 default" if is_default else "")
        ax.plot(rec, prec, color=TREE_COLORS.get(n, C_ACCENT),
                lw=3.2 if is_default else 1.6,
                zorder=5 if is_default else 3,
                label=label)
    baseline = float(np.mean(y_test))
    ax.axhline(baseline, color=C_MUTED, lw=1, linestyle="--",
               label=f"Baseline ({baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="lower left", fontsize=9)

    fig.tight_layout()
    save(fig, "exp_05_roc_pr_curves")


# ── Thesis-ready table export ──────────────────────────────────────────────────

TABLE_COLUMNS = [
    ("n_estimators",    "Trees",       "{:d}"),
    ("accuracy",        "Accuracy",    "{:.4f}"),
    ("precision",       "Precision",   "{:.4f}"),
    ("recall",          "Recall",      "{:.4f}"),
    ("f1",              "F1",          "{:.4f}"),
    ("auc",             "AUC-ROC",     "{:.4f}"),
    ("avg_precision",   "Avg Prec.",   "{:.4f}"),
    ("fpr",             "FPR",         "{:.4f}"),
    ("fnr",             "FNR",         "{:.4f}"),
    ("threshold",       "Threshold",   "{:.2f}"),
    ("pred_ms_per_url", "ms/URL",      "{:.2f}"),
    ("tn", "TN", "{:d}"), ("fp", "FP", "{:d}"),
    ("fn", "FN", "{:d}"), ("tp", "TP", "{:d}"),
]


def export_csv(results):
    import csv
    path = os.path.join(METRICS_DIR, "tree_experiment_results.csv")
    keys, headers, _ = zip(*TABLE_COLUMNS)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(list(headers) + ["is_default"])
        for r in results:
            writer.writerow([r[k] for k in keys] + [r["n_estimators"] == DEFAULT_N])
    print(f"  CSV saved: {path}")


def export_latex(results):
    """Writes a ready-to-\\input{} LaTeX table (booktabs style) for the thesis appendix.
    The row matching DEFAULT_N is bolded so it's visually obvious which model
    was actually shipped, independent of which row has the single best metric."""
    keys, headers, fmts = zip(*TABLE_COLUMNS)
    path = os.path.join(METRICS_DIR, "tree_experiment_results.tex")

    lines = []
    lines.append("% Auto-generated by tree_experiment.py — requires \\usepackage{booktabs}")
    lines.append(f"% Bolded row = selected default model (n_estimators={DEFAULT_N}, class_weight=None)")
    lines.append("\\begin{table}[ht]")
    lines.append("\\centering")
    lines.append("\\caption{Random Forest performance across different tree counts "
                  f"(bold row = selected default, $n={DEFAULT_N}$)}}")
    lines.append("\\label{tab:tree_experiment}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{" + "c" * len(headers) + "}")
    lines.append("\\toprule")
    lines.append(" & ".join(headers) + " \\\\")
    lines.append("\\midrule")
    for r in results:
        row = [fmt.format(r[k]) for k, fmt in zip(keys, fmts)]
        if r["n_estimators"] == DEFAULT_N:
            row = [f"\\textbf{{{cell}}}" for cell in row]
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table}")

    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  LaTeX table saved: {path}")


def print_summary(results):
    print("\n" + "=" * 96)
    print("EXPERIMENT SUMMARY")
    print("=" * 96)
    print(f"  {'Trees':>6} {'F1':>8} {'AUC':>8} {'Prec':>8} {'Recall':>8} "
          f"{'FPR%':>7} {'FNR%':>7} {'Thresh':>7} {'ms/URL':>8} "
          f"{'TN':>7} {'FP':>6} {'FN':>6} {'TP':>7} {'Train':>10}")
    print("  " + "-" * 94)
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
            f"{r['tn']:>7} "
            f"{r['fp']:>6} "
            f"{r['fn']:>6} "
            f"{r['tp']:>7} "
            f"{t_str:>10}"
        )
    print("=" * 96)

    f1_150 = next((r["f1"] for r in results if r["n_estimators"] == 150), None)
    f1_300 = next((r["f1"] for r in results if r["n_estimators"] == 300), None)
    if f1_150 and f1_300:
        delta = f1_300 - f1_150
        print(f"\n  F1 gain 150 -> 300 trees : {delta:.4f} "
              f"({'significant' if abs(delta) > 0.005 else 'negligible'})")

    fp_150 = next((r["fp"] for r in results if r["n_estimators"] == 150), None)
    fp_300 = next((r["fp"] for r in results if r["n_estimators"] == 300), None)
    if fp_150 is not None and fp_300 is not None:
        print(f"  False positives 150 -> 300 trees : {fp_150} -> {fp_300} "
              f"({fp_300 - fp_150:+d})")

    fn_150 = next((r["fn"] for r in results if r["n_estimators"] == 150), None)
    fn_300 = next((r["fn"] for r in results if r["n_estimators"] == 300), None)
    if fn_150 is not None and fn_300 is not None:
        print(f"  False negatives 150 -> 300 trees : {fn_150} -> {fn_300} "
              f"({fn_300 - fn_150:+d})")

    best = max(results, key=lambda r: r["f1"])
    print(f"  Best F1               : {best['f1']:.4f} "
          f"at n_estimators = {best['n_estimators']}")

    default_row = next((r for r in results if r["n_estimators"] == DEFAULT_N), None)
    if default_row:
        print(f"\n  Selected default model : n_estimators = {DEFAULT_N} "
              f"(class_weight=None, i.e. no re-weighting for class imbalance)")
        print(f"    F1={default_row['f1']:.4f}  AUC={default_row['auc']:.4f}  "
              f"FPR={default_row['fpr']*100:.2f}%  FNR={default_row['fnr']*100:.2f}%  "
              f"Speed={default_row['pred_ms_per_url']:.2f}ms/URL")
        if best["n_estimators"] == DEFAULT_N:
            print(f"    -> This is also the best-performing model on F1 among "
                  f"{N_ESTIMATORS_LIST}, despite fewer trees than the 250/300 variants.")
        else:
            gap = default_row["f1"] - best["f1"]
            print(f"    -> Best F1 belongs to n_estimators={best['n_estimators']} "
                  f"(F1={best['f1']:.4f}); default trails by {abs(gap):.4f} "
                  f"— re-check whether {DEFAULT_N} is still the right production choice.")


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

    results, roc_data = run_experiments(
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
    )

    out_path = os.path.join(METRICS_DIR, "tree_experiment_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results JSON saved: {out_path}")

    print("\nExporting tables for thesis...")
    export_csv(results)
    export_latex(results)

    print("\nGenerating plots...")
    plot_performance(results)
    plot_training_time(results)
    plot_fpr_fnr_speed(results)
    plot_confusion_matrices(results)
    plot_roc_pr_curves(roc_data, y_test, results)

    print_summary(results)

    print(f"\nAll plots saved to: {RESULTS_DIR}")
    print(f"All tables saved to: {METRICS_DIR}")
    print("Formats: PNG + PDF (plots), JSON + CSV + LaTeX (tables)\n")


if __name__ == "__main__":
    main()