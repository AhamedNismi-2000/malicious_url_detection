#!/usr/bin/env python3
"""
error_analysis.py
-----------------
Analyses False Positives and False Negatives from the test set.

What it does:
  1. Loads test features + test URLs
  2. Runs predictions with tuned threshold
  3. Extracts FP and FN cases
  4. Analyses patterns in each error type
  5. Saves detailed report + plots

Output:
  results/error_analysis/error_report.txt
  results/error_analysis/fp_urls.csv
  results/error_analysis/fn_urls.csv
  results/plots/08_error_analysis_fp_features.png / .pdf
  results/plots/09_error_analysis_fn_features.png / .pdf
  results/plots/10_error_confidence_distribution.png / .pdf

Run from project root:
  python scripts/error_analysis.py
"""

import os
import sys
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DIR = os.path.join(BASE_DIR, "features")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
SPLITS_DIR   = os.path.join(BASE_DIR, "data", "splits")
OUTPUT_DIR   = os.path.join(BASE_DIR, "results", "error_analysis")
PLOTS_DIR    = os.path.join(BASE_DIR, "results", "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

TEST_NPZ  = os.path.join(FEATURES_DIR, "features_test.npz")
TEST_CSV  = os.path.join(SPLITS_DIR,   "test_urls.csv")

# ── Heuristic feature names (first 48) ───────────────────────────────────────
HEURISTIC_FEATURES = [
    "url_len", "path_len", "num_dots", "path_dots", "num_hyphens",
    "num_underscores", "num_at", "num_qmark", "num_equal", "num_amp",
    "num_percent", "num_digits", "num_letters", "num_subdirs", "num_frag",
    "num_special", "num_repeating", "num_upper", "num_non_ascii",
    "num_slashes", "num_params", "ratio_digits", "ratio_letters",
    "url_entropy", "ip_flag", "subdomain_parts", "has_multi_subdomain",
    "tld_len", "risky_tld", "https_flag", "shortened", "sus_words",
    "brand_mismatch", "puny", "susp_ext", "suspicious_port",
    "max_consonants", "max_vowels", "max_digits",
    "leet_speak_score", "homoglyph_suspicious", "encoding_ratio",
    "punycode_suspicious", "subdomain_spam_score", "visual_brand_similarity",
    "brand_in_domain", "leet_in_domain", "brand_hyphen_suspicious",
]
N_HEURISTIC = len(HEURISTIC_FEATURES)  # 48

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
C_BEN    = "#22c55e"
C_ACCENT = "#818cf8"
C_WARN   = "#f59e0b"
C_MUTED  = "#6b7280"
SAVE_DPI = 200


def save(fig, name):
    for ext in ("png", "pdf"):
        path = os.path.join(PLOTS_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    print(f"  ✓  {name}.png / .pdf")
    plt.close(fig)


# ── Load data ─────────────────────────────────────────────────────────────────

def load_test_data():
    print("Loading test features...")
    data = np.load(TEST_NPZ, allow_pickle=True)
    X = sp.csr_matrix(
        (data["data"], data["indices"], data["indptr"]),
        shape=tuple(data["shape"])
    ).toarray().astype(np.float32)
    y = data["labels"].astype(int)
    print(f"  Shape: {X.shape}  |  Malicious: {y.sum():,}  Benign: {(y==0).sum():,}")

    print("Loading test URLs...")
    if os.path.exists(TEST_CSV):
        df   = pd.read_csv(TEST_CSV, dtype={"url": str, "label": int})
        urls = df["url"].tolist()
        print(f"  Loaded {len(urls):,} URLs from test_urls.csv")
    else:
        urls = [f"url_{i}" for i in range(len(y))]
        print(f"  WARNING: {TEST_CSV} not found — using placeholder URLs")

    return X, y, urls


# ── Predict ───────────────────────────────────────────────────────────────────

def get_predictions(model, X, threshold):
    print("Running predictions...")
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    return y_prob, y_pred


# ── Extract error cases ───────────────────────────────────────────────────────

def extract_errors(X, y, y_pred, y_prob, urls, n_samples=50):
    """Extract FP and FN cases with their feature vectors and confidence."""
    fp_idx = np.where((y == 0) & (y_pred == 1))[0]  # Benign but predicted Malicious
    fn_idx = np.where((y == 1) & (y_pred == 0))[0]  # Malicious but predicted Benign

    print(f"\n  Total FP: {len(fp_idx):,}  |  Total FN: {len(fn_idx):,}")

    # Sample for analysis
    np.random.seed(42)
    fp_sample = np.random.choice(fp_idx, min(n_samples, len(fp_idx)), replace=False)
    fn_sample = np.random.choice(fn_idx, min(n_samples, len(fn_idx)), replace=False)

    fp_data = {
        "urls"      : [urls[i] for i in fp_sample],
        "confidence": y_prob[fp_sample] * 100,
        "features"  : X[fp_sample, :N_HEURISTIC],
        "indices"   : fp_sample,
    }
    fn_data = {
        "urls"      : [urls[i] for i in fn_sample],
        "confidence": y_prob[fn_sample] * 100,
        "features"  : X[fn_sample, :N_HEURISTIC],
        "indices"   : fn_sample,
    }

    return fp_data, fn_data, fp_idx, fn_idx


# ── Analyse patterns ──────────────────────────────────────────────────────────

def analyse_patterns(error_data, all_data_features, label: str) -> dict:
    """
    Compare mean feature values of error cases vs correct cases.
    High difference = feature contributed to the error.
    """
    error_features  = error_data["features"]        # (n_errors, 48)
    correct_features = all_data_features             # (all, 48)

    error_mean   = np.mean(error_features,   axis=0)
    correct_mean = np.mean(correct_features, axis=0)
    diff         = error_mean - correct_mean

    # Top features that differ most between errors and correct cases
    top_idx  = np.argsort(np.abs(diff))[::-1][:15]
    patterns = {
        HEURISTIC_FEATURES[i]: {
            "error_mean"  : round(float(error_mean[i]), 4),
            "overall_mean": round(float(correct_mean[i]), 4),
            "difference"  : round(float(diff[i]), 4),
        }
        for i in top_idx
    }

    print(f"\n  Top distinguishing features for {label}:")
    print(f"  {'Feature':<30} {'Error Mean':>12} {'Overall Mean':>13} {'Diff':>8}")
    print("  " + "-" * 65)
    for feat, vals in patterns.items():
        print(f"  {feat:<30} {vals['error_mean']:>12.4f} "
              f"{vals['overall_mean']:>13.4f} {vals['difference']:>8.4f}")

    return patterns


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_feature_diff(patterns: dict, title: str, color: str, filename: str):
    """Bar chart of feature differences between error cases and overall."""
    features = list(patterns.keys())[:12]
    diffs    = [patterns[f]["difference"] for f in features]
    labels   = [f.replace("_", " ") for f in features]

    colors = [color if d > 0 else C_BEN for d in diffs]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0f1117")

    bars = ax.barh(range(len(features)), diffs[::-1],
                   color=colors[::-1], edgecolor="none", height=0.65)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(labels[::-1], fontsize=9)
    ax.axvline(x=0, color=C_MUTED, lw=1, linestyle="--")
    ax.set_xlabel("Mean Difference (Error cases vs Overall)")
    ax.set_title(title)

    for bar, val in zip(bars, diffs[::-1]):
        ax.text(bar.get_width() + (0.002 if val >= 0 else -0.002),
                bar.get_y() + bar.get_height() / 2,
                f"{val:+.3f}", va="center", fontsize=8, color=C_MUTED)

    fig.tight_layout()
    save(fig, filename)


def plot_confidence_distribution(fp_conf, fn_conf):
    """Show confidence distribution of FP and FN cases."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#0f1117")

    # FP confidence — model was confident but wrong
    ax = axes[0]
    ax.hist(fp_conf, bins=20, color=C_MAL, alpha=0.8, edgecolor="none")
    ax.axvline(x=np.mean(fp_conf), color=C_WARN, lw=2,
               linestyle="--", label=f"Mean: {np.mean(fp_conf):.1f}%")
    ax.set_xlabel("Model Confidence (%)")
    ax.set_ylabel("Number of URLs")
    ax.set_title("False Positives — Confidence Distribution\n"
                 "(Benign URLs wrongly classified as Malicious)")
    ax.legend()

    # FN confidence — model was not confident enough
    ax = axes[1]
    ax.hist(fn_conf, bins=20, color=C_WARN, alpha=0.8, edgecolor="none")
    ax.axvline(x=np.mean(fn_conf), color=C_MAL, lw=2,
               linestyle="--", label=f"Mean: {np.mean(fn_conf):.1f}%")
    ax.set_xlabel("Model Confidence (%)")
    ax.set_ylabel("Number of URLs")
    ax.set_title("False Negatives — Confidence Distribution\n"
                 "(Malicious URLs wrongly classified as Benign)")
    ax.legend()

    fig.tight_layout()
    save(fig, "10_error_confidence_distribution")


# ── Save CSVs ─────────────────────────────────────────────────────────────────

def save_error_csvs(fp_data, fn_data):
    fp_df = pd.DataFrame({
        "url"           : fp_data["urls"],
        "confidence_pct": np.round(fp_data["confidence"], 2),
        "true_label"    : "BENIGN",
        "predicted"     : "MALICIOUS",
    })
    fn_df = pd.DataFrame({
        "url"           : fn_data["urls"],
        "confidence_pct": np.round(fn_data["confidence"], 2),
        "true_label"    : "MALICIOUS",
        "predicted"     : "BENIGN",
    })

    fp_path = os.path.join(OUTPUT_DIR, "fp_urls.csv")
    fn_path = os.path.join(OUTPUT_DIR, "fn_urls.csv")
    fp_df.to_csv(fp_path, index=False)
    fn_df.to_csv(fn_path, index=False)
    print(f"\n  FP URLs saved: {fp_path}")
    print(f"  FN URLs saved: {fn_path}")
    return fp_df, fn_df


# ── Write report ──────────────────────────────────────────────────────────────

def write_report(fp_data, fn_data, fp_idx, fn_idx,
                 fp_patterns, fn_patterns,
                 fp_df, fn_df, threshold):

    report_path = os.path.join(OUTPUT_DIR, "error_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:

        f.write("=" * 70 + "\n")
        f.write("MALICIOUS URL DETECTOR — ERROR ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Threshold used : {threshold:.2f}\n")
        f.write(f"Total FP       : {len(fp_idx):,} "
                f"(benign URLs wrongly blocked)\n")
        f.write(f"Total FN       : {len(fn_idx):,} "
                f"(malicious URLs missed)\n\n")

        # FP section
        f.write("=" * 70 + "\n")
        f.write("FALSE POSITIVES — Benign URLs flagged as Malicious\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Mean confidence: {np.mean(fp_data['confidence']):.1f}%\n")
        f.write(f"Max confidence : {np.max(fp_data['confidence']):.1f}%\n")
        f.write(f"Min confidence : {np.min(fp_data['confidence']):.1f}%\n\n")

        f.write("Sample FP URLs (sorted by confidence desc):\n")
        f.write("-" * 70 + "\n")
        fp_sorted = fp_df.sort_values("confidence_pct", ascending=False)
        for _, row in fp_sorted.head(20).iterrows():
            f.write(f"  [{row['confidence_pct']:.1f}%]  {row['url']}\n")

        f.write("\nTop distinguishing features:\n")
        f.write("-" * 70 + "\n")
        for feat, vals in fp_patterns.items():
            f.write(f"  {feat:<30} diff={vals['difference']:+.4f}  "
                    f"(error={vals['error_mean']:.3f}, "
                    f"overall={vals['overall_mean']:.3f})\n")

        # FN section
        f.write("\n" + "=" * 70 + "\n")
        f.write("FALSE NEGATIVES — Malicious URLs missed by model\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Mean confidence: {np.mean(fn_data['confidence']):.1f}%\n")
        f.write(f"Max confidence : {np.max(fn_data['confidence']):.1f}%\n")
        f.write(f"Min confidence : {np.min(fn_data['confidence']):.1f}%\n\n")

        f.write("Sample FN URLs (sorted by confidence asc — hardest to catch):\n")
        f.write("-" * 70 + "\n")
        fn_sorted = fn_df.sort_values("confidence_pct", ascending=True)
        for _, row in fn_sorted.head(20).iterrows():
            f.write(f"  [{row['confidence_pct']:.1f}%]  {row['url']}\n")

        f.write("\nTop distinguishing features:\n")
        f.write("-" * 70 + "\n")
        for feat, vals in fn_patterns.items():
            f.write(f"  {feat:<30} diff={vals['difference']:+.4f}  "
                    f"(error={vals['error_mean']:.3f}, "
                    f"overall={vals['overall_mean']:.3f})\n")

        # Insights
        f.write("\n" + "=" * 70 + "\n")
        f.write("KEY INSIGHTS\n")
        f.write("=" * 70 + "\n\n")
        f.write("FALSE POSITIVES (legitimate URLs incorrectly blocked):\n")
        f.write("  - High confidence FPs suggest model learned wrong patterns\n")
        f.write("  - Check top features above for common FP characteristics\n")
        f.write("  - Consider adding frequently FP'd domains to whitelist\n\n")
        f.write("FALSE NEGATIVES (malicious URLs that evaded detection):\n")
        f.write("  - Low confidence FNs are borderline cases near threshold\n")
        f.write("  - High confidence FNs suggest novel attack patterns\n")
        f.write("  - These URLs likely lack obvious malicious features\n")

    print(f"  Report saved: {report_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  Malicious URL Detector — Error Analysis")
    print("=" * 60 + "\n")

    # Load model
    print("Loading model...")
    model = joblib.load(os.path.join(MODELS_DIR, "rf_model_latest.joblib"))
    with open(os.path.join(MODELS_DIR, "threshold.json")) as f:
        threshold = float(json.load(f).get("threshold", 0.44))
    print(f"  Threshold: {threshold:.2f}")

    # Load data
    X, y, urls = load_test_data()

    # Predict
    y_prob, y_pred = get_predictions(model, X, threshold)

    # Confusion matrix summary
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    print(f"\n  TN={tn:,}  FP={fp:,}  FN={fn:,}  TP={tp:,}")
    print(f"  FPR={fp/(fp+tn)*100:.2f}%  FNR={fn/(fn+tp)*100:.2f}%")

    # Extract errors
    fp_data, fn_data, fp_idx, fn_idx = extract_errors(
        X, y, y_pred, y_prob, urls, n_samples=50
    )

    # Analyse patterns
    print("\nAnalysing False Positive patterns...")
    fp_patterns = analyse_patterns(fp_data, X[:, :N_HEURISTIC], "False Positives")

    print("\nAnalysing False Negative patterns...")
    fn_patterns = analyse_patterns(fn_data, X[:, :N_HEURISTIC], "False Negatives")

    # Save CSVs
    fp_df, fn_df = save_error_csvs(fp_data, fn_data)

    # Generate plots
    print("\nGenerating plots...")
    plot_feature_diff(
        fp_patterns,
        "False Positives — Feature Differences\n"
        "(Positive = feature higher in FP cases than overall)",
        C_MAL, "08_error_fp_feature_diff"
    )
    plot_feature_diff(
        fn_patterns,
        "False Negatives — Feature Differences\n"
        "(Positive = feature higher in FN cases than overall)",
        C_WARN, "09_error_fn_feature_diff"
    )
    plot_confidence_distribution(
        fp_data["confidence"],
        fn_data["confidence"]
    )

    # Write report
    print("\nWriting report...")
    write_report(
        fp_data, fn_data, fp_idx, fn_idx,
        fp_patterns, fn_patterns,
        fp_df, fn_df, threshold
    )

    print("\n" + "=" * 60)
    print("  Error Analysis Complete")
    print("=" * 60)
    print(f"\n  Outputs saved to: {OUTPUT_DIR}")
    print(f"  Plots saved to  : {PLOTS_DIR}")
    print("\n  Files generated:")
    print("    error_report.txt       — full analysis report")
    print("    fp_urls.csv            — false positive URLs")
    print("    fn_urls.csv            — false negative URLs")
    print("    08_error_fp_feature_diff.png/pdf")
    print("    09_error_fn_feature_diff.png/pdf")
    print("    10_error_confidence_distribution.png/pdf\n")


if __name__ == "__main__":
    main()