#!/usr/bin/env python3
"""
snapshot_baseline.py  (v2)  -  freeze the CURRENT model so the new one can be compared fairly
-------------------------------------------------------------------------------------------
Run ONCE, BEFORE you change feature_extraction.py. No flags needed:

    python scripts/snapshot_baseline.py

What it does
  1. Copies the whole models/ folder to models_backup_baseline/ (your rollback point).
  2. Scores features_val.npz and features_test.npz with the current model and stores the
     probabilities in results/baseline/baseline_probs.npz  (train_compare.py reads them).
  3. Scores every URL in data/reference/regression_urls.csv with the current model
     (model only: no whitelist, rules or Safe Browsing) and stores the result.

It refuses to run if models_backup_baseline/ already exists, so a good backup is never
overwritten by a later (already changed) model.
"""

import json
import os
import shutil
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp

warnings.filterwarnings("ignore")

# ─────────────────────────── PATHS ───────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR  = os.path.join(BASE_DIR, "scripts")
MODELS_DIR   = os.path.join(BASE_DIR, "models")
FEATURES_DIR = os.path.join(BASE_DIR, "features")
OUT_DIR      = os.path.join(BASE_DIR, "results", "baseline")

# ═════════════════════════ SETTINGS (edit here) ═════════════════════════
BACKUP_DIR     = os.path.join(BASE_DIR, "models_backup_baseline")
REGRESSION_CSV = os.path.join(BASE_DIR, "data", "reference", "regression_urls.csv")
MODEL_FILE     = "rf_model_latest.joblib"


def load_dense(path: str):
    d = np.load(path, allow_pickle=True)
    X = sp.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"]))
    return X.toarray().astype(np.float32), d["labels"].astype(int)


def proba_of(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, list(model.classes_).index(1)]


def main() -> int:
    print("SNAPSHOT OF THE CURRENT (BASELINE) MODEL")
    print("=" * 60)
    if os.path.exists(BACKUP_DIR):
        print(f"{BACKUP_DIR} already exists: the baseline was already saved.\n"
              "Delete that folder only if you really want to take a new snapshot.")
        return 1
    for p in (os.path.join(MODELS_DIR, MODEL_FILE), os.path.join(MODELS_DIR, "threshold.json"),
              os.path.join(FEATURES_DIR, "features_val.npz"), os.path.join(FEATURES_DIR, "features_test.npz")):
        if not os.path.exists(p):
            print(f"Missing: {p}")
            return 1

    shutil.copytree(MODELS_DIR, BACKUP_DIR)
    print(f"[1/3] models/ copied to {BACKUP_DIR}")

    model = joblib.load(os.path.join(MODELS_DIR, MODEL_FILE))
    with open(os.path.join(MODELS_DIR, "threshold.json"), encoding="utf-8") as fh:
        threshold = float(json.load(fh)["threshold"])
    Xva, yva = load_dense(os.path.join(FEATURES_DIR, "features_val.npz"))
    Xte, yte = load_dense(os.path.join(FEATURES_DIR, "features_test.npz"))
    if Xva.shape[1] != model.n_features_in_:
        print(f"features_val.npz has {Xva.shape[1]} columns but the model expects "
              f"{model.n_features_in_}: the features were already re-extracted. Restore the "
              "old features first (re-run the OLD feature_extraction.py), then run this again.")
        shutil.rmtree(BACKUP_DIR)
        return 1
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(OUT_DIR, "baseline_probs.npz"),
                        val_proba=proba_of(model, Xva), val_y=yva,
                        test_proba=proba_of(model, Xte), test_y=yte,
                        threshold=np.array(threshold))
    print(f"[2/3] val/test probabilities saved (threshold {threshold:.2f})")

    reg = {}
    if os.path.exists(REGRESSION_CSV):
        sys.path.insert(0, SCRIPTS_DIR)
        from feature_extraction import extract_heuristic_features, segment_url   # OLD version, on purpose
        scaler   = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
        vec_char = joblib.load(os.path.join(MODELS_DIR, "vectorizer_char.joblib"))
        vec_word = joblib.load(os.path.join(MODELS_DIR, "vectorizer_word.joblib"))
        df = pd.read_csv(REGRESSION_CSV, dtype=str, keep_default_na=False)
        rows = []
        for u in df["url"]:
            h = scaler.transform(np.array(extract_heuristic_features(u), dtype=np.float32).reshape(1, -1)).ravel()
            s = segment_url(u)
            rows.append(np.concatenate([h, vec_char.transform([s]).toarray().ravel(),
                                        vec_word.transform([s]).toarray().ravel()]))
        probs = proba_of(model, np.array(rows, dtype=np.float32))
        reg = {u: round(float(p), 4) for u, p in zip(df["url"], probs)}
        print(f"[3/3] {len(reg)} regression URLs scored")
    else:
        print(f"[3/3] {REGRESSION_CSV} not found: regression part skipped")
    with open(os.path.join(OUT_DIR, "baseline_regression.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(threshold=threshold, proba=reg), fh, indent=2)

    print("\nDONE. Now apply the feature_extraction.py patch and run:")
    print("    python scripts/feature_extraction.py")
    print("    python scripts/train_compare.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())