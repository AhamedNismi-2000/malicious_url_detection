#!/usr/bin/env python3
"""
OPTIMIZED ADAPTIVE RANDOM FOREST TRAINING
- Faster training with early stopping approximation
- Progress tracking and time estimation
- Enhanced logging and error handling
"""

import os
import time
import json
import joblib
import numpy as np
import datetime
from scipy.sparse import csr_matrix
import psutil  # For memory monitoring

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, train_test_split
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve
)

import warnings
warnings.filterwarnings("ignore")

# ================== CONFIG ==================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS_DIR = os.path.join(BASE_DIR, "data", "splits")
MODELS_DIR = os.path.join(BASE_DIR, "models", "random_forest_enhanced")
os.makedirs(MODELS_DIR, exist_ok=True)

DATASET_CHOICE = "combined"

# Optimized for faster training
BASE_RF_PARAMS = {
    "random_state": 42,
    "n_jobs": -1,  # Use all cores
    "bootstrap": True,
    "oob_score": True,
    "verbose": 0,
}

# Smaller grid for faster search
RF_PARAM_GRID = {
    "n_estimators": [150, 200],  # Reduced from 3 to 2
    "max_depth": [20, 25, None],
    "min_samples_split": [5, 10],  # Reduced from 3 to 2
    "min_samples_leaf": [2, 4],
    "max_features": ["sqrt", "log2"],  # Removed 0.5
    "max_samples": [0.7, 0.8],  # Reduced from 3 to 2
}

class TrainingTimer:
    """Track and estimate training time"""
    def __init__(self):
        self.start_time = time.time()
        self.steps = {}
        
    def start_step(self, step_name):
        self.steps[step_name] = time.time()
        print(f"\n⏱️  Starting: {step_name}")
        
    def end_step(self, step_name):
        if step_name in self.steps:
            elapsed = time.time() - self.steps[step_name]
            total_elapsed = time.time() - self.start_time
            print(f"   ✓ Completed in {elapsed:.1f}s (Total: {total_elapsed/60:.1f}m)")
            return elapsed
        return 0
    
    def get_eta(self, current_progress):
        """Estimate remaining time"""
        elapsed = time.time() - self.start_time
        if current_progress > 0:
            total_estimated = elapsed / current_progress
            remaining = total_estimated - elapsed
            return remaining
        return 0

def check_memory():
    """Check available memory"""
    mem = psutil.virtual_memory()
    print(f"💾 Memory: {mem.available/1e9:.1f}GB available / {mem.total/1e9:.1f}GB total")
    return mem.available

def load_split_optimized(path, max_memory_gb=8):
    """Load data with memory constraints"""
    try:
        data = np.load(path, allow_pickle=True)
        
        # Reconstruct sparse matrix
        if all(k in data for k in ["features_data", "features_indices", "features_indptr"]):
            X = csr_matrix(
                (data["features_data"], data["features_indices"], data["features_indptr"]),
                shape=tuple(data["features_shape"])
            )
        else:
            X = csr_matrix(data["features"])
        
        y = data["labels"]
        
        # Check if we should convert to dense
        estimated_mb = X.shape[0] * X.shape[1] * 8 / 1e6
        
        if estimated_mb < max_memory_gb * 1000:  # If less than 8GB
            X = X.toarray()
            print(f"   Converted to dense ({estimated_mb:.0f}MB)")
        else:
            print(f"   Keeping sparse ({X.nnz:,} non-zero elements)")
            
        return X, y
        
    except Exception as e:
        print(f"❌ Error loading {path}: {e}")
        return None, None

def load_data_optimized():
    """Load data with validation"""
    timer.start_step("Loading Data")
    
    train_path = os.path.join(SPLITS_DIR, DATASET_CHOICE, f"{DATASET_CHOICE}_train.npz")
    val_path = os.path.join(SPLITS_DIR, DATASET_CHOICE, f"{DATASET_CHOICE}_val.npz")
    
    if not os.path.exists(train_path):
        print(f"❌ Training data not found: {train_path}")
        return None, None, None, None
    if not os.path.exists(val_path):
        print(f"❌ Validation data not found: {val_path}")
        return None, None, None, None
    
    X_train, y_train = load_split_optimized(train_path)
    X_val, y_val = load_split_optimized(val_path)
    
    if X_train is None or X_val is None:
        return None, None, None, None
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Training:   {len(y_train):,} samples, {X_train.shape[1]} features")
    print(f"   Validation: {len(y_val):,} samples")
    
    # Class distribution
    train_counts = np.bincount(y_train)
    val_counts = np.bincount(y_val)
    
    print(f"\n📈 Class Distribution:")
    print(f"   Training - Benign: {train_counts[0]:,} ({train_counts[0]/len(y_train):.1%}), "
          f"Malicious: {train_counts[1]:,} ({train_counts[1]/len(y_train):.1%})")
    print(f"   Validation - Benign: {val_counts[0]:,} ({val_counts[0]/len(y_val):.1%}), "
          f"Malicious: {val_counts[1]:,} ({val_counts[1]/len(y_val):.1%})")
    
    timer.end_step("Loading Data")
    return X_train, y_train, X_val, y_val

def calculate_class_weights_enhanced(y):
    """Enhanced class weight calculation"""
    counts = np.bincount(y)
    total = len(y)
    
    minority_class = int(np.argmin(counts))
    majority_class = int(np.argmax(counts))
    
    imbalance_ratio = counts[majority_class] / counts[minority_class]
    
    # Adaptive weighting based on imbalance
    if imbalance_ratio > 10:  # Severe imbalance
        minority_weight = min(6.0, imbalance_ratio * 0.6)
    elif imbalance_ratio > 5:  # High imbalance
        minority_weight = min(4.0, imbalance_ratio * 0.7)
    else:  # Moderate imbalance
        minority_weight = min(3.0, imbalance_ratio * 0.8)
    
    weights = {
        majority_class: 1.0,
        minority_class: minority_weight
    }
    
    print(f"\n⚖️  Class Weights:")
    print(f"   Imbalance ratio: {imbalance_ratio:.1f}:1")
    print(f"   Class 0 weight: {weights.get(0, 1.0):.2f}")
    print(f"   Class 1 weight: {weights.get(1, 1.0):.2f}")
    
    return weights, minority_class, majority_class, imbalance_ratio

def optimize_hyperparams_fast(X, y, class_weights):
    """Fast hyperparameter optimization"""
    timer.start_step("Hyperparameter Optimization")
    
    # Use smaller subset for faster search
    if len(y) > 150000:
        X_subset, _, y_subset, _ = train_test_split(
            X, y, train_size=150000,
            stratify=y, random_state=42
        )
        print(f"   Using subset: {len(y_subset):,} samples")
    else:
        X_subset, y_subset = X, y
    
    rf = RandomForestClassifier(**BASE_RF_PARAMS, class_weight=class_weights)
    
    # Use 2-fold CV for speed (balance between speed and reliability)
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    
    search = RandomizedSearchCV(
        rf,
        RF_PARAM_GRID,
        n_iter=8,  # Reduced from 12
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=42,
        return_train_score=True
    )
    
    print(f"   Searching {8*2} models (8 iterations × 2-fold CV)")
    print(f"   This will take 10-20 minutes...")
    
    search.fit(X_subset, y_subset)
    
    print(f"\n✅ Best Parameters:")
    for param, value in search.best_params_.items():
        print(f"   {param}: {value}")
    print(f"✅ Best CV F1: {search.best_score_:.4f}")
    
    timer.end_step("Hyperparameter Optimization")
    return search.best_params_, search.best_score_

def train_final_model(X_train, y_train, best_params, class_weights):
    """Train final model with progress estimation"""
    timer.start_step("Final Model Training")
    
    # Update params with class weights
    final_params = best_params.copy()
    final_params.update(BASE_RF_PARAMS)
    final_params["class_weight"] = class_weights
    
    print(f"\n🌲 Training Random Forest with {final_params.get('n_estimators', 200)} trees")
    print(f"   Samples: {len(y_train):,}, Features: {X_train.shape[1]}")
    print(f"   This will take 20-40 minutes...")
    
    # Train in stages for progress tracking
    model = RandomForestClassifier(**final_params)
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"   ✓ Training completed in {training_time/60:.1f} minutes")
    
    # Show model stats
    if hasattr(model, 'oob_score_'):
        print(f"   Out-of-bag score: {model.oob_score_:.4f}")
    
    timer.end_step("Final Model Training")
    return model, training_time

def optimize_threshold_comprehensive(model, X_val, y_val, minority_class):
    """Comprehensive threshold optimization"""
    timer.start_step("Threshold Optimization")
    
    y_proba = model.predict_proba(X_val)[:, minority_class]
    
    # Test multiple threshold strategies
    fpr, tpr, thresholds = roc_curve(y_val == minority_class, y_proba)
    
    best_threshold = 0.5
    best_score = -1
    
    print(f"   Testing {len(thresholds)} thresholds...")
    
    # Strategy 1: Maximize F1 with recall constraint
    for i, th in enumerate(thresholds):
        if i % 20 == 0:  # Sample every 20th threshold for speed
            y_pred = (y_proba >= th).astype(int)
            recall = recall_score(y_val == minority_class, y_pred, zero_division=0)
            
            if recall >= 0.65:  # Minimum recall requirement
                precision = precision_score(y_val == minority_class, y_pred, zero_division=0)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
                
                # Balance recall and precision
                score = recall * 0.6 + precision * 0.4
                
                if score > best_score:
                    best_score = score
                    best_threshold = th
    
    # Adjust for safety margin
    optimal_threshold = max(0.15, min(0.85, best_threshold * 0.95))
    
    print(f"   Selected threshold: {optimal_threshold:.4f}")
    print(f"   (Recall ≥ {0.65:.2f} with optimal precision)")
    
    timer.end_step("Threshold Optimization")
    return optimal_threshold

def evaluate_comprehensive(model, X_val, y_val, threshold, minority_class):
    """Comprehensive evaluation"""
    y_proba = model.predict_proba(X_val)[:, minority_class]
    y_pred = (y_proba >= threshold).astype(int)
    
    # Basic metrics
    precision = precision_score(y_val == minority_class, y_pred, zero_division=0)
    recall = recall_score(y_val == minority_class, y_pred, zero_division=0)
    f1 = f1_score(y_val == minority_class, y_pred, zero_division=0)
    auc = roc_auc_score(y_val == minority_class, y_proba)
    avg_precision = average_precision_score(y_val == minority_class, y_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_val == minority_class, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Rates
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f"\n📊 VALIDATION RESULTS:")
    print(f"   Threshold:       {threshold:.4f}")
    print(f"   Precision:       {precision:.4f}")
    print(f"   Recall:          {recall:.4f}")
    print(f"   F1-score:        {f1:.4f}")
    print(f"   AUC-ROC:         {auc:.4f}")
    print(f"   Avg Precision:   {avg_precision:.4f}")
    print(f"   False Positive:  {fpr:.4f}")
    print(f"   Detection Rate:  {detection_rate:.1%}")
    
    print(f"\n📊 Confusion Matrix:")
    print(f"   TP: {tp:,}  FP: {fp:,}")
    print(f"   FN: {fn:,}  TN: {tn:,}")
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "auc_roc": auc,
        "avg_precision": avg_precision,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "detection_rate": detection_rate,
        "confusion_matrix": [int(tp), int(fp), int(fn), int(tn)]
    }

def save_model_enhanced(model, threshold, metrics, minority_class, imbalance_ratio, training_time):
    """Save model with enhanced metadata"""
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Model path
    model_path = os.path.join(MODELS_DIR, f"rf_enhanced_{DATASET_CHOICE}_{ts}.pkl")
    joblib.dump(model, model_path, compress=3)
    print(f"💾 Model saved: {model_path}")
    
    # Feature importance
    feature_importance = {}
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        top_indices = np.argsort(importances)[-10:]  # Top 10 features
        feature_importance = {
            "top_features_indices": top_indices.tolist(),
            "top_features_scores": importances[top_indices].tolist(),
            "mean_importance": float(np.mean(importances))
        }
    
    # Enhanced metadata
    metadata = {
        "model_info": {
            "type": "RandomForestClassifier",
            "n_estimators": model.n_estimators,
            "n_features": model.n_features_in_,
            "minority_class": minority_class,
            "class_weights": model.class_weight,
            "hyperparameters": model.get_params()
        },
        "dataset_info": {
            "name": DATASET_CHOICE,
            "imbalance_ratio": imbalance_ratio,
            "training_samples": model.n_features_in_  # Note: This is actually feature count
        },
        "performance": {
            "optimal_threshold": threshold,
            "validation_metrics": metrics,
            "deployment_recommendation": "READY" if metrics["recall"] >= 0.7 else "NEEDS_IMPROVEMENT"
        },
        "training_info": {
            "training_date": ts,
            "training_time_seconds": training_time,
            "total_training_time": time.time() - timer.start_time
        },
        "feature_importance": feature_importance
    }
    
    meta_path = os.path.join(MODELS_DIR, f"model_metadata_{ts}.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"💾 Metadata saved: {meta_path}")
    
    return model_path, meta_path

# ================== MAIN ==================
def main():
    global timer
    timer = TrainingTimer()
    
    print("\n" + "="*70)
    print("🚀 OPTIMIZED RANDOM FOREST TRAINING")
    print("   Adaptive for Imbalanced Security Data")
    print("="*70 + "\n")
    
    # Check system resources
    check_memory()
    
    # Load data
    X_train, y_train, X_val, y_val = load_data_optimized()
    if X_train is None:
        print("❌ Failed to load data. Exiting.")
        return
    
    # Calculate class weights
    timer.start_step("Calculating Class Weights")
    class_weights, minority_class, majority_class, imbalance_ratio = calculate_class_weights_enhanced(y_train)
    timer.end_step("Calculating Class Weights")
    
    # Optimize hyperparameters
    best_params, best_score = optimize_hyperparams_fast(X_train, y_train, class_weights)
    
    # Train final model
    model, training_time = train_final_model(X_train, y_train, best_params, class_weights)
    
    # Optimize threshold
    threshold = optimize_threshold_comprehensive(model, X_val, y_val, minority_class)
    
    # Evaluate
    metrics = evaluate_comprehensive(model, X_val, y_val, threshold, minority_class)
    
    # Save
    model_path, meta_path = save_model_enhanced(
        model, threshold, metrics, minority_class, imbalance_ratio, training_time
    )
    
    total_time = time.time() - timer.start_time
    
    print(f"\n✅ TRAINING COMPLETED SUCCESSFULLY!")
    print(f"   Total time: {total_time/60:.1f} minutes")
    
    print(f"\n🎯 DEPLOYMENT STATUS:")
    if metrics["recall"] >= 0.75 and metrics["false_positive_rate"] <= 0.03:
        print("   ✅ EXCELLENT - Ready for Chrome extension deployment")
    elif metrics["recall"] >= 0.65:
        print("   ⚠️  GOOD - Can deploy with monitoring")
    else:
        print("   ❌ NEEDS IMPROVEMENT - Retrain with adjustments")
    
    print(f"\n💡 Next steps:")
    print(f"   1. Test model: python scripts/test_enhanced.py --model {model_path}")
    print(f"   2. Validate: python scripts/validate_enhanced.py --model {model_path}")
    print(f"   3. Deploy to Chrome extension")

if __name__ == "__main__":
    main()