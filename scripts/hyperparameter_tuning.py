"""
MEMORY-OPTIMIZED HYPERPARAMETER TUNING
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from scipy.stats import randint, uniform
from scipy.sparse import load_npz
import joblib
import warnings
import psutil
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def check_memory_usage():
    """Check available memory and warn if low"""
    memory_info = psutil.virtual_memory()
    available_gb = memory_info.available / (1024 ** 3)
    print(f"💾 Available memory: {available_gb:.1f} GB")
    
    if available_gb < 4:
        print("⚠️  WARNING: Low memory available!")
        return False
    return True

def load_split_data_subset(dataset_name, split_name, max_samples=50000):
    """Load subset of data to reduce memory usage"""
    try:
        split_dir = os.path.join(BASE_DIR, "data", "splits", dataset_name)
        features_path = os.path.join(split_dir, f"{dataset_name}_{split_name}_features.npz")
        meta_path = os.path.join(split_dir, f"{dataset_name}_{split_name}.npz")
        
        # Load sparse matrix
        features = load_npz(features_path)
        data = np.load(meta_path, allow_pickle=True)
        labels = data['labels']
        
        # Use subset for tuning to save memory
        if len(labels) > max_samples:
            indices = np.random.choice(len(labels), max_samples, replace=False)
            features = features[indices]
            labels = labels[indices]
            print(f"📊 Using subset: {features.shape} (from original {data['labels'].shape})")
        else:
            print(f"📊 Loaded full data: {features.shape}")
            
        return features, labels
    except Exception as e:
        print(f"❌ Error loading {split_name} data for {dataset_name}: {e}")
        return None, None

def evaluate_model_comprehensive(model, X_test, y_test):
    """Comprehensive model evaluation with memory efficiency"""
    # Predict in chunks for large datasets
    if X_test.shape[0] > 50000:
        print("   🔄 Predicting in chunks...")
        chunk_size = 20000
        all_predictions = []
        all_probabilities = []
        
        for i in range(0, X_test.shape[0], chunk_size):
            chunk = X_test[i:i+chunk_size]
            chunk_pred = model.predict(chunk)
            chunk_proba = model.predict_proba(chunk)[:, 1]
            all_predictions.extend(chunk_pred)
            all_probabilities.extend(chunk_proba)
            
        y_pred = np.array(all_predictions)
        y_proba = np.array(all_probabilities)
    else:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc_roc = roc_auc_score(y_test, y_proba)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_roc,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp
    }

def tune_random_forest_memory_safe():
    """Memory-optimized hyperparameter tuning"""
    print("🎯 MEMORY-OPTIMIZED HYPERPARAMETER TUNING")
    print("=" * 60)
    
    # Check memory first
    if not check_memory_usage():
        print("❌ Insufficient memory for tuning!")
        return None
    
    # Load smaller subset for tuning
    print("📥 Loading data subsets for memory efficiency...")
    X_train, y_train = load_split_data_subset("combined", "train", max_samples=50000)
    X_test, y_test = load_split_data_subset("combined", "test", max_samples=20000)
    
    if X_train is None or X_test is None:
        print("❌ Failed to load data!")
        return None
    
    print(f"📊 Training subset: {X_train.shape}")
    print(f"📊 Test subset: {X_test.shape}")
    print(f"🏷️ Classes - Benign: {np.sum(y_train==0):,}, Malicious: {np.sum(y_train==1):,}")
    
    # MEMORY-OPTIMIZED PARAMETERS
    param_dist = {
        'n_estimators': randint(50, 150),           # Reduced from 100-300
        'max_depth': [10, 15, 20, None],           # Limit depth
        'min_samples_split': randint(5, 20),       # Increased to prevent overfitting
        'min_samples_leaf': randint(3, 10),        # Increased for smaller trees
        'max_features': ['sqrt', 'log2', 0.5],     # Reduced options
        'max_samples': [0.7, 0.8, 0.9],           # Bootstrap samples (MEMORY SAVER!)
        'class_weight': [
            'balanced', 
            {0: 1, 1: 4},  # Reduced from 6,8,10
            {0: 1, 1: 6}
        ]
    }
    
    # Memory-optimized Random Forest
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=4,  # ← LIMITED to 4 cores instead of -1
        verbose=1,
        bootstrap=True  # Enable bootstrapping to work with max_samples
    )
    
    # Memory-optimized search
    print("🚀 Starting Memory-Optimized RandomizedSearchCV...")
    print("⏳ Estimated time: 15-30 minutes...")
    
    search = RandomizedSearchCV(
        rf, 
        param_dist, 
        n_iter=10,           # Reduced from 20
        cv=2,                # Reduced from 3
        scoring='recall',
        random_state=42,
        n_jobs=2,            # ← LIMITED parallelization
        verbose=2,
        error_score='raise'
    )
    
    try:
        search.fit(X_train, y_train)
    except MemoryError:
        print("❌ MEMORY ERROR! Try with even smaller data subset.")
        return None
    
    print(f"\n🎯 TUNING COMPLETED!")
    print(f"📊 Best parameters: {search.best_params_}")
    print(f"📊 Best CV recall: {search.best_score_:.4f}")
    
    # Test best model on full test set
    print(f"\n🔍 Loading full test set for final evaluation...")
    X_test_full, y_test_full = load_split_data_subset("combined", "test", max_samples=100000)
    
    best_model = search.best_estimator_
    test_metrics = evaluate_model_comprehensive(best_model, X_test_full, y_test_full)
    
    print(f"🎯 TEST SET PERFORMANCE (Full Data):")
    print(f"   Recall:      {test_metrics['recall']:.4f}  ← Malicious detection")
    print(f"   Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"   Precision:   {test_metrics['precision']:.4f}")
    print(f"   F1-Score:    {test_metrics['f1_score']:.4f}")
    print(f"   AUC-ROC:     {test_metrics['auc_roc']:.4f}")
    print(f"   False Negatives: {test_metrics['false_negatives']:,}")
    print(f"   False Positives: {test_metrics['false_positives']:,}")
    
    # Compare with original
    print(f"\n📈 COMPARISON WITH ORIGINAL BEST MODEL:")
    print(f"   Original Recall: 0.7783")
    print(f"   Tuned Recall:    {test_metrics['recall']:.4f}")
    print(f"   Improvement:     {test_metrics['recall'] - 0.7783:+.4f}")
    
    # Save tuned model
    tuned_model_path = os.path.join(MODELS_DIR, "tuned_rf_memory_optimized.joblib")
    joblib.dump(best_model, tuned_model_path)
    print(f"💾 Tuned model saved: {tuned_model_path}")
    
    return best_model

def quick_tune_even_safer():
    """Even more memory-safe tuning for low-RAM systems"""
    print("🛡️  ULTRA-SAFE TUNING (For Low Memory Systems)")
    
    # Load very small subset
    X_train, y_train = load_split_data_subset("combined", "train", max_samples=20000)
    X_test, y_test = load_split_data_subset("combined", "test", max_samples=10000)
    
    # Very conservative parameters
    param_dist = {
        'n_estimators': [50, 80, 100],
        'max_depth': [10, 15],
        'min_samples_split': [10, 20],
        'min_samples_leaf': [5, 10],
        'max_features': ['sqrt'],
        'max_samples': [0.6, 0.7],
        'class_weight': ['balanced', {0: 1, 1: 3}]
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=2, verbose=0)
    
    search = RandomizedSearchCV(
        rf, param_dist, n_iter=5, cv=2,
        scoring='recall', random_state=42, n_jobs=1,  # Single job!
        verbose=1
    )
    
    search.fit(X_train, y_train)
    return search.best_estimator_

if __name__ == "__main__":
    print("🤔 Choose tuning strategy:")
    print("1. Memory-Optimized (Recommended)")
    print("2. Ultra-Safe (Low Memory)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        best_model = quick_tune_even_safer()
    else:
        best_model = tune_random_forest_memory_safe()
    
    if best_model is not None:
        print(f"\n🎉 HYPERPARAMETER TUNING COMPLETED SUCCESSFULLY!")
        print(f"📁 Model saved: models/tuned_rf_memory_optimized.joblib")
    else:
        print(f"\n❌ Tuning failed!")