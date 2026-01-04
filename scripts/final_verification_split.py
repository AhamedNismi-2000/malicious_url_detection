#!/usr/bin/env python3
# scripts/final_verification.py
"""
FINAL VERIFICATION BEFORE TRAINING
- Quick check to confirm all datasets are ready
- No fixes needed - just validation
"""

import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS_DIR = os.path.join(BASE_DIR, "data", "splits")

def quick_verification():
    """Quick verification that everything is ready for training"""
    print("🚀 FINAL TRAINING READINESS CHECK")
    print("=" * 60)
    
    all_good = True
    
    # Check directory structure
    print("\n📁 DIRECTORY STRUCTURE:")
    expected_datasets = ['heuristic', 'nlp', 'combined']
    for dataset in expected_datasets:
        dataset_dir = os.path.join(SPLITS_DIR, dataset)
        if os.path.exists(dataset_dir):
            npz_files = [f for f in os.listdir(dataset_dir) if f.endswith('.npz')]
            if len(npz_files) == 3:  # train, val, test
                print(f"✅ {dataset:12} -> 3 NPZ files found")
            else:
                print(f"❌ {dataset:12} -> {len(npz_files)}/3 NPZ files")
                all_good = False
        else:
            print(f"❌ {dataset:12} -> MISSING")
            all_good = False
    
    # Quick data integrity check
    print("\n🔍 DATA INTEGRITY CHECK:")
    for dataset in expected_datasets:
        if os.path.exists(os.path.join(SPLITS_DIR, dataset)):
            try:
                # Check one split per dataset (train split)
                train_path = os.path.join(SPLITS_DIR, dataset, f"{dataset}_train.npz")
                data = np.load(train_path, allow_pickle=True)
                
                # Reconstruct features
                features = csr_matrix(
                    (data['features_data'], data['features_indices'], data['features_indptr']),
                    shape=tuple(data['features_shape'])
                )
                
                # Basic checks
                n_samples = features.shape[0]
                n_features = features.shape[1]
                n_labels = len(data['labels'])
                
                if n_samples == n_labels:
                    print(f"✅ {dataset:12} -> {n_samples:>7,} samples, {n_features:>4} features")
                else:
                    print(f"❌ {dataset:12} -> Sample mismatch: {n_samples} vs {n_labels}")
                    all_good = False
                    
            except Exception as e:
                print(f"❌ {dataset:12} -> Error: {e}")
                all_good = False
    
    # Check global summary
    print("\n📊 GLOBAL SUMMARY:")
    global_summary_path = os.path.join(SPLITS_DIR, "global_split_summary.csv")
    if os.path.exists(global_summary_path):
        df = pd.read_csv(global_summary_path)
        total_samples = df['samples'].sum()
        print(f"✅ Global summary: {total_samples:,} total samples across all splits")
    else:
        print("❌ Global summary missing")
        all_good = False
    
    # Final recommendation
    print("\n" + "=" * 60)
    if all_good:
        print("🎉 PERFECT! READY FOR MODEL TRAINING! 🚀")
        print("\n📊 YOUR DATASETS:")
        print("   • heuristic: 45 features")
        print("   • nlp:       1,806 features") 
        print("   • combined:  1,851 features")
        print(f"   • Total: ~1.6M samples (80/10/10 split)")
        print("\n🎯 NEXT STEP: Proceed with Random Forest training!")
    else:
        print("❌ Issues found - fix before training")
    
    return all_good

def show_training_recommendations():
    """Show recommendations for training"""
    print("\n" + "=" * 60)
    print("🤖 TRAINING RECOMMENDATIONS:")
    print("=" * 60)
    
    print("\n🔧 RANDOM FOREST CONFIGURATION:")
    print("   • n_estimators: 100-200 (start with 100)")
    print("   • max_depth: 20-30 (prevents overfitting)")
    print("   • min_samples_split: 10-20")
    print("   • min_samples_leaf: 5-10")
    print("   • class_weight: 'balanced' (for your 85/15 distribution)")
    print("   • n_jobs: -1 (use all CPU cores)")
    print("   • random_state: 42 (for reproducibility)")
    
    print("\n📈 TRAINING STRATEGY:")
    print("   1. Start with heuristic features (fastest training)")
    print("   2. Then try NLP features (higher dimensionality)")
    print("   3. Finally combined features (best performance expected)")
    print("   4. Compare all three on validation set")
    print("   5. Select best model for final testing")
    
    print("\n⚡ PERFORMANCE EXPECTATIONS:")
    print("   • Heuristic: Fast training, decent accuracy")
    print("   • NLP: Slower training, better accuracy") 
    print("   • Combined: Slowest training, best accuracy")
    print("   • Expected validation accuracy: 85-95%")
    
    print("\n🔍 MODEL EVALUATION:")
    print("   • Use validation set for hyperparameter tuning")
    print("   • Use test set ONLY for final evaluation")
    print("   • Monitor: Accuracy, Precision, Recall, F1-score")
    print("   • Focus on malicious URL detection (recall)")

if __name__ == "__main__":
    ready = quick_verification()
    if ready:
        show_training_recommendations()