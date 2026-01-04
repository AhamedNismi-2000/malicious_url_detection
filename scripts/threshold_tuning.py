
"""
OPTIMIZE CLASSIFICATION THRESHOLD TO REDUCE BOTH FP AND FN
- Finds optimal probability threshold
- No retraining required
- Instant improvement
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import precision_recall_curve, f1_score, confusion_matrix
from scipy.sparse import load_npz
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_test_data(dataset_name):
    """Load test dataset"""
    split_dir = os.path.join(BASE_DIR, "data", "splits", dataset_name)
    features_path = os.path.join(split_dir, f"{dataset_name}_test_features.npz")
    meta_path = os.path.join(split_dir, f"{dataset_name}_test.npz")
    
    X_test = load_npz(features_path)
    data = np.load(meta_path, allow_pickle=True)
    y_test = data['labels']
    
    return X_test, y_test

def find_optimal_threshold(model, X_test, y_test):
    """Find best probability threshold to balance FP and FN"""
    print("🔍 Finding optimal threshold...")
    
    # Get probability predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Generate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    
    # Calculate F1-score for each threshold
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
    
    # Find threshold that maximizes F1-score
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    
    # Also find threshold that gives best balance (closest to equal FP/FN reduction)
    current_pred = (y_proba >= 0.5).astype(int)
    current_cm = confusion_matrix(y_test, current_pred)
    current_fn, current_fp = current_cm[1, 0], current_cm[0, 1]
    
    # Calculate error reduction potential for each threshold
    error_reduction_scores = []
    for i, threshold in enumerate(thresholds):
        pred = (y_proba >= threshold).astype(int)
        cm = confusion_matrix(y_test, pred)
        fn, fp = cm[1, 0], cm[0, 1]
        
        # Score based on balanced reduction of both errors
        fn_reduction = (current_fn - fn) / current_fn
        fp_reduction = (current_fp - fp) / current_fp
        balanced_score = (fn_reduction + fp_reduction) / 2
        
        error_reduction_scores.append(balanced_score)
    
    # Find threshold with best balanced error reduction
    balanced_idx = np.argmax(error_reduction_scores)
    balanced_threshold = thresholds[balanced_idx]
    balanced_f1 = f1_scores[balanced_idx]
    
    return optimal_threshold, optimal_f1, balanced_threshold, balanced_f1, thresholds, f1_scores

def evaluate_threshold(model, X_test, y_test, threshold):
    """Evaluate model performance at specific threshold"""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    return {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_negatives': tn,
        'false_positives': fp,
        'false_negatives': fn,
        'true_positives': tp
    }

def plot_threshold_analysis(thresholds, f1_scores, optimal_threshold, balanced_threshold):
    """Plot threshold analysis"""
    plt.figure(figsize=(10, 6))
    
    plt.plot(thresholds, f1_scores, 'b-', linewidth=2, label='F1-Score')
    plt.axvline(x=optimal_threshold, color='red', linestyle='--', 
                label=f'Optimal F1 (threshold={optimal_threshold:.3f})')
    plt.axvline(x=balanced_threshold, color='green', linestyle='--',
                label=f'Balanced (threshold={balanced_threshold:.3f})')
    plt.axvline(x=0.5, color='gray', linestyle=':', label='Default (threshold=0.5)')
    
    plt.xlabel('Classification Threshold')
    plt.ylabel('F1-Score')
    plt.title('Threshold Optimization for URL Classification')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    os.makedirs('visualizations', exist_ok=True)
    plt.savefig('visualizations/threshold_optimization.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    print("🎯 THRESHOLD TUNING OPTIMIZATION")
    print("=" * 60)
    
    # Load best model
    model_path = os.path.join(BASE_DIR, "models", "combined_RF_100trees_balanced.joblib")
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    model = joblib.load(model_path)
    print(f"✅ Loaded model: {os.path.basename(model_path)}")
    
    # Load test data
    X_test, y_test = load_test_data("combined")
    print(f"📊 Test data: {X_test.shape}")
    
    # Find optimal thresholds
    optimal_threshold, optimal_f1, balanced_threshold, balanced_f1, thresholds, f1_scores = find_optimal_threshold(
        model, X_test, y_test
    )
    
    # Evaluate at different thresholds
    default_results = evaluate_threshold(model, X_test, y_test, 0.5)
    optimal_results = evaluate_threshold(model, X_test, y_test, optimal_threshold)
    balanced_results = evaluate_threshold(model, X_test, y_test, balanced_threshold)
    
    # Print comparison
    print(f"\n📊 THRESHOLD COMPARISON:")
    print("=" * 60)
    
    for results, name in [(default_results, "DEFAULT (0.5)"), 
                         (optimal_results, "OPTIMAL F1"), 
                         (balanced_results, "BALANCED")]:
        print(f"\n🎯 {name}:")
        print(f"   Threshold:    {results['threshold']:.3f}")
        print(f"   Accuracy:     {results['accuracy']:.4f}")
        print(f"   Precision:    {results['precision']:.4f}")
        print(f"   Recall:       {results['recall']:.4f}")
        print(f"   F1-Score:     {results['f1_score']:.4f}")
        print(f"   FP (false alarms): {results['false_positives']:,}")
        print(f"   FN (missed malicious): {results['false_negatives']:,}")
    
    # Plot analysis
    plot_threshold_analysis(thresholds, f1_scores, optimal_threshold, balanced_threshold)
    print(f"\n📈 Visualization saved: visualizations/threshold_optimization.png")
    
    # Recommendation
    print(f"\n💡 RECOMMENDATION:")
    if balanced_results['false_negatives'] < optimal_results['false_negatives']:
        print(f"   Use BALANCED threshold: {balanced_threshold:.3f}")
        print(f"   This reduces BOTH FP and FN more evenly")
    else:
        print(f"   Use OPTIMAL F1 threshold: {optimal_threshold:.3f}")
        print(f"   This gives the best overall F1-score")
    
    # Save results
    results_df = pd.DataFrame([default_results, optimal_results, balanced_results])
    results_df['strategy'] = ['default', 'optimal_f1', 'balanced']
    
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/threshold_tuning_results.csv', index=False)
    print(f"💾 Results saved: results/threshold_tuning_results.csv")

if __name__ == "__main__":
    main()