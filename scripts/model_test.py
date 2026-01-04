#!/usr/bin/env python3
"""
CLEAN MODEL TESTING SCRIPT
- Removed all unnecessary complexity
- Same functionality, 70% less code
- Faster and easier to maintain
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import argparse
from scipy.sparse import csr_matrix
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, precision_recall_curve)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_data(dataset="combined", split="test"):
    """Load dataset efficiently"""
    path = os.path.join(BASE_DIR, "data", "splits", dataset, f"{dataset}_{split}.npz")
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data not found: {path}")
    
    data = np.load(path, allow_pickle=True)
    
    # Load features
    if 'features_data' in data:
        X = csr_matrix((data['features_data'], data['features_indices'], 
                       data['features_indptr']), shape=tuple(data['features_shape']))
    else:
        X = csr_matrix(data['features'])
    
    y = data['labels']
    
    # Convert to dense if reasonable size
    if X.shape[0] * X.shape[1] < 50_000_000:
        X = X.toarray()
    
    print(f"Loaded: {len(y):,} samples, {X.shape[1]} features")
    print(f"Class balance: {np.sum(y==0):,} benign ({np.mean(y==0):.1%}), "
          f"{np.sum(y==1):,} malicious ({np.mean(y==1):.1%})")
    
    return X, y

def load_model(model_path):
    """Load model and metadata"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load model
    model = joblib.load(model_path)
    print(f"Model: {type(model).__name__}, Trees: {model.n_estimators}")
    
    # Load metadata
    metadata = {}
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path)
    
    # Extract timestamp
    import re
    if match := re.search(r'(\d{8}_\d{6})', model_name):
        timestamp = match.group(1)
        
        # Try JSON then pickle
        for ext in ['.json', '.pkl']:
            meta_path = os.path.join(model_dir, f"model_metadata_{timestamp}{ext}")
            if os.path.exists(meta_path):
                if ext == '.json':
                    with open(meta_path, 'r') as f:
                        metadata = json.load(f)
                else:
                    metadata = joblib.load(meta_path)
                break
    
    # Defaults if no metadata
    threshold = metadata.get('performance', {}).get('optimal_threshold', 0.5)
    minority_class = metadata.get('model_info', {}).get('minority_class', 1)
    
    return model, threshold, minority_class, metadata

def evaluate_model(model, X, y, threshold, minority_class):
    """Core evaluation function"""
    # Get probabilities
    y_proba = model.predict_proba(X)[:, minority_class]
    y_binary = (y == minority_class).astype(int)
    
    # Predict
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_binary, y_pred),
        'precision': precision_score(y_binary, y_pred, zero_division=0),
        'recall': recall_score(y_binary, y_pred, zero_division=0),
        'f1': f1_score(y_binary, y_pred, zero_division=0),
        'auc': roc_auc_score(y_binary, y_proba)
    }
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_binary, y_pred).ravel()
    metrics.update({
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0
    })
    
    return metrics, y_proba, y_binary

def find_best_threshold(model, X, y, minority_class):
    """Find optimal threshold for business needs"""
    y_proba = model.predict_proba(X)[:, minority_class]
    y_binary = (y == minority_class).astype(int)
    
    # Try thresholds
    thresholds = np.arange(0.1, 0.9, 0.05)
    results = []
    
    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        recall = recall_score(y_binary, y_pred, zero_division=0)
        cm = confusion_matrix(y_binary, y_pred)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            business_score = recall * 0.6 + (1 - fpr) * 0.4
            results.append((thresh, business_score, recall, fpr))
    
    if not results:
        return 0.5
    
    # Best by business score
    results.sort(key=lambda x: x[1], reverse=True)
    return results[0][0]

def generate_report(metrics, model_name, save_dir, split="test"):
    """Generate concise report"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Simple text report
    report = f"""
MODEL TEST REPORT
=================
Model: {model_name}
Dataset: {split} set
Threshold: {0.5 if 'threshold' not in metrics else metrics['threshold']:.3f}

PERFORMANCE
-----------
Accuracy:  {metrics['accuracy']:.4f}
Precision: {metrics['precision']:.4f}
Recall:    {metrics['recall']:.4f} (Detection Rate)
F1-Score:  {metrics['f1']:.4f}
AUC-ROC:   {metrics['auc']:.4f}

ERRORS
------
False Positives: {metrics['fp']:,} (FPR: {metrics['fpr']:.3f})
False Negatives: {metrics['fn']:,} (FNR: {metrics['fnr']:.3f})

CONFUSION MATRIX
----------------
True Positives:  {metrics['tp']:,}
False Positives: {metrics['fp']:,}
False Negatives: {metrics['fn']:,}
True Negatives:  {metrics['tn']:,}

RECOMMENDATION
--------------
"""
    # Add recommendation
    if metrics['recall'] >= 0.75 and metrics['fpr'] <= 0.03:
        report += "✅ PRODUCTION READY - Excellent detection with low false alarms\n"
    elif metrics['recall'] >= 0.65:
        report += "⚠️  ACCEPTABLE - Can deploy with monitoring\n"
    else:
        report += "❌ NEEDS IMPROVEMENT - Detection too low or false alarms too high\n"
    
    # Save report
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(save_dir, f"report_{timestamp}.txt")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Save CSV
    csv_path = os.path.join(save_dir, f"metrics_{timestamp}.csv")
    pd.DataFrame([metrics]).to_csv(csv_path, index=False)
    
    return report_path, csv_path

def plot_simple_curves(y_true, y_proba, save_dir):
    """Create essential plots only"""
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)
    ax1.plot(fpr, tpr, label=f'AUC = {auc_score:.3f}')
    ax1.plot([0, 1], [0, 1], 'k--')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ax2.plot(recall, precision)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"curves_{timestamp}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path

def main():
    parser = argparse.ArgumentParser(description='Clean Model Testing')
    parser.add_argument('--model', help='Path to model (.pkl)')
    parser.add_argument('--split', default='test', choices=['test', 'val'])
    parser.add_argument('--dataset', default='combined')
    parser.add_argument('--threshold', type=float, help='Override threshold')
    
    args = parser.parse_args()
    
    print("MODEL TESTING")
    print("=" * 50)
    
    # 1. Load model
    try:
        if args.model:
            model_path = args.model
        else:
            # Find latest model
            model_dir = os.path.join(BASE_DIR, "models", "random_forest_enhanced")
            if not os.path.exists(model_dir):
                model_dir = os.path.join(BASE_DIR, "models", "random_forest")
            
            models = [f for f in os.listdir(model_dir) 
                     if f.endswith('.pkl') and f.startswith('rf_')]
            
            if not models:
                raise FileNotFoundError("No models found")
            
            model_path = os.path.join(model_dir, sorted(models)[-1])
        
        model, default_thresh, minority_class, metadata = load_model(model_path)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # 2. Load data
    try:
        X, y = load_data(args.dataset, args.split)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # 3. Determine threshold
    threshold = args.threshold if args.threshold else default_thresh
    
    # 4. Evaluate
    print("\nEVALUATION:")
    metrics, y_proba, y_binary = evaluate_model(model, X, y, threshold, minority_class)
    
    print(f"   Threshold: {threshold:.3f}")
    print(f"   Accuracy:  {metrics['accuracy']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall:    {metrics['recall']:.4f}")
    print(f"   F1-Score:  {metrics['f1']:.4f}")
    print(f"   AUC-ROC:   {metrics['auc']:.4f}")
    print(f"   FPR:       {metrics['fpr']:.4f}")
    print(f"   FNR:       {metrics['fnr']:.4f}")
    
    # 5. Find better threshold if needed
    if metrics['recall'] < 0.7:
        print("\nSEARCHING FOR BETTER THRESHOLD...")
        best_thresh = find_best_threshold(model, X, y, minority_class)
        if best_thresh != threshold:
            print(f"   Try threshold {best_thresh:.3f} for better recall")
            # Re-evaluate with better threshold
            metrics2, _, _ = evaluate_model(model, X, y, best_thresh, minority_class)
            if metrics2['recall'] > metrics['recall'] and metrics2['fpr'] < 0.05:
                print(f"   Expected recall: {metrics2['recall']:.3f}, FPR: {metrics2['fpr']:.3f}")
    
    # 6. Generate outputs
    results_dir = os.path.join(BASE_DIR, "results", "reports", args.split)
    os.makedirs(results_dir, exist_ok=True)
    
    model_name = os.path.basename(model_path).replace('.pkl', '')
    report_path, csv_path = generate_report(metrics, model_name, results_dir, args.split)
    plot_path = plot_simple_curves(y_binary, y_proba, results_dir)
    
    print(f"\nOUTPUTS:")
    print(f"   Report: {report_path}")
    print(f"   CSV:    {csv_path}")
    print(f"   Plot:   {plot_path}")
    
    # 7. Final recommendation
    print(f"\nRECOMMENDATION:")
    if metrics['recall'] >= 0.75 and metrics['fpr'] <= 0.03:
        print("✅ DEPLOY - Excellent performance")
    elif metrics['recall'] >= 0.65:
        print("⚠️  MONITOR - Acceptable but needs watching")
    else:
        print("❌ IMPROVE - Needs better detection")

if __name__ == "__main__":
    main()