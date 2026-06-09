#!/usr/bin/env python3
"""
COMPREHENSIVE MODEL VALIDATION SCRIPT
- Tests multiple models on validation/test sets
- Compares performance across different algorithms
- Provides deployment recommendations
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import argparse
from datetime import datetime
from scipy.sparse import csr_matrix
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, precision_recall_curve)
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Fix encoding issues for Windows
sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf-8', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', encoding='utf-8', buffering=1)

# Config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_data(dataset="combined", split="val"):
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
    
    print(f"Loaded {split.upper()} set: {len(y):,} samples, {X.shape[1]:,} features")
    print(f"Class balance: {np.sum(y==0):,} benign ({np.mean(y==0):.1%}), "
          f"{np.sum(y==1):,} malicious ({np.mean(y==1):.1%})")
    
    return X, y

def load_all_models():
    """Load all trained models from models directory"""
    models_dir = os.path.join(BASE_DIR, "models")
    all_models = {}
    
    if not os.path.exists(models_dir):
        print(f"❌ Models directory not found: {models_dir}")
        return all_models
    
    # Look for model directories
    for model_type in os.listdir(models_dir):
        model_type_dir = os.path.join(models_dir, model_type)
        
        if not os.path.isdir(model_type_dir):
            continue
        
        # Look for .pkl files
        for file in os.listdir(model_type_dir):
            if file.endswith('.pkl') and file.startswith(('rf_', 'model_')):
                model_path = os.path.join(model_type_dir, file)
                
                try:
                    model = joblib.load(model_path)
                    model_info = {
                        'path': model_path,
                        'model': model,
                        'type': model_type,
                        'name': os.path.splitext(file)[0]
                    }
                    
                    # Try to load metadata
                    timestamp = None
                    if 'timestamp' in file:
                        import re
                        match = re.search(r'(\d{8}_\d{6})', file)
                        if match:
                            timestamp = match.group(1)
                    
                    if timestamp:
                        meta_json = os.path.join(model_type_dir, f"model_metadata_{timestamp}.json")
                        meta_pkl = os.path.join(model_type_dir, f"model_metadata_{timestamp}.pkl")
                        
                        if os.path.exists(meta_json):
                            with open(meta_json, 'r', encoding='utf-8') as f:
                                model_info['metadata'] = json.load(f)
                        elif os.path.exists(meta_pkl):
                            model_info['metadata'] = joblib.load(meta_pkl)
                    
                    all_models[model_info['name']] = model_info
                    print(f"✅ Loaded: {model_info['name']} ({model_type})")
                    
                except Exception as e:
                    print(f"❌ Failed to load {file}: {e}")
    
    return all_models

def evaluate_model(model, X, y, threshold=0.5, minority_class=1):
    """Evaluate a single model"""
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
    
    # Determine deployment readiness
    recall = metrics['recall']
    fpr = metrics['fpr']
    
    if recall >= 0.75 and fpr <= 0.03:
        deployment_ready = True
        recommendation = "Excellent - Ready for production"
    elif recall >= 0.65 and fpr <= 0.05:
        deployment_ready = True
        recommendation = "Good - Can deploy with monitoring"
    else:
        deployment_ready = False
        if recall < 0.65:
            recommendation = "Needs improvement - Detection rate too low"
        elif fpr > 0.05:
            recommendation = "Needs improvement - False alarm rate too high"
        else:
            recommendation = "Needs improvement - Review model performance"
    
    metrics.update({
        'deployment_ready': deployment_ready,
        'recommendation': recommendation
    })
    
    return metrics, y_proba, y_binary

def compare_models(models_dict, X, y, minority_class=1):
    """Compare multiple models"""
    results = []
    
    for model_name, model_info in models_dict.items():
        print(f"\nEvaluating: {model_name}")
        print("-" * 30)
        
        # Get threshold from metadata or use default
        threshold = 0.5
        if 'metadata' in model_info:
            threshold = model_info['metadata'].get('performance', {}).get('optimal_threshold', 0.5)
        
        try:
            metrics, y_proba, y_binary = evaluate_model(
                model_info['model'], X, y, threshold, minority_class
            )
            
            result = {
                'model_name': model_name,
                'model_type': model_info['type'],
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'auc': metrics['auc'],
                'tp': metrics['tp'],
                'fp': metrics['fp'],
                'fn': metrics['fn'],
                'tn': metrics['tn'],
                'fpr': metrics['fpr'],
                'fnr': metrics['fnr'],
                'threshold': threshold,
                'deployment_ready': metrics['deployment_ready'],
                'recommendation': metrics['recommendation']
            }
            
            results.append(result)
            
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1']:.4f}")
            print(f"  AUC-ROC:   {metrics['auc']:.4f}")
            print(f"  FPR:       {metrics['fpr']:.4f}")
            print(f"  Deployment: {'READY' if metrics['deployment_ready'] else 'NOT READY'}")
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
    
    return results

def save_validation_results(results, save_dir, dataset_name):
    """Save validation results to files"""
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert results to DataFrame
    df = pd.DataFrame(results)
    
    # Save CSV
    csv_path = os.path.join(save_dir, f"validation_metrics_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    
    # Save text report WITH UTF-8 ENCODING
    report_path = os.path.join(save_dir, f"validation_report_{timestamp}.txt")
    with open(report_path, 'w', encoding='utf-8') as f:  # FIXED: Added encoding='utf-8'
        f.write(f"Validation Report - {timestamp}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Total Models Tested: {len(results)}\n")
        f.write(f"Deployment Ready Models: {sum(r['deployment_ready'] for r in results)}\n\n")
        
        for _, row in df.iterrows():
            f.write(f"Model: {row['model_name']}\n")
            f.write(f"  Type: {row['model_type']}\n")
            f.write(f"  Accuracy:  {row['accuracy']:.4f}\n")
            f.write(f"  Precision: {row['precision']:.4f}\n")
            f.write(f"  Recall:    {row['recall']:.4f}\n")
            f.write(f"  F1-Score:  {row['f1']:.4f}\n")
            f.write(f"  AUC-ROC:   {row['auc']:.4f}\n")
            f.write(f"  FPR:       {row['fpr']:.4f}\n")
            f.write(f"  Threshold: {row['threshold']:.4f}\n")
            f.write(f"  Deployment Ready: {'✅ YES' if row['deployment_ready'] else '❌ NO'}\n")
            f.write(f"  Recommendation: {row['recommendation']}\n")
            f.write("\n")
    
    print(f"\n✅ Validation results saved:")
    print(f"   CSV: {csv_path}")
    print(f"   Report: {report_path}")
    
    return df

def create_comparison_plot(results_df, save_dir):
    """Create comparison visualization"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if len(results_df) == 0:
        print("❌ No results to plot")
        return None
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'auc', 'fpr']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'False Positive Rate']
    
    for idx, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Sort by metric value
        sorted_df = results_df.sort_values(by=metric, ascending=(metric == 'fpr'))
        
        # Create bar chart
        bars = ax.barh(sorted_df['model_name'], sorted_df[metric], 
                      color=['#2ecc71' if ready else '#e74c3c' 
                            for ready in sorted_df['deployment_ready']])
        
        ax.set_xlabel(name)
        ax.set_title(name)
        ax.set_xlim(0, 1 if metric != 'fpr' else max(0.1, sorted_df[metric].max() * 1.2))
        
        # Add value labels
        for bar, val in zip(bars, sorted_df[metric]):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', fontsize=9)
    
    plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_dir, f"model_comparison_{timestamp}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comparison plot saved: {plot_path}")
    return plot_path

def create_summary_report(results_df, dataset_name, save_dir):
    """Create executive summary report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if len(results_df) == 0:
        print("❌ No results for summary")
        return None
    
    # Find best model by F1 score
    best_by_f1 = results_df.loc[results_df['f1'].idxmax()]
    best_by_auc = results_df.loc[results_df['auc'].idxmax()]
    best_by_recall = results_df.loc[results_df['recall'].idxmax()]
    
    # Create summary
    summary = f"""
{'='*60}
MODEL VALIDATION EXECUTIVE SUMMARY
{'='*60}

Dataset: {dataset_name}
Validation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Models Tested: {len(results_df)}

OVERALL RESULTS:
----------------
• Models Deployment Ready: {sum(results_df['deployment_ready'])}/{len(results_df)}
• Average Recall: {results_df['recall'].mean():.3f}
• Average Precision: {results_df['precision'].mean():.3f}
• Average F1-Score: {results_df['f1'].mean():.3f}

BEST PERFORMING MODELS:
-----------------------
1. Best F1-Score ({best_by_f1['f1']:.3f}): {best_by_f1['model_name']}
   • Recall: {best_by_f1['recall']:.3f}, Precision: {best_by_f1['precision']:.3f}
   • Deployment: {'✅ READY' if best_by_f1['deployment_ready'] else '❌ NOT READY'}

2. Best AUC-ROC ({best_by_auc['auc']:.3f}): {best_by_auc['model_name']}
   • Recall: {best_by_auc['recall']:.3f}, FPR: {best_by_auc['fpr']:.3f}

3. Best Recall ({best_by_recall['recall']:.3f}): {best_by_recall['model_name']}
   • Precision: {best_by_recall['precision']:.3f}, F1: {best_by_recall['f1']:.3f}

DEPLOYMENT RECOMMENDATIONS:
---------------------------
"""
    
    # Add recommendations for each ready model
    ready_models = results_df[results_df['deployment_ready']]
    if len(ready_models) > 0:
        summary += "The following models are ready for deployment:\n"
        for _, model in ready_models.iterrows():
            summary += f"\n• {model['model_name']}:\n"
            summary += f"  - Recall: {model['recall']:.3f}, FPR: {model['fpr']:.3f}\n"
            summary += f"  - Recommendation: {model['recommendation']}\n"
    else:
        summary += "❌ No models are currently deployment ready.\n"
        summary += "Consider retraining with more data or adjusting thresholds.\n"
    
    summary += f"\n{'='*60}\n"
    
    # Save summary
    summary_path = os.path.join(save_dir, f"executive_summary_{timestamp}.txt")
    with open(summary_path, 'w', encoding='utf-8') as f:  # FIXED: Added encoding='utf-8'
        f.write(summary)
    
    print(f"✅ Executive summary saved: {summary_path}")
    return summary_path

def main():
    parser = argparse.ArgumentParser(description='Comprehensive Model Validation')
    parser.add_argument('--dataset', default='combined', help='Dataset name')
    parser.add_argument('--split', default='val', choices=['val', 'test'], 
                       help='Data split to use')
    parser.add_argument('--output', default='validation', 
                       help='Output directory name')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("COMPREHENSIVE MODEL VALIDATION")
    print("=" * 60)
    
    # 1. Load data
    try:
        X, y = load_data(args.dataset, args.split)
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return
    
    # 2. Load all models
    print(f"\n📦 Loading models...")
    models = load_all_models()
    
    if not models:
        print("❌ No models found to validate")
        return
    
    print(f"✅ Loaded {len(models)} models")
    
    # 3. Evaluate all models
    print(f"\n🧪 Evaluating models on {args.split.upper()} set...")
    results = compare_models(models, X, y)
    
    if not results:
        print("❌ No results generated")
        return
    
    # 4. Save results
    save_dir = os.path.join(BASE_DIR, "results","reports", args.output)
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # 5. Save files
    try:
        # Save CSV and text report
        save_validation_results(results, save_dir, args.dataset)
        
        # Create visualizations
        create_comparison_plot(results_df, save_dir)
        
        # Create executive summary
        create_summary_report(results_df, args.dataset, save_dir)
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")
    
    # 6. Print final summary
    print(f"\n{'='*60}")
    print("VALIDATION COMPLETE")
    print("=" * 60)
    
    # Count deployment ready models
    ready_count = sum(1 for r in results if r['deployment_ready'])
    
    print(f"\n📊 Summary Statistics:")
    print(f"   Total Models Tested: {len(results)}")
    print(f"   Deployment Ready: {ready_count} ({ready_count/len(results)*100:.0f}%)")
    
    if ready_count > 0:
        print(f"\n✅ DEPLOYMENT RECOMMENDATIONS:")
        for result in results:
            if result['deployment_ready']:
                print(f"   • {result['model_name']}: {result['recommendation']}")
    else:
        print(f"\n❌ NO MODELS READY FOR DEPLOYMENT")
        print(f"   Consider retraining with:")
        print(f"   1. More balanced dataset")
        print(f"   2. Different feature engineering")
        print(f"   3. Adjusted classification thresholds")

if __name__ == "__main__":
    main()