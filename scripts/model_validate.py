#!/usr/bin/env python3
"""
COMPATIBLE ENHANCED VALIDATION CODE
- Specifically for validation set (re-evaluates after training)
- Compatible with optimized training output
- Business-focused validation metrics
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import argparse
from scipy.sparse import csr_matrix
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, average_precision_score
)
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_validation_data(dataset="combined"):
    """Load validation data with same format as training"""
    print(f"📊 Loading {dataset} validation data...")
    
    split_dir = os.path.join(BASE_DIR, "data", "splits", dataset)
    val_path = os.path.join(split_dir, f"{dataset}_val.npz")
    
    if not os.path.exists(val_path):
        print(f"❌ Validation data not found: {val_path}")
        return None, None
    
    try:
        val_data = np.load(val_path, allow_pickle=True)
        
        # Reconstruct sparse matrix (same as training)
        def reconstruct_features(data):
            if all(key in data for key in ['features_data', 'features_indices', 'features_indptr', 'features_shape']):
                return csr_matrix(
                    (data['features_data'], data['features_indices'], data['features_indptr']),
                    shape=tuple(data['features_shape'])
                )
            elif 'features' in data:
                return csr_matrix(data['features'])
            else:
                raise ValueError("Unknown data format")
        
        X_val = reconstruct_features(val_data)
        y_val = val_data['labels']
        
        # Convert to dense if manageable
        if X_val.shape[0] * X_val.shape[1] < 50000000:  # 50M elements
            X_val = X_val.toarray()
            print("✅ Converted to dense array")
        
        # Class distribution
        unique, counts = np.unique(y_val, return_counts=True)
        total = len(y_val)
        
        print(f"✅ Validation data loaded: {total:,} samples, {X_val.shape[1]:,} features")
        print(f"📈 Class distribution: Benign={counts[0]:,} ({100*counts[0]/total:.1f}%), "
              f"Malicious={counts[1]:,} ({100*counts[1]/total:.1f}%)")
        
        return X_val, y_val
        
    except Exception as e:
        print(f"❌ Error loading validation data: {e}")
        return None, None

def load_model_with_json_metadata(model_path):
    """Load model and JSON metadata (compatible with optimized training)"""
    print(f"📦 Loading model: {os.path.basename(model_path)}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return None, None, None
    
    try:
        # Load model
        model = joblib.load(model_path)
        print(f"✅ Model loaded: {type(model).__name__}")
        
        # Extract timestamp from model name
        import re
        timestamp_match = re.search(r'(\d{8}_\d{6})', os.path.basename(model_path))
        timestamp = timestamp_match.group(1) if timestamp_match else None
        
        metadata = {}
        model_dir = os.path.dirname(model_path)
        
        # Try to load JSON metadata (optimized training saves JSON)
        if timestamp:
            json_meta_path = os.path.join(model_dir, f"model_metadata_{timestamp}.json")
            if os.path.exists(json_meta_path):
                with open(json_meta_path, 'r') as f:
                    metadata = json.load(f)
                print(f"✅ JSON metadata loaded")
            else:
                # Try pickle metadata for backward compatibility
                pkl_meta_path = os.path.join(model_dir, f"model_metadata_{timestamp}.pkl")
                if os.path.exists(pkl_meta_path):
                    metadata = joblib.load(pkl_meta_path)
                    print(f"✅ Pickle metadata loaded")
        
        # Default if no metadata found
        if not metadata:
            print("⚠️  No metadata found. Using default settings.")
            metadata = {
                'performance': {'optimal_threshold': 0.5},
                'model_info': {'minority_class': 1},
                'training_info': {'training_date': 'Unknown'}
            }
        
        # Get minority class (critical for validation)
        minority_class = metadata.get('model_info', {}).get('minority_class', 1)
        print(f"   Minority class: {minority_class}")
        print(f"   Training date: {metadata.get('training_info', {}).get('training_date', 'Unknown')}")
        
        return model, metadata, minority_class
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, None

def validate_single_model(model, metadata, minority_class, X_val, y_val):
    """Validate a single model with comprehensive metrics"""
    
    # Get threshold from metadata
    threshold = metadata.get('performance', {}).get('optimal_threshold', 0.5)
    
    print(f"\n🔍 Validating with threshold: {threshold:.4f}")
    
    # Get probabilities for minority class
    y_proba = model.predict_proba(X_val)[:, minority_class]
    
    # Convert y_val to binary for minority class
    y_val_binary = (y_val == minority_class).astype(int)
    
    # Apply threshold
    y_pred = (y_proba >= threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_val_binary, y_pred)
    precision = precision_score(y_val_binary, y_pred, zero_division=0)
    recall = recall_score(y_val_binary, y_pred, zero_division=0)
    f1 = f1_score(y_val_binary, y_pred, zero_division=0)
    auc_score = roc_auc_score(y_val_binary, y_proba)
    avg_precision = average_precision_score(y_val_binary, y_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_val_binary, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate rates
    benign_samples = np.sum(y_val_binary == 0)
    malicious_samples = np.sum(y_val_binary == 1)
    
    fpr = fp / benign_samples if benign_samples > 0 else 0
    fnr = fn / malicious_samples if malicious_samples > 0 else 0
    detection_rate = tp / malicious_samples if malicious_samples > 0 else 0
    
    # Business metrics
    business_score = recall * 0.6 + (1 - fpr) * 0.4
    deployment_ready = recall >= 0.70 and fpr <= 0.03
    
    # Test alternative thresholds
    thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
    best_alt_threshold = threshold
    best_alt_score = business_score
    
    for alt_threshold in thresholds:
        alt_pred = (y_proba >= alt_threshold).astype(int)
        alt_recall = recall_score(y_val_binary, alt_pred, zero_division=0)
        alt_cm = confusion_matrix(y_val_binary, alt_pred)
        alt_tn, alt_fp, alt_fn, alt_tp = alt_cm.ravel()
        alt_fpr = alt_fp / benign_samples if benign_samples > 0 else 0
        alt_score = alt_recall * 0.6 + (1 - alt_fpr) * 0.4
        
        if alt_score > best_alt_score:
            best_alt_score = alt_score
            best_alt_threshold = alt_threshold
    
    # Compile results
    results = {
        'model_name': os.path.basename(model_path) if 'model_path' in locals() else 'Unknown',
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc_roc': auc_score,
        'avg_precision': avg_precision,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn,
        'false_positive_rate': fpr,
        'false_negative_rate': fnr,
        'detection_rate': detection_rate,
        'business_score': business_score,
        'deployment_ready': deployment_ready,
        'suggested_threshold': best_alt_threshold,
        'suggested_business_score': best_alt_score,
        'n_estimators': model.n_estimators if hasattr(model, 'n_estimators') else 'N/A',
        'n_features': model.n_features_in_ if hasattr(model, 'n_features_in_') else 'N/A',
        'training_date': metadata.get('training_info', {}).get('training_date', 'Unknown'),
        'imbalance_ratio': metadata.get('dataset_info', {}).get('imbalance_ratio', 'N/A')
    }
    
    # Print results
    print(f"   📈 Performance:")
    print(f"     Recall:        {recall:.4f} (Detection Rate)")
    print(f"     Precision:     {precision:.4f}")
    print(f"     F1-Score:      {f1:.4f}")
    print(f"     AUC-ROC:       {auc_score:.4f}")
    
    print(f"\n   ⚠️  Error Metrics:")
    print(f"     False Positive: {fpr:.4f} ({fp}/{benign_samples})")
    print(f"     False Negative: {fnr:.4f} ({fn}/{malicious_samples})")
    
    print(f"\n   💡 Business Impact:")
    print(f"     Detection Rate: {detection_rate:.1%}")
    print(f"     Business Score: {business_score:.3f}")
    
    if best_alt_threshold != threshold:
        print(f"\n   🎯 Suggestion: Try threshold {best_alt_threshold:.3f}")
        print(f"     Business score improvement: +{best_alt_score - business_score:.3f}")
    
    print(f"\n   🚀 Deployment: {'✅ READY' if deployment_ready else '❌ NOT READY'}")
    
    return results

def validate_all_models(models_dir):
    """Validate all models in directory"""
    print(f"\n🔍 Searching for models in: {models_dir}")
    
    if not os.path.exists(models_dir):
        print(f"❌ Directory not found: {models_dir}")
        return []
    
    # Find model files
    model_files = [f for f in os.listdir(models_dir) 
                  if f.endswith('.pkl') and f.startswith('rf_enhanced')]
    
    if not model_files:
        print("❌ No enhanced models found")
        return []
    
    print(f"📁 Found {len(model_files)} model(s)")
    
    all_results = []
    
    for model_file in sorted(model_files):
        print(f"\n{'='*60}")
        model_path = os.path.join(models_dir, model_file)
        
        # Load model and metadata
        model, metadata, minority_class = load_model_with_json_metadata(model_path)
        if model is None:
            continue
        
        # Validate
        results = validate_single_model(model, metadata, minority_class, X_val, y_val)
        all_results.append(results)
    
    return all_results

def save_validation_results(results, save_dir, dataset):
    """Save validation results to files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    if df.empty:
        print("❌ No validation results to save")
        return
    
    # Sort by business score (higher is better)
    df = df.sort_values('business_score', ascending=False)
    
    # Save CSV
    csv_path = os.path.join(save_dir, f"validation_results_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    
    # Generate summary
    summary_path = os.path.join(save_dir, f"validation_summary_{timestamp}.txt")
    
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("MODEL VALIDATION SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Validation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Models Validated: {len(results)}\n\n")
        
        f.write("TOP 3 MODELS:\n")
        f.write("-" * 40 + "\n")
        
        for i, (_, row) in enumerate(df.head(3).iterrows(), 1):
            f.write(f"\n{i}. {row['model_name']}\n")
            f.write(f"   Business Score: {row['business_score']:.3f}\n")
            f.write(f"   Recall: {row['recall']:.4f}, FPR: {row['false_positive_rate']:.4f}\n")
            f.write(f"   F1: {row['f1_score']:.4f}, AUC: {row['auc_roc']:.4f}\n")
            f.write(f"   Deployment Ready: {'✅' if row['deployment_ready'] else '❌'}\n")
            f.write(f"   Suggested Threshold: {row['suggested_threshold']:.3f}\n")
        
        f.write(f"\nDEPLOYMENT RECOMMENDATION:\n")
        f.write("-" * 40 + "\n")
        
        ready_models = df[df['deployment_ready']]
        if len(ready_models) > 0:
            best_model = ready_models.iloc[0]
            f.write(f"\n✅ RECOMMENDED FOR DEPLOYMENT:\n")
            f.write(f"   Model: {best_model['model_name']}\n")
            f.write(f"   Detection Rate: {best_model['recall']:.1%}\n")
            f.write(f"   False Alarm Rate: {best_model['false_positive_rate']:.1%}\n")
            f.write(f"   Threshold: {best_model['threshold']:.3f}\n")
            f.write(f"   Business Score: {best_model['business_score']:.3f}\n")
        else:
            f.write(f"\n❌ NO MODELS READY FOR DEPLOYMENT\n")
            f.write(f"   Best candidate: {df.iloc[0]['model_name']}\n")
            f.write(f"   Issues: Recall={df.iloc[0]['recall']:.1%} (<70%) or FPR={df.iloc[0]['false_positive_rate']:.1%} (>3%)\n")
            f.write(f"   Suggested improvement: Try threshold {df.iloc[0]['suggested_threshold']:.3f}\n")
    
    print(f"\n💾 Results saved:")
    print(f"   CSV: {csv_path}")
    print(f"   Summary: {summary_path}")
    
    return df

def main():
    parser = argparse.ArgumentParser(description='Enhanced Model Validation')
    parser.add_argument('--model', type=str, help='Path to specific model to validate')
    parser.add_argument('--all', action='store_true', help='Validate all models in directory')
    parser.add_argument('--dataset', type=str, default='combined', help='Dataset name')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("🧪 ENHANCED MODEL VALIDATION")
    print("   Compatible with optimized training output")
    print("=" * 70 + "\n")
    
    # Load validation data
    X_val, y_val = load_validation_data(args.dataset)
    if X_val is None:
        return
    
    # Determine which models to validate
    if args.model:
        # Validate specific model
        if not os.path.exists(args.model):
            print(f"❌ Model not found: {args.model}")
            return
        
        model, metadata, minority_class = load_model_with_json_metadata(args.model)
        if model is None:
            return
        
        results = [validate_single_model(model, metadata, minority_class, X_val, y_val)]
        
    elif args.all:
        # Validate all models in enhanced directory
        enhanced_dir = os.path.join(BASE_DIR, "models", "random_forest_enhanced")
        results = validate_all_models(enhanced_dir)
        
        # Also check legacy directory if enhanced is empty
        if not results:
            legacy_dir = os.path.join(BASE_DIR, "models", "random_forest")
            results = validate_all_models(legacy_dir)
    else:
        # Validate latest model
        enhanced_dir = os.path.join(BASE_DIR, "models", "random_forest_enhanced")
        if not os.path.exists(enhanced_dir):
            enhanced_dir = os.path.join(BASE_DIR, "models", "random_forest")
        
        if not os.path.exists(enhanced_dir):
            print(f"❌ Model directory not found: {enhanced_dir}")
            return
        
        model_files = [f for f in os.listdir(enhanced_dir) 
                      if f.endswith('.pkl') and f.startswith('rf_enhanced')]
        
        if not model_files:
            print("❌ No trained models found")
            return
        
        latest_model = sorted(model_files)[-1]
        model_path = os.path.join(enhanced_dir, latest_model)
        
        model, metadata, minority_class = load_model_with_json_metadata(model_path)
        if model is None:
            return
        
        results = [validate_single_model(model, metadata, minority_class, X_val, y_val)]
    
    # Save results
    if results:
        save_dir = os.path.join(BASE_DIR, "results", "reports", "validation")
        df = save_validation_results(results, save_dir, args.dataset)
        
        # Final recommendation
        print(f"\n🎯 FINAL VALIDATION SUMMARY:")
        print(f"=" * 50)
        
        if not df.empty:
            best_model = df.iloc[0]
            if best_model['deployment_ready']:
                print(f"✅ RECOMMENDED FOR DEPLOYMENT: {best_model['model_name']}")
                print(f"   Detection: {best_model['recall']:.1%}, False Alarms: {best_model['false_positive_rate']:.1%}")
            else:
                print(f"⚠️  BEST MODEL NEEDS IMPROVEMENT: {best_model['model_name']}")
                print(f"   Detection: {best_model['recall']:.1%}, False Alarms: {best_model['false_positive_rate']:.1%}")
                print(f"   Try threshold {best_model['suggested_threshold']:.3f} for better balance")
    else:
        print("\n❌ No models were successfully validated")

if __name__ == "__main__":
    main()