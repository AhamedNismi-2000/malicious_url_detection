#!/usr/bin/env python3
# scripts/split_dataset_final_fixed.py
"""
FINAL FIXED DATASET SPLITTING - HANDLES MISSING COMBINED.NPZ
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix, save_npz, load_npz, issparse, hstack
import warnings
warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DIR = os.path.join(BASE_DIR, "features")
SPLITS_DIR = os.path.join(BASE_DIR, "data", "splits")
os.makedirs(SPLITS_DIR, exist_ok=True)

# DATASETS WITH FALLBACK PATHS
DATASETS = {
    "heuristic": [
        os.path.join(FEATURES_DIR, "heuristic", "heuristic_features_enhanced.npz"),
        os.path.join(FEATURES_DIR, "heuristic_features_enhanced.npz")
    ],
    "nlp": [
        os.path.join(FEATURES_DIR, "nlp", "nlp_features_enhanced_aligned.npz"),
        os.path.join(FEATURES_DIR, "nlp_features_enhanced_aligned.npz")
    ],
    "combined": [
        os.path.join(FEATURES_DIR, "combined", "combined_features.npz"),
        os.path.join(FEATURES_DIR, "combined_features.npz"),
        os.path.join(FEATURES_DIR, "combined", "combined.npz")
    ]
}

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_STATE = 42

def find_dataset_path(dataset_name):
    """Find the actual path of the dataset with multiple fallbacks"""
    possible_paths = DATASETS[dataset_name]
    for path in possible_paths:
        if os.path.exists(path):
            print(f"   ✅ Found: {path}")
            return path
    print(f"   ❌ Not found in any location")
    return None

# ---------------- ROBUST DATA LOADING ----------------
def load_npz_smart(npz_path):
    """Smart NPZ loading that handles different file structures"""
    print(f"\n📥 Loading: {os.path.basename(npz_path)}")
    
    if not os.path.exists(npz_path):
        print(f"   ❌ File not found: {npz_path}")
        return None, None, None, None
    
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            available_keys = list(data.keys())
            print(f"   🔍 Available keys: {available_keys}")
            
            # Debug: Show what's inside
            for key in available_keys:
                array_data = data[key]
                if hasattr(array_data, 'shape'):
                    print(f"      {key:20}: shape={array_data.shape}, dtype={array_data.dtype}")
                else:
                    print(f"      {key:20}: value={array_data}")
            
            # Extract features
            features = extract_features_smart(data, available_keys)
            if features is None:
                return None, None, None, None
            
            # Extract labels
            labels = extract_labels_smart(data, available_keys)
            if labels is None:
                return None, None, None, None
            
            # Extract metadata - FIXED: Handle empty URLs array
            feature_names = data['feature_names'] if 'feature_names' in data else None
            
            # Handle URLs - if empty array, set to None
            if 'urls' in data:
                urls = data['urls']
                if hasattr(urls, 'shape') and urls.shape[0] == 0:
                    print("   ℹ️  URLs array is empty, setting to None")
                    urls = None
            else:
                urls = None
            
            # Convert to sparse for consistency
            if not issparse(features):
                features = csr_matrix(features)
            
            print(f"   ✅ Loaded: {features.shape[0]:,} samples, {features.shape[1]:,} features")
            return features, labels, feature_names, urls
            
    except Exception as e:
        print(f"   ❌ Error loading: {e}")
        return None, None, None, None

def extract_features_smart(data, available_keys):
    """Smart feature extraction with multiple strategies"""
    
    # Strategy 1: Direct features array
    if 'features' in available_keys:
        print("   ✅ Using 'features' array")
        return data['features']
    
    # Strategy 2: Sparse matrix components
    elif all(key in available_keys for key in ['data', 'indices', 'indptr', 'shape']):
        try:
            shape = tuple(data['shape'])
            features = csr_matrix((data['data'], data['indices'], data['indptr']), shape=shape)
            print("   ✅ Reconstructed sparse matrix from components")
            return features
        except Exception as e:
            print(f"   ❌ Sparse reconstruction failed: {e}")
    
    # Strategy 3: Look for any 2D array
    for key in available_keys:
        array_data = data[key]
        if hasattr(array_data, 'shape') and len(array_data.shape) == 2:
            # Skip obvious non-feature arrays
            if key not in ['labels', 'target', 'urls', 'feature_names']:
                print(f"   ✅ Using '{key}' as features")
                return array_data
    
    print("   ❌ No suitable features found")
    return None

def extract_labels_smart(data, available_keys):
    """Smart label extraction"""
    
    # Common label keys
    for key in ['labels', 'target', 'y']:
        if key in available_keys:
            labels = data[key]
            print(f"   ✅ Using '{key}' as labels: {len(np.unique(labels))} classes")
            return labels
    
    # Last resort: find 1D array with few unique values
    for key in available_keys:
        array_data = data[key]
        if hasattr(array_data, 'shape') and len(array_data.shape) == 1:
            unique_vals = np.unique(array_data)
            if 2 <= len(unique_vals) <= 10:  # Reasonable for classification
                print(f"   🔍 Using '{key}' as labels: {len(unique_vals)} classes")
                return array_data
    
    print("   ❌ No labels found")
    return None

# ---------------- CORE SPLITTING LOGIC ----------------
def split_dataset_perfect(dataset_path, dataset_name):
    """Perfect dataset splitting"""
    print(f"\n🎯 SPLITTING: {dataset_name.upper()}")
    print("=" * 60)
    
    # Load data
    features, labels, feature_names, urls = load_npz_smart(dataset_path)
    
    if features is None or labels is None:
        print(f"❌ Failed to load {dataset_name}")
        return None
    
    # Validate data - FIXED: Don't fail on URL mismatch if URLs is None
    if not validate_dataset_fixed(features, labels, urls):
        return None
    
    # Fix alignment if needed
    features, labels, urls = fix_data_alignment(features, labels, urls)
    
    print(f"✅ Ready: {features.shape[0]:,} samples, {features.shape[1]:,} features")
    
    # Perform perfect 80:10:10 split
    splits, final_feature_names = perfect_stratified_split(
        features, labels, urls, feature_names
    )
    
    if splits is None:
        return None
    
    # Show split analysis
    analyze_splits_perfect(splits, dataset_name)
    
    # Save splits
    dataset_dir = os.path.join(SPLITS_DIR, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    
    if save_splits_perfect(splits, final_feature_names, dataset_dir, dataset_name):
        save_split_summary_perfect(splits, dataset_name, dataset_dir)
        return splits
    
    return None

def validate_dataset_fixed(features, labels, urls):
    """Fixed validation - don't fail on URL issues"""
    if len(features.shape) != 2:
        print(f"❌ Features must be 2D, got {features.shape}")
        return False
    
    if features.shape[0] != len(labels):
        print(f"❌ Sample mismatch: features={features.shape[0]}, labels={len(labels)}")
        return False
    
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        print(f"❌ Need 2+ classes, found: {unique_labels}")
        return False
    
    # FIXED: Only check URL length if URLs is not None and not empty
    if urls is not None and len(urls) > 0 and len(urls) != len(labels):
        print(f"❌ URL mismatch: {len(urls)} vs {len(labels)}")
        return False
    
    print("   ✅ Data validation passed")
    return True

def fix_data_alignment(features, labels, urls):
    """Fix data alignment safely"""
    n_features = features.shape[0]
    n_labels = len(labels)
    
    if n_features == n_labels:
        return features, labels, urls
    
    print(f"🔄 Aligning: features={n_features}, labels={n_labels}")
    min_samples = min(n_features, n_labels)
    
    # Safe slicing for sparse matrices
    if issparse(features):
        features = features[:min_samples]
    else:
        features = features[:min_samples]
    
    labels = labels[:min_samples]
    
    # Only slice URLs if they exist and have data
    if urls is not None and len(urls) > 0:
        urls = urls[:min_samples]
    
    print(f"   ✅ Aligned to {min_samples} samples")
    return features, labels, urls

def perfect_stratified_split(features, labels, urls, feature_names):
    """Perfect 80:10:10 stratified split"""
    
    # Handle missing metadata
    if urls is None:
        urls = np.array([f"sample_{i}" for i in range(len(labels))])
    
    if feature_names is None:
        feature_names = np.array([f"feature_{i}" for i in range(features.shape[1])])
    
    try:
        # First split: 80% train + 20% temp
        X_train, X_temp, y_train, y_temp, url_train, url_temp = train_test_split(
            features, labels, urls, 
            test_size=0.2, 
            random_state=RANDOM_STATE,
            stratify=labels
        )
        
        # Split temp into 10% val + 10% test (50/50 split of the 20%)
        X_val, X_test, y_val, y_test, url_val, url_test = train_test_split(
            X_temp, y_temp, url_temp,
            test_size=0.5,  # Half of 20% = 10%
            random_state=RANDOM_STATE,
            stratify=y_temp
        )
        
        splits = {
            'train': (X_train, y_train, url_train),
            'val': (X_val, y_val, url_val), 
            'test': (X_test, y_test, url_test)
        }
        
        return splits, feature_names
        
    except Exception as e:
        print(f"❌ Split failed: {e}")
        return None, None

def analyze_splits_perfect(splits, dataset_name):
    """Perfect split analysis"""
    print(f"\n📊 {dataset_name.upper()} SPLIT ANALYSIS:")
    print("=" * 50)
    
    for split_name, (features, labels, urls) in splits.items():
        total = len(labels)
        benign = np.sum(labels == 0)
        malicious = np.sum(labels == 1)
        
        print(f"   {split_name:6}: {total:>6,} samples")
        print(f"           Benign:    {benign:>5,} ({(benign/total)*100:5.1f}%)")
        print(f"           Malicious: {malicious:>3,} ({(malicious/total)*100:5.1f}%)")
        print(f"           Features:  {features.shape[1]:>5,}")

def save_splits_perfect(splits, feature_names, output_dir, dataset_name):
    """Save splits perfectly - 3 NPZ files only"""
    success = True
    
    for split_name, (features, labels, urls) in splits.items():
        output_path = os.path.join(output_dir, f"{dataset_name}_{split_name}.npz")
        
        try:
            if issparse(features):
                # Save sparse matrix components
                np.savez_compressed(
                    output_path,
                    # Sparse matrix
                    features_data=features.data,
                    features_indices=features.indices,
                    features_indptr=features.indptr,
                    features_shape=np.array(features.shape),
                    # Metadata
                    labels=labels,
                    feature_names=feature_names,
                    urls=urls,
                    # Info
                    split_name=split_name,
                    dataset_name=dataset_name,
                    sparse_format='csr'
                )
            else:
                # Save dense array
                np.savez_compressed(
                    output_path,
                    features=features,
                    labels=labels,
                    feature_names=feature_names,
                    urls=urls,
                    split_name=split_name,
                    dataset_name=dataset_name,
                    sparse_format='dense'
                )
            
            print(f"   💾 {split_name}: {features.shape[0]:,} samples")
            
        except Exception as e:
            print(f"   ❌ Failed to save {split_name}: {e}")
            success = False
    
    return success

def save_split_summary_perfect(splits, dataset_name, output_dir):
    """Save perfect split summary"""
    summary_data = []
    
    for split_name, (features, labels, urls) in splits.items():
        total = len(labels)
        benign = np.sum(labels == 0)
        malicious = np.sum(labels == 1)
        
        summary_data.append({
            'dataset': dataset_name,
            'split': split_name,
            'total_samples': total,
            'benign_count': benign,
            'malicious_count': malicious,
            'benign_percentage': (benign / total) * 100,
            'malicious_percentage': (malicious / total) * 100,
            'feature_count': features.shape[1],
            'sparse_format': 'csr' if issparse(features) else 'dense'
        })
    
    df = pd.DataFrame(summary_data)
    csv_path = os.path.join(output_dir, f"{dataset_name}_split_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"   📈 Summary: {csv_path}")

def save_global_summary_perfect(all_splits):
    """Save perfect global summary"""
    global_data = []
    
    for dataset_name, splits in all_splits.items():
        for split_name, (features, labels, urls) in splits.items():
            global_data.append({
                'dataset': dataset_name,
                'split': split_name,
                'samples': len(labels),
                'features': features.shape[1],
                'benign': np.sum(labels == 0),
                'malicious': np.sum(labels == 1)
            })
    
    df = pd.DataFrame(global_data)
    global_path = os.path.join(SPLITS_DIR, "global_split_summary.csv")
    df.to_csv(global_path, index=False)
    print(f"📊 Global summary: {global_path}")

# ---------------- CREATE COMBINED DATASET IF MISSING ----------------
def create_combined_dataset_from_splits(heuristic_splits, nlp_splits):
    """Create combined dataset from existing splits if combined.npz is missing"""
    print(f"\n🔗 CREATING COMBINED DATASET FROM EXISTING SPLITS")
    print("=" * 50)
    
    if not heuristic_splits or not nlp_splits:
        print("❌ Need both heuristic and NLP splits to create combined dataset")
        return None
    
    combined_splits = {}
    combined_output_dir = os.path.join(SPLITS_DIR, "combined")
    os.makedirs(combined_output_dir, exist_ok=True)
    
    success = True
    
    for split_name in ['train', 'val', 'test']:
        # Get features from both datasets
        h_features, h_labels, h_urls = heuristic_splits[split_name]
        n_features, n_labels, n_urls = nlp_splits[split_name]
        
        # Validate alignment
        if len(h_labels) != len(n_labels) or not np.array_equal(h_labels, n_labels):
            print(f"❌ Label mismatch in {split_name} split")
            success = False
            continue
        
        # Combine features horizontally
        try:
            combined_features = hstack([h_features, n_features])
            
            # Create combined feature names
            heuristic_feature_names = [f"heuristic_{i}" for i in range(h_features.shape[1])]
            nlp_feature_names = [f"nlp_{i}" for i in range(n_features.shape[1])]
            combined_feature_names = np.array(heuristic_feature_names + nlp_feature_names)
            
            combined_splits[split_name] = (combined_features, h_labels, h_urls)
            
            # Save combined split
            output_path = os.path.join(combined_output_dir, f"combined_{split_name}.npz")
            
            # Save sparse matrix components
            np.savez_compressed(
                output_path,
                features_data=combined_features.data,
                features_indices=combined_features.indices,
                features_indptr=combined_features.indptr,
                features_shape=np.array(combined_features.shape),
                labels=h_labels,
                feature_names=combined_feature_names,
                urls=h_urls,
                split_name=split_name,
                dataset_name="combined",
                sparse_format='csr',
                heuristic_feature_count=h_features.shape[1],
                nlp_feature_count=n_features.shape[1]
            )
            
            print(f"   💾 combined_{split_name}: {combined_features.shape[0]:,} samples, {combined_features.shape[1]:,} features")
            
        except Exception as e:
            print(f"❌ Error combining {split_name}: {e}")
            success = False
            continue
    
    if success and combined_splits:
        analyze_splits_perfect(combined_splits, "combined")
        save_split_summary_perfect(combined_splits, "combined", combined_output_dir)
        print("✅ Combined dataset created successfully from splits")
        return combined_splits
    else:
        print("❌ Failed to create combined dataset")
        return None

# ---------------- MAIN EXECUTION ----------------
def main():
    print("🚀 MALICIOUS URL DETECTION - DATASET SPLITTING")
    print("🎯 FOR CHROMIUM EXTENSION + RANDOM FOREST")
    print("📊 SPLITTING: heuristic + nlp + combined")
    print("⚡ RATIO: 80% Train / 10% Validation / 10% Test")
    print("=" * 60)
    
    # Find available datasets with fallback paths
    available_datasets = {}
    for dataset_name in DATASETS.keys():
        print(f"\n🔍 Looking for {dataset_name}:")
        path = find_dataset_path(dataset_name)
        if path:
            available_datasets[dataset_name] = path
    
    if not available_datasets:
        print("❌ No datasets found!")
        return
    
    print(f"\n🎯 Processing {len(available_datasets)} datasets...")
    
    # Split available datasets
    all_splits = {}
    heuristic_splits = None
    nlp_splits = None
    
    for dataset_name, dataset_path in available_datasets.items():
        splits = split_dataset_perfect(dataset_path, dataset_name)
        if splits:
            all_splits[dataset_name] = splits
            if dataset_name == 'heuristic':
                heuristic_splits = splits
            elif dataset_name == 'nlp':
                nlp_splits = splits
    
    # If combined dataset is missing but we have both heuristic and NLP, create it
    if 'combined' not in all_splits and heuristic_splits and nlp_splits:
        print(f"\n🔄 Combined dataset not found, creating from heuristic + NLP splits...")
        combined_splits = create_combined_dataset_from_splits(heuristic_splits, nlp_splits)
        if combined_splits:
            all_splits['combined'] = combined_splits
    
    # Save global summary
    if all_splits:
        save_global_summary_perfect(all_splits)
    
    # Show final structure
    print("\n" + "=" * 60)
    print("🎉 SPLITTING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    show_final_structure()

def show_final_structure():
    """Show the perfect final structure"""
    print("\n📁 FINAL OUTPUT STRUCTURE:")
    print("data/splits/")
    
    # Check each dataset directory
    for dataset in ['heuristic', 'nlp', 'combined']:
        dataset_dir = os.path.join(SPLITS_DIR, dataset)
        if os.path.exists(dataset_dir):
            print(f"├── {dataset}/")
            # List all NPZ and CSV files
            files = [f for f in os.listdir(dataset_dir) 
                    if f.endswith('.npz') or f.endswith('.csv')]
            files.sort()
            
            for i, file in enumerate(files):
                if i == len(files) - 1:
                    print(f"│   └── {file}")
                else:
                    print(f"│   ├── {file}")
    
    # Global summary
    global_file = os.path.join(SPLITS_DIR, "global_split_summary.csv")
    if os.path.exists(global_file):
        print(f"└── global_split_summary.csv")
    
    print(f"\n✅ Ready for model training!")
    datasets_created = []
    for dataset in ['heuristic', 'nlp', 'combined']:
        if os.path.exists(os.path.join(SPLITS_DIR, dataset)):
            datasets_created.append(dataset)
    
    print(f"   - {', '.join(datasets_created)}: 3 NPZ files each (train/val/test)")
    print(f"   - All NPZ files contain features + metadata")

if __name__ == "__main__":
    main()