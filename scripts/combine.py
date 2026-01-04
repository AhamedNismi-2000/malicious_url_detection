#!/usr/bin/env python3
# scripts/combine_features.py
"""
COMBINE HEURISTIC AND NLP FEATURES - WORKING VERSION
- Loads heuristic features from CSV
- Loads NLP features from NPZ  
- Combines them horizontally
- Saves as NPZ for splitting
"""

import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, save_npz
import warnings
warnings.filterwarnings('ignore')

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Your ACTUAL file paths (from previous outputs)
HEURISTIC_CSV = os.path.join(BASE_DIR, "features", "heuristic", "heuristic_features_enhanced.csv")
NLP_NPZ = os.path.join(BASE_DIR, "features", "nlp", "nlp_features_enhanced_aligned.npz")

# Output directory
COMBINED_DIR = os.path.join(BASE_DIR, "features", "combined")
os.makedirs(COMBINED_DIR, exist_ok=True)

# Output files
COMBINED_NPZ = os.path.join(COMBINED_DIR, "combined_features.npz")
COMBINED_CSV = os.path.join(COMBINED_DIR, "combined_features_sample.csv")  # Small sample
COMBINED_INFO = os.path.join(COMBINED_DIR, "combined_features_info.txt")

# ---------------- LOAD HEURISTIC (CSV) ----------------
def load_heuristic_features():
    """Load heuristic features from CSV"""
    print("📥 Loading heuristic features from CSV...")
    
    if not os.path.exists(HEURISTIC_CSV):
        print(f"❌ File not found: {HEURISTIC_CSV}")
        return None, None, None, None
    
    try:
        df = pd.read_csv(HEURISTIC_CSV)
        print(f"✅ Loaded: {df.shape[0]:,} samples, {df.shape[1]} columns")
        
        # Check column structure
        print(f"   Columns: {list(df.columns)[:5]}...")  # Show first 5
        
        # Identify feature columns (exclude 'url' and 'label')
        feature_cols = [col for col in df.columns if col not in ['url', 'label']]
        
        if len(feature_cols) == 0:
            print("❌ No feature columns found!")
            return None, None, None, None
        
        print(f"   Feature columns: {len(feature_cols)}")
        
        # Extract features
        X_heuristic = df[feature_cols].values.astype(np.float32)
        
        # Convert to sparse for efficiency
        X_heuristic_sparse = csr_matrix(X_heuristic)
        
        # Extract labels
        labels = df['label'].values if 'label' in df.columns else None
        
        # Extract URLs
        urls = df['url'].values if 'url' in df.columns else None
        
        # Create feature names
        heuristic_feature_names = np.array([f"heuristic_{i}" for i in range(X_heuristic_sparse.shape[1])])
        
        # Show info
        print(f"   Features shape: {X_heuristic_sparse.shape}")
        if labels is not None:
            unique, counts = np.unique(labels, return_counts=True)
            print(f"   Labels: 0={counts[0]:,}, 1={counts[1]:,}")
        
        return X_heuristic_sparse, labels, urls, heuristic_feature_names
        
    except Exception as e:
        print(f"❌ Error loading heuristic CSV: {e}")
        return None, None, None, None

# ---------------- LOAD NLP (NPZ) ----------------
def load_nlp_features():
    """Load NLP features from NPZ"""
    print("\n📥 Loading NLP features from NPZ...")
    
    if not os.path.exists(NLP_NPZ):
        print(f"❌ File not found: {NLP_NPZ}")
        return None, None, None, None
    
    try:
        data = np.load(NLP_NPZ, allow_pickle=True)
        print(f"✅ Loaded NPZ file")
        
        # Show what's inside
        print(f"   Keys: {list(data.keys())}")
        
        # Check if it's sparse matrix format
        if all(key in data for key in ['data', 'indices', 'indptr', 'shape']):
            X_nlp = csr_matrix(
                (data['data'], data['indices'], data['indptr']),
                shape=tuple(data['shape'])
            )
            print(f"   Sparse matrix reconstructed: {X_nlp.shape}")
        else:
            print("❌ NPZ doesn't have sparse matrix components")
            return None, None, None, None
        
        # Extract labels
        labels = data['labels'] if 'labels' in data else None
        
        # Extract URLs
        urls = data['urls'] if 'urls' in data else None
        
        # Extract feature names
        feature_names = data['feature_names'] if 'feature_names' in data else None
        
        # Show info
        print(f"   Features: {X_nlp.shape[1]:,}")
        if labels is not None:
            unique, counts = np.unique(labels, return_counts=True)
            print(f"   Labels: 0={counts[0]:,}, 1={counts[1]:,}")
        
        return X_nlp, labels, urls, feature_names
        
    except Exception as e:
        print(f"❌ Error loading NLP NPZ: {e}")
        return None, None, None, None

# ---------------- COMBINE FEATURES ----------------
def combine_features():
    """Combine heuristic and NLP features"""
    print("\n" + "=" * 60)
    print("🔗 COMBINING HEURISTIC + NLP FEATURES")
    print("=" * 60)
    
    # Load both datasets
    X_heuristic, h_labels, h_urls, h_feature_names = load_heuristic_features()
    X_nlp, n_labels, n_urls, n_feature_names = load_nlp_features()
    
    if X_heuristic is None or X_nlp is None:
        print("❌ Failed to load one or both datasets")
        return False
    
    print("\n🔍 Verifying datasets...")
    
    # Check sample counts
    print(f"   Heuristic samples: {X_heuristic.shape[0]:,}")
    print(f"   NLP samples: {X_nlp.shape[0]:,}")
    
    # Align to same sample count
    if X_heuristic.shape[0] != X_nlp.shape[0]:
        print("⚠️  Sample count mismatch!")
        min_samples = min(X_heuristic.shape[0], X_nlp.shape[0])
        print(f"   Aligning to {min_samples:,} samples")
        
        X_heuristic = X_heuristic[:min_samples]
        X_nlp = X_nlp[:min_samples]
        
        if h_labels is not None:
            h_labels = h_labels[:min_samples]
        if n_labels is not None:
            n_labels = n_labels[:min_samples]
        
        if h_urls is not None and len(h_urls) > 0:
            h_urls = h_urls[:min_samples]
        if n_urls is not None and len(n_urls) > 0:
            n_urls = n_urls[:min_samples]
    
    # Choose which labels to use
    if h_labels is not None and n_labels is not None:
        if np.array_equal(h_labels, n_labels):
            print("✅ Labels match perfectly!")
            labels = h_labels
        else:
            print("⚠️  Labels don't match exactly, using heuristic labels")
            labels = h_labels
    elif h_labels is not None:
        labels = h_labels
        print("⚠️  Using heuristic labels only")
    elif n_labels is not None:
        labels = n_labels
        print("⚠️  Using NLP labels only")
    else:
        labels = None
        print("❌ No labels found!")
    
    # Choose which URLs to use
    urls = h_urls if h_urls is not None and len(h_urls) > 0 else n_urls
    
    # Combine features horizontally
    print("\n🔗 Combining features...")
    X_combined = hstack([X_heuristic, X_nlp], format='csr')
    
    # Create feature names
    if h_feature_names is None:
        h_feature_names = np.array([f"heuristic_{i}" for i in range(X_heuristic.shape[1])])
    
    if n_feature_names is None:
        n_feature_names = np.array([f"nlp_{i}" for i in range(X_nlp.shape[1])])
    
    combined_feature_names = np.concatenate([h_feature_names, n_feature_names])
    
    print(f"✅ Combined successfully!")
    print(f"   Total samples: {X_combined.shape[0]:,}")
    print(f"   Total features: {X_combined.shape[1]:,}")
    print(f"     - Heuristic: {X_heuristic.shape[1]:,}")
    print(f"     - NLP: {X_nlp.shape[1]:,}")
    
    if labels is not None:
        unique, counts = np.unique(labels, return_counts=True)
        print(f"   Labels: 0={counts[0]:,}, 1={counts[1]:,}")
    
    # Save combined features
    return save_combined(X_combined, labels, urls, combined_feature_names)

# ---------------- SAVE COMBINED ----------------
def save_combined(X_combined, labels, urls, feature_names):
    """Save combined features"""
    print("\n💾 Saving combined features...")
    
    try:
        # Save as NPZ
        np.savez_compressed(
            COMBINED_NPZ,
            data=X_combined.data,
            indices=X_combined.indices,
            indptr=X_combined.indptr,
            shape=np.array(X_combined.shape),
            feature_names=feature_names,
            labels=labels if labels is not None else np.array([]),
            urls=urls if urls is not None else np.array([]),
            heuristic_feature_count=X_combined.shape[1] - len([f for f in feature_names if f.startswith('nlp_')]),
            nlp_feature_count=len([f for f in feature_names if f.startswith('nlp_')]),
            matrix_format='csr'
        )
        print(f"✅ NPZ saved: {COMBINED_NPZ}")
        
        # Save info file
        with open(COMBINED_INFO, 'w') as f:
            f.write("COMBINED FEATURES INFORMATION\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Creation time: {pd.Timestamp.now()}\n")
            f.write(f"Total samples: {X_combined.shape[0]:,}\n")
            f.write(f"Total features: {X_combined.shape[1]:,}\n")
            f.write(f"Heuristic features: {X_combined.shape[1] - len([f for f in feature_names if f.startswith('nlp_')]):,}\n")
            f.write(f"NLP features: {len([f for f in feature_names if f.startswith('nlp_')]):,}\n")
            if labels is not None:
                unique, counts = np.unique(labels, return_counts=True)
                f.write(f"Benign samples (0): {counts[0]:,}\n")
                f.write(f"Malicious samples (1): {counts[1]:,}\n")
        
        print(f"✅ Info saved: {COMBINED_INFO}")
        
        # Save small sample as CSV (for verification)
        print("\n📊 Creating sample CSV (first 1000 samples)...")
        sample_size = min(1000, X_combined.shape[0])
        
        # Get sample data
        X_sample = X_combined[:sample_size].toarray()
        
        # Create DataFrame
        sample_data = {}
        for i in range(min(50, X_combined.shape[1])):  # First 50 features
            sample_data[feature_names[i]] = X_sample[:, i]
        
        if labels is not None:
            sample_data['label'] = labels[:sample_size]
        
        if urls is not None and len(urls) > 0:
            sample_data['url'] = urls[:sample_size]
        
        df_sample = pd.DataFrame(sample_data)
        df_sample.to_csv(COMBINED_CSV, index=False)
        print(f"✅ Sample CSV saved: {COMBINED_CSV}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving combined features: {e}")
        return False

# ---------------- MAIN ----------------
def main():
    print("🚀 COMBINE HEURISTIC + NLP FEATURES")
    print("=" * 60)
    
    # Check if input files exist
    print("🔍 Checking input files...")
    
    if not os.path.exists(HEURISTIC_CSV):
        print(f"❌ Heuristic CSV not found: {HEURISTIC_CSV}")
        print("\n💡 SOLUTION: Run heuristic feature extraction first:")
        print("   python scripts/extract_heuristic_features.py")
        return
    
    if not os.path.exists(NLP_NPZ):
        print(f"❌ NLP NPZ not found: {NLP_NPZ}")
        print("\n💡 SOLUTION: Run NLP feature extraction first:")
        print("   python scripts/extract_nlp_features_enhanced_aligned.py")
        return
    
    print(f"✅ Heuristic CSV: {HEURISTIC_CSV}")
    print(f"✅ NLP NPZ: {NLP_NPZ}")
    
    # Combine features
    success = combine_features()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 COMBINED FEATURES CREATED SUCCESSFULLY!")
        print("=" * 60)
        print(f"📁 Output directory: {COMBINED_DIR}")
        print(f"📦 NPZ file: {COMBINED_NPZ}")
        print(f"📄 Sample CSV: {COMBINED_CSV}")
        print(f"📋 Info file: {COMBINED_INFO}")
        
        print("\n✅ NOW YOU CAN RUN SPLITTING:")
        print("   python scripts/split_dataset_robust.py")
    else:
        print("\n❌ Failed to combine features")

if __name__ == "__main__":
    main()