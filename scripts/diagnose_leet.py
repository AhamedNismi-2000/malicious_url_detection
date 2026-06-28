#!/usr/bin/env python3
"""
diagnose_leet.py
----------------
Run from project root:
  python scripts/diagnose_leet.py
"""

import sys
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'app')

from feature_extraction import (
    extract_heuristic_features,
    HEURISTIC_FEATURE_NAMES,
    leet_in_domain_only,
    leet_brand_score,
    detect_leet_speak,
    decode_leet
)

url         = "http://go0gle.com/"
domain_only = "go0gle"
reg_domain  = "go0gle.com"

# Raw feature vector
fv = extract_heuristic_features(url)

# Key indices
idx_leet_dom   = HEURISTIC_FEATURE_NAMES.index("leet_in_domain")
idx_leet_brand = HEURISTIC_FEATURE_NAMES.index("leet_brand_score")
idx_leet_score = HEURISTIC_FEATURE_NAMES.index("leet_speak_score")
idx_https      = HEURISTIC_FEATURE_NAMES.index("https_flag")

print("=" * 55)
print("LEET DETECTION DIAGNOSTIC — http://go0gle.com/")
print("=" * 55)

print("\n--- Feature vector values ---")
print(f"leet_in_domain   : {fv[idx_leet_dom]}")
print(f"leet_brand_score : {fv[idx_leet_brand]}")
print(f"leet_speak_score : {fv[idx_leet_score]:.4f}")
print(f"https_flag       : {fv[idx_https]}")

print("\n--- Direct function calls ---")
print(f"leet_in_domain_only(go0gle)        : {leet_in_domain_only(domain_only)}")
print(f"leet_brand_score(go0gle, go0gle.com): {leet_brand_score(domain_only, reg_domain)}")
print(f"detect_leet_speak(url)             : {detect_leet_speak(url):.4f}")
print(f"decode_leet(go0gle)                : {decode_leet(domain_only)}")

print("\n--- Override condition ---")
leet_detected = (fv[idx_leet_dom] == 1.0) or (fv[idx_leet_brand] == 1.0)
https_off     = fv[idx_https] == 0.0
print(f"leet_detected  : {leet_detected}")
print(f"https_off      : {https_off}")
print(f"Override fires : {leet_detected and https_off}")

print("\n--- Model loader import check ---")
try:
    from model_loader import classifier
    raw = extract_heuristic_features(url)
    from feature_extraction import HEURISTIC_FEATURE_NAMES as HFN
    from app.model_loader import _FEAT_IDX
    leet_dom_val   = raw[_FEAT_IDX.get("leet_in_domain", -1)]
    leet_brand_val = raw[_FEAT_IDX.get("leet_brand_score", -1)]
    https_val      = raw[_FEAT_IDX.get("https_flag", -1)]
    print(f"_FEAT_IDX leet_in_domain   : {_FEAT_IDX.get('leet_in_domain')}")
    print(f"_FEAT_IDX leet_brand_score : {_FEAT_IDX.get('leet_brand_score')}")
    print(f"leet_dom_val   : {leet_dom_val}")
    print(f"leet_brand_val : {leet_brand_val}")
    print(f"https_val      : {https_val}")
    print(f"Override fires : {(leet_dom_val == 1.0 or leet_brand_val == 1.0) and https_val == 0.0}")
except Exception as e:
    print(f"model_loader import error: {e}")

print("=" * 55)