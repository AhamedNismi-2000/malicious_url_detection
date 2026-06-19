"""
model_loader.py
---------------
Loads model artefacts and exposes three public methods:

  predict_url(url)                   -> dict
  predict_batch(urls)                -> list[dict]
  explain_url(url, num_features=10)  -> dict   (prediction + LIME explanation)

The URLClassifier is a singleton; import `classifier` directly:

  from model_loader import classifier
"""

import json
import os
import re
import sys
import threading
import warnings
from typing import Optional

import joblib
import numpy as np

# ── Locate project root and add scripts/ to path ─────────────────────────────
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.abspath(os.path.join(_APP_DIR, ".."))
_SCRIPTS = os.path.join(_ROOT, "scripts")

if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from feature_extraction import extract_heuristic_features, preprocess_url_for_nlp

# ── Path constants ────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(_ROOT, "models")
DATA_DIR   = os.path.join(_ROOT, "data")

# ── Feature names (must match training order) ─────────────────────────────────
HEURISTIC_FEATURES: list[str] = [
    "url_len", "path_len", "num_dots", "path_dots", "num_hyphens",
    "num_underscores", "num_at", "num_qmark", "num_equal", "num_amp",
    "num_percent", "num_digits", "num_letters", "num_subdirs", "num_frag",
    "num_special", "num_repeating", "num_upper", "num_non_ascii",
    "num_slashes", "num_params", "ratio_digits", "ratio_letters",
    "url_entropy", "ip_flag", "subdomain_parts", "has_multi_subdomain",
    "tld_len", "risky_tld", "https_flag", "shortened", "sus_words",
    "brand_mismatch", "puny", "susp_ext", "suspicious_port",
    "max_consonants", "max_vowels", "max_digits",
    "leet_speak_score", "homoglyph_suspicious", "encoding_ratio",
    "punycode_suspicious", "subdomain_spam_score", "visual_brand_similarity",
    "brand_in_domain", "leet_in_domain", "brand_hyphen_suspicious",
]

FEATURE_NAMES: list[str] = (
    HEURISTIC_FEATURES
    + [f"char_{i}" for i in range(300)]
    + [f"word_{i}" for i in range(200)]
)

# Boolean/flag features — LIME treats these as categorical
_CATEGORICAL_FEATURE_NAMES: list[str] = [
    "ip_flag", "has_multi_subdomain", "risky_tld", "https_flag",
    "shortened", "sus_words", "brand_mismatch", "puny", "susp_ext",
    "suspicious_port", "brand_in_domain", "leet_in_domain",
    "brand_hyphen_suspicious",
]

# 54 trusted domains — instant BENIGN without hitting the model
WHITELIST: frozenset[str] = frozenset({
    "google.com", "youtube.com", "facebook.com", "twitter.com", "instagram.com",
    "linkedin.com", "wikipedia.org", "amazon.com", "apple.com", "microsoft.com",
    "github.com", "stackoverflow.com", "reddit.com", "netflix.com", "spotify.com",
    "dropbox.com", "slack.com", "zoom.us", "adobe.com", "salesforce.com",
    "paypal.com", "ebay.com", "walmart.com", "target.com", "bestbuy.com",
    "nytimes.com", "bbc.com", "cnn.com", "theguardian.com", "reuters.com",
    "harvard.edu", "mit.edu", "stanford.edu", "coursera.org", "udemy.com",
    "python.org", "npmjs.com", "pypi.org", "docker.com", "kubernetes.io",
    "cloudflare.com", "aws.amazon.com", "azure.microsoft.com", "cloud.google.com",
    "stripe.com", "twilio.com", "sendgrid.com", "mailchimp.com", "hubspot.com",
    "wordpress.com", "shopify.com", "squarespace.com", "wix.com", "weebly.com",
    "anthropic.com", "openai.com",
})


# ── URLClassifier (singleton) ─────────────────────────────────────────────────

class URLClassifier:
    """Thread-safe singleton. Load artefacts once; serve predictions forever."""

    _instance: Optional["URLClassifier"] = None
    _init_lock = threading.Lock()

    def __new__(cls):
        with cls._init_lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._initialized = False
                cls._instance = instance
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.model    = joblib.load(os.path.join(MODELS_DIR, "rf_model_latest.joblib"))
        self.vec_char = joblib.load(os.path.join(MODELS_DIR, "vectorizer_char.joblib"))
        self.vec_word = joblib.load(os.path.join(MODELS_DIR, "vectorizer_word.joblib"))
        self.scaler   = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))

        with open(os.path.join(MODELS_DIR, "threshold.json")) as fh:
            self.threshold = float(json.load(fh).get("threshold", 0.44))

        # LIME explainer — built lazily, cached after first call
        self._explainer: Optional[object] = None
        self._explainer_lock = threading.Lock()

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _registered_domain(url: str) -> str:
        """Return 'example.com' from any URL (no external deps)."""
        cleaned = re.sub(r"^https?://", "", url, flags=re.IGNORECASE)
        host    = cleaned.split("/")[0].split(":")[0].split("?")[0].lower()
        parts   = host.split(".")
        return ".".join(parts[-2:]) if len(parts) >= 2 else host

    def _feature_vector(self, url: str) -> np.ndarray:
        """Build the 548-dim feature vector for *url*."""
        # 48 heuristic features — scaler was fitted on these only
        heuristic = np.array(
            extract_heuristic_features(url), dtype=np.float32
        ).reshape(1, -1)                                            # (1, 48)
        heuristic_scaled = self.scaler.transform(heuristic).flatten()  # (48,)

        # 300 + 200 NLP features — unscaled, exactly as during training
        processed  = preprocess_url_for_nlp(url)
        char_dense = self.vec_char.transform([processed]).toarray().flatten()  # (300,)
        word_dense = self.vec_word.transform([processed]).toarray().flatten()  # (200,)

        # Concatenate in training order: heuristic_scaled + char + word
        return np.concatenate([heuristic_scaled, char_dense, word_dense])  # (548,)

    # ── Public: prediction ────────────────────────────────────────────────────

    def predict_url(self, url: str) -> dict:
        if not url or not isinstance(url, str):
            return {
                "url": url, "prediction": "BENIGN",
                "confidence": 0.0,
                "threshold": round(self.threshold * 100, 2),
                "source": "invalid",
            }

        if self._registered_domain(url) in WHITELIST:
            return {
                "url": url, "prediction": "BENIGN",
                "confidence": 100.0,
                "threshold": round(self.threshold * 100, 2),
                "source": "whitelist",
            }

        try:
            fv    = self._feature_vector(url)
            proba = float(self.model.predict_proba(fv.reshape(1, -1))[0][1])
            label = "MALICIOUS" if proba >= self.threshold else "BENIGN"
            return {
                "url": url, "prediction": label,
                "confidence": round(proba * 100, 2),
                "threshold" : round(self.threshold * 100, 2),
                "source"    : "model",
            }
        except Exception as exc:
            return {
                "url": url, "prediction": "BENIGN",
                "confidence": 0.0,
                "threshold": round(self.threshold * 100, 2),
                "source": "invalid", "error": str(exc),
            }

    def predict_batch(self, urls: list[str]) -> list[dict]:
        return [self.predict_url(u) for u in urls]

    # ── Public: LIME explanation ──────────────────────────────────────────────

    def explain_url(self, url: str, num_features: int = 10) -> dict:
        """
        Predict + explain. Returns standard predict dict plus:
          "explanation": [{"feature": str, "weight": float, "value": float}, ...]
        Whitelist / invalid URLs return an empty explanation list.
        """
        base = self.predict_url(url)
        if base["source"] in ("whitelist", "invalid"):
            return {**base, "explanation": []}

        try:
            explainer = self._get_explainer()
            fv        = self._feature_vector(url)

            exp = explainer.explain_instance(
                data_row     = fv,
                predict_fn   = self._lime_predict_fn,
                num_features = num_features,
                top_labels   = 1,
            )

            raw_list = exp.as_list(label=1)  # label 1 = MALICIOUS

            explanation = []
            for condition_str, weight in raw_list:
                feat_name = _parse_lime_feature(condition_str)
                feat_idx  = FEATURE_NAMES.index(feat_name) \
                            if feat_name in FEATURE_NAMES else -1
                feat_val  = float(fv[feat_idx]) if feat_idx >= 0 else 0.0
                explanation.append({
                    "feature": feat_name,
                    "weight" : round(float(weight), 6),
                    "value"  : round(feat_val, 6),
                })

            explanation.sort(key=lambda x: abs(x["weight"]), reverse=True)
            return {**base, "explanation": explanation}

        except Exception as exc:
            return {**base, "explanation": [], "explain_error": str(exc)}

    # ── LIME internals ────────────────────────────────────────────────────────

    def _get_explainer(self):
        """Build LimeTabularExplainer lazily; cache forever (thread-safe)."""
        if self._explainer is not None:
            return self._explainer

        with self._explainer_lock:
            if self._explainer is not None:  # double-checked
                return self._explainer

            try:
                from lime.lime_tabular import LimeTabularExplainer
            except ImportError as exc:
                raise ImportError(
                    "lime is not installed. Run: pip install lime"
                ) from exc

            bg = self._load_background()

            cat_indices = [
                FEATURE_NAMES.index(n)
                for n in _CATEGORICAL_FEATURE_NAMES
                if n in FEATURE_NAMES
            ]

            self._explainer = LimeTabularExplainer(
                training_data        = bg,
                feature_names        = FEATURE_NAMES,
                class_names          = ["BENIGN", "MALICIOUS"],
                categorical_features = cat_indices,
                mode                 = "classification",
                discretize_continuous= True,
                random_state         = 42,
            )
            return self._explainer

    def _load_background(self, n_samples: int = 5000) -> np.ndarray:
        """Load lime_background.npz; fall back to building from train_urls.csv."""
        bg_path = os.path.join(MODELS_DIR, "lime_background.npz")
        if os.path.exists(bg_path):
            return np.load(bg_path)["X"]

        warnings.warn(
            f"{bg_path} not found — building LIME background from train_urls.csv. "
            "This runs once then saves the result.",
            RuntimeWarning, stacklevel=3,
        )

        import csv, random
        csv_path = os.path.join(_ROOT, "data", "splits", "train_urls.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Cannot find {bg_path} or {csv_path}.\n"
                "Run: python -c \"from model_loader import classifier; "
                "classifier._load_background()\""
            )

        with open(csv_path, newline="", encoding="utf-8") as fh:
            reader   = csv.DictReader(fh)
            all_urls = [row["url"] for row in reader if row.get("url")]

        sample = random.sample(all_urls, min(n_samples, len(all_urls)))
        rows   = []
        for u in sample:
            try:
                rows.append(self._feature_vector(u))
            except Exception:
                pass

        bg = np.array(rows, dtype=np.float32)
        np.savez_compressed(bg_path, X=bg)
        return bg

    def _lime_predict_fn(self, X: np.ndarray) -> np.ndarray:
        """predict_proba wrapper for LIME (input is already-scaled vectors)."""
        return self.model.predict_proba(X)


# ── Helper ────────────────────────────────────────────────────────────────────

def _parse_lime_feature(condition_str: str) -> str:
    """
    Recover bare feature name from a LIME condition string.
    e.g. 'brand_in_domain=1'  →  'brand_in_domain'
         'url_len > 45.00'    →  'url_len'
    Uses longest-match to handle underscored names correctly.
    """
    for name in sorted(FEATURE_NAMES, key=len, reverse=True):
        if condition_str.startswith(name):
            return name
    return re.split(r"[\s<>=!]", condition_str)[0]


# ── Module-level singleton ────────────────────────────────────────────────────
classifier = URLClassifier()