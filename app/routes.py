"""
routes.py
---------
Flask route definitions for the Malicious URL Detection API.

Endpoints:
  GET  /health          — liveness check
  GET  /                — API info
  POST /predict         — classify a single URL
  POST /predict/batch   — classify up to 500 URLs
  POST /explain         — classify + LIME explanation for a single URL
"""

from flask import Blueprint, jsonify, request

from model_loader import classifier           # direct import (sys.path set by app.py)

api = Blueprint("api", __name__)             # named `api` to match your convention


# ── Helpers ───────────────────────────────────────────────────────────────────

def _bad_request(msg: str):
    return jsonify({"error": msg}), 400


def _require_url(data: dict):
    """Return (url, error_response).  error_response is None on success."""
    if not data or "url" not in data:
        return None, _bad_request("Request body must include a 'url' field.")
    url = data["url"]
    if not isinstance(url, str) or not url.strip():
        return None, _bad_request("'url' must be a non-empty string.")
    return url.strip(), None


# ── Routes ────────────────────────────────────────────────────────────────────

@api.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status"   : "ok",
        "model"    : "rf_model_latest",
        "threshold": classifier.threshold,
    }), 200


@api.route("/", methods=["GET"])
def index():
    return jsonify({
        "service"  : "Malicious URL Detector",
        "version"  : "1.0",
        "endpoints": {
            "GET  /health"        : "Health check",
            "POST /predict"       : "Classify a single URL",
            "POST /predict/batch" : "Classify multiple URLs",
            "POST /explain"       : "Classify + LIME explanation for a single URL",
        },
    }), 200


@api.route("/predict", methods=["POST"])
def predict():
    url, err = _require_url(request.get_json(silent=True) or {})
    if err:
        return err
    return jsonify(classifier.predict_url(url)), 200


@api.route("/predict/batch", methods=["POST"])
def predict_batch():
    data = request.get_json(silent=True) or {}
    urls = data.get("urls")
    if not urls or not isinstance(urls, list):
        return _bad_request("Request body must include a 'urls' list.")
    if len(urls) > 500:
        return _bad_request("Batch size limited to 500 URLs per request.")
    return jsonify({"results": classifier.predict_batch(urls)}), 200


@api.route("/explain", methods=["POST"])
def explain():
    """
    POST /explain
    Request  : {"url": "https://example.com", "num_features": 10}
    Response : {
        "url"        : "...",
        "prediction" : "MALICIOUS" | "BENIGN",
        "confidence" : 87.3,
        "threshold"  : 44.0,
        "source"     : "model" | "whitelist" | "invalid",
        "explanation": [
            {"feature": "brand_in_domain", "weight": 0.42, "value": 1.0},
            ...up to num_features entries...
        ]
    }
    """
    data = request.get_json(silent=True) or {}
    url, err = _require_url(data)
    if err:
        return err

    num_features = int(data.get("num_features", 10))
    num_features = max(1, min(num_features, 30))   # clamp to 1–30

    result = classifier.explain_url(url, num_features=num_features)
    return jsonify(result), 200