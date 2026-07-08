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
  GET  /stats           — get summary statistics
  GET  /history         — get full URL history
  POST /stats/clear     — clear all history
"""

import json
import os
import threading
from datetime import datetime, timezone
from flask import Blueprint, jsonify, request
from model_loader import classifier

api = Blueprint("api", __name__)

# ── Stats file path ───────────────────────────────────────────────────────────
_APP_DIR   = os.path.dirname(os.path.abspath(__file__))
STATS_PATH = os.path.join(_APP_DIR, "stats.json")
_stats_lock = threading.Lock()

# ── Stats helpers ─────────────────────────────────────────────────────────────

def _load_stats() -> dict:
    """Load stats from disk."""
    if os.path.exists(STATS_PATH):
        try:
            with open(STATS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "total"    : 0,
        "malicious": 0,
        "benign"   : 0,
        "history"  : [],
    }


def _save_stats(data: dict):
    """Save stats to disk."""
    try:
        with open(STATS_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def _record(result: dict):
    """
    Record a URL check result into stats.json.
    Called after every /predict and /explain.
    Keeps last 1000 entries in history.
    """
    with _stats_lock:
        data = _load_stats()

        prediction = result.get("prediction", "UNKNOWN")
        url        = result.get("url", "")
        confidence = result.get("confidence", 0)
        source     = result.get("source", "model")
        brand      = result.get("brand_detected", None)

        # Update counters
        data["total"] += 1
        if prediction == "MALICIOUS":
            data["malicious"] += 1
        elif prediction == "BENIGN":
            data["benign"] += 1

        # Add to history (newest first)
        entry = {
            "url"       : url,
            "prediction": prediction,
            "confidence": round(confidence, 1),
            "source"    : source,
            "timestamp" : datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        }
        if brand:
            entry["brand"] = brand

        data["history"].insert(0, entry)

        # Keep only last 1000
        data["history"] = data["history"][:1000]

        _save_stats(data)


# ── Request helpers ───────────────────────────────────────────────────────────

def _bad_request(msg: str):
    return jsonify({"error": msg}), 400


def _require_url(data: dict):
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
            "POST /explain"       : "Classify + LIME explanation",
            "GET  /stats"         : "Summary statistics",
            "GET  /history"       : "Full URL history",
            "POST /stats/clear"   : "Clear all history",
        },
    }), 200


@api.route("/predict", methods=["POST"])
def predict():
    url, err = _require_url(request.get_json(silent=True) or {})
    if err:
        return err
    result = classifier.predict_url(url)
    _record(result)
    return jsonify(result), 200


@api.route("/predict/batch", methods=["POST"])
def predict_batch():
    data = request.get_json(silent=True) or {}
    urls = data.get("urls")
    if not urls or not isinstance(urls, list):
        return _bad_request("Request body must include a 'urls' list.")
    if len(urls) > 500:
        return _bad_request("Batch size limited to 500 URLs per request.")
    results = classifier.predict_batch(urls)
    for r in results:
        _record(r)
    return jsonify({"results": results}), 200


@api.route("/explain", methods=["POST"])
def explain():
    data = request.get_json(silent=True) or {}
    url, err = _require_url(data)
    if err:
        return err
    num_features = int(data.get("num_features", 10))
    num_features = max(1, min(num_features, 30))
    result = classifier.explain_url(url, num_features=num_features)
    return jsonify(result), 200


@api.route("/stats", methods=["GET"])
def get_stats():
    """
    GET /stats
    Returns summary statistics.
    Response: {
        "total"    : 150,
        "malicious": 23,
        "benign"   : 127,
        "mal_rate" : 15.3
    }
    """
    with _stats_lock:
        data = _load_stats()
    total = data["total"]
    mal   = data["malicious"]
    ben   = data["benign"]
    return jsonify({
        "total"    : total,
        "malicious": mal,
        "benign"   : ben,
        "mal_rate" : round((mal / total * 100) if total > 0 else 0, 1),
    }), 200


@api.route("/history", methods=["GET"])
def get_history():
    """
    GET /history?limit=100&filter=all|malicious|benign
    Returns URL history list.
    """
    limit  = min(int(request.args.get("limit", 100)), 1000)
    filter_ = request.args.get("filter", "all").lower()

    with _stats_lock:
        data = _load_stats()

    history = data.get("history", [])

    if filter_ == "malicious":
        history = [h for h in history if h["prediction"] == "MALICIOUS"]
    elif filter_ == "benign":
        history = [h for h in history if h["prediction"] == "BENIGN"]

    return jsonify({
        "total"  : len(history),
        "history": history[:limit],
    }), 200


@api.route("/stats/clear", methods=["POST"])
def clear_stats():
    """POST /stats/clear — wipe all history."""
    with _stats_lock:
        _save_stats({"total": 0, "malicious": 0, "benign": 0, "history": []})
    return jsonify({"status": "cleared"}), 200