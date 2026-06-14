#!/usr/bin/env python3
"""
routes.py
---------
API endpoints for the malicious URL detector.

Endpoints:
  GET  /                 - health check / info
  GET  /health           - health check
  POST /predict          - single URL: {"url": "..."}
  POST /predict/batch    - multiple URLs: {"urls": ["...", "..."]}

Response format (single):
  {
    "url": "...",
    "prediction": "MALICIOUS" | "BENIGN",
    "confidence": 0-100,
    "threshold": 0-100,
    "source": "whitelist" | "model" | "invalid"
  }
"""

from flask import Blueprint, request, jsonify
from model_loader import classifier

api = Blueprint("api", __name__)


@api.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "Malicious URL Detector API",
        "status": "running",
        "model_info": classifier.info(),
        "endpoints": {
            "health": "GET /health",
            "predict_single": "POST /predict  {url: string}",
            "predict_batch": "POST /predict/batch  {urls: [string]}"
        }
    })


@api.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": True})


@api.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    url = data.get("url")

    if not url or not isinstance(url, str):
        return jsonify({
            "error": "Missing or invalid 'url' field. "
                     "Expected JSON: {\"url\": \"https://example.com\"}"
        }), 400

    result = classifier.predict_url(url)
    return jsonify(result)


@api.route("/predict/batch", methods=["POST"])
def predict_batch():
    data = request.get_json(silent=True) or {}
    urls = data.get("urls")

    if not urls or not isinstance(urls, list):
        return jsonify({
            "error": "Missing or invalid 'urls' field. "
                     "Expected JSON: {\"urls\": [\"https://example.com\", ...]}"
        }), 400

    if len(urls) > 100:
        return jsonify({
            "error": "Too many URLs. Maximum 100 per batch request."
        }), 400

    results = classifier.predict_batch(urls)
    return jsonify({"results": results, "count": len(results)})