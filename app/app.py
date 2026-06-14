#!/usr/bin/env python3
"""
app.py
------
Flask application entry point for the malicious URL detector API.

Run:
  python app/app.py

The server starts on http://0.0.0.0:5000 by default.
Used by the browser extension (background.js) to classify URLs.
"""

import os
import sys

# Ensure this directory is on the path so `from model_loader import classifier`
# and `from routes import api` work regardless of working directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask
from flask_cors import CORS
from routes import api


def create_app() -> Flask:
    app = Flask(__name__)

    # Allow requests from the browser extension (any origin).
    # Browser extensions use chrome-extension:// / moz-extension:// origins,
    # so a wildcard is the simplest correct setup for a local API.
    CORS(app, resources={r"/*": {"origins": "*"}})

    app.register_blueprint(api)

    return app


app = create_app()


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"

    print("=" * 60)
    print("Malicious URL Detector API")
    print("=" * 60)
    print(f"  Model info : {app.view_functions}")
    print(f"  Starting on http://{host}:{port}")
    print("=" * 60)

    app.run(host=host, port=port, debug=debug)