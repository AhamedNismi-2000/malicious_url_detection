# Malicious URL Detector

An explainable, machine learning–based malicious URL detection system with a
Chrome/Brave browser extension front-end and a Flask REST API back-end. Combines
a Random Forest classifier trained on 559 URL-derived features with a rule-based
override layer, Google Safe Browsing threat-intel lookups, and WHOIS-based domain
age scoring — all wrapped in a real-time browser extension that blocks malicious
pages before they load, with LIME-powered natural-language explanations for
every verdict.

---

## Features

- **559-feature ML pipeline** — 57 heuristic URL features (leet-speak detection,
  brand impersonation, IP addresses, punycode, redirects, etc.) + 300 character
  n-gram + 202 word n-gram TF-IDF features
- **Random Forest classifier** — 300 trees, tuned for a 0.47 confidence threshold
- **4-layer detection pipeline** — Smart Whitelist → Google Safe Browsing →
  ML Model (+ rule-based overrides) → Domain Age (WHOIS) adjustment
- **Explainable AI (XAI)** — LIME-generated, human-readable reasons for every
  malicious verdict (e.g. *"This domain disguises a brand name using look-alike
  characters"*)
- **Real-time browser blocking** — intercepts navigation via
  `webNavigation.onBeforeNavigate` and redirects to a warning page **before**
  the malicious page loads
- **Voice alerts** — spoken warnings on blocked pages via the Web Speech API,
  toggleable in settings
- **History dashboard** — searchable, filterable URL history with a live
  auto-refreshing malicious-vs-benign bar chart
- **REST API** — `/predict`, `/predict/batch`, `/explain`, `/stats`, `/history`

---

## Architecture

```
URL Input
   │
   ├─ Layer 0: Smart Whitelist (Tranco top 50k + manual list + academic/gov TLDs)
   │     → BENIGN instantly if trusted domain/subdomain
   │
   ├─ Layer 1: Google Safe Browsing API v4
   │     → MALICIOUS instantly if known threat
   │
   ├─ Layer 2: ML Model (559 features → Random Forest → threshold 0.47)
   │     + rule-based overrides for verified model blind spots
   │     → MALICIOUS or BENIGN with confidence %
   │
   └─ Layer 3: Domain Age (WHOIS, cached) — adjusts confidence based on how
         recently the domain was registered
```

**Rule-based overrides** patch specific, verified gaps in the raw model output —
e.g. raw IP addresses served over HTTP, leet-speak brand impersonation
(`g00gle.com`), and embedded brand names in unrelated domains. These are a
deliberate hybrid ML + rule-based design choice; see [Limitations](#limitations).

---

## Model Performance

| Metric | Value |
|---|---|
| Accuracy | 95.55% |
| Precision | 96.85% |
| Recall | 94.17% |
| F1 Score | 0.9549 |
| AUC-ROC | 0.9889 |
| False Positive Rate | 3.06% |
| False Negative Rate | 5.83% |

Evaluated on a held-out 15% stratified test split. See [`scripts/evaluate_model.py`](scripts/evaluate_model.py).

---

## Project Structure

```
malicious_url_detection/
├── app/                    # Flask backend
│   ├── app.py               — entry point
│   ├── routes.py             — API endpoints
│   └── model_loader.py       — 4-layer prediction pipeline
├── scripts/                # Training & data pipeline (dev-only, not deployed)
│   ├── preprocessing.py, split.py, feature_extraction.py,
│   │   train_model.py, evaluate_model.py, test_model.py, etc.
├── extension/               # Chrome/Brave browser extension (Manifest V3)
│   ├── manifest.json
│   ├── background.js         — navigation interception + blocking logic
│   ├── popup.html/js         — toolbar popup UI
│   ├── content.js            — in-page warning banner
│   ├── blocked.html/js       — full-page block screen + voice alert
│   └── history.html/js       — history dashboard + live chart
├── models/                 # Trained model artifacts (required at runtime)
├── data/                   # Datasets (dev-only, not deployed)
└── requirements.txt
```

---

## Setup — Local Development

### Backend

```bash
git clone https://github.com/AhamedNismi-2000/malicious_url_detection.git
cd malicious_url_detection
python -m venv venv
.\venv\Scripts\activate        # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

Create a `.env` file in the project root:
```
GOOGLE_SAFE_BROWSING_API_KEY=your_key_here
```

Run the server:
```bash
python app/app.py
```
API available at `http://localhost:5000`.

### Browser Extension

1. Open `brave://extensions/` (or `chrome://extensions/`)
2. Enable **Developer mode**
3. **Load unpacked** → select the `extension/` folder
4. Click the extension icon → gear icon → confirm **API Base URL** points to
   your running Flask instance (`http://localhost:5000` by default)

**Supported browsers:** Chrome, Brave, Edge, Opera, Vivaldi (Chromium-based
only). Not currently supported on Firefox or Safari.

---

## Deployment (Free, No Credit Card)

Backend deployed on [Render](https://render.com) free tier:

1. Push only the runtime-required files (`app/`, `scripts/feature_extraction.py`,
   `models/`, `requirements.txt`) to GitHub
2. Render Dashboard → **New Web Service** → connect repo
3. Build command: `pip install -r requirements.txt`
4. Start command: `python app/app.py`
5. Set `GOOGLE_SAFE_BROWSING_API_KEY` under the **Environment** tab (never
   commit the real key to the repo)
6. Deploy → update the extension's API Base URL to the resulting
   `https://your-app.onrender.com` URL

Free tier note: the service sleeps after 15 minutes of inactivity (30–60s cold
start on the next request). Optionally ping `/health` every ~10 minutes via a
free uptime monitor to keep it warm during demos.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check, model status + threshold |
| `/predict` | POST | Classify a single URL |
| `/predict/batch` | POST | Classify up to 500 URLs |
| `/explain` | POST | Classify + LIME explanation |
| `/stats` | GET | Summary statistics |
| `/history` | GET | Full URL check history |
| `/stats/clear` | POST | Wipe all history |

Example:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'
```

---

## Limitations

- **Confidence scores are not uniformly calibrated.** Rule-based override cases
  return fixed sentinel values (85%/90%/100%) rather than genuine model
  probabilities, while standard ML-path predictions are computed scores. This
  is a deliberate hybrid design trade-off for reliability on well-understood
  attack patterns, documented explicitly rather than hidden.
- **TF-IDF vocabulary bias.** The word n-gram features were trained mostly on
  English/`.com`-heavy benign URLs; legitimate sites with unfamiliar vocabulary
  (non-Western TLDs, uncommon words) can be scored higher than warranted even
  with zero structural red flags. Partially mitigated via confidence dampening
  for clean HTTPS sites, at a small cost to recall.
- **WHOIS coverage gaps.** Domain age lookups can fail for some ccTLDs (e.g.
  `.lk`), returning `"unknown"` and skipping that adjustment layer entirely.
- **URL-structure detection only.** The system detects phishing-pattern URLs;
  it does not assess downloaded file safety or site content/reputation.
  Content-based threats (e.g. malware-bundling piracy sites with structurally
  clean URLs) fall outside its detection scope by design.
- **Chromium-based browsers only.** Firefox and Safari are not currently supported.

---

## Future Work

- Train `domain_age_days` as a direct model feature (currently a post-prediction
  multiplier) for a more principled, learned age-based adjustment
- Reputation-API integration (e.g. VirusTotal, Scamadviser) to catch
  content/behavior-based threats outside current URL-structure scope
- Calibrated confidence scores across ML and override paths
- Firefox support via the `webextension-polyfill`
- Broader false-positive evaluation across non-English/non-`.com` domains

---

## Tech Stack

Python · scikit-learn · Flask · LIME · pandas · NumPy · tldextract ·
python-whois · Chrome Extension (Manifest V3) · Chart.js

---

## License

_(Add your chosen license here — e.g. MIT)_

---

## Author

**Ahamed Nismi**
Final-year Computer Science undergraduate, Uva Wellassa University of Sri Lanka
[GitHub](https://github.com/AhamedNismi-2000) ·
[LinkedIn](https://linkedin.com/in/ahamednismi312) ·
[Portfolio](https://ahamednismi-2000.github.io/nismi_portfolio/)