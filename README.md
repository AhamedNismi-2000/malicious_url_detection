# Malicious URL Detector — Full Session Summary & Context Prompt

Use this prompt at the start of a new chat to restore full project context.

---

## PROJECT OVERVIEW

**Title:** Explainable Machine Learning-Based Malicious URL Detection Browser Extension
**Stack:** Python 3.14 · scikit-learn · Flask · Chrome/Brave Extension (Manifest V3)
**Project path:** `D:\Research Project\Research_Working_dir_2\malicious_url_detection`
**venv:** `.\venv\Scripts\activate`
**Flask port:** 5000

---

## FINAL MODEL PERFORMANCE

| Metric | Value |
|--------|-------|
| F1 Score | 0.9568 |
| AUC-ROC | 0.9908 |
| Accuracy | 0.9571 |
| Precision | 0.9639 |
| Recall | 0.9499 |
| FPR | 3.54% |
| FNR | 5.01% |
| Threshold | 0.47 |
| Features | 559 |

**Algorithm:** RandomForestClassifier — 300 trees, max_depth=25, min_samples_split=10,
min_samples_leaf=4, max_features="sqrt", max_samples=0.8, bootstrap=True, random_state=42

---

## FEATURE ENGINEERING (559 total)

```
Heuristic features     : 57  (was 56 — added leet_brand_score)
Char n-gram TF-IDF     : 300 (analyzer=char_wb, ngram_range=(2,4))
Word n-gram TF-IDF     : 202 (analyzer=word, ngram_range=(1,2))
─────────────────────────────
TOTAL                  : 559
```

### Key heuristic features added this session:
- `leet_brand_score` (NEW #57) — decodes leet substitutions dynamically and checks
  against BRANDS set. Returns 1.0 when leet + brand name confirmed.
- `leet_in_domain` — REWRITTEN to catch word-boundary leet (paypa1, 1nstagram),
  double substitution (g00gle), end-of-word leet. Old version missed these.
- `detect_leet_speak()` — REWRITTEN to density-based scoring (not rigid pattern match)
- `decode_leet()` — NEW dynamic leet decoder that reverses substitutions at runtime,
  generates all candidate plain-text strings. No hardcoded lookup table needed.
- `has_redirect`, `double_slash_in_path`, `abnormal_subdomain` — new structural features
- `http_no_brand_no_age` — HTTPS bias correction feature

### NLP pipeline:
- `segment_url()` — tokenises URL by stripping scheme/www, decoding percent-encoding,
  splitting on special chars before TF-IDF
- Both vectorisers fitted on TRAIN only (no data leakage)
- TRANCO_PATH bug fixed — trailing space in "raw " directory name caused silent fallback

---

## 4-LAYER PREDICTION PIPELINE

```
URL Input
   │
   ├─ Layer 0: Smart Whitelist (Tranco top 50k + manual 91 domains)
   │     → BENIGN instantly if trusted domain/subdomain
   │
   ├─ Layer 1: Google Safe Browsing API v4
   │     → MALICIOUS instantly if known threat
   │
   ├─ Layer 2: ML Model (559 features → RF → threshold 0.47)
   │     + Rule-based overrides (see below)
   │     → MALICIOUS or BENIGN with confidence %
   │
   └─ Layer 3: Domain Age WHOIS (cached in domain_age_cache.json)
         < 30 days   → × 1.4   (boost malicious)
         30-180 days → × 1.15
         180-365days → × 1.0
         1-2 years   → × 0.7   (reduce malicious)
         2-5 years   → × 0.5
         > 5 years   → × 0.3   (reduce significantly)
```

---

## RULE-BASED OVERRIDES IN _classify() (all have early return — skip domain age)

| Fix | Trigger | Action |
|-----|---------|--------|
| FIX 9 | leet_in_domain=1.0 OR leet_brand_score=1.0 AND no HTTPS | Force 85%, return MALICIOUS, skip domain age |
| FIX 12 | Suspicious word in domain label (phishing, malware, trojan...) | Force 90%, return MALICIOUS |
| FIX 13 | brand_in_domain=1.0 OR brand_mismatch=1.0 AND no HTTPS | Force 85%, return MALICIOUS |
| FIX 14 | ip_flag=1.0 AND no HTTPS (checked in predict_url BEFORE reverse DNS) | Force 85%, return MALICIOUS |

**Critical bug fixed:** Domain age multiplier was crushing leet override.
`0.85 × 0.3 (old domain) = 0.255 → BENIGN` — fixed by adding early return
before domain age adjustment for all 4 override cases.

---

## SMART WHITELIST (_is_safe_whitelist_url)

Replaces simple `registered_domain in WHITELIST` check. Handles:

1. **Legit subdomains** — `mail.google.com` → google.com whitelisted +
   "mail" in TRUSTED_SUBDOMAIN_PREFIXES → BENIGN ✓
2. **Bypass attacks** — `google.com.evil.tk` → registered=evil.tk →
   NOT whitelisted → ML model ✓
3. **Brand spoofing in subdomain** — `paypal.evil.tk` → "paypal" detected
   as brand label in subdomain → NOT trusted → ML model ✓
4. **Academic/Government TLDs** — `.ac.lk`, `.edu`, `.gov`, `.ac.uk` etc →
   trusted automatically (fixes Sri Lankan university LMS false positives)
5. **Shorteners removed** — `whitelist = whitelist - SHORTENERS` so
   tinyurl.com, bit.ly go through ML model

### TRUSTED_SUBDOMAIN_PREFIXES includes:
www, mail, api, docs, login, signin, auth, accounts, app, dashboard,
support, blog, dev, shop, secure, cdn, static, lms, vle, moodle, etc.

---

## FILE STRUCTURE

```
malicious_url_detection/
│
├── scripts/
│   ├── preprocessing.py       — dataset loading, cleaning, 1:1 balance
│   ├── split.py               — 70/15/15 stratified split
│   ├── feature_extraction.py  — 559 features, segment_url, decode_leet (UPDATED)
│   ├── train_model.py         — RF 300 trees, FPR-constrained threshold
│   ├── evaluate_model.py      — full evaluation + plots
│   ├── test_model.py          — 25-URL sanity check (update EXPECTED_HEURISTIC=57)
│   ├── generate_typosquatting.py — synthetic leet/typosquatting malicious URLs
│   ├── generate_whitelist.py  — generates models/whitelist.txt from top-1m.csv
│   └── diagnose_leet.py       — debug script for leet detection
│
├── app/
│   ├── app.py                 — Flask entry point (CORS allows chrome-extension://)
│   ├── routes.py              — API endpoints incl. /stats /history /stats/clear (UPDATED)
│   └── model_loader.py        — 4-layer pipeline + all fixes (UPDATED)
│
├── extension/
│   ├── manifest.json          — added webNavigation, web_accessible_resources (UPDATED)
│   ├── background.js          — blocks BEFORE page loads via onBeforeNavigate (UPDATED)
│   ├── popup.html             — added history button + View History button (UPDATED)
│   ├── popup.js               — history navigation, details button (UPDATED)
│   ├── content.js             — banner injection (unchanged)
│   ├── blocked.html           — NEW: malicious warning page with reasons + 2 buttons
│   └── history.html           — NEW: full dashboard with stats + searchable history
│
├── models/
│   ├── rf_model_latest.joblib — 559-feature RF model
│   ├── vectorizer_char.joblib
│   ├── vectorizer_word.joblib
│   ├── scaler.joblib
│   ├── threshold.json         — {"threshold": 0.47}
│   ├── lime_background.npz    — 500-sample background (559 features)
│   ├── whitelist.txt          — Tranco top 50k domains
│   └── domain_age_cache.json  — WHOIS cache (builds automatically)
│
├── data/
│   ├── raw/
│   │   ├── top-1m.csv         — Tranco top 1M list
│   │   ├── Phish.csv          — PhishTank phishing URLs
│   │   ├── urlhaus.txt        — URLhaus malware URLs
│   │   ├── openphish.txt      — OpenPhish active phishing
│   │   ├── data.csv           — faizann24 dataset
│   │   └── synthetic_malicious.csv — generated typosquatting/leet URLs
│   ├── processed/
│   │   └── cleaned_urls.csv
│   └── splits/
│       ├── train_urls.csv
│       ├── val_urls.csv
│       └── test_urls.csv
│
└── .env                       — GOOGLE_SAFE_BROWSING_API_KEY=your_key_here
```

---

## API ENDPOINTS

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness check, returns model status + threshold |
| `/predict` | POST | Classify single URL, saves to stats.json |
| `/predict/batch` | POST | Classify up to 500 URLs |
| `/explain` | POST | Classify + LIME explanation (num_features 1-30) |
| `/stats` | GET | Summary stats (total, malicious, benign, mal_rate) |
| `/history` | GET | Full URL history (limit, filter params) |
| `/stats/clear` | POST | Wipe all history |

---

## BROWSER EXTENSION — HOW IT WORKS

### Blocking flow (background.js):
1. `webNavigation.onBeforeNavigate` fires BEFORE page loads
2. Calls `/predict` API
3. If MALICIOUS → calls `/explain` for reasons → stores in
   `chrome.storage.session` as `block_{tabId}` → redirects to `blocked.html?tabId=X`
4. If BENIGN → allows navigation, fetches explanation quietly

### blocked.html:
- Reads block data from `chrome.storage.session` (NOT URL params — avoids
  encoding issues and length limits)
- Shows confidence score, blocked URL, brand impersonation warning, reasons list
- **"Take Me Back to Safety"** → `chrome.tabs.create({ url: "chrome://newtab" })`
- **"Proceed Anyway"** → sends `PROCEED_ANYWAY` message to background.js
  which adds URL to `_pendingChecks` set to bypass the intercept, then navigates

### history.html:
- Reads API base from `chrome.storage.sync` (matches user's popup setting)
- Fetches `/stats` and `/history` from Flask with no-cache headers
- Filter buttons: All / Malicious / Benign
- Search box: real-time URL filter
- Clear button: calls `/stats/clear`

### popup.html / popup.js:
- Clock icon in header → opens `history.html` in new tab
- "View History & Statistics" button → opens `history.html` in new tab
- Results cached in `chrome.storage.session` as `tab_{tabId}`
- LIME reasons pre-fetched by background.js and cached for instant display

---

## STATS TRACKING (routes.py + stats.json)

Every `/predict` call saves to `app/stats.json`:
```json
{
  "total": 150,
  "malicious": 23,
  "benign": 127,
  "history": [
    {
      "url": "http://g00gle.com/login",
      "prediction": "MALICIOUS",
      "confidence": 85.0,
      "source": "model",
      "timestamp": "2025-07-12 10:23:45",
      "brand": "Google"
    }
  ]
}
```
Thread-safe using `threading.Lock()`. Keeps last 1,000 entries.

---

## HOW TO RUN

```bash
cd "D:\Research Project\Research_Working_dir_2\malicious_url_detection"
.\venv\Scripts\activate

# Only if retraining needed (feature count changed):
python scripts/feature_extraction.py
python scripts/train_model.py
python scripts/evaluate_model.py

# Generate whitelist.txt (run once):
python scripts/generate_whitelist.py

# Pre-build LIME background (run once after retraining):
python -c "
import sys
sys.path.insert(0, 'app')
from model_loader import classifier
classifier._load_background()
print('Done')
"

# Start Flask
python app/app.py

# Load extension in Brave/Chrome:
# brave://extensions/ → Developer mode → Load unpacked → select extension/
```

---

## KNOWN ISSUES FIXED THIS SESSION

| Issue | Root Cause | Fix Applied |
|-------|-----------|-------------|
| g00gle.com → BENIGN 25.5% | Domain age × 0.3 crushing 0.85 override | Early return before domain age for leet |
| phishing.ru → BENIGN 19.1% | ML missed "phishing" in domain label | FIX 12: domain label word check |
| microsoft-support-center.com → BENIGN 44.6% | 44.6% just below 47% threshold | FIX 13: brand + no HTTPS override |
| 185.220.101.45 → BENIGN 8.8% | Reverse DNS resolved IP before ip_flag check | FIX 14 moved to predict_url() before DNS |
| mail.google.com → ML model | Simple whitelist only matched root domains | Smart whitelist with trusted subdomains |
| vle.uwu.ac.lk → MALICIOUS 70% | .ac.lk not whitelisted | Academic TLD whitelist rule |
| tinyurl.com → BENIGN (whitelist) | Tranco listed it | whitelist = whitelist - SHORTENERS |
| blocked.html buttons not working | URL params encoding issue + wrong tabId method | Data via chrome.storage.session |
| history.html stats not loading | Hardcoded API URL, no chrome.storage | Reads from chrome.storage.sync |
| LIME wrong reasons | Flag sanity not checking actual values | _flag_sanity dict + value validation |
| GSB API key not loading | Spaces around = in .env file | Fixed .env format |
| TRANCO_PATH wrong | Trailing space in "raw " directory | Fixed to "raw" |

---

## OUTSTANDING / FUTURE IMPROVEMENTS

1. Add `domain_age_days` as feature #58 — let RF learn age directly
2. Expand synthetic_malicious.csv with more multi-substitution leet variants
3. Add chart visualizations to history dashboard (pie/bar over time)
4. Pre-seed domain_age_cache.json with common domains
5. Comprehensive end-to-end user testing across all URL categories
6. Thesis documentation — feature engineering, evaluation results, system design

---

## TEST URLS (quick sanity check after any changes)

```
# Should be MALICIOUS (leet)
http://g00gle.com/login         → 85%+ MALICIOUS
http://paypa1.tk/secure         → 85%+ MALICIOUS
http://amaz0n-login.xyz/verify  → 85%+ MALICIOUS

# Should be MALICIOUS (other)
http://phishing.ru/             → 90% MALICIOUS
http://185.220.101.45/steal     → 85% MALICIOUS
http://microsoft-support-center.com/fix → 85% MALICIOUS
http://paypal-security-alert.com/verify → MALICIOUS

# Should be BENIGN (whitelist)
https://google.com              → BENIGN (whitelist)
https://mail.google.com/mail/   → BENIGN (whitelist)
https://docs.python.org/3/      → BENIGN (whitelist)
https://vle.uwu.ac.lk/          → BENIGN (whitelist - academic)

# Should be BENIGN (ML)
https://roadmap.sh              → BENIGN
https://neverssl.com            → BENIGN
```

---

## IMPORTANT NOTES FOR NEW CHAT

- **Do NOT run** `update_whitelist.py` — it is an old script incompatible with current architecture
- **Do NOT use** the old `model_loader.py` with hardcoded `LEET_BRAND_MAP` — replaced with dynamic decode
- **After any feature_extraction.py change** → must retrain model + delete lime_background.npz
- **test_model.py** has `EXPECTED_HEURISTIC = 57` (not 56) and `EXPECTED_TOTAL = 559` (not 558)
- **blocked.html** reads data from `chrome.storage.session` key `block_{tabId}` NOT URL params
- **history.html** reads API base from `chrome.storage.sync` NOT hardcoded localhost
- **stats.json** is saved in `app/stats.json` (same directory as app.py and routes.py)