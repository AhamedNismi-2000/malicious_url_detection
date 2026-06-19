/**
 * popup.js — Fetches URL, prediction and LIME explanation from the Flask API
 * and renders everything in popup.html.
 */

"use strict";

const DEFAULT_API = "http://localhost:5000";

// ── Human-readable labels for heuristic feature names ────────────────────────
const FEATURE_LABELS = {
  // Brand / impersonation
  brand_in_domain         : "Brand impersonation detected",
  brand_hyphen_suspicious : "Fake brand domain pattern",
  brand_mismatch          : "Brand used outside real domain",
  leet_in_domain          : "Disguised brand name (e.g. amaz0n)",
  visual_brand_similarity : "Visually similar to known brand",
  homoglyph_suspicious    : "Look-alike characters detected",
  punycode_suspicious     : "Internationalized domain trick",
  puny                    : "Punycode domain detected",

  // Domain / TLD
  risky_tld               : "Suspicious domain ending (.tk .xyz)",
  ip_flag                 : "IP address used as domain",
  shortened               : "URL shortener detected",
  suspicious_port         : "Unusual port number",
  has_multi_subdomain     : "Excessive subdomains",
  subdomain_spam_score    : "Subdomain spam pattern",
  tld_len                 : "Unusually long domain suffix",

  // URL structure
  url_len                 : "Unusually long URL",
  num_hyphens             : "Excessive hyphens in URL",
  num_at                  : "@ symbol in URL",
  num_percent             : "Percent-encoding in URL",
  num_non_ascii           : "Non-standard characters in URL",
  num_special             : "Excessive special characters",
  url_entropy             : "Randomly generated domain",
  ratio_digits            : "High proportion of digits",
  encoding_ratio          : "Heavy URL encoding",
  susp_ext                : "Suspicious file extension",

  // Content signals
  sus_words               : "Phishing keywords found",
  leet_speak_score        : "Leet-speak obfuscation",

  // HTTPS — direction depends on weight sign, handled in render
  https_flag              : "Missing HTTPS security",
};

// Features to always hide from the popup (n-gram TF-IDF — not user-readable)
function isInterpretable(featureName) {
  return (
    !featureName.startsWith("char_") &&
    !featureName.startsWith("word_")
  );
}

function friendlyLabel(feature, weight) {
  // Special case: https_flag with positive weight means HTTP (no HTTPS) is suspicious
  if (feature === "https_flag") {
    return weight > 0 ? "Missing HTTPS security" : "HTTPS present (safe signal)";
  }
  return FEATURE_LABELS[feature] || feature.replace(/_/g, " ");
}

// ── API helpers ───────────────────────────────────────────────────────────────

async function getApiBase() {
  return new Promise((resolve) =>
    chrome.storage.sync.get({ apiBase: DEFAULT_API }, (s) =>
      resolve((s.apiBase || DEFAULT_API).replace(/\/$/, ""))
    )
  );
}

async function callPredict(url, apiBase) {
  const res = await fetch(`${apiBase}/predict`, {
    method : "POST",
    headers: { "Content-Type": "application/json" },
    body   : JSON.stringify({ url }),
  });
  if (!res.ok) throw new Error(`/predict returned HTTP ${res.status}`);
  return res.json();
}

async function callExplain(url, apiBase) {
  const res = await fetch(`${apiBase}/explain`, {
    method : "POST",
    headers: { "Content-Type": "application/json" },
    body   : JSON.stringify({ url, num_features: 20 }),
  });
  if (!res.ok) throw new Error(`/explain returned HTTP ${res.status}`);
  return res.json();
}

// ── Render helpers ─────────────────────────────────────────────────────────────

function verdictClass(prediction) {
  if (prediction === "MALICIOUS") return "malicious";
  if (prediction === "BENIGN")    return "benign";
  return "unknown";
}

function verdictEmoji(prediction) {
  if (prediction === "MALICIOUS") return "⚠";
  if (prediction === "BENIGN")    return "✓";
  return "?";
}

function renderCard(result) {
  const vc         = verdictClass(result.prediction);
  const confidence = result.confidence ?? 0;
  const truncUrl   = (result.url || "").length > 80
    ? result.url.slice(0, 77) + "…"
    : result.url || "—";

  document.getElementById("main-content").innerHTML = `
    <div class="card ${vc}">
      <div class="verdict ${vc}">
        <div class="verdict-icon">${verdictEmoji(result.prediction)}</div>
        <div>
          <div class="verdict-label">${result.prediction || "UNKNOWN"}</div>
          <div class="verdict-sub">${
            result.source === "whitelist"
              ? "Trusted domain — whitelist"
              : result.source === "error"
              ? "API unreachable"
              : "ML model classification"
          }</div>
        </div>
      </div>

      ${result.source !== "whitelist" && result.source !== "invalid" ? `
      <div class="conf-row ${vc}">
        <div class="conf-labels">
          <span>Confidence</span>
          <span class="conf-value">${confidence.toFixed(1)}%</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="width:${confidence}%"></div>
        </div>
      </div>` : ""}

      <div class="url-row">
        <div class="url-label">URL</div>
        <div class="url-text">${escapeHtml(truncUrl)}</div>
      </div>
    </div>
  `;

  // Footer
  const footer = document.getElementById("footer-row");
  footer.style.display = "flex";
  document.getElementById("src-badge").textContent = result.source || "—";
  if (result.threshold != null) {
    document.getElementById("threshold-txt").textContent =
      `threshold ${result.threshold}%`;
  }
}

function renderExplanation(explanation, prediction) {
  const ec = document.getElementById("explain-content");
  if (!explanation || explanation.length === 0) {
    ec.innerHTML = "";
    return;
  }

  // Filter out n-gram features (char_*, word_*) — not user-readable
  // Then take top 3 by absolute weight
  const top3 = explanation
    .filter(f => isInterpretable(f.feature))
    .slice(0, 3);

  if (top3.length === 0) {
    ec.innerHTML = "";
    return;
  }

  const isMal = prediction === "MALICIOUS";

  const items = top3.map(({ feature, weight, value }) => {
    const positive    = weight > 0;
    const dotClass    = positive ? "mal" : "ben";
    const labelText   = friendlyLabel(feature, weight);

    // Value display — show meaningful context instead of raw numbers
    let valText;
    if (value === 1 || value > 0.9)      valText = "detected";
    else if (value === 0 || value < 0.1) valText = "not present";
    else                                  valText = `score ${value.toFixed(2)}`;

    return `
      <div class="reason-pill">
        <div class="reason-dot ${dotClass}"></div>
        <div class="reason-name">${escapeHtml(labelText)}</div>
        <div class="reason-meta">${escapeHtml(valText)}</div>
      </div>`;
  }).join("");

  ec.innerHTML = `
    <div class="explain-section">
      <div class="explain-title">
        ${isMal ? "Why malicious?" : "Why benign?"}
      </div>
      <div class="explain-list">${items}</div>
    </div>
  `;
}

function renderError(msg) {
  document.getElementById("main-content").innerHTML = `
    <div class="state-msg">
      ⚠ ${escapeHtml(msg)}
    </div>`;
  document.getElementById("explain-content").innerHTML = "";
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// ── Main flow ─────────────────────────────────────────────────────────────────

async function run() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const url   = tab?.url;

  if (!url || (!url.startsWith("http://") && !url.startsWith("https://"))) {
    renderError("Not a web page — nothing to check.");
    return;
  }

  const apiBase = await getApiBase();

  // Check session cache first
  let prediction;
  try {
    const store = await chrome.storage.session.get(`tab_${tab.id}`);
    const cached = store[`tab_${tab.id}`];
    if (cached && cached.url === url) {
      prediction = cached;
    }
  } catch (_) {}

  if (!prediction) {
    try {
      prediction      = await callPredict(url, apiBase);
      prediction.url  = url;
    } catch (err) {
      renderError(`Could not reach API at ${apiBase}.\nMake sure Flask is running.`);
      return;
    }
  }

  renderCard(prediction);

  // Fetch LIME explanation for model-classified URLs only
  if (prediction.source === "model") {
    try {
      const explained = await callExplain(url, apiBase);
      renderExplanation(explained.explanation, prediction.prediction);
    } catch (_) {
      // Explanation is best-effort — silently ignore
    }
  }
}

// ── Settings panel ─────────────────────────────────────────────────────────────

document.getElementById("settings-toggle").addEventListener("click", () => {
  const panel   = document.getElementById("settings-panel");
  const visible = panel.style.display === "block";
  panel.style.display = visible ? "none" : "block";
  if (!visible) {
    chrome.storage.sync.get({ apiBase: DEFAULT_API }, (s) => {
      document.getElementById("api-url").value = s.apiBase;
    });
  }
});

document.getElementById("save-settings").addEventListener("click", () => {
  const val = document.getElementById("api-url").value.trim() || DEFAULT_API;
  chrome.storage.sync.set({ apiBase: val }, () => {
    document.getElementById("settings-panel").style.display = "none";
    document.getElementById("main-content").innerHTML =
      '<div class="state-msg"><div class="spinner"></div>Checking URL…</div>';
    document.getElementById("explain-content").innerHTML = "";
    document.getElementById("footer-row").style.display = "none";
    run();
  });
});

// ── Boot ───────────────────────────────────────────────────────────────────────
run();