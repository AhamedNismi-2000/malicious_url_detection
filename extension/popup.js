/**
 * popup.js — Fetches URL, prediction and LIME explanation from the Flask API
 * and renders everything in popup.html.
 */

"use strict";

const DEFAULT_API = "http://localhost:5000";

// ── Human-readable labels for feature names ───────────────────────────────────
const FEATURE_LABELS = {
  brand_in_domain         : "Brand impersonation",
  leet_in_domain          : "Leet-speak in domain",
  brand_hyphen_suspicious : "Suspicious brand-hyphen pattern",
  brand_mismatch          : "Brand–domain mismatch",
  risky_tld               : "Risky TLD (e.g. .tk, .xyz)",
  ip_flag                 : "IP address used as host",
  shortened               : "URL shortener detected",
  sus_words               : "Suspicious keywords",
  puny                    : "Punycode / IDN domain",
  punycode_suspicious     : "Suspicious punycode",
  homoglyph_suspicious    : "Homoglyph character",
  visual_brand_similarity : "Visual brand similarity",
  leet_speak_score        : "Leet-speak score",
  encoding_ratio          : "High encoding ratio",
  suspicious_port         : "Suspicious port number",
  subdomain_spam_score    : "Subdomain spam score",
  has_multi_subdomain     : "Multiple subdomains",
  susp_ext                : "Suspicious file extension",
  num_at                  : "@ symbol in URL",
  num_percent             : "Percent-encoding in URL",
  num_non_ascii           : "Non-ASCII characters",
  url_entropy             : "High URL entropy",
  ratio_digits            : "High digit ratio",
  url_len                 : "Very long URL",
  num_hyphens             : "Many hyphens",
  num_special             : "Many special characters",
  https_flag              : "HTTPS present",
};

function label(name) {
  return FEATURE_LABELS[name] || name.replace(/_/g, " ");
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
    body   : JSON.stringify({ url, num_features: 10 }),
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

  // Show top 3
  const top3 = explanation.slice(0, 3);
  const isMal = prediction === "MALICIOUS";

  const items = top3.map(({ feature, weight, value }) => {
    const positive    = weight > 0;
    const dotClass    = positive ? "mal" : "ben";
    const featureText = label(feature);
    const valText     = value === 1 ? "detected"
                      : value === 0 ? "not present"
                      : `value ${value.toFixed(2)}`;
    return `
      <div class="reason-pill">
        <div class="reason-dot ${dotClass}"></div>
        <div class="reason-name">${escapeHtml(featureText)}</div>
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
  // 1. Get the active tab's URL
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const url   = tab?.url;

  if (!url || (!url.startsWith("http://") && !url.startsWith("https://"))) {
    renderError("Not a web page — nothing to check.");
    return;
  }

  // 2. Check session cache first
  const cacheKey = `tab_${tab.id}`;
  let cached;
  try {
    const store = await chrome.storage.session.get(cacheKey);
    cached = store[cacheKey];
  } catch (_) { /* session storage may not be available */ }

  const apiBase = await getApiBase();

  // 3. Use cached prediction if available
  let prediction = cached;
  if (!cached || cached.url !== url) {
    try {
      prediction = await callPredict(url, apiBase);
      prediction.url = url;
    } catch (err) {
      renderError(`Could not reach API at ${apiBase}.\n${err.message}`);
      return;
    }
  }

  renderCard(prediction);

  // 4. Fetch LIME explanation if model-classified
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
  const panel = document.getElementById("settings-panel");
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
    // Re-run with new API base
    document.getElementById("main-content").innerHTML =
      '<div class="state-msg"><div class="spinner"></div>Checking URL…</div>';
    document.getElementById("explain-content").innerHTML = "";
    document.getElementById("footer-row").style.display = "none";
    run();
  });
});

// ── Boot ───────────────────────────────────────────────────────────────────────
run();
