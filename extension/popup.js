/**
 * popup.js — Fetches prediction and LIME explanation, renders user-friendly
 * natural language reasons instead of technical feature names.
 */

"use strict";

const DEFAULT_API = "http://localhost:5000";

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

  // Source subtitle
  let subtitle = "ML model classification";
  if (result.source === "whitelist")    subtitle = "Trusted domain — whitelist";
  else if (result.source === "error")   subtitle = "API unreachable";
  else if (result.brand_detected)       subtitle = `Impersonating ${result.brand_detected}`;

  document.getElementById("main-content").innerHTML = `
    <div class="card ${vc}">
      <div class="verdict ${vc}">
        <div class="verdict-icon">${verdictEmoji(result.prediction)}</div>
        <div>
          <div class="verdict-label">${result.prediction || "UNKNOWN"}</div>
          <div class="verdict-sub">${escapeHtml(subtitle)}</div>
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

      ${result.brand_detected ? `
      <div class="brand-row">
        <span class="brand-icon">⚠</span>
        <span>Real website: <strong>${escapeHtml(result.real_domain || "")}</strong></span>
      </div>` : ""}

      <div class="url-row">
        <div class="url-label">URL</div>
        <div class="url-text">${escapeHtml(truncUrl)}</div>
      </div>

      ${result.unshortened ? `
      <div class="url-row" style="border-top:1px solid var(--c-border);">
        <div class="url-label">Resolved to</div>
        <div class="url-text">${escapeHtml(result.unshortened.slice(0, 80))}</div>
      </div>` : ""}
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

function renderReasons(reasons, prediction) {
  const ec = document.getElementById("explain-content");

  if (!reasons || reasons.length === 0) {
    ec.innerHTML = "";
    return;
  }

  const isMal  = prediction === "MALICIOUS";
  const title  = isMal ? "Why is this dangerous?" : "Why is this safe?";
  const dotCls = isMal ? "mal" : "ben";

  const items = reasons.map((reason, i) => `
    <div class="reason-pill">
      <div class="reason-dot ${dotCls}">${i + 1}</div>
      <div class="reason-text">${escapeHtml(reason)}</div>
    </div>
  `).join("");

  ec.innerHTML = `
    <div class="explain-section">
      <div class="explain-title">${title}</div>
      <div class="explain-list">${items}</div>
    </div>
  `;
}

function renderError(msg) {
  document.getElementById("main-content").innerHTML = `
    <div class="state-msg">⚠ ${escapeHtml(msg)}</div>`;
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

  // Check session cache
  let prediction;
  try {
    const store  = await chrome.storage.session.get(`tab_${tab.id}`);
    const cached = store[`tab_${tab.id}`];
    if (cached && cached.url === url) prediction = cached;
  } catch (_) {}

  if (!prediction) {
    try {
      prediction     = await callPredict(url, apiBase);
      prediction.url = url;
    } catch (err) {
      renderError(`Could not reach API at ${apiBase}.\nMake sure Flask is running.`);
      return;
    }
  }

  renderCard(prediction);

  // Fetch LIME explanation for model-classified URLs
  if (prediction.source === "model") {
    try {
      const explained = await callExplain(url, apiBase);

      // Use natural language reasons from backend
      const reasons = explained.reasons || [];

      // Update brand info if explanation has it
      if (explained.brand_detected && !prediction.brand_detected) {
        prediction.brand_detected = explained.brand_detected;
        prediction.real_domain    = explained.real_domain;
        renderCard(prediction);   // re-render with brand info
      }

      renderReasons(reasons, prediction.prediction);
    } catch (_) {
      // Best-effort — silently ignore
    }
  }
}

// ── Settings panel ────────────────────────────────────────────────────────────

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

// ── Boot ──────────────────────────────────────────────────────────────────────
run();
