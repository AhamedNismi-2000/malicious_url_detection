/**
 * popup.js — Renders prediction + cached natural language reasons.
 *
 * FIXES:
 *   - History button uses chrome.runtime.getURL directly (no message needed)
 *   - Details button always visible after result loads
 *   - Handles blocked.html and history.html gracefully
 */

"use strict";

const DEFAULT_API = "http://localhost:5000";

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
    body   : JSON.stringify({ url, num_features: 30 }),
  });
  if (!res.ok) throw new Error(`/explain returned HTTP ${res.status}`);
  return res.json();
}

// ── Open history in new tab ───────────────────────────────────────────────────
function openHistory() {
  const url = chrome.runtime.getURL("history.html");
  chrome.tabs.create({ url });
  window.close(); // close popup
}

// ── Render helpers ────────────────────────────────────────────────────────────

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

  let subtitle = "ML model classification";
  if (result.source === "whitelist")                 subtitle = "Trusted domain — whitelist";
  else if (result.source === "google_safe_browsing") subtitle = "Flagged by Google Safe Browsing";
  else if (result.source === "error")                subtitle = "API unreachable";
  else if (result.brand_detected)                    subtitle = `Impersonating ${result.brand_detected}`;

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

  // Show footer
  const footer = document.getElementById("footer-row");
  footer.style.display = "flex";
  document.getElementById("src-badge").textContent = result.source || "—";
  if (result.threshold != null) {
    document.getElementById("threshold-txt").textContent =
      `threshold ${result.threshold}%`;
  }

  // Show details button
  document.getElementById("details-btn").style.display = "flex";
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

function renderLoading() {
  document.getElementById("explain-content").innerHTML = `
    <div class="explain-section">
      <div class="explain-title">Analysing reasons…</div>
      <div style="padding:8px 0;color:var(--c-muted);font-size:12px;">
        <div class="spinner" style="width:16px;height:16px;margin:0 0 6px 0;"></div>
        Fetching explanation…
      </div>
    </div>`;
}

function renderError(msg) {
  document.getElementById("main-content").innerHTML = `
    <div class="state-msg">⚠ ${escapeHtml(msg)}</div>`;
  document.getElementById("explain-content").innerHTML = "";
  // Still show details button so user can check history
  document.getElementById("details-btn").style.display = "flex";
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
    document.getElementById("main-content").innerHTML = `
      <div class="state-msg">
        <div style="font-size:28px;margin-bottom:8px;">🛡</div>
        Open a website to check it
      </div>`;
    document.getElementById("details-btn").style.display = "flex";
    return;
  }

  // Don't try to classify our own pages
  const ownPages = ["blocked.html", "history.html"];
  if (ownPages.some(p => url.includes(p))) {
    document.getElementById("main-content").innerHTML = `
      <div class="state-msg">
        <div style="font-size:28px;margin-bottom:8px;">🛡</div>
        URL Detector is active
      </div>`;
    document.getElementById("details-btn").style.display = "flex";
    return;
  }

  const apiBase = await getApiBase();

  // ── Step 1: Check session cache ───────────────────────────────────────────
  let cached;
  try {
    const store = await chrome.storage.session.get(`tab_${tab.id}`);
    cached = store[`tab_${tab.id}`];
  } catch (_) {}

  if (cached && cached.url === url) {
    renderCard(cached);
    if (cached.prediction === "MALICIOUS") {
      if (cached.reasons && cached.reasons.length > 0) {
        renderReasons(cached.reasons, cached.prediction);
      } else {
        renderLoading();
        fetchAndRenderReasons(url, apiBase, cached);
      }
    }
    return;
  }

  // ── Step 2: No cache — fetch prediction ───────────────────────────────────
  let prediction;
  try {
    prediction     = await callPredict(url, apiBase);
    prediction.url = url;
  } catch (err) {
    renderError(
      `Could not reach API at ${apiBase}.\nMake sure Flask is running.\n\n${err.message}`
    );
    return;
  }

  renderCard(prediction);

  if (prediction.source === "model" && prediction.prediction === "MALICIOUS") {
    renderLoading();
    fetchAndRenderReasons(url, apiBase, prediction);
  }
}

async function fetchAndRenderReasons(url, apiBase, prediction) {
  try {
    const explained = await callExplain(url, apiBase);
    const reasons   = explained.reasons || [];

    if (explained.brand_detected && !prediction.brand_detected) {
      prediction.brand_detected = explained.brand_detected;
      prediction.real_domain    = explained.real_domain;
      renderCard(prediction);
    }

    renderReasons(reasons, prediction.prediction);

    // Update session cache
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      const store = await chrome.storage.session.get(`tab_${tab.id}`);
      const entry = store[`tab_${tab.id}`] || prediction;
      entry.reasons = reasons;
      if (explained.brand_detected) {
        entry.brand_detected = explained.brand_detected;
        entry.real_domain    = explained.real_domain;
      }
      await chrome.storage.session.set({ [`tab_${tab.id}`]: entry });
    } catch (_) {}

  } catch (_) {
    document.getElementById("explain-content").innerHTML = "";
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
    document.getElementById("footer-row").style.display  = "none";
    document.getElementById("details-btn").style.display = "none";
    run();
  });
});

// ── History / Details buttons ─────────────────────────────────────────────────

document.getElementById("history-btn").addEventListener("click", openHistory);
document.getElementById("details-btn").addEventListener("click", openHistory);

// ── Boot ──────────────────────────────────────────────────────────────────────
run();