/**
 * blocked.js — Logic for blocked.html.
 * Extracted from an inline <script> tag because Manifest V3's default CSP
 * (script-src 'self') silently blocks inline scripts on extension pages.
 * That was the actual root cause of "buttons don't work / prediction
 * never shows" — the old inline script simply never executed.
 */

"use strict";

const params = new URLSearchParams(window.location.search);
const tabId  = parseInt(params.get("tabId") || "0");

let _blockedUrl = "";
let _confidence = 0;

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function loadBlockData() {
  chrome.storage.session.get(`block_${tabId}`, function (result) {
    const data = result[`block_${tabId}`];

    if (!data) {
      // Fallback: try URL params directly (older format)
      _blockedUrl = params.get("url") || "";
      _confidence = parseFloat(params.get("confidence") || "0");
      const brand      = params.get("brand") || "";
      const realDomain = params.get("real")  || "";
      let reasons = [];
      try { reasons = JSON.parse(params.get("reasons") || "[]"); } catch (_) {}
      renderData({ url: _blockedUrl, confidence: _confidence, brand, real: realDomain, reasons });
      return;
    }

    _blockedUrl = data.url        || "";
    _confidence = data.confidence || 0;
    renderData(data);
  });
}

// ── Voice alert (Web Speech API) ────────────────────────────────────────────
// Respects the user's "voiceAlertsEnabled" setting (default: on).
function speakWarning(data) {
  if (!("speechSynthesis" in window)) return;

  chrome.storage.sync.get({ voiceAlertsEnabled: true }, function (s) {
    if (!s.voiceAlertsEnabled) return;

    let message = "Warning. This website has been identified as malicious and blocked.";
    if (data.brand) {
      message = `Warning. This site is impersonating ${data.brand} and has been blocked.`;
    }

    try {
      speechSynthesis.cancel(); // avoid overlapping utterances on repeat blocks
      const utterance = new SpeechSynthesisUtterance(message);
      utterance.rate   = 1.1;
      utterance.pitch  = 2;
      utterance.volume = 0.8;
      speechSynthesis.speak(utterance);
    } catch (_) {
      // Speech synthesis unavailable or blocked — fail silently,
      // the visual warning is already sufficient.
    }
  });
}

function renderData(data) {
  document.getElementById("loading-state").style.display = "none";
  document.getElementById("content").style.display       = "block";

  speakWarning(data);

  document.getElementById("conf-pct").textContent =
    (_confidence || data.confidence || 0).toFixed(1) + "%";
  setTimeout(() => {
    document.getElementById("conf-bar").style.width =
      Math.min(data.confidence || 0, 100) + "%";
  }, 100);

  document.getElementById("blocked-url").textContent = data.url || "—";

  if (data.brand) {
    document.getElementById("brand-box").style.display = "flex";
    document.getElementById("brand-name").textContent  = data.brand;
    document.getElementById("brand-real").textContent  = data.real || (data.brand + ".com");
  }

  const reasons = data.reasons || [];
  if (reasons.length > 0) {
    document.getElementById("reasons-section").style.display = "block";
    const list = document.getElementById("reason-list");
    reasons.forEach(function (r, i) {
      const item = document.createElement("div");
      item.className = "reason-item";
      item.innerHTML =
        '<div class="reason-num">' + (i + 1) + '</div>' +
        '<div class="reason-text">' + escapeHtml(String(r)) + '</div>';
      list.appendChild(item);
    });
  }
}

function doProceeed() {
  const confirmed = confirm(
    "⚠ WARNING\n\n" +
    "This site was classified as MALICIOUS.\n\n" +
    "Proceeding may expose you to phishing, malware, or data theft.\n\n" +
    "Are you sure you want to continue?"
  );
  if (!confirmed) return;

  chrome.runtime.sendMessage(
    { type: "PROCEED_ANYWAY", url: _blockedUrl, tabId: tabId },
    function (response) {
      if (chrome.runtime.lastError) {
        window.location.href = _blockedUrl;
      }
    }
  );
}

document.getElementById("btn-back").addEventListener("click", function () {
  chrome.tabs.create({ url: "chrome://newtab" });
});

document.getElementById("btn-proceed").addEventListener("click", function () {
  if (!_blockedUrl) {
    chrome.storage.session.get("block_" + tabId, function (result) {
      const data = result["block_" + tabId];
      if (data && data.url) {
        _blockedUrl = data.url;
        doProceeed();
      }
    });
    return;
  }
  doProceeed();
});

loadBlockData();