/**
 * background.js — Service worker for the Malicious URL Detector extension.
 *
 * BLOCKING FLOW (new):
 *   1. webNavigation.onBeforeNavigate fires BEFORE page loads
 *   2. Calls POST /predict immediately
 *   3. If MALICIOUS → redirect tab to blocked.html BEFORE the page loads
 *   4. If BENIGN    → allow navigation, fetch explanation in background
 *
 * HISTORY (new):
 *   Every URL checked is recorded via POST /predict which saves to stats.json
 *
 * POPUP FLOW (unchanged):
 *   Results cached in chrome.storage.session for instant popup display
 */

"use strict";

const DEFAULT_API = "http://localhost:5000";

// URLs currently being checked — prevent double-checking blocked.html itself
const _pendingChecks = new Set();
const _checkedUrls   = new Map(); // url → result cache (session)

async function getApiBase() {
  return new Promise((resolve) =>
    chrome.storage.sync.get({ apiBase: DEFAULT_API }, (s) =>
      resolve((s.apiBase || DEFAULT_API).replace(/\/$/, ""))
    )
  );
}

async function classifyUrl(url, api) {
  const res = await fetch(`${api}/predict`, {
    method : "POST",
    headers: { "Content-Type": "application/json" },
    body   : JSON.stringify({ url }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function getExplanation(url, api) {
  try {
    const res = await fetch(`${api}/explain`, {
      method : "POST",
      headers: { "Content-Type": "application/json" },
      body   : JSON.stringify({ url, num_features: 30 }),
    });
    if (!res.ok) return null;
    return res.json();
  } catch (_) {
    return null;
  }
}

function setBadge(tabId, prediction) {
  const isMal = prediction === "MALICIOUS";
  chrome.action.setBadgeText({ tabId, text: isMal ? "!" : "✓" });
  chrome.action.setBadgeBackgroundColor({
    tabId,
    color: isMal ? "#E53935" : "#43A047",
  });
}

function isClassifiable(url) {
  if (!url) return false;
  if (!url.startsWith("http://") && !url.startsWith("https://")) return false;
  // Don't check our own extension pages
  if (url.includes("blocked.html") || url.includes("history.html")) return false;
  // Don't check localhost (Flask API itself)
  if (url.startsWith("http://localhost")) return false;
  return true;
}

function showNotification(url, confidence, reasons, brandDetected) {
  let message = `Confidence: ${confidence.toFixed(1)}%`;
  if (brandDetected) message = `Impersonating ${brandDetected} — ${message}`;

  const topReasons  = (reasons || []).slice(0, 2);
  const contextMsg  = topReasons.length > 0
    ? topReasons.join(" • ")
    : "Suspicious URL patterns detected";

  chrome.notifications.create({
    type             : "basic",
    iconUrl          : "icons/icon128.png",
    title            : "⚠ Malicious URL Blocked",
    message          : message,
    contextMessage   : contextMsg,
    priority         : 2,
    requireInteraction: true,
  });
}

function buildBlockedUrl(tabId, originalUrl, result) {
  const params = new URLSearchParams({
    url       : originalUrl,
    confidence: result.confidence || 0,
    source    : result.source || "model",
    brand     : result.brand_detected || "",
    real      : result.real_domain || "",
    reasons   : JSON.stringify(result.reasons || []),
    tabId     : tabId,
  });
  return chrome.runtime.getURL(`blocked.html?${params.toString()}`);
}

// ════════════════════════════════════════════════════════════════
// MAIN BLOCKING LOGIC — fires BEFORE page loads
// ════════════════════════════════════════════════════════════════

chrome.webNavigation.onBeforeNavigate.addListener(async (details) => {
  // Only main frame navigations (not iframes)
  if (details.frameId !== 0) return;

  const url   = details.url;
  const tabId = details.tabId;

  if (!isClassifiable(url)) return;
  if (_pendingChecks.has(url)) return;

  _pendingChecks.add(url);

  try {
    const api    = await getApiBase();
    const result = await classifyUrl(url, api);
    result.url   = url;

    // Cache result for popup
    await chrome.storage.session.set({ [`tab_${tabId}`]: result });
    setBadge(tabId, result.prediction);

    if (result.prediction === "MALICIOUS") {
      // ── BLOCK: redirect to warning page BEFORE site loads ──
      let reasons       = result.reasons || [];
      let brandDetected = result.brand_detected || null;

      // Fetch explanation for better reasons
      const explained = await getExplanation(url, api);
      if (explained) {
        reasons                = explained.reasons || reasons;
        brandDetected          = explained.brand_detected || brandDetected;
        result.reasons         = reasons;
        result.brand_detected  = brandDetected;
        result.real_domain     = explained.real_domain || result.real_domain;
        result.explanation     = explained.explanation || [];
        // Update cache with full result
        await chrome.storage.session.set({ [`tab_${tabId}`]: result });
      }

      // Show notification
      showNotification(url, result.confidence, reasons, brandDetected);

      // Redirect to blocked page
      const blockedUrl = buildBlockedUrl(tabId, url, result);
      chrome.tabs.update(tabId, { url: blockedUrl });

    } else {
      // BENIGN — allow navigation, fetch explanation quietly
      if (result.source === "model") {
        const explained = await getExplanation(url, api);
        if (explained) {
          result.reasons    = explained.reasons || [];
          result.explanation = explained.explanation || [];
          await chrome.storage.session.set({ [`tab_${tabId}`]: result });
        }
      }
      // Send hide banner message (in case content.js is loaded)
      chrome.tabs.sendMessage(tabId, { type: "HIDE_BANNER" }).catch(() => {});
    }

  } catch (err) {
    chrome.action.setBadgeText({ tabId, text: "?" });
    chrome.action.setBadgeBackgroundColor({ tabId, color: "#9E9E9E" });
    await chrome.storage.session.set({
      [`tab_${tabId}`]: {
        url,
        prediction: "UNKNOWN",
        confidence: 0,
        source    : "error",
        error     : err.message,
      },
    });
  } finally {
    _pendingChecks.delete(url);
  }
});

// ════════════════════════════════════════════════════════════════
// HANDLE PROCEED ANYWAY MESSAGE from blocked.html
// ════════════════════════════════════════════════════════════════

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.type === "PROCEED_ANYWAY") {
    const { tabId, url } = message;
    // Add to bypass list temporarily
    _pendingChecks.add(url);
    chrome.tabs.update(tabId, { url }, () => {
      // Remove from bypass after a short delay
      setTimeout(() => _pendingChecks.delete(url), 3000);
    });
    sendResponse({ ok: true });
  }

  if (message.type === "GET_HISTORY_URL") {
    sendResponse({ url: chrome.runtime.getURL("history.html") });
  }
});

// ════════════════════════════════════════════════════════════════
// CLEANUP on tab close
// ════════════════════════════════════════════════════════════════

chrome.tabs.onRemoved.addListener((tabId) => {
  chrome.storage.session.remove(`tab_${tabId}`);
});