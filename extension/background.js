/**
 * background.js — Service worker for the Malicious URL Detector extension.
 *
 * On every tab navigation:
 *   1. Calls POST /predict to classify the URL.
 *   2. Stores result in chrome.storage.session keyed by tabId.
 *   3. Updates the extension badge.
 *   4. If MALICIOUS — auto shows Chrome notification + sends banner to content.js
 */

"use strict";

const DEFAULT_API = "http://localhost:5000";

// ── Helpers ───────────────────────────────────────────────────────────────────

async function getApiBase() {
  return new Promise((resolve) =>
    chrome.storage.sync.get({ apiBase: DEFAULT_API }, (s) =>
      resolve((s.apiBase || DEFAULT_API).replace(/\/$/, ""))
    )
  );
}

async function classifyUrl(url) {
  const api = await getApiBase();
  const res  = await fetch(`${api}/predict`, {
    method : "POST",
    headers: { "Content-Type": "application/json" },
    body   : JSON.stringify({ url }),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function getExplanation(url) {
  const api = await getApiBase();
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
  const isMalicious = prediction === "MALICIOUS";
  chrome.action.setBadgeText({ tabId, text: isMalicious ? "!" : "✓" });
  chrome.action.setBadgeBackgroundColor({
    tabId,
    color: isMalicious ? "#E53935" : "#43A047",
  });
}

function isClassifiable(url) {
  if (!url) return false;
  return url.startsWith("http://") || url.startsWith("https://");
}

// ── Auto notification when MALICIOUS ─────────────────────────────────────────

function showMaliciousNotification(url, confidence, reasons, brandDetected) {
  // Build notification message
  let message = `Confidence: ${confidence.toFixed(1)}%`;
  if (brandDetected) {
    message = `Impersonating ${brandDetected} — ${message}`;
  }

  // Build reason lines (top 2 from natural language reasons)
  const topReasons = (reasons || []).slice(0, 2);
  const reasonText = topReasons.length > 0
    ? topReasons.join(" • ")
    : "Suspicious URL patterns detected";

  chrome.notifications.create({
    type    : "basic",
    iconUrl : "icons/icon128.png",
    title   : "⚠ Malicious URL Detected",
    message : message,
    contextMessage: reasonText,
    priority: 2,
    requireInteraction: false,
  });
}

// ── Tab listeners ─────────────────────────────────────────────────────────────

chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  if (changeInfo.status !== "complete") return;
  const url = tab.url || changeInfo.url;
  if (!isClassifiable(url)) return;

  try {
    // Step 1: Get prediction
    const result  = await classifyUrl(url);
    result.url    = url;

    // Step 2: If malicious, get explanation for notification
    let reasons      = [];
    let brandDetected = result.brand_detected || null;

    if (result.prediction === "MALICIOUS" && result.source === "model") {
      const explained = await getExplanation(url);
      if (explained) {
        reasons       = explained.reasons || [];
        brandDetected = explained.brand_detected || brandDetected;
        result.reasons = reasons;
      }
    }

    // Step 3: Cache result
    await chrome.storage.session.set({ [`tab_${tabId}`]: result });

    // Step 4: Update badge
    setBadge(tabId, result.prediction);

    // Step 5: Auto notification + banner if MALICIOUS
    if (result.prediction === "MALICIOUS") {

      // Chrome notification (top-right corner)
      showMaliciousNotification(
        url,
        result.confidence,
        reasons,
        brandDetected
      );

      // Content script banner (on the page itself)
      chrome.tabs.sendMessage(tabId, {
        type         : "SHOW_BANNER",
        prediction   : result.prediction,
        confidence   : result.confidence,
        source       : result.source,
        reasons      : reasons,
        brandDetected: brandDetected,
      }).catch(() => {});

    } else {
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
  }
});

// Clean up on tab close
chrome.tabs.onRemoved.addListener((tabId) => {
  chrome.storage.session.remove(`tab_${tabId}`);
});
