/**
 * background.js — Service worker for the Malicious URL Detector extension.
 *
 * On every tab navigation:
 *   1. Calls POST /predict to classify the URL
 *   2. If MALICIOUS — calls POST /explain to get natural language reasons
 *   3. Caches full result (including reasons) in chrome.storage.session
 *   4. Updates badge
 *   5. Shows Chrome notification + content script banner automatically
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
  return url && (url.startsWith("http://") || url.startsWith("https://"));
}

function showNotification(url, confidence, reasons, brandDetected) {
  let message = `Confidence: ${confidence.toFixed(1)}%`;
  if (brandDetected) message = `Impersonating ${brandDetected} — ${message}`;

  const topReasons = (reasons || []).slice(0, 2);
  const contextMsg = topReasons.length > 0
    ? topReasons.join(" • ")
    : "Suspicious URL patterns detected";

  chrome.notifications.create({
    type             : "basic",
    iconUrl          : "icons/icon128.png",
    title            : "⚠ Malicious URL Detected",
    message          : message,
    contextMessage   : contextMsg,
    priority         : 2,
    requireInteraction: false,
  });
}

chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  if (changeInfo.status !== "complete") return;
  const url = tab.url || changeInfo.url;
  if (!isClassifiable(url)) return;

  try {
    const api    = await getApiBase();
    const result = await classifyUrl(url, api);
    result.url   = url;

    let reasons       = [];
    let brandDetected = result.brand_detected || null;

    // Fetch explanation immediately in background for MALICIOUS URLs
    if (result.prediction === "MALICIOUS" && result.source === "model") {
      const explained = await getExplanation(url, api);
      if (explained) {
        reasons                = explained.reasons || [];
        brandDetected          = explained.brand_detected || brandDetected;
        result.reasons         = reasons;
        result.brand_detected  = brandDetected;
        result.real_domain     = explained.real_domain || result.real_domain;
        result.explanation     = explained.explanation || [];
      }
    }

    // Cache complete result including reasons
    await chrome.storage.session.set({ [`tab_${tabId}`]: result });

    setBadge(tabId, result.prediction);

    if (result.prediction === "MALICIOUS") {
      showNotification(url, result.confidence, reasons, brandDetected);

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

chrome.tabs.onRemoved.addListener((tabId) => {
  chrome.storage.session.remove(`tab_${tabId}`);
});