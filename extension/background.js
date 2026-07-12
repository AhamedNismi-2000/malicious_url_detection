/**
 * background.js — Service worker for the Malicious URL Detector extension.
 *
 * BLOCKING FLOW:
 *   1. webNavigation.onBeforeNavigate fires BEFORE page loads
 *   2. Calls POST /predict immediately
 *   3. If MALICIOUS → redirect tab to blocked.html BEFORE the page loads
 *   4. If BENIGN    → allow navigation, fetch explanation in background
 *
 * FIX: reasons passed to blocked.html via chrome.storage.session
 *      (not URL params) to avoid URL length limits and encoding issues
 */

"use strict";

const DEFAULT_API = "http://localhost:5000";

const _pendingChecks = new Set();

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
  if (url.includes("blocked.html"))  return false;
  if (url.includes("history.html"))  return false;
  if (url.startsWith("http://localhost")) return false;
  return true;
}

function showNotification(confidence, reasons, brandDetected) {
  let message = `Confidence: ${confidence.toFixed(1)}%`;
  if (brandDetected) message = `Impersonating ${brandDetected} — ${message}`;
  const contextMsg = (reasons || []).slice(0, 2).join(" • ") || "Suspicious URL patterns detected";
  chrome.notifications.create({
    type             : "basic",
    iconUrl          : "icons/icon.png",
    title            : "⚠ Malicious URL Blocked",
    message          : message,
    contextMessage   : contextMsg,
    priority         : 2,
    requireInteraction: true,
  });
}

// ════════════════════════════════════════════════════════════════
// MAIN BLOCKING LOGIC
// ════════════════════════════════════════════════════════════════

chrome.webNavigation.onBeforeNavigate.addListener(async (details) => {
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
        await chrome.storage.session.set({ [`tab_${tabId}`]: result });
      }

      showNotification(result.confidence, reasons, brandDetected);

      // FIX: Store block data in session storage instead of URL params
      // URL params have length limits and encoding issues with JSON
      const blockKey = `block_${tabId}`;
      await chrome.storage.session.set({
        [blockKey]: {
          url          : url,
          confidence   : result.confidence,
          source       : result.source || "model",
          brand        : brandDetected || "",
          real         : result.real_domain || "",
          reasons      : reasons,
        }
      });

      // Redirect to blocked page — just pass tabId so blocked.html can read storage
      const blockedPageUrl = chrome.runtime.getURL(`blocked.html?tabId=${tabId}`);
      chrome.tabs.update(tabId, { url: blockedPageUrl });

    } else {
      // BENIGN — fetch explanation quietly
      if (result.source === "model") {
        const explained = await getExplanation(url, api);
        if (explained) {
          result.reasons     = explained.reasons || [];
          result.explanation = explained.explanation || [];
          await chrome.storage.session.set({ [`tab_${tabId}`]: result });
        }
      }
      chrome.tabs.sendMessage(tabId, { type: "HIDE_BANNER" }).catch(() => {});
    }

  } catch (err) {
    chrome.action.setBadgeText({ tabId, text: "?" });
    chrome.action.setBadgeBackgroundColor({ tabId, color: "#9E9E9E" });
    await chrome.storage.session.set({
      [`tab_${tabId}`]: {
        url, prediction: "UNKNOWN", confidence: 0,
        source: "error", error: err.message,
      },
    });
  } finally {
    _pendingChecks.delete(url);
  }
});

// ════════════════════════════════════════════════════════════════
// MESSAGE HANDLER
// ════════════════════════════════════════════════════════════════

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {

  // FIX: PROCEED_ANYWAY — use sender tab, bypass check
  if (message.type === "PROCEED_ANYWAY") {
    const { url, tabId } = message;
    _pendingChecks.add(url);
    const targetTab = tabId || (sender.tab ? sender.tab.id : null);
    if (targetTab) {
      chrome.tabs.update(targetTab, { url }, () => {
        setTimeout(() => _pendingChecks.delete(url), 5000);
      });
    } else {
      setTimeout(() => _pendingChecks.delete(url), 5000);
    }
    sendResponse({ ok: true });
    return true;
  }

  if (message.type === "GET_BLOCK_DATA") {
    const { tabId } = message;
    chrome.storage.session.get(`block_${tabId}`, (result) => {
      sendResponse(result[`block_${tabId}`] || null);
    });
    return true; // keep channel open for async response
  }

});

// ════════════════════════════════════════════════════════════════
// CLEANUP
// ════════════════════════════════════════════════════════════════

chrome.tabs.onRemoved.addListener((tabId) => {
  chrome.storage.session.remove([`tab_${tabId}`, `block_${tabId}`]);
});