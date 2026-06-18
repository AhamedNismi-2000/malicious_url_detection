/**
 * background.js — Service worker for the Malicious URL Detector extension.
 *
 * On every tab navigation:
 *   1. Calls POST /predict to classify the URL.
 *   2. Stores the result in chrome.storage.session keyed by tabId.
 *   3. Updates the extension badge (R = red malicious, G = green benign).
 *   4. If malicious, sends a message to the content script to show a banner.
 */

const DEFAULT_API = "http://localhost:5000";

// ── Helpers ──────────────────────────────────────────────────────────────────

async function getApiBase() {
  return new Promise((resolve) => {
    chrome.storage.sync.get({ apiBase: DEFAULT_API }, (s) =>
      resolve(s.apiBase.replace(/\/$/, ""))
    );
  });
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

function setBadge(tabId, prediction) {
  const isMalicious = prediction === "MALICIOUS";
  chrome.action.setBadgeText ({ tabId, text: isMalicious ? "!" : "✓" });
  chrome.action.setBadgeBackgroundColor({
    tabId,
    color: isMalicious ? "#E53935" : "#43A047",
  });
}

function isClassifiable(url) {
  if (!url) return false;
  return url.startsWith("http://") || url.startsWith("https://");
}

// ── Tab listeners ─────────────────────────────────────────────────────────────

chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  // Only fire once per navigation, when the URL is committed
  if (changeInfo.status !== "complete") return;
  const url = tab.url || changeInfo.url;
  if (!isClassifiable(url)) return;

  try {
    const result = await classifyUrl(url);
    result.url   = url;                         // ensure url is in result

    // Cache in session storage (cleared when browser closes)
    await chrome.storage.session.set({ [`tab_${tabId}`]: result });

    setBadge(tabId, result.prediction);

    // Notify content script if malicious
    if (result.prediction === "MALICIOUS") {
      chrome.tabs.sendMessage(tabId, {
        type      : "SHOW_BANNER",
        prediction: result.prediction,
        confidence: result.confidence,
        source    : result.source,
      }).catch(() => {/* content script may not be ready */});
    } else {
      chrome.tabs.sendMessage(tabId, { type: "HIDE_BANNER" })
        .catch(() => {});
    }
  } catch (err) {
    // API unreachable — clear badge
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
