/**
 * content.js — Optional warning banner injected into malicious pages.
 * Listens for SHOW_BANNER / HIDE_BANNER messages from background.js.
 */

"use strict";

const BANNER_ID = "__mud_warning_banner__";

function createBanner(confidence) {
  const existing = document.getElementById(BANNER_ID);
  if (existing) return; // already shown

  const banner = document.createElement("div");
  banner.id = BANNER_ID;
  banner.setAttribute("role", "alert");
  banner.innerHTML = `
    <span style="font-size:18px;line-height:1;">⚠</span>
    <span>
      <strong>Malicious URL Detected</strong> — This page may be dangerous.
      Confidence: ${confidence.toFixed(1)}%.
      Proceed with extreme caution.
    </span>
    <button id="${BANNER_ID}_close" aria-label="Dismiss warning">&times;</button>
  `;

  Object.assign(banner.style, {
    position       : "fixed",
    top            : "0",
    left           : "0",
    width          : "100%",
    zIndex         : "2147483647",
    background     : "#b91c1c",
    color          : "#fff",
    padding        : "10px 16px",
    display        : "flex",
    alignItems     : "center",
    gap            : "10px",
    fontFamily     : "system-ui, sans-serif",
    fontSize       : "13px",
    boxShadow      : "0 2px 12px rgba(0,0,0,.4)",
    boxSizing      : "border-box",
  });

  document.documentElement.prepend(banner);

  document.getElementById(`${BANNER_ID}_close`).addEventListener("click", () => {
    banner.remove();
  });
}

function removeBanner() {
  document.getElementById(BANNER_ID)?.remove();
}

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === "SHOW_BANNER") createBanner(msg.confidence ?? 0);
  if (msg.type === "HIDE_BANNER") removeBanner();
});
