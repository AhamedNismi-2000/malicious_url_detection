/**
 * content.js — Enhanced warning banner injected into malicious pages.
 * Shows auto-dismissing banner with natural language reasons and brand name.
 */

"use strict";

const BANNER_ID    = "__mud_warning_banner__";
const REASONS_ID   = "__mud_warning_reasons__";
const AUTO_DISMISS = 15000;   // auto-dismiss after 15 seconds

function createBanner(confidence, reasons, brandDetected) {
  // Remove existing banner
  document.getElementById(BANNER_ID)?.remove();

  const banner = document.createElement("div");
  banner.id = BANNER_ID;
  banner.setAttribute("role", "alert");

  // Build headline
  let headline = "⚠ Malicious Website Detected";
  if (brandDetected) {
    headline = `⚠ Warning: This site is impersonating ${brandDetected}`;
  }

  // Build reasons HTML
  let reasonsHTML = "";
  const topReasons = (reasons || []).slice(0, 3);
  if (topReasons.length > 0) {
    reasonsHTML = `
      <ul style="
        margin: 6px 0 0 0;
        padding-left: 18px;
        font-size: 12px;
        opacity: 0.92;
        line-height: 1.6;
      ">
        ${topReasons.map(r => `<li>${escapeHtml(r)}</li>`).join("")}
      </ul>
    `;
  }

  banner.innerHTML = `
    <div style="display:flex;align-items:flex-start;gap:12px;width:100%;">
      <div style="font-size:24px;line-height:1;flex-shrink:0;margin-top:2px;">⚠</div>
      <div style="flex:1;">
        <div style="font-weight:700;font-size:14px;letter-spacing:.3px;">
          ${escapeHtml(headline)}
        </div>
        <div style="font-size:12px;opacity:.85;margin-top:2px;">
          Confidence: ${confidence.toFixed(1)}% — Proceed with extreme caution or close this tab.
        </div>
        ${reasonsHTML}
      </div>
      <div style="display:flex;flex-direction:column;gap:6px;flex-shrink:0;">
        <button id="${BANNER_ID}_close" style="
          background:rgba(255,255,255,0.2);
          border:1px solid rgba(255,255,255,0.4);
          color:#fff;
          border-radius:4px;
          padding:4px 10px;
          cursor:pointer;
          font-size:12px;
          font-weight:600;
        ">Dismiss</button>
        <button id="${BANNER_ID}_close_tab" style="
          background:#7f1d1d;
          border:1px solid rgba(255,255,255,0.3);
          color:#fff;
          border-radius:4px;
          padding:4px 10px;
          cursor:pointer;
          font-size:12px;
          font-weight:600;
        ">Close Tab</button>
      </div>
    </div>
    <div id="${BANNER_ID}_timer" style="
      position:absolute;
      bottom:0;left:0;
      height:3px;
      background:rgba(255,255,255,0.4);
      width:100%;
      transform-origin:left;
      transition:transform ${AUTO_DISMISS}ms linear;
    "></div>
  `;

  Object.assign(banner.style, {
    position  : "fixed",
    top       : "0",
    left      : "0",
    width     : "100%",
    zIndex    : "2147483647",
    background: "linear-gradient(135deg, #b91c1c 0%, #7f1d1d 100%)",
    color     : "#fff",
    padding   : "12px 16px 16px",
    display   : "flex",
    alignItems: "flex-start",
    gap       : "10px",
    fontFamily: "system-ui, -apple-system, sans-serif",
    boxShadow : "0 4px 20px rgba(0,0,0,.5)",
    boxSizing : "border-box",
    cursor    : "default",
  });

  document.documentElement.prepend(banner);

  // Timer bar animation
  setTimeout(() => {
    const timerBar = document.getElementById(`${BANNER_ID}_timer`);
    if (timerBar) {
      timerBar.style.transform = "scaleX(0)";
    }
  }, 100);

  // Auto dismiss
  const autoTimer = setTimeout(() => banner.remove(), AUTO_DISMISS);

  // Dismiss button
  document.getElementById(`${BANNER_ID}_close`)?.addEventListener("click", () => {
    clearTimeout(autoTimer);
    banner.remove();
  });

  // Close tab button
  document.getElementById(`${BANNER_ID}_close_tab`)?.addEventListener("click", () => {
    window.close();
    // Fallback if window.close() blocked
    setTimeout(() => {
      banner.innerHTML = `
        <div style="padding:8px;font-size:13px;">
          ⚠ Please close this tab manually — this site is dangerous.
        </div>`;
    }, 500);
  });
}

function removeBanner() {
  document.getElementById(BANNER_ID)?.remove();
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === "SHOW_BANNER") {
    createBanner(
      msg.confidence   ?? 0,
      msg.reasons      ?? [],
      msg.brandDetected ?? null,
    );
  }
  if (msg.type === "HIDE_BANNER") {
    removeBanner();
  }
});
