/**
 * history.js — Logic for history.html.
 * Extracted from an inline <script> tag, and inline onclick/oninput
 * attributes converted to addEventListener, because Manifest V3's default
 * CSP (script-src 'self') blocks both inline scripts and inline event
 * handler attributes on extension pages.
 */

"use strict";

const DEFAULT_API = "http://localhost:5000";
const POLL_INTERVAL_MS = 5000; // live refresh every 5s
let _allHistory    = [];
let _filter        = "all";
let _apiBase       = DEFAULT_API;
let _chart         = null;
let _pollHandle    = null;

function init() {
  try {
    chrome.storage.sync.get({ apiBase: DEFAULT_API }, function (s) {
      _apiBase = (s.apiBase || DEFAULT_API).replace(/\/$/, "");
      loadData();
      startPolling();
    });
  } catch (e) {
    _apiBase = DEFAULT_API;
    loadData();
    startPolling();
  }
}

function startPolling() {
  if (_pollHandle) clearInterval(_pollHandle);
  _pollHandle = setInterval(loadStatsAndChartOnly, POLL_INTERVAL_MS);
}

// Lightweight poll: refreshes stat cards + chart only, without re-rendering
// the (potentially large) table or disturbing the user's current filter/search.
async function loadStatsAndChartOnly() {
  try {
    const statsRes = await fetch(_apiBase + "/stats", {
      method : "GET",
      headers: { "Cache-Control": "no-cache" },
    });
    if (!statsRes.ok) return;
    const stats = await statsRes.json();

    document.getElementById("stat-total").textContent = stats.total     ?? 0;
    document.getElementById("stat-mal"  ).textContent = stats.malicious ?? 0;
    document.getElementById("stat-ben"  ).textContent = stats.benign    ?? 0;
    document.getElementById("stat-rate" ).textContent =
      (stats.mal_rate ?? 0).toFixed(1) + "%";

    updateChart(stats.malicious ?? 0, stats.benign ?? 0);
  } catch (_) {
    // Silent — a failed background poll shouldn't interrupt the page.
    // The next successful manual Refresh will surface any real connection error.
  }
}

function initChart(malicious, benign) {
  const ctx = document.getElementById("verdict-chart");
  if (!ctx || typeof Chart === "undefined") return;

  _chart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Malicious", "Benign"],
      datasets: [{
        data: [malicious, benign],
        backgroundColor: ["#ef4444", "#22c55e"],
        borderRadius: 6,
        maxBarThickness: 64,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 400 },
      plugins: { legend: { display: false } },
      scales: {
        x: {
          ticks: { color: "#e2e4ec", font: { size: 11 } },
          grid: { display: false },
        },
        y: {
          beginAtZero: true,
          ticks: { color: "#6b7280", precision: 0 },
          grid: { color: "#2a2d3a" },
        },
      },
    },
  });
}

function updateChart(malicious, benign) {
  if (!_chart) {
    initChart(malicious, benign);
    return;
  }
  _chart.data.datasets[0].data = [malicious, benign];
  _chart.update();
}

async function loadData() {
  document.getElementById("table-container").innerHTML =
    '<div class="loading"><div class="spinner"></div>Loading…</div>';

  try {
    const statsRes = await fetch(_apiBase + "/stats", {
      method : "GET",
      headers: { "Cache-Control": "no-cache" },
    });
    if (!statsRes.ok) throw new Error("HTTP " + statsRes.status);
    const stats = await statsRes.json();

    document.getElementById("stat-total").textContent = stats.total     ?? 0;
    document.getElementById("stat-mal"  ).textContent = stats.malicious ?? 0;
    document.getElementById("stat-ben"  ).textContent = stats.benign    ?? 0;
    document.getElementById("stat-rate" ).textContent =
      (stats.mal_rate ?? 0).toFixed(1) + "%";

    updateChart(stats.malicious ?? 0, stats.benign ?? 0);

    const histRes = await fetch(_apiBase + "/history?limit=1000", {
      method : "GET",
      headers: { "Cache-Control": "no-cache" },
    });
    if (!histRes.ok) throw new Error("History HTTP " + histRes.status);
    const hist  = await histRes.json();
    _allHistory = hist.history || [];
    renderTable(_allHistory);

  } catch (err) {
    document.getElementById("table-container").innerHTML =
      '<div class="error-box">' +
      '⚠ Cannot reach Flask API at <strong>' + _apiBase + '</strong><br>' +
      'Make sure <code>python app/app.py</code> is running on port 5000.<br><br>' +
      'Error: ' + err.message +
      '</div>';

    ["stat-total", "stat-mal", "stat-ben", "stat-rate"].forEach(function (id) {
      document.getElementById(id).textContent = "—";
    });
  }
}

function setFilter(filter, btn) {
  _filter = filter;
  document.querySelectorAll(".filter-btn").forEach(function (b) {
    b.classList.remove("active");
  });
  btn.classList.add("active");
  applySearch();
}

function applySearch() {
  const query = (document.getElementById("search-input").value || "").toLowerCase();
  let data    = _allHistory;
  if (_filter === "malicious") data = data.filter(function (h) { return h.prediction === "MALICIOUS"; });
  if (_filter === "benign")    data = data.filter(function (h) { return h.prediction === "BENIGN"; });
  if (query) data = data.filter(function (h) { return (h.url || "").toLowerCase().includes(query); });
  renderTable(data);
}

function renderTable(history) {
  const container = document.getElementById("table-container");

  if (!history || history.length === 0) {
    container.innerHTML =
      '<div class="empty-state">' +
      '<div class="big">🔍</div>' +
      '<div>No URLs found</div>' +
      '<div style="margin-top:6px;font-size:11px;">Browse some websites and results will appear here</div>' +
      '</div>';
    return;
  }

  let rows = "";
  history.forEach(function (h, i) {
    const isMal = h.prediction === "MALICIOUS";
    const vc    = isMal ? "mal" : "ben";
    const icon  = isMal ? "⚠" : "✓";
    const conf  = h.source === "whitelist"
      ? "—"
      : (h.confidence ?? 0).toFixed(1) + "%";
    const brand = h.brand
      ? '<div style="color:var(--amber);font-size:11px;margin-top:2px;">🎭 ' + escapeHtml(h.brand) + '</div>'
      : "";

    rows +=
      "<tr>" +
      '<td style="color:var(--muted);font-size:11px;">' + (i + 1) + "</td>" +
      '<td><span class="verdict-badge ' + vc + '">' + icon + " " + h.prediction + "</span></td>" +
      '<td class="url-cell"><div title="' + escapeHtml(h.url || "") + '">' + escapeHtml(h.url || "—") + "</div>" + brand + "</td>" +
      '<td class="conf-cell">' + conf + "</td>" +
      '<td><span class="src-badge">' + escapeHtml(h.source || "model") + "</span></td>" +
      '<td class="time-cell">' + escapeHtml(h.timestamp || "—") + "</td>" +
      "</tr>";
  });

  container.innerHTML =
    "<table>" +
    "<thead><tr>" +
    "<th>#</th><th>Verdict</th><th>URL</th><th>Confidence</th><th>Source</th><th>Time (UTC)</th>" +
    "</tr></thead>" +
    "<tbody>" + rows + "</tbody>" +
    "</table>";
}

async function clearHistory() {
  if (!confirm("Clear all URL history and statistics?\nThis cannot be undone.")) return;
  try {
    await fetch(_apiBase + "/stats/clear", { method: "POST" });
    _allHistory = [];
    ["stat-total", "stat-mal", "stat-ben"].forEach(function (id) {
      document.getElementById(id).textContent = "0";
    });
    document.getElementById("stat-rate").textContent = "0.0%";
    updateChart(0, 0);
    renderTable([]);
  } catch (_) {
    alert("Failed to clear — is Flask running?");
  }
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;")
    .replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

// ── Wire up controls (previously inline onclick/oninput — blocked by CSP) ──
document.getElementById("refresh-btn").addEventListener("click", loadData);
document.getElementById("clear-btn").addEventListener("click", clearHistory);
document.getElementById("filter-all").addEventListener("click", function () { setFilter("all", this); });
document.getElementById("filter-mal").addEventListener("click", function () { setFilter("malicious", this); });
document.getElementById("filter-ben").addEventListener("click", function () { setFilter("benign", this); });
document.getElementById("search-input").addEventListener("input", applySearch);

window.addEventListener("beforeunload", function () {
  if (_pollHandle) clearInterval(_pollHandle);
});

// Boot
init();