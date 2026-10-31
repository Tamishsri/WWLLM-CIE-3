// App shell: routing, theme, engine status, dialogs and global shortcuts.

import { getJSON } from "./api.js";
import { mountBenchmark, unmountBenchmark } from "./benchmark.js";
import { openPalette, setPaletteProvider } from "./palette.js";
import { initStudio, mountStudio, paletteProvider, setStatus, studioKeydown, unmountStudio } from "./studio.js";
import { isTheatreOpen } from "./theatre.js";
import { esc, icon, isTyping, toast } from "./util.js";

const viewEl = document.getElementById("view");
const html = document.documentElement;
let status = null;
let current = null;

// ── Theme ─────────────────────────────────────────────────────────────────
const themeBtn = document.getElementById("themeToggle");

function paintThemeButton() {
  const dark = html.dataset.theme !== "light";
  themeBtn.innerHTML = icon(dark ? "sun" : "moon");
  themeBtn.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
}

function toggleTheme() {
  const apply = () => {
    html.dataset.theme = html.dataset.theme === "light" ? "dark" : "light";
    try {
      localStorage.setItem("wwllm.theme", html.dataset.theme);
    } catch {
      /* private mode */
    }
    paintThemeButton();
  };
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (!document.startViewTransition || reduced) return apply();

  // Circular reveal from the toggle button.
  const r = themeBtn.getBoundingClientRect();
  const x = r.left + r.width / 2;
  const y = r.top + r.height / 2;
  const radius = Math.hypot(Math.max(x, innerWidth - x), Math.max(y, innerHeight - y));
  document.startViewTransition(apply).ready.then(() => {
    html.animate(
      { clipPath: [`circle(0px at ${x}px ${y}px)`, `circle(${radius}px at ${x}px ${y}px)`] },
      { duration: 560, easing: "cubic-bezier(0.2, 0.8, 0.2, 1)", pseudoElement: "::view-transition-new(root)" }
    );
  });
}

themeBtn.addEventListener("click", toggleTheme);
matchMedia("(prefers-color-scheme: light)").addEventListener("change", (e) => {
  let saved = null;
  try {
    saved = localStorage.getItem("wwllm.theme");
  } catch {
    /* ignore */
  }
  if (!saved) {
    html.dataset.theme = e.matches ? "light" : "dark";
    paintThemeButton();
  }
});
paintThemeButton();

// ── Top bar ───────────────────────────────────────────────────────────────
const topbar = document.getElementById("topbar");
addEventListener("scroll", () => topbar.classList.toggle("scrolled", scrollY > 4), { passive: true });

// ── Engine status ─────────────────────────────────────────────────────────
const pill = document.getElementById("enginePill");
const STAGE_META = {
  vad: { label: "Voice activity detection", icon: "wave" },
  asr: { label: "Speech recognition", icon: "mic" },
  llm: { label: "Notes & quiz", icon: "sparkles" },
  mt: { label: "Translation", icon: "translate" },
};

function paintPill() {
  const footer = document.getElementById("footerEngines");
  if (footer && status) {
    const s = status.stages;
    footer.textContent = `WWLLM v2 · ${s.vad.name} · ${s.asr.name} · ${s.llm.name} · ${s.mt.name}`;
  }
  const label = pill.querySelector(".pill-label");
  if (!status) {
    pill.dataset.mode = "offline";
    label.textContent = "Server offline";
    return;
  }
  const stages = Object.values(status.stages);
  const live = stages.filter((s) => s.live).length;
  pill.dataset.mode = status.mode;
  label.textContent =
    status.mode === "live" ? "All models live" : status.mode === "hybrid" ? `Hybrid · ${live}/${stages.length} live` : "Demo mode · offline engines";
  pill.title = "Engine status";
}

function openEngines() {
  const dlg = document.getElementById("enginesDialog");
  if (!status) {
    toast("The studio server isn't reachable — start it with: python server.py", { kind: "error", ms: 5000 });
    return;
  }
  const d = status.device || {};
  const device = !d.probed
    ? `<span class="chip"><span class="spinner"></span>Probing hardware…</span>`
    : [
        `<span class="chip">${icon("cpu", "icon-sm")}${d.cuda ? esc(d.gpu) : "CPU only"}</span>`,
        d.vram_gb ? `<span class="chip">${d.vram_gb} GB VRAM</span>` : "",
        d.torch ? `<span class="chip">PyTorch ${esc(d.torch)}</span>` : `<span class="chip">PyTorch not installed</span>`,
        d.cuda_version ? `<span class="chip">CUDA ${esc(d.cuda_version)}</span>` : "",
      ].join("");

  const rows = Object.entries(status.stages)
    .map(([key, s]) => {
      const meta = STAGE_META[key];
      const detail = s.live
        ? `Loads on first use · <code>${esc(s.model)}</code>${s.note ? `<br>${esc(s.note)}` : ""}`
        : `Missing <code>${esc(s.missing.join(", ") || "—")}</code> → ${esc(s.fallback)}`;
      return `<div class="engine-row"><span class="step-badge">${icon(meta.icon, "icon-sm")}</span>
          <div><h4>${esc(s.name)} <span class="muted" style="font-weight:400">· ${meta.label}</span></h4><p>${detail}</p></div>
          <span class="chip" data-mode="${s.live ? "live" : "fallback"}"><span class="dot"></span>${s.live ? "Live" : "Fallback"}</span></div>`;
    })
    .join("");

  const sentence = {
    live: "Every stage runs a real model on this machine.",
    hybrid: "Some stages run real models; the rest use offline fallbacks.",
    demo: status.forced_demo
      ? "Demo mode was forced with --demo: stored results and offline engines only."
      : "The ML stack isn't installed here, so stages use stored results and offline engines. The sample lecture still shows real model output.",
  }[status.mode];

  const install =
    status.mode === "live"
      ? ""
      : `<div class="callout is-info">${icon("info")}<div class="callout-body"><strong>Run the real models</strong>
          <span>Install <code>requirements.txt</code> (see README), then restart <code>python server.py</code>. Each stage switches to its real model automatically.</span></div></div>`;

  dlg.innerHTML = `
    <div class="sheet-head"><div><h2 id="enginesTitle">Engines</h2><p>${sentence}</p></div>
      <button class="icon-btn" data-close aria-label="Close">${icon("x", "icon-sm")}</button></div>
    <div class="sheet-body">${rows}<div class="device-line">${device}</div>${install}</div>`;
  dlg.querySelector("[data-close]").onclick = () => dlg.close();
  if (!dlg.open) dlg.showModal();
}

pill.addEventListener("click", openEngines);

async function loadStatus() {
  try {
    status = await getJSON("/api/status");
  } catch {
    status = null;
    toast("Can't reach the studio server — start it with: python server.py", { kind: "error", ms: 6000 });
  }
  paintPill();
  return status;
}

async function pollDevice(tries = 12) {
  if (!status || status.device?.probed || tries <= 0) return;
  await new Promise((r) => setTimeout(r, 1500));
  try {
    status = await getJSON("/api/status");
    setStatus(status);
    paintPill();
    if (document.getElementById("enginesDialog").open) openEngines();
  } catch {
    /* keep the last known status */
  }
  pollDevice(tries - 1);
}

// ── Shortcuts dialog ──────────────────────────────────────────────────────
function openKeys() {
  const dlg = document.getElementById("keysDialog");
  const row = (keys, label) => `<div><dt>${label}</dt><dd>${keys.map((k) => `<kbd>${k}</kbd>`).join("")}</dd></div>`;
  dlg.innerHTML = `
    <div class="sheet-head"><div><h2 id="keysTitle">Shortcuts</h2></div>
      <button class="icon-btn" data-close aria-label="Close">${icon("x", "icon-sm")}</button></div>
    <div class="sheet-body"><dl class="keys-list">
      ${row(["Space"], "Play / pause the lecture")}
      ${row(["Ctrl", "K"], "Search the lecture in any script")}
      ${row(["1", "–", "5"], "Jump to a stage")}
      ${row(["F"], "Enter Theatre")}
      ${row(["Ctrl", "Enter"], "Run the current stage")}
      ${row(["T"], "Toggle light / dark theme")}
      ${row(["?"], "Show this list")}
      ${row(["Esc"], "Close dialogs")}
    </dl></div>`;
  dlg.querySelector("[data-close]").onclick = () => dlg.close();
  dlg.showModal();
}

// Close any sheet by clicking its backdrop.
document.querySelectorAll("dialog.sheet").forEach((dlg) =>
  dlg.addEventListener("click", (e) => {
    if (e.target === dlg) dlg.close();
  })
);

setPaletteProvider(() => paletteProvider({ toggleTheme, openEngines, openKeys }));
document.getElementById("searchBtn").addEventListener("click", () => openPalette());

addEventListener("keydown", (e) => {
  if (document.querySelector("dialog[open]") || isTheatreOpen()) return;
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
    e.preventDefault();
    openPalette();
    return;
  }
  if (e.key === "/" && !isTyping(e.target)) {
    e.preventDefault();
    openPalette();
    return;
  }
  if (current === "studio" && studioKeydown(e)) {
    e.preventDefault();
    return;
  }
  if (isTyping(e.target) || e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.key === "t" || e.key === "T") toggleTheme();
  else if (e.key === "?") openKeys();
});

// ── Routing ───────────────────────────────────────────────────────────────
function route() {
  const target = location.hash.replace(/^#\/?/, "") === "benchmark" ? "benchmark" : "studio";
  if (target === current) return;
  if (current === "studio") unmountStudio();
  if (current === "benchmark") unmountBenchmark();
  current = target;
  document.querySelectorAll(".nav a").forEach((a) => a.classList.toggle("active", a.dataset.route === target));
  document.title = target === "benchmark" ? "Benchmark · WWLLM Lecture Studio" : "WWLLM Lecture Studio";
  viewEl.innerHTML = "";
  scrollTo({ top: 0 });
  if (target === "studio") mountStudio(viewEl);
  else mountBenchmark(viewEl);
}

// ── Boot ──────────────────────────────────────────────────────────────────
(async function boot() {
  await loadStatus();
  initStudio(status);
  route();
  addEventListener("hashchange", route);
  pollDevice();
})();
