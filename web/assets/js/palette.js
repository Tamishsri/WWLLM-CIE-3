// Ctrl+K palette: polyglot search across the lecture, plus app commands.
// Results come from a provider so this module knows nothing about the session.

import { esc, icon } from "./util.js";

let provider = () => ({ groups: () => [], hints: [], note: "" });
let items = [];
let selected = 0;

const dialog = () => document.getElementById("palette");

export function setPaletteProvider(fn) {
  provider = fn;
}

export function openPalette(initial = "") {
  const dlg = dialog();
  if (dlg.open) return;
  const { placeholder } = provider();
  dlg.innerHTML = `
    <div class="pal-input">${icon("search")}
      <input id="palInput" type="search" autocomplete="off" spellcheck="false" aria-label="Search" placeholder="${esc(placeholder || "Search commands…")}">
      <kbd>Esc</kbd></div>
    <div class="pal-results" id="palResults" role="listbox"></div>
    <div class="pal-foot"><span><kbd>↑</kbd><kbd>↓</kbd> move</span><span><kbd>Enter</kbd> open</span>
      <span class="pal-note" id="palNote"></span></div>`;
  const input = dlg.querySelector("#palInput");
  input.value = initial;
  input.addEventListener("input", refresh);
  input.addEventListener("keydown", onKey);
  dlg.querySelector("#palResults").addEventListener("click", onClick);
  dlg.showModal();
  input.focus();
  refresh();
}

function refresh() {
  const dlg = dialog();
  const query = dlg.querySelector("#palInput").value.trim();
  const { groups, hints, note } = provider();
  const results = groups(query);
  const host = dlg.querySelector("#palResults");
  dlg.querySelector("#palNote").textContent = note || "";

  items = [];
  let html = "";
  if (!query && hints?.length) {
    html += `<div class="pal-hints"><span>Try</span>${hints
      .map((h) => `<button type="button" class="chip pal-hint" data-hint="${esc(h)}">${esc(h)}</button>`)
      .join("")}</div>`;
  }
  for (const group of results) {
    if (!group.items.length) continue;
    html += `<div class="pal-group">${esc(group.title)}</div>`;
    for (const item of group.items) {
      const i = items.push(item) - 1;
      html += `<button type="button" class="pal-item" role="option" data-i="${i}">
          <span class="pal-glyph">${item.time != null ? `<b>${esc(item.timeLabel)}</b>` : icon(item.icon || "arrow", "icon-sm")}</span>
          <span class="pal-text">${item.html}</span>
          ${item.badge ? `<span class="chip pal-badge">${esc(item.badge)}</span>` : ""}
        </button>`;
    }
  }
  if (!html) html = `<div class="pal-empty">Nothing matches “${esc(query)}”.</div>`;
  host.innerHTML = html;
  selected = 0;
  paint();
}

function paint() {
  const nodes = dialog().querySelectorAll(".pal-item");
  nodes.forEach((n, i) => {
    n.classList.toggle("sel", i === selected);
    n.setAttribute("aria-selected", i === selected);
  });
  nodes[selected]?.scrollIntoView({ block: "nearest" });
}

function activate(i) {
  const item = items[i];
  if (!item) return;
  dialog().close();
  item.run();
}

function onKey(e) {
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault();
    if (!items.length) return;
    const step = e.key === "ArrowDown" ? 1 : -1;
    selected = (selected + step + items.length) % items.length;
    paint();
  } else if (e.key === "Enter") {
    e.preventDefault();
    activate(selected);
  }
}

function onClick(e) {
  const hint = e.target.closest("[data-hint]");
  if (hint) {
    const input = dialog().querySelector("#palInput");
    input.value = hint.dataset.hint;
    input.focus();
    refresh();
    return;
  }
  const row = e.target.closest(".pal-item");
  if (row) activate(Number(row.dataset.i));
}
