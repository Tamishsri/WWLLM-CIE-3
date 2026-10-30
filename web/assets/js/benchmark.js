// Benchmark dashboard: headline metrics, WER charts, error composition and a
// sample table whose rows expand into a word-level alignment diff.

import { getJSON, postForm } from "./api.js";
import { debounce, downloadFile, esc, fixed, fmtNum, hideTip, icon, pct, plural, showTip, toast } from "./util.js";

let report = null;
let root = null;
let resizeObserver = null;
const view = { status: "all", lang: "all", q: "", sort: { key: "number", dir: 1 }, open: null, groupBy: "language" };

const STATUS_LABEL = { pass: "Pass", fail: "Fail", na: "No prediction" };
const STATUS_ICON = { pass: "check", fail: "x", na: "minus" };

export async function mountBenchmark(el) {
  root = el;
  if (!report) {
    el.innerHTML = skeletonHTML();
    try {
      report = await getJSON("/api/benchmark");
    } catch (err) {
      if (root === el) el.innerHTML = errorHTML(err);
      return;
    }
  }
  if (root === el) render();
}

export function unmountBenchmark() {
  resizeObserver?.disconnect();
  resizeObserver = null;
  root = null;
  hideTip();
}

// ===========================================================================
// Layout
// ===========================================================================
function render() {
  const s = report.summary;
  root.innerHTML = `
    <section class="bench">
      ${headHTML()}
      <div class="kpis rise" style="--i:1">${kpisHTML()}</div>
      <div class="chart-card card rise" style="--i:2" id="werCard">
        <div class="chart-head"><h3>Word error rate per lecture</h3>
          <p>${plural(s.evaluated, "lecture")} in dataset order, grouped by course module</p>
          <div class="legend" aria-hidden="true">
            <span><i style="background:var(--series-1)"></i>Pass · WER ≤ ${report.threshold.toFixed(2)}</span>
            <span><i style="background:var(--critical)"></i>Fail · WER > ${report.threshold.toFixed(2)}</span>
            <span><i class="line"></i>Threshold</span></div></div>
        <div class="chart" id="werChart" role="img" aria-label="Column chart of word error rate for each lecture"></div>
      </div>
      <div class="bench-grid rise" style="--i:3">
        <div class="chart-card card">
          <div class="chart-head"><h3>Average WER by ${view.groupBy}</h3>
            <div class="legend"><div class="seg" role="group">
              <button type="button" data-b="group" data-value="language" aria-pressed="${view.groupBy === "language"}">Language</button>
              <button type="button" data-b="group" data-value="domain" aria-pressed="${view.groupBy === "domain"}">Module</button></div></div></div>
          <div class="chart" id="groupChart" role="img" aria-label="Bar chart of average word error rate by ${view.groupBy}"></div>
        </div>
        <div class="chart-card card">${compositionHTML()}</div>
      </div>
      <div class="card table-card rise" style="--i:4" id="tableCard">
        ${tableToolsHTML()}
        <div class="table-scroll"><table class="data">
          <thead>${theadHTML()}</thead><tbody id="tbody">${tbodyHTML()}</tbody></table></div>
      </div>
    </section>`;

  root.onclick = onClick;
  root.oninput = onInput;
  root.onchange = onInput;
  const stack = root.querySelector(".stack");
  stack?.addEventListener("mousemove", (e) => {
    const part = e.target.closest("[data-tip]");
    if (part) showTip(part.dataset.tip, e.clientX, e.clientY);
  });
  stack?.addEventListener("mouseleave", hideTip);
  drawCharts();
  resizeObserver?.disconnect();
  resizeObserver = new ResizeObserver(debounce(drawCharts, 80));
  resizeObserver.observe(root.querySelector("#werCard"));
}

function headHTML() {
  const s = report.summary;
  const custom = report.source !== "bundled";
  return `
    <div class="bench-head rise" style="--i:0">
      <div>
        <p class="eyebrow"><span class="spark"></span>Evaluation · ${custom ? esc(report.source) : "bundled 40-lecture set"}</p>
        <h1>Benchmark</h1>
        <p>Predicted transcripts scored against references with word error rate — ${plural(s.samples, "lecture")} across
          ${plural(s.languages, "language")} and ${plural(s.domains, "module")}. Anything above WER ${report.threshold.toFixed(2)} is flagged as a failure.</p>
      </div>
      <div class="bench-actions">
        ${custom ? `<button class="btn btn-quiet" data-b="bundled">${icon("retry", "icon-sm")}Bundled set</button>` : ""}
        <button class="btn" data-b="upload">${icon("upload", "icon-sm")}Evaluate your own set</button>
        <button class="btn" data-b="csv" ${s.failures ? "" : "disabled"}>${icon("download", "icon-sm")}Failures CSV</button>
      </div>
    </div>
    ${
      report.simulated
        ? `<div class="callout is-warn rise" style="--i:1">${icon("alert")}<div class="callout-body">
            <strong>This bundled set is simulated — it is not a measurement of IndicConformer.</strong>
            <span>The predictions are hand-written in <code>generate_test_dataset.py</code> and the audio clips are synthetic tones.
            It demonstrates the evaluation pipeline. To measure the real model, run it on real recordings and use “Evaluate your own set”.</span>
          </div></div>`
        : ""
    }`;
}

function kpisHTML() {
  const s = report.summary;
  const m = report.metrics;
  const bert = !m.bertscore ? "BERTScore not installed" : m.bertscore_used ? `BERTScore F1 ${fixed(s.bert_f1, 3)}` : "BERTScore available";
  return `
    <div class="kpi card hero-kpi">
      <span class="kpi-label">${icon("chart", "icon-sm")}Average WER</span>
      <span class="kpi-value">${fixed(s.avg_wer, 3)}</span>
      <span class="kpi-sub">median ${fixed(s.median_wer, 3)} · best ${fixed(s.best_wer, 2)} · worst ${fixed(s.worst_wer, 2)}</span>
    </div>
    <div class="kpi card">
      <span class="kpi-label">Pass rate</span>
      <span class="kpi-value">${pct(s.pass_rate)}</span>
      <div class="meter" aria-hidden="true"><i style="--p:${s.pass_rate || 0}"></i></div>
      <span class="kpi-sub">${s.evaluated - s.failures} of ${s.evaluated} at or below ${report.threshold.toFixed(2)}</span>
    </div>
    <div class="kpi card">
      <span class="kpi-label">Failures</span>
      <span class="kpi-value">${fmtNum(s.failures)}</span>
      <span class="kpi-sub">logged for failure analysis</span>
    </div>
    <div class="kpi card">
      <span class="kpi-label">Word accuracy</span>
      <span class="kpi-value">${pct(s.word_accuracy)}</span>
      <span class="kpi-sub">${fmtNum(s.totals.hits)} of ${fmtNum(s.totals.ref_words)} reference words</span>
    </div>
    <div class="kpi card">
      <span class="kpi-label">Summary ROUGE-L</span>
      <span class="kpi-value">${fixed(s.rougeL, 3)}</span>
      <span class="kpi-sub">${m.rouge === "built-in" ? "built-in ROUGE" : "rouge-score"} · ${bert}</span>
    </div>`;
}

function compositionHTML() {
  const t = report.summary.totals;
  const errors = t.substitutions + t.deletions + t.insertions;
  const parts = [
    ["Substitutions", t.substitutions, "var(--series-1)", "a word recognised as a different word"],
    ["Deletions", t.deletions, "var(--series-2)", "reference words the ASR dropped"],
    ["Insertions", t.insertions, "var(--series-3)", "extra words the ASR added"],
  ];
  const stack = parts
    .map(([name, n, color, desc]) => {
      const share = errors ? n / errors : 0;
      if (!n) return "";
      return `<div style="flex:${n};background:${color}" data-tip="${esc(`<div class="tt-title">${name}</div>${fmtNum(n)} errors · ${pct(share)}<br>${desc}`)}">${share >= 0.12 ? pct(share, 0) : ""}</div>`;
    })
    .join("");
  return `
    <div class="chart-head"><h3>Where the errors come from</h3><p>${fmtNum(errors)} word errors across ${fmtNum(t.ref_words)} reference words</p></div>
    <div class="composition">
      <div class="stack" role="img" aria-label="Error composition">${stack}</div>
      <div class="comp-rows">${parts
        .map(
          ([name, n, color, desc]) => `
          <div class="comp-row"><i style="background:${color}"></i><span>${name} <span class="muted">— ${desc}</span></span>
            <b>${fmtNum(n)}</b><span class="muted">${pct(errors ? n / errors : 0)}</span></div>`
        )
        .join("")}</div>
    </div>`;
}

// ===========================================================================
// Charts (SVG)
// ===========================================================================
function drawCharts() {
  if (!root) return;
  drawWerChart();
  drawGroupChart();
}

function barPath(x, base, w, h, r) {
  if (h <= 0.5) return "";
  r = Math.min(r, w / 2, h);
  const top = base - h;
  return `M${x},${base}V${top + r}Q${x},${top} ${x + r},${top}H${x + w - r}Q${x + w},${top} ${x + w},${top + r}V${base}Z`;
}

function hBarPath(x, y, w, h, r) {
  if (w <= 0.5) return "";
  r = Math.min(r, h / 2, w);
  const right = x + w;
  return `M${x},${y}H${right - r}Q${right},${y} ${right},${y + r}V${y + h - r}Q${right},${y + h} ${right - r},${y + h}H${x}Z`;
}

function niceMax(v) {
  return Math.max(0.5, Math.ceil((v || 0) / 0.25) * 0.25);
}

function drawWerChart() {
  const el = root.querySelector("#werChart");
  const W = el?.clientWidth;
  if (!W) return;
  const samples = report.samples;
  const n = samples.length;
  const H = W < 600 ? 230 : 270;
  const m = { t: 18, r: 10, b: 54, l: 38 };
  const pw = W - m.l - m.r;
  const ph = H - m.t - m.b;
  const band = pw / n;
  const bw = Math.max(2, Math.min(18, band - 2));
  const ymax = niceMax(report.summary.worst_wer);
  const y = (v) => m.t + ph - (v / ymax) * ph;
  const base = m.t + ph;

  const ticks = [];
  for (let v = 0; v <= ymax + 1e-9; v += 0.25) ticks.push(v);
  const grid = ticks
    .map((v) => `<line class="${v === 0 ? "axis-line" : "grid-line"}" x1="${m.l}" x2="${W - m.r}" y1="${y(v)}" y2="${y(v)}"/>
      <text x="${m.l - 8}" y="${y(v) + 4}" text-anchor="end">${v.toFixed(2)}</text>`)
    .join("");

  const bars = samples
    .map((s, i) => {
      if (s.wer == null) return "";
      const x = m.l + i * band + (band - bw) / 2;
      const color = s.status === "fail" ? "var(--critical)" : "var(--series-1)";
      return `<path class="bar" data-i="${i}" d="${barPath(x, base, bw, Math.max(1.5, (s.wer / ymax) * ph), 4)}" fill="${color}"/>`;
    })
    .join("");

  const every = band >= 15 ? 1 : band >= 8 ? 5 : 10;
  const xLabels = samples
    .map((s, i) => {
      const num = s.number ?? i + 1;
      if (num !== 1 && num % every !== 0 && every !== 1) return "";
      return `<text x="${m.l + i * band + band / 2}" y="${base + 14}" text-anchor="middle">${num}</text>`;
    })
    .join("");

  // Course-module brackets under the axis
  const groups = [];
  samples.forEach((s, i) => {
    const last = groups[groups.length - 1];
    if (last && last.name === s.domain) last.end = i;
    else groups.push({ name: s.domain, start: i, end: i });
  });
  const brackets = groups
    .map((g) => {
      const x1 = m.l + g.start * band + 2;
      const x2 = m.l + (g.end + 1) * band - 2;
      const width = x2 - x1;
      const full = g.name;
      const short = full.split(/[ ,&]/)[0];
      const label = full.length * 6.2 < width ? full : short.length * 6.2 < width ? short : "";
      const by = base + 26;
      return `<path class="band-sep" d="M${x1},${by - 4}V${by}H${x2}V${by - 4}" fill="none"/>
        ${label ? `<text class="band-label" x="${(x1 + x2) / 2}" y="${by + 15}" text-anchor="middle">${esc(label)}</text>` : ""}`;
    })
    .join("");

  // Threshold line with a pill label so it stays legible over bars.
  const ty = y(report.threshold);
  const tLabel = `WER ${report.threshold.toFixed(2)} threshold`;
  const tw = tLabel.length * 6.1 + 14;
  const threshold = `<line class="threshold" x1="${m.l}" x2="${W - m.r}" y1="${ty}" y2="${ty}"/>
    <rect class="threshold-pill" x="${W - m.r - tw}" y="${ty - 10}" width="${tw}" height="20" rx="10"/>
    <text class="threshold-label" x="${W - m.r - tw / 2}" y="${ty + 4}" text-anchor="middle">${tLabel}</text>`;

  const hits = samples
    .map((s, i) => `<rect class="hit" data-i="${i}" x="${m.l + i * band}" y="${m.t}" width="${band}" height="${ph + 20}"/>`)
    .join("");

  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${grid}${bars}${threshold}${xLabels}${brackets}${hits}</svg>`;
  bindChartHover(el, (i) => sampleTip(samples[i]), (i) => openSample(samples[i].id));
}

function drawGroupChart() {
  const el = root.querySelector("#groupChart");
  const W = el?.clientWidth;
  if (!W) return;
  const rows = (view.groupBy === "language" ? report.by_language : report.by_domain).filter((r) => r.avg_wer != null);
  const rowH = 44;
  const labelW = Math.min(170, W * 0.38);
  const m = { t: 8, r: 54, b: 26, l: labelW };
  const H = m.t + rows.length * rowH + m.b;
  const pw = W - m.l - m.r;
  const xmax = niceMax(Math.max(...rows.map((r) => r.avg_wer), report.threshold));
  const x = (v) => m.l + (v / xmax) * pw;
  const barH = 16;

  const ticks = [];
  for (let v = 0; v <= xmax + 1e-9; v += 0.25) ticks.push(v);
  const grid = ticks
    .map((v) => `<line class="${v === 0 ? "axis-line" : "grid-line"}" x1="${x(v)}" x2="${x(v)}" y1="${m.t}" y2="${H - m.b}"/>
      <text x="${x(v)}" y="${H - m.b + 16}" text-anchor="middle">${v.toFixed(2)}</text>`)
    .join("");

  const bars = rows
    .map((r, i) => {
      const cy = m.t + i * rowH + rowH / 2;
      return `<text x="0" y="${cy - 3}" style="fill:var(--text-1);font-size:12.5px;font-weight:550">${esc(r.name)}</text>
        <text x="0" y="${cy + 13}">${plural(r.count, "lecture")}${r.failures ? ` · ${r.failures} failed` : ""}</text>
        <path class="bar" data-i="${i}" d="${hBarPath(m.l, cy - barH / 2, (r.avg_wer / xmax) * pw, barH, 4)}" fill="var(--series-1)"/>
        <text class="value-label" x="${x(r.avg_wer) + 6}" y="${cy + 4}">${r.avg_wer.toFixed(3)}</text>
        <rect class="hit" data-i="${i}" x="0" y="${cy - rowH / 2}" width="${W}" height="${rowH}"/>`;
    })
    .join("");

  const tx = x(report.threshold);
  const threshold = `<line class="threshold" x1="${tx}" x2="${tx}" y1="${m.t - 4}" y2="${H - m.b}"/>`;
  el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">${grid}${threshold}${bars}</svg>`;
  bindChartHover(el, (i) => {
    const r = rows[i];
    return `<div class="tt-title">${esc(r.name)}</div>
      <div class="tt-row"><span>Average WER</span><b>${r.avg_wer.toFixed(3)}</b></div>
      <div class="tt-row"><span>Lectures</span><b>${r.count}</b></div>
      <div class="tt-row"><span>Failures</span><b>${r.failures}</b></div>`;
  });
}

function bindChartHover(el, tip, onPick) {
  const svg = el.querySelector("svg");
  svg.addEventListener("mousemove", (e) => {
    const hit = e.target.closest(".hit");
    if (!hit) return;
    const i = hit.dataset.i;
    el.classList.add("hovering");
    svg.querySelectorAll(".bar").forEach((b) => b.classList.toggle("hot", b.dataset.i === i));
    showTip(tip(Number(i)), e.clientX, e.clientY);
  });
  svg.addEventListener("mouseleave", () => {
    el.classList.remove("hovering");
    hideTip();
  });
  if (onPick) {
    svg.addEventListener("click", (e) => {
      const hit = e.target.closest(".hit");
      if (hit) onPick(Number(hit.dataset.i));
    });
  }
}

function sampleTip(s) {
  const status = `<span class="status ${s.status}">${icon(STATUS_ICON[s.status])}${STATUS_LABEL[s.status]}</span>`;
  return `<div class="tt-title">#${s.number ?? ""} ${esc(s.title)}</div>
    <div class="tt-row"><span>WER</span><b>${fixed(s.wer, 3)}</b></div>
    <div class="tt-row"><span>Status</span>${status}</div>
    <div class="tt-row"><span>Language</span><b>${esc(s.language)}</b></div>
    <div class="tt-row"><span>Sub / Del / Ins</span><b>${s.substitutions} / ${s.deletions} / ${s.insertions}</b></div>
    ${s.condition ? `<div style="margin-top:6px">${esc(s.condition)}</div>` : ""}
    <div class="muted" style="margin-top:6px">Click to inspect the alignment</div>`;
}

// ===========================================================================
// Table
// ===========================================================================
function tableToolsHTML() {
  const counts = { all: report.samples.length, fail: 0, pass: 0 };
  report.samples.forEach((s) => (counts[s.status] = (counts[s.status] || 0) + 1));
  const langs = [...new Set(report.samples.map((s) => s.language))];
  return `<div class="table-tools">
      <h3>Lectures</h3>
      <div class="seg" role="group" aria-label="Filter by status">
        ${["all", "fail", "pass"]
          .map((k) => `<button type="button" data-b="status" data-value="${k}" aria-pressed="${view.status === k}">${k === "all" ? "All" : STATUS_LABEL[k] === "Fail" ? "Failures" : "Passing"} <span class="muted">${counts[k] || 0}</span></button>`)
          .join("")}
      </div>
      <select class="select" style="width:auto;height:34px;padding:0 10px" data-b="lang" aria-label="Filter by language">
        <option value="all">All languages</option>
        ${langs.map((l) => `<option value="${esc(l)}" ${view.lang === l ? "selected" : ""}>${esc(l)}</option>`).join("")}
      </select>
      <label class="search">${icon("search", "icon-sm")}<input type="search" data-b="q" placeholder="Search lectures, conditions, words…" value="${esc(view.q)}" aria-label="Search lectures"></label>
    </div>`;
}

function theadHTML() {
  const col = (key, label, cls = "") => {
    const sorted = view.sort.key === key;
    return `<th class="${cls}"><button type="button" data-b="sort" data-value="${key}" class="${sorted ? "sorted" : ""}">${label}${icon("sort", "icon-sm")}</button></th>`;
  };
  return `<tr>${col("number", "#")}${col("title", "Lecture")}${col("language", "Language")}<th>Recording condition</th>${col("wer", "WER", "num")}<th class="num">Sub · Del · Ins</th><th>Status</th></tr>`;
}

function visibleSamples() {
  const q = view.q.trim().toLowerCase();
  const rows = report.samples.filter(
    (s) =>
      (view.status === "all" || s.status === view.status) &&
      (view.lang === "all" || s.language === view.lang) &&
      (!q || [s.title, s.condition, s.domain, s.reference, s.prediction].some((v) => (v || "").toLowerCase().includes(q)))
  );
  const { key, dir } = view.sort;
  return rows.sort((a, b) => {
    const av = a[key] ?? -1;
    const bv = b[key] ?? -1;
    return (typeof av === "string" ? av.localeCompare(bv) : av - bv) * dir;
  });
}

function tbodyHTML() {
  const rows = visibleSamples();
  if (!rows.length) return `<tr><td colspan="7" class="muted" style="padding:28px;text-align:center">No lectures match these filters.</td></tr>`;
  const t = report.threshold;
  return rows
    .map((s) => {
      const open = view.open === s.id;
      const werCell =
        s.wer == null
          ? '<span class="muted">—</span>'
          : `<div class="wer-cell"><div class="wer-bar ${s.status === "fail" ? "fail" : ""}" style="--w:${s.wer};--t:${t}"><i></i></div><span>${s.wer.toFixed(3)}</span></div>`;
      const row = `<tr class="row ${open ? "open" : ""}" data-b="row" data-value="${esc(s.id)}" id="row-${esc(s.id)}" aria-expanded="${open}">
          <td><span class="lec-num">${s.number ?? ""}</span></td>
          <td><div class="lec-title">${esc(s.title)}</div><div class="cond">${esc(s.domain)}</div></td>
          <td>${esc(s.language)}</td>
          <td><div class="cond" title="${esc(s.condition)}">${esc(s.condition || "—")}</div></td>
          <td class="num">${werCell}</td>
          <td class="num"><span class="sdi"><b>${s.substitutions}</b> · <b>${s.deletions}</b> · <b>${s.insertions}</b></span></td>
          <td><span class="status ${s.status}">${icon(STATUS_ICON[s.status])}${STATUS_LABEL[s.status]}</span></td>
        </tr>`;
      return open ? row + detailHTML(s) : row;
    })
    .join("");
}

function diffHTML(alignment) {
  if (!alignment?.length) return '<span class="muted">No alignment — the sample has no prediction.</span>';
  return alignment
    .map((op) => {
      if (op.op === "equal") return esc(op.ref.join(" "));
      if (op.op === "delete") return op.ref.map((w) => `<span class="w del" title="Deletion — missing from the ASR output">${esc(w)}</span>`).join(" ");
      if (op.op === "insert") return op.hyp.map((w) => `<span class="w ins" title="Insertion — not in the reference">${esc(w)}</span>`).join(" ");
      return op.ref
        .map((w, i) => `<span class="w sub-ref" title="Reference word">${esc(w)}</span> <span class="w sub-hyp" title="Substitution — ASR heard this instead">${esc(op.hyp[i] ?? "")}</span>`)
        .join(" ");
    })
    .join(" ");
}

function detailHTML(s) {
  const summary =
    s.summary_ref && s.summary_pred
      ? `<div class="summary-pair"><h4>Summary · ROUGE-1 ${fixed(s.rouge1, 3)} · ROUGE-2 ${fixed(s.rouge2, 3)} · ROUGE-L ${fixed(s.rougeL, 3)}${
          s.bert_f1 != null ? ` · BERTScore ${fixed(s.bert_f1, 3)}` : ""
        }</h4>
          <p><b>Reference</b> ${esc(s.summary_ref)}</p><p><b>Generated</b> ${esc(s.summary_pred)}</p></div>`
      : "";
  return `<tr class="detail-row"><td colspan="7"><div class="detail">
      <div class="mini-metrics">
        <span class="chip">WER <b>&nbsp;${fixed(s.wer, 3)}</b></span>
        <span class="chip">${s.ref_words} reference words</span>
        <span class="chip">${s.hits} correct</span>
        ${s.reasons.map((r) => `<span class="chip">${icon("alert", "icon-sm")}${esc(r)}</span>`).join("")}
      </div>
      <div><h4>Alignment · reference vs ASR output</h4>
        <div class="diff lang-text">${diffHTML(s.alignment)}</div></div>
      <div class="diff-legend">
        <span><span class="sw w sub-ref">word</span>→<span class="sw w sub-hyp">heard</span>Substitution</span>
        <span><span class="sw w del">word</span>Deletion</span>
        <span><span class="sw w ins">word</span>Insertion</span>
      </div>
      <div class="detail-grid">
        <div><h4>Reference transcript</h4><div class="lang-text">${esc(s.reference)}</div></div>
        <div><h4>ASR prediction</h4><div class="lang-text">${esc(s.prediction) || '<span class="muted">—</span>'}</div></div>
      </div>
      ${summary}
    </div></td></tr>`;
}

function refreshTable() {
  const tbody = root?.querySelector("#tbody");
  if (tbody) tbody.innerHTML = tbodyHTML();
  const thead = root?.querySelector("thead");
  if (thead) thead.innerHTML = theadHTML();
}

function openSample(id) {
  view.open = id;
  view.status = "all";
  view.lang = "all";
  view.q = "";
  const tools = root.querySelector(".table-tools");
  if (tools) tools.outerHTML = tableToolsHTML();
  refreshTable();
  root.querySelector(`[id="row-${CSS.escape(id)}"]`)?.scrollIntoView({ behavior: "smooth", block: "center" });
}

// ===========================================================================
// Events
// ===========================================================================
function onClick(e) {
  const el = e.target.closest("[data-b]");
  if (!el) return;
  const { b, value } = el.dataset;
  if (b === "row") {
    view.open = view.open === value ? null : value;
    refreshTable();
  } else if (b === "status") {
    view.status = value;
    root.querySelectorAll('[data-b="status"]').forEach((x) => x.setAttribute("aria-pressed", x.dataset.value === value));
    refreshTable();
  } else if (b === "sort") {
    view.sort = { key: value, dir: view.sort.key === value ? -view.sort.dir : value === "wer" ? -1 : 1 };
    refreshTable();
  } else if (b === "group") {
    view.groupBy = value;
    render();
  } else if (b === "csv") {
    exportCSV();
  } else if (b === "upload") {
    openUploadDialog();
  } else if (b === "bundled") {
    resetToBundled();
  }
}

function onInput(e) {
  const b = e.target.dataset?.b;
  if (b === "q") {
    view.q = e.target.value;
    refreshTable();
  } else if (b === "lang" && e.type === "change") {
    view.lang = e.target.value;
    refreshTable();
  }
}

function csvCell(v) {
  const s = String(v ?? "");
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function exportCSV() {
  const header = ["audio_file", "language", "condition", "wer", "substitutions", "deletions", "insertions", "rouge1_f", "rouge2_f", "rougeL_f", "failure_reasons", "reference_transcript", "predicted_transcript"];
  const rows = report.samples
    .filter((s) => s.status === "fail")
    .map((s) => [s.id, s.language, s.condition, s.wer, s.substitutions, s.deletions, s.insertions, s.rouge1, s.rouge2, s.rougeL, s.reasons.join("; "), s.reference, s.prediction]);
  // BOM so spreadsheet apps read Indic scripts as UTF-8
  const csv = "﻿" + [header, ...rows].map((r) => r.map(csvCell).join(",")).join("\r\n");
  downloadFile("failure_analysis.csv", csv, "text/csv");
}

async function resetToBundled() {
  try {
    report = await getJSON("/api/benchmark");
    view.open = null;
    render();
  } catch (err) {
    toast(err.message, { kind: "error" });
  }
}

function openUploadDialog() {
  let dlg = document.getElementById("benchDialog");
  if (!dlg) {
    dlg = Object.assign(document.createElement("dialog"), { id: "benchDialog", className: "sheet" });
    dlg.setAttribute("aria-labelledby", "benchTitle");
    document.body.append(dlg);
    dlg.addEventListener("click", (e) => e.target === dlg && dlg.close());
  }
  const bert = report.metrics.bertscore;
  dlg.innerHTML = `
    <form id="benchForm">
      <div class="sheet-head"><div><h2 id="benchTitle">Evaluate your own set</h2>
        <p>Score any ASR output against your references. Files stay on this machine.</p></div>
        <button class="icon-btn" type="button" data-close aria-label="Close">${icon("x", "icon-sm")}</button></div>
      <div class="sheet-body">
        <div class="field"><label for="refsFile">Ground truth JSON <span class="muted">(required)</span></label>
          <input class="input" type="file" id="refsFile" name="references" accept=".json,application/json" required>
          <span class="kpi-sub">Array of <code>{"audio_file", "transcript", "summary"?, "condition"?}</code></span></div>
        <div class="field"><label for="predsFile">Predictions JSON</label>
          <input class="input" type="file" id="predsFile" name="predictions" accept=".json,application/json">
          <span class="kpi-sub"><code>{"audio_file": "transcript"}</code>, or <code>{"predictions", "summaries_pred", "summaries_ref"}</code></span></div>
        <label class="kpi-sub" style="display:flex;gap:8px;align-items:center">
          <input type="checkbox" id="bertBox" ${bert ? "" : "disabled"}> Compute BERTScore ${bert ? "(slow — downloads a BERT model on first use)" : "(bert-score isn't installed)"}</label>
      </div>
      <div class="sheet-foot"><button class="btn btn-quiet" type="button" data-close>Cancel</button>
        <button class="btn btn-primary" type="submit" id="benchGo">${icon("chart", "icon-sm")}Evaluate</button></div>
    </form>`;
  dlg.querySelectorAll("[data-close]").forEach((b) => (b.onclick = () => dlg.close()));
  dlg.querySelector("#benchForm").onsubmit = async (e) => {
    e.preventDefault();
    const go = dlg.querySelector("#benchGo");
    go.disabled = true;
    go.innerHTML = '<span class="spinner"></span>Evaluating…';
    const form = new FormData(e.target);
    if (!dlg.querySelector("#predsFile").files.length) form.delete("predictions");
    try {
      const query = dlg.querySelector("#bertBox").checked ? "?bertscore=1" : "";
      report = await postForm(`/api/benchmark${query}`, form);
      view.open = null;
      view.status = "all";
      view.lang = "all";
      view.q = "";
      dlg.close();
      if (root) render();
      toast(`Evaluated ${plural(report.summary.samples, "sample")} from ${report.source}`, { kind: "success" });
    } catch (err) {
      toast(err.message, { kind: "error", ms: 5000 });
      go.disabled = false;
      go.innerHTML = `${icon("chart", "icon-sm")}Evaluate`;
    }
  };
  dlg.showModal();
}

// ===========================================================================
// States
// ===========================================================================
function skeletonHTML() {
  const block = (h) => `<div class="card" style="height:${h}px;padding:20px"><div class="skeleton"><i style="width:40%"></i><i style="width:70%"></i><i style="width:55%"></i></div></div>`;
  return `<section class="bench">
      <div class="bench-head"><div><p class="eyebrow"><span class="spark"></span>Evaluation</p><h1>Benchmark</h1><p>Scoring 40 lectures…</p></div></div>
      <div class="kpis">${block(120)}${block(120)}${block(120)}${block(120)}${block(120)}</div>${block(300)}</section>`;
}

function errorHTML(err) {
  return `<div class="callout is-error" role="alert">${icon("alert")}
    <div class="callout-body"><strong>Couldn't load the benchmark.</strong><span>${esc(err.message)}</span></div></div>`;
}
