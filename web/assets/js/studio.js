// Studio view: source → transcript → notes → quiz → translate.
// State lives in `S` (persisted to localStorage); the DOM is re-rendered in
// regions (deck, stepper, panel, log) as pipeline events stream in.

import { anchorNotes, anchorQuiz, sectionIndex } from "./anchors.js";
import { postJSON, stream, uploadFile } from "./api.js";
import { extractConcepts, layoutConcepts } from "./concepts.js";
import { LectureIndex, retimeSegments } from "./lecture.js";
import { inline, lineKind, render as renderMarkdown } from "./markdown.js";
import { detectLanguage, tokenize } from "./phonetic.js";
import { mountQuiz } from "./quiz.js";
import { buildRangoli, rangoliSVG } from "./rangoli.js";
import { GlyphRiver } from "./river.js";
import { isTheatreOpen, openTheatre } from "./theatre.js";
import {
  copyText,
  debounce,
  downloadFile,
  esc,
  fmtBytes,
  fmtTime,
  icon,
  isTyping,
  plural,
  rafThrottle,
  readMinutes,
  safeName,
  toast,
  wordCount,
} from "./util.js";
import { Waveform } from "./waveform.js";

const STAGES = [
  { key: "transcript", label: "Transcript", icon: "mic" },
  { key: "notes", label: "Notes", icon: "notes" },
  { key: "quiz", label: "Quiz", icon: "quiz" },
  { key: "translate", label: "Translate", icon: "translate" },
  { key: "concepts", label: "Concepts", icon: "layers" },
];
const SESSION_KEY = "wwllm.session.v2";
const PERSISTED = [
  "source", "sourceLang", "chunks", "segments", "transcript", "transcriptRev", "transcriptMeta",
  "notes", "notesMeta", "notesRev", "quiz", "quizMeta", "quizRev", "numQuestions", "quizMode",
  "quizAnswers", "target", "translations", "translateView", "translateTab", "active", "notesView",
];
const PREFS = ["sourceLang", "target", "numQuestions", "notesView", "translateView"];
const MEDIA_RE = /\.(wav|mp3|flac|ogg|m4a|mp4|mkv|webm)$/i;
const MODE_LABEL = {
  live: "Live model",
  fallback: "Offline fallback",
  replay: "Stored prediction",
  manual: "Your text",
};
const RATES = [1, 1.25, 1.5, 2, 0.75];
const ROTATOR = [
  { t: "notes" },
  { t: "नोट्स", indic: true },
  { t: "quizzes" },
  { t: "குறிப்புகள்", indic: true },
  { t: "translations" },
  { t: "ಟಿಪ್ಪಣಿಗಳು", indic: true },
];
const T0 = performance.now();

const freshStages = () =>
  Object.fromEntries(STAGES.map((s) => [s.key, { status: "idle", progress: 0, detail: "", error: null, phase: 0 }]));

function freshState(prefs = {}) {
  return {
    source: null,
    sourceLang: "Hindi",
    chunks: [],
    segments: [],
    transcript: "",
    transcriptRev: 0,
    transcriptMeta: null,
    transcriptView: "segments",
    notes: "",
    notesMeta: null,
    notesRev: 0,
    notesView: "rendered",
    quiz: [],
    quizMeta: null,
    quizRev: 0,
    numQuestions: 5,
    quizMode: "practice",
    quizAnswers: {},
    target: "Hindi",
    translations: {},
    translateView: "pairs",
    translateTab: "notes",
    active: "transcript",
    stages: freshStages(),
    running: null,
    log: [],
    ...prefs,
  };
}

let S = freshState();
const ctx = {
  status: null,
  root: null,
  wave: null,
  audio: null,
  audioFor: null,
  abort: null,
  uploadAbort: null,
  localFile: null,
  notesStream: "",
  rotatorTimer: null,
  rate: 1,
  river: null,
  index: null,
  indexKey: "",
  memo: new Map(),
  concept: null,
  sentence: -2,
  words: [],
  deckObserver: null,
  deckVisible: true,
  pendingJump: null,
};

// ===========================================================================
// Persistence
// ===========================================================================
const persist = debounce(() => {
  try {
    const data = Object.fromEntries(PERSISTED.map((k) => [k, S[k]]));
    localStorage.setItem(SESSION_KEY, JSON.stringify(data));
  } catch {
    /* storage full or blocked — the session just won't survive a reload */
  }
}, 400);

function restore() {
  try {
    const data = JSON.parse(localStorage.getItem(SESSION_KEY) || "null");
    if (data && typeof data === "object") {
      for (const k of PERSISTED) if (k in data) S[k] = data[k];
    }
  } catch {
    /* ignore corrupt sessions */
  }
}

// ===========================================================================
// Public API (used by main.js)
// ===========================================================================
export function initStudio(status) {
  ctx.status = status;
  restore();
}

export function setStatus(status) {
  ctx.status = status;
  if (!ctx.root) return;
  if (S.source) return updateRegions();
  const signal = ctx.root.querySelector(".signal");
  if (signal) signal.innerHTML = signalHTML();
}

export function mountStudio(root) {
  ctx.root = root;
  render();
  if (ctx.pendingJump) {
    const jump = ctx.pendingJump;
    ctx.pendingJump = null;
    jump();
  }
}

export function unmountStudio() {
  clearInterval(ctx.rotatorTimer);
  ctx.river?.destroy();
  ctx.river = null;
  ctx.audio?.pause();
  ctx.wave?.destroy();
  ctx.wave = null;
  ctx.deckObserver?.disconnect();
  ctx.root = null;
}

export function studioKeydown(e) {
  if (!ctx.root || isTyping(e.target) || e.metaKey || e.altKey) return false;
  if (!S.source) return false;
  if (e.key === " " && !e.ctrlKey && e.target.tagName !== "BUTTON") {
    togglePlay();
    return true;
  }
  if (e.ctrlKey && e.key === "Enter") {
    runStage(S.active);
    return true;
  }
  const n = Number(e.key);
  if (!e.ctrlKey && n >= 1 && n <= STAGES.length) {
    setActive(STAGES[n - 1].key);
    return true;
  }
  if (!e.ctrlKey && (e.key === "f" || e.key === "F") && S.source.kind !== "text") {
    enterTheatre();
    return true;
  }
  return false;
}

// ===========================================================================
// Derived state
// ===========================================================================
const srcKey = () => `${S.notesRev}:${S.quizRev}`;
const notesStale = () => Boolean(S.notes) && S.notesMeta?.rev !== S.transcriptRev;
const quizStale = () => S.quiz.length > 0 && S.quizMeta?.rev !== S.transcriptRev;
const trStale = (lang) =>
  Boolean(S.translations[lang]) && (S.translations[lang].srcKey !== srcKey() || notesStale() || quizStale());
const stageLive = (key) => Boolean(ctx.status?.stages?.[key]?.live);
const targets = () => ctx.status?.languages?.translate || [];
const asrLangs = () => ctx.status?.languages?.asr || [];
const nativeOf = (name) => targets().find((t) => t.name === name)?.native || asrLangs().find((t) => t.name === name)?.native || name;
const sampleContent = () => S.source?.kind === "sample";
const stageName = (key, fallback) => ctx.status?.stages?.[key]?.name || fallback;

/** Search index of the current transcript (rebuilt when the transcript changes). */
function lecture() {
  const key = `${S.transcriptRev}:${S.segments.length}:${S.transcript.length}`;
  if (ctx.indexKey !== key) {
    ctx.index = new LectureIndex({ segments: S.segments, transcript: S.transcript });
    ctx.indexKey = key;
    ctx.memo.clear();
  }
  return ctx.index;
}

function memo(key, build) {
  lecture();
  if (!ctx.memo.has(key)) ctx.memo.set(key, build());
  return ctx.memo.get(key);
}

const noteSections = () => memo(`notes:${S.notesRev}`, () => (S.notes ? anchorNotes(lecture(), S.notes) : []));
const quizMoments = () => memo(`quiz:${S.notesRev}:${S.quizRev}`, () => anchorQuiz(lecture(), noteSections(), S.quiz));
const lectureDuration = () => S.source?.duration || (Number.isFinite(ctx.audio?.duration) ? ctx.audio.duration : 0);
const conceptGraph = () =>
  memo(`concepts:${S.notesRev}`, () => extractConcepts(lecture(), { notes: S.notes, duration: lectureDuration() }));

function rangoliModel() {
  return buildRangoli({
    peaks: ctx.wave?.peaks,
    chunks: S.chunks,
    duration: lectureDuration(),
    seed: `${S.source?.title || ""}${S.transcript.slice(0, 240)}`,
  });
}

function availability(lang) {
  if (stageLive("mt")) return "live";
  return lang.sample && sampleContent() ? "sample" : "locked";
}

function stageInfo(key) {
  const st = S.stages[key];
  if (st.status === "running") return { status: "running", text: st.detail || "Working…" };
  const err = st.status === "error" && (key !== "translate" || st.error?.target === S.target);
  if (err) return { status: "error", text: "Needs attention" };

  if (key === "transcript") {
    return S.transcript
      ? { status: "done", text: plural(wordCount(S.transcript), "word") }
      : { status: "ready", text: "Ready to transcribe" };
  }
  if (key === "notes") {
    if (S.notes) return { status: "done", stale: notesStale(), text: notesStale() ? "Out of date" : plural(wordCount(S.notes), "word") };
    return S.transcript ? { status: "ready", text: "Ready to write" } : { status: "locked", text: "Needs a transcript" };
  }
  if (key === "quiz") {
    if (S.quiz.length) return { status: "done", stale: quizStale(), text: quizStale() ? "Out of date" : plural(S.quiz.length, "question") };
    return S.transcript ? { status: "ready", text: "Ready to generate" } : { status: "locked", text: "Needs a transcript" };
  }
  if (key === "concepts") {
    if (!S.transcript) return { status: "locked", text: "Needs a transcript" };
    const n = conceptGraph().concepts.length;
    return n ? { status: "done", text: plural(n, "concept") } : { status: "ready", text: "Too little text yet" };
  }
  const tr = S.translations[S.target];
  if (tr) return { status: "done", stale: trStale(S.target), text: trStale(S.target) ? "Out of date" : `${tr.native} ready` };
  if (!S.notes && !S.quiz.length) return { status: "locked", text: "Needs notes or a quiz" };
  const count = Object.keys(S.translations).length;
  return { status: "ready", text: count ? `${count} done · ${S.target} next` : `Ready · ${S.target}` };
}

function needsRun(key) {
  if (key === "concepts") return false; // derived from the transcript and notes
  if (key === "transcript") return S.source?.kind !== "text" && !S.transcript;
  if (key === "notes") return !S.notes || notesStale();
  if (key === "quiz") return !S.quiz.length || quizStale();
  return !S.translations[S.target] || trStale(S.target);
}

// ===========================================================================
// Logging
// ===========================================================================
function log(tag, message, level = "info") {
  const entry = { t: (performance.now() - T0) / 1000, tag, message, level };
  S.log.push(entry);
  if (S.log.length > 250) S.log.shift();
  const list = ctx.root?.querySelector("#logList");
  if (list) {
    list.querySelector(".log-empty")?.remove();
    list.insertAdjacentHTML("beforeend", logItem(entry));
    list.scrollTop = list.scrollHeight;
  }
}

const LOG_TAGS = { transcript: "asr", notes: "notes", quiz: "quiz", translate: "mt" };

const logItem = (e) =>
  `<li class="${e.level}"><span class="log-t">${fmtTime(e.t)}</span><span class="log-tag">${esc(LOG_TAGS[e.tag] || e.tag)}</span><span class="log-msg">${esc(e.message)}</span></li>`;

// ===========================================================================
// Rendering — hero
// ===========================================================================
function render() {
  const root = ctx.root;
  if (!root) return;
  clearInterval(ctx.rotatorTimer);
  ctx.river?.destroy();
  ctx.river = null;
  ctx.wave?.destroy();
  ctx.wave = null;
  ctx.deckObserver?.disconnect();

  if (!S.source) {
    root.innerHTML = heroHTML();
    bindHero();
    startRotator();
    ctx.river = new GlyphRiver(root.querySelector("#river"));
    return;
  }
  root.innerHTML = `
    <section class="workspace">
      <div class="deck card rise" style="--i:0" id="deck">${deckHTML()}</div>
      <nav class="stepper rise" style="--i:1" id="stepper" aria-label="Pipeline stages">${stepperHTML()}</nav>
      <div class="stage-grid rise" style="--i:2">
        <section class="panel card" id="panel"></section>
        <aside class="log card" aria-label="Run log">${logHTML()}</aside>
      </div>
      <div class="mini" id="mini" hidden>
        <button class="play-btn" data-action="play" aria-label="Play or pause">${icon("pause")}</button>
        <span class="mini-text lang-text" id="miniText"></span>
        <span class="transport-time" id="miniTime">0:00</span>
      </div>
    </section>`;
  mountWave();
  paintEmblem();
  renderPanel(true);
  root.onclick = onWorkspaceClick;

  ctx.deckVisible = true;
  ctx.deckObserver = new IntersectionObserver(([entry]) => {
    ctx.deckVisible = entry.isIntersecting;
    paintMini();
  });
  ctx.deckObserver.observe(root.querySelector("#deck"));
}

function heroHTML() {
  const st = ctx.status;
  const sampleDur = st?.sample?.duration ? ` · ${fmtTime(st.sample.duration)}` : "";
  return `
  <section class="hero">
    <canvas class="river-canvas" id="river" aria-hidden="true"></canvas>
    <div class="hero-copy">
      <p class="eyebrow rise" style="--i:0"><span class="spark"></span>AI4Bharat lecture studio</p>
      <h1 class="display rise" style="--i:1">Turn every lecture into <span class="rotator" id="rotator" aria-hidden="true">${ROTATOR.map(
        (w) => `<span class="${w.indic ? "indic" : ""}">${w.t}</span>`
      ).join("")}</span><span class="sr-only">notes, quizzes and translations</span></h1>
      <p class="lede rise" style="--i:2">Drop a classroom recording in Hindi, Tamil, Kannada, English and more. Get a synced transcript, structured notes, a practice quiz, and translations into ten Indian languages.</p>
      <div class="hero-actions rise" style="--i:3">
        <button class="btn btn-primary btn-lg" data-action="sample">${icon("play", "icon-sm")}Try the sample lecture<span class="btn-sub">Hindi${sampleDur}</span></button>
        <button class="btn btn-lg" data-action="paste">${icon("clipboard", "icon-sm")}Paste a transcript</button>
      </div>
    </div>
    <div class="dropzone rise" style="--i:2" id="dropzone" role="button" tabindex="0" aria-label="Upload a lecture recording">
      <input type="file" id="fileInput" accept="audio/*,video/*,.wav,.mp3,.mp4,.flac,.ogg,.m4a,.mkv,.webm" hidden>
      <div class="dz-inner" id="dzInner">${dzIdleHTML()}</div>
    </div>
    <div class="signal rise" style="--i:4">${signalHTML()}</div>
  </section>`;
}

function dzIdleHTML() {
  const bars = [38, 64, 92, 55, 78, 100, 46, 70, 88, 40, 60];
  return `
    <div class="dz-visual" aria-hidden="true">
      ${bars.map((h, i) => `<i style="--h:${h};--d:${(i * 0.17).toFixed(2)}"></i>`).join("")}
      <span class="arrow">${icon("arrow")}</span>
      <div class="dz-lines"><i style="width:68px"></i><i style="width:88px"></i><i style="width:52px"></i></div>
    </div>
    <div class="dz-title">Drop a lecture recording</div>
    <div class="dz-sub">WAV · MP3 · MP4 · FLAC · OGG · MKV — or click to browse</div>`;
}

function signalHTML() {
  const stages = ctx.status?.stages || {};
  const chip = (key) => {
    if (!ctx.status) return `<span class="chip"><span class="dot"></span>Checking…</span>`;
    const live = stages[key]?.live;
    return `<span class="chip" data-mode="${live ? "live" : "fallback"}" title="${esc(live ? stages[key].model : stages[key]?.fallback || "")}"><span class="dot"></span>${live ? "Live" : "Fallback"}</span>`;
  };
  const nodes = [
    ["01 · Input", "Audio or video", "Resampled to 16 kHz mono", `<span class="chip" data-mode="live"><span class="dot"></span>Ready</span>`],
    ["02 · VAD", "Silero-VAD", "Speech chunks ≤ 25 s, silence dropped", chip("vad")],
    ["03 · ASR", stageName("asr", "Speech recognition"), "A model for each language", chip("asr")],
    ["04 · LLM", "BharatGen Param-1", "Structured notes + MCQs", chip("llm")],
    ["05 · MT", stageName("mt", "Translation"), "English → 10 Indian languages", chip("mt")],
  ];
  return nodes
    .map(
      ([step, name, desc, state]) => `
      <div class="signal-node"><span class="signal-step">${step}</span><span class="signal-name">${name}</span>
        <span class="signal-desc">${desc}</span><span class="signal-state">${state}</span></div>`
    )
    .join("");
}

function startRotator() {
  const el = ctx.root?.querySelector("#rotator");
  if (!el) return;
  const words = [...el.children];
  let i = 0;
  const show = () => {
    words.forEach((w, j) => {
      w.classList.toggle("on", j === i);
      w.classList.toggle("off", j !== i && w.classList.contains("on"));
    });
    el.style.width = `${words[i].offsetWidth + 4}px`;
  };
  show();
  // Fonts can load after first paint and change word widths.
  document.fonts?.ready.then(show);
  ctx.rotatorTimer = setInterval(() => {
    words[i].classList.remove("on");
    words[i].classList.add("off");
    i = (i + 1) % words.length;
    show();
  }, 2300);
}

function bindHero() {
  const root = ctx.root;
  const dz = root.querySelector("#dropzone");
  const input = root.querySelector("#fileInput");

  root.onclick = (e) => {
    const action = e.target.closest("[data-action]")?.dataset.action;
    if (action === "sample") return loadSample();
    if (action === "paste") return openPasteDialog();
    if (action === "cancel-upload") return ctx.uploadAbort?.abort();
  };
  dz.addEventListener("click", (e) => {
    if (!ctx.uploadAbort && !e.target.closest("[data-action]")) input.click();
  });
  dz.addEventListener("keydown", (e) => {
    if ((e.key === "Enter" || e.key === " ") && !ctx.uploadAbort) {
      e.preventDefault();
      input.click();
    }
  });
  input.addEventListener("change", () => input.files[0] && startUpload(input.files[0]));
  ["dragenter", "dragover"].forEach((t) =>
    dz.addEventListener(t, (e) => {
      e.preventDefault();
      dz.classList.add("dragging");
      ctx.river?.setRush(true);
    })
  );
  ["dragleave", "drop"].forEach((t) =>
    dz.addEventListener(t, () => {
      dz.classList.remove("dragging");
      ctx.river?.setRush(false);
    })
  );
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) startUpload(file);
  });
}

// ===========================================================================
// Sources
// ===========================================================================
function resetSession() {
  const prefs = Object.fromEntries(PREFS.map((k) => [k, S[k]]));
  S = freshState(prefs);
  ctx.audio?.pause();
  ctx.audio = null;
  ctx.audioFor = null;
  ctx.notesStream = "";
}

async function startUpload(file) {
  if (!MEDIA_RE.test(file.name)) {
    toast("That file type isn't supported — use WAV, MP3, MP4, FLAC, OGG or MKV.", { kind: "error" });
    return;
  }
  const inner = ctx.root?.querySelector("#dzInner");
  ctx.uploadAbort = new AbortController();
  const paint = (p) => {
    if (!inner) return;
    inner.innerHTML = `
      <div class="dz-upload">
        <div class="dz-title">Uploading ${esc(file.name)}</div>
        <div class="progress"><i style="--p:${p}"></i></div>
        <div class="stats"><span><b>${Math.round(p * 100)}%</b> of ${fmtBytes(file.size)}</span></div>
        <div><button class="btn btn-sm btn-quiet" data-action="cancel-upload">Cancel</button></div>
      </div>`;
  };
  paint(0);
  try {
    const record = await uploadFile(file, paint, ctx.uploadAbort.signal);
    resetSession();
    ctx.localFile = { id: record.id, file };
    S.source = { kind: "audio", ...record, url: `/api/files/${record.id}`, title: prettyName(record.name) };
    log("upload", `${record.name} · ${fmtBytes(record.size)}`, "ok");
    persist();
    render();
  } catch (err) {
    if (err.name !== "AbortError") toast(err.message, { kind: "error" });
    if (inner) inner.innerHTML = dzIdleHTML();
  } finally {
    ctx.uploadAbort = null;
  }
}

async function loadSample() {
  try {
    const record = await postJSON("/api/sample");
    resetSession();
    S.source = { kind: "sample", ...record, url: "/api/files/sample" };
    S.sourceLang = record.language;
    log("sample", `${record.title} · ${record.speaker}`, "ok");
    persist();
    render();
    runAll();
  } catch (err) {
    toast(err.message, { kind: "error" });
  }
}

function prettyName(name) {
  return name
    .replace(/\.[a-z0-9]+$/i, "")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/^./, (c) => c.toUpperCase());
}

export function openPasteDialog() {
  const dlg = document.getElementById("pasteDialog");
  const langs = asrLangs().length ? asrLangs() : [{ name: "English", native: "English" }];
  dlg.innerHTML = `
    <form method="dialog" id="pasteForm">
      <div class="sheet-head">
        <div><h2 id="pasteTitle">Paste a transcript</h2>
          <p>Skip the audio — notes, quiz and translations run straight from text.</p></div>
        <button class="icon-btn" value="cancel" aria-label="Close">${icon("x", "icon-sm")}</button>
      </div>
      <div class="sheet-body">
        <div class="field"><label for="pasteName">Lecture title</label>
          <input class="input" id="pasteName" placeholder="e.g. Raft consensus — Distributed Systems, week 6"></div>
        <div class="field"><label for="pasteLang">Spoken language</label>
          <select class="select" id="pasteLang"><option value="">Detect from the text</option>${langs
            .map((l) => `<option value="${esc(l.name)}">${esc(l.name)} · ${esc(l.native)}</option>`)
            .join("")}</select></div>
        <div class="field"><label for="pasteText">Transcript</label>
          <textarea class="textarea" id="pasteText" required placeholder="Paste or type the lecture transcript…"></textarea></div>
      </div>
      <div class="sheet-foot">
        <button class="btn btn-quiet" value="cancel">Cancel</button>
        <button class="btn btn-primary" value="ok" id="pasteOk">${icon("arrow", "icon-sm")}Use transcript</button>
      </div>
    </form>`;
  dlg.showModal();
  dlg.querySelector("#pasteText").focus();
  dlg.onclose = () => {
    if (dlg.returnValue !== "ok") return;
    const text = dlg.querySelector("#pasteText").value.trim();
    if (!text) return;
    const title = dlg.querySelector("#pasteName").value.trim();
    resetSession();
    S.source = { kind: "text", id: null, name: title || "Pasted transcript", title: title || "Pasted transcript" };
    S.sourceLang = dlg.querySelector("#pasteLang").value || detectLanguage(text) || "English";
    setTranscript(text, { mode: "manual", engine: "Pasted text", language: S.sourceLang });
    S.active = "notes";
    log("source", `Pasted transcript · ${plural(wordCount(text), "word")}`, "ok");
    persist();
    if (location.hash !== "#/studio") location.hash = "#/studio";
    render();
  };
}

function setTranscript(text, meta) {
  S.transcript = text;
  S.transcriptMeta = meta;
  S.transcriptRev += 1;
}

// ===========================================================================
// Rendering — workspace regions
// ===========================================================================
function updateRegions() {
  if (!ctx.root || !S.source) return;
  const stepper = ctx.root.querySelector("#stepper");
  if (stepper) stepper.innerHTML = stepperHTML();
  const actions = ctx.root.querySelector("#deckActions");
  if (actions) actions.innerHTML = deckActionsHTML();
  const chunkInfo = ctx.root.querySelector("#chunkInfo");
  if (chunkInfo) chunkInfo.textContent = chunkLabel();
  ctx.root.querySelector(".log-live")?.classList.toggle("on", Boolean(S.running));
  renderPanel();
}

const liveUpdate = rafThrottle(updateRegions);

function deckHTML() {
  const src = S.source;
  const isText = src.kind === "text";
  const title = src.title || src.name;
  const lang = S.transcriptMeta?.language || S.sourceLang;
  const meta = [
    src.kind === "sample" ? "Sample lecture · text-to-speech voice" : isText ? "Pasted transcript" : src.name,
    src.duration ? fmtTime(src.duration) : null,
    src.size ? fmtBytes(src.size) : null,
    `${lang} · ${nativeOf(lang)}`,
  ].filter(Boolean);
  const wave = isText
    ? ""
    : `<div class="wave" id="wave" aria-label="Waveform with speech chunks — click to seek"></div>
       <div class="transport">
         <button class="play-btn" data-action="play" aria-label="Play">${icon("play")}</button>
         <span class="transport-time"><span id="tNow">0:00</span> / <span id="tDur">${fmtTime(src.duration || 0)}</span></span>
         <span id="chunkInfo">${chunkLabel()}</span>
         <span class="spacer"></span>
         <button class="rate-btn" data-action="rate" title="Playback speed">${ctx.rate}×</button>
         <button class="btn btn-sm theatre-btn" data-action="theatre" title="Immersive listening (F)">${icon("sparkles", "icon-sm")}Theatre</button>
       </div>`;
  return `
    <div class="deck-head">
      <div class="deck-title">
        <button class="deck-emblem" id="emblem" data-action="theatre" title="This lecture's rangoli — every ring is a speech chunk" ${isText ? "disabled" : ""}>${icon(isText ? "clipboard" : "wave", "icon-lg")}</button>
        <div style="min-width:0"><h2 class="deck-name" title="${esc(title)}">${esc(title)}</h2>
          <div class="deck-meta">${meta.map((m) => `<span>${esc(m)}</span>`).join("")}</div></div>
      </div>
      <div class="deck-actions" id="deckActions">${deckActionsHTML()}</div>
    </div>${wave}`;
}

const chunkLabel = () =>
  S.chunks.length ? `${plural(S.chunks.length, "speech chunk")} · click a chunk to jump` : "Speech chunks appear after VAD";

function deckActionsHTML() {
  if (S.running) {
    return `<button class="btn" data-action="cancel">${icon("stop", "icon-sm")}Stop</button>
            <button class="btn btn-primary" disabled><span class="spinner"></span>Running ${esc(STAGES.find((s) => s.key === S.running)?.label || "")}…</button>`;
  }
  const pending = STAGES.filter((s) => needsRun(s.key)).length;
  return `<button class="btn btn-quiet" data-action="new">${icon("plus", "icon-sm")}New lecture</button>
          <button class="btn btn-primary" data-action="run-all" ${pending ? "" : 'title="Every stage is up to date"'}>
            ${icon(pending ? "bolt" : "check", "icon-sm")}${pending ? "Run pipeline" : "All stages done"}</button>`;
}

function stepperHTML() {
  return STAGES.map((s, i) => {
    const info = stageInfo(s.key);
    const st = S.stages[s.key];
    const current = S.active === s.key;
    const badge =
      info.status === "running"
        ? '<span class="spinner"></span>'
        : info.status === "error"
          ? icon("alert", "icon-sm")
          : info.status === "done" && !info.stale
            ? icon("check")
            : icon(info.status === "locked" ? "lock" : s.icon, "icon-sm");
    const indeterminate = info.status === "running" && !st.progress;
    return `<button class="step ${current ? "current" : ""}" data-action="stage" data-stage="${s.key}"
        data-status="${info.status}" data-stale="${Boolean(info.stale)}" ${current ? 'aria-current="step"' : ""}>
        <span class="step-badge">${badge}</span>
        <span class="step-label">${s.label}<kbd>${i + 1}</kbd></span>
        <span class="step-status">${info.stale ? '<i class="warn-dot"></i>' : ""}${esc(info.text)}</span>
        <span class="progress ${indeterminate ? "indeterminate" : ""}"><i style="--p:${st.progress || 0}"></i></span>
      </button>`;
  }).join("");
}

function logHTML() {
  return `
    <div class="log-head"><span class="log-live ${S.running ? "on" : ""}"></span><h3>Run log</h3>
      <button class="btn btn-sm btn-quiet" data-action="clear-log">Clear</button></div>
    <ol class="log-list" id="logList">${
      S.log.map(logItem).join("") ||
      '<li class="log-empty">Pipeline events stream here — stage starts, chunk progress, model loads and timings.</li>'
    }</ol>`;
}

// ── Audio + waveform ────────────────────────────────────────────────────
function mountWave() {
  const src = S.source;
  const el = ctx.root?.querySelector("#wave");
  if (!el || src.kind === "text") return;

  if (ctx.audioFor !== src.url) {
    ctx.audio?.pause();
    ctx.audio = new Audio();
    ctx.audio.preload = "metadata";
    ctx.audio.src = src.url;
    ctx.audio.playbackRate = ctx.rate;
    ctx.audioFor = src.url;
    const sync = () => {
      const btn = ctx.root?.querySelector('[data-action="play"]');
      if (btn) {
        btn.innerHTML = icon(ctx.audio.paused ? "play" : "pause");
        btn.setAttribute("aria-label", ctx.audio.paused ? "Play" : "Pause");
      }
    };
    ctx.audio.addEventListener("play", sync);
    ctx.audio.addEventListener("pause", sync);
    ctx.audio.addEventListener("ended", sync);
    ctx.audio.addEventListener("loadedmetadata", () => {
      const dur = ctx.root?.querySelector("#tDur");
      if (dur) dur.textContent = fmtTime(ctx.audio.duration);
    });
  }

  ctx.wave = new Waveform(el, ctx.audio, { duration: src.duration, onTime: onAudioTime });
  ctx.wave.setChunks(S.chunks);
  const local = ctx.localFile?.id === src.id ? ctx.localFile.file : null;
  ctx.wave.load(src.id, local || src.url, src.size).then(paintEmblem);
  if (!ctx.audio.paused) ctx.audio.dispatchEvent(new Event("play"));
}

function paintEmblem() {
  const el = ctx.root?.querySelector("#emblem");
  if (!el || !S.source) return;
  if (S.source.kind === "text" ? !S.transcript : !ctx.wave?.peaks) return;
  el.innerHTML = rangoliSVG(rangoliModel(), { dots: false });
}

function onAudioTime(t) {
  const root = ctx.root;
  if (!root) return;
  const now = root.querySelector("#tNow");
  if (now) now.textContent = fmtTime(t);
  const rows = root.querySelectorAll(".segment");
  if (rows.length) {
    const idx = S.segments.findIndex((s) => t >= s.start && t < s.end);
    rows.forEach((row) => row.classList.toggle("active", Number(row.dataset.i) === idx && t > 0));
  }
  syncKaraoke(t);
  paintMini();
}

/** Word-by-word follow-along in the transcript (timing is interpolated). */
function syncKaraoke(t) {
  const sentence = S.transcript && t > 0 ? lecture().sentenceAt(t) : null;
  const id = sentence ? sentence.id : -1;
  if (id !== ctx.sentence) {
    ctx.sentence = id;
    const panel = ctx.root.querySelector("#panel");
    panel?.querySelectorAll(".sent.now").forEach((el) => {
      el.classList.remove("now");
      el.querySelectorAll(".said").forEach((w) => w.classList.remove("said"));
    });
    const el = panel?.querySelector(`.sent[data-s="${id}"]`);
    el?.classList.add("now");
    ctx.words = el ? [...el.querySelectorAll(".kw")] : [];
  }
  for (const w of ctx.words) w.classList.toggle("said", Number(w.dataset.t) <= t);
}

function paintMini() {
  const mini = ctx.root?.querySelector("#mini");
  if (!mini) return;
  const a = ctx.audio;
  const show = Boolean(a) && !ctx.deckVisible && !a.ended && (!a.paused || a.currentTime > 0) && !isTheatreOpen();
  mini.hidden = !show;
  if (!show) return;
  mini.querySelector(".play-btn").innerHTML = icon(a.paused ? "play" : "pause");
  mini.querySelector("#miniTime").textContent = fmtTime(a.currentTime);
  const text = S.transcript ? lecture().sentenceAt(a.currentTime)?.text : "";
  const label = mini.querySelector("#miniText");
  const next = text || S.source?.title || "";
  if (label.textContent !== next) label.textContent = next;
}

function enterTheatre() {
  const src = S.source;
  if (!ctx.audio || !src || src.kind === "text") return;
  openTheatre({
    audio: ctx.audio,
    index: lecture(),
    model: rangoliModel(),
    concepts: S.transcript ? conceptGraph().concepts : [],
    title: src.title || src.name,
    duration: lectureDuration() || 1,
  });
}

/** Play a sentence, or show it in the transcript when there's no audio. */
function jumpToSentence(id, time) {
  const sentence = lecture().sentences[id];
  if (!sentence) return;
  if (sentence.start != null && ctx.audio) {
    ctx.audio.currentTime = (time ?? sentence.start) + 0.01;
    ctx.audio.play().catch(() => {});
    return;
  }
  S.transcriptView = "segments";
  setActive("transcript");
  renderPanel(true);
  const el = ctx.root?.querySelector(`.sent[data-s="${id}"]`);
  if (el) {
    el.scrollIntoView({ behavior: "smooth", block: "center" });
    el.classList.add("flash");
    setTimeout(() => el.classList.remove("flash"), 1800);
  }
}

function anchorHTML(sentence, label = "") {
  if (!sentence) return "";
  const timed = sentence.start != null;
  const text = timed ? fmtTime(sentence.start) : "Source";
  return `<button class="anchor" type="button" data-action="jump" data-s="${sentence.id}" title="${esc(sentence.text)}">${icon(timed ? "play" : "arrow", "icon-xs")}${text}${label ? ` · ${esc(label)}` : ""}</button>`;
}

function togglePlay() {
  const a = ctx.audio;
  if (!a) return;
  if (a.paused) a.play().catch((err) => toast(`Playback failed: ${err.message}`, { kind: "error" }));
  else a.pause();
}

// ===========================================================================
// Panels
// ===========================================================================
function renderPanel(force = false) {
  const el = ctx.root?.querySelector("#panel");
  if (!el) return;
  if (!force && el.contains(document.activeElement) && document.activeElement.tagName === "TEXTAREA") return;
  const builders = {
    transcript: transcriptPanel,
    notes: notesPanel,
    quiz: quizPanel,
    translate: translatePanel,
    concepts: conceptsPanel,
  };
  el.innerHTML = builders[S.active]();
  el.dataset.stage = S.active;
  ctx.sentence = -2;
  ctx.words = [];

  if (S.active === "quiz" && S.quiz.length && S.stages.quiz.status !== "running") {
    mountQuizHost(el.querySelector("#quizHost"), S.quiz, "en");
  }
  if (S.active === "notes" && S.notes && S.notesView === "rendered" && S.stages.notes.status !== "running") {
    const headings = el.querySelectorAll("#notesDoc h2");
    noteSections().forEach((section, i) => {
      if (section.sentence) headings[i]?.insertAdjacentHTML("beforeend", anchorHTML(section.sentence));
    });
  }
  if (S.active === "concepts") mountConceptMap(el);
  if (S.active === "translate" && S.translateTab === "quiz") {
    const tr = S.translations[S.target];
    if (tr?.quiz?.length) mountQuizHost(el.querySelector("#quizHost"), tr.quiz, S.target, S.target);
  }
  if (S.active === "transcript" && ctx.audio) onAudioTime(ctx.audio.currentTime);
  el.querySelector("#transcriptEditor")?.focus();
}

function mountQuizHost(host, quiz, key, lang = null) {
  if (!host) return;
  S.quizAnswers[key] ||= {};
  mountQuiz(host, quiz, {
    mode: S.quizMode,
    answers: S.quizAnswers[key],
    lang,
    anchors: quiz.length === S.quiz.length ? quizMoments().map((m) => anchorHTML(m, "Hear it in the lecture")) : [],
    onChange: persist,
    onMode: (mode) => {
      S.quizMode = mode;
      persist();
      renderPanel();
    },
  });
}

// ── Shared fragments ────────────────────────────────────────────────────
function panelHead(title, chips = [], tools = "") {
  return `<div class="panel-head"><h2 class="panel-title">${title}</h2>
    <div class="panel-meta">${chips.filter(Boolean).join("")}</div>
    ${tools ? `<div class="panel-tools">${tools}</div>` : ""}</div>`;
}

const chip = (text, attrs = "") => `<span class="chip" ${attrs}>${text}</span>`;

function modeChip(meta) {
  if (!meta?.mode) return "";
  const label = meta.cached ? "Model output · stored" : MODE_LABEL[meta.mode] || meta.mode;
  const when = meta.cached && meta.created ? ` · generated ${new Date(meta.created * 1000).toLocaleDateString()}` : "";
  return `<span class="chip" data-mode="${esc(meta.mode)}" title="${esc((meta.engine || "") + when)}"><span class="dot"></span>${esc(label)}</span>`;
}

const timeChip = (seconds) => (seconds != null ? chip(`${icon("clock", "icon-sm")}${seconds}s`) : "");

function seg(action, current, options, attrs = "") {
  return `<div class="seg" role="group" ${attrs}>${options
    .map(
      ([value, label, glyph]) =>
        `<button type="button" data-action="${action}" data-value="${value}" aria-pressed="${current === value}">${glyph ? icon(glyph, "icon-sm") : ""}${label}</button>`
    )
    .join("")}</div>`;
}

function engineLine(stage) {
  const info = ctx.status?.stages?.[stage];
  if (!info) return "";
  return info.live
    ? `${icon("cpu", "icon-sm")} ${esc(info.name)} · <span class="mono">${esc(info.model)}</span>`
    : `${icon("info", "icon-sm")} ${esc(info.name)} isn't installed — using: ${esc(info.fallback)}`;
}

function runnerHTML(title, steps, phase, detail, progress) {
  return `<div class="runner">
      <div class="runner-head"><span class="spinner"></span>${esc(title)}</div>
      <ul class="runner-steps">${steps
        .map(([label, sub], i) => {
          const cls = i < phase ? "done" : i === phase ? "active" : "";
          return `<li class="${cls}"><span class="tick">${i < phase ? icon("check", "icon-sm") : ""}</span>${esc(label)}<span class="muted">${esc(sub)}</span>${
            i === phase && detail ? `<span class="runner-detail">${esc(detail)}</span>` : ""
          }</li>`;
        })
        .join("")}</ul>
      <div class="progress ${progress ? "" : "indeterminate"}"><i style="--p:${progress || 0}"></i></div>
    </div>`;
}

function errorHTML(error, actions = []) {
  return `<div class="callout is-error" role="alert">${icon("alert")}
      <div class="callout-body"><strong>${esc(error?.message || "Something went wrong.")}</strong>
        ${error?.hint ? `<span>${esc(error.hint)}</span>` : ""}
        ${actions.length ? `<div class="callout-actions">${actions.join("")}</div>` : ""}</div></div>`;
}

function staleHTML(what, action) {
  return `<div class="callout is-warn">${icon("alert")}
      <div class="callout-body"><strong>The ${what} changed since this was generated.</strong>
        <span>Regenerate to bring it back in sync.</span>
        <div class="callout-actions"><button class="btn btn-sm" data-action="${action}">${icon("refresh", "icon-sm")}Regenerate</button></div></div></div>`;
}

function lockedHTML(glyph, title, text, goto, gotoLabel) {
  return `<div class="empty"><div class="empty-art">${icon(glyph, "icon-lg")}</div><h3>${title}</h3><p>${text}</p>
      <div class="empty-controls"><button class="btn" data-action="stage" data-stage="${goto}">${icon("arrow", "icon-sm")}${gotoLabel}</button></div></div>`;
}

const btn = (action, label, glyph, cls = "", extra = "") =>
  `<button class="btn ${cls}" data-action="${action}" ${extra}>${glyph ? icon(glyph, "icon-sm") : ""}${label}</button>`;

// ── Transcript ──────────────────────────────────────────────────────────
function transcriptPanel() {
  const st = S.stages.transcript;
  const meta = S.transcriptMeta;
  const running = st.status === "running";
  const editing = S.transcriptView === "edit";
  const has = Boolean(S.transcript);

  const tools =
    has && !running && !editing
      ? `${seg("transcript-view", S.transcriptView, [["segments", "Synced", "wave"], ["edit", "Edit", "edit"]])}
         ${btn("copy-transcript", "", "copy", "btn-sm btn-quiet", 'title="Copy transcript" aria-label="Copy transcript"')}
         ${btn("dl-transcript", ".txt", "download", "btn-sm")}
         ${S.source.kind === "text" ? "" : btn("transcribe", "Re-transcribe", "refresh", "btn-sm")}`
      : "";
  const head = panelHead(
    "Transcript",
    has ? [modeChip(meta), meta?.edited ? chip("Edited") : "", chip(esc(`${meta?.language || S.sourceLang} · ${nativeOf(meta?.language || S.sourceLang)}`))] : [],
    tools
  );

  let body;
  if (running) {
    body = runnerHTML(
      "Transcribing the lecture",
      [["Detect speech", "voice activity detection"], ["Transcribe chunks", stageName("asr", "speech recognition")]],
      st.phase,
      st.detail,
      st.progress
    );
  } else if (editing) {
    const downstream = S.notes || S.quiz.length;
    body = `<div class="field"><label class="sr-only" for="transcriptEditor">Transcript</label>
        <textarea class="textarea lang-text" id="transcriptEditor" rows="14" placeholder="Paste or type the lecture transcript…">${esc(S.transcript)}</textarea></div>
      <div class="edit-actions"><span class="muted">${downstream ? "Saving marks notes, quiz and translations as out of date." : "Notes and the quiz are generated from this text."}</span>
        ${btn("cancel-edit", "Cancel", "", "btn-quiet")}${btn("save-transcript", "Save transcript", "check", "btn-primary")}</div>`;
  } else if (st.status === "error") {
    body = errorHTML(st.error, [
      btn("paste-inline", "Type or paste the transcript", "edit", "btn-sm btn-primary"),
      btn("transcribe", "Try again", "retry", "btn-sm"),
    ]);
  } else if (!has) {
    const langs = asrLangs();
    const locked = S.source.kind === "sample";
    body = `<div class="empty">
        <div class="empty-art">${icon("mic", "icon-lg")}</div>
        <h3>${locked ? "Transcribe the sample lecture" : "What language is the lecture in?"}</h3>
        <p>Silero-VAD splits the recording into speech chunks, then ${esc(stageName("asr", "the speech model"))} transcribes each chunk${locked ? "." : " in the language you pick."}</p>
        ${locked ? "" : `<div class="lang-grid">${langs
          .map(
            (l) => `<button class="lang-chip" type="button" data-action="asr-lang" data-value="${esc(l.name)}" aria-pressed="${S.sourceLang === l.name}">
                <b>${esc(l.native)}</b><small>${esc(l.name)}</small></button>`
          )
          .join("")}</div>`}
        <div class="empty-controls">${btn("transcribe", `Transcribe${locked ? "" : ` in ${esc(S.sourceLang)}`}`, "wave", "btn-primary btn-lg")}
          ${btn("paste-inline", "or paste a transcript", "", "btn-quiet")}</div>
        <div class="engine-line">${engineLine("asr")}</div></div>`;
  } else {
    const words = wordCount(S.transcript);
    const t = meta?.timings;
    const stats = `<div class="stats"><span><b>${words}</b> words</span><span><b>${S.transcript.length}</b> characters</span>
        ${S.segments.length ? `<span><b>${S.segments.length}</b> segments</span>` : ""}<span>~<b>${readMinutes(words)}</b> min read</span>
        ${t ? `<span>VAD <b>${t.vad}s</b> · ASR <b>${t.asr}s</b></span>` : ""}</div>`;
    const index = lecture();
    const content = S.segments.length
      ? `<ol class="segments">${S.segments
          .map((s, i) => {
            const sentences = index.sentences.filter((x) => x.segment === i);
            return `<li class="segment" data-i="${i}">
                <button class="segment-time" data-action="seek" data-t="${s.start}" title="Play from ${fmtTime(s.start)}"><b>${fmtTime(s.start)}</b><span>${fmtTime(s.end)}</span></button>
                <div class="segment-text lang-text">${sentences.map(sentenceHTML).join(" ") || '<span class="muted">(no speech recognised)</span>'}</div></li>`;
          })
          .join("")}</ol>`
      : `<div class="transcript-block lang-text">${index.sentences.map(sentenceHTML).join(" ")}</div>`;
    body = stats + content;
  }
  return head + body;
}

/** A sentence whose words can be followed and clicked during playback. */
function sentenceHTML(s) {
  if (s.start == null) return `<span class="sent" data-s="${s.id}">${esc(s.text)}</span>`;
  let html = "";
  let cursor = 0;
  for (const t of s.tokens) {
    html += esc(s.text.slice(cursor, t.start));
    html += `<span class="kw" data-action="seek" data-t="${lecture().timeOf(s, t).toFixed(2)}">${esc(t.text)}</span>`;
    cursor = t.end;
  }
  return `<span class="sent" data-s="${s.id}">${html}${esc(s.text.slice(cursor))}</span>`;
}

// ── Notes ───────────────────────────────────────────────────────────────
function notesPanel() {
  const st = S.stages.notes;
  const running = st.status === "running";
  if (!S.transcript && !running) {
    return panelHead("Notes") + lockedHTML("notes", "Notes need a transcript", "Transcribe the lecture or paste a transcript first.", "transcript", "Go to transcript");
  }
  const meta = S.notesMeta;
  const has = Boolean(S.notes) && !running;
  const words = wordCount(S.notes);
  const tools = has
    ? `${seg("notes-view", S.notesView, [["rendered", "Document", "doc"], ["markdown", "Markdown", "code"]])}
       ${btn("copy-notes", "", "copy", "btn-sm btn-quiet", 'title="Copy Markdown" aria-label="Copy Markdown"')}
       ${btn("print-notes", "", "print", "btn-sm btn-quiet", 'title="Print or save as PDF" aria-label="Print notes"')}
       ${btn("dl-notes", ".md", "download", "btn-sm")}
       ${btn("notes", "Regenerate", "refresh", "btn-sm")}`
    : "";
  const head = panelHead(
    "Notes",
    has ? [modeChip(meta), chip(`${words} words · ${readMinutes(words)} min`), timeChip(meta?.seconds)] : [],
    tools
  );

  if (running) {
    if (!ctx.notesStream) {
      return head + runnerHTML("Writing notes", [["Read transcript", ""], ["Write structured notes", "streaming"]], 0, st.detail, 0);
    }
    const { html } = renderMarkdown(ctx.notesStream);
    return head + `<div class="doc-layout"><div class="toc"></div><article class="doc streaming">${html}</article></div>`;
  }
  if (st.status === "error") return head + errorHTML(st.error, [btn("notes", "Try again", "retry", "btn-sm")]);
  if (!S.notes) {
    return head + `<div class="empty"><div class="empty-art">${icon("sparkles", "icon-lg")}</div>
        <h3>Write structured notes</h3>
        <p>Title, sections, bolded key terms and a takeaways box — streamed live as they're written.</p>
        <div class="empty-controls">${btn("notes", "Generate notes", "sparkles", "btn-primary btn-lg")}</div>
        <div class="engine-line">${engineLine("llm")}</div></div>`;
  }

  const stale = notesStale() ? staleHTML("transcript", "notes") : "";
  if (S.notesView === "markdown") return head + stale + `<pre class="md-source">${esc(S.notes)}</pre>`;
  const { html, toc } = renderMarkdown(S.notes);
  const tocHTML = toc.length
    ? `<nav class="toc" aria-label="Sections"><span class="toc-title">On this page</span>${toc
        .map((t) => `<a href="#${t.id}" data-action="toc" data-id="${t.id}">${esc(t.text)}</a>`)
        .join("")}</nav>`
    : "<div></div>";
  return head + stale + `<div class="doc-layout">${tocHTML}<article class="doc" id="notesDoc">${html}</article></div>`;
}

// ── Concepts ────────────────────────────────────────────────────────────
function conceptsPanel() {
  if (!S.transcript) {
    return panelHead("Concepts") + lockedHTML("layers", "Concepts need a transcript", "Transcribe the lecture or paste a transcript first.", "transcript", "Go to transcript");
  }
  const { concepts, edges } = conceptGraph();
  const head = panelHead("Concepts", [chip(plural(concepts.length, "concept")), chip(plural(edges.length, "link"))]);
  if (!concepts.length) {
    return head + `<div class="empty"><div class="empty-art">${icon("layers", "icon-lg")}</div><h3>Not enough to map yet</h3>
        <p>Concepts come from your notes and from words the lecturer repeats. Add a longer transcript or generate notes.</p></div>`;
  }
  return head + `<p class="panel-sub">The ideas in this lecture, left to right in the order they come up. Linked ideas are discussed together. Pick one to hear every moment it's mentioned${
      S.notes ? "" : " — generate notes for cleaner English labels"
    }.</p>
    <div class="cmap" id="conceptMap"></div><div id="moments"></div>`;
}

const measureCtx = document.createElement("canvas").getContext("2d");

function mountConceptMap(panel) {
  const host = panel.querySelector("#conceptMap");
  if (!host) return;
  const { concepts, edges } = conceptGraph();
  const W = host.clientWidth;
  if (!W) return;

  measureCtx.font = '600 13px "Geist", system-ui, sans-serif';
  const labelWidth = (c) => measureCtx.measureText(c.label).width;
  measureCtx.font = '12px "Noto Sans Devanagari", "Noto Sans Tamil", system-ui, sans-serif';
  const widths = new Map(concepts.map((c) => [c.id, Math.max(labelWidth(c), c.surface ? measureCtx.measureText(c.surface).width : 0)]));

  const area = concepts.reduce((sum, c) => sum + (widths.get(c.id) + 40) * (c.surface ? 56 : 42), 0);
  const stage = Math.max(320, Math.min(620, (area / W) * 2.1));
  const timeline = 58;
  const H = stage + timeline;
  const nodes = layoutConcepts(concepts, edges, W, 8, stage - 6, (c) => widths.get(c.id));

  const pad = 12;
  const span = lectureDuration() || lecture().sentences.at(-1)?.end || lecture().sentences.length || 1;
  const tx = (time) => pad + Math.min(1, time / span) * (W - 2 * pad);
  const hitX = (h) => tx(h.time ?? h.sentence.id);
  const bandY = stage + 14;

  const edgeSVG = edges
    .map((e) => {
      const a = nodes[e.a];
      const b = nodes[e.b];
      const mx = (a.x + b.x) / 2;
      return `<path class="c-edge" data-a="${e.a}" data-b="${e.b}" stroke-width="${Math.min(2.4, 0.8 + e.weight * 0.5)}" d="M${a.x} ${a.y}C${mx} ${a.y} ${mx} ${b.y} ${b.x} ${b.y}"/>`;
    })
    .join("");

  const nodeSVG = nodes
    .map((n) => {
      const c = n.c;
      const labelY = c.surface ? 18 : n.h / 2 + 4.5;
      return `<g class="c-node ${c.fromNotes ? "" : "c-raw"}" data-id="${c.id}" tabindex="0" role="button" aria-label="${esc(c.label)}, ${plural(c.hits.length, "mention")}" transform="translate(${n.x - n.w / 2} ${n.y - n.h / 2})">
          <rect width="${n.w}" height="${n.h}" rx="${Math.min(15, n.h / 2)}"/>
          <text class="c-label" x="${n.w / 2}" y="${labelY}" text-anchor="middle">${esc(c.label)}</text>
          ${c.surface ? `<text class="c-sub" x="${n.w / 2}" y="35" text-anchor="middle">${esc(c.surface)}</text>` : ""}
        </g>`;
    })
    .join("");

  const chunkSVG = (S.chunks.length ? S.chunks : [{ start: 0, end: span }])
    .map((c) => `<rect class="c-chunk" x="${tx(c.start)}" y="${bandY}" width="${Math.max(1, tx(c.end) - tx(c.start))}" height="18" rx="4"/>`)
    .join("");

  const ticks = nodes
    .flatMap((n) =>
      n.c.hits.map((h) => `<line class="c-tick" data-id="${n.c.id}" x1="${hitX(h)}" x2="${hitX(h)}" y1="${bandY + 3}" y2="${bandY + 15}"/>`)
    )
    .join("");

  const end = lecture().timed ? fmtTime(span) : "end";
  host.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">
      <g class="c-links" id="cLinks"></g>${edgeSVG}${nodeSVG}
      ${chunkSVG}${ticks}
      <text class="c-axis" x="${pad}" y="${bandY + 34}">${lecture().timed ? "0:00" : "start"}</text>
      <text class="c-axis" x="${W - pad}" y="${bandY + 34}" text-anchor="end">${end}</text>
    </svg>`;

  const svg = host.querySelector("svg");
  const neighbours = (id) => {
    const set = new Set([id]);
    edges.forEach((e) => {
      if (e.a === id) set.add(e.b);
      if (e.b === id) set.add(e.a);
    });
    return set;
  };

  const highlight = (id) => {
    const near = id == null ? null : neighbours(id);
    svg.classList.toggle("focus", id != null);
    svg.querySelectorAll(".c-node").forEach((g) => {
      const n = Number(g.dataset.id);
      g.classList.toggle("near", Boolean(near?.has(n)));
      g.classList.toggle("sel", n === id);
    });
    svg.querySelectorAll(".c-edge").forEach((p) =>
      p.classList.toggle("near", id != null && (Number(p.dataset.a) === id || Number(p.dataset.b) === id))
    );
    svg.querySelectorAll(".c-tick").forEach((t) => t.classList.toggle("sel", Number(t.dataset.id) === id));
    const links = svg.querySelector("#cLinks");
    if (id == null) return (links.innerHTML = "");
    const n = nodes[id];
    links.innerHTML = n.c.hits
      .map((h) => {
        const x = hitX(h);
        const y0 = n.y + n.h / 2;
        const midY = (y0 + bandY) / 2;
        return `<path class="c-link" d="M${n.x} ${y0}C${n.x} ${midY} ${x} ${midY} ${x} ${bandY}"/>`;
      })
      .join("");
  };

  const select = (id) => {
    ctx.concept = id;
    highlight(id);
    paintMoments(panel, concepts[id]);
    ctx.wave?.setMarks(concepts[id].hits.map((h) => h.time).filter((t) => t != null));
  };

  svg.addEventListener("pointerover", (e) => {
    const g = e.target.closest(".c-node");
    if (g) highlight(Number(g.dataset.id));
  });
  svg.addEventListener("pointerleave", () => highlight(ctx.concept));
  svg.addEventListener("click", (e) => {
    const g = e.target.closest(".c-node");
    if (g) select(Number(g.dataset.id));
  });
  svg.addEventListener("keydown", (e) => {
    const g = e.target.closest?.(".c-node");
    if (g && (e.key === "Enter" || e.key === " ")) {
      e.preventDefault();
      e.stopPropagation();
      select(Number(g.dataset.id));
    }
  });

  select(ctx.concept != null && concepts[ctx.concept] ? ctx.concept : 0);
}

function markedSentence(sentence, from, to) {
  const toks = sentence.tokens;
  const a = toks[from].start;
  const b = toks[to - 1].end;
  return `${esc(sentence.text.slice(0, a))}<mark>${esc(sentence.text.slice(a, b))}</mark>${esc(sentence.text.slice(b))}`;
}

function paintMoments(panel, concept) {
  const host = panel.querySelector("#moments");
  if (!host) return;
  host.innerHTML = `
    <h3 class="moments-title">${esc(concept.label)}${concept.surface ? ` <span class="lang-text">${esc(concept.surface)}</span>` : ""}
      <span class="muted">· ${plural(concept.hits.length, "mention")}</span></h3>
    <ol class="moments">${concept.hits
      .map(
        (h) => `<li><button class="moment" type="button" data-action="jump" data-s="${h.sentence.id}" ${h.time != null ? `data-t="${h.time.toFixed(2)}"` : ""}>
            <span class="moment-time">${h.time != null ? `${icon("play", "icon-xs")}${fmtTime(h.time)}` : icon("arrow", "icon-xs")}</span>
            <span class="lang-text">${markedSentence(h.sentence, h.from, h.to)}</span></button></li>`
      )
      .join("")}</ol>`;
}

// ── Quiz ────────────────────────────────────────────────────────────────
function quizPanel() {
  const st = S.stages.quiz;
  const running = st.status === "running";
  if (!S.transcript && !running) {
    return panelHead("Quiz") + lockedHTML("quiz", "The quiz needs a transcript", "Transcribe the lecture or paste a transcript first.", "transcript", "Go to transcript");
  }
  const counter = `<div class="stepper-num" title="Number of questions">
      <button type="button" data-action="q-count" data-value="-1" aria-label="Fewer questions" ${S.numQuestions <= 3 ? "disabled" : ""}>${icon("minus", "icon-sm")}</button>
      <output aria-live="polite">${S.numQuestions}</output>
      <button type="button" data-action="q-count" data-value="1" aria-label="More questions" ${S.numQuestions >= 10 ? "disabled" : ""}>${icon("plus", "icon-sm")}</button></div>`;
  const has = S.quiz.length > 0 && !running;
  const tools = has
    ? `${seg("quiz-mode", S.quizMode, [["practice", "Practice", "bolt"], ["review", "Review", "eye"]])}${counter}
       ${btn("quiz", "Regenerate", "refresh", "btn-sm")}${btn("dl-quiz", ".json", "download", "btn-sm")}`
    : "";
  const head = panelHead("Quiz", has ? [modeChip(S.quizMeta), chip(plural(S.quiz.length, "question")), timeChip(S.quizMeta?.seconds)] : [], tools);

  if (running) {
    return head + runnerHTML("Writing questions", [["Pick key concepts", ""], ["Write options & explanations", ""]], st.progress ? 1 : 0, st.detail, st.progress);
  }
  if (st.status === "error") return head + errorHTML(st.error, [btn("quiz", "Try again", "retry", "btn-sm")]);
  if (!S.quiz.length) {
    return head + `<div class="empty"><div class="empty-art">${icon("trophy", "icon-lg")}</div>
        <h3>Test yourself on this lecture</h3>
        <p>Multiple-choice questions with plausible distractors and an explanation for every answer.</p>
        <div class="empty-controls">${counter}<span class="muted">questions</span>${btn("quiz", "Generate quiz", "sparkles", "btn-primary btn-lg")}</div>
        <div class="engine-line">${engineLine("llm")}</div></div>`;
  }
  return head + (quizStale() ? staleHTML("transcript", "quiz") : "") + `<div id="quizHost"></div>`;
}

// ── Translate ───────────────────────────────────────────────────────────
function translatePanel() {
  const st = S.stages.translate;
  const running = st.status === "running";
  if (!S.notes && !S.quiz.length && !running) {
    return panelHead("Translate") + lockedHTML("translate", "Nothing to translate yet", "Generate notes or a quiz, then translate them into Indian languages with IndicTrans2.", "notes", "Go to notes");
  }
  const tr = S.translations[S.target];
  const showResult = tr && !running;
  const tools = showResult
    ? `${S.translateTab === "notes" && tr.notes ? seg("tr-view", S.translateView, [["pairs", "Side by side", "columns"], ["doc", "Translation", "doc"]]) : ""}
       ${btn("dl-translation", S.translateTab === "notes" ? ".md" : ".json", "download", "btn-sm")}
       ${btn("translate", "Retranslate", "refresh", "btn-sm")}`
    : "";
  const head = panelHead("Translate", showResult ? [modeChip(tr), chip(esc(`${tr.native} · ${tr.code}`)), timeChip(tr.seconds)] : [], tools);

  const picker = `<div class="translate-controls"><span class="field"><span class="label">Target language</span></span>
      <div class="lang-grid">${targets()
        .map((l) => {
          const av = availability(l);
          const done = S.translations[l.name] ? " · done" : "";
          const sub = av === "live" ? stageName("mt", "model") : av === "sample" ? "stored" : "needs model";
          return `<button class="lang-chip ${av === "locked" ? "locked" : ""}" type="button" data-action="target" data-value="${esc(l.name)}" aria-pressed="${S.target === l.name}">
              <b>${esc(l.native)}</b><small><i class="avail ${av === "live" ? "on" : av}"></i>${esc(l.name)} · ${sub}${done}</small></button>`;
        })
        .join("")}</div></div>`;

  let body;
  const err = st.status === "error" && st.error?.target === S.target;
  if (running) {
    body = runnerHTML(`Translating to ${S.target}`, [["Notes", ""], ["Quiz", ""]], st.progress >= 0.5 ? 1 : 0, st.detail, st.progress);
  } else if (err) {
    body = errorHTML(st.error, [btn("translate", "Try again", "retry", "btn-sm")]);
  } else if (!tr) {
    const what = [S.notes ? "notes" : "", S.quiz.length ? "quiz" : ""].filter(Boolean).join(" and ");
    body = `<div class="empty"><div class="empty-art">${icon("globe", "icon-lg")}</div>
        <h3>Translate into ${esc(nativeOf(S.target))}</h3>
        <p>Your ${what} will be translated line by line, keeping headings, lists and the quiz answer key intact.</p>
        <div class="empty-controls">${btn("translate", `Translate to ${esc(S.target)}`, "translate", "btn-primary btn-lg")}</div>
        <div class="engine-line">${engineLine("mt")}</div></div>`;
  } else {
    const tabs = [tr.notes ? ["notes", "Notes", "notes"] : null, tr.quiz?.length ? ["quiz", "Quiz", "quiz"] : null].filter(Boolean);
    if (!tabs.some(([v]) => v === S.translateTab)) S.translateTab = tabs[0][0];
    const stale = trStale(S.target) ? staleHTML("notes or quiz", "translate") : "";
    let content;
    if (S.translateTab === "quiz") content = `<div id="quizHost"></div>`;
    else if (S.translateView === "doc") content = `<article class="doc lang-text">${renderMarkdown(tr.notes).html}</article>`;
    else content = pairsHTML(S.notes, tr.notes, tr.native, S.target);
    body = stale + (tabs.length > 1 ? seg("tr-tab", S.translateTab, tabs) : "") + content;
  }
  return head + picker + body;
}

function pairsHTML(source, translated, native, lang) {
  const a = (source || "").split("\n");
  const b = (translated || "").split("\n");
  const rows = [];
  let lastGap = true;
  for (let i = 0; i < Math.max(a.length, b.length); i++) {
    const ka = lineKind(a[i] || "");
    const kb = lineKind(b[i] || "");
    const kind = ka.kind === "gap" ? kb.kind : ka.kind;
    if (kind === "gap") {
      if (!lastGap) rows.push('<div class="pair gap"><div></div><div></div></div>');
      lastGap = true;
      continue;
    }
    lastGap = false;
    const cell = (k) => `${k.mark ? `<span class="li-mark">${esc(k.mark)}</span>` : ""}${inline(k.text)}`;
    rows.push(`<div class="pair ${kind}"><div>${cell(ka)}</div><div class="lang-text">${cell(kb)}</div></div>`);
  }
  return `<div class="pairs"><div class="pairs-head"><div>English · source</div><div>${esc(native)} · ${esc(lang)}</div></div>${rows.join("")}</div>`;
}

// ===========================================================================
// Pipeline runs
// ===========================================================================
function setActive(key) {
  if (S.active === key) return;
  S.active = key;
  if (key !== "concepts") ctx.wave?.setMarks([]);
  persist();
  updateRegions();
  renderPanel(true);
}

function onStreamEvent(stageKey, ev) {
  const st = S.stages[stageKey];
  switch (ev.event) {
    case "stage": {
      st.detail = ev.label;
      if (stageKey === "transcript") {
        if (ev.stage === "vad" && ev.status === "start") st.progress = 0.04;
        if (ev.stage === "vad" && ev.status === "done") {
          st.phase = 1;
          st.progress = 0.2;
          S.chunks = ev.chunks || [];
          ctx.wave?.setChunks(S.chunks);
        }
        if (ev.stage === "asr" && ev.status === "start") ctx.wave?.setChunkState(0, "scanning");
        if (ev.stage === "asr" && ev.status === "done") st.phase = 2;
      } else if (stageKey === "notes" && ev.status === "start") {
        st.phase = 1;
      }
      const secs = ev.seconds != null ? ` · ${ev.seconds}s` : "";
      log(ev.stage, `${ev.label}${secs}`, ev.status === "done" ? "ok" : "info");
      break;
    }
    case "progress": {
      st.detail = ev.detail;
      st.progress = stageKey === "transcript" ? 0.2 + 0.8 * ev.value : ev.value;
      if (stageKey === "transcript" && ev.chunk != null) {
        ctx.wave?.setChunkState(ev.chunk, "done");
        ctx.wave?.setChunkState(ev.chunk + 1, "scanning");
      }
      break;
    }
    case "token":
      ctx.notesStream += ev.text;
      st.detail = `${plural(wordCount(ctx.notesStream), "word")} written`;
      break;
    case "log":
      log(stageKey, ev.message, ev.level === "warning" ? "warn" : "info");
      break;
    case "error":
      log(stageKey, ev.message, "error");
      break;
  }
  liveUpdate();
}

const RUNNERS = {
  async transcript(signal) {
    ctx.wave?.clearChunkStates();
    const r = await stream(
      "/api/transcribe",
      { file_id: S.source.id, language: S.sourceLang, fresh: Boolean(S.transcript) },
      (ev) => onStreamEvent("transcript", ev),
      signal
    );
    S.chunks = r.chunks;
    S.segments = r.segments;
    setTranscript(r.transcript, {
      engine: r.engine,
      mode: r.mode,
      cached: r.cached,
      created: r.created,
      timings: r.timings,
      language: r.language,
    });
    S.transcriptView = "segments";
    ctx.wave?.setChunks(S.chunks);
  },
  async notes(signal) {
    ctx.notesStream = "";
    const title = S.source.kind === "text" && S.source.title !== "Pasted transcript" ? S.source.title : null;
    const fresh = Boolean(S.notes) && !notesStale();
    const r = await stream("/api/notes", { transcript: S.transcript, title, fresh }, (ev) => onStreamEvent("notes", ev), signal);
    S.notes = r.notes;
    S.notesMeta = { engine: r.engine, mode: r.mode, cached: r.cached, created: r.created, seconds: r.seconds, rev: S.transcriptRev };
    S.notesRev += 1;
    ctx.notesStream = "";
  },
  async quiz(signal) {
    const r = await stream(
      "/api/quiz",
      { transcript: S.transcript, num_questions: S.numQuestions, fresh: S.quiz.length > 0 && !quizStale() },
      (ev) => onStreamEvent("quiz", ev),
      signal
    );
    S.quiz = r.quiz;
    S.quizMeta = { engine: r.engine, mode: r.mode, cached: r.cached, created: r.created, seconds: r.seconds, rev: S.transcriptRev };
    S.quizRev += 1;
    S.quizAnswers = {};
    S.quizMode = "practice";
  },
  async translate(signal) {
    const target = S.target;
    S.stages.translate.target = target;
    const r = await stream(
      "/api/translate",
      { notes: S.notes, quiz: S.quiz, target, fresh: Boolean(S.translations[target]) && !trStale(target) },
      (ev) => onStreamEvent("translate", ev),
      signal
    );
    S.translations[target] = { ...r, srcKey: srcKey() };
    delete S.quizAnswers[target];
  },
};

function canRun(key) {
  if (!S.source) return false;
  if (key === "transcript") return S.source.kind !== "text";
  if (key === "translate") return Boolean(S.notes || S.quiz.length);
  return Boolean(S.transcript);
}

async function runStage(key, { follow = true } = {}) {
  if (S.running || !canRun(key)) return false;
  const st = S.stages[key];
  Object.assign(st, { status: "running", progress: 0, detail: "Starting…", error: null, phase: 0 });
  S.running = key;
  if (follow) S.active = key;
  if (key === "transcript") S.transcriptView = "segments";
  ctx.abort = new AbortController();
  updateRegions();
  renderPanel(true);

  let ok = false;
  try {
    await RUNNERS[key](ctx.abort.signal);
    st.status = "idle";
    ok = true;
  } catch (err) {
    if (err.name === "AbortError") {
      st.status = "idle";
      log(key, "Stopped", "warn");
    } else {
      st.status = "error";
      st.error = { message: err.message, hint: err.hint || "", target: S.stages.translate.target };
      if (!err.fromStream) log(key, err.message, "error");
    }
  } finally {
    S.running = null;
    ctx.abort = null;
    ctx.notesStream = "";
    if (key === "transcript") setTimeout(() => ctx.wave?.clearChunkStates(), 900);
    persist();
    updateRegions();
    renderPanel(true);
  }
  return ok;
}

async function runAll() {
  if (S.running) return;
  if (!STAGES.some((s) => needsRun(s.key))) {
    toast("Every stage is up to date — regenerate one from its panel.", { kind: "success" });
    return;
  }
  for (const { key } of STAGES) {
    // Checked as we go: regenerating notes makes the translation stale.
    if (!needsRun(key) || !canRun(key)) continue;
    const ok = await runStage(key);
    if (!ok) {
      if (S.stages[key].status === "error") {
        toast(key === "translate" ? "Pipeline done — translation needs the IndicTrans2 model." : `Stopped at ${key}: ${S.stages[key].error?.message}`, {
          kind: key === "translate" ? "info" : "error",
          ms: 4200,
        });
      }
      return;
    }
  }
  toast("Pipeline complete.", { kind: "success" });
}

// ===========================================================================
// Event handling
// ===========================================================================
function onWorkspaceClick(e) {
  const target = e.target.closest("[data-action]");
  if (!target || target.disabled) return;
  const { action, value, stage } = target.dataset;

  switch (action) {
    case "stage":
      return setActive(stage);
    case "run-all":
      return runAll();
    case "cancel":
      return ctx.abort?.abort();
    case "new":
      if ((S.transcript || S.notes) && !confirm("Start a new lecture? The current transcript, notes and quiz will be cleared.")) return;
      resetSession();
      persist();
      return render();
    case "transcribe":
      return runStage("transcript");
    case "notes":
    case "quiz":
    case "translate":
      return runStage(action);
    case "play":
      return togglePlay();
    case "rate": {
      ctx.rate = RATES[(RATES.indexOf(ctx.rate) + 1) % RATES.length];
      if (ctx.audio) ctx.audio.playbackRate = ctx.rate;
      target.textContent = `${ctx.rate}×`;
      return;
    }
    case "seek": {
      if (!ctx.audio) return;
      ctx.audio.currentTime = Number(target.dataset.t) + 0.01;
      ctx.audio.play().catch(() => {});
      return;
    }
    case "jump":
      return jumpToSentence(Number(target.dataset.s), target.dataset.t ? Number(target.dataset.t) : undefined);
    case "theatre":
      return enterTheatre();
    case "clear-log": {
      S.log = [];
      const list = ctx.root.querySelector("#logList");
      if (list) list.innerHTML = '<li class="log-empty">Log cleared.</li>';
      return;
    }
    case "asr-lang":
      S.sourceLang = value;
      persist();
      return renderPanel(true);
    case "transcript-view":
      S.transcriptView = value;
      return renderPanel(true);
    case "paste-inline":
      S.transcriptView = "edit";
      S.active = "transcript";
      updateRegions();
      return renderPanel(true);
    case "cancel-edit":
      S.transcriptView = "segments";
      return renderPanel(true);
    case "save-transcript":
      return saveTranscript();
    case "copy-transcript":
      return copyText(S.transcript).then(() => toast("Transcript copied", { kind: "success" }));
    case "dl-transcript":
      return downloadFile(`${safeName(S.source.title || S.source.name)}_transcript.txt`, S.transcript);
    case "notes-view":
      S.notesView = value;
      persist();
      return renderPanel(true);
    case "copy-notes":
      return copyText(S.notes).then(() => toast("Markdown copied", { kind: "success" }));
    case "dl-notes":
      return downloadFile(`${safeName(S.source.title || S.source.name)}_notes.md`, S.notes, "text/markdown");
    case "print-notes":
      return printNotes();
    case "toc": {
      e.preventDefault();
      document.getElementById(target.dataset.id)?.scrollIntoView({ behavior: "smooth", block: "start" });
      return;
    }
    case "q-count":
      S.numQuestions = Math.min(10, Math.max(3, S.numQuestions + Number(value)));
      persist();
      return renderPanel(true);
    case "quiz-mode":
      S.quizMode = value;
      persist();
      return renderPanel(true);
    case "dl-quiz":
      return downloadFile(`${safeName(S.source.title || S.source.name)}_quiz.json`, JSON.stringify(S.quiz, null, 2), "application/json");
    case "target":
      S.target = value;
      persist();
      return updateRegions();
    case "tr-tab":
      S.translateTab = value;
      persist();
      return renderPanel(true);
    case "tr-view":
      S.translateView = value;
      persist();
      return renderPanel(true);
    case "dl-translation": {
      const tr = S.translations[S.target];
      const base = `${safeName(S.source.title || S.source.name)}_${S.target.toLowerCase()}`;
      if (S.translateTab === "quiz") return downloadFile(`${base}_quiz.json`, JSON.stringify(tr.quiz, null, 2), "application/json");
      return downloadFile(`${base}_notes.md`, tr.notes, "text/markdown");
    }
  }
}

function saveTranscript() {
  const text = ctx.root.querySelector("#transcriptEditor")?.value.trim() || "";
  if (!text) {
    toast("The transcript is empty.", { kind: "error" });
    return;
  }
  if (text !== S.transcript) {
    const prev = S.transcriptMeta || {};
    const mode = S.transcript ? prev.mode : "manual";
    const language = prev.language || detectLanguage(text) || S.sourceLang;
    setTranscript(text, { ...prev, mode, engine: prev.engine || "Typed by you", language, edited: Boolean(S.transcript) });
    S.segments = retimeSegments(S.segments, text);
    S.stages.transcript.status = "idle";
    S.stages.transcript.error = null;
    log("transcript", `Saved · ${plural(wordCount(text), "word")}`, "ok");
  }
  S.transcriptView = "segments";
  persist();
  updateRegions();
  renderPanel(true);
}

function printNotes() {
  if (S.notesView !== "rendered") {
    S.notesView = "rendered";
    renderPanel(true);
  }
  document.body.classList.add("print-notes");
  const done = () => document.body.classList.remove("print-notes");
  window.addEventListener("afterprint", done, { once: true });
  window.print();
  setTimeout(done, 1000);
}

// ===========================================================================
// Palette (Ctrl+K)
// ===========================================================================
function highlightTokens(sentence, indices) {
  const set = new Set(indices);
  let html = "";
  let cursor = 0;
  sentence.tokens.forEach((t, i) => {
    if (!set.has(i)) return;
    html += esc(sentence.text.slice(cursor, t.start)) + `<mark>${esc(t.text)}</mark>`;
    cursor = t.end;
  });
  return html + esc(sentence.text.slice(cursor));
}

/** Run something in the Studio view, switching to it first if needed. */
function inStudio(fn) {
  if (ctx.root) return fn();
  ctx.pendingJump = fn;
  location.hash = "#/studio";
}

function searchGroups(query) {
  if (!query || !S.transcript) return [];
  const queryScripts = new Set(tokenize(query).map((t) => t.script));
  const crossScript = (sentence, tokens) => tokens.some((i) => !queryScripts.has(sentence.tokens[i].script));
  const groups = [];

  const moments = lecture()
    .search(query, { limit: 6 })
    .map((hit) => ({
      html: `<span class="lang-text">${highlightTokens(hit.sentence, hit.tokens)}</span>`,
      time: hit.sentence.start,
      timeLabel: hit.sentence.start != null ? fmtTime(hit.sentence.start) : "¶",
      badge: crossScript(hit.sentence, hit.tokens) ? "by sound" : "",
      run: () => inStudio(() => jumpToSentence(hit.sentence.id)),
    }));
  groups.push({ title: "In the lecture", items: moments });

  const concepts = conceptGraph().concepts.filter((c) => {
    const q = query.toLowerCase();
    return c.label.toLowerCase().includes(q) || c.surface?.toLowerCase().includes(q);
  });
  groups.push({
    title: "Concepts",
    items: concepts.slice(0, 3).map((c) => ({
      html: `${esc(c.label)}${c.surface ? ` <span class="muted lang-text">${esc(c.surface)}</span>` : ""}`,
      icon: "layers",
      badge: plural(c.hits.length, "mention"),
      run: () =>
        inStudio(() => {
          ctx.concept = c.id;
          setActive("concepts");
          renderPanel(true);
        }),
    })),
  });

  if (S.notes) {
    const sections = noteSections();
    const hits = sectionIndex(sections).search(query, { limit: 4 });
    groups.push({
      title: "In your notes",
      items: hits.map((hit) => {
        const section = sections[hit.sentence.segment];
        return {
          html: `<span class="lang-text">${highlightTokens(hit.sentence, hit.tokens)}</span>${
            hit.sentence.text === section.heading ? "" : ` <span class="muted">— ${esc(section.heading)}</span>`
          }`,
          time: section.sentence?.start ?? null,
          timeLabel: section.sentence?.start != null ? fmtTime(section.sentence.start) : "",
          icon: "notes",
          run: () =>
            inStudio(() => {
              if (section.sentence?.start != null) jumpToSentence(section.sentence.id);
              setActive("notes");
            }),
        };
      }),
    });
  }

  if (S.quiz.length) {
    const index = new LectureIndex({
      segments: S.quiz.map((q, i) => ({ start: i, end: i + 1, text: `${q.question}\n${q.explanation}` })),
    });
    groups.push({
      title: "In the quiz",
      items: [...new Set(index.search(query, { limit: 6 }).map((hit) => hit.sentence.segment))].slice(0, 3).map((i) => ({
        html: esc(S.quiz[i].question),
        icon: "quiz",
        run: () => inStudio(() => setActive("quiz")),
      })),
    });
  }

  for (const [lang, tr] of Object.entries(S.translations)) {
    if (!tr.notes) continue;
    const index = new LectureIndex({ transcript: tr.notes.replace(/[*#_`]/g, "") });
    const hits = index.search(query, { limit: 2 });
    if (!hits.length) continue;
    groups.push({
      title: `In ${lang}`,
      items: hits.map((hit) => ({
        html: `<span class="lang-text">${highlightTokens(hit.sentence, hit.tokens)}</span>`,
        icon: "translate",
        badge: crossScript(hit.sentence, hit.tokens) ? "by sound" : "",
        run: () =>
          inStudio(() => {
            S.target = lang;
            S.translateTab = "notes";
            setActive("translate");
            renderPanel(true);
          }),
      })),
    });
  }
  return groups;
}

function commandList(shell) {
  const has = Boolean(S.source);
  const go = (stage) => () => inStudio(() => setActive(stage));
  return [
    has && { label: "Run pipeline", icon: "bolt", run: () => inStudio(runAll) },
    has && S.source.kind !== "text" && { label: "Enter Theatre", icon: "sparkles", run: () => inStudio(enterTheatre) },
    has && { label: "Go to Transcript", icon: "mic", run: go("transcript") },
    has && { label: "Go to Notes", icon: "notes", run: go("notes") },
    has && { label: "Go to Quiz", icon: "quiz", run: go("quiz") },
    has && { label: "Go to Translate", icon: "translate", run: go("translate") },
    has && { label: "Go to Concepts", icon: "layers", run: go("concepts") },
    { label: "Try the sample lecture", icon: "play", run: () => inStudio(loadSample) },
    { label: "Paste a transcript", icon: "clipboard", run: openPasteDialog },
    { label: "Open Benchmark", icon: "chart", run: () => (location.hash = "#/benchmark") },
    { label: "Engine status", icon: "cpu", run: shell.openEngines },
    { label: "Toggle light / dark theme", icon: "sun", run: shell.toggleTheme },
    { label: "Keyboard shortcuts", icon: "keyboard", run: shell.openKeys },
  ].filter(Boolean);
}

export function paletteProvider(shell) {
  const ready = Boolean(S.transcript);
  return {
    placeholder: ready ? "Search the lecture in any language — gradient, लर्निंग रेट…" : "Search commands…",
    note: ready ? "Finds words across scripts by how they sound" : "",
    hints: ready ? conceptGraph().concepts.slice(0, 4).map((c) => c.label) : [],
    groups(query) {
      const q = query.toLowerCase();
      const commands = commandList(shell).filter((c) => !q || c.label.toLowerCase().includes(q));
      return [
        ...searchGroups(query),
        {
          title: query ? "Commands" : "Jump to",
          items: commands.slice(0, query ? 4 : 12).map((c) => ({ html: esc(c.label), icon: c.icon, run: c.run })),
        },
      ];
    },
  };
}
