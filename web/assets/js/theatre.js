// Theatre mode: listen full-screen while the lecture's rangoli draws itself,
// one ring per speech chunk, pulsing with the live audio. Captions follow word
// by word (timing is interpolated, so it is an estimate, not forced alignment).

import { downloadFile, esc, fmtTime, icon, safeName, toast } from "./util.js";
import { drawRangoli, rangoliPNG } from "./rangoli.js";

const graphs = new WeakMap();
let active = null;

/** One analyser per audio element: a media source can only be created once. */
function analyserFor(audio) {
  if (graphs.has(audio)) return graphs.get(audio);
  try {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    const ac = new Ctx();
    const source = ac.createMediaElementSource(audio);
    const analyser = ac.createAnalyser();
    analyser.fftSize = 512;
    analyser.smoothingTimeConstant = 0.72;
    source.connect(analyser);
    analyser.connect(ac.destination);
    const graph = { ac, analyser, data: new Uint8Array(analyser.fftSize) };
    graphs.set(audio, graph);
    return graph;
  } catch (err) {
    console.warn("Audio analyser unavailable", err);
    graphs.set(audio, null);
    return null;
  }
}

function loudness(graph) {
  if (!graph) return 0;
  graph.analyser.getByteTimeDomainData(graph.data);
  let sum = 0;
  for (const v of graph.data) sum += ((v - 128) / 128) ** 2;
  return Math.min(1, Math.sqrt(sum / graph.data.length) * 3.2);
}

function captionHTML(sentence) {
  let html = "";
  let cursor = 0;
  for (const t of sentence.tokens) {
    html += esc(sentence.text.slice(cursor, t.start));
    html += `<span class="t-w" data-at="${t.start}">${esc(t.text)}</span>`;
    cursor = t.end;
  }
  return html + esc(sentence.text.slice(cursor));
}

export const isTheatreOpen = () => Boolean(active);
export const closeTheatre = () => active?.close();

/**
 * @param {object} opts
 * @param {HTMLAudioElement} opts.audio
 * @param {import("./lecture.js").LectureIndex} opts.index
 * @param {object} opts.model      rangoli model
 * @param {Array}  opts.concepts   extracted concepts (their hits drive the chips)
 * @param {string} opts.title
 * @param {number} opts.duration
 */
export function openTheatre({ audio, index, model, concepts = [], title, duration }) {
  if (active) return;
  const root = document.createElement("div");
  root.className = "theatre";
  root.setAttribute("role", "dialog");
  root.setAttribute("aria-label", `Theatre mode: ${title}`);
  root.innerHTML = `
    <canvas class="t-canvas"></canvas>
    <header class="t-top">
      <div class="t-title"><span class="eyebrow"><span class="spark"></span>Now playing</span><h2>${esc(title)}</h2></div>
      <div class="t-actions">
        <button class="t-btn" data-t="save" title="Save this lecture's rangoli as an image">${icon("download", "icon-sm")}Save rangoli</button>
        <button class="t-btn t-icon" data-t="close" aria-label="Close theatre">${icon("x", "icon-sm")}</button>
      </div>
    </header>
    <div class="t-caption" aria-live="off">
      <div class="t-concepts" id="tConcepts"></div>
      <p class="t-prev" id="tPrev"></p>
      <p class="t-now" id="tcNow"></p>
      <p class="t-next" id="tNext"></p>
      <p class="t-hint" id="tHint">Word timing is estimated from each speech chunk.</p>
    </div>
    <footer class="t-bar">
      <button class="t-play" data-t="play" aria-label="Play or pause">${icon(audio.paused ? "play" : "pause")}</button>
      <span class="t-time" id="tTime">0:00</span>
      <div class="t-scrub" id="tScrub" role="slider" aria-label="Seek" tabindex="0" aria-valuemin="0" aria-valuemax="${Math.round(duration)}">
        ${model.rings
          .map((r) => `<i style="left:${(r.start / duration) * 100}%;width:${((r.end - r.start) / duration) * 100}%;background:${r.color}"></i>`)
          .join("")}
        <b id="tHead"></b>
      </div>
      <span class="t-time">${fmtTime(duration)}</span>
    </footer>`;
  document.body.append(root);
  document.body.classList.add("theatre-open");

  const canvas = root.querySelector(".t-canvas");
  const ctx = canvas.getContext("2d");
  const $ = (id) => root.querySelector(`#${id}`);
  const graph = analyserFor(audio);
  graph?.ac.resume();

  let raf = 0;
  let energy = 0;
  let spin = 0;
  let last = performance.now();
  let sentenceId = -2;
  let tokens = [];

  const resize = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = innerWidth * dpr;
    canvas.height = innerHeight * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  };
  resize();
  addEventListener("resize", resize);

  const setCaption = (t) => {
    const s = index.sentenceAt(t);
    const id = s ? s.id : -1;
    if (id !== sentenceId) {
      sentenceId = id;
      const list = index.sentences;
      if (s) {
        $("tPrev").textContent = list[s.id - 1]?.text || "";
        $("tNext").textContent = list[s.id + 1]?.text || "";
        $("tcNow").innerHTML = captionHTML(s);
        tokens = [...$("tcNow").querySelectorAll(".t-w")];
      }
      const here = s ? concepts.filter((c) => c.hits.some((h) => h.sentence.id === s.id)) : [];
      $("tConcepts").innerHTML = here.map((c) => `<span class="t-chip">${esc(c.label)}</span>`).join("");
    }
    if (s) {
      const ratio = (t - s.start) / (s.end - s.start || 1);
      const at = ratio * s.text.length;
      tokens.forEach((w) => w.classList.toggle("on", Number(w.dataset.at) <= at));
    }
  };

  const frame = (now) => {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    const t = audio.currentTime || 0;
    const target = audio.paused ? 0 : loudness(graph);
    energy += (target - energy) * 0.2;
    if (!audio.paused) spin += dt * 0.025;

    const w = innerWidth;
    const h = innerHeight;
    ctx.clearRect(0, 0, w, h);
    const radius = Math.min(w * 0.46, h * 0.36);
    drawRangoli(ctx, model, w / 2, h * 0.4, radius, { time: t, energy, spinOffset: spin });

    $("tTime").textContent = fmtTime(t);
    $("tHead").style.left = `${Math.min(1, t / duration) * 100}%`;
    setCaption(t);
    raf = requestAnimationFrame(frame);
  };
  raf = requestAnimationFrame(frame);

  const syncPlay = () => (root.querySelector('[data-t="play"]').innerHTML = icon(audio.paused ? "play" : "pause"));
  const togglePlay = () => (audio.paused ? audio.play().catch(() => {}) : audio.pause());
  audio.addEventListener("play", syncPlay);
  audio.addEventListener("pause", syncPlay);

  const seekTo = (clientX) => {
    const rect = $("tScrub").getBoundingClientRect();
    audio.currentTime = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width)) * duration;
  };
  let scrubbing = false;
  $("tScrub").addEventListener("pointerdown", (e) => {
    scrubbing = true;
    $("tScrub").setPointerCapture(e.pointerId);
    seekTo(e.clientX);
  });
  $("tScrub").addEventListener("pointermove", (e) => scrubbing && seekTo(e.clientX));
  $("tScrub").addEventListener("pointerup", () => (scrubbing = false));

  const save = async () => {
    const blob = await rangoliPNG(model, { title, subtitle: "A lecture rangoli · WWLLM Lecture Studio" });
    if (!blob) return toast("Couldn't render the image.", { kind: "error" });
    downloadFile(`${safeName(title)}_rangoli.png`, blob);
    toast("Rangoli saved", { kind: "success" });
  };

  root.addEventListener("click", (e) => {
    const action = e.target.closest("[data-t]")?.dataset.t;
    if (action === "close") close();
    else if (action === "play") togglePlay();
    else if (action === "save") save();
  });

  const onKey = (e) => {
    if (e.key === "Escape") close();
    else if (e.key === " ") togglePlay();
    else if (e.key === "ArrowRight") audio.currentTime = Math.min(duration, audio.currentTime + 5);
    else if (e.key === "ArrowLeft") audio.currentTime = Math.max(0, audio.currentTime - 5);
    else return;
    e.preventDefault();
    e.stopPropagation();
  };
  addEventListener("keydown", onKey, true);

  function close() {
    cancelAnimationFrame(raf);
    removeEventListener("resize", resize);
    removeEventListener("keydown", onKey, true);
    audio.removeEventListener("play", syncPlay);
    audio.removeEventListener("pause", syncPlay);
    document.body.classList.remove("theatre-open");
    root.classList.add("closing");
    setTimeout(() => root.remove(), 260);
    active = null;
  }

  active = { close };
  if (audio.paused) audio.play().catch(() => {});
}
