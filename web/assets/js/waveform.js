// Canvas waveform with VAD chunk overlays, scrubbing, hover time and a live
// playhead. Peaks are decoded in the browser at 8 kHz to keep memory low.

import { clamp, fmtTime } from "./util.js";

const PEAK_BUCKETS = 1600;
const MAX_DECODE_BYTES = 200 * 1024 * 1024;
const peakCache = new Map();

async function decodePeaks(arrayBuffer) {
  const Ctx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  const ctx = new Ctx(1, 1, 8000);
  const audio = await ctx.decodeAudioData(arrayBuffer);
  const data = audio.getChannelData(0);
  const size = Math.max(1, Math.floor(data.length / PEAK_BUCKETS));
  const peaks = new Float32Array(Math.ceil(data.length / size));
  for (let b = 0; b < peaks.length; b++) {
    let max = 0;
    const end = Math.min(data.length, (b + 1) * size);
    for (let i = b * size; i < end; i++) {
      const v = Math.abs(data[i]);
      if (v > max) max = v;
    }
    peaks[b] = max;
  }
  // Normalise to the 99th percentile so one click doesn't flatten the rest.
  const sorted = Array.from(peaks).sort((a, b) => a - b);
  const ref = sorted[Math.floor(sorted.length * 0.99)] || 1;
  for (let i = 0; i < peaks.length; i++) peaks[i] = Math.min(1, peaks[i] / ref);
  return { peaks, duration: audio.duration };
}

export class Waveform {
  /**
   * @param {HTMLElement} root  the .wave container
   * @param {HTMLAudioElement} audio
   * @param {{onTime?: (t:number)=>void}} opts
   */
  constructor(root, audio, opts = {}) {
    this.root = root;
    this.audio = audio;
    this.onTime = opts.onTime;
    this.peaks = null;
    this.duration = opts.duration || 0;
    this.chunks = [];
    this.states = new Map();
    this.active = -1;

    root.innerHTML = `
      <canvas></canvas>
      <div class="wave-chunks"></div>
      <div class="wave-marks"></div>
      <div class="wave-playhead" hidden></div>
      <div class="wave-cursor"><b>0:00</b></div>
      <div class="wave-empty" hidden></div>`;
    this.canvas = root.querySelector("canvas");
    this.ctx = this.canvas.getContext("2d");
    this.chunkLayer = root.querySelector(".wave-chunks");
    this.markLayer = root.querySelector(".wave-marks");
    this.marks = [];
    this.playhead = root.querySelector(".wave-playhead");
    this.cursor = root.querySelector(".wave-cursor");
    this.empty = root.querySelector(".wave-empty");

    this.resizeObserver = new ResizeObserver(() => this.draw());
    this.resizeObserver.observe(root);
    this.themeObserver = new MutationObserver(() => this.draw());
    this.themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });

    this.bindPointer();
    this.bindAudio();
  }

  destroy() {
    this.resizeObserver.disconnect();
    this.themeObserver.disconnect();
    cancelAnimationFrame(this.raf);
    this.unbindAudio?.();
  }

  // ── Loading ─────────────────────────────────────────────────────────
  async load(key, source, size = 0) {
    if (peakCache.has(key)) return this.setPeaks(peakCache.get(key));
    if (size > MAX_DECODE_BYTES) return this.setMessage("Waveform preview skipped for very large files");
    this.setMessage("Rendering waveform…");
    try {
      let buffer;
      if (source instanceof Blob) buffer = await source.arrayBuffer();
      else {
        const res = await fetch(source);
        if (!res.ok) return this.setMessage("Audio not found on the server — upload the recording again");
        buffer = await res.arrayBuffer();
      }
      const decoded = await decodePeaks(buffer);
      peakCache.set(key, decoded);
      this.setPeaks(decoded);
    } catch (err) {
      console.warn("Waveform decode failed", err);
      this.setMessage("This browser can't preview this format — playback may still work");
    }
  }

  setPeaks({ peaks, duration }) {
    this.peaks = peaks;
    if (!this.duration && duration) this.duration = duration;
    this.empty.hidden = true;
    this.layoutChunks();
    this.paintMarks();
    this.draw();
  }

  setMessage(text) {
    this.empty.textContent = text;
    this.empty.hidden = false;
    this.draw();
  }

  setDuration(seconds) {
    if (seconds && Math.abs(seconds - this.duration) > 0.05) {
      this.duration = seconds;
      this.layoutChunks();
      this.draw();
    }
  }

  // ── Chunks ──────────────────────────────────────────────────────────
  setChunks(chunks = []) {
    this.chunks = chunks;
    this.states.clear();
    this.layoutChunks();
  }

  /** Highlight moments on the wave (e.g. every mention of a concept). */
  setMarks(times = []) {
    this.marks = times;
    this.paintMarks();
  }

  paintMarks() {
    const d = this.duration;
    this.markLayer.innerHTML = d
      ? this.marks.map((t) => `<i class="wave-mark" style="left:${(t / d) * 100}%"></i>`).join("")
      : "";
  }

  setChunkState(index, state) {
    if (index < 0 || index >= this.chunks.length) return;
    if (state) this.states.set(index, state);
    else this.states.delete(index);
    this.paintChunkStates();
  }

  clearChunkStates() {
    this.states.clear();
    this.paintChunkStates();
  }

  setActive(index) {
    if (index === this.active) return;
    this.active = index;
    this.paintChunkStates();
  }

  layoutChunks() {
    const d = this.duration;
    if (!d || !this.chunks.length) {
      this.chunkLayer.innerHTML = "";
      return;
    }
    this.chunkLayer.innerHTML = this.chunks
      .map((c, i) => {
        const left = (c.start / d) * 100;
        const width = Math.max(0.3, ((c.end - c.start) / d) * 100);
        return `<div class="wave-chunk" data-i="${i}" style="left:${left}%;width:${width}%"><span>${String(i + 1).padStart(2, "0")}</span></div>`;
      })
      .join("");
    this.paintChunkStates();
  }

  paintChunkStates() {
    this.chunkLayer.querySelectorAll(".wave-chunk").forEach((el) => {
      const i = Number(el.dataset.i);
      const state = this.states.get(i);
      el.classList.toggle("scanning", state === "scanning");
      el.classList.toggle("done", state === "done");
      el.classList.toggle("active", i === this.active && !state);
    });
  }

  // ── Drawing ─────────────────────────────────────────────────────────
  draw() {
    const { canvas, ctx } = this;
    const dpr = window.devicePixelRatio || 1;
    const w = this.root.clientWidth;
    const h = this.root.clientHeight;
    if (!w || !h) return;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const css = getComputedStyle(this.root);
    const idle = css.getPropertyValue("--wave-idle").trim() || "#444";
    const played = css.getPropertyValue("--wave-played").trim() || "#f90";
    const progress = this.duration ? clamp(this.audio.currentTime / this.duration, 0, 1) : 0;

    if (!this.peaks) {
      ctx.fillStyle = idle;
      ctx.fillRect(0, h / 2 - 0.5, w, 1);
      return;
    }

    const barW = 2;
    const gap = 1.6;
    const bars = Math.max(1, Math.floor(w / (barW + gap)));
    const per = this.peaks.length / bars;
    const mid = h / 2;
    const maxH = h * 0.42;
    const cut = progress * bars;

    for (let b = 0; b < bars; b++) {
      let peak = 0;
      const end = Math.min(this.peaks.length, Math.ceil((b + 1) * per));
      for (let i = Math.floor(b * per); i < end; i++) if (this.peaks[i] > peak) peak = this.peaks[i];
      const bh = Math.max(1, peak * maxH);
      ctx.fillStyle = b < cut ? played : idle;
      const x = b * (barW + gap);
      ctx.beginPath();
      if (ctx.roundRect) ctx.roundRect(x, mid - bh, barW, bh * 2, 1);
      else ctx.rect(x, mid - bh, barW, bh * 2);
      ctx.fill();
    }
  }

  updatePlayhead() {
    const d = this.duration;
    const t = this.audio.currentTime || 0;
    const show = d > 0 && (t > 0 || !this.audio.paused);
    this.playhead.hidden = !show;
    if (show) this.playhead.style.left = `${clamp(t / d, 0, 1) * 100}%`;
    const idx = this.chunks.findIndex((c) => t >= c.start && t < c.end);
    this.setActive(show ? idx : -1);
    this.onTime?.(t);
  }

  // ── Interaction ─────────────────────────────────────────────────────
  timeAt(clientX) {
    const rect = this.root.getBoundingClientRect();
    return clamp((clientX - rect.left) / rect.width, 0, 1) * (this.duration || 0);
  }

  bindPointer() {
    const root = this.root;
    let dragging = false;
    const seek = (e) => {
      if (!this.duration || !this.audio.src) return;
      this.audio.currentTime = this.timeAt(e.clientX);
      this.updatePlayhead();
      this.draw();
    };
    root.addEventListener("pointermove", (e) => {
      const rect = root.getBoundingClientRect();
      const x = clamp(e.clientX - rect.left, 0, rect.width);
      this.cursor.style.left = `${x}px`;
      const label = this.cursor.querySelector("b");
      label.textContent = fmtTime(this.timeAt(e.clientX));
      label.style.transform = x > rect.width - 60 ? "translateX(calc(-100% - 12px))" : "";
      if (dragging) seek(e);
    });
    root.addEventListener("pointerdown", (e) => {
      dragging = true;
      root.setPointerCapture(e.pointerId);
      seek(e);
    });
    const stop = () => (dragging = false);
    root.addEventListener("pointerup", stop);
    root.addEventListener("pointercancel", stop);
  }

  bindAudio() {
    const a = this.audio;
    const loop = () => {
      this.updatePlayhead();
      this.draw();
      if (!a.paused) this.raf = requestAnimationFrame(loop);
    };
    const onPlay = () => {
      cancelAnimationFrame(this.raf);
      this.raf = requestAnimationFrame(loop);
    };
    const onMeta = () => this.setDuration(a.duration);
    const onSeek = () => {
      this.updatePlayhead();
      this.draw();
    };
    a.addEventListener("play", onPlay);
    a.addEventListener("pause", onSeek);
    a.addEventListener("seeked", onSeek);
    a.addEventListener("loadedmetadata", onMeta);
    this.unbindAudio = () => {
      a.removeEventListener("play", onPlay);
      a.removeEventListener("pause", onSeek);
      a.removeEventListener("seeked", onSeek);
      a.removeEventListener("loadedmetadata", onMeta);
    };
    if (a.readyState >= 1) onMeta();
  }
}
