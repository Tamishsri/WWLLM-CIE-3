// Lecture Rangoli: a generative fingerprint of a lecture.
//
// Every speech chunk becomes one ring of petals, shaped by that chunk's real
// loudness contour, with the mirrored symmetry of a floor rangoli. No two
// recordings look alike. In Theatre mode the rangoli draws itself as the
// lecture plays; the finished pattern can be exported as an image.

const COLORS = ["#ff9f2e", "#e8487f", "#8a8bff", "#22b8a6", "#ffd166", "#ff7a59", "#b98cff", "#4cc9f0"];
const PETALS = [8, 8, 16, 16, 24, 24, 32, 32];
const MAX_RINGS = 8;
const INNER = 0.1;
const OUTER = 0.97;
const POINTS_PER_PETAL = 18;

function hash(text) {
  let h = 2166136261;
  for (let i = 0; i < text.length; i++) h = Math.imul(h ^ text.charCodeAt(i), 16777619);
  return h >>> 0;
}

function seededEnvelope(seed, size) {
  let x = seed || 1;
  const out = [];
  for (let i = 0; i < size; i++) {
    x = Math.imul(x, 1664525) + 1013904223;
    out.push(((x >>> 16) & 255) / 255);
  }
  return out;
}

/** Resample peaks[from..to) to *size* points and smooth lightly. */
function envelope(peaks, from, to, size) {
  const raw = [];
  const span = Math.max(1, to - from);
  for (let i = 0; i < size; i++) {
    const a = from + Math.floor((i / size) * span);
    const b = Math.max(a + 1, from + Math.floor(((i + 1) / size) * span));
    let max = 0;
    for (let j = a; j < b && j < peaks.length; j++) if (peaks[j] > max) max = peaks[j];
    raw.push(max);
  }
  return raw.map((v, i) => (0.25 * (raw[i - 1] ?? v) + 0.5 * v + 0.25 * (raw[i + 1] ?? v)));
}

/** Group chunks so a long lecture still fits in MAX_RINGS rings. */
function ringSpans(chunks, duration) {
  if (!chunks?.length) {
    const n = 6;
    return Array.from({ length: n }, (_, i) => ({ start: (i / n) * duration, end: ((i + 1) / n) * duration }));
  }
  if (chunks.length <= MAX_RINGS) return chunks;
  const per = chunks.length / MAX_RINGS;
  return Array.from({ length: MAX_RINGS }, (_, i) => ({
    start: chunks[Math.floor(i * per)].start,
    end: chunks[Math.min(chunks.length - 1, Math.ceil((i + 1) * per) - 1)].end,
  }));
}

/**
 * @param {{peaks?: Float32Array, chunks?: Array, duration?: number, seed?: string}} input
 *   Without audio (a pasted transcript) the rings grow from a hash of the text.
 */
export function buildRangoli({ peaks, chunks, duration = 0, seed = "" }) {
  const hasAudio = Boolean(peaks?.length && duration);
  const spans = ringSpans(chunks, duration || 1);
  const band = (OUTER - INNER) / spans.length;
  const seedHash = hash(seed);

  const rings = spans.map((span, i) => {
    const env = hasAudio
      ? envelope(
          peaks,
          Math.floor((span.start / duration) * peaks.length),
          Math.floor((span.end / duration) * peaks.length),
          24
        )
      : seededEnvelope(seedHash + i * 7919, 24);
    return {
      env,
      petals: PETALS[Math.min(i, PETALS.length - 1)],
      inner: INNER + i * band,
      outer: INNER + (i + 1) * band,
      color: COLORS[(seedHash + i) % COLORS.length],
      start: span.start,
      end: span.end,
    };
  });
  return { rings, spin: (seedHash % 360) * (Math.PI / 180) };
}

/** Polar outline of one ring as [x, y] points on a unit circle. */
function ringPoints(ring, spin, boost = 0) {
  const pts = [];
  const { petals, inner, outer, env } = ring;
  const width = outer - inner;
  for (let p = 0; p < petals; p++) {
    for (let j = 0; j < POINTS_PER_PETAL; j++) {
      const u = j / POINTS_PER_PETAL;
      const mirrored = Math.abs(2 * u - 1); // 0 at the tip, 1 at the base: symmetric petals
      const e = env[Math.min(env.length - 1, Math.floor((1 - mirrored) * env.length))];
      const shape = Math.sin(Math.PI * u) ** 1.25;
      const r = inner + width * Math.min(1, 0.08 + 0.92 * shape * (0.4 + 0.6 * e) + boost * shape * 0.25);
      const a = spin + ((p + u) / petals) * Math.PI * 2;
      pts.push([r * Math.cos(a), r * Math.sin(a)]);
    }
  }
  return pts;
}

function tipPoints(ring, spin) {
  const out = [];
  for (let p = 0; p < ring.petals; p++) {
    const a = spin + ((p + 0.5) / ring.petals) * Math.PI * 2;
    const r = ring.outer + 0.012;
    out.push([r * Math.cos(a), r * Math.sin(a)]);
  }
  return out;
}

/** Render as a self-contained SVG string (used for the deck emblem). */
export function rangoliSVG(model, { dots = true } = {}) {
  const path = (pts) => `M${pts.map(([x, y]) => `${(x * 100).toFixed(1)} ${(y * 100).toFixed(1)}`).join("L")}Z`;
  const rings = model.rings
    .map((ring) => {
      const tips = dots
        ? tipPoints(ring, model.spin)
            .map(([x, y]) => `<circle cx="${(x * 100).toFixed(1)}" cy="${(y * 100).toFixed(1)}" r="0.9" fill="${ring.color}"/>`)
            .join("")
        : "";
      return `<path d="${path(ringPoints(ring, model.spin))}" fill="${ring.color}" fill-opacity=".2" stroke="${ring.color}" stroke-width="1"/>${tips}`;
    })
    .join("");
  return `<svg class="rangoli" viewBox="-104 -104 208 208" role="img" aria-label="Lecture rangoli: a pattern generated from this recording">${rings}<circle r="5" fill="${model.rings[0]?.color || "#ff9f2e"}"/></svg>`;
}

/**
 * Draw on a 2D canvas.
 * @param {number} time        playback position; rings ahead of it stay faint
 * @param {number} energy      0–1 live loudness, makes the current ring pulse
 * @param {number} spinOffset  slow rotation (radians)
 */
export function drawRangoli(ctx, model, cx, cy, radius, { time = Infinity, energy = 0, spinOffset = 0 } = {}) {
  const spin = model.spin + spinOffset;
  ctx.save();
  ctx.translate(cx, cy);
  ctx.scale(radius, radius);
  ctx.lineJoin = "round";

  for (const ring of model.rings) {
    const done = time >= ring.end;
    const active = time >= ring.start && !done;
    const progress = done ? 1 : active ? (time - ring.start) / (ring.end - ring.start || 1) : 0;
    const pts = ringPoints(ring, spin, active ? energy : 0);

    // Ghost of what is still to come
    if (!done) {
      ctx.beginPath();
      pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
      ctx.closePath();
      ctx.globalAlpha = 0.14;
      ctx.lineWidth = 0.002;
      ctx.strokeStyle = ring.color;
      ctx.stroke();
    }
    if (progress <= 0) continue;

    // The ring draws itself around the circle as its chunk plays
    const shown = Math.max(2, Math.floor(pts.length * progress));
    ctx.beginPath();
    for (let i = 0; i < shown; i++) (i ? ctx.lineTo : ctx.moveTo).call(ctx, pts[i][0], pts[i][1]);
    if (done) ctx.closePath();
    ctx.globalAlpha = 0.22;
    ctx.fillStyle = ring.color;
    if (done) ctx.fill();
    ctx.globalAlpha = 0.95;
    ctx.lineWidth = active ? 0.004 + energy * 0.004 : 0.0028;
    ctx.strokeStyle = ring.color;
    if (active) {
      ctx.shadowColor = ring.color;
      ctx.shadowBlur = 18 * (radius / 300);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    if (done) {
      ctx.globalAlpha = 0.9;
      ctx.fillStyle = ring.color;
      for (const [x, y] of tipPoints(ring, spin)) {
        ctx.beginPath();
        ctx.arc(x, y, 0.006, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  ctx.globalAlpha = 1;
  ctx.fillStyle = model.rings[0]?.color || "#ff9f2e";
  ctx.beginPath();
  ctx.arc(0, 0, 0.05 + energy * 0.02, 0, Math.PI * 2);
  ctx.fill();
  ctx.restore();
}

/** A finished rangoli on a dark card, as a PNG blob, for saving and sharing. */
export function rangoliPNG(model, { title = "", subtitle = "" } = {}) {
  const size = 1200;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size + 150;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#0b0c10";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  drawRangoli(ctx, model, size / 2, size / 2 + 20, size * 0.44);
  ctx.textAlign = "center";
  ctx.fillStyle = "#f3f1ea";
  ctx.font = '44px "Instrument Serif", Georgia, serif';
  ctx.fillText(title.slice(0, 60), size / 2, size + 56);
  ctx.fillStyle = "#85827a";
  ctx.font = '22px "Geist", system-ui, sans-serif';
  ctx.fillText(subtitle, size / 2, size + 100);
  return new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
}
