// Glyph River: the hero's background. Letters from ten scripts ride a sound
// wave across the page, scatter from the pointer, and rush when a file is
// dragged over the page. It is the product in one image: speech becoming text
// in every language.

const SCRIPTS = [
  "कखगघचजटडतदनपबमयरलवशसह",
  "கஙசஞடணதநபமயரலவழளறன",
  "ಕಖಗಘಚಜಟಡತದನಪಬಮಯರಲವಶಸಹ",
  "కఖగఘచజటడతదనపబమయరలవశసహ",
  "കഖഗഘചജടഡതദനപബമയരലവശസഹ",
  "কখগঘচজটডতদনপবমযরলশসহ",
  "કખગઘચજટડતદનપબમયરલવશસહ",
  "ਕਖਗਘਚਜਟਡਤਦਨਪਬਮਯਰਲਵਸਹ",
  "କଖଗଘଚଜଟଡତଦନପବମଯରଲଶସହ",
  "abcdefghijklmnopqrstuvwxyz",
];

const FONT_FAMILIES = [
  "Noto Sans Devanagari", "Noto Sans Tamil", "Noto Sans Kannada", "Noto Sans Telugu", "Noto Sans Malayalam",
  "Noto Sans Bengali", "Noto Sans Gujarati", "Noto Sans Gurmukhi", "Noto Sans Oriya", "Geist",
];

const RIBBONS = [
  { base: 0.5, amp: 0.22, freq: 1.7, speed: 1, size: 19, token: "--accent-text", alpha: 0.75 },
  { base: 0.56, amp: 0.3, freq: 1.2, speed: 0.72, size: 15, token: "--indigo", alpha: 0.5 },
  { base: 0.44, amp: 0.14, freq: 2.4, speed: 0.5, size: 12, token: "--text-3", alpha: 0.42 },
];
const PER_RIBBON = 54;

export class GlyphRiver {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.rush = 0;
    this.rushTarget = 0;
    this.pointer = { x: -9999, y: -9999 };
    this.time = 0;
    this.last = performance.now();
    this.reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

    this.glyphs = [];
    RIBBONS.forEach((ribbon, r) => {
      for (let i = 0; i < PER_RIBBON; i++) {
        const script = SCRIPTS[(i + r * 3) % SCRIPTS.length];
        const chars = [...script];
        this.glyphs.push({
          ribbon,
          u: i / PER_RIBBON + Math.random() * 0.012,
          char: chars[Math.floor(Math.random() * chars.length)],
          jitter: Math.random() * Math.PI * 2,
          dx: 0,
          dy: 0,
        });
      }
    });

    this.onResize = () => this.resize();
    this.onMove = (e) => {
      const rect = canvas.getBoundingClientRect();
      this.pointer.x = e.clientX - rect.left;
      this.pointer.y = e.clientY - rect.top;
    };
    addEventListener("resize", this.onResize);
    addEventListener("pointermove", this.onMove, { passive: true });
    this.resize();
    // Canvas text doesn't trigger web-font loading the way the DOM does.
    SCRIPTS.forEach((chars, i) => document.fonts?.load(`16px "${FONT_FAMILIES[i]}"`, chars));
    if (this.reduced) this.draw(0);
    else this.raf = requestAnimationFrame((t) => this.frame(t));
  }

  destroy() {
    cancelAnimationFrame(this.raf);
    removeEventListener("resize", this.onResize);
    removeEventListener("pointermove", this.onMove);
  }

  /** Speed the river up while a file hovers over the dropzone. */
  setRush(on) {
    this.rushTarget = on ? 1 : 0;
  }

  resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const rect = this.canvas.getBoundingClientRect();
    this.w = rect.width;
    this.h = rect.height;
    this.canvas.width = rect.width * dpr;
    this.canvas.height = rect.height * dpr;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (this.reduced) this.draw(0);
  }

  frame(now) {
    const dt = Math.min(0.05, (now - this.last) / 1000);
    this.last = now;
    if (!document.hidden) {
      this.rush += (this.rushTarget - this.rush) * 0.06;
      this.time += dt * (1 + this.rush * 2.5);
      this.draw(dt);
    }
    this.raf = requestAnimationFrame((t) => this.frame(t));
  }

  draw(dt) {
    const { ctx, w, h, time, pointer } = this;
    if (!w || !h) return;
    ctx.clearRect(0, 0, w, h);
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const css = getComputedStyle(document.documentElement);

    for (const ribbon of RIBBONS) {
      ribbon.color = css.getPropertyValue(ribbon.token).trim();
    }

    for (const g of this.glyphs) {
      const r = g.ribbon;
      g.u = (g.u + dt * 0.022 * r.speed * (1 + this.rush * 4)) % 1;
      const x = g.u * (w + 80) - 40;
      const phase = g.u * Math.PI * 2 * r.freq;
      // A speech-like wave: a carrier inside a slower loudness envelope
      const envelope = 0.45 + 0.55 * Math.sin(phase * 0.5 + time * 0.35) ** 2;
      const wave = Math.sin(phase + time * 0.9) * envelope + 0.3 * Math.sin(phase * 2.7 - time * 0.6);
      const y = h * (r.base + r.amp * wave * 0.72) + Math.sin(time + g.jitter) * 3;

      // Scatter away from the pointer, then drift back
      const px = x + g.dx - pointer.x;
      const py = y + g.dy - pointer.y;
      const dist = Math.hypot(px, py);
      if (dist < 130 && dist > 0.1) {
        const force = (1 - dist / 130) * 2.6;
        g.dx += (px / dist) * force;
        g.dy += (py / dist) * force;
      }
      g.dx *= 0.94;
      g.dy *= 0.94;

      const edge = Math.min(1, Math.min(g.u, 1 - g.u) * 9);
      ctx.globalAlpha = r.alpha * edge * (0.55 + 0.45 * envelope);
      ctx.fillStyle = r.color;
      ctx.font = `${r.size}px "Geist", "Noto Sans Devanagari", "Noto Sans Tamil", "Noto Sans Kannada", "Noto Sans Telugu", "Noto Sans Malayalam", "Noto Sans Bengali", "Noto Sans Gujarati", "Noto Sans Gurmukhi", "Noto Sans Oriya", sans-serif`;
      ctx.fillText(g.char, x + g.dx, y + g.dy);
    }
    ctx.globalAlpha = 1;
  }
}
