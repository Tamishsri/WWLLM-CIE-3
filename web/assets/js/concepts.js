// Concept extraction and the "concept river" layout: the ideas of a lecture
// arranged along its timeline, linked by where they are discussed together.
//
// Candidates come from the notes' bold terms (good English labels) and from
// words the lecturer repeats. Each is then located in the transcript phonetically,
// so "Learning rate" in the notes resolves to every लर्निंग रेट in the speech.

import { tokenize } from "./phonetic.js";

const MAX_CONCEPTS = 22;

function cleanTerm(raw) {
  return raw
    .replace(/\([^)]*\)/g, " ")
    .replace(/[^\p{L}\p{M}\p{N}\s-]/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

const capitalise = (s) => (/^[a-z]/.test(s) ? s[0].toUpperCase() + s.slice(1) : s);

function notesCandidates(notes) {
  const seen = new Set();
  const out = [];
  for (const m of String(notes || "").matchAll(/\*\*([^*\n]+?)\*\*/g)) {
    const term = cleanTerm(m[1]);
    const key = term.toLowerCase();
    if (term && !seen.has(key) && tokenize(term).length <= 5) {
      seen.add(key);
      out.push(term);
    }
  }
  return out;
}

/** Words and word pairs the lecturer repeats. */
function transcriptCandidates(index) {
  const counts = new Map();
  const bump = (key, surface) => {
    if (!counts.has(key)) counts.set(key, { surface, count: 0 });
    counts.get(key).count += 1;
  };
  for (const s of index.sentences) {
    let prev = null;
    for (const t of s.tokens) {
      const content = !t.stop && !t.common && t.skel.length >= 3;
      if (content) {
        bump(t.norm, t.text);
        if (prev) bump(`${prev.norm} ${t.norm}`, `${prev.text} ${t.text}`);
      }
      prev = content ? t : null;
    }
  }
  return [...counts.values()]
    .filter((c) => c.count >= 2)
    .sort((a, b) => b.count - a.count)
    .slice(0, 30)
    .map((c) => c.surface);
}

/** Occurrences of a term, shrinking it until something matches. */
function resolve(index, term) {
  const words = term.split(/\s+/).filter((w) => !tokenize(w).every((t) => t.stop));
  for (let size = words.length; size >= 1; size--) {
    for (let start = 0; start + size <= words.length; start++) {
      const phrase = words.slice(start, start + size).join(" ");
      if (size === 1 && tokenize(phrase).every((t) => t.skel.length < 3 || t.stop)) continue;
      const hits = index.occurrences(phrase);
      if (hits.length) return { phrase, hits };
    }
  }
  return null;
}

function mostCommon(values) {
  const counts = new Map();
  values.forEach((v) => counts.set(v, (counts.get(v) || 0) + 1));
  return [...counts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0];
}

/**
 * @returns {{concepts: Array, edges: Array}} concepts sorted by importance, each
 *   with `pos` (0–1 position in the lecture), `hits` and a `surface` form as
 *   actually spoken; edges carry a co-occurrence `weight`.
 */
export function extractConcepts(index, { notes = "", duration = 0 } = {}) {
  const candidates = [
    ...notesCandidates(notes).map((term) => ({ term, fromNotes: true })),
    ...transcriptCandidates(index).map((term) => ({ term, fromNotes: false })),
  ];

  // Longest phrases claim their words first, so "Batch gradient descent" isn't
  // also counted as "Gradient descent" and "Gradient".
  const resolved = candidates
    .map((c) => ({ ...c, ...resolve(index, c.term) }))
    .filter((c) => c.hits)
    .sort((a, b) => tokenize(b.phrase).length - tokenize(a.phrase).length || b.fromNotes - a.fromNotes);

  const claimed = new Set();
  const seen = new Set();
  const spoken = new Set(); // words inside concepts the notes already name
  const concepts = [];
  for (const c of resolved) {
    if (!c.fromNotes && tokenize(c.phrase).every((t) => spoken.has(t.skel))) continue;
    const free = c.hits.filter((h) => {
      for (let i = h.from; i < h.to; i++) if (claimed.has(`${h.sentence.id}:${i}`)) return false;
      return true;
    });
    if (!free.length) continue;
    const key = tokenize(c.phrase).map((t) => t.skel).join(" ");
    if (seen.has(key)) continue;
    seen.add(key);
    tokenize(c.phrase).forEach((t) => spoken.add(t.skel));
    free.forEach((h) => {
      for (let i = h.from; i < h.to; i++) claimed.add(`${h.sentence.id}:${i}`);
    });
    const surface = mostCommon(free.map((h) => h.surface));
    concepts.push({
      label: capitalise(c.phrase),
      surface: surface && surface.toLowerCase() !== c.phrase.toLowerCase() ? surface : null,
      hits: free,
      fromNotes: c.fromNotes,
      score: free.length * (c.fromNotes ? 1.6 : 1) + (tokenize(c.phrase).length > 1 ? 0.5 : 0),
    });
  }

  const top = concepts.sort((a, b) => b.score - a.score).slice(0, MAX_CONCEPTS);
  const span = duration || index.sentences.at(-1)?.end || index.sentences.length || 1;
  top.forEach((c, i) => {
    c.id = i;
    const mean = c.hits.reduce((sum, h) => sum + (h.time ?? h.sentence.id), 0) / c.hits.length;
    c.pos = Math.min(1, mean / span);
  });

  const edges = [];
  for (let a = 0; a < top.length; a++) {
    const inA = new Set(top[a].hits.map((h) => h.sentence.id));
    const segA = new Set(top[a].hits.map((h) => h.sentence.segment));
    for (let b = a + 1; b < top.length; b++) {
      let weight = 0;
      for (const h of top[b].hits) {
        if (inA.has(h.sentence.id)) weight += 1;
        else if (h.sentence.segment != null && segA.has(h.sentence.segment)) weight += 0.2;
      }
      if (weight >= 0.4) edges.push({ a, b, weight });
    }
  }
  return { concepts: top, edges };
}

/**
 * Place pills along the timeline: x follows when a concept is discussed, y is
 * resolved by pushing overlapping pills apart (rectangles, because labels are wide).
 */
export function layoutConcepts(concepts, edges, width, top, bottom, measure) {
  const pad = 10;
  const nodes = concepts.map((c, i) => {
    const w = measure(c) + 28;
    const h = c.surface ? 44 : 30;
    const x = pad + w / 2 + c.pos * (width - 2 * pad - w);
    return { c, w, h, x, tx: x, y: (top + bottom) / 2 + ((i * 2654435761) % 97) - 48 };
  });
  const mid = (top + bottom) / 2;

  for (let iter = 0; iter < 420; iter++) {
    for (const e of edges) {
      const a = nodes[e.a];
      const b = nodes[e.b];
      const pull = 0.004 * Math.min(e.weight, 3);
      const dy = b.y - a.y;
      a.y += dy * pull;
      b.y -= dy * pull;
    }
    for (const n of nodes) {
      n.x += (n.tx - n.x) * 0.06;
      n.y += (mid - n.y) * 0.006;
    }
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i];
        const b = nodes[j];
        const ox = (a.w + b.w) / 2 + 10 - Math.abs(a.x - b.x);
        const oy = (a.h + b.h) / 2 + 12 - Math.abs(a.y - b.y);
        if (ox <= 0 || oy <= 0) continue;
        // Separate along y (x carries meaning), nudging x only slightly.
        const dir = a.y === b.y ? (i % 2 ? 1 : -1) : Math.sign(b.y - a.y);
        a.y -= (dir * oy) / 2;
        b.y += (dir * oy) / 2;
        const dx = Math.sign(b.x - a.x) || 1;
        a.x -= dx * Math.min(ox, 4) * 0.1;
        b.x += dx * Math.min(ox, 4) * 0.1;
      }
    }
    for (const n of nodes) {
      n.x = Math.max(pad + n.w / 2, Math.min(width - pad - n.w / 2, n.x));
      n.y = Math.max(top + n.h / 2, Math.min(bottom - n.h / 2, n.y));
    }
  }
  return nodes;
}
