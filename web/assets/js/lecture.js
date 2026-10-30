// Searchable index of a lecture: sentences with estimated timestamps, ranked
// BM25-style across scripts. Everything that ties an output back to a moment in
// the recording (search, notes/quiz anchors, concepts) is built on this.

import { tokenize, tokenSimilarity } from "./phonetic.js";

const SENTENCE_RE = /[^.!?।॥\n]+[.!?।॥]*/gu;
const K1 = 1.2;
const B = 0.6;

function splitSentences(text) {
  const out = [];
  for (const m of String(text || "").matchAll(SENTENCE_RE)) {
    const raw = m[0];
    const trimmed = raw.trim();
    if (trimmed.length > 1) out.push({ text: trimmed, offset: m.index + raw.indexOf(trimmed) });
  }
  return out;
}

/**
 * Keep a transcript's timing after an edit. Words before and after the changed
 * region stay in their chunk; the changed region spreads across the chunks it
 * replaced. A one-word fix therefore moves nothing else.
 */
export function retimeSegments(segments, text) {
  const words = (s) => s.split(/\s+/).filter(Boolean);
  const before = segments.flatMap((seg, i) => words(seg.text).map((word) => ({ word, seg: i })));
  const after = words(text);
  if (!before.length) return segments;

  let head = 0;
  while (head < before.length && head < after.length && before[head].word === after[head]) head++;
  let tail = 0;
  while (
    tail < before.length - head &&
    tail < after.length - head &&
    before[before.length - 1 - tail].word === after[after.length - 1 - tail]
  ) {
    tail++;
  }

  const changedOld = before.length - head - tail;
  const changedNew = after.length - head - tail;
  const owner = (i) => {
    if (i < head) return before[i].seg;
    if (i >= after.length - tail) return before[before.length - (after.length - i)].seg;
    const at = head + Math.floor(((i - head) / changedNew) * changedOld);
    return before[Math.min(at, before.length - 1)].seg;
  };

  const texts = segments.map(() => []);
  after.forEach((word, i) => texts[owner(i)].push(word));
  return segments.map((seg, i) => ({ ...seg, text: texts[i].join(" ") }));
}

export class LectureIndex {
  /**
   * @param {{segments?: Array<{start:number,end:number,text:string}>, transcript?: string}} source
   *   With segments, sentence times are interpolated inside each segment by
   *   character position (speech rate is roughly constant within a chunk).
   */
  constructor({ segments = [], transcript = "" } = {}) {
    this.sentences = [];
    this.timed = segments.length > 0;
    const blocks = this.timed
      ? segments.map((s, i) => ({ ...s, segment: i }))
      : [{ text: transcript, start: null, end: null, segment: null }];

    for (const block of blocks) {
      const length = block.text.length || 1;
      for (const part of splitSentences(block.text)) {
        const at = (offset) => (block.start == null ? null : block.start + (offset / length) * (block.end - block.start));
        this.sentences.push({
          id: this.sentences.length,
          text: part.text,
          segment: block.segment,
          start: at(part.offset),
          end: at(part.offset + part.text.length),
          tokens: tokenize(part.text),
        });
      }
    }

    // Vocabulary: unique token forms → where they occur
    this.vocab = new Map();
    let totalLength = 0;
    for (const s of this.sentences) {
      totalLength += s.tokens.length;
      s.tokens.forEach((t, i) => {
        const key = `${t.script}:${t.norm}`;
        if (!this.vocab.has(key)) this.vocab.set(key, { token: t, postings: [] });
        this.vocab.get(key).postings.push([s.id, i]);
      });
    }
    this.avgLength = this.sentences.length ? totalLength / this.sentences.length : 1;
  }

  /** Vocabulary entries that match a query token, with IDF-weighted scores. */
  matchesFor(queryToken) {
    const N = this.sentences.length || 1;
    const hits = [];
    for (const entry of this.vocab.values()) {
      const sim = tokenSimilarity(queryToken, entry.token);
      if (sim <= 0) continue;
      const df = new Set(entry.postings.map(([sid]) => sid)).size;
      hits.push({ entry, sim, idf: Math.log(1 + (N - df + 0.5) / (df + 0.5)) });
    }
    return hits;
  }

  /** Rank sentences for free text in any script. */
  search(text, { limit = 8, minScore = 0 } = {}) {
    const all = tokenize(text);
    const content = all.filter((t) => !t.stop);
    const query = content.length ? content : all;
    if (!query.length) return [];

    const scored = new Map();
    for (const q of query) {
      const best = new Map(); // one query word counts once per sentence
      for (const { entry, sim, idf } of this.matchesFor(q)) {
        for (const [sid, tokenIndex] of entry.postings) {
          const value = sim * idf;
          const prev = best.get(sid);
          if (!prev || value > prev.value) best.set(sid, { value, tokenIndex, tokens: [tokenIndex] });
          else if (value === prev.value) prev.tokens.push(tokenIndex);
        }
      }
      for (const [sid, { value, tokens }] of best) {
        if (!scored.has(sid)) scored.set(sid, { score: 0, tokens: new Set(), matched: 0 });
        const row = scored.get(sid);
        const len = this.sentences[sid].tokens.length;
        row.score += (value * (K1 + 1)) / (1 + K1 * (1 - B + (B * len) / this.avgLength));
        row.matched += 1;
        tokens.forEach((i) => row.tokens.add(i));
      }
    }

    return [...scored.entries()]
      .map(([sid, row]) => ({
        sentence: this.sentences[sid],
        // Reward covering more of the query, and consecutive matched words.
        score: row.score * (0.5 + (0.5 * row.matched) / query.length) * adjacencyBoost(row.tokens),
        matched: row.matched,
        tokens: [...row.tokens].sort((a, b) => a - b),
      }))
      .filter((r) => r.score > minScore)
      .sort((a, b) => b.score - a.score || a.sentence.id - b.sentence.id)
      .slice(0, limit);
  }

  /** Best sentence for a passage (a quiz answer, a note), or null if unsure. */
  locate(text, { minMatched = 2 } = {}) {
    const [best] = this.search(text, { limit: 1 });
    if (!best) return null;
    const strong = best.tokens.some((i) => best.sentence.tokens[i].skel.length >= 4);
    return best.matched >= minMatched || (strong && best.matched >= 1) ? best : null;
  }

  /**
   * Align ordered passages (note sections) to sentences, keeping lecture order:
   * notes follow the lecture, so a term repeated throughout ("gradient") can't
   * pull a section to the wrong moment. Dynamic programme over passages ×
   * sentences; returns one sentence (or null) per passage.
   */
  alignOrdered(passages) {
    const n = passages.length;
    const m = this.sentences.length;
    if (!n || !m) return passages.map(() => null);

    const scores = passages.map((p) => {
      const row = new Float64Array(m);
      for (const hit of this.search(p, { limit: m })) row[hit.sentence.id] = hit.score;
      return row;
    });

    // best[i][j]: best total with passage i placed at or before sentence j
    const best = Array.from({ length: n }, () => new Float64Array(m));
    const pick = Array.from({ length: n }, () => new Int32Array(m));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < m; j++) {
        const take = scores[i][j] + (i ? best[i - 1][j] : 0);
        const skip = j ? best[i][j - 1] : -Infinity;
        if (take >= skip) {
          best[i][j] = take;
          pick[i][j] = j;
        } else {
          best[i][j] = skip;
          pick[i][j] = pick[i][j - 1];
        }
      }
    }

    const placed = new Array(n).fill(null);
    let j = m - 1;
    for (let i = n - 1; i >= 0; i--) {
      const at = pick[i][j];
      if (scores[i][at] > 0) placed[i] = this.sentences[at];
      j = at;
    }
    return placed;
  }

  /** Every place a phrase occurs (words in order, phonetically across scripts). */
  occurrences(phrase) {
    const words = tokenize(phrase).filter((t) => t.skel.length >= 2 || !t.stop);
    if (!words.length) return [];
    const out = [];
    for (const s of this.sentences) {
      const toks = s.tokens;
      for (let i = 0; i + words.length <= toks.length; i++) {
        let total = 0;
        let ok = true;
        for (let w = 0; w < words.length; w++) {
          const sim = tokenSimilarity(words[w], toks[i + w]);
          if (sim < 0.45) {
            ok = false;
            break;
          }
          total += sim;
        }
        // Weak two-consonant words need a strong partner ("loss function").
        if (ok && (words.length === 1 || total / words.length >= 0.7)) {
          out.push({
            sentence: s,
            from: i,
            to: i + words.length,
            score: total / words.length,
            surface: s.text.slice(toks[i].start, toks[i + words.length - 1].end),
            time: this.timeOf(s, toks[i]),
          });
        }
      }
    }
    return out;
  }

  /** Estimated moment a token is spoken. */
  timeOf(sentence, token) {
    if (sentence.start == null) return null;
    const ratio = token.start / (sentence.text.length || 1);
    return sentence.start + ratio * (sentence.end - sentence.start);
  }

  /** Sentence being spoken at time t. */
  sentenceAt(t) {
    return this.sentences.find((s) => s.start != null && t >= s.start && t < s.end) || null;
  }
}

function adjacencyBoost(tokens) {
  const sorted = [...tokens].sort((a, b) => a - b);
  let runs = 0;
  for (let i = 1; i < sorted.length; i++) if (sorted[i] === sorted[i - 1] + 1) runs++;
  return 1 + 0.25 * Math.min(runs, 4);
}
