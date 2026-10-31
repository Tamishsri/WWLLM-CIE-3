// Ties generated content back to moments in the recording.
//
// Notes sections are aligned to the transcript in lecture order. Those sections
// then act as a bridge for everything else: notes are usually English while the
// lecture may be Hindi, so a word like "slope" (spoken as ढलान) has no phonetic
// link to the audio, but it does appear in a note section whose position in the
// lecture is known.

import { LectureIndex } from "./lecture.js";
import { levelHeadings } from "./markdown.js";

const SUMMARY_RE = /takeaway|key points|key terms|summary|overview|निष्कर्ष|सारांश/i;

/** Split Markdown notes into `##` sections (the h1 title is skipped). */
export function parseSections(notes) {
  const sections = [];
  let current = null;
  for (const line of levelHeadings(notes).split(/\r?\n/)) {
    const heading = line.match(/^\s{0,3}##\s+(.*?)\s*#*\s*$/);
    if (heading) {
      current = { heading: heading[1].replace(/[*_`]/g, ""), lines: [] };
      sections.push(current);
    } else if (current) {
      const text = line.replace(/^\s*([-*+]|\d+[.)])\s+/, "").replace(/[*_`]/g, "").trim();
      if (text) current.lines.push(text);
    }
  }
  return sections.map((s) => ({
    heading: s.heading,
    body: s.lines.join("\n"),
    summary: SUMMARY_RE.test(s.heading),
    sentence: null,
  }));
}

/**
 * Place each notes section at a moment in the lecture. Summary-style sections
 * restate the whole lecture, so they get no single moment.
 */
export function anchorNotes(index, notes) {
  const sections = parseSections(notes);
  const ordered = sections.filter((s) => !s.summary);
  const passages = ordered.map((s) => `${s.heading}\n${s.body}`);
  const placed = index.alignOrdered(passages);

  // The alignment finds a section's strongest sentence; a listener wants to
  // start where the topic begins, so step back to its earliest solid mention.
  let previous = -1;
  ordered.forEach((section, i) => {
    const best = placed[i];
    if (!best) return;
    const hits = index.search(passages[i], { limit: index.sentences.length });
    const top = hits.find((h) => h.sentence.id === best.id)?.score ?? 0;
    const start = hits
      .filter((h) => h.sentence.id > previous && h.sentence.id <= best.id && h.score >= top * 0.5)
      .reduce((earliest, h) => (h.sentence.id < earliest.id ? h.sentence : earliest), best);
    section.sentence = start;
    previous = best.id;
  });
  return sections;
}

/** Index of the note sections themselves, used as the bridge for lookups. */
function sectionIndex(sections) {
  return new LectureIndex({
    segments: sections.map((s, i) => ({ start: i, end: i + 1, text: `${s.heading}\n${s.body}` })),
  });
}

/** Find the moment a passage refers to, going through the notes when needed. */
export function locateVia(index, sections, bridge, text) {
  const direct = index.search(text, { limit: 40 });
  const viaNotes = bridge?.search(text, { limit: 1 })[0];
  const section = viaNotes && viaNotes.matched >= 2 ? sections[viaNotes.sentence.segment] : null;

  if (section?.sentence) {
    // Prefer a direct hit inside the part of the lecture this section covers.
    const anchored = sections.filter((s) => s.sentence);
    const next = anchored[anchored.indexOf(section) + 1];
    const from = section.sentence.id;
    const to = next ? next.sentence.id : Infinity;
    const inside = direct.find((h) => h.sentence.id >= from && h.sentence.id < to);
    return inside ? inside.sentence : section.sentence;
  }
  const best = direct[0];
  if (!best) return null;
  const strong = best.tokens.some((i) => best.sentence.tokens[i].skel.length >= 4);
  return best.matched >= 2 || strong ? best.sentence : null;
}

/** One moment per quiz question (or null when nothing supports it). */
export function anchorQuiz(index, sections, quiz) {
  const bridge = sections.length ? sectionIndex(sections) : null;
  return quiz.map((q) =>
    locateVia(index, sections, bridge, `${q.question} ${q.options?.[q.answer_index] ?? q.correct_answer ?? ""} ${q.explanation ?? ""}`)
  );
}

export { sectionIndex };
