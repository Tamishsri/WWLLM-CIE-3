// node --test "tests/web/*.test.mjs"
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { anchorNotes, anchorQuiz, parseSections } from "../../web/assets/js/anchors.js";
import { extractConcepts, layoutConcepts } from "../../web/assets/js/concepts.js";
import { LectureIndex } from "../../web/assets/js/lecture.js";
import { levelHeadings, render } from "../../web/assets/js/markdown.js";

const sample = JSON.parse(readFileSync(new URL("./fixtures/gradient_descent.json", import.meta.url), "utf8"));

let t = 0.6;
const segments = sample.segments.map((text, i) => {
  const seg = { start: t, end: t + sample.layout[i], text };
  t = seg.end + 1.4;
  return seg;
});
const index = new LectureIndex({ segments });

test("parseSections splits notes and spots the summary section", () => {
  const sections = parseSections(sample.notes);
  assert.equal(sections.length, 5);
  assert.equal(sections[1].heading, "The Mountain Analogy");
  assert.ok(sections[4].summary);
  assert.ok(!sections[0].summary);
});

test("note sections land on the chunk that covers them", () => {
  const sections = anchorNotes(index, sample.notes);
  assert.deepEqual(sections.map((s) => s.sentence?.segment ?? null), [0, 1, 2, 3, null]);
});

test("a section starts where its topic begins, not at its strongest line", () => {
  const sections = anchorNotes(index, sample.notes);
  const variants = sections.find((s) => s.heading === "Three Variants");
  const strongest = index.occurrences("mini-batch gradient descent")[0].sentence;
  assert.equal(variants.sentence.segment, 3);
  assert.ok(variants.sentence.id < strongest.id, variants.sentence.text);
});

test("quiz questions point at the exact sentence when it is said verbatim", () => {
  const sections = anchorNotes(index, sample.notes);
  const placed = anchorQuiz(index, sections, sample.quiz);
  assert.ok(placed[3].text.includes("स्टोकेस्टिक"), placed[3].text);
});

test("a translated term reaches the lecture through the notes bridge", () => {
  // "slope" is spoken as ढलान: no phonetic link, so only the notes can connect it.
  const sections = anchorNotes(index, sample.notes);
  const placed = anchorQuiz(index, sections, sample.quiz);
  assert.deepEqual(placed.map((s) => s?.segment), [0, 1, 2, 3, 3]);
});

test("quiz anchoring works without notes, directly by sound", () => {
  const placed = anchorQuiz(index, [], sample.quiz);
  assert.equal(placed[3]?.segment, 3); // "stochastic gradient descent" is said verbatim
});

test("extractive notes in the transcript's own language anchor exactly", () => {
  const notes = `# T\n\n## Part\n- ${index.sentences[9].text}\n`;
  const [section] = anchorNotes(index, notes);
  assert.equal(section.sentence.id, 9);
});

test("concepts carry English labels and the words actually spoken", () => {
  const { concepts, edges } = extractConcepts(index, { notes: sample.notes, duration: 78.6 });
  const byLabel = Object.fromEntries(concepts.map((c) => [c.label.toLowerCase(), c]));
  assert.ok(byLabel["learning rate"], concepts.map((c) => c.label).join(", "));
  assert.equal(byLabel["learning rate"].surface, "लर्निंग रेट");
  assert.ok(byLabel["learning rate"].hits.length >= 2);
  assert.ok(byLabel["momentum"].pos > byLabel["loss function"].pos, "later ideas sit further along the timeline");
  assert.ok(edges.length > 0);
});

test("longest phrases claim their words", () => {
  const { concepts } = extractConcepts(index, { notes: sample.notes, duration: 78.6 });
  const mini = concepts.find((c) => c.label.toLowerCase().startsWith("mini"));
  assert.ok(mini, "mini-batch concept exists");
  const claimed = new Set(mini.hits.map((h) => h.sentence.id));
  const plain = concepts.find((c) => c.label.toLowerCase() === "gradient descent");
  for (const h of plain?.hits ?? []) {
    if (claimed.has(h.sentence.id)) assert.ok(h.from >= mini.hits[0].to || h.to <= mini.hits[0].from);
  }
});

test("without notes, repeated words still become concepts", () => {
  const { concepts } = extractConcepts(index, { duration: 78.6 });
  assert.ok(concepts.length >= 3);
  assert.ok(concepts.some((c) => c.label.includes("ग्रेडिएंट")));
});

test("layout keeps pills inside the box without overlaps", () => {
  const { concepts, edges } = extractConcepts(index, { notes: sample.notes, duration: 78.6 });
  const nodes = layoutConcepts(concepts, edges, 900, 10, 400, (c) => c.label.length * 7);
  for (const n of nodes) {
    assert.ok(n.x - n.w / 2 >= 0 && n.x + n.w / 2 <= 900);
    assert.ok(n.y - n.h / 2 >= 9 && n.y + n.h / 2 <= 401);
  }
  let overlaps = 0;
  nodes.forEach((a, i) =>
    nodes.slice(i + 1).forEach((b) => {
      if (Math.abs(a.x - b.x) < (a.w + b.w) / 2 && Math.abs(a.y - b.y) < (a.h + b.h) / 2) overlaps++;
    })
  );
  assert.equal(overlaps, 0);
});

test("notes written with deeper headings still get sections and a contents list", () => {
  const notes = "### Title\n\n#### One\nBody\n\n#### Two\nBody";
  assert.deepEqual(parseSections(notes).map((s) => s.heading), ["One", "Two"]);
  assert.equal(render(notes).toc.length, 2);
});

test("heading levels: a title stays one level above its sections", () => {
  assert.equal(levelHeadings("# Title\n\n## One"), "# Title\n\n## One");
  assert.equal(levelHeadings("### One\n\n### Two"), "## One\n\n## Two");
  assert.equal(levelHeadings("no headings"), "no headings");
  assert.equal(levelHeadings(""), "");
});
