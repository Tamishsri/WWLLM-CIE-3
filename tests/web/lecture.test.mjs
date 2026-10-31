// node --test "tests/web/*.test.mjs"
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";

import { LectureIndex, retimeSegments } from "../../web/assets/js/lecture.js";

const sample = JSON.parse(readFileSync(new URL("./fixtures/gradient_descent.json", import.meta.url), "utf8"));

let t = 0.6;
const segments = sample.segments.map((text, i) => {
  const seg = { start: t, end: t + sample.layout[i], text };
  t = seg.end + 1.4;
  return seg;
});
const index = new LectureIndex({ segments });

test("sentences get monotonic interpolated timestamps", () => {
  assert.ok(index.sentences.length >= 12);
  for (let i = 1; i < index.sentences.length; i++) {
    assert.ok(index.sentences[i].start >= index.sentences[i - 1].start);
  }
  assert.equal(index.sentenceAt(5).segment, 0);
  assert.equal(index.sentenceAt(45).segment, 2);
  assert.equal(index.sentenceAt(18.6), null); // the pause between chunks
});

test("an English query finds the Hindi moment", () => {
  const [hit] = index.search("learning rate");
  assert.equal(hit.sentence.segment, 2);
  assert.ok(hit.sentence.text.includes("लर्निंग रेट"));
});

test("a Hindi query works too", () => {
  const [hit] = index.search("स्टोकेस्टिक");
  assert.equal(hit.sentence.segment, 3);
});

test("a short word doesn't match a longer word that merely starts the same", () => {
  const plain = new LectureIndex({ transcript: "We minimise the loss here. A mini batch is small. Batches help." });
  const hits = plain.search("mini");
  assert.equal(hits.length, 1);
  assert.equal(hits[0].sentence.id, 1);
  assert.equal(plain.search("batch").length, 2); // plural still matches
});

test("nonsense finds nothing", () => {
  assert.equal(index.search("zebra crossing").length, 0);
});

test("occurrences locate a phrase across scripts with a timestamp", () => {
  const hits = index.occurrences("mini-batch gradient descent");
  assert.equal(hits.length, 1);
  assert.equal(hits[0].sentence.segment, 3);
  assert.ok(hits[0].time > hits[0].sentence.start);
  assert.ok(hits[0].surface.includes("ग्रेडिएंट"));
});

test("note sections align to the right chunks, in order", () => {
  const sections = sample.notes.split(/^## /m).slice(1, 5);
  const placed = index.alignOrdered(sections);
  assert.deepEqual(placed.map((s) => s?.segment), [0, 1, 2, 3]);
});

test("editing a word keeps every chunk's timing", () => {
  const segs = [
    { start: 0, end: 5, text: "one two three" },
    { start: 6, end: 9, text: "four five" },
    { start: 10, end: 12, text: "six seven" },
  ];
  const edited = retimeSegments(segs, "one two three four FIVE six seven");
  assert.deepEqual(edited.map((s) => s.text), ["one two three", "four FIVE", "six seven"]);
  assert.deepEqual(edited.map((s) => s.start), [0, 6, 10]);
});

test("added text joins the chunk where it was added", () => {
  const segs = [{ start: 0, end: 5, text: "a b" }, { start: 6, end: 9, text: "c d" }];
  assert.deepEqual(retimeSegments(segs, "a b c d e f").map((s) => s.text), ["a b", "c d e f"]);
  assert.deepEqual(retimeSegments(segs, "a x y b c d").map((s) => s.text), ["a x y b", "c d"]);
});

test("removed text leaves the other chunks alone", () => {
  const segs = [{ start: 0, end: 5, text: "a b" }, { start: 6, end: 9, text: "c d" }];
  assert.deepEqual(retimeSegments(segs, "a c d").map((s) => s.text), ["a", "c d"]);
});

test("a full rewrite still spreads across every chunk", () => {
  const segs = [{ start: 0, end: 5, text: "a b" }, { start: 6, end: 9, text: "c d" }];
  const out = retimeSegments(segs, "w x y z");
  assert.equal(out.map((s) => s.text).join(" "), "w x y z");
  assert.ok(out.every((s) => s.text));
});

test("retiming with no segments is a no-op", () => {
  assert.deepEqual(retimeSegments([], "text"), []);
});

test("an untimed transcript still indexes, with null times", () => {
  const plain = new LectureIndex({ transcript: "Raft elects a leader. The leader replicates the log." });
  assert.equal(plain.sentences.length, 2);
  assert.equal(plain.sentences[0].start, null);
  assert.equal(plain.search("log")[0].sentence.id, 1);
});
