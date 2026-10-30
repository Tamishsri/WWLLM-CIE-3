// node --test "tests/web/*.test.mjs"
import assert from "node:assert/strict";
import { test } from "node:test";

import { detectLanguage, skeleton, skeletonSimilarity, tokenize, tokenSimilarity } from "../../web/assets/js/phonetic.js";

const same = (a, b) => assert.equal(skeleton(a), skeleton(b), `${a} (${skeleton(a)}) vs ${b} (${skeleton(b)})`);

test("English loanwords match Devanagari", () => {
  same("gradient", "ग्रेडिएंट");
  same("descent", "डिसेंट");
  same("learning", "लर्निंग");
  same("rate", "रेट");
  same("function", "फंक्शन");
  same("momentum", "मोमेंटम");
  same("hyperparameter", "हाइपरपैरामीटर");
  same("optimization", "ऑप्टिमाइज़ेशन");
  same("dataset", "डेटासेट");
  same("sample", "सैंपल");
  same("neural", "न्यूरल");
  same("network", "नेटवर्क");
  same("algorithm", "अल्गोरिदम");
});

test("a word-final c is hard, not soft", () => {
  assert.ok(skeleton("topic").endsWith("K"));
  assert.equal(tokenSimilarity(tokenize("stochastic")[0], tokenize("स्टोकेस्टिक")[0]), 0.75);
});

test("the same word matches across every Indic script", () => {
  const gradient = skeleton("gradient");
  for (const word of ["ग्रेडिएंट", "கிரேடியன்ட்", "ಗ್ರೇಡಿಯಂಟ್", "గ్రేడియంట్", "ഗ്രേഡിയന്റ്", "গ্রেডিয়েন্ট", "ગ્રેડિયન્ટ", "ਗ੍ਰੇਡੀਐਂਟ", "ଗ୍ରାଡିଏଣ୍ଟ"]) {
    assert.equal(skeleton(word), gradient, word);
  }
});

test("Tamil's unvoiced spelling of voiced sounds still matches", () => {
  same("data", "டேட்டா");
  same("model", "மாடல்");
});

test("short skeletons need an exact match and score lower", () => {
  assert.equal(skeletonSimilarity("LS", "LS"), 0.45);
  assert.equal(skeletonSimilarity("LS", "LT"), 0);
  assert.equal(skeletonSimilarity("KRTNT", "KRTNT"), 1);
  assert.equal(skeletonSimilarity("KRTNT", "KRTN"), 0.8);
});

test("everyday words don't collide with native vocabulary", () => {
  const sim = (a, b) => tokenSimilarity(tokenize(a)[0], tokenize(b)[0]);
  assert.equal(sim("imagine", "मशीन"), 0);
  assert.equal(sim("trying", "ट्रेनिंग"), 0);
  assert.equal(sim("mountain", "मोमेंटम"), 0);
  assert.equal(sim("gradient", "करते"), 0);
  assert.equal(sim("gradient", "ग्रेडिएंट"), 1);
});

test("two Indic scripts compare by sound, not by spelling", () => {
  const sim = (a, b) => tokenSimilarity(tokenize(a)[0], tokenize(b)[0]);
  assert.equal(sim("கிரேடியன்ட்", "ग्रेडिएंट"), 1);
});

test("tokenize keeps Indic vowel signs inside words", () => {
  const words = tokenize("मशीन लर்னिंग, deep-learning!").map((t) => t.text);
  assert.deepEqual(words[0], "मशीन");
  assert.ok(words.includes("deep"));
  assert.ok(words.includes("learning"));
});

test("detectLanguage reads the dominant script", () => {
  assert.equal(detectLanguage("यह एक वाक्य है"), "Hindi");
  assert.equal(detectLanguage("இது ஒரு வாக்கியம்"), "Tamil");
  assert.equal(detectLanguage("plain english text"), "English");
  assert.equal(detectLanguage("123"), null);
});
