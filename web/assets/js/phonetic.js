// Cross-script phonetic matching for code-mixed Indian lectures.
//
// A lecturer says "ग्रेडिएंट डिसेंट"; a student types "gradient descent". Both
// reduce to the same consonant skeleton, KRTNT STNT, so they match with no
// model and no dictionary. This works because:
//   • The nine major Indic Unicode blocks share one layout (inherited from
//     ISCII): offset 0x15 is "ka" in Devanagari, Bengali, Gurmukhi, Gujarati,
//     Odia, Tamil, Telugu, Kannada and Malayalam alike. One table covers all.
//   • Consonants carry loanwords across scripts; vowels don't. So vowels are
//     dropped, and sounds that scripts merge are folded into one class (Tamil
//     writes k/g, t/d and p/b with a single letter, so voicing must not count).
//
// Skeleton alphabet:  K  S  T  P  N  M  L  R

const OFFSET_CLASS = {};
const assign = (cls, ...offsets) => offsets.forEach((o) => (OFFSET_CLASS[o] = cls));
assign("N", 0x01, 0x02, 0x19, 0x1e, 0x23, 0x28, 0x29); // nasals + anusvara / candrabindu
assign("K", 0x15, 0x16, 0x17, 0x18, 0x58, 0x59, 0x5a); // ka kha ga gha (+ qa etc.)
assign("S", 0x1a, 0x1b, 0x1c, 0x1d, 0x36, 0x37, 0x38, 0x5b); // ca cha ja jha sha ssa sa za
assign("T", 0x1f, 0x20, 0x21, 0x22, 0x24, 0x25, 0x26, 0x27); // retroflex + dental stops
assign("P", 0x2a, 0x2b, 0x2c, 0x2d, 0x35, 0x5e); // pa pha ba bha va fa
assign("M", 0x2e);
assign("R", 0x30, 0x31, 0x5c, 0x5d);
assign("L", 0x32, 0x33, 0x34);
assign("N", 0x7a, 0x7b); // Malayalam chillu ṇ, n
assign("R", 0x7c);
assign("L", 0x7d, 0x7e);
assign("K", 0x7f);

const INDIC_START = 0x0900;
const INDIC_END = 0x0d7f;
const GURMUKHI_TIPPI = 0x0a70;
const BENGALI_KHANDA_TA = 0x09ce;

const isIndicCode = (cp) => cp >= INDIC_START && cp <= INDIC_END;

/** Script of a token: "latin", or the Indic block ("indic0" Devanagari … "indic8" Malayalam). */
export function scriptOf(word) {
  for (const ch of word) {
    const cp = ch.codePointAt(0);
    if (isIndicCode(cp)) return `indic${(cp - INDIC_START) >> 7}`;
  }
  return "latin";
}

function indicSkeleton(word) {
  // Malayalam clusters: nna+virama+rra is "nt" and rra+virama+rra is "tt".
  const text = word.replace(/\u0D28\u0D4D\u0D31/g, "\u0D28\u0D4D\u0D1F").replace(/\u0D31\u0D4D\u0D31/g, "\u0D1F");
  let out = "";
  for (const ch of text) {
    const cp = ch.codePointAt(0);
    if (!isIndicCode(cp)) continue;
    if (cp === GURMUKHI_TIPPI) out += "N";
    else if (cp === BENGALI_KHANDA_TA) out += "T";
    else out += OFFSET_CLASS[(cp - INDIC_START) & 0x7f] ?? "";
  }
  return out;
}

function latinSkeleton(word) {
  const w = word
    .toLowerCase()
    .replace(/[^a-z]/g, "")
    .replace(/^(kn|gn|pn)/, "n")
    .replace(/^wr/, "r")
    .replace(/^ps/, "s")
    .replace(/tch/g, "ch")
    .replace(/dge/g, "j")
    .replace(/sch/g, "sk")
    .replace(/[tsc]ion/g, "shn")
    .replace(/ture/g, "chr")
    .replace(/ph/g, "f")
    .replace(/ck/g, "k")
    .replace(/qu/g, "k")
    .replace(/x/g, "ks")
    .replace(/gh/g, "")
    .replace(/mb$/, "m");

  let out = "";
  for (let i = 0; i < w.length; i++) {
    const c = w[i];
    const next = w[i + 1] || "";
    if ("aeiouyh".includes(c)) continue;
    if (c === "c") out += next === "h" || (next && "eiy".includes(next)) ? "S" : "K";
    else if (c === "g") out += next && "eiy".includes(next) ? "S" : "K";
    else if ("bpfvw".includes(c)) out += "P";
    else if ("dt".includes(c)) out += "T";
    else if ("kq".includes(c)) out += "K";
    else if ("jsz".includes(c)) out += "S";
    else if (c === "l") out += "L";
    else if (c === "m") out += "M";
    else if (c === "n") out += "N";
    else if (c === "r") out += "R";
  }
  return out;
}

/** Reduce one word in any supported script to its consonant skeleton. */
export function skeleton(word) {
  const raw = scriptOf(word) === "latin" ? latinSkeleton(word) : indicSkeleton(word);
  return raw.replace(/M(?=P)/g, "N").replace(/(.)\1+/g, "$1");
}

export function editDistance(a, b) {
  if (a === b) return 0;
  if (!a.length) return b.length;
  if (!b.length) return a.length;
  let prev = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    for (let j = 1; j <= b.length; j++) {
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    }
    prev = cur;
  }
  return prev[b.length];
}

/**
 * Similarity (0–1) of two skeletons. Deliberately strict: consonant skeletons
 * collide easily, and a false link between a note and a moment is worse than
 * none. Two-consonant words ("loss", "rate") only count for much inside a
 * phrase; fuzzy matches must be long and agree on the first consonant.
 */
export function skeletonSimilarity(a, b) {
  if (!a || !b) return 0;
  if (a === b) return a.length >= 3 ? 1 : 0.45;
  const min = Math.min(a.length, b.length);
  const max = Math.max(a.length, b.length);
  if (min < 4 || a[0] !== b[0]) return 0;
  // Inflection: learnings/लर्निंग, models/मॉडल
  if ((a.startsWith(b) || b.startsWith(a)) && max - min === 1) return 0.8;
  if (min >= 5 && editDistance(a, b) <= 1) return 0.75;
  if (min >= 8 && editDistance(a, b) <= 2) return 0.65;
  return 0;
}

// ── Tokens ────────────────────────────────────────────────────────────────
const WORD_RE = /[\p{L}\p{M}\p{N}]+/gu;

const STOPWORDS = new Set(
  `a about above after again all also am an and any are as at be because been before being
  below between both but by can could did do does doing down during each few for from further
  had has have having he her here him his how i if in into is it its itself just let like me
  more most my no nor not now of off on once only or other our out over own same she should so
  some such than that the their them then there these they this those through to too under
  until up very was we were what when where which while who whom why will with would you your
  okay today class lecture going get got one two three use used using called means way thing
  things lot really next see look
  है हैं था थे थी हो होता होती होते होगा हुआ हुए के का की को में से पर और या भी तो ही एक यह वह ये वो
  इस उस इन उन इसे उसे जो कि जब तक तब कर करता करते करती करना किया किए लिए साथ बहुत सबसे अब आज हम आप
  हमारा हमारे हमें आपको कोई कुछ यानी इसलिए अगर नहीं दोस्तों नमस्ते जाता जाती जाते रहे रहा रही बात बारे
  सकता सकते देखते कहते करेंगे करेगा करेगी होगी होंगे बताता बताते समझते चाहिए वाले वाली वाला ओर तरफ
  पार पूरे देखेंगे देख मैं तुम उसी इसी उसे दूर नीचे ऊपर आगे पीछे हर किस कौन क्या कैसे क्यों कहाँ यहाँ वहाँ
  ஒரு இது அது மற்றும் என்று என்ற இந்த அந்த உள்ள ஆகும் ஆனால் போது
  ಒಂದು ಈ ಆ ಮತ್ತು ಎಂದು ಇದು ಅದು ಆಗಿದೆ ಇರುವ ಹಾಗೂ`
    .split(/\s+/)
    .filter(Boolean)
);

// Everyday English that is never a loanword in a lecture: without this, "imagine"
// would match मशीन and "trying" ट्रेनिंग, because their consonants collide.
const COMMON_ENGLISH = new Set(
  `able across actually add after against ago almost along already always among another answer
  anything around ask away back based become begin behind believe best better big bit body book
  bring build called came care carry case cause certain change clear close come common consider
  could course cut day different direction done during early easy either else end enough even
  ever every everything example far feel find first follow form found full general give given
  go good great group grow hand happen hard head hear help high hold home however idea imagine
  important include instead into keep kind know known large last late later lead learn least
  leave left less life light line little live long lot low made main make many may maybe mean
  might mind more morning most move much must name near need never new night note number often
  old once only open order part pass people perhaps person place play point possible power
  present problem put question quite rather reach read real reason right run said say second
  seem sense set several show side simple since small something sometimes soon start state
  stand standing step still stop story such sure take talk tell term than then thing think
  though thought three time together told took top toward trying try turn type understand until
  upon usually value want watch water way well went whole wide without word work world write
  year yes yet young below above slow fast steep steepest`
    .split(/\s+/)
    .filter(Boolean)
);

export const isStopword = (word) => STOPWORDS.has(word.toLowerCase());

/** Words with their character offsets, skeleton and script. */
export function tokenize(text) {
  const tokens = [];
  for (const m of String(text || "").matchAll(WORD_RE)) {
    const word = m[0];
    if (!/\p{L}/u.test(word)) continue;
    tokens.push({
      text: word,
      norm: word.toLowerCase(),
      start: m.index,
      end: m.index + word.length,
      skel: skeleton(word),
      script: scriptOf(word),
      stop: STOPWORDS.has(word.toLowerCase()),
      common: COMMON_ENGLISH.has(word.toLowerCase()),
    });
  }
  return tokens;
}

/** Similarity of two tokens: exact-ish within a script, phonetic across scripts. */
export function tokenSimilarity(a, b) {
  if (a.norm === b.norm) return 1;
  if (a.script === b.script) {
    const min = Math.min(a.norm.length, b.norm.length);
    const max = Math.max(a.norm.length, b.norm.length);
    let shared = 0;
    while (shared < min && a.norm[shared] === b.norm[shared]) shared++;
    if (shared >= 4 && shared >= 0.6 * max) return 0.85; // same stem, different ending
    return 0;
  }
  if (a.stop || b.stop || a.common || b.common) return 0;
  return skeletonSimilarity(a.skel, b.skel);
}

/** Dominant language of a text, judged by script. */
export function detectLanguage(text) {
  const blocks = [
    ["Hindi", 0x0900], ["Bengali", 0x0980], ["Punjabi", 0x0a00], ["Gujarati", 0x0a80], ["Odia", 0x0b00],
    ["Tamil", 0x0b80], ["Telugu", 0x0c00], ["Kannada", 0x0c80], ["Malayalam", 0x0d00],
  ];
  const counts = new Map();
  for (const ch of String(text || "")) {
    const cp = ch.codePointAt(0);
    if (/[a-z]/i.test(ch)) counts.set("English", (counts.get("English") || 0) + 1);
    else if (isIndicCode(cp)) {
      const name = blocks.find(([, base]) => cp >= base && cp < base + 0x80)?.[0];
      if (name) counts.set(name, (counts.get(name) || 0) + 1);
    }
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] || null;
}
