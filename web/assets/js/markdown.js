// Small, safe Markdown renderer for LLM notes (escape first, then format).
// Supports headings, lists, paragraphs, blockquotes, code, hr, bold/italic,
// inline code and http(s) links. Returns the HTML plus an h2 table of contents.

import { esc, slugify } from "./util.js";

const TAKEAWAY_RE = /takeaway|key points|summary|निष्कर्ष|सारांश|முக்கியக் குறிப்புகள்|ಪ್ರಮುಖ ಅಂಶಗಳು/i;

export function inline(text) {
  const codes = [];
  let s = esc(text).replace(/`([^`]+)`/g, (_, code) => {
    codes.push(code);
    return `\u0000${codes.length - 1}\u0000`;
  });
  s = s
    .replace(/\*\*([^*]+?)\*\*|__([^_]+?)__/g, (_, a, b) => `<strong>${a ?? b}</strong>`)
    .replace(/(^|[^*\w])\*(?!\s)([^*]+?)\*(?!\w)/g, "$1<em>$2</em>")
    .replace(/(^|[^_\w])_(?!\s)([^_]+?)_(?!\w)/g, "$1<em>$2</em>")
    .replace(
      /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
    );
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${codes[+i]}</code>`);
}

/** Classify one Markdown line for the side-by-side translation view. */
export function lineKind(line) {
  const t = line.trim();
  if (!t) return { kind: "gap", text: "" };
  let m = t.match(/^(#{1,6})\s+(.*)$/);
  if (m) return { kind: m[1].length === 1 ? "h1" : "h2", text: m[2] };
  m = t.match(/^[-*+]\s+(.*)$/);
  if (m) return { kind: "li", mark: "•", text: m[1] };
  m = t.match(/^(\d+)[.)]\s+(.*)$/);
  if (m) return { kind: "li", mark: `${m[1]}.`, text: m[2] };
  return { kind: "p", text: t };
}

/** Models pick their own heading depth; the UI expects a `#` title and `##` sections. */
export function levelHeadings(md) {
  const depths = [...String(md || "").matchAll(/^ {0,3}(#{1,6})\s/gm)].map((m) => m[1].length);
  if (!depths.length) return String(md || "");
  const top = Math.min(...depths);
  const titled = depths.filter((d) => d === top).length === 1;
  const shift = top - (titled ? 1 : 2);
  if (shift <= 0) return String(md);
  return String(md).replace(/^( {0,3})(#{1,6})(?=\s)/gm, (_, pad, hashes) => pad + "#".repeat(Math.max(1, hashes.length - shift)));
}

export function render(md) {
  const lines = levelHeadings(md).replace(/\r\n?/g, "\n").split("\n");
  const out = [];
  const toc = [];
  const ids = new Set();
  let para = [];
  let list = null;
  let inTakeaways = false;

  const uniqueId = (text) => {
    let id = slugify(text);
    for (let n = 2; ids.has(id); n++) id = `${slugify(text)}-${n}`;
    ids.add(id);
    return id;
  };
  const flushPara = () => {
    if (para.length) out.push(`<p>${inline(para.join(" "))}</p>`);
    para = [];
  };
  const flushList = () => {
    if (!list) return;
    const start = list.type === "ol" && list.start !== 1 ? ` start="${list.start}"` : "";
    out.push(`<${list.type}${start}>${list.items.map((li) => `<li>${li}</li>`).join("")}</${list.type}>`);
    list = null;
  };
  const flush = () => {
    flushPara();
    flushList();
  };

  for (let i = 0; i < lines.length; i++) {
    const raw = lines[i];
    const line = raw.trimEnd();

    if (/^\s*```/.test(line)) {
      flush();
      const buf = [];
      for (i++; i < lines.length && !/^\s*```/.test(lines[i]); i++) buf.push(lines[i]);
      out.push(`<pre><code>${esc(buf.join("\n"))}</code></pre>`);
      continue;
    }

    const heading = line.match(/^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$/);
    if (heading) {
      flush();
      const level = heading[1].length;
      const text = heading[2];
      const id = uniqueId(text);
      if (level <= 2 && inTakeaways) {
        out.push("</section>");
        inTakeaways = false;
      }
      if (level === 2) {
        toc.push({ id, text: text.replace(/[*_`]/g, "") });
        if (TAKEAWAY_RE.test(text)) {
          out.push('<section class="takeaways">');
          inTakeaways = true;
        }
      }
      out.push(`<h${level} id="${id}">${inline(text)}</h${level}>`);
      continue;
    }

    if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) {
      flush();
      out.push("<hr>");
      continue;
    }

    const ul = line.match(/^\s*[-*+]\s+(.*)$/);
    const ol = line.match(/^\s*(\d+)[.)]\s+(.*)$/);
    if (ul || ol) {
      flushPara();
      const type = ul ? "ul" : "ol";
      if (list && list.type !== type) flushList();
      if (!list) list = { type, items: [], start: ol ? Number(ol[1]) : 1 };
      list.items.push(inline(ul ? ul[1] : ol[2]));
      continue;
    }

    if (/^\s*>\s?/.test(line)) {
      flush();
      const buf = [];
      for (; i < lines.length && /^\s*>\s?/.test(lines[i]); i++) buf.push(lines[i].replace(/^\s*>\s?/, ""));
      i--;
      out.push(`<blockquote>${inline(buf.join(" "))}</blockquote>`);
      continue;
    }

    if (!line.trim()) {
      flush();
      continue;
    }

    if (list && /^\s{2,}\S/.test(raw)) {
      list.items[list.items.length - 1] += " " + inline(line.trim());
      continue;
    }
    para.push(line.trim());
  }
  flush();
  if (inTakeaways) out.push("</section>");
  return { html: out.join("\n"), toc };
}
