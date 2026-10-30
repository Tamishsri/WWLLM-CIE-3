// Interactive MCQ component: practice (answer, get feedback, score) and
// review (answers + explanations shown) modes.

import { esc, icon } from "./util.js";

const LETTERS = "ABCDEFGH";

function verdict(score) {
  if (score === 1) return ["Flawless.", "Every answer right — this lecture is yours."];
  if (score >= 0.8) return ["Great work.", "Just a detail or two to revisit."];
  if (score >= 0.5) return ["Good effort.", "Review the explanations and try again."];
  return ["Keep practising.", "Re-read the notes, then take another pass."];
}

/**
 * Mount a quiz into *el*.
 * @param {HTMLElement} el
 * @param {Array} quiz          canonical MCQs (question, options, answer_index, explanation)
 * @param {object} opts
 * @param {"practice"|"review"} opts.mode
 * @param {Record<number, number>} opts.answers  persisted answers (mutated in place)
 * @param {string} [opts.lang]  language name for Indic fonts
 * @param {string[]} [opts.anchors]  per-question HTML linking to the lecture moment
 * @param {() => void} [opts.onChange]  called after answers change
 * @param {(mode: string) => void} [opts.onMode]
 */
export function mountQuiz(el, quiz, opts) {
  const answers = opts.answers;
  const review = opts.mode === "review";
  const lang = opts.lang ? ` lang-text` : "";
  let fresh = -1; // card answered by the latest click (animates once)

  const optionClass = (q, qi, oi) => {
    const chosen = answers[qi];
    if (review) return oi === q.answer_index ? "is-correct" : "";
    if (chosen == null) return "";
    if (oi === q.answer_index) return "is-correct";
    if (oi === chosen) return "is-wrong";
    return "is-dim";
  };

  const card = (q, qi) => {
    const chosen = answers[qi];
    const answered = review || chosen != null;
    let state = !review && chosen != null ? (chosen === q.answer_index ? "correct" : "wrong") : "";
    if (qi === fresh) state += state === "correct" ? " fresh just-correct" : " fresh";
    const options = q.options
      .map((opt, oi) => {
        const cls = optionClass(q, qi, oi);
        const mark =
          cls === "is-correct"
            ? `<span class="mark">${icon("check", "icon-sm")}${review || chosen === oi ? "" : "Answer"}</span>`
            : cls === "is-wrong"
              ? `<span class="mark">${icon("x", "icon-sm")}Your pick</span>`
              : "<span></span>";
        return `<button class="q-opt ${cls}" type="button" data-q="${qi}" data-o="${oi}" ${answered ? "disabled" : ""}
                  aria-label="Option ${LETTERS[oi]}: ${esc(opt)}">
                  <span class="q-letter">${LETTERS[oi]}</span><span class="${lang}">${esc(opt)}</span>${mark}
                </button>`;
      })
      .join("");
    const anchor = answered ? opts.anchors?.[qi] || "" : "";
    const explanation =
      answered && (q.explanation || anchor)
        ? `<div class="q-explain">${icon("bulb", "icon-sm")}<div><span class="${lang}">${esc(q.explanation)}</span>${
            anchor ? `<div class="q-anchor">${anchor}</div>` : ""
          }</div></div>`
        : "";
    return `<article class="q-card ${state}" data-card="${qi}">
        <div class="q-head"><span class="q-num">Q${String(qi + 1).padStart(2, "0")}</span>
          <p class="q-text${lang}">${esc(q.question)}</p></div>
        <div class="q-options" role="group" aria-label="Question ${qi + 1} options">${options}</div>
        ${explanation}
      </article>`;
  };

  const render = () => {
    const total = quiz.length;
    const answered = Object.keys(answers).filter((k) => k < total).length;
    const correct = quiz.filter((q, i) => answers[i] === q.answer_index).length;
    const done = !review && answered === total && total > 0;
    const [title, line] = verdict(total ? correct / total : 0);

    const bar = review
      ? `<div class="quiz-bar">${icon("eye", "icon-sm")}<span>Review mode — correct answers and explanations are shown.</span>
           <button class="btn btn-sm" data-quiz="practice">${icon("bolt", "icon-sm")}Practice instead</button></div>`
      : `<div class="quiz-bar"><span><b>${answered}</b>/${total} answered · <b>${correct}</b> correct</span>
           <div class="progress"><i style="--p:${total ? answered / total : 0}"></i></div>
           <button class="btn btn-sm btn-quiet" data-quiz="reset" ${answered ? "" : "disabled"}>${icon("retry", "icon-sm")}Reset</button></div>`;

    const score = done
      ? `<div class="score">
           <div class="score-ring" style="--p:${Math.round((correct / total) * 100)}">${Math.round((correct / total) * 100)}%</div>
           <div><h3>${title}</h3><p>${correct} of ${total} correct. ${line}</p></div>
           <div class="panel-tools"><button class="btn" data-quiz="review">${icon("eye", "icon-sm")}Review</button>
             <button class="btn btn-primary" data-quiz="reset">${icon("retry", "icon-sm")}Retry</button></div>
         </div>`
      : "";

    el.innerHTML = `${bar}${score}<div class="quiz-list">${quiz.map(card).join("")}</div>`;
  };

  el.onclick = (e) => {
    const opt = e.target.closest(".q-opt");
    if (opt && !opt.disabled) {
      const qi = Number(opt.dataset.q);
      answers[qi] = Number(opt.dataset.o);
      fresh = qi;
      render();
      fresh = -1;
      opts.onChange?.();
      return;
    }
    const action = e.target.closest("[data-quiz]")?.dataset.quiz;
    if (action === "reset") {
      Object.keys(answers).forEach((k) => delete answers[k]);
      opts.onChange?.();
      opts.onMode?.("practice");
    } else if (action === "review" || action === "practice") {
      opts.onMode?.(action);
    }
  };

  render();
}
