"""
quiz_schema.py — MCQ normalisation shared by every quiz producer
================================================================
LLMs are loose with the quiz format: options arrive as ``"A) text"`` or
``"text"``, and ``correct_answer`` may be the letter, the labelled option
or a paraphrase of it.  ``normalise_quiz`` converts any of these into one
canonical schema the UI and translator can rely on::

    {
        "question": str,
        "options": [str, ...],          # option text without "A)" labels
        "answer_index": int,            # index into options
        "correct_answer": str,          # == options[answer_index]
        "explanation": str,
    }

Questions whose answer cannot be resolved are dropped rather than shown
with a wrong key.  This module is dependency-free so it can be used in
demo mode without PyTorch.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

_LABEL_RE = re.compile(r"^\s*\(?([A-Ha-h])\s*[\)\.:\-]\s+")
_BARE_LETTER_RE = re.compile(r"^\(?([A-Ha-h])[\)\.:]?$")


def strip_option_label(option: str) -> str:
    """Remove a leading ``A)`` / ``(b)`` / ``C.`` label from an option."""
    return _LABEL_RE.sub("", option, count=1).strip()


def resolve_answer_index(
    options: List[str],
    answer: Any,
    hint: Any = None,
) -> Optional[int]:
    """Find which option *answer* refers to.

    Tries, in order: an explicit integer *hint*, exact text, a bare letter
    (``"B"``), label-insensitive text, and finally containment.
    """
    if isinstance(hint, int) and not isinstance(hint, bool) and 0 <= hint < len(options):
        return hint
    if answer is None:
        return None

    raw = str(answer).strip()
    if not raw:
        return None

    for i, opt in enumerate(options):
        if opt == raw:
            return i

    letter = _BARE_LETTER_RE.match(raw)
    if letter:
        idx = ord(letter.group(1).upper()) - ord("A")
        return idx if idx < len(options) else None

    core = strip_option_label(raw).casefold()
    plain = [strip_option_label(o).casefold() for o in options]
    for i, opt in enumerate(plain):
        if opt == core:
            return i

    labelled = _LABEL_RE.match(raw)
    if labelled:
        idx = ord(labelled.group(1).upper()) - ord("A")
        if idx < len(options):
            return idx

    for i, opt in enumerate(plain):
        if core and opt and (core in opt or opt in core):
            return i
    return None


def normalise_quiz(items: Iterable[Any]) -> List[Dict[str, Any]]:
    """Coerce raw MCQ dicts into the canonical schema (see module docstring)."""
    quiz: List[Dict[str, Any]] = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        raw_options = [str(o).strip() for o in item.get("options", []) if str(o).strip()]
        if not question or len(raw_options) < 2:
            continue

        idx = resolve_answer_index(
            raw_options, item.get("correct_answer"), item.get("answer_index")
        )
        if idx is None:
            continue

        options = [strip_option_label(o) or o for o in raw_options]
        quiz.append(
            {
                "question": question,
                "options": options,
                "answer_index": idx,
                "correct_answer": options[idx],
                "explanation": str(item.get("explanation", "")).strip(),
            }
        )
    return quiz
