"""
translator.py — AI4Bharat IndicTrans2 Translation Pipeline
==========================================================
Translates English text (Markdown notes and JSON MCQs) into Indian
languages (Hindi, Tamil, Kannada, Telugu, Bengali, ...) using the
``ai4bharat/indictrans2-en-indic-1B`` seq2seq model.

This module uses the **IndicTransToolkit** preprocessor for script-
aware tokenisation and post-processing, which is required for correct
IndicTrans2 inference.

Installation::

    pip install git+https://github.com/VarunGumma/IndicTransToolkit.git

Usage::

    translator = TranslationEngine()
    hindi_notes = translator.translate(notes_md, target_lang="hin_Deva")
"""

from __future__ import annotations

import json
import re
from contextlib import nullcontext
from typing import Any, Dict, List, Optional

import torch
from loguru import logger

# Friendly name → FLORES-200 code (shared with the UI, re-exported here)
from src.pipelines.languages import SUPPORTED_TARGETS

try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

try:
    from IndicTransToolkit import IndicProcessor

    _INDIC_TOOLKIT_AVAILABLE = True
except ImportError:
    _INDIC_TOOLKIT_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_TRANS_MODEL: str = "ai4bharat/indictrans2-en-indic-1B"
"""HuggingFace model ID for IndicTrans2 (English → Indic, 1B params)."""

SRC_LANG: str = "eng_Latn"
"""FLORES-200 source language code (English, Latin script)."""

MAX_SEQ_LEN: int = 256
"""Maximum token sequence length per translation batch element."""


# ---------------------------------------------------------------------------
# TranslationEngine
# ---------------------------------------------------------------------------
class TranslationEngine:
    """IndicTrans2 translation wrapper.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID.
    device : str | None
        ``"cuda"`` or ``"cpu"``.  Auto-detects if *None*.
    num_beams : int
        Beam search width for translation decoding.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_TRANS_MODEL,
        device: Optional[str] = None,
        num_beams: int = 5,
    ) -> None:
        if not _HF_AVAILABLE:
            raise ImportError("transformers is required.  pip install transformers")
        if not _INDIC_TOOLKIT_AVAILABLE:
            raise ImportError(
                "IndicTransToolkit is required for IndicTrans2.  "
                "pip install git+https://github.com/VarunGumma/IndicTransToolkit.git"
            )

        self.model_id = model_id
        self.num_beams = num_beams

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # ── Load tokenizer & model ──────────────────────────────────────
        logger.info(f"Loading IndicTrans2: {model_id} …")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype=torch.float16
        ).to(self.device)
        self.model.eval()

        # ── IndicProcessor for pre/post processing ─────────────────────
        self.processor = IndicProcessor(inference=True)
        logger.success(f"IndicTrans2 ready on {self.device}.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def translate(
        self,
        text: str,
        target_lang: str = "hin_Deva",
        batch_size: int = 16,
    ) -> str:
        """Translate a block of English text into the target language.

        Handles multi-line / multi-paragraph text by splitting into
        sentences, batching through the model, then re-joining.

        Parameters
        ----------
        text : str
            English source text (plain text or Markdown).
        target_lang : str
            FLORES-200 target code (e.g. ``"hin_Deva"``).
        batch_size : int
            Sentences per inference batch.

        Returns
        -------
        str
            Translated text with original line structure preserved.
        """
        if not text.strip():
            return ""

        # Resolve friendly name → FLORES code if needed
        target_lang = SUPPORTED_TARGETS.get(target_lang, target_lang)

        # ── Split into translatable segments ────────────────────────────
        lines = text.split("\n")
        translated_lines: List[str] = []

        # Collect non-empty lines for batch translation
        non_empty: List[tuple[int, str]] = []
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if stripped and not self._is_markdown_only(stripped):
                # The prefix ("## ", "- ") is re-attached after translation.
                body = line[len(self._extract_md_prefix(line)):].strip()
                if body:
                    non_empty.append((idx, body))

        # ── Batch translate ─────────────────────────────────────────────
        source_texts = [t for _, t in non_empty]
        translated_texts = self._batch_translate(
            source_texts, target_lang, batch_size
        )

        # Build output preserving blank / markdown-only lines
        trans_map = {idx: trans for (idx, _), trans in zip(non_empty, translated_texts)}
        for idx, line in enumerate(lines):
            if idx in trans_map:
                # Preserve leading whitespace / markdown prefixes
                prefix = self._extract_md_prefix(line)
                translated_lines.append(prefix + trans_map[idx])
            else:
                translated_lines.append(line)

        return "\n".join(translated_lines)

    def translate_quiz(
        self,
        quiz: List[Dict[str, Any]],
        target_lang: str = "hin_Deva",
    ) -> List[Dict[str, Any]]:
        """Translate each field of every MCQ dict.

        Parameters
        ----------
        quiz : List[Dict[str, Any]]
            MCQ dicts (as returned by ``LLMEngine.generate_quiz``).
        target_lang : str
            FLORES-200 target code.

        Returns
        -------
        List[Dict[str, Any]]
            Translated MCQ dicts.
        """
        target_lang = SUPPORTED_TARGETS.get(target_lang, target_lang)
        translated: List[Dict[str, Any]] = []

        for q in quiz:
            tq: Dict[str, Any] = {}
            tq["question"] = self._translate_single(q.get("question", ""), target_lang)
            tq["options"] = [
                self._translate_single(opt, target_lang)
                for opt in q.get("options", [])
            ]
            idx = q.get("answer_index")
            if isinstance(idx, int) and 0 <= idx < len(tq["options"]):
                # Re-use the translated option so the key always matches it.
                tq["answer_index"] = idx
                tq["correct_answer"] = tq["options"][idx]
            else:
                tq["correct_answer"] = self._translate_single(
                    q.get("correct_answer", ""), target_lang
                )
            tq["explanation"] = self._translate_single(
                q.get("explanation", ""), target_lang
            )
            translated.append(tq)

        return translated

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _batch_translate(
        self,
        sentences: List[str],
        target_lang: str,
        batch_size: int,
    ) -> List[str]:
        """Translate a list of sentences in batches.

        Parameters
        ----------
        sentences : List[str]
        target_lang : str
        batch_size : int

        Returns
        -------
        List[str]
        """
        all_translated: List[str] = []

        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]

            # IndicTransToolkit preprocessing
            preprocessed = self.processor.preprocess_batch(
                batch, src_lang=SRC_LANG, tgt_lang=target_lang
            )

            inputs = self.tokenizer(
                preprocessed,
                padding="longest",
                truncation=True,
                max_length=MAX_SEQ_LEN,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    num_beams=self.num_beams,
                    max_length=MAX_SEQ_LEN,
                )

            # Removed from transformers' base tokenizer in v5; keep using it
            # where a custom tokenizer still provides it.
            target_mode = getattr(self.tokenizer, "as_target_tokenizer", nullcontext)
            with target_mode():
                decoded = self.tokenizer.batch_decode(
                    generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )

            # IndicTransToolkit postprocessing
            postprocessed = self.processor.postprocess_batch(
                decoded, lang=target_lang
            )
            all_translated.extend(postprocessed)

        return all_translated

    def _translate_single(self, text: str, target_lang: str) -> str:
        """Translate a single string."""
        if not text.strip():
            return text
        result = self._batch_translate([text], target_lang, batch_size=1)
        return result[0] if result else text

    @staticmethod
    def _is_markdown_only(line: str) -> bool:
        """Check if a line is pure markdown syntax (headings, separators, etc.)."""
        stripped = line.strip()
        if re.fullmatch(r"#{1,6}\s*", stripped):
            return True
        if re.fullmatch(r"[-*_]{3,}", stripped):
            return True
        if stripped == "":
            return True
        return False

    @staticmethod
    def _extract_md_prefix(line: str) -> str:
        """Extract leading markdown syntax (e.g. '## ', '- ', '  * ')."""
        match = re.match(r"^(\s*(?:#{1,6}\s+|[-*+]\s+|\d+\.\s+)?)", line)
        return match.group(1) if match else ""

    @property
    def model_info(self) -> Dict[str, str]:
        """Return a summary dict for UI display."""
        return {
            "model_id": self.model_id,
            "device": str(self.device),
            "supported_targets": ", ".join(SUPPORTED_TARGETS.keys()),
        }
