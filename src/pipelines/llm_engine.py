"""
llm_engine.py — BharatGen Param1 Summarisation & Quiz Generation
================================================================
Wraps the ``bharatgenai/Param-1-2.9B-Instruct`` decoder-only causal LM
with structured prompting for:
    1. **Lecture Notes Generation** — zero-shot markdown-structured notes.
    2. **MCQ Quiz Generation** — few-shot prompt enforcing strict JSON
       output for easy UI rendering.

Memory Optimisation
-------------------
* Uses ``bitsandbytes`` 8-bit quantisation (``load_in_8bit=True``) via
  the HuggingFace ``BitsAndBytesConfig`` to fit the 2.9 B-parameter
  model within ~6 GB VRAM.
* ``device_map="auto"`` lets ``accelerate`` shard across available
  GPUs or spill to CPU RAM.

Usage::

    engine = LLMEngine()
    notes_md = engine.generate_notes(transcript)
    for piece in engine.stream_notes(transcript):   # token streaming for UIs
        print(piece, end="")
    quiz_json = engine.generate_quiz(transcript, num_questions=5)
"""

from __future__ import annotations

import json
import re
from threading import Thread
from typing import Any, Dict, Iterator, List, Optional

import torch
from loguru import logger

from src.pipelines.quiz_schema import normalise_quiz

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TextIteratorStreamer,
    )

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_LLM_MODEL: str = "bharatgenai/Param-1-2.9B-Instruct"
"""HuggingFace model ID for BharatGen Param-1 (instruct-tuned)."""

MAX_NEW_TOKENS_NOTES: int = 2048
"""Maximum generation length for lecture notes."""

MAX_NEW_TOKENS_QUIZ: int = 2048
"""Maximum generation length for MCQ quiz output."""


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------
SYSTEM_PROMPT_NOTES: str = """\
You are an expert academic note-taker. Given a raw lecture transcript, \
produce well-structured **Markdown** lecture notes. Follow these rules:

1. Start with a concise **Title** (inferred from the content).
2. Organise content under clear **## Section Headings**.
3. Use bullet points for key concepts and numbered lists for steps/processes.
4. Highlight important terms in **bold**.
5. Include a **## Key Takeaways** section at the end with 3-5 bullet points.
6. Do NOT add information that is not present in the transcript.
7. Preserve technical accuracy — do not paraphrase formulas or proper nouns.
"""

SYSTEM_PROMPT_QUIZ: str = """\
You are a quiz-generation assistant. Given a lecture transcript, generate \
exactly {num_questions} multiple-choice questions (MCQs) that test \
understanding of the key concepts.

**Output format**: Return ONLY a valid JSON array. Each element must be:
{{
  "question": "<question text>",
  "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
  "correct_answer": "<full text of correct option>",
  "explanation": "<1-2 sentence explanation>"
}}

Rules:
- Each question must have exactly 4 options labelled A–D.
- Distractors should be plausible but clearly incorrect.
- Questions should cover different parts of the lecture.
- Output ONLY the JSON array — no preamble, no markdown fences.
"""


# ---------------------------------------------------------------------------
# LLMEngine
# ---------------------------------------------------------------------------
class LLMEngine:
    """BharatGen Param-1 inference engine for notes and quiz generation.

    Parameters
    ----------
    model_id : str
        HuggingFace model identifier.
    quantize_8bit : bool
        If *True* (default), load with ``bitsandbytes`` 8-bit quantisation.
    device_map : str
        Accelerate device map strategy.  ``"auto"`` is recommended.
    temperature : float
        Sampling temperature.  Lower → more deterministic.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_LLM_MODEL,
        quantize_8bit: bool = True,
        device_map: str = "auto",
        temperature: float = 0.4,
    ) -> None:
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "HuggingFace Transformers is required.  "
                "pip install transformers accelerate bitsandbytes"
            )

        self.model_id = model_id
        self.temperature = temperature

        # ── Quantisation config ─────────────────────────────────────────
        quant_config: Optional[BitsAndBytesConfig] = None
        if quantize_8bit:
            quant_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,
            )
            logger.info("8-bit quantisation enabled (bitsandbytes).")

        # ── Load tokenizer ──────────────────────────────────────────────
        logger.info(f"Loading tokenizer: {model_id} …")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True
        )
        # Ensure pad token is set (some causal LMs omit it)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── Load model ──────────────────────────────────────────────────
        logger.info(f"Loading LLM: {model_id} …")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map=device_map,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        self.model.eval()
        logger.success(f"LLM ready — {model_id}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_notes(self, transcript: str) -> str:
        """Generate structured Markdown lecture notes from a transcript.

        Parameters
        ----------
        transcript : str
            Raw ASR transcript text.

        Returns
        -------
        str
            Markdown-formatted lecture notes.
        """
        if not transcript.strip():
            return "_No transcript provided._"
        return self._generate(
            self._notes_messages(transcript), max_new_tokens=MAX_NEW_TOKENS_NOTES
        )

    def stream_notes(self, transcript: str) -> Iterator[str]:
        """Yield the notes as they are generated (decoded text pieces).

        Uses ``TextIteratorStreamer`` with generation running in a worker
        thread, so callers can forward tokens to a UI in real time.
        """
        if not transcript.strip():
            yield "_No transcript provided._"
            return
        yield from self._generate_stream(
            self._notes_messages(transcript), max_new_tokens=MAX_NEW_TOKENS_NOTES
        )

    def _notes_messages(self, transcript: str) -> List[Dict[str, str]]:
        # Truncate extremely long transcripts to fit context window (4096 tok)
        truncated = self._truncate_to_tokens(transcript, max_tokens=2800)
        return [
            {"role": "system", "content": SYSTEM_PROMPT_NOTES},
            {"role": "user", "content": f"Transcript:\n\n{truncated}"},
        ]

    def generate_quiz(
        self,
        transcript: str,
        num_questions: int = 5,
    ) -> List[Dict[str, Any]]:
        """Generate MCQs in strict JSON format.

        Parameters
        ----------
        transcript : str
            Raw ASR transcript text.
        num_questions : int
            Number of MCQs to generate.

        Returns
        -------
        List[Dict[str, Any]]
            Parsed list of MCQ dicts in the canonical schema of
            :func:`~src.pipelines.quiz_schema.normalise_quiz`: ``question``,
            ``options``, ``answer_index``, ``correct_answer``, ``explanation``.
        """
        if not transcript.strip():
            return []

        truncated = self._truncate_to_tokens(transcript, max_tokens=2400)

        system = SYSTEM_PROMPT_QUIZ.format(num_questions=num_questions)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Transcript:\n\n{truncated}"},
        ]
        raw_output = self._generate(messages, max_new_tokens=MAX_NEW_TOKENS_QUIZ)

        # ── Parse JSON from LLM output, then validate the answer keys ──
        return normalise_quiz(self._parse_quiz_json(raw_output))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
    ) -> str:
        """Run chat-style generation.

        Uses the tokenizer's ``apply_chat_template`` for proper
        formatting with BharatGen's expected control tokens.

        Parameters
        ----------
        messages : List[Dict[str, str]]
            Chat messages with ``role`` and ``content`` keys.
        max_new_tokens : int
            Generation ceiling.

        Returns
        -------
        str
            Decoded model output (assistant reply only).
        """
        input_ids = self._encode(messages)
        input_length = input_ids.shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids, **self._sampling_kwargs(max_new_tokens)
            )

        # Decode only the newly generated tokens (skip input prompt)
        generated = output_ids[0, input_length:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _generate_stream(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
    ) -> Iterator[str]:
        """Streaming variant of :meth:`_generate`."""
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        kwargs = dict(
            input_ids=self._encode(messages),
            streamer=streamer,
            **self._sampling_kwargs(max_new_tokens),
        )
        worker = Thread(target=self._generate_no_grad, kwargs=kwargs, daemon=True)
        worker.start()
        for piece in streamer:
            if piece:
                yield piece
        worker.join()

    def _generate_no_grad(self, **kwargs: Any) -> None:
        with torch.no_grad():
            self.model.generate(**kwargs)

    def _encode(self, messages: List[Dict[str, str]]) -> torch.Tensor:
        """Apply the chat template and return ``input_ids`` on the model device.

        Newer ``transformers`` releases may return a ``BatchEncoding`` rather
        than a bare tensor, so both shapes are accepted.
        """
        encoded = self.tokenizer.apply_chat_template(
            conversation=messages,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        input_ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        return input_ids.to(self.model.device)

    def _sampling_kwargs(self, max_new_tokens: int) -> Dict[str, Any]:
        return dict(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=self.tokenizer.pad_token_id,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate *text* to fit within *max_tokens*.

        Uses the tokenizer for accurate counting, then decodes back.

        Parameters
        ----------
        text : str
        max_tokens : int

        Returns
        -------
        str
            Possibly shortened text.
        """
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            return text
        logger.warning(
            f"Transcript truncated from {len(token_ids)} → {max_tokens} tokens."
        )
        return self.tokenizer.decode(
            token_ids[:max_tokens], skip_special_tokens=True
        )

    @staticmethod
    def _parse_quiz_json(raw: str) -> List[Dict[str, Any]]:
        """Extract and parse a JSON array from potentially noisy LLM output.

        The model sometimes wraps JSON in markdown fences or adds preamble
        text.  This method tries several extraction strategies:

        1. Direct ``json.loads()``.
        2. Regex extraction of ``[...]`` from the string.
        3. Line-by-line object accumulation.

        Parameters
        ----------
        raw : str
            Raw LLM output string.

        Returns
        -------
        List[Dict[str, Any]]
            Parsed MCQ dicts.  Returns an empty list on parse failure.
        """
        # Strategy 1: direct parse
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        # Strategy 2: extract JSON array via regex
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                if isinstance(data, list):
                    return data
            except json.JSONDecodeError:
                pass

        # Strategy 3: look for individual JSON objects
        objects: List[Dict[str, Any]] = []
        for obj_match in re.finditer(r"\{[^{}]+\}", raw):
            try:
                obj = json.loads(obj_match.group(0))
                if "question" in obj:
                    objects.append(obj)
            except json.JSONDecodeError:
                continue

        if objects:
            return objects

        logger.error(f"Could not parse quiz JSON from LLM output:\n{raw[:500]}")
        return []

    @property
    def model_info(self) -> Dict[str, str]:
        """Return a summary dict for UI display."""
        return {
            "model_id": self.model_id,
            "quantized": "8-bit" if hasattr(self.model, "is_loaded_in_8bit") and self.model.is_loaded_in_8bit else "fp16",
            "temperature": str(self.temperature),
        }
