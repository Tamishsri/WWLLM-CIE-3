"""
llm_engine.py — BharatGen Param1 Summarisation & Quiz Generation
================================================================
Wraps the ``bharatgenai/Param-1-2.9B-Instruct`` decoder-only causal LM
with structured prompting for:
    1. **Lecture Notes Generation** — zero-shot markdown-structured notes.
    2. **MCQ Quiz Generation** — one-shot prompt with a fixed plain format,
       parsed into the canonical quiz schema.

Memory Optimisation
-------------------
* Uses ``bitsandbytes`` 8-bit quantisation (``load_in_8bit=True``) via
  the HuggingFace ``BitsAndBytesConfig`` to fit the 2.9 B-parameter
  model within ~6 GB VRAM.
* ``device_map="auto"`` lets ``accelerate`` shard across available
  GPUs or spill to CPU RAM.
* On CPU the checkpoint's own bf16 weights load directly (about 6 GB of RAM).

The quiz is written in a plain ``Q: / A) / Answer: / Why:`` format rather than
JSON: Param-1 follows it far more reliably, and a reply cut off by the token
limit loses one question instead of the whole quiz.

Usage::

    engine = LLMEngine()
    notes_md = engine.generate_notes(transcript)
    for piece in engine.stream_notes(transcript):   # token streaming for UIs
        print(piece, end="")
    quiz_json = engine.generate_quiz(transcript, num_questions=5)
    raw = "".join(engine.stream_quiz(transcript, 5))    # or stream the raw JSON
    quiz_json = engine.parse_quiz(raw)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import torch
from loguru import logger

from src.pipelines.quiz_schema import normalise_quiz

try:
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        LogitsProcessorList,
        RepetitionPenaltyLogitsProcessor,
        TemperatureLogitsWarper,
        TopPLogitsWarper,
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

TOP_P: float = 0.9
REPETITION_PENALTY: float = 1.15


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
8. Write in English, whatever language the transcript is in.
"""

SYSTEM_PROMPT_QUIZ: str = """\
You are a quiz-generation assistant. You write multiple-choice questions about \
the lecture you are given, in English, whatever language the lecture is in.
"""

# The task comes after the transcript, and the format has no worked example:
# Param-1 copies a worked example instead of writing about the lecture.
USER_PROMPT_QUIZ: str = """\
Lecture transcript:

{transcript}

Write {num_questions} multiple-choice questions that test understanding of this \
lecture. Cover different parts of it. Use this format for each question:

Q1: the question
A) first option
B) second option
C) third option
D) fourth option
Answer: the letter of the correct option
Why: one sentence explaining the answer
"""

_QUESTION_RE = re.compile(r"^\s*Q\d*[:.)]\s*", re.MULTILINE)
_OPTION_RE = re.compile(r"^[A-Ha-h][).:]\s")
_ANSWER_RE = re.compile(r"^(?:correct\s+)?answer\s*[:\-]\s*(.+)", re.IGNORECASE)
_WHY_RE = re.compile(r"^(?:why|explanation)\s*[:\-]\s*(.+)", re.IGNORECASE)


@dataclass
class _Reply:
    """A reply in progress: every token so far, and those the model hasn't read."""

    sequence: torch.Tensor
    unread: torch.Tensor
    past: Any = None


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
    max_new_tokens_notes, max_new_tokens_quiz : int
        Length ceilings.  Lower them on CPU, where each token costs real time.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_LLM_MODEL,
        quantize_8bit: bool = True,
        device_map: str = "auto",
        temperature: float = 0.4,
        max_new_tokens_notes: int = MAX_NEW_TOKENS_NOTES,
        max_new_tokens_quiz: int = MAX_NEW_TOKENS_QUIZ,
    ) -> None:
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "HuggingFace Transformers is required.  "
                "pip install transformers accelerate bitsandbytes"
            )

        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens_notes = max_new_tokens_notes
        self.max_new_tokens_quiz = max_new_tokens_quiz

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
        # fp16 is GPU-only in practice.  On CPU, keep the checkpoint's bf16 and
        # skip device_map, which would offload layers to disk when RAM is tight.
        cuda = torch.cuda.is_available()
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        rope = getattr(config, "rope_scaling", None)
        if isinstance(rope, dict) and "type" not in rope:
            # transformers 5 fills this in with defaults, and Param-1's remote
            # code reads any non-empty value as a scaling request.
            config.rope_scaling = None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            quantization_config=quant_config,
            device_map=device_map if cuda else None,
            trust_remote_code=True,
            torch_dtype=torch.float16 if cuda else torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self._rebuild_rotary_tables()
        logger.success(f"LLM ready — {model_id}")

    def _rebuild_rotary_tables(self) -> None:
        """Recompute Param-1's rotary position tables.

        They aren't stored in the checkpoint, and transformers 5 builds models
        without real memory first, so they load uninitialised.  The weights are
        fine but the model loses all sense of word order and writes gibberish.
        """
        for module in self.model.modules():
            if hasattr(module, "_set_cos_sin_cache"):
                steps = torch.arange(0, module.dim, 2).float()
                module.inv_freq = (1.0 / (module.base ** (steps / module.dim))).to(module.inv_freq.device)
                module._set_cos_sin_cache(module.max_seq_len_cached, module.inv_freq.device, torch.float32)

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
            self._notes_messages(transcript), max_new_tokens=self.max_new_tokens_notes
        )

    def stream_notes(self, transcript: str) -> Iterator[str]:
        """Yield the notes as they are generated (decoded text pieces).

        Callers can forward the pieces to a UI in real time; closing this
        generator stops the model.
        """
        if not transcript.strip():
            yield "_No transcript provided._"
            return
        yield from self._generate_stream(
            self._notes_messages(transcript), max_new_tokens=self.max_new_tokens_notes
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
        return self.parse_quiz("".join(self.stream_quiz(transcript, num_questions)))

    def stream_quiz(self, transcript: str, num_questions: int = 5) -> Iterator[str]:
        """Yield the quiz's raw text as it is generated, stopping after *num_questions*."""
        truncated = self._truncate_to_tokens(transcript, max_tokens=2400)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_QUIZ},
            {"role": "user", "content": USER_PROMPT_QUIZ.format(transcript=truncated, num_questions=num_questions)},
        ]
        # Param-1 stops after one question, so each one is started for it.  The
        # reply carries on, so the transcript is only read once.
        input_ids = self._encode(messages)
        reply = _Reply(input_ids, input_ids)
        budget = self.max_new_tokens_quiz // num_questions
        for number in range(1, num_questions + 1):
            question = ""
            for piece in self._continue(reply, f"Q{number}:" if number == 1 else f"\n\nQ{number}:", budget):
                question += piece
                yield piece
            if "Why:" not in question:
                yield from self._continue(reply, "\nWhy:", budget // 3)

    @staticmethod
    def count_questions(raw: str) -> int:
        """How many questions a quiz reply has started so far."""
        return len(_QUESTION_RE.findall(raw))

    def parse_quiz(self, raw: str) -> List[Dict[str, Any]]:
        """Parse the model's reply, then validate the answer keys."""
        return normalise_quiz(self._parse_quiz_text(raw) or self._parse_quiz_json(raw.strip()))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def _generate(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
    ) -> str:
        """Run chat-style generation and return the assistant's reply.

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
        return "".join(self._generate_stream(messages, max_new_tokens)).strip()

    def _generate_stream(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
    ) -> Iterator[str]:
        """Yield the reply as it is generated.

        Closing the generator stops the model, which is how the UI's Stop
        button works.
        """
        input_ids = self._encode(messages)
        yield from self._decode(self._sample(_Reply(input_ids, input_ids), max_new_tokens))

    def _continue(self, reply: _Reply, opening: str, max_new_tokens: int) -> Iterator[str]:
        """Write *opening* into *reply*, then let the model carry on from it."""
        ids = self.tokenizer(opening, add_special_tokens=False, return_tensors="pt").input_ids
        ids = ids.to(reply.sequence.device)
        reply.sequence = torch.cat([reply.sequence, ids], dim=-1)
        reply.unread = torch.cat([reply.unread, ids], dim=-1)
        yield opening
        # The format repeats on purpose, so repetition isn't penalised here.
        yield from self._decode(self._sample(reply, max_new_tokens, repetition_penalty=1.0))

    def _decode(self, tokens: Iterator[int]) -> Iterator[str]:
        """Turn token ids into text pieces as they arrive."""
        seen: List[int] = []
        sent = 0
        for token in tokens:
            seen.append(token)
            text = self.tokenizer.decode(seen, skip_special_tokens=True)
            if len(text) > sent and not text.endswith("\ufffd"):  # U+FFFD: a character split across tokens
                yield text[sent:]
                sent = len(text)

    def _sample(
        self,
        reply: _Reply,
        max_new_tokens: int,
        repetition_penalty: float = REPETITION_PENALTY,
    ) -> Iterator[int]:
        """Yield new token ids one at a time, keeping *reply* up to date.

        Param-1's remote code keeps its key/value cache as tuples, which
        ``generate()`` in current transformers can't drive.  Without the cache
        every token re-runs the whole prompt, so this loop carries it instead.
        """
        processors = LogitsProcessorList(
            [
                RepetitionPenaltyLogitsProcessor(repetition_penalty),
                TemperatureLogitsWarper(self.temperature),
                TopPLogitsWarper(TOP_P),
            ]
        )
        eos = self.tokenizer.eos_token_id
        with torch.no_grad():
            for _ in range(max_new_tokens):
                out = self.model(
                    input_ids=reply.unread, past_key_values=reply.past, use_cache=True, return_dict=True
                )
                reply.past = out.past_key_values
                scores = processors(reply.sequence, out.logits[:, -1, :].float())
                token = torch.multinomial(scores.softmax(dim=-1), num_samples=1)
                if token.item() == eos:
                    reply.unread = token[:, :0]
                    return
                reply.sequence = torch.cat([reply.sequence, token], dim=-1)
                reply.unread = token
                yield token.item()

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
    def _parse_quiz_text(raw: str) -> List[Dict[str, Any]]:
        """Read the ``Q: / A) / Answer: / Why:`` format into MCQ dicts."""
        items: List[Dict[str, Any]] = []
        for block in _QUESTION_RE.split(raw)[1:]:
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if not lines:
                continue
            item: Dict[str, Any] = {"question": lines[0], "options": []}
            for line in lines[1:]:
                answer, why = _ANSWER_RE.match(line), _WHY_RE.match(line)
                if _OPTION_RE.match(line):
                    item["options"].append(line)
                elif answer:
                    item["correct_answer"] = answer.group(1)
                elif why:
                    item["explanation"] = why.group(1)
            items.append(item)
        return items

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
