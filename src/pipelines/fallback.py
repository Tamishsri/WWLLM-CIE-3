"""
fallback.py — Torch-free stand-ins so every stage runs on any machine
=====================================================================
When the heavy ML stack (PyTorch, NeMo, Transformers) is not installed,
the Studio API swaps in these lightweight engines:

    • ``EnergyVAD``                  — RMS-energy voice activity detection on
                                        PCM WAV (NumPy only).  Mirrors the
                                        chunk-merging rules of ``AudioProcessor``.
    • ``generate_extractive_notes``  — frequency-scored extractive Markdown notes.
    • ``generate_cloze_quiz``        — fill-in-the-blank MCQs built from key terms.

They are honest approximations, not replacements: the UI labels their
output as "fallback" so nobody mistakes it for model output.
"""

from __future__ import annotations

import math
import random
import re
import wave
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Span = Tuple[float, float]


# ===========================================================================
# Energy VAD
# ===========================================================================
@dataclass
class VADResult:
    """Output of :meth:`EnergyVAD.process_wav`."""

    duration: float
    sample_rate: int
    spans: List[Span]
    raw_segments: int
    samples: Optional[np.ndarray] = field(default=None, repr=False)


class EnergyVAD:
    """Frame-energy voice activity detector with Silero-compatible chunking.

    Parameters mirror :class:`~src.pipelines.audio_utils.AudioProcessor` so
    the two produce comparable chunk layouts.

    Parameters
    ----------
    max_chunk_sec, min_chunk_sec : float
        Chunk merging bounds (same semantics as ``AudioProcessor``).
    min_silence_ms : int
        Gaps shorter than this are bridged.
    min_speech_ms : int
        Speech runs shorter than this are dropped.
    sensitivity : float
        Fraction of the noise-floor → peak dynamic range a frame must exceed
        to count as speech.  Lower is more permissive.
    """

    def __init__(
        self,
        max_chunk_sec: float = 25.0,
        min_chunk_sec: float = 2.0,
        min_silence_ms: int = 300,
        min_speech_ms: int = 250,
        sensitivity: float = 0.35,
        frame_ms: int = 30,
    ) -> None:
        self.max_chunk_sec = max_chunk_sec
        self.min_chunk_sec = min_chunk_sec
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.sensitivity = sensitivity
        self.frame_ms = frame_ms

    # ------------------------------------------------------------------
    def process_wav(self, path: str | Path, keep_samples: bool = False) -> VADResult:
        """Read a PCM WAV file and return merged speech spans (seconds)."""
        samples, sr = read_wav(path)
        raw = self.detect(samples, sr)
        duration = len(samples) / sr if sr else 0.0
        spans = self.merge(raw, duration)
        return VADResult(
            duration=duration,
            sample_rate=sr,
            spans=spans,
            raw_segments=len(raw),
            samples=samples if keep_samples else None,
        )

    def detect(self, samples: np.ndarray, sr: int) -> List[Span]:
        """Return raw speech segments ``[(start_s, end_s), …]``."""
        frame = max(1, int(sr * self.frame_ms / 1000))
        n_frames = len(samples) // frame
        if n_frames == 0:
            return []

        frames = samples[: n_frames * frame].reshape(n_frames, frame).astype(np.float64)
        rms = np.sqrt(np.mean(frames**2, axis=1))
        db = 20.0 * np.log10(rms + 1e-10)

        floor, peak = np.percentile(db, 10), np.percentile(db, 98)
        if peak < -60.0:
            return []  # effectively digital silence
        if peak - floor < 10.0:
            # Steady signal (tone, constant hum): no dynamic range to split on.
            speech = db > max(peak - 20.0, -60.0)
        else:
            speech = db > floor + (peak - floor) * self.sensitivity

        # Run-length encode speech frames → [start, end) frame pairs.
        edges = np.flatnonzero(np.diff(np.concatenate(([0], speech.astype(np.int8), [0]))))
        runs = edges.reshape(-1, 2).tolist()

        gap_frames = math.ceil(self.min_silence_ms / self.frame_ms)
        bridged: List[List[int]] = []
        for start, end in runs:
            if bridged and start - bridged[-1][1] < gap_frames:
                bridged[-1][1] = end
            else:
                bridged.append([start, end])

        min_frames = math.ceil(self.min_speech_ms / self.frame_ms)
        pad = 2  # ~60 ms of context either side, like Silero's speech_pad_ms
        sec = self.frame_ms / 1000.0
        segments: List[Span] = []
        for start, end in bridged:
            if end - start < min_frames:
                continue
            s = max(0, start - pad) * sec
            e = min(n_frames, end + pad) * sec
            segments.extend(self._split_long((s, e)))
        return segments

    def _split_long(self, span: Span) -> List[Span]:
        """Split a segment longer than ``max_chunk_sec`` (Silero's max_speech_duration_s)."""
        start, end = span
        length = end - start
        if length <= self.max_chunk_sec:
            return [span]
        parts = math.ceil(length / self.max_chunk_sec)
        step = length / parts
        return [(start + i * step, start + (i + 1) * step) for i in range(parts)]

    def merge(self, segments: Sequence[Span], duration: float) -> List[Span]:
        """Merge segments into ≤ ``max_chunk_sec`` chunks (mirrors AudioProcessor)."""
        if not segments:
            return [(0.0, round(duration, 3))] if duration > 0 else []

        chunks: List[Span] = []
        run_start, run_end = segments[0]
        for seg_start, seg_end in segments[1:]:
            if seg_end - run_start <= self.max_chunk_sec:
                run_end = seg_end
            else:
                chunks.append((run_start, run_end))
                run_start, run_end = seg_start, seg_end

        if run_end - run_start >= self.min_chunk_sec or not chunks:
            chunks.append((run_start, run_end))
        else:
            chunks[-1] = (chunks[-1][0], run_end)
        return [(round(s, 3), round(e, 3)) for s, e in chunks]


def read_wav(path: str | Path) -> Tuple[np.ndarray, int]:
    """Read a PCM WAV into a mono float32 array in [-1, 1].

    Raises
    ------
    ValueError
        If the file is not uncompressed PCM (e.g. float or compressed WAV).
    """
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            width = wav.getsampwidth()
            sr = wav.getframerate()
            raw = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as exc:
        raise ValueError(f"Unsupported WAV encoding ({exc}). Re-export as 16-bit PCM.") from exc

    if width == 1:
        data = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, np.uint8).reshape(-1, 3).astype(np.int32)
        ints = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        ints = np.where(ints >= 1 << 23, ints - (1 << 24), ints)
        data = ints.astype(np.float32) / float(1 << 23)
    elif width == 4:
        data = np.frombuffer(raw, "<i4").astype(np.float32) / float(1 << 31)
    else:
        raise ValueError(f"Unsupported sample width: {width} bytes")

    if channels > 1:
        data = data[: len(data) - len(data) % channels].reshape(-1, channels).mean(axis=1)
    return data.astype(np.float32), sr


def wav_duration(path: str | Path) -> Optional[float]:
    """Duration of a WAV file in seconds, or *None* if it can't be read."""
    try:
        with wave.open(str(path), "rb") as wav:
            rate = wav.getframerate()
            return wav.getnframes() / rate if rate else None
    except (wave.Error, EOFError, OSError):
        return None


# ===========================================================================
# Text utilities (script-aware — Python's \w splits Indic words at matras)
# ===========================================================================
_SENTENCE_RE = re.compile(r"(?<=[.!?।॥])\s+|\n+")
_TOKEN_RE = re.compile(r"[^\s.,;:!?।॥()\[\]{}\"“”‘’'«»—–/|]+")
_DELIMS = r"\s.,;:!?।॥()\[\]{}\"“”‘’'«»—–/|"

_STOPWORDS = set(
    """
    a about above after again against all also am an and any are as at be because been
    before being below between both but by can could did do does doing down during each
    few for from further had has have having he her here hers him his how i if in into is
    it its itself just let like me more most my no nor not now of off on once only or other
    our out over own same she should so some such than that the their them then there these
    they this those through to too under until up very was we were what when where which
    while who whom why will with would you your yours okay today class lecture let's going
    get got one two three us use used using called means mean way thing things lot really
    so-called basically actually next see look
    है हैं था थे थी हो होता होती होते होगा हुआ हुए के का की को में से पर और या भी तो ही
    एक यह वह ये वो इस उस इन उन इसे उसे जो कि जब तक तब कर करता करते करती करना किया किए
    लिए साथ बहुत सबसे अब आज हम आप हमारा हमारे हमें आपको कोई कुछ यानी इसलिए अगर नहीं
    दोस्तों नमस्ते जाता जाती जाते रहे रहा रही बात बारे सकता सकते देखते कहते होती
    करेंगे करेगा करेगी होगी होंगे बताता बताते समझते चाहिए वाले वाली वाला ओर तरफ
    ஒரு இது அது மற்றும் என்று என்ற இந்த அந்த உள்ள ஆகும் ஆனால் போது
    ಒಂದು ಈ ಆ ಮತ್ತು ಎಂದು ಇದು ಅದು ಆಗಿದೆ ಇರುವ ಹಾಗೂ
    """.split()
)


def split_sentences(text: str) -> List[str]:
    """Split on sentence punctuation (incl. the Devanagari danda) and newlines."""
    return [s.strip() for s in _SENTENCE_RE.split(text) if s and len(s.strip()) > 1]


def tokenize(text: str) -> List[str]:
    """Script-aware word tokens (keeps Indic vowel signs attached)."""
    return [t for t in _TOKEN_RE.findall(text) if any(ch.isalpha() for ch in t)]


def _norm(token: str) -> str:
    return token.casefold().strip("-’'")


def _is_content(token: str) -> bool:
    norm = _norm(token)
    if norm in _STOPWORDS or norm.isdigit():
        return False
    is_latin = all(ord(ch) < 0x0250 for ch in norm)
    return len(norm) >= (4 if is_latin else 3)


def key_terms(text: str, k: int = 12) -> List[Tuple[str, float]]:
    """Rank salient unigrams and repeated bigrams by frequency.

    Returns ``[(surface_form, weight), …]`` with the first-seen surface form
    of each term (so casing and script are preserved).
    """
    tokens = tokenize(text)
    surface: Dict[str, str] = {}
    unigrams: Counter = Counter()
    bigrams: Counter = Counter()

    prev: Optional[str] = None
    for tok in tokens:
        norm = _norm(tok)
        if _is_content(tok):
            unigrams[norm] += 1
            surface.setdefault(norm, tok)
            if prev is not None:
                pair = f"{prev} {norm}"
                bigrams[pair] += 1
                surface.setdefault(pair, f"{surface[prev]} {tok}")
            prev = norm
        else:
            prev = None

    scored: Dict[str, float] = {}
    for term, count in unigrams.items():
        scored[term] = count * (1.0 + min(len(term), 12) / 12.0)
    for pair, count in bigrams.items():
        if count >= 2:
            scored[pair] = count * 2.6
            for part in pair.split():  # a strong phrase absorbs its parts
                if part in scored:
                    scored[part] *= 0.55

    ranked = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
    return [(surface[t], w) for t, w in ranked[:k]]


def _sentence_scores(sentences: List[str], weights: Dict[str, float]) -> List[float]:
    scores = []
    for i, sent in enumerate(sentences):
        toks = [_norm(t) for t in tokenize(sent)]
        if not toks:
            scores.append(0.0)
            continue
        if sum(1 for t in toks if _is_content(t)) < 2:
            scores.append(0.0)  # greetings and fillers ("नमस्ते दोस्तों।")
            continue
        joined = " ".join(toks)
        score = sum(w for term, w in weights.items() if term in joined)
        score /= max(len(toks), 1) ** 0.35
        if i == 0:
            score *= 1.15  # lecturers usually state the topic up front
        scores.append(score)
    return scores


def _contains_term(sentence: str, term: str) -> bool:
    return _term_pattern(term).search(sentence) is not None


def _term_pattern(term: str) -> re.Pattern:
    return re.compile(
        rf"(?<![^{_DELIMS}]){re.escape(term)}(?![^{_DELIMS}])", re.IGNORECASE
    )


def _trim(text: str, limit: int = 170) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return cut.rstrip(",;:—– ") + "…"


def _title_case(term: str) -> str:
    return " ".join(w[:1].upper() + w[1:] if w.isascii() else w for w in term.split())


# ===========================================================================
# Extractive notes
# ===========================================================================
def generate_extractive_notes(transcript: str, title: Optional[str] = None) -> str:
    """Build structured Markdown notes by extracting the most salient sentences.

    Sections: Overview → Key Concepts → Lecture Flow → Key Takeaways, matching
    the layout the LLM prompt asks Param-1 for.
    """
    sentences = split_sentences(transcript)
    if not sentences:
        return "_No transcript provided._"

    terms = key_terms(transcript, k=14)
    weights = {_norm(t): w for t, w in terms}
    scores = _sentence_scores(sentences, weights)
    ranked = sorted(range(len(sentences)), key=lambda i: -scores[i])

    # The topic is usually announced in the opening lines: prefer the
    # strongest key term that appears there.
    opening = " ".join(sentences[:2])
    topic = next((t for t, _ in terms[:8] if _contains_term(opening, t)), None)
    topic = topic or (terms[0][0] if terms else None)
    heading = title or (f"{_title_case(topic)}: Lecture Notes" if topic else "Lecture Notes")
    lines: List[str] = [f"# {heading}", ""]

    if len(sentences) < 4:
        # Too little text to summarise: anything more would just repeat it.
        lines += ["## Overview", " ".join(sentences), ""]
        if terms:
            lines += ["## Key Terms", *(f"- **{t}**" for t, _ in terms[:5])]
        return "\n".join(lines).strip() + "\n"

    overview = sorted(ranked[: min(2, len(ranked))])
    lines += ["## Overview", " ".join(sentences[i] for i in overview), ""]
    used = set(overview)

    concepts: List[str] = []
    concept_lines: List[str] = []
    concept_used: set = set()
    for term, _ in terms:
        if len(concepts) >= 6:
            break
        if any(_norm(term) in _norm(c) or _norm(c) in _norm(term) for c in concepts):
            continue
        holders = [i for i in range(len(sentences)) if _contains_term(sentences[i], term)]
        if not holders:
            continue
        best = min(holders, key=lambda i: (i in concept_used, i in used, len(sentences[i])))
        concept_used.add(best)
        concepts.append(term)
        concept_lines.append(f"- **{term}** — {_trim(sentences[best])}")
    if concept_lines:
        lines += ["## Key Concepts", *concept_lines, ""]

    if len(sentences) >= 4:
        parts = min(4, len(sentences) // 2)
        size = math.ceil(len(sentences) / parts)
        lines.append("## Lecture Flow")
        for p in range(parts):
            idxs = list(range(p * size, min((p + 1) * size, len(sentences))))
            if not idxs:
                continue
            best = max(idxs, key=lambda i: scores[i])
            used.add(best)
            lines.append(f"{p + 1}. {_trim(sentences[best], 200)}")
        lines.append("")

    takeaways = [i for i in ranked if i not in used and scores[i] > 0][:4] or ranked[:3]
    lines.append("## Key Takeaways")
    lines += [f"- {_trim(sentences[i], 200)}" for i in sorted(takeaways)]
    if concepts:
        lines.append(f"- Terms to revise: {', '.join(f'**{c}**' for c in concepts[:5])}.")
    return "\n".join(lines).strip() + "\n"


# ===========================================================================
# Cloze quiz
# ===========================================================================
def generate_cloze_quiz(
    transcript: str,
    num_questions: int = 5,
    seed: int = 7,
) -> List[Dict[str, object]]:
    """Generate fill-in-the-blank MCQs from salient sentences.

    The blank is a key term; distractors are other key terms from the same
    lecture (preferring similar length), so every option is plausible.
    Output already follows the canonical quiz schema.
    """
    sentences = split_sentences(transcript)
    terms = [t for t, _ in key_terms(transcript, k=30)]
    if len(terms) < 4 or not sentences:
        return []

    weights = {_norm(t): 1.0 for t in terms}
    scores = _sentence_scores(sentences, weights)
    order = sorted(range(len(sentences)), key=lambda i: -scores[i])
    rng = random.Random(seed)

    quiz: List[Dict[str, object]] = []
    used_terms: set = set()
    for idx in order:
        if len(quiz) >= num_questions:
            break
        sent = sentences[idx]
        if not 5 <= len(tokenize(sent)) <= 60:
            continue
        candidates = [
            t
            for t in terms
            if _contains_term(sent, t)
            and not any(_norm(t) in u or u in _norm(t) for u in used_terms)
        ]
        if not candidates:
            continue
        answer = max(candidates, key=len)
        pool = [
            t
            for t in terms
            if _norm(t) != _norm(answer)
            and not _contains_term(sent, t)
            and _norm(t) not in _norm(answer)
            and _norm(answer) not in _norm(t)
        ]
        if len(pool) < 3:
            continue
        pool.sort(key=lambda t: (abs(len(t) - len(answer)), rng.random()))
        options = [answer] + rng.sample(pool[:6], 3)
        rng.shuffle(options)

        cloze = _term_pattern(answer).sub("_____", sent, count=1)
        quiz.append(
            {
                "question": f"Fill in the blank: “{_trim(cloze, 240)}”",
                "options": options,
                "answer_index": options.index(answer),
                "correct_answer": answer,
                "explanation": f"From the lecture: “{_trim(sent, 240)}”",
            }
        )
        used_terms.add(_norm(answer))
    return quiz
