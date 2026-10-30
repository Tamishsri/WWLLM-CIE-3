"""
engines.py — Pipeline orchestration for the Studio API
======================================================
``PipelineHub`` owns every model, loads each one lazily on first use and
falls back to the offline engines in ``src.pipelines.fallback`` when a
stage's dependencies are missing.

Every operation is a generator of event dicts, which the API streams to
the browser as NDJSON so the UI can show live progress::

    {"event": "stage",    "stage": "vad", "status": "start", "label": "…"}
    {"event": "progress", "stage": "asr", "value": 0.5, "detail": "…"}
    {"event": "log",      "level": "info", "message": "…"}
    {"event": "token",    "text": "…"}                  # streamed notes
    {"event": "result",   "data": {…}}
    {"event": "error",    "message": "…", "hint": "…"}

Result ``mode`` tells the UI where content came from:
``live`` (real model), ``fallback`` (offline heuristic), ``sample``
(pre-rendered sample lecture) or ``replay`` (evaluation-set prediction).
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from loguru import logger

from src.pipelines.fallback import (
    EnergyVAD,
    generate_cloze_quiz,
    generate_extractive_notes,
)
from src.pipelines.languages import ASR_LANGUAGES, NATIVE_NAMES, SUPPORTED_TARGETS
from src.pipelines.quiz_schema import normalise_quiz
from src.services.capabilities import Capabilities
from src.services.sample import SampleLecture

Event = Dict[str, Any]

DEFAULT_ASR_TEMPLATE = "ai4bharat/indicconformer_stt_{iso}_hybrid_rnnt_large"


# ---------------------------------------------------------------------------
# Event constructors
# ---------------------------------------------------------------------------
def _stage(stage: str, status: str, label: str, **extra: Any) -> Event:
    return {"event": "stage", "stage": stage, "status": status, "label": label, **extra}


def _progress(stage: str, value: float, detail: str = "", **extra: Any) -> Event:
    return {
        "event": "progress",
        "stage": stage,
        "value": round(max(0.0, min(1.0, value)), 4),
        "detail": detail,
        **extra,
    }


def _log(message: str, level: str = "info") -> Event:
    return {"event": "log", "level": level, "message": message}


def _error(message: str, hint: str = "") -> Event:
    return {"event": "error", "message": message, "hint": hint}


def _result(data: Dict[str, Any]) -> Event:
    return {"event": "result", "data": data}


def _pieces(text: str) -> List[str]:
    """Split text into word-sized pieces for simulated token streaming."""
    return re.findall(r"\S+\s*|\s+", text)


def _distribute(text: str, spans: List[Tuple[float, float]]) -> List[str]:
    """Spread a transcript's words across chunks in proportion to duration."""
    words = text.split()
    if len(spans) <= 1:
        return [text]
    total = sum(e - s for s, e in spans) or 1.0
    out, cursor, acc = [], 0, 0.0
    for i, (s, e) in enumerate(spans):
        acc += (e - s) / total
        end = len(words) if i == len(spans) - 1 else round(acc * len(words))
        out.append(" ".join(words[cursor:end]))
        cursor = end
    return out


# ---------------------------------------------------------------------------
# PipelineHub
# ---------------------------------------------------------------------------
class PipelineHub:
    """Lazily-loaded models plus offline fallbacks behind one streaming API.

    Parameters
    ----------
    cfg : dict
        Parsed ``configs/config.yaml``.
    caps : Capabilities
        Result of :func:`~src.services.capabilities.detect_capabilities`.
    sample : SampleLecture
        The built-in sample lecture.
    eval_predictions : dict
        ``{audio_file: transcript}`` from the evaluation set, replayed when
        one of its clips is uploaded without a live ASR model.
    demo_pace : float
        Multiplier for the small delays that let offline stages animate.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        caps: Capabilities,
        sample: SampleLecture,
        eval_predictions: Optional[Dict[str, str]] = None,
        demo_pace: float = 1.0,
    ) -> None:
        self.cfg = cfg
        self.caps = caps
        self.sample = sample
        self.eval_predictions = eval_predictions or {}
        self.demo_pace = demo_pace

        audio_cfg = cfg.get("audio", {})
        self.energy_vad = EnergyVAD(
            max_chunk_sec=audio_cfg.get("max_chunk_sec", 25.0),
            min_chunk_sec=audio_cfg.get("min_chunk_sec", 2.0),
            min_silence_ms=audio_cfg.get("min_silence_ms", 300),
        )

        self._load_lock = threading.Lock()
        # Plain Lock (not RLock): streaming generators may resume on a
        # different worker thread than the one that acquired it.
        self._infer_lock = threading.Lock()
        self._audio = None
        self._asr = None
        self._asr_model_id: Optional[str] = None
        self._llm = None
        self._mt = None

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def asr_model_id(self, language: str) -> str:
        """Checkpoint for *language*: ``asr.models`` override, else the template."""
        asr_cfg = self.cfg.get("asr", {})
        overrides = asr_cfg.get("models") or {}
        if language in overrides:
            return overrides[language]
        iso = ASR_LANGUAGES.get(language, "hi")
        default = asr_cfg.get("model_id", "")
        if default and f"_{iso}_" in default:
            return default
        return asr_cfg.get("model_template", DEFAULT_ASR_TEMPLATE).format(iso=iso)

    def status(self) -> Dict[str, Any]:
        caps = self.caps

        def stage(key: str, name: str, model: str, fallback: str) -> Dict[str, Any]:
            return {
                "live": caps.live(key),
                "name": name,
                "model": model,
                "fallback": fallback,
                "missing": caps.missing(key),
            }

        llm_id = self.cfg.get("llm", {}).get("model_id", "bharatgenai/Param-1-2.9B-Instruct")
        mt_id = self.cfg.get("translation", {}).get("model_id", "ai4bharat/indictrans2-en-indic-1B")
        sample_langs = set(self.sample.translated_languages)
        return {
            "app": {
                "name": "WWLLM Lecture Studio",
                "version": self.cfg.get("app", {}).get("version", "2.0.0"),
            },
            "mode": caps.mode,
            "forced_demo": caps.forced_demo,
            "device": caps.device,
            "stages": {
                "vad": stage("vad", "Silero-VAD", "silero-vad v5", "Energy VAD · NumPy · .wav only"),
                "asr": stage("asr", "IndicConformer", self.asr_model_id("Hindi"), "Sample lecture & eval-set replay"),
                "llm": stage("llm", "BharatGen Param-1", llm_id, "Extractive notes + cloze quiz"),
                "mt": stage("mt", "IndicTrans2", mt_id, "Pre-rendered sample translations"),
            },
            "languages": {
                "asr": [
                    {"name": n, "native": NATIVE_NAMES.get(n, n), "iso": iso}
                    for n, iso in ASR_LANGUAGES.items()
                ],
                "translate": [
                    {
                        "name": n,
                        "native": NATIVE_NAMES.get(n, n),
                        "code": code,
                        "sample": n in sample_langs,
                    }
                    for n, code in SUPPORTED_TARGETS.items()
                ],
            },
            "sample": {
                "title": self.sample.title,
                "language": self.sample.language,
                "duration": self.sample.duration,
            },
            "loaded": {
                "vad": self._audio is not None,
                "asr": self._asr_model_id,
                "llm": self._llm is not None,
                "mt": self._mt is not None,
            },
        }

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------
    def _audio_processor(self):
        with self._load_lock:
            if self._audio is None:
                from src.pipelines.audio_utils import AudioProcessor

                audio_cfg = self.cfg.get("audio", {})
                self._audio = AudioProcessor(
                    max_chunk_sec=audio_cfg.get("max_chunk_sec", 25.0),
                    min_chunk_sec=audio_cfg.get("min_chunk_sec", 2.0),
                    vad_threshold=audio_cfg.get("vad_threshold", 0.45),
                    min_silence_ms=audio_cfg.get("min_silence_ms", 300),
                )
            return self._audio

    def _asr_engine(self, model_id: str):
        with self._load_lock:
            if self._asr is not None and self._asr_model_id == model_id:
                return self._asr
            if self._asr is not None:  # one ~600M checkpoint in memory at a time
                self._asr = None
                self._release_gpu()
            from src.pipelines.asr_engine import ASREngine

            asr_cfg = self.cfg.get("asr", {})
            self._asr = ASREngine(
                model_id=model_id,
                device=asr_cfg.get("device"),
                batch_size=asr_cfg.get("batch_size", 8),
            )
            self._asr_model_id = model_id
            return self._asr

    def _llm_engine(self):
        with self._load_lock:
            if self._llm is None:
                import torch

                from src.pipelines.llm_engine import LLMEngine

                llm_cfg = self.cfg.get("llm", {})
                quantize = (
                    bool(llm_cfg.get("quantize_8bit", True))
                    and self.caps.has("bitsandbytes")
                    and torch.cuda.is_available()
                )
                if llm_cfg.get("quantize_8bit", True) and not quantize:
                    logger.warning("8-bit quantisation needs CUDA + bitsandbytes — loading fp16.")
                self._llm = LLMEngine(
                    model_id=llm_cfg.get("model_id", "bharatgenai/Param-1-2.9B-Instruct"),
                    quantize_8bit=quantize,
                    temperature=llm_cfg.get("temperature", 0.4),
                )
            return self._llm

    def _mt_engine(self):
        with self._load_lock:
            if self._mt is None:
                from src.pipelines.translator import TranslationEngine

                mt_cfg = self.cfg.get("translation", {})
                self._mt = TranslationEngine(
                    model_id=mt_cfg.get("model_id", "ai4bharat/indictrans2-en-indic-1B"),
                    num_beams=mt_cfg.get("num_beams", 5),
                )
            return self._mt

    @staticmethod
    def _release_gpu() -> None:
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _pace(self, seconds: float) -> None:
        """Small delay so offline stages visibly progress in the UI."""
        if self.demo_pace > 0:
            time.sleep(seconds * self.demo_pace)

    # ------------------------------------------------------------------
    # Stage 1 + 2: VAD → ASR
    # ------------------------------------------------------------------
    def transcribe(
        self, record: Dict[str, Any], path: Path, language: str
    ) -> Iterator[Event]:
        """Chunk the audio with VAD, then transcribe each chunk."""
        if record.get("sample"):
            yield from self._transcribe_sample()
            return

        language = language if language in ASR_LANGUAGES else "Hindi"
        vad_live = self.caps.live("vad")
        t0 = time.perf_counter()

        # ── VAD ─────────────────────────────────────────────────────────
        chunks = None
        if vad_live:
            yield _stage("vad", "start", "Detecting speech · Silero-VAD")
            if self._audio is None:
                yield _log("Loading Silero-VAD…")
            chunks, spans = self._audio_processor().process_with_spans(path)
        else:
            yield _stage("vad", "start", "Detecting speech · energy VAD (offline)")
            if path.suffix.lower() != ".wav":
                yield _error(
                    "The offline VAD can only read .wav files.",
                    "Convert the recording to 16-bit WAV, or install torch, "
                    "torchaudio and silero-vad to process MP3/MP4.",
                )
                return
            try:
                spans = self.energy_vad.process_wav(path).spans
            except ValueError as exc:
                yield _error(str(exc), "Re-export the file as 16-bit PCM WAV.")
                return
            self._pace(0.5)

        vad_seconds = time.perf_counter() - t0
        yield _stage(
            "vad",
            "done",
            f"{len(spans)} speech chunk{'s' if len(spans) != 1 else ''}",
            chunks=[{"start": s, "end": e} for s, e in spans],
            seconds=round(vad_seconds, 2),
        )

        # ── ASR ─────────────────────────────────────────────────────────
        t1 = time.perf_counter()
        if self.caps.live("asr") and chunks is not None:
            model_id = self.asr_model_id(language)
            if self._asr_model_id != model_id:
                yield _log(f"Loading {model_id} (first run downloads the checkpoint)…")
            asr = self._asr_engine(model_id)
            yield _stage("asr", "start", f"Transcribing · IndicConformer ({language})")
            texts: List[str] = []
            for i in range(0, len(chunks), asr.batch_size):
                with self._infer_lock:
                    texts.extend(asr.transcribe(chunks[i : i + asr.batch_size]))
                done = min(i + asr.batch_size, len(chunks))
                yield _progress("asr", done / len(chunks), f"Chunk {done}/{len(chunks)}", chunk=done - 1)
            engine, mode = model_id, "live"
        else:
            reference = self.eval_predictions.get(record.get("name", ""))
            if reference is None:
                yield _error(
                    "IndicConformer isn't installed, so new audio can't be transcribed here.",
                    "Paste or type the transcript to keep going, or install the "
                    "NeMo ASR stack (see README) for live transcription.",
                )
                return
            yield _stage("asr", "start", "Replaying the stored evaluation prediction")
            yield _log(f"{record['name']} is in the benchmark set — replaying its stored, simulated prediction.")
            texts = _distribute(reference, spans)
            for i in range(len(spans)):
                self._pace(0.35)
                yield _progress("asr", (i + 1) / len(spans), f"Chunk {i + 1}/{len(spans)}", chunk=i)
            engine, mode = "Evaluation-set replay", "replay"

        segments = [
            {"start": s, "end": e, "text": t.strip()} for (s, e), t in zip(spans, texts)
        ]
        transcript = " ".join(seg["text"] for seg in segments if seg["text"])
        asr_seconds = time.perf_counter() - t1
        yield _stage("asr", "done", f"{len(transcript.split())} words", seconds=round(asr_seconds, 2))
        yield _result(
            {
                "transcript": transcript,
                "segments": segments,
                "chunks": [{"start": s, "end": e} for s, e in spans],
                "language": language,
                "engine": engine,
                "mode": mode,
                "timings": {"vad": round(vad_seconds, 2), "asr": round(asr_seconds, 2)},
            }
        )

    def _transcribe_sample(self) -> Iterator[Event]:
        sample = self.sample
        yield _stage("vad", "start", "Detecting speech in the sample lecture")
        self._pace(0.6)
        chunks = [{"start": s, "end": e} for s, e in sample.spans]
        yield _stage("vad", "done", f"{len(chunks)} speech chunks", chunks=chunks, seconds=0.6)

        yield _stage("asr", "start", "Loading the pre-written sample transcript")
        yield _log("The sample's audio is synthetic, so its transcript is pre-written.")
        for i in range(len(chunks)):
            self._pace(0.45)
            yield _progress("asr", (i + 1) / len(chunks), f"Chunk {i + 1}/{len(chunks)}", chunk=i)
        yield _stage("asr", "done", f"{len(sample.transcript.split())} words", seconds=1.8)
        yield _result(
            {
                "transcript": sample.transcript,
                "segments": sample.segments,
                "chunks": chunks,
                "language": sample.language,
                "engine": "Pre-rendered sample",
                "mode": "sample",
                "timings": {"vad": 0.6, "asr": 1.8},
            }
        )

    # ------------------------------------------------------------------
    # Stage 3: notes
    # ------------------------------------------------------------------
    def notes(self, transcript: str, title: Optional[str] = None) -> Iterator[Event]:
        """Stream structured Markdown notes token by token."""
        transcript = (transcript or "").strip()
        if not transcript:
            yield _error("The transcript is empty.", "Transcribe the audio or paste a transcript first.")
            return

        t0 = time.perf_counter()
        if self.caps.live("llm"):
            if self._llm is None:
                yield _log("Loading BharatGen Param-1 (first run downloads the checkpoint)…")
            llm = self._llm_engine()
            yield _stage("notes", "start", "Writing notes · BharatGen Param-1")
            pieces: List[str] = []
            with self._infer_lock:
                for piece in llm.stream_notes(transcript):
                    pieces.append(piece)
                    yield {"event": "token", "text": piece}
            notes = "".join(pieces).strip()
            engine, mode = llm.model_id, "live"
        else:
            if self.sample.matches_transcript(transcript):
                notes, engine, mode = self.sample.notes, "Pre-rendered sample", "sample"
                yield _stage("notes", "start", "Loading the sample lecture's notes")
            else:
                notes = generate_extractive_notes(transcript, title)
                engine, mode = "Extractive summariser (offline)", "fallback"
                yield _stage("notes", "start", "Writing notes · offline extractive summariser")
            for piece in _pieces(notes):
                self._pace(0.012)
                yield {"event": "token", "text": piece}

        seconds = time.perf_counter() - t0
        yield _stage("notes", "done", f"{len(notes.split())} words", seconds=round(seconds, 2))
        yield _result({"notes": notes, "engine": engine, "mode": mode, "seconds": round(seconds, 2)})

    # ------------------------------------------------------------------
    # Stage 4: quiz
    # ------------------------------------------------------------------
    def quiz(self, transcript: str, num_questions: int = 5) -> Iterator[Event]:
        """Generate multiple-choice questions in the canonical schema."""
        transcript = (transcript or "").strip()
        if not transcript:
            yield _error("The transcript is empty.", "Transcribe the audio or paste a transcript first.")
            return
        n = max(3, min(10, int(num_questions or 5)))

        t0 = time.perf_counter()
        if self.caps.live("llm"):
            if self._llm is None:
                yield _log("Loading BharatGen Param-1…")
            llm = self._llm_engine()
            yield _stage("quiz", "start", f"Writing {n} questions · BharatGen Param-1")
            with self._infer_lock:
                quiz = llm.generate_quiz(transcript, num_questions=n)
            if not quiz:
                yield _error(
                    "Param-1 didn't return a valid quiz this time.",
                    "Sampling is stochastic — regenerating usually fixes it.",
                )
                return
            engine, mode = llm.model_id, "live"
        else:
            if self.sample.matches_transcript(transcript):
                quiz = self.sample.quiz(n)
                engine, mode = "Pre-rendered sample", "sample"
                yield _stage("quiz", "start", "Loading the sample lecture's quiz")
                if n > len(quiz):
                    yield _log(f"The sample ships {len(quiz)} pre-written questions.")
            else:
                yield _stage("quiz", "start", f"Writing {n} questions · offline cloze generator")
                quiz = generate_cloze_quiz(transcript, n)
                engine, mode = "Cloze generator (offline)", "fallback"
                if not quiz:
                    yield _error(
                        "Not enough distinct key terms to build a quiz.",
                        "Try a longer transcript, or install the LLM stack for abstractive questions.",
                    )
                    return
            for i in range(len(quiz)):
                self._pace(0.25)
                yield _progress("quiz", (i + 1) / len(quiz), f"Question {i + 1}/{len(quiz)}")

        seconds = time.perf_counter() - t0
        yield _stage("quiz", "done", f"{len(quiz)} questions", seconds=round(seconds, 2))
        yield _result({"quiz": quiz, "engine": engine, "mode": mode, "seconds": round(seconds, 2)})

    # ------------------------------------------------------------------
    # Stage 5: translation
    # ------------------------------------------------------------------
    def translate(
        self,
        notes: str,
        quiz: List[Dict[str, Any]],
        target: str,
    ) -> Iterator[Event]:
        """Translate notes and quiz into *target* (an IndicTrans2 language)."""
        if target not in SUPPORTED_TARGETS:
            yield _error(f"Unsupported target language: {target}.")
            return
        notes = (notes or "").strip()
        quiz = normalise_quiz(quiz or [])
        if not notes and not quiz:
            yield _error("Nothing to translate yet.", "Generate notes or a quiz first.")
            return

        flores = SUPPORTED_TARGETS[target]
        t0 = time.perf_counter()
        if self.caps.live("mt"):
            if self._mt is None:
                yield _log("Loading IndicTrans2 (first run downloads the checkpoint)…")
            translator = self._mt_engine()
            yield _stage("translate", "start", f"Translating to {target} · IndicTrans2")
            steps = (1 if notes else 0) + len(quiz)
            done = 0
            out_notes = ""
            if notes:
                with self._infer_lock:
                    out_notes = translator.translate(notes, target_lang=flores)
                done += 1
                yield _progress("translate", done / steps, "Notes translated")
            out_quiz: List[Dict[str, Any]] = []
            for q in quiz:
                with self._infer_lock:
                    out_quiz.extend(translator.translate_quiz([q], target_lang=flores))
                done += 1
                yield _progress("translate", done / steps, f"Question {done - (1 if notes else 0)}/{len(quiz)}")
            out_quiz = normalise_quiz(out_quiz)
            engine, mode = translator.model_id, "live"
        else:
            curated = self.sample.translation(target)
            notes_ok = not notes or self.sample.matches_notes(notes)
            quiz_ok = not quiz or self.sample.matches_quiz(quiz)
            if not (curated and notes_ok and quiz_ok):
                langs = ", ".join(self.sample.translated_languages)
                sample_hint = (
                    f" The sample lecture includes pre-rendered {langs}."
                    if not (notes_ok and quiz_ok)
                    else f" The sample lecture is pre-rendered in {langs} only."
                )
                yield _error(
                    "IndicTrans2 isn't installed, so live translation is unavailable.",
                    "Install transformers + IndicTransToolkit (see README)." + sample_hint,
                )
                return
            yield _stage("translate", "start", f"Loading the sample's {target} translation")
            out_notes = curated["notes"] if notes else ""
            out_quiz = [
                {
                    **tq,
                    "answer_index": src["answer_index"],
                    "correct_answer": tq["options"][src["answer_index"]],
                }
                for tq, src in zip(curated["quiz"], quiz)
            ]
            for i in range(4):
                self._pace(0.3)
                yield _progress("translate", (i + 1) / 4, "Translating")
            engine, mode = "Pre-rendered sample", "sample"

        seconds = time.perf_counter() - t0
        yield _stage("translate", "done", f"{target} ready", seconds=round(seconds, 2))
        yield _result(
            {
                "target": target,
                "native": NATIVE_NAMES.get(target, target),
                "code": flores,
                "notes": out_notes,
                "quiz": out_quiz,
                "engine": engine,
                "mode": mode,
                "seconds": round(seconds, 2),
            }
        )


def load_eval_predictions(path: Path) -> Dict[str, str]:
    """Read ``{audio_file: transcript}`` (or the richer eval_inputs format)."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("predictions"), dict):
        data = data["predictions"]
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}
