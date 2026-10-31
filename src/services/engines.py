"""
engines.py — Pipeline orchestration for the Studio API
======================================================
``PipelineHub`` runs each stage on the best thing available, in this order:

    1. A stored result, if this exact input has been processed before.
    2. The real model, loaded on first use.
    3. An offline engine from ``src.pipelines.fallback``.

A model that fails to load (not downloaded, no network, access not granted)
drops the stage to the next option instead of failing the request.

Every operation is a generator of event dicts, which the API streams to
the browser as NDJSON so the UI can show live progress::

    {"event": "stage",    "stage": "vad", "status": "start", "label": "…"}
    {"event": "progress", "stage": "asr", "value": 0.5, "detail": "…"}
    {"event": "log",      "level": "info", "message": "…"}
    {"event": "token",    "text": "…"}                  # streamed notes
    {"event": "result",   "data": {…}}
    {"event": "error",    "message": "…", "hint": "…"}

Result ``mode`` tells the UI where content came from: ``live`` (a real model,
with ``cached: true`` if replayed), ``fallback`` (offline heuristic) or
``replay`` (a stored evaluation prediction).
"""

from __future__ import annotations

import gc
import json
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from loguru import logger

from src.pipelines.fallback import (
    EnergyVAD,
    generate_cloze_quiz,
    generate_extractive_notes,
)
from src.pipelines.languages import (
    ASR_LANGUAGES,
    NATIVE_NAMES,
    SUPPORTED_TARGETS,
    WHISPER_MODELS,
)
from src.pipelines.quiz_schema import normalise_quiz
from src.services.capabilities import Capabilities
from src.services.result_cache import ResultCache, file_digest
from src.services.sample import SampleLecture

Event = Dict[str, Any]

DEFAULT_ASR_TEMPLATE = "ai4bharat/indicconformer_stt_{iso}_hybrid_rnnt_large"
DEFAULT_LLM = "bharatgenai/Param-1-2.9B-Instruct"
DEFAULT_MT = "ai4bharat/indictrans2-en-indic-1B"
DEFAULT_OPEN_MT = "facebook/nllb-200-distilled-600M"


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
    """Split text into word-sized pieces for replayed token streaming."""
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


def _norm(text: str) -> str:
    return " ".join((text or "").split())


def _complete_sentences(text: str) -> str:
    """Drop a last sentence cut off by the model's token limit."""
    lines = text.rstrip().split("\n")
    last = lines[-1]
    if last.lstrip().startswith("#") or re.search(r"[.!?।॥:)\"*]$", last):
        return "\n".join(lines)
    ends = [m.end() for m in re.finditer(r"[.!?।॥](?=\s)", last)]
    if ends:
        lines[-1] = last[: ends[-1]]
    else:
        lines.pop()
    while lines and (not lines[-1].strip() or lines[-1].lstrip().startswith("#")):
        lines.pop()  # a heading left with nothing under it
    return "\n".join(lines) or text.rstrip()  # very short notes are kept as they are


# ---------------------------------------------------------------------------
# Engine construction (shared with the Streamlit UI)
# ---------------------------------------------------------------------------
def asr_model_id(cfg: Dict[str, Any], language: str) -> str:
    """IndicConformer checkpoint for *language*: ``asr.models`` override, else the template."""
    asr_cfg = cfg.get("asr", {})
    overrides = asr_cfg.get("models") or {}
    if language in overrides:
        return overrides[language]
    iso = ASR_LANGUAGES.get(language, "hi")
    default = asr_cfg.get("model_id", "")
    if default and f"_{iso}_" in default:
        return default
    return asr_cfg.get("model_template", DEFAULT_ASR_TEMPLATE).format(iso=iso)


def build_asr(cfg: Dict[str, Any], caps: Capabilities, language: str):
    """IndicConformer when NeMo and the checkpoint are available, else open Whisper."""
    asr_cfg = cfg.get("asr", {})
    if caps.has("nemo"):
        try:
            from src.pipelines.asr_engine import ASREngine

            return ASREngine(
                model_id=asr_model_id(cfg, language),
                device=asr_cfg.get("device"),
                batch_size=asr_cfg.get("batch_size", 8),
            )
        except Exception as exc:  # gated checkpoint, broken NeMo install
            logger.warning(f"IndicConformer unavailable ({exc}); using Whisper.")

    from src.pipelines.whisper_engine import WhisperEngine

    return WhisperEngine(language, device=asr_cfg.get("device"))


def build_llm(cfg: Dict[str, Any], caps: Capabilities):
    import torch

    from src.pipelines.llm_engine import LLMEngine

    llm_cfg = cfg.get("llm", {})
    limit = "max_new_tokens_{}" if torch.cuda.is_available() else "cpu_max_new_tokens_{}"
    wanted = bool(llm_cfg.get("quantize_8bit", True))
    quantize = wanted and caps.has("bitsandbytes") and torch.cuda.is_available()
    if wanted and not quantize:
        logger.info("8-bit quantisation needs CUDA + bitsandbytes; loading unquantised.")
    return LLMEngine(
        model_id=llm_cfg.get("model_id", DEFAULT_LLM),
        quantize_8bit=quantize,
        temperature=llm_cfg.get("temperature", 0.4),
        max_new_tokens_notes=llm_cfg.get(limit.format("notes"), 2048),
        max_new_tokens_quiz=llm_cfg.get(limit.format("quiz"), 2048),
    )


def build_translator(cfg: Dict[str, Any], caps: Capabilities):
    """IndicTrans2 when its toolkit and weights are available, else open NLLB-200."""
    from src.pipelines.translator import NLLBEngine, TranslationEngine

    mt_cfg = cfg.get("translation", {})
    if caps.has("IndicTransToolkit"):
        try:
            return TranslationEngine(
                model_id=mt_cfg.get("model_id", DEFAULT_MT),
                num_beams=mt_cfg.get("num_beams", 5),
            )
        except Exception as exc:  # gated checkpoint
            logger.warning(f"IndicTrans2 unavailable ({exc}); using NLLB-200.")
    return NLLBEngine(model_id=mt_cfg.get("open_model_id", DEFAULT_OPEN_MT))


# ---------------------------------------------------------------------------
# PipelineHub
# ---------------------------------------------------------------------------
class PipelineHub:
    """Stored results, lazily-loaded models and offline fallbacks behind one API.

    Parameters
    ----------
    cfg : dict
        Parsed ``configs/config.yaml``.
    caps : Capabilities
        Result of :func:`~src.services.capabilities.detect_capabilities`.
    sample : SampleLecture
        The built-in sample lecture.
    cache : ResultCache
        Where model outputs are stored and replayed from.
    eval_predictions : dict
        ``{audio_file: transcript}`` from the evaluation set, replayed when
        one of its clips is uploaded without a working ASR model.
    demo_pace : float
        Multiplier for the small delays that let instant stages animate.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        caps: Capabilities,
        sample: SampleLecture,
        cache: ResultCache,
        eval_predictions: Optional[Dict[str, str]] = None,
        demo_pace: float = 1.0,
    ) -> None:
        self.cfg = cfg
        self.caps = caps
        self.sample = sample
        self.cache = cache
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
        self._models: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def status(self) -> Dict[str, Any]:
        caps = self.caps

        def stage(key: str, name: str, model: str, fallback: str, note: str = "") -> Dict[str, Any]:
            return {
                "live": caps.live(key),
                "name": name,
                "model": model,
                "fallback": fallback,
                "missing": caps.missing(key),
                "note": note,
            }

        nemo = caps.has("nemo")
        toolkit = caps.has("IndicTransToolkit")
        mt_cfg = self.cfg.get("translation", {})
        return {
            "app": {
                "name": "WWLLM Lecture Studio",
                "version": self.cfg.get("app", {}).get("version", "2.0.0"),
            },
            "mode": caps.mode,
            "forced_demo": caps.forced_demo,
            "device": caps.device,
            "stages": {
                "vad": stage("vad", "Silero-VAD", "silero-vad v5", "Energy VAD · NumPy"),
                "asr": stage(
                    "asr",
                    "IndicConformer" if nemo else "Whisper",
                    asr_model_id(self.cfg, "Hindi") if nemo else WHISPER_MODELS["Hindi"],
                    "Stored results only",
                    "" if nemo else "IndicConformer is gated: it needs a Hugging Face login plus NeMo.",
                ),
                "llm": stage(
                    "llm",
                    "BharatGen Param-1",
                    self.cfg.get("llm", {}).get("model_id", DEFAULT_LLM),
                    "Extractive notes + cloze quiz",
                ),
                "mt": stage(
                    "mt",
                    "IndicTrans2" if toolkit else "NLLB-200",
                    mt_cfg.get("model_id", DEFAULT_MT) if toolkit else mt_cfg.get("open_model_id", DEFAULT_OPEN_MT),
                    "Stored results only",
                    "" if toolkit else "IndicTrans2 is gated: it needs a Hugging Face login plus IndicTransToolkit.",
                ),
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
                        "sample": n in self.sample.translated_languages,
                    }
                    for n, code in SUPPORTED_TARGETS.items()
                ],
            },
            "sample": {
                "title": self.sample.title,
                "language": self.sample.language,
                "duration": self.sample.duration,
            },
            "loaded": sorted(self._models),
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
                    vad_threshold=audio_cfg.get("vad_threshold", 0.45),
                    min_silence_ms=audio_cfg.get("min_silence_ms", 300),
                )
            return self._audio

    def _model(self, key: str, build: Callable[[], Any]) -> Optional[Any]:
        """Return a loaded model, or *None* if it can't be loaded.

        Only one large model stays in memory: they rarely all fit on a laptop.
        """
        with self._load_lock:
            if key in self._models:
                return self._models[key]
            if self._models:
                self._models.clear()
                self._release_memory()
            try:
                self._models[key] = build()
            except Exception as exc:
                logger.warning(f"Could not load {key}: {exc}")
                return None
            return self._models[key]

    @staticmethod
    def _release_memory() -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _pace(self, seconds: float) -> None:
        """Small delay so instant stages visibly progress in the UI."""
        if self.demo_pace > 0:
            time.sleep(seconds * self.demo_pace)

    def _llm_label(self, label: str) -> str:
        """A stage label that spells out the wait when there's no GPU."""
        return label if self.caps.device.get("cuda") else f"{label} · about 10 minutes without a GPU"

    def _store(self, stage: str, payload: Dict[str, Any], data: Dict[str, Any]) -> None:
        try:
            self.cache.put(stage, payload, data)
        except OSError as exc:  # a full disk shouldn't lose the result on screen
            logger.warning(f"Could not cache {stage}: {exc}")

    # ------------------------------------------------------------------
    # Stage 1 + 2: VAD → ASR
    # ------------------------------------------------------------------
    def transcribe(
        self,
        record: Dict[str, Any],
        path: Path,
        language: str,
        fresh: bool = False,
    ) -> Iterator[Event]:
        """Chunk the audio with VAD, then transcribe each chunk."""
        language = language if language in ASR_LANGUAGES else "Hindi"
        payload = {"audio": file_digest(path), "language": language}
        stored = None if fresh else self.cache.get("asr", payload)
        if stored:
            yield from self._replay_transcript(stored)
            return

        t0 = time.perf_counter()

        # ── VAD ─────────────────────────────────────────────────────────
        chunks = None
        spans: Optional[List[Tuple[float, float]]] = None
        if self.caps.live("vad"):
            yield _stage("vad", "start", "Detecting speech · Silero-VAD")
            if self._audio is None:
                yield _log("Loading Silero-VAD…")
            try:
                chunks, spans = self._audio_processor().process_with_spans(path)
            except Exception as exc:
                yield _log(f"Silero-VAD failed ({exc}); using the energy VAD.", "warning")
        if spans is None:
            yield _stage("vad", "start", "Detecting speech · energy VAD (offline)")
            try:
                spans = self._energy_spans(path)
            except (ValueError, OSError, subprocess.SubprocessError) as exc:
                yield _error(f"Couldn't read this recording: {exc}", "Try re-exporting it as WAV or MP3.")
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
        asr = None
        if self.caps.live("asr") and chunks is not None:
            name = "IndicConformer" if self.caps.has("nemo") else "Whisper"
            if f"asr:{language}" not in self._models:
                yield _log(f"Loading {name} for {language} (the first run downloads it)…")
            asr = self._model(f"asr:{language}", lambda: build_asr(self.cfg, self.caps, language))
            if asr is None:
                yield _log(f"{name} couldn't be loaded.", "warning")

        if asr is not None:
            yield _stage("asr", "start", f"Transcribing {language}")
            texts: List[str] = []
            for i in range(0, len(chunks), asr.batch_size):
                with self._infer_lock:
                    texts.extend(asr.transcribe(chunks[i : i + asr.batch_size]))
                done = min(i + asr.batch_size, len(chunks))
                yield _progress("asr", done / len(chunks), f"Chunk {done}/{len(chunks)}", chunk=done - 1)
            engine, mode = asr.model_id, "live"
        else:
            reference = self.eval_predictions.get(record.get("name", ""))
            if reference is None:
                yield _error(
                    "No speech recognition model is available here.",
                    "Paste or type the transcript to keep going, or install the ML stack (see README).",
                )
                return
            yield _stage("asr", "start", "Replaying the stored evaluation prediction")
            yield _log(f"{record['name']} is in the benchmark set: replaying its stored prediction.")
            texts = _distribute(reference, spans)
            for i in range(len(spans)):
                self._pace(0.35)
                yield _progress("asr", (i + 1) / len(spans), f"Chunk {i + 1}/{len(spans)}", chunk=i)
            engine, mode = "Evaluation-set replay", "replay"

        segments = [{"start": s, "end": e, "text": t.strip()} for (s, e), t in zip(spans, texts)]
        transcript = " ".join(seg["text"] for seg in segments if seg["text"])
        asr_seconds = time.perf_counter() - t1
        yield _stage("asr", "done", f"{len(transcript.split())} words", seconds=round(asr_seconds, 2))
        data = {
            "transcript": transcript,
            "segments": segments,
            "chunks": [{"start": s, "end": e} for s, e in spans],
            "language": language,
            "engine": engine,
            "mode": mode,
            "timings": {"vad": round(vad_seconds, 2), "asr": round(asr_seconds, 2)},
        }
        if mode == "live":
            self._store("asr", payload, data)
        yield _result(data)

    def _energy_spans(self, path: Path) -> List[Tuple[float, float]]:
        """Offline VAD. Formats other than WAV go through ffmpeg first."""
        if path.suffix.lower() == ".wav":
            return self.energy_vad.process_wav(path).spans
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise ValueError("the offline VAD reads .wav only, and ffmpeg isn't installed")
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "audio.wav"
            subprocess.run(
                [ffmpeg, "-nostdin", "-loglevel", "error", "-i", str(path),
                 "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(target)],
                check=True,
                capture_output=True,
            )
            return self.energy_vad.process_wav(target).spans

    def _replay_transcript(self, stored: Dict[str, Any]) -> Iterator[Event]:
        chunks = stored["chunks"]
        timings = stored.get("timings", {})
        yield _stage("vad", "start", "Detecting speech")
        self._pace(0.4)
        yield _stage("vad", "done", f"{len(chunks)} speech chunks", chunks=chunks, seconds=timings.get("vad"))
        yield _stage("asr", "start", f"Stored transcript · {stored.get('engine', 'model')}")
        for i in range(len(chunks)):
            self._pace(0.3)
            yield _progress("asr", (i + 1) / len(chunks), f"Chunk {i + 1}/{len(chunks)}", chunk=i)
        yield _stage(
            "asr",
            "done",
            f"{len(stored['transcript'].split())} words",
            seconds=timings.get("asr"),
        )
        yield _result(stored)

    # ------------------------------------------------------------------
    # Stage 3: notes
    # ------------------------------------------------------------------
    def notes(
        self,
        transcript: str,
        title: Optional[str] = None,
        fresh: bool = False,
    ) -> Iterator[Event]:
        """Stream structured Markdown notes token by token."""
        transcript = (transcript or "").strip()
        if not transcript:
            yield _error("The transcript is empty.", "Transcribe the audio or paste a transcript first.")
            return

        payload = {"transcript": _norm(transcript)}
        stored = None if fresh else self.cache.get("notes", payload)
        if stored:
            yield _stage("notes", "start", f"Stored notes · {stored.get('engine', 'model')}")
            for piece in _pieces(stored["notes"]):
                self._pace(0.012)
                yield {"event": "token", "text": piece}
            yield _stage("notes", "done", f"{len(stored['notes'].split())} words", seconds=stored.get("seconds"))
            yield _result(stored)
            return

        t0 = time.perf_counter()
        notes, engine, mode = "", "", ""
        llm = None
        if self.caps.live("llm"):
            if "llm" not in self._models:
                yield _log("Loading BharatGen Param-1 (the first run downloads it)…")
            llm = self._model("llm", lambda: build_llm(self.cfg, self.caps))
            if llm is None:
                yield _log("Param-1 couldn't be loaded; using the offline summariser.", "warning")

        if llm is not None:
            yield _stage("notes", "start", self._llm_label("Writing notes · BharatGen Param-1"))
            pieces: List[str] = []
            with self._infer_lock:
                for piece in llm.stream_notes(transcript):
                    pieces.append(piece)
                    yield {"event": "token", "text": piece}
            notes = _complete_sentences("".join(pieces).strip())
            engine, mode = llm.model_id, "live"
            if not notes:
                yield _log("Param-1 returned nothing; using the offline summariser.", "warning")

        if not notes:
            notes = generate_extractive_notes(transcript, title)
            engine, mode = "Extractive summariser (offline)", "fallback"
            yield _stage("notes", "start", "Writing notes · offline extractive summariser")
            for piece in _pieces(notes):
                self._pace(0.012)
                yield {"event": "token", "text": piece}

        seconds = time.perf_counter() - t0
        yield _stage("notes", "done", f"{len(notes.split())} words", seconds=round(seconds, 2))
        data = {"notes": notes, "engine": engine, "mode": mode, "seconds": round(seconds, 2)}
        if mode == "live":
            self._store("notes", payload, data)
        yield _result(data)

    # ------------------------------------------------------------------
    # Stage 4: quiz
    # ------------------------------------------------------------------
    def quiz(
        self,
        transcript: str,
        num_questions: int = 5,
        fresh: bool = False,
    ) -> Iterator[Event]:
        """Generate multiple-choice questions in the canonical schema."""
        transcript = (transcript or "").strip()
        if not transcript:
            yield _error("The transcript is empty.", "Transcribe the audio or paste a transcript first.")
            return
        n = max(3, min(10, int(num_questions or 5)))

        payload = {"transcript": _norm(transcript), "n": n}
        stored = None if fresh else self.cache.get("quiz", payload)
        if stored:
            quiz = stored["quiz"]
            yield _stage("quiz", "start", f"Stored quiz · {stored.get('engine', 'model')}")
            for i in range(len(quiz)):
                self._pace(0.2)
                yield _progress("quiz", (i + 1) / len(quiz), f"Question {i + 1}/{len(quiz)}")
            yield _stage("quiz", "done", f"{len(quiz)} questions", seconds=stored.get("seconds"))
            yield _result(stored)
            return

        t0 = time.perf_counter()
        quiz: List[Dict[str, Any]] = []
        engine, mode = "", ""
        llm = None
        if self.caps.live("llm"):
            if "llm" not in self._models:
                yield _log("Loading BharatGen Param-1 (the first run downloads it)…")
            llm = self._model("llm", lambda: build_llm(self.cfg, self.caps))
            if llm is None:
                yield _log("Param-1 couldn't be loaded; using the offline quiz generator.", "warning")

        if llm is not None:
            yield _stage("quiz", "start", self._llm_label(f"Writing {n} questions · BharatGen Param-1"))
            raw, written = "", 0
            with self._infer_lock:
                for piece in llm.stream_quiz(transcript, n):
                    raw += piece
                    seen = min(llm.count_questions(raw), n)
                    if seen != written:
                        written = seen
                        yield _progress("quiz", (written - 1) / n, f"Question {written}/{n}")
            quiz = llm.parse_quiz(raw)[:n]
            engine, mode = llm.model_id, "live"
            if not quiz:
                yield _log("Param-1 didn't return a valid quiz; using the offline generator.", "warning")

        if not quiz:
            yield _stage("quiz", "start", f"Writing {n} questions · offline cloze generator")
            quiz = generate_cloze_quiz(transcript, n)
            engine, mode = "Cloze generator (offline)", "fallback"
            if not quiz:
                yield _error(
                    "Not enough distinct key terms to build a quiz.",
                    "Try a longer transcript.",
                )
                return
            for i in range(len(quiz)):
                self._pace(0.2)
                yield _progress("quiz", (i + 1) / len(quiz), f"Question {i + 1}/{len(quiz)}")

        seconds = time.perf_counter() - t0
        yield _stage("quiz", "done", f"{len(quiz)} questions", seconds=round(seconds, 2))
        data = {"quiz": quiz, "engine": engine, "mode": mode, "seconds": round(seconds, 2)}
        if mode == "live":
            self._store("quiz", payload, data)
        yield _result(data)

    # ------------------------------------------------------------------
    # Stage 5: translation
    # ------------------------------------------------------------------
    def translate(
        self,
        notes: str,
        quiz: List[Dict[str, Any]],
        target: str,
        fresh: bool = False,
    ) -> Iterator[Event]:
        """Translate notes and quiz into *target*."""
        if target not in SUPPORTED_TARGETS:
            yield _error(f"Unsupported target language: {target}.")
            return
        notes = (notes or "").strip()
        quiz = normalise_quiz(quiz or [])
        if not notes and not quiz:
            yield _error("Nothing to translate yet.", "Generate notes or a quiz first.")
            return

        flores = SUPPORTED_TARGETS[target]
        payload = {"notes": _norm(notes), "quiz": quiz, "target": target}
        stored = None if fresh else self.cache.get("translate", payload)
        if stored:
            yield _stage("translate", "start", f"Stored {target} translation · {stored.get('engine', 'model')}")
            for i in range(4):
                self._pace(0.2)
                yield _progress("translate", (i + 1) / 4, "Translating")
            yield _stage("translate", "done", f"{target} ready", seconds=stored.get("seconds"))
            yield _result(stored)
            return

        translator = None
        t0 = time.perf_counter()
        if self.caps.live("mt"):
            if "mt" not in self._models:
                yield _log("Loading the translation model (the first run downloads it)…")
            translator = self._model("mt", lambda: build_translator(self.cfg, self.caps))
        if translator is None:
            yield _error(
                "No translation model is available here.",
                "Install the ML stack (see README). The sample lecture's translations are stored.",
            )
            return

        yield _stage("translate", "start", f"Translating to {target}")
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

        seconds = time.perf_counter() - t0
        yield _stage("translate", "done", f"{target} ready", seconds=round(seconds, 2))
        data = {
            "target": target,
            "native": NATIVE_NAMES.get(target, target),
            "code": flores,
            "notes": out_notes,
            "quiz": normalise_quiz(out_quiz),
            "engine": translator.model_id,
            "mode": "live",
            "seconds": round(seconds, 2),
        }
        self._store("translate", payload, data)
        yield _result(data)


def load_eval_predictions(path: Path) -> Dict[str, str]:
    """Read ``{audio_file: transcript}`` (or the richer eval_inputs format)."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("predictions"), dict):
        data = data["predictions"]
    return {str(k): str(v) for k, v in data.items()} if isinstance(data, dict) else {}


__all__ = [
    "PipelineHub",
    "asr_model_id",
    "build_asr",
    "build_llm",
    "build_translator",
    "load_eval_predictions",
]
