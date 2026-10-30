"""
sample.py — The built-in sample lecture
=======================================
A short Hindi lecture on gradient descent that lets anyone walk the whole
pipeline without installing models or finding a recording.

* The **transcript, notes, quiz and translations** are hand-written and
  stored in ``data/sample/gradient_descent.json``.  The UI labels them as
  pre-rendered whenever a real model did not produce them.
* The **audio** is synthetic: a speech-like babble (voiced syllables with
  formants, pauses and pitch declination) generated with NumPy so the
  waveform, VAD chunks and synced transcript have something real to show.
  Its phrase layout matches the transcript segments one-to-one.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from loguru import logger

SAMPLE_RATE = 16_000
LEAD_IN_SEC = 0.6
GAP_SEC = 1.4
TAIL_SEC = 0.8

_VOWELS: Sequence[Tuple[float, float]] = (
    (730, 1090),  # a
    (270, 2290),  # i
    (530, 1840),  # e
    (300, 870),  # u
    (570, 840),  # o
)


def _normalise_ws(text: str) -> str:
    return " ".join(text.split())


class SampleLecture:
    """Loads the curated sample and synthesises its audio on first use."""

    def __init__(self, content_path: Path, audio_dir: Path) -> None:
        self.data: Dict[str, Any] = json.loads(content_path.read_text(encoding="utf-8"))
        self.audio_path = audio_dir / self.data["file_name"]

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    @property
    def title(self) -> str:
        return self.data["title"]

    @property
    def language(self) -> str:
        return self.data["language"]

    @property
    def spans(self) -> List[Tuple[float, float]]:
        spans, t = [], LEAD_IN_SEC
        for dur in self.data["layout"]:
            spans.append((round(t, 3), round(t + dur, 3)))
            t += dur + GAP_SEC
        return spans

    @property
    def duration(self) -> float:
        return round(self.spans[-1][1] + TAIL_SEC, 3)

    @property
    def segments(self) -> List[Dict[str, Any]]:
        return [
            {"start": s, "end": e, "text": text}
            for (s, e), text in zip(self.spans, self.data["segments"])
        ]

    @property
    def transcript(self) -> str:
        return " ".join(self.data["segments"])

    @property
    def notes(self) -> str:
        return self.data["notes"]

    def quiz(self, n: Optional[int] = None) -> List[Dict[str, Any]]:
        items = self.data["quiz"][: n or None]
        return [
            {**q, "correct_answer": q["options"][q["answer_index"]]} for q in items
        ]

    def record(self) -> Dict[str, Any]:
        """File-store-shaped record for the sample audio."""
        self.ensure_audio()
        return {
            "id": "sample",
            "name": self.data["file_name"],
            "ext": ".wav",
            "size": self.audio_path.stat().st_size,
            "duration": self.duration,
            "title": self.title,
            "language": self.language,
            "speaker": self.data.get("speaker", ""),
            "sample": True,
        }

    # ------------------------------------------------------------------
    # Matching (decides when curated content may stand in for a model)
    # ------------------------------------------------------------------
    def matches_transcript(self, transcript: str) -> bool:
        return _normalise_ws(transcript) == _normalise_ws(self.transcript)

    def matches_notes(self, notes: str) -> bool:
        return _normalise_ws(notes) == _normalise_ws(self.notes)

    def matches_quiz(self, quiz: List[Dict[str, Any]]) -> bool:
        source = self.data["quiz"]
        return bool(quiz) and len(quiz) <= len(source) and all(
            q.get("question") == s["question"] for q, s in zip(quiz, source)
        )

    def translation(self, language: str) -> Optional[Dict[str, Any]]:
        return self.data["translations"].get(language)

    @property
    def translated_languages(self) -> List[str]:
        return list(self.data["translations"])

    # ------------------------------------------------------------------
    # Audio synthesis
    # ------------------------------------------------------------------
    def ensure_audio(self) -> Path:
        if not self.audio_path.exists():
            self.audio_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Synthesising sample lecture audio → {self.audio_path}")
            synthesise_babble(self.audio_path, self.spans, self.duration)
        return self.audio_path


def synthesise_babble(
    path: Path,
    spans: Sequence[Tuple[float, float]],
    duration: float,
    sr: int = SAMPLE_RATE,
    seed: int = 4,
) -> None:
    """Write a 16-bit mono WAV of speech-like babble filling each span.

    Intra-phrase pauses stay under 300 ms so a VAD treats each span as one
    continuous utterance, while the 1.4 s gaps between spans separate them.
    """
    rng = np.random.default_rng(seed)
    out = np.zeros(int(duration * sr), dtype=np.float64)

    for start, end in spans:
        cursor = start
        while cursor < end - 0.25:
            for _ in range(int(rng.integers(2, 5))):  # syllables per word
                length = float(rng.uniform(0.11, 0.22))
                if cursor + length > end:
                    break
                progress = (cursor - start) / (end - start)
                f0 = 178.0 - 48.0 * progress + float(rng.normal(0.0, 7.0))
                _add_syllable(out, sr, cursor, length, f0, rng)
                cursor += length + float(rng.uniform(0.0, 0.03))
            cursor += float(rng.uniform(0.05, 0.12))  # word gap
            if rng.random() < 0.12:
                cursor += float(rng.uniform(0.08, 0.14))  # comma pause

    out += rng.normal(0.0, 0.002, size=out.shape)  # room tone
    out *= 0.55 / max(np.abs(out).max(), 1e-9)
    pcm = (np.clip(out, -1.0, 1.0) * 32767).astype("<i2")

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sr)
        wav.writeframes(pcm.tobytes())


def _add_syllable(
    out: np.ndarray,
    sr: int,
    start: float,
    length: float,
    f0: float,
    rng: np.random.Generator,
) -> None:
    n = int(length * sr)
    i0 = int(start * sr)
    if n <= 0 or i0 + n > len(out):
        return
    t = np.arange(n) / sr
    f1, f2 = _VOWELS[int(rng.integers(0, len(_VOWELS)))]

    f0_track = f0 * (1.0 + 0.03 * np.sin(2 * np.pi * float(rng.uniform(3, 6)) * t))
    phase = 2 * np.pi * np.cumsum(f0_track) / sr

    voiced = np.zeros(n)
    for h in range(1, 16):
        freq = h * f0
        gain = (
            np.exp(-(((freq - f1) / 160.0) ** 2))
            + 0.6 * np.exp(-(((freq - f2) / 240.0) ** 2))
            + 0.05
        )
        voiced += (gain / h) * np.sin(h * phase)

    envelope = np.sin(np.pi * t / length) ** 0.8
    burst = rng.normal(0.0, 1.0, n) * np.exp(-t / 0.012) * 0.25  # consonant onset
    out[i0 : i0 + n] += (voiced * envelope + burst) * float(rng.uniform(0.55, 1.0))
