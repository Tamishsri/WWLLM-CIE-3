#!/usr/bin/env python3
"""
generate_speech.py — Real spoken audio for the sample lecture and benchmark
===========================================================================
Reads the reference transcripts aloud with Meta's MMS text-to-speech models,
giving the pipeline genuine speech to work on:

    * data/sample/<file_name>          the sample lecture
    * data/ground_truth/audio/*.wav    one clip per benchmark lecture

Usage::

    python scripts/generate_speech.py            # both
    python scripts/generate_speech.py --sample
    python scripts/generate_speech.py --benchmark

MMS models are CC-BY-NC 4.0: fine for this academic project, not for commercial use.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import wave
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoTokenizer, VitsModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipelines.eval_pipeline import detect_language  # noqa: E402

VOICES: Dict[str, str] = {
    "Hindi": "facebook/mms-tts-hin",
    "Tamil": "facebook/mms-tts-tam",
    "Kannada": "facebook/mms-tts-kan",
    "English": "facebook/mms-tts-eng",
}
SAMPLE_RATE = 16_000
SENTENCE_PAUSE = 0.35
PARAGRAPH_PAUSE = 1.0
LEAD_IN = 0.6
TAIL = 0.8


class Voice:
    """One MMS model, loaded on first use."""

    _loaded: Dict[str, "Voice"] = {}

    def __init__(self, language: str) -> None:
        model_id = VOICES[language]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = VitsModel.from_pretrained(model_id).eval()

    @classmethod
    def get(cls, language: str) -> "Voice":
        if language not in cls._loaded:
            cls._loaded.clear()  # one voice at a time keeps memory flat
            cls._loaded[language] = cls(language)
        return cls._loaded[language]

    def say(self, text: str) -> np.ndarray:
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            audio = self.model(**inputs).waveform[0].numpy()
        return audio


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


def read_aloud(text: str, language: str) -> np.ndarray:
    """Speak sentence by sentence so the pacing has natural pauses."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?।])\s+", text) if s.strip()]
    voice = Voice.get(language)
    parts: List[np.ndarray] = []
    for sentence in sentences:
        parts += [voice.say(sentence), silence(SENTENCE_PAUSE)]
    return np.concatenate(parts[:-1])


def save(path: Path, audio: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.abs(audio).max()) or 1.0
    pcm = (audio / peak * 0.89 * 32767).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(pcm.tobytes())


def build_sample() -> None:
    meta_path = ROOT / "data/sample/gradient_descent.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    parts: List[np.ndarray] = [silence(LEAD_IN)]
    for paragraph in meta["segments"]:
        parts += [read_aloud(paragraph, meta["language"]), silence(PARAGRAPH_PAUSE)]
    audio = np.concatenate(parts[:-1] + [silence(TAIL)])
    target = ROOT / "data/sample" / meta["file_name"]
    save(target, audio)
    print(f"sample: {len(audio) / SAMPLE_RATE:.1f} s -> {target.name}", flush=True)


def build_benchmark() -> None:
    records = json.loads((ROOT / "data/ground_truth/references.json").read_text(encoding="utf-8"))
    # Grouped by language so each voice loads once.
    records.sort(key=lambda r: detect_language(r["transcript"]))
    for i, record in enumerate(records, 1):
        target = ROOT / "data/ground_truth/audio" / record["audio_file"]
        if target.exists():
            continue
        language = detect_language(record["transcript"])
        save(target, read_aloud(record["transcript"], language))
        print(f"[{i}/{len(records)}] {language}: {record['audio_file']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--sample", action="store_true", help="Only the sample lecture")
    parser.add_argument("--benchmark", action="store_true", help="Only the benchmark clips")
    args = parser.parse_args()
    both = not (args.sample or args.benchmark)
    if args.sample or both:
        build_sample()
    if args.benchmark or both:
        build_benchmark()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
