"""
sample.py — The built-in sample lecture
=======================================
A short Hindi lecture on gradient descent that lets anyone walk the whole
pipeline without finding a recording.

The audio is real speech, generated from the reference text in
``data/sample/gradient_descent.json`` by ``scripts/generate_speech.py``.  Its
transcript, notes, quiz and translations are real model outputs, produced by
``scripts/precompute_sample.py`` and stored in ``data/sample/results/``, so the
sample also works on a laptop without the models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.pipelines.fallback import wav_duration


class SampleLecture:
    """Metadata and audio of the sample lecture."""

    def __init__(self, content_path: Path, audio_dir: Path) -> None:
        self.data: Dict[str, Any] = json.loads(content_path.read_text(encoding="utf-8"))
        self.audio_path = audio_dir / self.data["file_name"]
        self.results_dir = audio_dir / "results"

    @property
    def title(self) -> str:
        return self.data["title"]

    @property
    def language(self) -> str:
        return self.data["language"]

    @property
    def reference(self) -> str:
        """The text the voice reads: ground truth for this lecture's ASR."""
        return " ".join(self.data["segments"])

    @property
    def duration(self) -> float:
        return round(wav_duration(self.audio_path) or 0.0, 3)

    @property
    def translated_languages(self) -> List[str]:
        """Languages with a stored translation of the sample's notes and quiz."""
        manifest = self.results_dir / "manifest.json"
        if not manifest.exists():
            return []
        return json.loads(manifest.read_text(encoding="utf-8")).get("translations", [])

    def record(self) -> Dict[str, Any]:
        """File-store-shaped record for the sample audio."""
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
