"""
files.py — Upload storage for the Studio
========================================
Uploaded media is written to ``data/uploads/<id><ext>`` with a JSON
sidecar holding the original name, so sessions survive server restarts.
"""

from __future__ import annotations

import json
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, BinaryIO, Dict, Optional

from src.pipelines.fallback import wav_duration

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".mp4", ".mkv", ".webm"}
_ID_RE = re.compile(r"^[a-f0-9]{32}$")


class FileStore:
    """Tiny content store keyed by random hex ids."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, filename: str, stream: BinaryIO) -> Dict[str, Any]:
        """Persist an upload and return its public record.

        Raises
        ------
        ValueError
            If the extension is not an accepted audio/video type.
        """
        ext = Path(filename or "").suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            accepted = ", ".join(sorted(e.lstrip(".") for e in ALLOWED_EXTENSIONS))
            raise ValueError(f"Unsupported file type '{ext or '?'}'. Accepted: {accepted}.")

        file_id = uuid.uuid4().hex
        path = self.root / f"{file_id}{ext}"
        with open(path, "wb") as out:
            shutil.copyfileobj(stream, out, length=1024 * 1024)

        record = {
            "id": file_id,
            "name": Path(filename).name,
            "ext": ext,
            "size": path.stat().st_size,
            "duration": wav_duration(path) if ext == ".wav" else None,
            "created": time.time(),
        }
        (self.root / f"{file_id}.json").write_text(json.dumps(record), encoding="utf-8")
        return record

    def get(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Look up a record by id (``None`` if unknown or malformed)."""
        if not _ID_RE.match(file_id or ""):
            return None
        meta = self.root / f"{file_id}.json"
        if not meta.exists():
            return None
        record = json.loads(meta.read_text(encoding="utf-8"))
        return record if self.path(record).exists() else None

    def path(self, record: Dict[str, Any]) -> Path:
        return self.root / f"{record['id']}{record['ext']}"
