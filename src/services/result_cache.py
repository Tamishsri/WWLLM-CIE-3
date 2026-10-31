"""
result_cache.py — Keep what the models produced
===============================================
Real models are slow on a CPU, so every result is stored and replayed the next
time the same input comes back.  The content is still genuine model output;
the UI marks it as cached, and "Regenerate" asks for a fresh run.

Two locations are read:

    * ``data/cache/``          written at runtime (not committed)
    * ``data/sample/results/`` real outputs for the sample lecture, committed so
                               the sample works on a laptop without the models
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


def file_digest(path: Path) -> str:
    """SHA-256 of a file, read in blocks so large recordings stay cheap."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ResultCache:
    """Stage results on disk, keyed by a hash of the stage's input.

    Parameters
    ----------
    writable : Path
        Where new results go.
    read_only : Sequence[Path]
        Extra locations to look in.
    """

    def __init__(self, writable: Path, read_only: Sequence[Path] = ()) -> None:
        self.writable = writable
        self.read_only = list(read_only)

    @staticmethod
    def _name(stage: str, payload: Dict[str, Any]) -> str:
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return f"{stage}-{hashlib.sha256(blob).hexdigest()[:24]}.json"

    def get(self, stage: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return a stored result, or *None*.  Unreadable entries count as misses."""
        name = self._name(stage, payload)
        for root in [self.writable, *self.read_only]:
            path = root / name
            if not path.exists():
                continue
            try:
                return {**json.loads(path.read_text(encoding="utf-8")), "cached": True}
            except (OSError, json.JSONDecodeError):
                continue
        return None

    def put(self, stage: str, payload: Dict[str, Any], value: Dict[str, Any]) -> None:
        self.writable.mkdir(parents=True, exist_ok=True)
        entry = {k: v for k, v in value.items() if k != "cached"}
        path = self.writable / self._name(stage, payload)
        path.write_text(
            json.dumps({**entry, "created": time.time()}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
