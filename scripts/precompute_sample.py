#!/usr/bin/env python3
"""
precompute_sample.py — Run the real pipeline on the sample lecture
==================================================================
Sends the sample's audio through the actual models and stores every output in
``data/sample/results/``, so the sample plays instantly and works on a laptop
without the models.  Nothing here is written by hand.

Usage::

    python scripts/precompute_sample.py                    # whatever is missing
    python scripts/precompute_sample.py --only transcript
    python scripts/precompute_sample.py --only notes quiz translations
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jiwer import wer  # noqa: E402

from src.config import load_config  # noqa: E402
from src.pipelines.languages import SUPPORTED_TARGETS  # noqa: E402
from src.services.capabilities import detect_capabilities  # noqa: E402
from src.services.engines import PipelineHub  # noqa: E402
from src.services.result_cache import ResultCache  # noqa: E402
from src.services.sample import SampleLecture  # noqa: E402

STAGES = ("transcript", "notes", "quiz", "translations")


def run(events: Iterator[Dict[str, Any]]) -> Dict[str, Any]:
    """Drain a hub stream, echoing progress, and return its result."""
    result: Dict[str, Any] = {}
    tokens = 0
    for event in events:
        kind = event["event"]
        if kind == "stage":
            print(f"  {event['stage']}: {event['label']}", flush=True)
        elif kind == "log":
            print(f"  {event['message']}", flush=True)
        elif kind == "token":
            tokens += 1
            print(event["text"], end="", flush=True)
        elif kind == "error":
            raise RuntimeError(f"{event['message']} {event.get('hint', '')}")
        elif kind == "result":
            result = event["data"]
    if tokens:
        print(flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--only", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--fresh", action="store_true", help="Ignore stored results")
    args = parser.parse_args()

    cfg = load_config()
    caps = detect_capabilities()
    sample = SampleLecture(ROOT / "data/sample/gradient_descent.json", ROOT / "data/sample")
    hub = PipelineHub(cfg, caps, sample, ResultCache(sample.results_dir), demo_pace=0)

    print("Transcribing", flush=True)
    only = set(args.only)
    fresh = args.fresh
    transcript = run(hub.transcribe(sample.record(), sample.audio_path, sample.language, fresh="transcript" in only and fresh))
    plain = lambda text: re.sub(r"[।॥,.!?\-]", " ", text)  # noqa: E731
    measured = wer(plain(sample.reference), plain(transcript["transcript"]))
    print(f"  WER against the text the voice read: {measured:.3f}", flush=True)
    print(transcript["transcript"], flush=True)

    started = time.perf_counter()
    notes: Dict[str, Any] = {}
    quiz: Dict[str, Any] = {}
    if only & {"notes", "translations"}:
        print("Notes", flush=True)
        notes = run(hub.notes(transcript["transcript"], fresh="notes" in only and fresh))
    if only & {"quiz", "translations"}:
        print("Quiz", flush=True)
        quiz = run(hub.quiz(transcript["transcript"], 5, fresh="quiz" in only and fresh))
        print(json.dumps(quiz.get("quiz"), ensure_ascii=False, indent=1), flush=True)

    translated = []
    if "translations" in only:
        for target in SUPPORTED_TARGETS:
            print(f"Translating: {target}", flush=True)
            run(hub.translate(notes.get("notes", ""), quiz.get("quiz", []), target, fresh=fresh))
            translated.append(target)

    manifest_path = sample.results_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest.update(
        {
            "translations": translated or manifest.get("translations", []),
            "asr_model": transcript.get("engine"),
            "asr_wer": round(measured, 4),
            "date": time.strftime("%Y-%m-%d"),
        }
    )
    sample.results_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done in {time.perf_counter() - started:.0f}s: {manifest}", flush=True)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
