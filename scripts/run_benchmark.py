#!/usr/bin/env python3
"""
run_benchmark.py — Measure the real pipeline on the 40-lecture benchmark
========================================================================
Transcribes every clip in ``data/ground_truth/audio/`` with the real VAD and
ASR models and writes the output for the Benchmark page and ``run_eval.py``:

    outputs/transcripts/predictions.json    {audio_file: transcript}
    outputs/transcripts/eval_inputs.json    predictions, summaries, measurement details
    outputs/evaluations/failure_analysis.csv

Usage::

    python scripts/run_benchmark.py               # transcripts
    python scripts/run_benchmark.py --summaries   # also summarise with Param-1 (slow on CPU)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config  # noqa: E402
from src.pipelines.audio_utils import AudioProcessor  # noqa: E402
from src.pipelines.eval_pipeline import EvalPipeline, detect_language  # noqa: E402
from src.services.capabilities import detect_capabilities  # noqa: E402
from src.services.engines import build_asr, build_llm  # noqa: E402

OUTPUTS = ROOT / "outputs" / "transcripts"
EVAL_INPUTS = OUTPUTS / "eval_inputs.json"
AUDIO = ROOT / "data" / "ground_truth" / "audio"

SUMMARY_PROMPT = (
    "Summarise this lecture excerpt in one short English sentence. "
    "Use only facts from the excerpt."
)


def transcribe_all(records: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    caps = detect_capabilities()
    audio_cfg = cfg.get("audio", {})
    vad = AudioProcessor(
        max_chunk_sec=audio_cfg.get("max_chunk_sec", 25.0),
        vad_threshold=audio_cfg.get("vad_threshold", 0.45),
        min_silence_ms=audio_cfg.get("min_silence_ms", 300),
    )

    predictions: Dict[str, str] = {}
    models: Dict[str, str] = {}
    asr, loaded = None, None
    started = time.perf_counter()
    # Sorted by language so each model loads once.
    ordered = sorted(records, key=lambda r: detect_language(r["transcript"]))
    for i, record in enumerate(ordered, 1):
        language = detect_language(record["transcript"])
        if language != loaded:
            asr, loaded = None, None  # release the previous model first
            asr, loaded = build_asr(cfg, caps, language), language
            models[language] = asr.model_id
        chunks, _ = vad.process_with_spans(AUDIO / record["audio_file"])
        texts = asr.transcribe(chunks)
        predictions[record["audio_file"]] = " ".join(t for t in texts if t.strip())
        print(f"[{i}/{len(ordered)}] {language}: {record['audio_file']}", flush=True)

    return {
        "predictions": {r["audio_file"]: predictions[r["audio_file"]] for r in records},
        "measurement": {
            "asr_models": models,
            "speech": "Meta MMS text-to-speech reading each reference",
            "seconds": round(time.perf_counter() - started, 1),
            "date": time.strftime("%Y-%m-%d"),
        },
    }


def summarise_all(records: List[Dict[str, Any]], cfg: Dict[str, Any], inputs: Dict[str, Any]) -> None:
    llm = build_llm(cfg, detect_capabilities())
    summaries: Dict[str, str] = dict(inputs.get("summaries_pred") or {})
    for i, record in enumerate(records, 1):
        name = record["audio_file"]
        if not summaries.get(name):
            messages = [
                {"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user", "content": inputs["predictions"][name]},
            ]
            summaries[name] = llm._generate(messages, max_new_tokens=60)
            inputs["summaries_pred"] = summaries  # a long run can resume from here
            EVAL_INPUTS.write_text(json.dumps(inputs, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{i}/{len(records)}] summary: {name}", flush=True)
    inputs["measurement"]["summary_model"] = llm.model_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--summaries", action="store_true", help="Also summarise with Param-1")
    parser.add_argument("--reuse", action="store_true", help="Keep existing transcripts")
    args = parser.parse_args()

    cfg = load_config()
    references = ROOT / "data" / "ground_truth" / "references.json"
    records = json.loads(references.read_text(encoding="utf-8"))

    if args.reuse and EVAL_INPUTS.exists():
        inputs = json.loads(EVAL_INPUTS.read_text(encoding="utf-8"))
    else:
        inputs = transcribe_all(records, cfg)
        inputs["summaries_ref"] = {r["audio_file"]: r.get("summary", "") for r in records}
        OUTPUTS.mkdir(parents=True, exist_ok=True)
        (OUTPUTS / "predictions.json").write_text(
            json.dumps(inputs["predictions"], ensure_ascii=False, indent=2), encoding="utf-8"
        )
    EVAL_INPUTS.write_text(json.dumps(inputs, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.summaries:
        summarise_all(records, cfg, inputs)
        EVAL_INPUTS.write_text(json.dumps(inputs, ensure_ascii=False, indent=2), encoding="utf-8")

    evaluator = EvalPipeline(output_dir=ROOT / "outputs" / "evaluations", use_bertscore=False)
    results = evaluator.run(
        references,
        predictions=inputs["predictions"],
        summaries_pred=inputs.get("summaries_pred"),
        summaries_ref=inputs.get("summaries_ref"),
    )
    frame = evaluator.generate_report_df(results)
    print(f"\nAverage WER {frame['WER'].mean():.4f}, failures {len(frame[frame['Failure'] == 'FAIL'])}/{len(frame)}")
    print(frame.groupby("Language")["WER"].agg(["count", "mean"]).round(4).to_string())


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
