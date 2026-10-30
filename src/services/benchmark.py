"""
benchmark.py — Evaluation reports for the Benchmark dashboard
=============================================================
Wraps :class:`~src.pipelines.eval_pipeline.EvalPipeline` and shapes its
results into one JSON document: headline metrics, per-language and
per-domain breakdowns, and every sample with its word alignment.
"""

from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.pipelines.eval_pipeline import WER_THRESHOLD, EvalPipeline, EvalResult

Payload = Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]

# Lecture-number ranges of the bundled 40-lecture set (see generate_test_dataset.py)
DOMAINS: List[Tuple[int, int, str]] = [
    (1, 10, "AI & Machine Learning"),
    (11, 15, "Computer Systems"),
    (16, 20, "Databases & Cloud"),
    (21, 25, "Indic Language Tech"),
    (26, 30, "OS, Networks & Security"),
    (31, 40, "Advanced Topics"),
]

_ACRONYMS = {
    "ml": "ML", "cuda": "CUDA", "acid": "ACID", "nosql": "NoSQL", "cap": "CAP",
    "tcp": "TCP", "kv": "KV", "lora": "LoRA", "vad": "VAD", "nlp": "NLP",
    "mt": "MT", "indictrans": "IndicTrans", "rag": "RAG",
}


def lecture_meta(audio_file: str) -> Dict[str, Any]:
    """Derive number, readable title and domain from a ``lecture_NN_slug.wav`` name."""
    stem = Path(audio_file).stem
    match = re.match(r"^lecture_(\d+)_(.+)$", stem)
    number = int(match.group(1)) if match else None
    slug = match.group(2) if match else stem
    words = [_ACRONYMS.get(w, w.capitalize()) for w in slug.split("_") if w]
    title = " ".join(words).replace(" B Trees", " B-Trees")
    domain = next(
        (name for lo, hi, name in DOMAINS if number is not None and lo <= number <= hi),
        "Uploaded set",
    )
    return {"number": number, "title": title, "domain": domain}


def parse_predictions(data: Any) -> Payload:
    """Accept ``{audio_file: text}`` or ``{"predictions": …, "summaries_pred": …}``."""
    if not isinstance(data, dict):
        raise ValueError("Predictions must be a JSON object.")
    if isinstance(data.get("predictions"), dict):
        return (
            data["predictions"],
            data.get("summaries_pred") or {},
            data.get("summaries_ref") or {},
        )
    return data, {}, {}


def load_default_dataset(references: Path, predictions: Path) -> Tuple[List[Dict[str, Any]], Payload]:
    records = json.loads(references.read_text(encoding="utf-8"))
    payload: Payload = ({}, {}, {})
    if predictions.exists():
        payload = parse_predictions(json.loads(predictions.read_text(encoding="utf-8")))
    return records, payload


def _sample(r: EvalResult) -> Dict[str, Any]:
    meta = lecture_meta(r.audio_file)
    status = "na" if r.wer is None else ("fail" if r.is_failure else "pass")
    return {
        "id": r.audio_file,
        **meta,
        "language": r.language,
        "condition": r.condition,
        "status": status,
        "wer": None if r.wer is None else round(r.wer, 4),
        "hits": r.hits,
        "substitutions": r.substitutions,
        "deletions": r.deletions,
        "insertions": r.insertions,
        "ref_words": r.ref_words,
        "reasons": r.failure_reasons,
        "reference": r.reference_transcript,
        "prediction": r.predicted_transcript,
        "alignment": r.alignment,
        "summary_ref": r.reference_summary,
        "summary_pred": r.predicted_summary,
        "rouge1": round(r.rouge1_f, 4),
        "rouge2": round(r.rouge2_f, 4),
        "rougeL": round(r.rougeL_f, 4),
        "bert_f1": round(r.bert_f1, 4) if r.bert_f1 else None,
    }


def _group(samples: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in samples:
        buckets[s[key]].append(s)
    rows = []
    for name, items in buckets.items():
        scored = [s["wer"] for s in items if s["wer"] is not None]
        rows.append(
            {
                "name": name,
                "count": len(items),
                "avg_wer": round(statistics.fmean(scored), 4) if scored else None,
                "failures": sum(1 for s in items if s["status"] == "fail"),
            }
        )
    return sorted(rows, key=lambda r: -r["count"])


def build_report(
    records: List[Dict[str, Any]],
    payload: Payload,
    *,
    source: str,
    output_dir: Path,
    bertscore: bool = False,
) -> Dict[str, Any]:
    """Evaluate *records* against *payload* and return the dashboard JSON."""
    predictions, summaries_pred, summaries_ref = payload
    pipeline = EvalPipeline(output_dir=output_dir, use_bertscore=bertscore)
    results = pipeline.run_records(
        records, predictions, summaries_pred, summaries_ref, write_csv=False
    )
    samples = [_sample(r) for r in results]
    scored = [s for s in samples if s["wer"] is not None]
    wers = [s["wer"] for s in scored]
    totals = {
        k: sum(s[k] for s in scored)
        for k in ("hits", "substitutions", "deletions", "insertions", "ref_words")
    }
    rouge = [s for s in samples if s["summary_ref"] and s["summary_pred"]]

    def mean(values: List[float]) -> Optional[float]:
        return round(statistics.fmean(values), 4) if values else None

    return {
        "source": source,
        "threshold": WER_THRESHOLD,
        "metrics": EvalPipeline.capabilities() | {"bertscore_used": pipeline.use_bertscore},
        "summary": {
            "samples": len(samples),
            "evaluated": len(scored),
            "avg_wer": mean(wers),
            "median_wer": round(statistics.median(wers), 4) if wers else None,
            "best_wer": min(wers) if wers else None,
            "worst_wer": max(wers) if wers else None,
            "failures": sum(1 for s in samples if s["status"] == "fail"),
            "pass_rate": round(sum(1 for s in scored if s["status"] == "pass") / len(scored), 4)
            if scored
            else None,
            "word_accuracy": round(totals["hits"] / totals["ref_words"], 4)
            if totals["ref_words"]
            else None,
            "totals": totals,
            "rouge1": mean([s["rouge1"] for s in rouge]),
            "rouge2": mean([s["rouge2"] for s in rouge]),
            "rougeL": mean([s["rougeL"] for s in rouge]),
            "bert_f1": mean([s["bert_f1"] for s in samples if s["bert_f1"] is not None]),
            "languages": len({s["language"] for s in samples}),
            "domains": len({s["domain"] for s in samples}),
        },
        "by_language": _group(samples, "language"),
        "by_domain": _group(samples, "domain"),
        "samples": samples,
    }
