"""
eval_pipeline.py — Evaluation Suite for ASR & Summarisation Quality
===================================================================
Automates:
    1. **WER (Word Error Rate)** calculation for ASR transcripts against
       ground-truth references using ``jiwer``.
    2. **BERTScore** for measuring factual consistency between generated
       notes and reference summaries.
    3. **ROUGE** scores (ROUGE-1, ROUGE-2, ROUGE-L) as an additional
       overlap-based metric.
    4. **Failure analysis** — logs all instances where WER > 0.3 or
       factual hallucination is suspected into ``failure_analysis.csv``.

Ground-Truth File Format
------------------------
The evaluation expects a JSON file at ``data/ground_truth/references.json``
with the structure::

    [
        {
            "audio_file": "lecture_01.wav",
            "transcript": "Ground truth transcript text ...",
            "summary": "Optional reference summary ..."
        },
        ...
    ]

Usage::

    evaluator = EvalPipeline()
    report = evaluator.run("data/ground_truth/references.json")
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

try:
    from jiwer import wer as compute_wer

    _JIWER_AVAILABLE = True
except ImportError:
    _JIWER_AVAILABLE = False

try:
    from bert_score import score as bert_score_fn

    _BERTSCORE_AVAILABLE = True
except ImportError:
    _BERTSCORE_AVAILABLE = False

try:
    from rouge_score import rouge_scorer

    _ROUGE_AVAILABLE = True
except ImportError:
    _ROUGE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class EvalResult:
    """Evaluation result for a single audio sample."""

    audio_file: str
    wer: float
    reference_transcript: str
    predicted_transcript: str
    bert_precision: float = 0.0
    bert_recall: float = 0.0
    bert_f1: float = 0.0
    rouge1_f: float = 0.0
    rouge2_f: float = 0.0
    rougeL_f: float = 0.0
    is_failure: bool = False
    failure_reasons: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WER_THRESHOLD: float = 0.30
"""Samples with WER above this are flagged as failures."""

BERTSCORE_THRESHOLD: float = 0.70
"""BERTScore F1 below this suggests possible hallucination."""

DEFAULT_OUTPUT_DIR: str = "outputs/evaluations"


# ---------------------------------------------------------------------------
# EvalPipeline
# ---------------------------------------------------------------------------
class EvalPipeline:
    """End-to-end evaluation pipeline.

    Parameters
    ----------
    output_dir : str | Path
        Directory to write ``failure_analysis.csv`` and summary reports.
    bert_model : str
        Model for BERTScore (e.g. ``"bert-base-multilingual-cased"``).
    """

    def __init__(
        self,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        bert_model: str = "bert-base-multilingual-cased",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bert_model = bert_model

        if not _JIWER_AVAILABLE:
            logger.warning("jiwer not installed — WER computation disabled.")
        if not _BERTSCORE_AVAILABLE:
            logger.warning("bert-score not installed — BERTScore disabled.")
        if not _ROUGE_AVAILABLE:
            logger.warning("rouge-score not installed — ROUGE disabled.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        ground_truth_path: str | Path,
        predictions: Optional[Dict[str, str]] = None,
        summaries_pred: Optional[Dict[str, str]] = None,
        summaries_ref: Optional[Dict[str, str]] = None,
    ) -> List[EvalResult]:
        """Run the full evaluation suite.

        Parameters
        ----------
        ground_truth_path : str | Path
            Path to the ground-truth JSON file.
        predictions : dict | None
            Mapping ``{audio_file: predicted_transcript}``.
            If *None*, only the ground-truth structure is validated.
        summaries_pred : dict | None
            Mapping ``{audio_file: predicted_summary}`` for BERTScore/ROUGE.
        summaries_ref : dict | None
            Mapping ``{audio_file: reference_summary}``.

        Returns
        -------
        List[EvalResult]
            Per-sample evaluation results.
        """
        gt_path = Path(ground_truth_path)
        if not gt_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

        with open(gt_path, "r", encoding="utf-8") as f:
            ground_truth: List[Dict[str, Any]] = json.load(f)

        logger.info(f"Loaded {len(ground_truth)} ground-truth samples.")

        results: List[EvalResult] = []
        predictions = predictions or {}
        summaries_pred = summaries_pred or {}
        summaries_ref = summaries_ref or {}

        for sample in ground_truth:
            audio_file = sample["audio_file"]
            ref_transcript = sample.get("transcript", "")
            pred_transcript = predictions.get(audio_file, "")

            result = EvalResult(
                audio_file=audio_file,
                wer=0.0,
                reference_transcript=ref_transcript,
                predicted_transcript=pred_transcript,
            )

            # ── WER ────────────────────────────────────────────────────
            if _JIWER_AVAILABLE and ref_transcript and pred_transcript:
                result.wer = compute_wer(ref_transcript, pred_transcript)
                if result.wer > WER_THRESHOLD:
                    result.is_failure = True
                    result.failure_reasons.append(
                        f"WER={result.wer:.3f} > {WER_THRESHOLD}"
                    )

            # ── BERTScore (on summaries) ───────────────────────────────
            ref_summary = summaries_ref.get(audio_file, sample.get("summary", ""))
            pred_summary = summaries_pred.get(audio_file, "")

            if _BERTSCORE_AVAILABLE and ref_summary and pred_summary:
                P, R, F1 = bert_score_fn(
                    [pred_summary],
                    [ref_summary],
                    model_type=self.bert_model,
                    lang="en",
                    verbose=False,
                )
                result.bert_precision = P[0].item()
                result.bert_recall = R[0].item()
                result.bert_f1 = F1[0].item()

                if result.bert_f1 < BERTSCORE_THRESHOLD:
                    result.is_failure = True
                    result.failure_reasons.append(
                        f"BERTScore F1={result.bert_f1:.3f} < {BERTSCORE_THRESHOLD} "
                        "(possible hallucination)"
                    )

            # ── ROUGE (on summaries) ───────────────────────────────────
            if _ROUGE_AVAILABLE and ref_summary and pred_summary:
                scorer = rouge_scorer.RougeScorer(
                    ["rouge1", "rouge2", "rougeL"], use_stemmer=True
                )
                scores = scorer.score(ref_summary, pred_summary)
                result.rouge1_f = scores["rouge1"].fmeasure
                result.rouge2_f = scores["rouge2"].fmeasure
                result.rougeL_f = scores["rougeL"].fmeasure

            results.append(result)

        # ── Write failure analysis CSV ─────────────────────────────────
        self._write_failure_csv(results)

        # ── Log summary ────────────────────────────────────────────────
        failures = [r for r in results if r.is_failure]
        avg_wer = (
            sum(r.wer for r in results) / len(results) if results else 0.0
        )
        logger.info(
            f"Evaluation complete: {len(results)} samples, "
            f"avg WER={avg_wer:.3f}, {len(failures)} failures."
        )

        return results

    # ------------------------------------------------------------------
    # CSV output
    # ------------------------------------------------------------------
    def _write_failure_csv(self, results: List[EvalResult]) -> Path:
        """Write ``failure_analysis.csv`` containing flagged samples.

        Parameters
        ----------
        results : List[EvalResult]

        Returns
        -------
        Path
            Path to the written CSV.
        """
        csv_path = self.output_dir / "failure_analysis.csv"
        fieldnames = [
            "audio_file",
            "wer",
            "bert_f1",
            "rouge1_f",
            "rouge2_f",
            "rougeL_f",
            "failure_reasons",
            "reference_transcript",
            "predicted_transcript",
        ]

        failures = [r for r in results if r.is_failure]

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in failures:
                writer.writerow(
                    {
                        "audio_file": r.audio_file,
                        "wer": f"{r.wer:.4f}",
                        "bert_f1": f"{r.bert_f1:.4f}",
                        "rouge1_f": f"{r.rouge1_f:.4f}",
                        "rouge2_f": f"{r.rouge2_f:.4f}",
                        "rougeL_f": f"{r.rougeL_f:.4f}",
                        "failure_reasons": "; ".join(r.failure_reasons),
                        "reference_transcript": r.reference_transcript[:500],
                        "predicted_transcript": r.predicted_transcript[:500],
                    }
                )

        logger.info(
            f"Failure analysis CSV written: {csv_path} ({len(failures)} rows)"
        )
        return csv_path

    def generate_report_df(self, results: List[EvalResult]) -> pd.DataFrame:
        """Convert results to a pandas DataFrame for display.

        Parameters
        ----------
        results : List[EvalResult]

        Returns
        -------
        pd.DataFrame
        """
        rows = []
        for r in results:
            rows.append(
                {
                    "audio_file": r.audio_file,
                    "WER": round(r.wer, 4),
                    "BERTScore_F1": round(r.bert_f1, 4),
                    "ROUGE-1": round(r.rouge1_f, 4),
                    "ROUGE-2": round(r.rouge2_f, 4),
                    "ROUGE-L": round(r.rougeL_f, 4),
                    "Failure": "FAIL" if r.is_failure else "PASS",
                    "Reasons": "; ".join(r.failure_reasons) if r.failure_reasons else "",
                }
            )
        return pd.DataFrame(rows)
