"""
eval_pipeline.py — Evaluation Suite for ASR & Summarisation Quality
===================================================================
Automates:
    1. **WER (Word Error Rate)** calculation for ASR transcripts against
       ground-truth references using ``jiwer``, including the word-level
       alignment (hits / substitutions / deletions / insertions) used by
       the Studio's diff view.
    2. **BERTScore** for measuring factual consistency between generated
       notes and reference summaries (batched in a single model call).
    3. **ROUGE** scores (ROUGE-1, ROUGE-2, ROUGE-L) as an additional
       overlap-based metric.  Falls back to a built-in implementation when
       ``rouge-score`` is not installed.
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
            "summary": "Optional reference summary ...",
            "condition": "Optional recording-condition note ..."
        },
        ...
    ]

Usage::

    evaluator = EvalPipeline()
    report = evaluator.run("data/ground_truth/references.json", predictions=preds)
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

try:
    import jiwer

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
    """Evaluation result for a single audio sample.

    ``wer`` is *None* when no prediction was supplied for the sample — a
    missing transcript must never count as a perfect score.
    """

    audio_file: str
    wer: Optional[float]
    reference_transcript: str
    predicted_transcript: str
    condition: str = ""
    language: str = ""
    hits: int = 0
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    alignment: List[Dict[str, Any]] = field(default_factory=list)
    reference_summary: str = ""
    predicted_summary: str = ""
    bert_precision: float = 0.0
    bert_recall: float = 0.0
    bert_f1: float = 0.0
    rouge1_f: float = 0.0
    rouge2_f: float = 0.0
    rougeL_f: float = 0.0
    is_failure: bool = False
    failure_reasons: List[str] = field(default_factory=list)

    @property
    def ref_words(self) -> int:
        return self.hits + self.substitutions + self.deletions


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WER_THRESHOLD: float = 0.30
"""Samples with WER above this are flagged as failures."""

BERTSCORE_THRESHOLD: float = 0.70
"""BERTScore F1 below this suggests possible hallucination."""

DEFAULT_OUTPUT_DIR: str = "outputs/evaluations"

_SCRIPT_RANGES: List[Tuple[str, int, int]] = [
    ("Hindi", 0x0900, 0x097F),  # Devanagari
    ("Bengali", 0x0980, 0x09FF),
    ("Punjabi", 0x0A00, 0x0A7F),
    ("Gujarati", 0x0A80, 0x0AFF),
    ("Odia", 0x0B00, 0x0B7F),
    ("Tamil", 0x0B80, 0x0BFF),
    ("Telugu", 0x0C00, 0x0C7F),
    ("Kannada", 0x0C80, 0x0CFF),
    ("Malayalam", 0x0D00, 0x0D7F),
]


_PUNCTUATION = re.compile(r"[^\w\s\u0900-\u0D7F]|[\u0964\u0965]|_")


def normalise_for_wer(text: str) -> str:
    """Lower-case and strip punctuation, the usual preparation before scoring.

    ASR output is rarely punctuated like a reference, and "है।" against "है"
    is not a word error.
    """
    return " ".join(_PUNCTUATION.sub(" ", text.lower()).split())


def detect_language(text: str) -> str:
    """Guess a transcript's language from its dominant script."""
    counts: Counter = Counter()
    for ch in text:
        cp = ord(ch)
        if ch.isascii():
            if ch.isalpha():
                counts["English"] += 1
            continue
        for name, lo, hi in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[name] += 1
                break
    return counts.most_common(1)[0][0] if counts else "Unknown"


# ---------------------------------------------------------------------------
# Built-in ROUGE (used when rouge-score is not installed)
# ---------------------------------------------------------------------------
_ROUGE_TOKEN = re.compile(r"[a-z0-9]+")


def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _f1(overlap: float, ref_total: int, pred_total: int) -> float:
    if not ref_total or not pred_total or not overlap:
        return 0.0
    precision, recall = overlap / pred_total, overlap / ref_total
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: List[str], b: List[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[j - 1]))
        prev = cur
    return prev[-1]


def simple_rouge(reference: str, prediction: str) -> Tuple[float, float, float]:
    """ROUGE-1/2/L F-measures with rouge-score's default tokenisation (no stemmer)."""
    ref = _ROUGE_TOKEN.findall(reference.lower())
    pred = _ROUGE_TOKEN.findall(prediction.lower())
    scores = []
    for n in (1, 2):
        r, p = _ngrams(ref, n), _ngrams(pred, n)
        scores.append(_f1(sum((r & p).values()), sum(r.values()), sum(p.values())))
    scores.append(_f1(_lcs_length(ref, pred), len(ref), len(pred)))
    return scores[0], scores[1], scores[2]


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
    use_bertscore : bool
        Set *False* to skip BERTScore even when installed (it downloads and
        runs a BERT model, which is slow for interactive dashboards).
    """

    def __init__(
        self,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        bert_model: str = "bert-base-multilingual-cased",
        use_bertscore: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bert_model = bert_model
        self.use_bertscore = use_bertscore and _BERTSCORE_AVAILABLE

        if not _JIWER_AVAILABLE:
            logger.warning("jiwer not installed — WER computation disabled.")
        if not _BERTSCORE_AVAILABLE:
            logger.warning("bert-score not installed — BERTScore disabled.")
        if not _ROUGE_AVAILABLE:
            logger.info("rouge-score not installed — using built-in ROUGE.")

        self._rouge = (
            rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            if _ROUGE_AVAILABLE
            else None
        )

    @staticmethod
    def capabilities() -> Dict[str, Any]:
        """Which metric backends are available in this environment."""
        return {
            "wer": _JIWER_AVAILABLE,
            "bertscore": _BERTSCORE_AVAILABLE,
            "rouge": "rouge-score" if _ROUGE_AVAILABLE else "built-in",
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(
        self,
        ground_truth_path: str | Path,
        predictions: Optional[Dict[str, str]] = None,
        summaries_pred: Optional[Dict[str, str]] = None,
        summaries_ref: Optional[Dict[str, str]] = None,
        write_csv: bool = True,
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
        write_csv : bool
            Write ``failure_analysis.csv`` to ``output_dir``.

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

        return self.run_records(
            ground_truth, predictions, summaries_pred, summaries_ref, write_csv
        )

    def run_records(
        self,
        ground_truth: List[Dict[str, Any]],
        predictions: Optional[Dict[str, str]] = None,
        summaries_pred: Optional[Dict[str, str]] = None,
        summaries_ref: Optional[Dict[str, str]] = None,
        write_csv: bool = True,
    ) -> List[EvalResult]:
        """Same as :meth:`run`, for ground truth already loaded in memory."""
        if not isinstance(ground_truth, list):
            raise ValueError("Ground truth must be a JSON array of sample objects.")
        logger.info(f"Loaded {len(ground_truth)} ground-truth samples.")

        predictions = predictions or {}
        summaries_pred = summaries_pred or {}
        summaries_ref = summaries_ref or {}
        results: List[EvalResult] = []

        for sample in ground_truth:
            if not isinstance(sample, dict) or "audio_file" not in sample:
                raise ValueError('Each ground-truth sample needs an "audio_file" key.')
            audio_file = sample["audio_file"]
            ref_transcript = sample.get("transcript", "") or ""
            pred_transcript = predictions.get(audio_file, "") or ""

            result = EvalResult(
                audio_file=audio_file,
                wer=None,
                reference_transcript=ref_transcript,
                predicted_transcript=pred_transcript,
                condition=sample.get("condition", "") or "",
                language=sample.get("language") or detect_language(ref_transcript),
                reference_summary=summaries_ref.get(audio_file, sample.get("summary", "")) or "",
                predicted_summary=summaries_pred.get(audio_file, "") or "",
            )

            # ── WER + word alignment ───────────────────────────────────
            if _JIWER_AVAILABLE and ref_transcript.strip() and pred_transcript.strip():
                self._score_transcript(result)
                if result.wer is not None and result.wer > WER_THRESHOLD:
                    result.is_failure = True
                    result.failure_reasons.append(
                        f"WER={result.wer:.3f} > {WER_THRESHOLD}"
                    )

            # ── ROUGE (on summaries) ───────────────────────────────────
            if result.reference_summary and result.predicted_summary:
                if self._rouge is not None:
                    scores = self._rouge.score(
                        result.reference_summary, result.predicted_summary
                    )
                    result.rouge1_f = scores["rouge1"].fmeasure
                    result.rouge2_f = scores["rouge2"].fmeasure
                    result.rougeL_f = scores["rougeL"].fmeasure
                else:
                    result.rouge1_f, result.rouge2_f, result.rougeL_f = simple_rouge(
                        result.reference_summary, result.predicted_summary
                    )

            results.append(result)

        # ── BERTScore (one batched call for every summary pair) ────────
        if self.use_bertscore:
            self._score_bertscore(results)

        # ── Write failure analysis CSV ─────────────────────────────────
        if write_csv:
            self._write_failure_csv(results)

        # ── Log summary ────────────────────────────────────────────────
        failures = [r for r in results if r.is_failure]
        scored = [r.wer for r in results if r.wer is not None]
        avg_wer = sum(scored) / len(scored) if scored else 0.0
        logger.info(
            f"Evaluation complete: {len(results)} samples ({len(scored)} with "
            f"predictions), avg WER={avg_wer:.3f}, {len(failures)} failures."
        )

        return results

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _score_transcript(result: EvalResult) -> None:
        """Fill WER, error counts and the word alignment via ``jiwer``."""
        reference = normalise_for_wer(result.reference_transcript)
        prediction = normalise_for_wer(result.predicted_transcript)
        if not reference or not prediction:
            return
        out = jiwer.process_words(reference, prediction)
        result.wer = float(out.wer)
        result.hits = out.hits
        result.substitutions = out.substitutions
        result.deletions = out.deletions
        result.insertions = out.insertions

        ref_words, hyp_words = out.references[0], out.hypotheses[0]
        result.alignment = [
            {
                "op": chunk.type,
                "ref": ref_words[chunk.ref_start_idx : chunk.ref_end_idx],
                "hyp": hyp_words[chunk.hyp_start_idx : chunk.hyp_end_idx],
            }
            for chunk in out.alignments[0]
        ]

    def _score_bertscore(self, results: List[EvalResult]) -> None:
        pairs = [r for r in results if r.reference_summary and r.predicted_summary]
        if not pairs:
            return
        P, R, F1 = bert_score_fn(
            [r.predicted_summary for r in pairs],
            [r.reference_summary for r in pairs],
            model_type=self.bert_model,
            lang="en",
            verbose=False,
        )
        for i, r in enumerate(pairs):
            r.bert_precision = P[i].item()
            r.bert_recall = R[i].item()
            r.bert_f1 = F1[i].item()
            if r.bert_f1 < BERTSCORE_THRESHOLD:
                r.is_failure = True
                r.failure_reasons.append(
                    f"BERTScore F1={r.bert_f1:.3f} < {BERTSCORE_THRESHOLD} "
                    "(possible hallucination)"
                )

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
                        "wer": "" if r.wer is None else f"{r.wer:.4f}",
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

        Samples without a prediction get ``NaN`` WER so they are excluded
        from averages rather than counted as perfect.

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
                    "Language": r.language,
                    "WER": float("nan") if r.wer is None else round(r.wer, 4),
                    "BERTScore_F1": round(r.bert_f1, 4),
                    "ROUGE-1": round(r.rouge1_f, 4),
                    "ROUGE-2": round(r.rouge2_f, 4),
                    "ROUGE-L": round(r.rougeL_f, 4),
                    "Failure": "FAIL" if r.is_failure else ("N/A" if r.wer is None else "PASS"),
                    "Reasons": "; ".join(r.failure_reasons) if r.failure_reasons else "",
                }
            )
        return pd.DataFrame(rows)
