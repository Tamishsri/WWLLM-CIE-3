#!/usr/bin/env python3
"""
run_eval.py — Standalone Evaluation Runner
===========================================
Run WER + BERTScore + ROUGE evaluation from the command line without
starting the Streamlit UI.

Usage::

    python scripts/run_eval.py \\
        --ground-truth data/ground_truth/references.json \\
        --predictions  outputs/transcripts/predictions.json \\
        --output-dir   outputs/evaluations

The predictions JSON is a mapping of ``{audio_file: transcript}``, or the richer
``{"predictions", "summaries_pred", "summaries_ref"}`` format, which also scores
the summaries with ROUGE and BERTScore.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipelines.eval_pipeline import EvalPipeline
from src.services.benchmark import parse_predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="WWLLM Evaluation Runner — WER, BERTScore, ROUGE"
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        required=True,
        help="Path to ground-truth JSON file.",
    )
    parser.add_argument(
        "--predictions",
        type=str,
        default=None,
        help="Path to predictions JSON (mapping audio_file → transcript).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/evaluations",
        help="Directory for failure_analysis.csv output.",
    )
    args = parser.parse_args()

    predictions, summaries_pred, summaries_ref = {}, {}, {}
    if args.predictions:
        with open(args.predictions, "r", encoding="utf-8") as f:
            predictions, summaries_pred, summaries_ref = parse_predictions(json.load(f))

    evaluator = EvalPipeline(output_dir=args.output_dir)
    results = evaluator.run(
        args.ground_truth,
        predictions=predictions,
        summaries_pred=summaries_pred,
        summaries_ref=summaries_ref,
    )

    df = evaluator.generate_report_df(results)
    print("\n" + "=" * 70)
    print("WWLLM EVALUATION REPORT")
    print("=" * 70)
    print(df.to_string(index=False))
    print(f"\nAvg WER: {df['WER'].mean():.4f}")
    print(f"Failures (WER > 0.3): {len(df[df['Failure'] == 'FAIL'])}/{len(df)}")
    print(f"\nCSV written to: {args.output_dir}/failure_analysis.csv")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
