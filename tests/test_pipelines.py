"""Quiz schema, offline fallbacks and the evaluation pipeline (no ML stack needed)."""

from __future__ import annotations

import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from src.pipelines.eval_pipeline import EvalPipeline, detect_language, normalise_for_wer, simple_rouge
from src.pipelines.fallback import (
    EnergyVAD,
    generate_cloze_quiz,
    generate_extractive_notes,
    key_terms,
    read_wav,
    split_sentences,
    tokenize,
)
from src.pipelines.quiz_schema import normalise_quiz, resolve_answer_index, strip_option_label
from src.services.benchmark import (
    build_report,
    lecture_meta,
    load_default_dataset,
    load_measurement,
    parse_predictions,
)

ROOT = Path(__file__).resolve().parent.parent

RAFT = (
    "Today we will study the Raft consensus algorithm. Raft maintains a replicated state machine "
    "across a cluster of nodes. The algorithm elects a leader node, and the leader handles all "
    "client requests. Log replication ensures every follower node stores the same sequence of "
    "commands. If the leader crashes, a new leader election starts after an election timeout. "
    "Each term has at most one leader, which guarantees safety. Followers reject log entries that "
    "conflict with the leader log. A command is committed once a majority of nodes have replicated it."
)


def write_wav(path: Path, samples: np.ndarray, rate: int = 16_000, width: int = 2) -> None:
    pcm = (np.clip(samples, -1, 1) * (2 ** (8 * width - 1) - 1)).astype(f"<i{width}")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(width)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


class QuizSchema(unittest.TestCase):
    OPTIONS = ["A) Red", "B) Green", "C) Blue", "D) Black"]

    def test_labels_are_stripped(self):
        self.assertEqual(strip_option_label("B) Green"), "Green")
        self.assertEqual(strip_option_label("(c) Blue"), "Blue")
        self.assertEqual(strip_option_label("Plain text"), "Plain text")

    def test_every_answer_style_resolves(self):
        for answer in ("B", "B)", "b", "B) Green", "Green", "green", "(B) Green"):
            self.assertEqual(resolve_answer_index(self.OPTIONS, answer), 1, answer)

    def test_answer_index_wins_and_bad_hint_falls_through(self):
        self.assertEqual(resolve_answer_index(self.OPTIONS, "Green", hint=2), 2)
        self.assertEqual(resolve_answer_index(self.OPTIONS, "Green", hint=9), 1)

    def test_normalise_drops_what_it_cannot_trust(self):
        quiz = normalise_quiz(
            [
                {"question": "Q1", "options": self.OPTIONS, "correct_answer": "C", "explanation": "e"},
                {"question": "No key", "options": self.OPTIONS},
                {"question": "Unknown key", "options": self.OPTIONS, "correct_answer": "Purple"},
                {"question": "", "options": self.OPTIONS, "correct_answer": "A"},
                {"question": "Too few", "options": ["only"], "correct_answer": "only"},
                "not a dict",
            ]
        )
        self.assertEqual(len(quiz), 1)
        self.assertEqual(quiz[0]["options"], ["Red", "Green", "Blue", "Black"])
        self.assertEqual(quiz[0]["answer_index"], 2)
        self.assertEqual(quiz[0]["correct_answer"], "Blue")


class Text(unittest.TestCase):
    def test_danda_splits_sentences(self):
        self.assertEqual(len(split_sentences("यह पहला वाक्य है। यह दूसरा है।")), 2)

    def test_indic_words_keep_their_vowel_signs(self):
        self.assertEqual(tokenize("मशीन लर्निंग"), ["मशीन", "लर्निंग"])

    def test_key_terms_prefers_repeated_phrases(self):
        terms = [t for t, _ in key_terms("loss function matters. the loss function is key. loss function again.")]
        self.assertEqual(terms[0], "loss function")


class ExtractiveNotes(unittest.TestCase):
    def test_structure(self):
        notes = generate_extractive_notes(RAFT)
        for heading in ("# ", "## Overview", "## Key Concepts", "## Lecture Flow", "## Key Takeaways"):
            self.assertIn(heading, notes)

    def test_title_is_used(self):
        self.assertTrue(generate_extractive_notes(RAFT, "My Title").startswith("# My Title"))

    def test_short_transcript_is_not_padded_out(self):
        text = "Raft elects a leader node for each term."
        notes = generate_extractive_notes(text)
        self.assertNotIn("## Lecture Flow", notes)
        self.assertEqual(notes.count(text), 1)

    def test_empty(self):
        self.assertEqual(generate_extractive_notes("   "), "_No transcript provided._")


class ClozeQuiz(unittest.TestCase):
    def test_a_transcript_with_no_punctuation_still_gets_a_quiz(self):
        words = RAFT.replace(".", "").replace(",", "").lower()
        self.assertEqual(len(generate_cloze_quiz(words, 3)), 3)

    def test_questions_are_valid_and_answerable(self):
        quiz = generate_cloze_quiz(RAFT, 5)
        self.assertGreaterEqual(len(quiz), 3)
        for q in quiz:
            self.assertEqual(len(q["options"]), 4)
            self.assertEqual(len(set(q["options"])), 4)
            self.assertEqual(q["options"][q["answer_index"]], q["correct_answer"])
            self.assertIn("_____", q["question"])
            self.assertNotIn(q["correct_answer"], q["question"])

    def test_deterministic(self):
        self.assertEqual(generate_cloze_quiz(RAFT, 3), generate_cloze_quiz(RAFT, 3))

    def test_too_little_text_gives_nothing(self):
        self.assertEqual(generate_cloze_quiz("Too short.", 5), [])


class EnergyVad(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def bursts(self, spans, total=10.0, rate=16_000):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 0.001, int(total * rate))
        for start, end in spans:
            a, b = int(start * rate), int(end * rate)
            x[a:b] += 0.4 * np.sin(2 * np.pi * 180 * np.arange(b - a) / rate)
        return x

    def test_finds_speech_between_silences(self):
        path = self.dir / "a.wav"
        write_wav(path, self.bursts([(1, 3), (6, 8)]))
        result = EnergyVAD(max_chunk_sec=2.5, min_chunk_sec=0.5).process_wav(path)
        self.assertEqual(len(result.spans), 2)
        self.assertAlmostEqual(result.spans[0][0], 1, delta=0.15)
        self.assertAlmostEqual(result.spans[1][1], 8, delta=0.15)
        self.assertAlmostEqual(result.duration, 10, delta=0.01)

    def test_short_gaps_are_bridged(self):
        path = self.dir / "b.wav"
        write_wav(path, self.bursts([(1, 3), (3.1, 5)]))
        self.assertEqual(EnergyVAD().process_wav(path).raw_segments, 1)

    def test_long_speech_is_capped_at_max_chunk(self):
        path = self.dir / "c.wav"
        write_wav(path, self.bursts([(0.5, 19.5)], total=20))
        spans = EnergyVAD(max_chunk_sec=5, min_chunk_sec=1).process_wav(path).spans
        self.assertGreaterEqual(len(spans), 4)
        self.assertTrue(all(e - s <= 5.01 for s, e in spans))

    def test_silence_returns_the_whole_file(self):
        path = self.dir / "d.wav"
        write_wav(path, np.zeros(16_000))
        self.assertEqual(EnergyVAD().process_wav(path).spans, [(0.0, 1.0)])

    def test_reads_other_pcm_widths_and_stereo(self):
        path = self.dir / "e.wav"
        write_wav(path, np.ones(1000) * 0.5, width=4)
        samples, rate = read_wav(path)
        self.assertEqual(rate, 16_000)
        self.assertAlmostEqual(float(samples.mean()), 0.5, delta=0.01)

    def test_non_pcm_is_a_clear_error(self):
        path = self.dir / "f.wav"
        path.write_bytes(b"this is not a wav file at all")
        with self.assertRaises(ValueError):
            read_wav(path)


class Evaluation(unittest.TestCase):
    def run_eval(self, records, predictions=None, **kw):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = EvalPipeline(output_dir=tmp, use_bertscore=False)
            return pipeline.run_records(records, predictions, write_csv=False, **kw)

    def test_missing_prediction_is_not_a_perfect_score(self):
        [result] = self.run_eval([{"audio_file": "a", "transcript": "hello world"}])
        self.assertIsNone(result.wer)
        self.assertFalse(result.is_failure)

    def test_wer_and_alignment(self):
        [r] = self.run_eval(
            [{"audio_file": "a", "transcript": "the cat sat on the mat"}],
            {"a": "the cat sit on mat"},
        )
        self.assertAlmostEqual(r.wer, 2 / 6)
        self.assertTrue(r.is_failure)
        self.assertEqual((r.hits, r.substitutions, r.deletions, r.insertions), (4, 1, 1, 0))
        self.assertEqual({a["op"] for a in r.alignment}, {"equal", "substitute", "delete"})

    def test_report_keeps_unscored_samples_out_of_the_average(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = EvalPipeline(output_dir=tmp, use_bertscore=False)
            results = pipeline.run_records(
                [{"audio_file": "a", "transcript": "one two"}, {"audio_file": "b", "transcript": "x"}],
                {"a": "one two"},
                write_csv=False,
            )
            df = pipeline.generate_report_df(results)
        self.assertEqual(df["WER"].mean(), 0)
        self.assertEqual(list(df["Failure"]), ["PASS", "N/A"])

    def test_bad_ground_truth_is_rejected(self):
        with self.assertRaises(ValueError):
            self.run_eval({"not": "a list"})
        with self.assertRaises(ValueError):
            self.run_eval([{"transcript": "no audio_file key"}])

    def test_builtin_rouge(self):
        self.assertEqual(simple_rouge("the cat sat", "the cat sat"), (1.0, 1.0, 1.0))
        self.assertEqual(simple_rouge("the cat", "dogs run"), (0.0, 0.0, 0.0))

    def test_language_comes_from_the_script(self):
        self.assertEqual(detect_language("மொழி"), "Tamil")
        self.assertEqual(detect_language("ಕನ್ನಡ"), "Kannada")
        self.assertEqual(detect_language("भाषा"), "Hindi")
        self.assertEqual(detect_language("hello"), "English")

    def test_failure_csv_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            refs = Path(tmp) / "refs.json"
            refs.write_text(json.dumps([{"audio_file": "a", "transcript": "one two three"}]), encoding="utf-8")
            pipeline = EvalPipeline(output_dir=tmp, use_bertscore=False)
            pipeline.run(refs, predictions={"a": "completely different words"})
            self.assertIn("a,1.0000", (Path(tmp) / "failure_analysis.csv").read_text(encoding="utf-8"))


class Normalisation(unittest.TestCase):
    def test_punctuation_and_case_are_not_word_errors(self):
        self.assertEqual(normalise_for_wer("Hello, World!"), "hello world")
        self.assertEqual(normalise_for_wer("यह है। वह है॥"), "यह है वह है")

    def test_combining_marks_survive(self):
        self.assertEqual(normalise_for_wer("கற்றல்"), "கற்றல்")
        self.assertEqual(normalise_for_wer("ಕಲಿಕೆ"), "ಕಲಿಕೆ")


class Benchmark(unittest.TestCase):
    def test_bundled_report_scores_every_lecture(self):
        predictions = ROOT / "outputs/transcripts/eval_inputs.json"
        records, payload = load_default_dataset(ROOT / "data/ground_truth/references.json", predictions)
        with tempfile.TemporaryDirectory() as tmp:
            report = build_report(records, payload, source="bundled", output_dir=Path(tmp))
        summary = report["summary"]
        self.assertEqual(summary["samples"], 40)
        self.assertEqual(summary["evaluated"], 40)
        self.assertTrue(0 < summary["avg_wer"] < 1)
        self.assertTrue(load_measurement(predictions)["asr_models"])
        self.assertEqual({r["name"] for r in report["by_language"]}, {"English", "Hindi", "Tamil", "Kannada"})
        self.assertEqual(len(report["by_domain"]), 6)

    def test_titles(self):
        self.assertEqual(lecture_meta("lecture_17_indexing_b_trees.wav")["title"], "Indexing B-Trees")
        self.assertEqual(lecture_meta("lecture_15_cuda_memory_hierarchy.wav")["title"], "CUDA Memory Hierarchy")
        self.assertEqual(lecture_meta("b.wav")["title"], "B")
        self.assertEqual(lecture_meta("anything.wav")["domain"], "Uploaded set")

    def test_both_prediction_formats(self):
        self.assertEqual(parse_predictions({"a.wav": "text"}), ({"a.wav": "text"}, {}, {}))
        rich = {"predictions": {"a": "x"}, "summaries_pred": {"a": "s"}, "summaries_ref": {"a": "r"}}
        self.assertEqual(parse_predictions(rich), ({"a": "x"}, {"a": "s"}, {"a": "r"}))
        with self.assertRaises(ValueError):
            parse_predictions(["not", "a", "dict"])


if __name__ == "__main__":
    unittest.main()
