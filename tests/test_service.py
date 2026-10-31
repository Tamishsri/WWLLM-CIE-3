"""PipelineHub (offline paths, plus live branches with fake engines) and the HTTP API."""

from __future__ import annotations

import io
import json
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from unittest.mock import patch

import uvicorn

from src.api.app import create_app
from src.config import ROOT, load_config
from src.pipelines.quiz_schema import normalise_quiz
from src.services.capabilities import STAGE_REQUIREMENTS, Capabilities
from src.services.engines import (
    PipelineHub,
    _complete_sentences,
    _distribute,
    asr_model_id,
    load_eval_predictions,
)
from src.services.files import FileStore
from src.services.result_cache import ResultCache
from src.services.sample import SampleLecture

SAMPLE = ROOT / "data/sample/gradient_descent.json"
EVAL_INPUTS = ROOT / "outputs/transcripts/eval_inputs.json"
RAFT = (
    "Today we will study the Raft consensus algorithm. Raft maintains a replicated state machine "
    "across a cluster of nodes. The algorithm elects a leader node, and the leader handles all "
    "client requests. Log replication ensures every follower node stores the same sequence of "
    "commands. If the leader crashes, a new leader election starts after an election timeout."
)


def caps(live: bool = False) -> Capabilities:
    modules = {m: live for reqs in STAGE_REQUIREMENTS.values() for m in reqs}
    return Capabilities(modules=modules, device={"probed": True})


def hub(live: bool = False, stored: bool = False) -> PipelineHub:
    sample = SampleLecture(SAMPLE, ROOT / "data/sample")
    cache = ResultCache(Path(tempfile.mkdtemp()), read_only=[sample.results_dir] if stored else [])
    return PipelineHub(load_config(), caps(live), sample, cache, load_eval_predictions(EVAL_INPUTS), demo_pace=0)


def events(gen):
    return list(gen)


def result(evs):
    assert evs[-1]["event"] == "result", evs[-1]
    return evs[-1]["data"]


class Offline(unittest.TestCase):
    """No models installed: offline engines, honest errors."""

    def setUp(self):
        self.hub = hub()

    def test_notes_fall_back_to_the_extractive_engine(self):
        data = result(events(self.hub.notes(RAFT)))
        self.assertEqual(data["mode"], "fallback")
        self.assertIn("## Overview", data["notes"])

    def test_quiz_falls_back_to_cloze(self):
        data = result(events(self.hub.quiz(RAFT, 3)))
        self.assertEqual(data["mode"], "fallback")
        self.assertEqual(len(data["quiz"]), 3)

    def test_fallback_output_is_never_stored_as_model_output(self):
        events(self.hub.notes(RAFT))
        self.assertIsNone(self.hub.cache.get("notes", {"transcript": " ".join(RAFT.split())}))

    def test_quiz_count_is_clamped(self):
        self.assertEqual(len(result(events(self.hub.quiz(RAFT, 1)))["quiz"]), 3)

    def test_translation_without_a_model_is_an_honest_error(self):
        evs = events(self.hub.translate("# Hello", [], "Hindi"))
        self.assertEqual(evs[-1]["event"], "error")
        self.assertIn("translation model", evs[-1]["message"])

    def test_empty_inputs_are_errors(self):
        for evs in (
            events(self.hub.notes("  ")),
            events(self.hub.quiz("", 5)),
            events(self.hub.translate("", [], "Hindi")),
            events(self.hub.translate("x", [], "Klingon")),
        ):
            self.assertEqual(evs[-1]["event"], "error")

    def test_known_eval_clip_replays_its_stored_prediction(self):
        path = ROOT / "data/ground_truth/audio/lecture_06_backpropagation_math.wav"
        data = result(events(self.hub.transcribe({"name": path.name}, path, "Hindi")))
        self.assertEqual(data["mode"], "replay")
        self.assertTrue(data["transcript"])

    def test_unknown_audio_gets_chunks_then_an_honest_error(self):
        evs = events(self.hub.transcribe({"name": "mine.wav"}, self.hub.sample.audio_path, "Hindi"))
        self.assertTrue(any(e.get("chunks") for e in evs))
        self.assertEqual(evs[-1]["event"], "error")

    def test_unreadable_audio_is_an_honest_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "broken.wav"
            bad.write_bytes(b"not audio")
            evs = events(self.hub.transcribe({"name": "broken.wav"}, bad, "Hindi"))
        self.assertEqual(evs[-1]["event"], "error")

    def test_distribute_keeps_every_word_in_order(self):
        text = "one two three four five six seven"
        parts = _distribute(text, [(0, 1), (1, 3), (3, 4)])
        self.assertEqual(" ".join(parts), text)
        self.assertEqual(len(parts), 3)
        self.assertEqual(_distribute(text, [(0, 1)]), [text])

    def test_a_sentence_cut_off_by_the_token_limit_is_dropped(self):
        self.assertEqual(_complete_sentences("One. Two is cut"), "One.")
        self.assertEqual(_complete_sentences("One.\n\n## Heading\n\nCut off here"), "One.")
        self.assertEqual(_complete_sentences("Finished.\n"), "Finished.")
        self.assertEqual(_complete_sentences("- **Bold ending**"), "- **Bold ending**")
        self.assertEqual(_complete_sentences("## Only a heading"), "## Only a heading")
        self.assertEqual(_complete_sentences("# Title\n\nBody"), "# Title\n\nBody")

    def test_status_shape(self):
        status = self.hub.status()
        self.assertEqual(status["mode"], "demo")
        self.assertEqual(set(status["stages"]), {"vad", "asr", "llm", "mt"})
        self.assertEqual(len(status["languages"]["translate"]), 10)


class StoredSample(unittest.TestCase):
    """The sample lecture works with no models, from real stored outputs."""

    def setUp(self):
        self.hub = hub(stored=True)
        self.record = self.hub.sample.record()
        self.transcript = result(events(self.hub.transcribe(self.record, self.hub.sample.audio_path, "Hindi")))

    def test_transcript_is_stored_model_output(self):
        self.assertEqual(self.transcript["mode"], "live")
        self.assertTrue(self.transcript["cached"])
        self.assertTrue(self.transcript["segments"])
        self.assertEqual(self.transcript["language"], "Hindi")

    def test_notes_stream_reassembles_exactly(self):
        evs = events(self.hub.notes(self.transcript["transcript"]))
        data = result(evs)
        self.assertEqual("".join(e["text"] for e in evs if e["event"] == "token"), data["notes"])
        self.assertTrue(data["cached"])

    def test_quiz_and_every_translation_are_stored(self):
        notes = result(events(self.hub.notes(self.transcript["transcript"])))["notes"]
        quiz = result(events(self.hub.quiz(self.transcript["transcript"], 5)))["quiz"]
        self.assertGreaterEqual(len(quiz), 3)
        self.assertEqual(len(self.hub.sample.translated_languages), 10)
        for language in self.hub.sample.translated_languages:
            data = result(events(self.hub.translate(notes, quiz, language)))
            self.assertTrue(data["cached"], language)
            self.assertTrue(data["notes"], language)
            for src, out in zip(quiz, data["quiz"]):
                self.assertEqual(src["answer_index"], out["answer_index"])

    def test_edited_transcript_does_not_get_the_stored_notes(self):
        data = result(events(self.hub.notes(self.transcript["transcript"] + " Extra sentence here.")))
        self.assertEqual(data["mode"], "fallback")

    def test_fresh_bypasses_the_store(self):
        data = result(events(self.hub.notes(self.transcript["transcript"], fresh=True)))
        self.assertEqual(data["mode"], "fallback")


class FakeAudio:
    def process_with_spans(self, path):
        return ["c1", "c2", "c3"], [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)]


class FakeASR:
    batch_size = 2
    model_id = "fake/asr"

    def transcribe(self, chunks):
        return [f"text {c}" for c in chunks]


class FakeLLM:
    model_id = "fake/llm"

    def stream_notes(self, transcript):
        yield from ["# Title", "\n\n", "Body"]

    def stream_quiz(self, transcript, num_questions):
        question = {"question": "Q", "options": ["a", "b"], "answer_index": 1, "correct_answer": "b", "explanation": ""}
        yield json.dumps([question] * num_questions)

    def count_questions(self, raw):
        return raw.count('"question"')

    def parse_quiz(self, raw):
        return normalise_quiz(json.loads(raw))


class FakeMT:
    model_id = "fake/mt"

    def translate(self, text, target_lang):
        return f"[{target_lang}] {text}"

    def translate_quiz(self, quiz, target_lang):
        return [{**q, "question": f"[{target_lang}] {q['question']}"} for q in quiz]


class Live(unittest.TestCase):
    """Real-model branches, with the engines swapped for fakes."""

    def setUp(self):
        self.hub = hub(live=True)
        self.hub._audio = FakeAudio()
        self.audio = self.hub.sample.audio_path

    def use(self, key, engine):
        self.hub._models = {key: engine}

    def test_transcribe_reports_per_batch_and_timestamps_segments(self):
        self.use("asr:Hindi", FakeASR())
        evs = events(self.hub.transcribe({"name": "a.wav"}, self.audio, "Hindi"))
        data = result(evs)
        self.assertEqual((data["mode"], data["engine"]), ("live", "fake/asr"))
        self.assertEqual(data["transcript"], "text c1 text c2 text c3")
        self.assertEqual(data["segments"][1], {"start": 2.0, "end": 3.0, "text": "text c2"})
        self.assertEqual([e["value"] for e in evs if e["event"] == "progress"], [0.6667, 1.0])

    def test_a_second_run_replays_and_fresh_reruns(self):
        self.use("asr:Hindi", FakeASR())
        events(self.hub.transcribe({"name": "a.wav"}, self.audio, "Hindi"))
        self.assertTrue(result(events(self.hub.transcribe({"name": "a.wav"}, self.audio, "Hindi")))["cached"])
        self.assertNotIn("cached", result(events(self.hub.transcribe({"name": "a.wav"}, self.audio, "Hindi", fresh=True))))

    def test_notes_stream_and_are_stored(self):
        self.use("llm", FakeLLM())
        evs = events(self.hub.notes("transcript"))
        self.assertEqual([e["text"] for e in evs if e["event"] == "token"], ["# Title", "\n\n", "Body"])
        self.assertEqual(result(evs)["notes"], "# Title\n\nBody")
        self.assertTrue(result(events(self.hub.notes("transcript")))["cached"])

    def test_quiz(self):
        self.use("llm", FakeLLM())
        evs = events(self.hub.quiz("transcript", 4))
        self.assertEqual(len(result(evs)["quiz"]), 4)
        self.assertTrue(any(e["event"] == "progress" for e in evs))

    def test_invalid_llm_quiz_falls_back_instead_of_failing(self):
        llm = FakeLLM()
        llm.parse_quiz = lambda raw: []
        self.use("llm", llm)
        data = result(events(self.hub.quiz(RAFT, 3)))
        self.assertEqual(data["mode"], "fallback")

    def test_translate_preserves_the_key(self):
        self.use("mt", FakeMT())
        quiz = [{"question": "Q", "options": ["a", "b"], "answer_index": 1, "correct_answer": "b"}]
        data = result(events(self.hub.translate("notes", quiz, "Hindi")))
        self.assertEqual(data["notes"], "[hin_Deva] notes")
        self.assertEqual(data["quiz"][0]["answer_index"], 1)
        self.assertEqual(data["mode"], "live")

    def test_only_one_large_model_stays_loaded(self):
        self.use("llm", FakeLLM())
        self.hub._model("mt", FakeMT)
        self.assertEqual(list(self.hub._models), ["mt"])

    def test_a_model_that_cannot_load_degrades_cleanly(self):
        def boom():
            raise OSError("not downloaded")

        self.hub.cfg = {**self.hub.cfg, "llm": {"model_id": "missing/model"}}
        self.assertIsNone(self.hub._model("llm", boom))
        with patch("src.services.engines.build_llm", side_effect=OSError("offline")):
            notes = result(events(self.hub.notes(RAFT)))
            quiz = result(events(self.hub.quiz(RAFT, 3)))
        self.assertEqual((notes["mode"], quiz["mode"]), ("fallback", "fallback"))
        with patch("src.services.engines.build_translator", side_effect=OSError("offline")):
            self.assertEqual(events(self.hub.translate("notes", [], "Hindi"))[-1]["event"], "error")

    def test_a_failed_vad_falls_back_to_the_energy_vad(self):
        class Broken:
            def process_with_spans(self, path):
                raise RuntimeError("codec")

        self.hub._audio = Broken()
        evs = events(self.hub.transcribe({"name": "mine.wav"}, self.audio, "Hindi"))
        self.assertTrue(any(e.get("chunks") for e in evs))

    def test_each_language_gets_its_own_checkpoint(self):
        self.assertIn("_hi_", asr_model_id(self.hub.cfg, "Hindi"))
        self.assertIn("_ta_", asr_model_id(self.hub.cfg, "Tamil"))
        self.assertEqual(asr_model_id({"asr": {"models": {"Tamil": "/local/ta.nemo"}}}, "Tamil"), "/local/ta.nemo")


class Cache(unittest.TestCase):
    def test_round_trip_and_read_only_layer(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = ResultCache(Path(tmp) / "a")
            first.put("notes", {"x": 1}, {"notes": "hello", "cached": True})
            second = ResultCache(Path(tmp) / "b", read_only=[Path(tmp) / "a"])
            stored = second.get("notes", {"x": 1})
            self.assertEqual(stored["notes"], "hello")
            self.assertTrue(stored["cached"])
            self.assertIn("created", stored)
            self.assertIsNone(second.get("notes", {"x": 2}))
            self.assertIsNone(second.get("quiz", {"x": 1}))

    def test_a_corrupt_entry_is_a_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = ResultCache(Path(tmp))
            cache.put("notes", {"x": 1}, {"notes": "hello"})
            next(Path(tmp).glob("*.json")).write_text("{broken", encoding="utf-8")
            self.assertIsNone(cache.get("notes", {"x": 1}))


class Store(unittest.TestCase):
    def test_rejects_bad_ids_and_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FileStore(Path(tmp))
            with self.assertRaises(ValueError):
                store.save("virus.exe", io.BytesIO(b"MZ"))
            self.assertIsNone(store.get("../../etc/passwd"))
            self.assertIsNone(store.get("0" * 32))


class Api(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = create_app(demo_pace=0)
        config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
        cls.server = uvicorn.Server(config)
        cls.thread = threading.Thread(target=cls.server.run, daemon=True)
        cls.thread.start()
        for _ in range(100):
            if cls.server.started:
                break
            time.sleep(0.05)
        port = cls.server.servers[0].sockets[0].getsockname()[1]
        cls.base = f"http://127.0.0.1:{port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.should_exit = True
        cls.thread.join(timeout=5)

    def get(self, path):
        with urllib.request.urlopen(self.base + path) as r:
            return r.status, r.read(), r.headers

    def post_json(self, path, body):
        req = urllib.request.Request(
            self.base + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"}
        )
        with urllib.request.urlopen(req) as r:
            return [json.loads(line) for line in r.read().splitlines() if line.strip()]

    def upload(self, name, data):
        boundary = uuid.uuid4().hex
        body = (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{name}"\r\n\r\n'.encode()
            + data
            + f"\r\n--{boundary}--\r\n".encode()
        )
        req = urllib.request.Request(
            self.base + "/api/files", data=body, headers={"content-type": f"multipart/form-data; boundary={boundary}"}
        )
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())

    def test_ui_is_served(self):
        status, body, headers = self.get("/")
        self.assertEqual(status, 200)
        self.assertIn(b"WWLLM Lecture Studio", body)
        self.assertEqual(headers["cache-control"], "no-cache")
        self.assertEqual(self.get("/assets/js/main.js")[0], 200)

    def test_status(self):
        data = json.loads(self.get("/api/status")[1])
        self.assertIn(data["mode"], {"demo", "hybrid", "live"})

    def test_sample_flow_over_http(self):
        record = json.loads(
            urllib.request.urlopen(urllib.request.Request(self.base + "/api/sample", method="POST")).read()
        )
        self.assertTrue(record["sample"])
        evs = self.post_json("/api/transcribe", {"file_id": "sample"})
        self.assertEqual(evs[-1]["event"], "result")
        notes = self.post_json("/api/notes", {"transcript": evs[-1]["data"]["transcript"]})
        self.assertEqual(notes[-1]["event"], "result")

    def test_audio_supports_range_requests(self):
        req = urllib.request.Request(self.base + "/api/files/sample", headers={"Range": "bytes=0-99"})
        with urllib.request.urlopen(req) as r:
            self.assertEqual(r.status, 206)
            self.assertEqual(len(r.read()), 100)

    def test_upload_round_trip_and_type_rejection(self):
        record = self.upload("talk.wav", (ROOT / "data/ground_truth/audio/lecture_01_ml_intro.wav").read_bytes())
        for suffix in (".wav", ".json"):
            self.addCleanup((ROOT / "data/uploads" / f"{record['id']}{suffix}").unlink, missing_ok=True)
        self.assertEqual(record["name"], "talk.wav")
        self.assertAlmostEqual(record["duration"], 14.08, places=1)
        self.assertEqual(self.get(f"/api/files/{record['id']}")[0], 200)
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.upload("evil.exe", b"MZ")
        self.assertEqual(ctx.exception.code, 415)

    def test_unknown_file_is_404(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.get("/api/files/nope")
        self.assertEqual(ctx.exception.code, 404)
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            self.post_json("/api/transcribe", {"file_id": "nope"})
        self.assertEqual(ctx.exception.code, 404)

    def test_malformed_json_body_does_not_crash(self):
        req = urllib.request.Request(self.base + "/api/notes", data=b"{not json")
        with urllib.request.urlopen(req) as r:
            self.assertEqual(json.loads(r.read().splitlines()[-1])["event"], "error")

    def test_benchmark_is_a_real_measurement(self):
        report = json.loads(self.get("/api/benchmark")[1])
        self.assertEqual(report["summary"]["samples"], 40)
        self.assertEqual(report["summary"]["evaluated"], 40)
        self.assertTrue(report["measurement"]["asr_models"])


if __name__ == "__main__":
    unittest.main()
