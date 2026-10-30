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

import uvicorn

from src.api.app import create_app
from src.config import ROOT, load_config
from src.services.capabilities import STAGE_REQUIREMENTS, Capabilities
from src.services.engines import PipelineHub, _distribute, load_eval_predictions
from src.services.files import FileStore
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


def hub(live: bool = False, tmp: Path | None = None) -> PipelineHub:
    sample = SampleLecture(SAMPLE, tmp or Path(tempfile.mkdtemp()))
    return PipelineHub(load_config(), caps(live), sample, load_eval_predictions(EVAL_INPUTS), demo_pace=0)


def events(gen):
    return list(gen)


def result(evs):
    assert evs[-1]["event"] == "result", evs[-1]
    return evs[-1]["data"]


class Offline(unittest.TestCase):
    def setUp(self):
        self.hub = hub()
        self.record = self.hub.sample.record()

    def test_sample_transcript(self):
        evs = events(self.hub.transcribe(self.record, self.hub.sample.audio_path, "Hindi"))
        data = result(evs)
        self.assertEqual(data["mode"], "sample")
        self.assertEqual(len(data["segments"]), 4)
        self.assertTrue(any(e["event"] == "stage" and e.get("chunks") for e in evs))

    def test_sample_notes_stream_reassembles_exactly(self):
        evs = events(self.hub.notes(self.hub.sample.transcript))
        text = "".join(e["text"] for e in evs if e["event"] == "token")
        self.assertEqual(text, self.hub.sample.notes)
        self.assertEqual(result(evs)["mode"], "sample")

    def test_edited_transcript_uses_the_fallback_not_the_sample(self):
        data = result(events(self.hub.notes(self.hub.sample.transcript + " Extra sentence here.")))
        self.assertEqual(data["mode"], "fallback")

    def test_quiz_is_capped_by_the_sample(self):
        evs = events(self.hub.quiz(self.hub.sample.transcript, 8))
        self.assertEqual(len(result(evs)["quiz"]), 5)
        self.assertTrue(any(e["event"] == "log" for e in evs))

    def test_quiz_count_is_clamped(self):
        self.assertEqual(len(result(events(self.hub.quiz(self.hub.sample.transcript, 1)))["quiz"]), 3)

    def test_fallback_quiz(self):
        data = result(events(self.hub.quiz(RAFT, 3)))
        self.assertEqual(data["mode"], "fallback")
        self.assertEqual(len(data["quiz"]), 3)

    def test_sample_translation_keeps_the_answer_key(self):
        quiz = self.hub.sample.quiz()
        data = result(events(self.hub.translate(self.hub.sample.notes, quiz, "Tamil")))
        self.assertEqual(data["mode"], "sample")
        for src, out in zip(quiz, data["quiz"]):
            self.assertEqual(src["answer_index"], out["answer_index"])
            self.assertEqual(out["correct_answer"], out["options"][out["answer_index"]])

    def test_uncurated_language_is_an_honest_error(self):
        evs = events(self.hub.translate(self.hub.sample.notes, [], "Telugu"))
        self.assertEqual(evs[-1]["event"], "error")
        self.assertIn("IndicTrans2", evs[-1]["message"])

    def test_custom_text_cannot_be_translated_offline(self):
        evs = events(self.hub.translate("# Hello", [], "Hindi"))
        self.assertEqual(evs[-1]["event"], "error")

    def test_empty_inputs_are_errors(self):
        for evs in (
            events(self.hub.notes("  ")),
            events(self.hub.quiz("", 5)),
            events(self.hub.translate("", [], "Hindi")),
            events(self.hub.translate("x", [], "Klingon")),
        ):
            self.assertEqual(evs[-1]["event"], "error")

    def test_known_eval_clip_replays_its_simulated_prediction(self):
        path = ROOT / "data/ground_truth/audio/lecture_06_backpropagation_math.wav"
        record = {"name": path.name}
        data = result(events(self.hub.transcribe(record, path, "Hindi")))
        self.assertEqual(data["mode"], "replay")
        self.assertTrue(data["transcript"])

    def test_unknown_audio_gets_chunks_then_an_honest_error(self):
        evs = events(self.hub.transcribe({"name": "mine.wav"}, self.hub.sample.audio_path, "Hindi"))
        self.assertTrue(any(e.get("chunks") for e in evs))
        self.assertEqual(evs[-1]["event"], "error")

    def test_non_wav_needs_the_real_vad(self):
        evs = events(self.hub.transcribe({"name": "x.mp3"}, Path("x.mp3"), "Hindi"))
        self.assertEqual(evs[-1]["event"], "error")
        self.assertIn(".wav", evs[-1]["message"])

    def test_distribute_keeps_every_word_in_order(self):
        text = "one two three four five six seven"
        parts = _distribute(text, [(0, 1), (1, 3), (3, 4)])
        self.assertEqual(" ".join(parts), text)
        self.assertEqual(len(parts), 3)
        self.assertEqual(_distribute(text, [(0, 1)]), [text])

    def test_status_shape(self):
        status = self.hub.status()
        self.assertEqual(status["mode"], "demo")
        self.assertEqual(set(status["stages"]), {"vad", "asr", "llm", "mt"})
        self.assertEqual(len(status["languages"]["translate"]), 10)


class FakeAudio:
    def process_with_spans(self, path):
        return ["c1", "c2", "c3"], [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)]


class FakeASR:
    batch_size = 2

    def transcribe(self, chunks):
        return [f"text {c}" for c in chunks]


class FakeLLM:
    model_id = "fake/llm"

    def stream_notes(self, transcript):
        yield from ["# Title", "\n\n", "Body"]

    def generate_quiz(self, transcript, num_questions):
        return [
            {"question": "Q", "options": ["a", "b"], "answer_index": 1, "correct_answer": "b", "explanation": ""}
        ] * num_questions


class FakeMT:
    model_id = "fake/mt"

    def translate(self, text, target_lang):
        return f"[{target_lang}] {text}"

    def translate_quiz(self, quiz, target_lang):
        return [{**q, "question": f"[{target_lang}] {q['question']}"} for q in quiz]


class LiveBranches(unittest.TestCase):
    """The real-model paths, with engines swapped for fakes."""

    def setUp(self):
        self.hub = hub(live=True)
        self.hub._audio = FakeAudio()
        self.hub._asr = FakeASR()
        self.hub._asr_model_id = self.hub.asr_model_id("Hindi")
        self.hub._llm = FakeLLM()
        self.hub._mt = FakeMT()

    def test_transcribe_reports_per_batch_and_timestamps_segments(self):
        evs = events(self.hub.transcribe({"name": "a.wav"}, Path("a.wav"), "Hindi"))
        data = result(evs)
        self.assertEqual(data["mode"], "live")
        self.assertEqual(data["transcript"], "text c1 text c2 text c3")
        self.assertEqual(data["segments"][1], {"start": 2.0, "end": 3.0, "text": "text c2"})
        self.assertEqual([e["value"] for e in evs if e["event"] == "progress"], [0.6667, 1.0])

    def test_notes_stream_tokens(self):
        evs = events(self.hub.notes("transcript"))
        self.assertEqual([e["text"] for e in evs if e["event"] == "token"], ["# Title", "\n\n", "Body"])
        data = result(evs)
        self.assertEqual((data["notes"], data["mode"]), ("# Title\n\nBody", "live"))

    def test_live_model_beats_the_curated_sample(self):
        data = result(events(self.hub.notes(self.hub.sample.transcript)))
        self.assertEqual(data["mode"], "live")

    def test_quiz(self):
        self.assertEqual(len(result(events(self.hub.quiz("transcript", 4)))["quiz"]), 4)

    def test_empty_llm_quiz_is_a_retryable_error(self):
        self.hub._llm.generate_quiz = lambda *a, **k: []
        self.assertEqual(events(self.hub.quiz("transcript", 4))[-1]["event"], "error")

    def test_translate_preserves_the_key(self):
        quiz = [{"question": "Q", "options": ["a", "b"], "answer_index": 1, "correct_answer": "b"}]
        data = result(events(self.hub.translate("notes", quiz, "Hindi")))
        self.assertEqual(data["notes"], "[hin_Deva] notes")
        self.assertEqual(data["quiz"][0]["answer_index"], 1)
        self.assertEqual(data["mode"], "live")

    def test_each_language_gets_its_own_checkpoint(self):
        self.assertIn("_hi_", self.hub.asr_model_id("Hindi"))
        self.assertIn("_ta_", self.hub.asr_model_id("Tamil"))
        self.assertIn("_kn_", self.hub.asr_model_id("Kannada"))
        self.hub.cfg = {"asr": {"models": {"Tamil": "/local/ta.nemo"}}}
        self.assertEqual(self.hub.asr_model_id("Tamil"), "/local/ta.nemo")


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
        self.assertAlmostEqual(record["duration"], 2.5, places=1)
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

    def test_benchmark_is_labelled_simulated(self):
        report = json.loads(self.get("/api/benchmark")[1])
        self.assertTrue(report["simulated"])
        self.assertEqual(report["summary"]["failures"], 11)


if __name__ == "__main__":
    unittest.main()
