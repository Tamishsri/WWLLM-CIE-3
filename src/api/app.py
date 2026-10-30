"""
app.py — Starlette application for WWLLM Lecture Studio
=======================================================
Routes
------
``GET  /api/status``           engines, languages, device, sample metadata
``POST /api/files``            multipart upload → file record
``GET  /api/files/{id}``       stream an uploaded file (Range-capable, for <audio>)
``POST /api/sample``           register the sample lecture → file record
``POST /api/transcribe``       NDJSON stream: VAD → ASR
``POST /api/notes``            NDJSON stream: notes tokens
``POST /api/quiz``             NDJSON stream: MCQs
``POST /api/translate``        NDJSON stream: IndicTrans2
``GET  /api/benchmark``        evaluation report for the bundled 40-lecture set
``POST /api/benchmark``        evaluation report for an uploaded ground truth
``/``                          the single-page web UI in ``web/``
"""

from __future__ import annotations

import json
import mimetypes
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from loguru import logger
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles
from starlette.types import ASGIApp, Receive, Scope, Send

from src.config import ROOT, load_config
from src.services.benchmark import build_report, load_default_dataset, parse_predictions
from src.services.capabilities import detect_capabilities
from src.services.engines import PipelineHub, load_eval_predictions
from src.services.files import FileStore
from src.services.sample import SampleLecture

WEB_DIR = ROOT / "web"


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


def _ndjson(events: Iterator[Dict[str, Any]]) -> StreamingResponse:
    """Stream generator events as newline-delimited JSON.

    Exceptions inside the generator become an ``error`` event, so the UI
    always gets a terminal message instead of a silently dropped stream.
    """

    def body() -> Iterator[str]:
        try:
            for event in events:
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as exc:  # surfaced to the UI, full trace in the log
            logger.exception("Pipeline stream failed")
            yield json.dumps(
                {
                    "event": "error",
                    "message": f"{type(exc).__name__}: {exc}",
                    "hint": "See the server console for the full traceback.",
                },
                ensure_ascii=False,
            ) + "\n"

    return StreamingResponse(
        body(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _json_body(request: Request) -> Dict[str, Any]:
    try:
        data = await request.json()
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


class _RevalidateStatic:
    """Ask browsers to revalidate UI assets so edits show up on reload."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"].startswith("/api/"):
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message: Dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                headers = [(k, v) for k, v in message.get("headers", []) if k != b"cache-control"]
                headers.append((b"cache-control", b"no-cache"))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, send_wrapper)


def create_app(demo_pace: float = 1.0) -> Starlette:
    """Build the ASGI app (models load lazily on first request)."""
    cfg = load_config()
    caps = detect_capabilities()
    caps.probe_device_async()

    eval_cfg = cfg.get("evaluation", {})
    references_path = ROOT / eval_cfg.get("references", "data/ground_truth/references.json")
    predictions_path = ROOT / eval_cfg.get("predictions", "outputs/transcripts/eval_inputs.json")
    output_dir = ROOT / eval_cfg.get("output_dir", "outputs/evaluations")

    store = FileStore(ROOT / "data" / "uploads")
    sample = SampleLecture(ROOT / "data" / "sample" / "gradient_descent.json", ROOT / "data" / "sample")
    hub = PipelineHub(
        cfg,
        caps,
        sample,
        eval_predictions=load_eval_predictions(predictions_path),
        demo_pace=demo_pace,
    )
    bench_cache: Dict[Any, Dict[str, Any]] = {}

    def resolve(file_id: str) -> Optional[tuple]:
        if file_id == "sample":
            record = sample.record()
            return record, sample.audio_path
        record = store.get(file_id)
        return (record, store.path(record)) if record else None

    # ── Routes ─────────────────────────────────────────────────────────
    async def status(request: Request) -> JSONResponse:
        return JSONResponse(hub.status())

    async def upload(request: Request) -> JSONResponse:
        form = await request.form(max_files=1, max_fields=4)
        upload_file = form.get("file")
        if upload_file is None or not hasattr(upload_file, "filename"):
            return _error(400, 'Send the media as multipart field "file".')
        try:
            record = await run_in_threadpool(store.save, upload_file.filename, upload_file.file)
        except ValueError as exc:
            return _error(415, str(exc))
        finally:
            await upload_file.close()
        logger.info(f"Stored upload {record['name']} → {record['id']} ({record['size'] / 1e6:.1f} MB)")
        return JSONResponse(record)

    async def get_file(request: Request):
        found = resolve(request.path_params["file_id"])
        if not found:
            return _error(404, "Unknown file id.")
        record, path = found
        media_type = mimetypes.guess_type(record["name"])[0] or "application/octet-stream"
        return FileResponse(path, media_type=media_type)

    async def load_sample(request: Request) -> JSONResponse:
        record = await run_in_threadpool(sample.record)
        return JSONResponse(record)

    async def transcribe(request: Request):
        body = await _json_body(request)
        found = await run_in_threadpool(resolve, str(body.get("file_id", "")))
        if not found:
            return _error(404, "Unknown file id — upload the recording again.")
        record, path = found
        return _ndjson(hub.transcribe(record, path, str(body.get("language", "Hindi"))))

    async def notes(request: Request):
        body = await _json_body(request)
        return _ndjson(hub.notes(str(body.get("transcript", "")), body.get("title") or None))

    async def quiz(request: Request):
        body = await _json_body(request)
        try:
            count = int(body.get("num_questions", 5))
        except (TypeError, ValueError):
            count = 5
        return _ndjson(hub.quiz(str(body.get("transcript", "")), count))

    async def translate(request: Request):
        body = await _json_body(request)
        quiz_items = body.get("quiz") if isinstance(body.get("quiz"), list) else []
        return _ndjson(
            hub.translate(str(body.get("notes", "")), quiz_items, str(body.get("target", "Hindi")))
        )

    async def benchmark(request: Request) -> JSONResponse:
        bertscore = request.query_params.get("bertscore") == "1"
        if request.method == "GET":
            if not references_path.exists():
                return _error(404, f"Ground truth not found: {references_path}")
            key = (
                references_path.stat().st_mtime,
                predictions_path.stat().st_mtime if predictions_path.exists() else 0,
                bertscore,
            )
            if key not in bench_cache:
                def run_default() -> Dict[str, Any]:
                    records, payload = load_default_dataset(references_path, predictions_path)
                    report = build_report(
                        records, payload, source="bundled", output_dir=output_dir, bertscore=bertscore
                    )
                    report["simulated"] = bool(eval_cfg.get("simulated", False))
                    return report

                t0 = time.perf_counter()
                bench_cache.clear()
                bench_cache[key] = await run_in_threadpool(run_default)
                logger.info(f"Benchmark computed in {time.perf_counter() - t0:.2f}s")
            return JSONResponse(bench_cache[key])

        form = await request.form(max_files=2, max_fields=4)
        try:
            refs_file = form.get("references")
            if refs_file is None or not hasattr(refs_file, "read"):
                return _error(400, 'Upload the ground truth as multipart field "references".')
            records = json.loads(await refs_file.read())
            payload = ({}, {}, {})
            preds_file = form.get("predictions")
            if preds_file is not None and hasattr(preds_file, "read"):
                raw = await preds_file.read()
                if raw.strip():
                    payload = parse_predictions(json.loads(raw))
            name = getattr(refs_file, "filename", "") or "uploaded set"
            report = await run_in_threadpool(
                lambda: build_report(
                    records, payload, source=name, output_dir=output_dir, bertscore=bertscore
                )
            )
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return _error(400, f"Invalid JSON: {exc}")
        except ValueError as exc:
            return _error(400, str(exc))
        return JSONResponse(report)

    routes = [
        Route("/api/status", status),
        Route("/api/files", upload, methods=["POST"]),
        Route("/api/files/{file_id}", get_file),
        Route("/api/sample", load_sample, methods=["POST"]),
        Route("/api/transcribe", transcribe, methods=["POST"]),
        Route("/api/notes", notes, methods=["POST"]),
        Route("/api/quiz", quiz, methods=["POST"]),
        Route("/api/translate", translate, methods=["POST"]),
        Route("/api/benchmark", benchmark, methods=["GET", "POST"]),
        Mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web"),
    ]
    app = Starlette(routes=routes)
    app.add_middleware(_RevalidateStatic)
    app.state.hub = hub
    return app
