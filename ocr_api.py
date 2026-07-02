"""FastAPI service exposing PDF → OCR jobs via an asyncio.Queue worker."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

import structlog
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

OcrFn = Callable[[Path], str]

log = structlog.get_logger("ocr_api")


def _write_job(jobs_dir: Path, job_id: str, **fields) -> dict:
    path = jobs_dir / f"{job_id}.json"
    data = json.loads(path.read_text()) if path.exists() else {"job_id": job_id}
    data.update(fields)
    path.write_text(json.dumps(data))
    return data


def _read_job(jobs_dir: Path, job_id: str) -> dict | None:
    path = jobs_dir / f"{job_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


async def _run_worker(
    queue: asyncio.Queue, jobs_dir: Path, results_dir: Path, ocr_fn: OcrFn
) -> None:
    log.info("worker.started")
    while True:
        job_id = await queue.get()
        structlog.contextvars.bind_contextvars(job_id=job_id)
        started = time.monotonic()
        try:
            job = _write_job(jobs_dir, job_id, status="running")
            pdf_path = Path(job["pdf_path"])
            size = pdf_path.stat().st_size if pdf_path.exists() else None
            log.info("job.running", pdf_path=str(pdf_path), pdf_bytes=size)
            text = await asyncio.to_thread(ocr_fn, pdf_path)
            result_path = results_dir / f"{job_id}.txt"
            result_path.write_text(text, encoding="utf-8")
            _write_job(
                jobs_dir, job_id, status="completed", result_path=str(result_path)
            )
            log.info(
                "job.completed",
                elapsed_s=round(time.monotonic() - started, 3),
                chars=len(text),
                result_path=str(result_path),
            )
        except Exception as exc:  # noqa: BLE001
            _write_job(jobs_dir, job_id, status="failed", error=repr(exc))
            log.exception(
                "job.failed",
                elapsed_s=round(time.monotonic() - started, 3),
                error=repr(exc),
            )
        finally:
            structlog.contextvars.unbind_contextvars("job_id")
            queue.task_done()


def create_app(*, data_dir: Path, ocr_fn: OcrFn) -> FastAPI:
    uploads_dir = data_dir / "uploads"
    jobs_dir = data_dir / "jobs"
    results_dir = data_dir / "results"
    for d in (uploads_dir, jobs_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.queue = asyncio.Queue()
        rehydrated = 0
        for path in sorted(jobs_dir.glob("*.json")):
            try:
                record = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                log.warning("job.rehydrate.skip_corrupt", path=str(path))
                continue
            if record.get("status") in {"queued", "running"}:
                job_id = record.get("job_id") or path.stem
                _write_job(jobs_dir, job_id, status="queued")
                app.state.queue.put_nowait(job_id)
                rehydrated += 1
                log.info(
                    "job.rehydrated",
                    job_id=job_id,
                    previous_status=record.get("status"),
                )
        log.info("app.startup", data_dir=str(data_dir), rehydrated_jobs=rehydrated)
        app.state.worker = asyncio.create_task(
            _run_worker(app.state.queue, jobs_dir, results_dir, ocr_fn)
        )
        try:
            yield
        finally:
            log.info("app.shutdown")
            app.state.worker.cancel()
            try:
                await app.state.worker
            except asyncio.CancelledError:
                pass

    app = FastAPI(lifespan=lifespan)

    @app.post("/jobs", status_code=202)
    async def create_job(file: UploadFile = File(...)) -> dict:
        if not (file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400, detail="only .pdf uploads are accepted"
            )
        job_id = uuid.uuid4().hex
        pdf_path = uploads_dir / f"{job_id}.pdf"
        contents = await file.read()
        pdf_path.write_bytes(contents)
        record = _write_job(jobs_dir, job_id, status="queued", pdf_path=str(pdf_path))
        await app.state.queue.put(job_id)
        log.info(
            "job.queued",
            job_id=job_id,
            pdf_bytes=len(contents),
            filename=file.filename,
            queue_depth=app.state.queue.qsize(),
        )
        return record

    @app.get("/jobs/{job_id}")
    async def get_job(job_id: str) -> dict:
        job = _read_job(jobs_dir, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        return job

    @app.get("/jobs/{job_id}/result", response_class=PlainTextResponse)
    async def get_result(job_id: str) -> str:
        job = _read_job(jobs_dir, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job")
        if job.get("status") != "completed":
            raise HTTPException(status_code=409, detail=f"job is {job.get('status')}")
        return Path(job["result_path"]).read_text(encoding="utf-8")

    return app
