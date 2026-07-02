"""FastAPI service exposing PDF → OCR jobs via an asyncio.Queue worker."""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

import structlog
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
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


def _acquire_claim(locks_dir: Path, job_id: str) -> int | None:
    """Atomically claim a job across processes with an exclusive advisory lock.

    Returns an open file descriptor on success (the caller must hold it for the
    job's lifetime and release it via ``_release_claim``), or ``None`` when
    another live process already owns the job. The OS drops the lock if the
    owning process dies, so a job left ``running`` by a crash is reclaimable.
    """
    fd = os.open(locks_dir / f"{job_id}.lock", os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd


def _release_claim(locks_dir: Path, job_id: str, fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
    (locks_dir / f"{job_id}.lock").unlink(missing_ok=True)


async def _run_worker(
    queue: asyncio.Queue,
    jobs_dir: Path,
    results_dir: Path,
    ocr_fn: OcrFn,
    claims: dict[str, int],
    locks_dir: Path,
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
            fd = claims.pop(job_id, None)
            if fd is not None:
                _release_claim(locks_dir, job_id, fd)
            structlog.contextvars.unbind_contextvars("job_id")
            queue.task_done()


_UPLOAD_CHUNK = 1 << 20  # 1 MiB


def create_app(
    *,
    data_dir: Path,
    ocr_fn: OcrFn,
    max_upload_bytes: int = 25 * 1024 * 1024,
    max_queue_size: int = 100,
    api_token: str | None = None,
) -> FastAPI:
    uploads_dir = data_dir / "uploads"
    jobs_dir = data_dir / "jobs"
    results_dir = data_dir / "results"
    locks_dir = data_dir / "locks"
    for d in (uploads_dir, jobs_dir, results_dir, locks_dir):
        d.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.queue = asyncio.Queue(maxsize=max_queue_size)
        app.state.claims = {}
        rehydrated = 0
        for path in sorted(jobs_dir.glob("*.json")):
            try:
                record = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                log.warning("job.rehydrate.skip_corrupt", path=str(path))
                continue
            if record.get("status") in {"queued", "running"}:
                job_id = record.get("job_id") or path.stem
                fd = _acquire_claim(locks_dir, job_id)
                if fd is None:
                    log.info("job.rehydrate.already_claimed", job_id=job_id)
                    continue
                try:
                    app.state.queue.put_nowait(job_id)
                except asyncio.QueueFull:
                    _release_claim(locks_dir, job_id, fd)
                    log.warning("job.rehydrate.queue_full", job_id=job_id)
                    break
                _write_job(jobs_dir, job_id, status="queued")
                app.state.claims[job_id] = fd
                rehydrated += 1
                log.info(
                    "job.rehydrated",
                    job_id=job_id,
                    previous_status=record.get("status"),
                )
        log.info("app.startup", data_dir=str(data_dir), rehydrated_jobs=rehydrated)
        app.state.worker = asyncio.create_task(
            _run_worker(
                app.state.queue,
                jobs_dir,
                results_dir,
                ocr_fn,
                app.state.claims,
                locks_dir,
            )
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

    async def require_auth(authorization: str | None = Header(default=None)) -> None:
        """Enforce a bearer token when one is configured; a no-op otherwise."""
        if api_token is None:
            return
        expected = f"Bearer {api_token}"
        if authorization is None or not secrets.compare_digest(
            authorization, expected
        ):
            raise HTTPException(
                status_code=401,
                detail="invalid or missing bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    app = FastAPI(lifespan=lifespan, dependencies=[Depends(require_auth)])

    @app.post("/jobs", status_code=202)
    async def create_job(file: UploadFile = File(...)) -> dict:
        if not (file.filename or "").lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400, detail="only .pdf uploads are accepted"
            )
        job_id = uuid.uuid4().hex
        pdf_path = uploads_dir / f"{job_id}.pdf"
        # Stream to disk in bounded chunks so a hostile upload can never buffer
        # more than one chunk in memory, and abort the moment it exceeds the cap.
        written = 0
        try:
            with pdf_path.open("wb") as out:
                while chunk := await file.read(_UPLOAD_CHUNK):
                    written += len(chunk)
                    if written > max_upload_bytes:
                        raise HTTPException(
                            status_code=413,
                            detail=f"upload exceeds {max_upload_bytes} bytes",
                        )
                    out.write(chunk)
        except HTTPException:
            pdf_path.unlink(missing_ok=True)
            log.warning(
                "job.rejected.too_large",
                filename=file.filename,
                max_upload_bytes=max_upload_bytes,
            )
            raise
        record = _write_job(jobs_dir, job_id, status="queued", pdf_path=str(pdf_path))
        # Claim the job before enqueueing so the worker holds exclusive ownership
        # for its whole lifetime (a fresh id is never contended, but this keeps a
        # single claim/release path shared with rehydration).
        fd = _acquire_claim(locks_dir, job_id)
        try:
            if fd is None:
                raise HTTPException(status_code=503, detail="server busy, retry later")
            app.state.queue.put_nowait(job_id)
        except (asyncio.QueueFull, HTTPException) as exc:
            if fd is not None:
                _release_claim(locks_dir, job_id, fd)
            pdf_path.unlink(missing_ok=True)
            (jobs_dir / f"{job_id}.json").unlink(missing_ok=True)
            log.warning(
                "job.rejected.queue_full",
                filename=file.filename,
                max_queue_size=max_queue_size,
            )
            if isinstance(exc, HTTPException):
                raise
            raise HTTPException(
                status_code=503, detail="server busy, retry later"
            ) from None
        app.state.claims[job_id] = fd
        log.info(
            "job.queued",
            job_id=job_id,
            pdf_bytes=written,
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
