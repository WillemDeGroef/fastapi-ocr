from __future__ import annotations

from pathlib import Path


def test_upload_pdf_returns_job_id_and_stores_file(client, sample_pdf: bytes, data_dir: Path):
    response = client.post(
        "/jobs",
        files={"file": ("invoice.pdf", sample_pdf, "application/pdf")},
    )
    assert response.status_code == 202, response.text
    body = response.json()
    job_id = body["job_id"]
    assert job_id and isinstance(job_id, str)
    assert body["status"] == "queued"

    stored = data_dir / "uploads" / f"{job_id}.pdf"
    assert stored.exists()
    assert stored.read_bytes() == sample_pdf


def test_upload_rejects_non_pdf(client):
    response = client.post(
        "/jobs",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400


def test_get_job_status_returns_queued_then_completed(client, sample_pdf: bytes):
    r = client.post("/jobs", files={"file": ("invoice.pdf", sample_pdf, "application/pdf")})
    job_id = r.json()["job_id"]

    # Immediately queryable
    status = client.get(f"/jobs/{job_id}")
    assert status.status_code == 200
    assert status.json()["job_id"] == job_id
    assert status.json()["status"] in {"queued", "running", "completed"}


def test_get_unknown_job_returns_404(client):
    r = client.get("/jobs/does-not-exist")
    assert r.status_code == 404


def _wait_for_status(client, job_id: str, target: str, timeout: float = 2.0) -> dict:
    import time as _t

    deadline = _t.monotonic() + timeout
    while _t.monotonic() < deadline:
        body = client.get(f"/jobs/{job_id}").json()
        if body["status"] == target:
            return body
        _t.sleep(0.02)
    raise AssertionError(f"job {job_id} never reached {target}; last={body}")


def test_worker_processes_job_and_writes_result_to_disk(client, sample_pdf: bytes, data_dir: Path):
    r = client.post("/jobs", files={"file": ("invoice.pdf", sample_pdf, "application/pdf")})
    job_id = r.json()["job_id"]

    body = _wait_for_status(client, job_id, "completed")
    assert body["status"] == "completed"

    result_file = data_dir / "results" / f"{job_id}.txt"
    assert result_file.exists()
    # fake_ocr returns "OCR[<pdf-name>]:<size>b"
    assert result_file.read_text().startswith(f"OCR[{job_id}.pdf]:")
    assert body["result_path"] == str(result_file)


def test_get_result_returns_text_after_completion(client, sample_pdf: bytes):
    r = client.post("/jobs", files={"file": ("invoice.pdf", sample_pdf, "application/pdf")})
    job_id = r.json()["job_id"]
    _wait_for_status(client, job_id, "completed")

    result = client.get(f"/jobs/{job_id}/result")
    assert result.status_code == 200
    assert result.headers["content-type"].startswith("text/plain")
    assert result.text.startswith(f"OCR[{job_id}.pdf]:")


def test_get_result_returns_404_for_unknown_job(client):
    r = client.get("/jobs/nope/result")
    assert r.status_code == 404


def test_restart_rehydrates_orphaned_jobs(data_dir: Path, sample_pdf: bytes, fake_ocr):
    """Jobs left in 'queued' or 'running' on disk must be re-enqueued on next startup."""
    import json

    from fastapi.testclient import TestClient

    from ocr_api import create_app

    jobs_dir = data_dir / "jobs"
    uploads_dir = data_dir / "uploads"
    for d in (jobs_dir, uploads_dir):
        d.mkdir(parents=True, exist_ok=True)

    orphan_id = "orphan-running-job"
    pdf_path = uploads_dir / f"{orphan_id}.pdf"
    pdf_path.write_bytes(sample_pdf)
    (jobs_dir / f"{orphan_id}.json").write_text(
        json.dumps(
            {"job_id": orphan_id, "status": "running", "pdf_path": str(pdf_path)}
        )
    )

    app = create_app(data_dir=data_dir, ocr_fn=fake_ocr)
    with TestClient(app) as c:
        body = _wait_for_status(c, orphan_id, "completed")
        assert body["status"] == "completed"
        assert Path(body["result_path"]).exists()


def test_upload_exceeding_max_bytes_is_rejected_and_leaves_no_partial(
    data_dir: Path, fake_ocr
):
    """Oversized uploads must be refused before they fill memory or disk."""
    from fastapi.testclient import TestClient

    from ocr_api import create_app

    app = create_app(data_dir=data_dir, ocr_fn=fake_ocr, max_upload_bytes=1024)
    with TestClient(app) as c:
        oversized = b"%PDF-1.4\n" + b"0" * 4096
        r = c.post(
            "/jobs",
            files={"file": ("big.pdf", oversized, "application/pdf")},
        )
        assert r.status_code == 413, r.text
        # No partial upload and no job record should survive a rejected upload.
        assert list((data_dir / "uploads").glob("*.pdf")) == []
        assert list((data_dir / "jobs").glob("*.json")) == []


def test_auth_required_when_token_configured(data_dir: Path, sample_pdf: bytes, fake_ocr):
    """With an api_token set, /jobs must reject unauthenticated and wrong-token callers."""
    from fastapi.testclient import TestClient

    from ocr_api import create_app

    app = create_app(data_dir=data_dir, ocr_fn=fake_ocr, api_token="s3cret")
    with TestClient(app) as c:
        files = {"file": ("invoice.pdf", sample_pdf, "application/pdf")}

        # No credentials.
        assert c.post("/jobs", files=files).status_code == 401
        # Wrong token.
        r = c.post(
            "/jobs", files=files, headers={"Authorization": "Bearer nope"}
        )
        assert r.status_code == 401
        # Correct token.
        r = c.post(
            "/jobs", files=files, headers={"Authorization": "Bearer s3cret"}
        )
        assert r.status_code == 202, r.text


def test_upload_rejected_when_queue_full_and_leaves_no_partial(
    data_dir: Path, sample_pdf: bytes
):
    """A full queue must shed load with 503, not grow memory/disk without bound."""
    import threading

    from fastapi.testclient import TestClient

    from ocr_api import create_app

    gate = threading.Event()

    def blocking_ocr(pdf_path: Path) -> str:
        gate.wait(timeout=5.0)
        return "done"

    # maxsize=1: first job is pulled by the worker and blocks, second fills the
    # queue, third must be rejected.
    app = create_app(
        data_dir=data_dir, ocr_fn=blocking_ocr, max_queue_size=1
    )
    with TestClient(app) as c:
        accepted = 0
        rejected = None
        for _ in range(5):
            r = c.post(
                "/jobs",
                files={"file": ("invoice.pdf", sample_pdf, "application/pdf")},
            )
            if r.status_code == 202:
                accepted += 1
            else:
                rejected = r
                break
        assert rejected is not None, "expected the queue to reject an upload"
        assert rejected.status_code == 503, rejected.text
        # The rejected upload must not leave an orphan file or job record behind.
        uploads = list((data_dir / "uploads").glob("*.pdf"))
        jobs = list((data_dir / "jobs").glob("*.json"))
        assert len(uploads) == accepted
        assert len(jobs) == accepted
        gate.set()


def test_concurrent_rehydration_processes_orphan_exactly_once(
    data_dir: Path, sample_pdf: bytes
):
    """Two processes sharing a data dir must not both re-run the same orphan job."""
    import json
    import threading

    from fastapi.testclient import TestClient

    from ocr_api import create_app

    jobs_dir = data_dir / "jobs"
    uploads_dir = data_dir / "uploads"
    for d in (jobs_dir, uploads_dir):
        d.mkdir(parents=True, exist_ok=True)

    orphan_id = "orphan-shared"
    pdf_path = uploads_dir / f"{orphan_id}.pdf"
    pdf_path.write_bytes(sample_pdf)
    (jobs_dir / f"{orphan_id}.json").write_text(
        json.dumps(
            {"job_id": orphan_id, "status": "running", "pdf_path": str(pdf_path)}
        )
    )

    calls: list[str] = []
    calls_lock = threading.Lock()
    gate = threading.Event()

    def gated_counting_ocr(p: Path) -> str:
        with calls_lock:
            calls.append(p.name)
        gate.wait(timeout=5.0)
        return f"OCR[{p.name}]"

    app_a = create_app(data_dir=data_dir, ocr_fn=gated_counting_ocr)
    app_b = create_app(data_dir=data_dir, ocr_fn=gated_counting_ocr)
    # app_a claims the orphan during its startup and holds the claim while it
    # processes; app_b's startup must find it already claimed and skip it.
    with TestClient(app_a) as ca, TestClient(app_b):
        gate.set()
        body = _wait_for_status(ca, orphan_id, "completed")
        assert body["status"] == "completed"

    assert calls == [f"{orphan_id}.pdf"], f"orphan processed more than once: {calls}"


def test_get_result_returns_409_if_not_completed(data_dir: Path, sample_pdf: bytes):
    """When the worker is blocked, /result must not silently 404 or 200 with empty text."""
    import threading

    gate = threading.Event()

    def slow_ocr(pdf_path: Path) -> str:
        gate.wait(timeout=5.0)
        return "done"

    from fastapi.testclient import TestClient

    from ocr_api import create_app

    app = create_app(data_dir=data_dir, ocr_fn=slow_ocr)
    with TestClient(app) as c:
        r = c.post("/jobs", files={"file": ("invoice.pdf", sample_pdf, "application/pdf")})
        job_id = r.json()["job_id"]
        result = c.get(f"/jobs/{job_id}/result")
        assert result.status_code == 409
        gate.set()
