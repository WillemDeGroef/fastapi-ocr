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
