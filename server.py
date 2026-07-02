"""FastAPI entrypoint: PDF upload → asyncio-queued PaddleOCR → text on disk.

Run with:
    uv run uvicorn server:app --host 0.0.0.0 --port 8000

Uploads land in ./data/uploads/<job_id>.pdf, results in ./data/results/<job_id>.txt,
job metadata in ./data/jobs/<job_id>.json.
"""

from __future__ import annotations

import os
from pathlib import Path

from logging_config import configure_logging

configure_logging()

from ocr_api import create_app  # noqa: E402
from ocr_service import create_ocr_fn  # noqa: E402

DATA_DIR = Path(os.environ.get("DATA_DIR", "./data"))

# Admission controls (see README "Deployment & limits"). Overridable via env so
# the documented 0.0.0.0 deployment can enforce a trust boundary and shed load.
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 25 * 1024 * 1024))
MAX_QUEUE_SIZE = int(os.environ.get("MAX_QUEUE_SIZE", 100))
API_TOKEN = os.environ.get("API_TOKEN") or None

app = create_app(
    data_dir=DATA_DIR,
    ocr_fn=create_ocr_fn(),
    max_upload_bytes=MAX_UPLOAD_BYTES,
    max_queue_size=MAX_QUEUE_SIZE,
    api_token=API_TOKEN,
)
