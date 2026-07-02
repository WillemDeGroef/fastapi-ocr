"""FastAPI entrypoint: PDF upload → asyncio-queued PaddleOCR → text on disk.

Run with:
    uv run uvicorn server:app --host 0.0.0.0 --port 8000

Uploads land in ./data/uploads/<job_id>.pdf, results in ./data/results/<job_id>.txt,
job metadata in ./data/jobs/<job_id>.json.
"""

from __future__ import annotations

from pathlib import Path

from logging_config import configure_logging

configure_logging()

from ocr_api import create_app  # noqa: E402
from ocr_service import create_ocr_fn  # noqa: E402

DATA_DIR = Path("./data")

app = create_app(data_dir=DATA_DIR, ocr_fn=create_ocr_fn())
