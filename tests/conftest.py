from __future__ import annotations

import io
from pathlib import Path

import pymupdf
import pytest
from fastapi.testclient import TestClient

from ocr_api import create_app


@pytest.fixture
def sample_pdf() -> bytes:
    """A minimal valid one-page PDF with a bit of text."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello OCR")
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def fake_ocr():
    """Fast fake OCR: reads first bytes of PDF and returns a deterministic string."""

    def _fake(pdf_path: Path) -> str:
        return f"OCR[{pdf_path.name}]:{pdf_path.stat().st_size}b"

    return _fake


@pytest.fixture
def client(data_dir: Path, fake_ocr):
    app = create_app(data_dir=data_dir, ocr_fn=fake_ocr)
    with TestClient(app) as c:
        yield c
