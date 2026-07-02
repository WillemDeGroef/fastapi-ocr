"""PaddleOCR wrapper using the benchmark-winning configuration.

Winning config (see bench_summary.csv):
  dpi=300, fmt=png, rec_batch=32  →  token-F1 0.9187 on the digital-invoice set.

Benchmark notes (full write-up in ../README.md):
  * DPI is the only knob that moves accuracy, and only slightly: 150→300 spans
    ~1 F1 point (0.9107→0.9187). dpi=300 is the accuracy max but costs +17%
    OCR time; dpi=150 is a near-tie for latency-sensitive use.
  * Image format is irrelevant here — page images go to a temp dir and are
    deleted after inference; rendering is <0.1% of total time. PNG is kept
    because it is lossless and costs nothing in this pipeline.

Heavy imports (paddleocr, torch, pymupdf) happen lazily so `ocr_api` and tests
can be imported without paying the PaddleOCR model-load cost.

Runtime tuning is picked via the ``OCR_PROFILE`` env var:

  * ``apple`` (default): the accuracy-max setup — PP-OCRv6 medium models,
    dpi=300, rec_batch=32, 8 threads. Fits Apple Silicon (M-series) where the
    unified-memory bandwidth and modern SIMD width keep the medium models fast
    enough to absorb the dpi=300 cost. Drop to dpi=150 if latency matters more
    than the last F1 point.
  * ``legacy-cpu``: tuned for old x86_64 without AVX2 (e.g. Xeon E7 v2 /
    Ivy Bridge in a VM) — swaps to PP-OCRv5 mobile models, drops DPI to 220,
    shrinks rec_batch to 8, and defaults to 4 threads so the working set stays
    in-cache. Pair it with running multiple uvicorn workers for throughput.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Callable

import structlog

PROFILES: dict[str, dict] = {
    "apple": {
        "det_model": "PP-OCRv6_medium_det",
        "rec_model": "PP-OCRv6_medium_rec",
        "dpi": 300,
        "rec_batch": 32,
        "threads": 8,
    },
    "legacy-cpu": {
        "det_model": "PP-OCRv5_mobile_det",
        "rec_model": "PP-OCRv5_mobile_rec",
        "dpi": 220,
        "rec_batch": 8,
        "threads": 4,
    },
}


def _active_profile() -> dict:
    name = os.environ.get("OCR_PROFILE", "apple").lower()
    if name not in PROFILES:
        raise ValueError(
            f"unknown OCR_PROFILE={name!r}; choose from {sorted(PROFILES)}"
        )
    return {"name": name, **PROFILES[name]}


_PROFILE = _active_profile()

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("OMP_NUM_THREADS", str(_PROFILE["threads"]))
os.environ.setdefault("MKL_NUM_THREADS", str(_PROFILE["threads"]))

log = structlog.get_logger("ocr_service")


def json_to_text(res_json: dict) -> str:
    """Convert one PaddleOCR page result into layout-preserving plaintext."""
    polys = res_json.get("rec_polys") or res_json.get("dt_polys") or []
    texts = res_json.get("rec_texts") or []
    if not texts:
        return ""

    items = []
    char_widths = []
    for poly, text in zip(polys, texts):
        ys = [pt[1] for pt in poly]
        xs = [pt[0] for pt in poly]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        items.append({"text": text, "x0": x0, "x1": x1, "y": y0, "h": y1 - y0})
        if text:
            char_widths.append((x1 - x0) / len(text))

    char_widths.sort()
    char_w = char_widths[len(char_widths) // 2] if char_widths else 10.0
    if char_w <= 0:
        char_w = 10.0

    items.sort(key=lambda it: (it["y"], it["x0"]))

    lines: list[list[dict]] = []
    for it in items:
        if lines and abs(it["y"] - lines[-1][0]["y"]) <= max(it["h"] * 0.6, 6):
            lines[-1].append(it)
        else:
            lines.append([it])

    min_x = min(it["x0"] for it in items)
    heights = sorted(it["h"] for it in items if it["h"] > 0)
    line_h = heights[len(heights) // 2] if heights else 14.0
    row_unit = line_h * 1.6
    line_ys = [min(it["y"] for it in line) for line in lines]

    out_lines: list[str] = []
    prev_y = None
    for line, y in zip(lines, line_ys):
        if prev_y is not None:
            gap = max(0, int(round((y - prev_y) / row_unit)) - 1)
            out_lines.extend([""] * gap)
        line.sort(key=lambda it: it["x0"])
        buf = ""
        for it in line:
            col = int(round((it["x0"] - min_x) / char_w))
            if col > len(buf):
                buf += " " * (col - len(buf))
            elif buf and not buf.endswith(" "):
                buf += " "
            buf += it["text"]
        out_lines.append(buf.rstrip())
        prev_y = y
    return "\n".join(out_lines)


def render_pdf_to_pngs(pdf_path: Path, out_dir: Path, *, dpi: int = 300) -> list[Path]:
    """Render each page of `pdf_path` as a PNG under `out_dir`. Returns page paths."""
    import pymupdf

    out_dir.mkdir(parents=True, exist_ok=True)
    zoom = dpi / 72.0
    pages: list[Path] = []
    started = time.monotonic()
    with pymupdf.open(pdf_path) as doc:
        n = len(doc)
        log.info("render.start", pages=n, dpi=dpi)
        matrix = pymupdf.Matrix(zoom, zoom)
        for i, page in enumerate(doc, start=1):
            page_started = time.monotonic()
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            out = out_dir / f"page_{i:04d}.png"
            pix.save(out)
            pages.append(out)
            log.debug(
                "render.page",
                page=i,
                elapsed_s=round(time.monotonic() - page_started, 3),
            )
    log.info(
        "render.done",
        pages=len(pages),
        elapsed_s=round(time.monotonic() - started, 3),
    )
    return pages


def create_ocr_fn(
    *, dpi: int | None = None, rec_batch: int | None = None
) -> Callable[[Path], str]:
    """Build a warm PaddleOCR callable that turns a PDF path into concatenated text.

    Model loading happens once, on first call to this factory. Unset ``dpi`` and
    ``rec_batch`` fall back to the active ``OCR_PROFILE`` defaults.
    """
    import torch
    from paddleocr import PaddleOCR

    if dpi is None:
        dpi = _PROFILE["dpi"]
    if rec_batch is None:
        rec_batch = _PROFILE["rec_batch"]

    num_threads = int(os.environ.get("TORCH_NUM_THREADS", str(_PROFILE["threads"])))
    torch.set_num_threads(num_threads)
    device = os.environ.get("PADDLE_DEVICE", "cpu")
    engine = os.environ.get("OCR_ENGINE", "transformers")
    det_model = _PROFILE["det_model"]
    rec_model = _PROFILE["rec_model"]

    log.info(
        "model.load.start",
        profile=_PROFILE["name"],
        det_model=det_model,
        rec_model=rec_model,
        engine=engine,
        device=device,
        torch_threads=num_threads,
        dpi=dpi,
        rec_batch=rec_batch,
    )
    started = time.monotonic()
    model = PaddleOCR(
        text_detection_model_name=det_model,
        text_recognition_model_name=rec_model,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_recognition_batch_size=rec_batch,
        engine=engine,
        device=device,
    )
    log.info("model.load.done", elapsed_s=round(time.monotonic() - started, 3))

    def ocr_pdf(pdf_path: Path) -> str:
        with tempfile.TemporaryDirectory(prefix="ocr_") as tmp:
            pages = render_pdf_to_pngs(pdf_path, Path(tmp), dpi=dpi)
            if not pages:
                log.warning("ocr.empty_pdf", pdf_path=str(pdf_path))
                return ""
            page_texts: dict[str, str] = {}
            infer_started = time.monotonic()
            log.info("ocr.infer.start", pages=len(pages))
            done = 0
            page_started = time.monotonic()
            for res in model.predict_iter(input=[str(p) for p in pages]):
                rj = json.loads(res.json) if isinstance(res.json, str) else res.json
                if isinstance(rj, dict) and "res" in rj:
                    rj = rj["res"]
                input_path = rj.get("input_path", "")
                page_texts[input_path] = json_to_text(rj)
                done += 1
                log.info(
                    "ocr.infer.page",
                    page=done,
                    total=len(pages),
                    input_path=input_path,
                    elapsed_s=round(time.monotonic() - page_started, 3),
                    chars=len(page_texts[input_path]),
                )
                page_started = time.monotonic()
            log.info(
                "ocr.infer.done",
                pages=len(pages),
                elapsed_s=round(time.monotonic() - infer_started, 3),
            )

            chunks: list[str] = []
            for i, page in enumerate(pages, start=1):
                text = page_texts.get(str(page), "")
                chunks.append(f"--- page {i} ({page.name}) ---\n{text}".rstrip())
            return "\n\n".join(chunks) + "\n"

    return ocr_pdf
