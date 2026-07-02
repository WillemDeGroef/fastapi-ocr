"""CLI to test the OCR conversion service end-to-end for a single PDF.

Three modes let you localize where a bug is in the pipeline:

  http (default): Upload PDF to a running server, poll job status, print text.
                  Exercises FastAPI + queue + worker + OCR.
  --direct:       Bypass HTTP and call the in-process ocr_fn on the PDF.
                  Skips FastAPI + queue; useful to prove the server layer is fine.
  --render-only:  Only render the PDF to PNGs (no OCR). Useful when text is
                  garbled — inspect the intermediate images.

Examples:
    uv run python convert_cli.py invoice.pdf
    uv run python convert_cli.py invoice.pdf --url http://localhost:8000 -o out.txt
    uv run python convert_cli.py invoice.pdf --direct
    uv run python convert_cli.py invoice.pdf --render-only --pages-dir /tmp/pages
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


def _multipart_body(pdf_path: Path) -> tuple[bytes, str]:
    """Build a minimal multipart/form-data body for a single 'file' field."""
    boundary = f"----ocrcli{uuid.uuid4().hex}"
    mime = mimetypes.guess_type(pdf_path.name)[0] or "application/pdf"
    head = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{pdf_path.name}"\r\n'
        f"Content-Type: {mime}\r\n\r\n"
    ).encode()
    tail = f"\r\n--{boundary}--\r\n".encode()
    body = head + pdf_path.read_bytes() + tail
    return body, f"multipart/form-data; boundary={boundary}"


def _http_json(method: str, url: str, body: bytes | None = None, content_type: str | None = None) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, method=method)
    if content_type:
        req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def run_http(pdf_path: Path, url: str, poll_interval: float, timeout: float, verbose: bool) -> str:
    body, ctype = _multipart_body(pdf_path)
    if verbose:
        print(f"[http] POST {url}/jobs  file={pdf_path.name} bytes={len(body)}", file=sys.stderr)
    status, resp = _http_json("POST", f"{url}/jobs", body, ctype)
    if status != 202:
        raise SystemExit(f"upload failed: HTTP {status}: {resp.decode(errors='replace')}")
    job = json.loads(resp)
    job_id = job["job_id"]
    print(f"[http] job_id={job_id} queued", file=sys.stderr)

    deadline = time.monotonic() + timeout
    last_status: str | None = None
    while time.monotonic() < deadline:
        status, resp = _http_json("GET", f"{url}/jobs/{job_id}")
        if status != 200:
            raise SystemExit(f"status query failed: HTTP {status}: {resp.decode(errors='replace')}")
        body_json = json.loads(resp)
        cur = body_json.get("status")
        if cur != last_status:
            print(f"[http] status={cur}", file=sys.stderr)
            last_status = cur
        if cur == "completed":
            status, resp = _http_json("GET", f"{url}/jobs/{job_id}/result")
            if status != 200:
                raise SystemExit(f"result fetch failed: HTTP {status}: {resp.decode(errors='replace')}")
            return resp.decode("utf-8")
        if cur == "failed":
            raise SystemExit(f"job failed: {body_json.get('error') or json.dumps(body_json)}")
        time.sleep(poll_interval)
    raise SystemExit(f"timed out after {timeout}s waiting for job {job_id}")


def run_direct(pdf_path: Path, verbose: bool) -> str:
    if verbose:
        print("[direct] loading OCR model (this can take a while on first run)…", file=sys.stderr)
    from logging_config import configure_logging
    from ocr_service import create_ocr_fn

    configure_logging()
    started = time.monotonic()
    ocr_fn = create_ocr_fn()
    if verbose:
        print(f"[direct] model ready in {time.monotonic() - started:.2f}s", file=sys.stderr)
    infer_started = time.monotonic()
    text = ocr_fn(pdf_path)
    if verbose:
        print(f"[direct] inference done in {time.monotonic() - infer_started:.2f}s, {len(text)} chars", file=sys.stderr)
    return text


def run_render_only(pdf_path: Path, pages_dir: Path, dpi: int, verbose: bool) -> str:
    from logging_config import configure_logging
    from ocr_service import render_pdf_to_pngs

    configure_logging()
    pages_dir.mkdir(parents=True, exist_ok=True)
    pages = render_pdf_to_pngs(pdf_path, pages_dir, dpi=dpi)
    lines = [f"rendered {len(pages)} page(s) to {pages_dir} at dpi={dpi}:"]
    for i, p in enumerate(pages, start=1):
        lines.append(f"  page {i}: {p}  ({p.stat().st_size} bytes)")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("pdf", type=Path, help="Path to the PDF to convert")
    p.add_argument("--url", default="http://localhost:8000", help="Server URL (default: %(default)s)")
    p.add_argument("--poll-interval", type=float, default=0.5, help="Poll interval in seconds (default: %(default)s)")
    p.add_argument("--timeout", type=float, default=120.0, help="Total timeout in seconds (default: %(default)s)")
    p.add_argument("--direct", action="store_true", help="Call ocr_fn in-process instead of via HTTP")
    p.add_argument("--render-only", action="store_true", help="Only render PDF pages to PNGs, skip OCR")
    p.add_argument("--pages-dir", type=Path, default=Path("./pages_debug"), help="Where to write PNGs in --render-only mode")
    p.add_argument("--dpi", type=int, default=300, help="Render DPI for --render-only mode")
    p.add_argument("-o", "--output", type=Path, help="Write result to this file instead of stdout")
    p.add_argument("-v", "--verbose", action="store_true", help="Verbose progress on stderr")
    args = p.parse_args(argv)

    if not args.pdf.exists():
        print(f"error: PDF not found: {args.pdf}", file=sys.stderr)
        return 2
    if args.direct and args.render_only:
        print("error: --direct and --render-only are mutually exclusive", file=sys.stderr)
        return 2

    if args.render_only:
        text = run_render_only(args.pdf, args.pages_dir, args.dpi, args.verbose)
    elif args.direct:
        text = run_direct(args.pdf, args.verbose)
    else:
        text = run_http(args.pdf, args.url.rstrip("/"), args.poll_interval, args.timeout, args.verbose)

    if args.output:
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote {len(text)} chars to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
