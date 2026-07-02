# ocr-service

FastAPI service that turns a PDF upload into OCR'd plaintext using PaddleOCR's
[**PP-OCRv6**](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det_safetensors)
models behind an asyncio queue.

```
POST /jobs           multipart PDF   → { job_id, status: "queued", ... }
GET  /jobs/{id}      → { status: "queued" | "running" | "completed" | "failed", ... }
GET  /jobs/{id}/result   plain text  (409 while not completed)
```

State lives on disk under `./data/{uploads,jobs,results}`. Jobs left in
`queued` / `running` when the process dies get rehydrated on next startup.

Run:

```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8000
```

---

## OCR model: PP-OCRv6

The service ships with **PP-OCRv6** as its default OCR engine (the `apple`
profile — see [Runtime profiles](#runtime-profiles-ocr_profile) below). PP-OCRv6
is the PaddleOCR team's 2025 model family, tagged *"From 1.5M to 34.5M
Parameters, Surpassing Billion-Scale VLMs on OCR Tasks."*

Why it's the default here:

- **Accuracy.** `PP-OCRv6_medium` reaches **86.2 % detection Hmean** and
  **83.2 % recognition accuracy**, beating the previous-generation
  `PP-OCRv5_server` by **+4.6 %** / **+5.1 %** respectively — while staying
  small enough to run on CPU. On the benchmark it edges out billion-parameter
  VLMs (Qwen3-VL-235B, GPT-5.5, Gemini-3.1-Pro) on OCR tasks with orders of
  magnitude fewer parameters.
- **A scalable family.** Three tiers span 1.5M → 34.5M parameters. This repo
  uses the **medium** detection (15.5M params) and recognition models, which
  hit the accuracy/latency sweet spot for invoice-style documents.
- **Modern lightweight architecture.** LCNetV4 backbone (MetaFormer-style with
  structural reparameterization), a RepLKFPN detection neck, and an
  EncoderWithLightSVTR recognition neck with local-global attention.
- **Broad coverage.** 48 languages plus tricky scenes — handwritten, printed,
  rotated, curved, blurred, artistic, and industrial text (digital displays,
  dot-matrix, tire prints, …).

The models are pulled from Hugging Face in **safetensors** format and run
through the `transformers` engine (`OCR_ENGINE=transformers`, the default), so
no `paddlepaddle` install is required. Model selection lives in one place —
the `PROFILES` table in [`ocr_service.py`](ocr_service.py) — and is wired into
every entry point (server, worker, and `convert_cli.py`) via `create_ocr_fn()`:

```python
PaddleOCR(
    text_detection_model_name="PP-OCRv6_medium_det",
    text_recognition_model_name="PP-OCRv6_medium_rec",
    engine="transformers",
    ...
)
```

Older x86_64 CPUs without AVX2 can fall back to the lighter PP-OCRv5 mobile
models via `OCR_PROFILE=legacy-cpu` — see
[Runtime profiles](#runtime-profiles-ocr_profile). Requires `paddleocr>=3.7.0`
and a `transformers` + `torch` environment (both already pinned in
`pyproject.toml`). Model card:
<https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det_safetensors>.

---

## CLI: `convert_cli.py`

A one-shot tool for OCR'ing a single PDF and printing the text — handy for
smoke tests, debugging, and one-off conversions without touching the HTTP
API. Three modes let you localize where in the pipeline a problem lives.

```bash
uv run python convert_cli.py <pdf> [flags]
```

### Modes

| Mode              | Flag             | What it exercises                                | When to use                                                     |
| ----------------- | ---------------- | ------------------------------------------------ | --------------------------------------------------------------- |
| HTTP (default)    | *(none)*         | FastAPI + queue + worker + OCR                   | End-to-end sanity check against a running server                |
| Direct            | `--direct`       | In-process `create_ocr_fn` → `ocr_fn(pdf)`       | Prove the OCR core works when the server layer is suspect       |
| Render-only       | `--render-only`  | Only PDF → PNG via PyMuPDF (no OCR)              | Text comes out garbled — inspect the intermediate images        |

`--direct` and `--render-only` are mutually exclusive.

### Common flags

| Flag                  | Default                 | Purpose                                                     |
| --------------------- | ----------------------- | ----------------------------------------------------------- |
| `--url`               | `http://localhost:8000` | Server URL for HTTP mode                                    |
| `--poll-interval`     | `0.5`                   | Seconds between job-status polls (HTTP mode)                |
| `--timeout`           | `120.0`                 | Give up after N seconds waiting for a job                   |
| `--pages-dir`         | `./pages_debug`         | Output dir for PNGs in `--render-only` mode                 |
| `--dpi`               | `300`                   | Render DPI in `--render-only` mode                          |
| `-o`, `--output PATH` | *(stdout)*              | Write result to a file instead of stdout                    |
| `-v`, `--verbose`     | off                     | Progress on stderr (timings, status transitions)            |

### Examples

```bash
# 1. Default: upload to a running server on localhost:8000 and print text.
uv run python convert_cli.py invoice.pdf

# 2. Point at a remote server and write to a file.
uv run python convert_cli.py invoice.pdf \
  --url http://ocr.internal:8000 -o invoice.txt -v

# 3. Skip the server — call the OCR model directly in-process.
#    Handy on the Xeon box to time the raw model without HTTP overhead.
uv run python convert_cli.py invoice.pdf --direct -v

# 4. Only render pages to PNG — no OCR. Inspect ./pages_debug/*.png.
uv run python convert_cli.py invoice.pdf --render-only --dpi 220

# 5. Combine with a profile — direct mode picks up OCR_PROFILE just like
#    the server does, so you can benchmark both profiles from the CLI.
OCR_PROFILE=legacy-cpu uv run python convert_cli.py invoice.pdf --direct -v
```

### Debugging playbook

1. Result is empty or bad → `--render-only` first. If the PNGs look wrong
   (blurry, cropped, wrong DPI), the problem is in `render_pdf_to_pngs`,
   not the OCR model.
2. PNGs look fine but text is bad → `--direct`. Isolates the OCR core.
   If it fails here, the issue is model / profile / thread config.
3. Direct works but HTTP mode doesn't → the server, queue, or worker is at
   fault. Check `./data/jobs/<job_id>.json` for the recorded status/error.

Exit codes: `0` on success, `2` on bad arguments, non-zero `SystemExit` with
a message on upload / status / result-fetch failure or job timeout.

---

## Runtime profiles (`OCR_PROFILE`)

Model choice + inference tuning is selected via the `OCR_PROFILE` env var so
Apple Silicon can keep its max-throughput config while old x86_64 hosts can
opt into a leaner setup.

| Profile        | det / rec models                              | DPI | rec_batch | threads |
| -------------- | --------------------------------------------- | --- | --------- | ------- |
| `apple` (default) | `PP-OCRv6_medium_det` / `PP-OCRv6_medium_rec` | 300 | 32        | 8       |
| `legacy-cpu`   | `PP-OCRv5_mobile_det`  / `PP-OCRv5_mobile_rec`  | 220 | 8         | 4       |

```bash
# Apple Silicon (M-series) — no action needed, defaults are correct.
uv run uvicorn server:app

# Old x86_64 without AVX2 (Xeon E7 v2 / Ivy Bridge, VMware, etc.)
export OCR_PROFILE=legacy-cpu
uv run uvicorn server:app --workers 8
```

Escape hatches (respected by both profiles):

| Env var             | Purpose                                                                 |
| ------------------- | ----------------------------------------------------------------------- |
| `OCR_PROFILE`       | `apple` (default) or `legacy-cpu`                                       |
| `OCR_ENGINE`        | `transformers` (default) or `paddlex` (needs `paddlepaddle` installed)  |
| `PADDLE_DEVICE`     | `cpu` (default), `gpu`, `mps`                                           |
| `TORCH_NUM_THREADS` | overrides the profile's thread count for the torch runtime              |
| `OMP_NUM_THREADS`   | OpenMP thread count (set from profile if unset)                         |
| `MKL_NUM_THREADS`   | Intel MKL thread count (set from profile if unset)                      |

---

## Why the `legacy-cpu` profile exists

Sample target host:

```
Intel(R) Xeon(R) CPU E7-8880 v2 @ 2.50GHz — 100 vCPU, VMware VM
Flags: sse4_2, avx, aes, f16c (no avx2, no avx-512, no vnni, no amx)
L1d 3.1 MiB · L2 25 MiB · L3 40 MiB
GPU: VMware SVGA II (paravirtual 2D framebuffer — not compute-capable)
```

**The CPU.** Ivy Bridge-EX (2013) has AVX but not AVX2. Modern OCR / DL
runtimes (PaddlePaddle, ONNX Runtime, OpenVINO, PyTorch) ship AVX2 and
AVX-512 kernels; on this box they fall back to AVX or SSE paths that run
2–4× slower per core. Cores are also slow individually (2.5 GHz, ~2013 IPC).
Upside: 100 vCPU + 40 MB shared L3 give lots of headroom for horizontal
parallelism.

**The "GPU".** VMware SVGA II is a paravirtual 2D framebuffer — no CUDA, no
OpenCL, no shader compute. It cannot run any ML model. GPU acceleration on
this VM would require passing through a real NVIDIA card (CUDA) or an Intel
Arc / iGPU (OpenVINO); neither is present.

### What the `legacy-cpu` profile changes, and why

1. **PP-OCRv6 medium → PP-OCRv5 mobile** — mobile det+rec models have ~5–10×
   fewer params. Biggest single win on a CPU that can't use the fast SIMD
   paths modern kernels were tuned for. Accuracy trade-off on invoices is
   small (~1–2 % token-F1).
2. **DPI 300 → 220** — ~46 % fewer pixels means ~half the render + inference
   time. Invoice text survives this cleanly.
3. **rec_batch 32 → 8** — keeps the recognition working set inside the 40 MB
   L3. Larger batches thrash DDR3 bandwidth, which is the actual bottleneck
   on these old Xeons.
4. **4 threads per worker (not 8)** — Amdahl kicks in fast without AVX2, and
   OpenMP contention across sockets in a VM is unpredictable. Fewer threads
   per worker + **many uvicorn workers** (`--workers 8` or more) is the way
   to use the 100 vCPUs.

### Rough runtime shape (single invoice page)

- Apple M-series, `apple` profile: sub-second per page.
- Xeon E7 v2, `apple` profile: ~4–8 s per page (medium model + AVX-only).
- Xeon E7 v2, `legacy-cpu` profile: ~1–2 s per page.
- Xeon E7 v2, `legacy-cpu` + `--workers 8`: ~1–2 s latency, ~8× throughput.

### Further wins (not enabled by default)

- `OCR_ENGINE=paddlex` with `paddlepaddle` installed → uses native Paddle
  inference with MKL-DNN kernels. Faster than the `transformers` backend on
  Intel, but adds a large dependency. Try after confirming the mobile-model
  win is not enough.
- INT8 quantization via ONNX Runtime — modest gain on this CPU (~1.5×
  because there's no VNNI), but halves memory bandwidth, which is the
  bottleneck. Requires re-exporting the mobile models.
- Do **not** bother with the VMware SVGA II. It's not compute-capable.
