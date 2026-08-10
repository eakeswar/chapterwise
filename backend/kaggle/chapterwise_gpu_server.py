#!/usr/bin/env python3
"""
chapterwise PDF extraction server — run inside a Kaggle notebook (GPU T4 x2).

The notebook uses Kaggle CPU cores for parallel PyMuPDF/pdfplumber extraction.
Attach dataset `chapterwise-models`, set KAGGLE_API_SECRET, run this script,
then copy the cloudflared URL into backend/.env on your laptop.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
WORKING.mkdir(parents=True, exist_ok=True)

# Kaggle speed defaults — applied before pdf_extract is imported
os.environ.setdefault("CHAPTERWISE_DEFER_IMAGE_B64", "1")
os.environ.setdefault("CHAPTERWISE_SKIP_OVERLAP", "1")
os.environ.setdefault("CHAPTERWISE_USE_PROCESS_POOL", "1")
os.environ.setdefault("CHAPTERWISE_EXTRACT_WORKERS", str(os.cpu_count() or 4))

_PIP_PACKAGES: list[tuple[str, list[str]]] = [
    ("fastapi", ["fastapi"]),
    ("uvicorn", ["uvicorn"]),
    ("fitz", ["pymupdf"]),
    ("pdfplumber", ["pdfplumber"]),
]


def _quiet_pip(*packages: str) -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", *packages],
        env={**os.environ, "PIP_PROGRESS_BAR": "off", "PIP_DISABLE_PIP_VERSION_CHECK": "1"},
    )


def _ensure_packages() -> None:
    for import_name, pip_args in _PIP_PACKAGES:
        try:
            __import__(import_name)
        except ImportError:
            print(f"chapterwise: installing {import_name} …")
            _quiet_pip(*pip_args)


_ensure_packages()

from fastapi import Depends, FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402
import uvicorn  # noqa: E402


def _import_shared_modules() -> None:
    for name in ("security.py", "pdf_extract.py"):
        for candidate in INPUT_ROOT.rglob(name):
            sys.path.insert(0, str(candidate.parent))
            break
        else:
            local = WORKING / name
            if local.exists():
                sys.path.insert(0, str(WORKING))
                break


_import_shared_modules()

from pdf_extract import (  # noqa: E402
    defer_image_b64,
    extract_document,
    extraction_worker_count,
    skip_overlap_counts,
    use_process_pool,
)
from security import (  # noqa: E402
    MAX_PDF_BYTES,
    cors_origins,
    require_kaggle_secret,
    validate_pdf_bytes,
)

API_SECRET = require_kaggle_secret(os.environ.get("KAGGLE_API_SECRET"))
PORT = int(os.environ.get("CHAPTERWISE_GPU_PORT", "8766"))
PDF_PATH = WORKING / "active_doc.pdf"
UPLOAD_CHUNKS_DIR = WORKING / "upload_chunks"

app = FastAPI(title="chapterwise Kaggle Extract Server")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
_bearer = HTTPBearer(auto_error=False)
_pdf_lock = threading.Lock()
_extract_lock = threading.Lock()
_extract_job: dict[str, object] = {
    "status": "idle",
    "error": None,
    "result": None,
    "pdf_mtime": None,
    "page_start": None,
    "page_end": None,
}


class ExtractRequest(BaseModel):
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)


class ChatMessage(BaseModel):
    role: str = Field(min_length=1, max_length=32)
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    model: str | None = None


def _reset_extract_job() -> None:
    with _extract_lock:
        _extract_job["status"] = "idle"
        _extract_job["error"] = None
        _extract_job["result"] = None
        _extract_job["pdf_mtime"] = None
        _extract_job["page_start"] = None
        _extract_job["page_end"] = None


def _format_extraction_result(result: dict) -> dict:
    total_chars = sum(page["text_length"] for page in result["pages"])
    total_images = sum(len(page["images"]) for page in result["pages"])
    return {
        "ok": True,
        "pdf_path": str(PDF_PATH),
        "num_pages": result["num_pages"],
        "page_start": result["page_start"],
        "page_end": result["page_end"],
        "total_chars": total_chars,
        "total_images": total_images,
        "pages": result["pages"],
    }


def _run_extraction_worker(page_start: int | None, page_end: int | None) -> None:
    if not PDF_PATH.exists():
        with _extract_lock:
            _extract_job["status"] = "error"
            _extract_job["error"] = "No PDF uploaded to Kaggle server"
        return

    range_label = (
        f"pages {page_start}-{page_end}"
        if page_start is not None or page_end is not None
        else "all pages"
    )
    print(
        f"chapterwise Kaggle: extracting {range_label} "
        f"with {extraction_worker_count()} worker(s) …"
    )
    try:
        result = extract_document(PDF_PATH, page_start=page_start, page_end=page_end)
        payload = _format_extraction_result(result)
        print(
            f"chapterwise Kaggle: done — pages {payload['page_start']}-{payload['page_end']} "
            f"of {payload['num_pages']}, {payload['total_chars']} chars, "
            f"{payload['total_images']} images"
        )
        with _extract_lock:
            _extract_job["status"] = "complete"
            _extract_job["error"] = None
            _extract_job["result"] = payload
            _extract_job["pdf_mtime"] = PDF_PATH.stat().st_mtime
            _extract_job["page_start"] = payload["page_start"]
            _extract_job["page_end"] = payload["page_end"]
    except Exception as exc:
        print(f"chapterwise Kaggle extract failed: {exc}")
        with _extract_lock:
            _extract_job["status"] = "error"
            _extract_job["error"] = str(exc)
            _extract_job["result"] = None


def _start_extraction_async(
    *,
    page_start: int | None = None,
    page_end: int | None = None,
) -> dict:
    if not PDF_PATH.exists():
        raise HTTPException(status_code=400, detail="No PDF uploaded to Kaggle server")

    pdf_mtime = PDF_PATH.stat().st_mtime
    with _extract_lock:
        status = _extract_job["status"]
        if status == "running":
            return {"ok": True, "status": "running", "started": False}
        if (
            status == "complete"
            and _extract_job.get("pdf_mtime") == pdf_mtime
            and _extract_job.get("page_start") == page_start
            and _extract_job.get("page_end") == page_end
            and _extract_job.get("result") is not None
        ):
            return {"ok": True, "status": "complete", "started": False}

        _extract_job["status"] = "running"
        _extract_job["error"] = None
        _extract_job["result"] = None
        _extract_job["pdf_mtime"] = pdf_mtime
        _extract_job["page_start"] = page_start
        _extract_job["page_end"] = page_end

    threading.Thread(
        target=_run_extraction_worker,
        args=(page_start, page_end),
        daemon=True,
    ).start()
    return {"ok": True, "status": "running", "started": True}


def verify_token(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> None:
    if creds is None or creds.credentials != API_SECRET:
        raise HTTPException(status_code=401, detail="Invalid or missing API token")


def _gpu_count() -> int:
    try:
        import torch

        return int(torch.cuda.device_count())
    except Exception:
        return 0


@app.get("/health")
def health() -> dict:
    llm_status: dict = {"enabled": False}
    try:
        import llm_server

        llm_status = {"enabled": True, **llm_server.status()}
    except Exception as exc:
        llm_status = {"enabled": False, "error": str(exc)}

    return {
        "ok": True,
        "status": "ok",
        "gpus": _gpu_count(),
        "workers": extraction_worker_count(),
        "process_pool": use_process_pool(),
        "defer_image_b64": defer_image_b64(),
        "skip_overlap": skip_overlap_counts(),
        "pdf_loaded": PDF_PATH.exists(),
        "extract_status": _extract_job["status"],
        "llm": llm_status,
    }


def _pdf_page_count() -> int:
    import fitz

    doc = fitz.open(str(PDF_PATH))
    try:
        return len(doc)
    finally:
        doc.close()


def _save_validated_pdf(pdf_bytes: bytes) -> dict:
    try:
        validate_pdf_bytes(pdf_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    PDF_PATH.write_bytes(pdf_bytes)
    num_pages = _pdf_page_count()
    _reset_extract_job()
    print(f"chapterwise Kaggle: saved PDF ({len(pdf_bytes)} bytes, {num_pages} pages)")
    return {"ok": True, "num_pages": num_pages, "bytes": len(pdf_bytes)}


def _assemble_chunked_upload(upload_id: str, total_chunks: int, total_bytes: int) -> dict:
    session_dir = UPLOAD_CHUNKS_DIR / upload_id
    if not session_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"Unknown upload session: {upload_id}")

    parts: list[bytes] = []
    for chunk_index in range(total_chunks):
        chunk_path = session_dir / f"chunk_{chunk_index:05d}"
        if not chunk_path.is_file():
            raise HTTPException(
                status_code=400,
                detail=f"Missing chunk {chunk_index} for upload {upload_id}",
            )
        parts.append(chunk_path.read_bytes())

    pdf_bytes = b"".join(parts)
    shutil.rmtree(session_dir, ignore_errors=True)

    if total_bytes > 0 and len(pdf_bytes) != total_bytes:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Assembled PDF size mismatch for upload {upload_id}: "
                f"expected {total_bytes}, got {len(pdf_bytes)}"
            ),
        )

    return _save_validated_pdf(pdf_bytes)


@app.post("/upload_chunk")
async def upload_chunk(request: Request, _: None = Depends(verify_token)) -> dict:
    """Receive a PDF fragment; assemble and validate on the final chunk."""
    upload_id = request.headers.get("x-upload-id", "").strip()
    chunk_index_raw = request.headers.get("x-chunk-index", "").strip()
    total_chunks_raw = request.headers.get("x-total-chunks", "").strip()
    total_bytes_raw = request.headers.get("x-total-bytes", "0").strip()

    if not upload_id or len(upload_id) > 128:
        raise HTTPException(status_code=400, detail="Missing or invalid X-Upload-Id header")
    try:
        chunk_index = int(chunk_index_raw)
        total_chunks = int(total_chunks_raw)
        total_bytes = int(total_bytes_raw)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid chunk upload headers") from exc

    if total_chunks < 1 or chunk_index < 0 or chunk_index >= total_chunks:
        raise HTTPException(status_code=400, detail="Chunk index out of range")
    if total_bytes < 0 or total_bytes > MAX_PDF_BYTES:
        raise HTTPException(status_code=400, detail="Invalid X-Total-Bytes header")

    chunk_bytes = await request.body()
    if not chunk_bytes:
        raise HTTPException(status_code=400, detail="Empty chunk bytes")

    with _pdf_lock:
        session_dir = UPLOAD_CHUNKS_DIR / upload_id
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / f"chunk_{chunk_index:05d}").write_bytes(chunk_bytes)

        if chunk_index + 1 < total_chunks:
            print(
                f"chapterwise Kaggle: chunk {chunk_index + 1}/{total_chunks} "
                f"for upload {upload_id[:8]}… ({len(chunk_bytes)} bytes)"
            )
            return {
                "ok": True,
                "upload_id": upload_id,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "complete": False,
            }

        print(f"chapterwise Kaggle: final chunk for upload {upload_id[:8]}… — assembling PDF")
        result = _assemble_chunked_upload(upload_id, total_chunks, total_bytes)
        result.update(
            {
                "upload_id": upload_id,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "complete": True,
            }
        )
        return result


@app.post("/upload_pdf")
async def upload_pdf(request: Request, _: None = Depends(verify_token)) -> dict:
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_PDF_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"PDF exceeds maximum size ({MAX_PDF_BYTES // (1024 * 1024)} MB)",
                )
        except ValueError:
            pass

    pdf_bytes = await request.body()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file bytes")
    try:
        validate_pdf_bytes(pdf_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with _pdf_lock:
        return _save_validated_pdf(pdf_bytes)


@app.post("/extract")
def extract_pdf(
    body: ExtractRequest | None = None,
    _: None = Depends(verify_token),
) -> dict:
    """Start extraction in the background (returns immediately for cloudflared tunnels)."""
    request = body or ExtractRequest()
    return _start_extraction_async(
        page_start=request.page_start,
        page_end=request.page_end,
    )


@app.get("/extract/status")
def extract_status(_: None = Depends(verify_token)) -> dict:
    with _extract_lock:
        status = _extract_job["status"]
        payload: dict[str, object] = {"ok": True, "status": status}
        if status == "error":
            payload["error"] = _extract_job["error"]
        elif status == "complete" and _extract_job["result"]:
            result = _extract_job["result"]
            assert isinstance(result, dict)
            payload["num_pages"] = result["num_pages"]
            payload["page_start"] = result.get("page_start")
            payload["page_end"] = result.get("page_end")
            payload["total_chars"] = result["total_chars"]
            payload["total_images"] = result["total_images"]
        return payload


@app.get("/extract/result")
def extract_result(_: None = Depends(verify_token)) -> dict:
    with _extract_lock:
        status = _extract_job["status"]
        if status == "running":
            raise HTTPException(status_code=409, detail="Extraction still running")
        if status == "error":
            raise HTTPException(status_code=500, detail=str(_extract_job["error"]))
        if status != "complete" or not _extract_job["result"]:
            raise HTTPException(status_code=400, detail="No extraction result available")
        assert isinstance(_extract_job["result"], dict)
        return _extract_job["result"]


def _run_extraction_sync(
    *,
    page_start: int | None = None,
    page_end: int | None = None,
) -> dict:
    if not PDF_PATH.exists():
        raise HTTPException(status_code=400, detail="No PDF uploaded to Kaggle server")

    range_label = (
        f"pages {page_start}-{page_end}"
        if page_start is not None or page_end is not None
        else "all pages"
    )
    print(
        f"chapterwise Kaggle: extracting {range_label} "
        f"with {extraction_worker_count()} worker(s) …"
    )
    try:
        result = extract_document(PDF_PATH, page_start=page_start, page_end=page_end)
        payload = _format_extraction_result(result)
        with _extract_lock:
            _extract_job["status"] = "complete"
            _extract_job["error"] = None
            _extract_job["result"] = payload
            _extract_job["pdf_mtime"] = PDF_PATH.stat().st_mtime
            _extract_job["page_start"] = payload["page_start"]
            _extract_job["page_end"] = payload["page_end"]
        print(
            f"chapterwise Kaggle: done — pages {payload['page_start']}-{payload['page_end']} "
            f"of {payload['num_pages']}, {payload['total_chars']} chars, "
            f"{payload['total_images']} images"
        )
        return payload
    except Exception as exc:
        print(f"chapterwise Kaggle extract failed: {exc}")
        with _extract_lock:
            _extract_job["status"] = "error"
            _extract_job["error"] = str(exc)
            _extract_job["result"] = None
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _run_extraction_locked() -> dict:
    start = _start_extraction_async()
    if start.get("status") == "complete":
        with _extract_lock:
            assert isinstance(_extract_job["result"], dict)
            return _extract_job["result"]
    raise HTTPException(
        status_code=409,
        detail="Extraction is running asynchronously — poll GET /extract/status",
    )


@app.post("/upload_and_extract")
async def upload_and_extract(request: Request, _: None = Depends(verify_token)) -> dict:
    """Save PDF and extract in one call — faster than upload + extract separately."""
    content_length = request.headers.get("content-length")
    if content_length:
        try:
            if int(content_length) > MAX_PDF_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=f"PDF exceeds maximum size ({MAX_PDF_BYTES // (1024 * 1024)} MB)",
                )
        except ValueError:
            pass

    pdf_bytes = await request.body()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file bytes")
    try:
        validate_pdf_bytes(pdf_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with _pdf_lock:
        PDF_PATH.write_bytes(pdf_bytes)
        _reset_extract_job()
        print(f"chapterwise Kaggle: saved PDF ({len(pdf_bytes)} bytes)")
        return _run_extraction_sync()


@app.post("/chat/completions")
def chat_completions(body: ChatRequest, _: None = Depends(verify_token)) -> dict:
    """OpenAI-compatible chat endpoint backed by a local Llama model on GPU."""
    try:
        import llm_server
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail="llm_server.py is not available on this Kaggle server.",
        ) from exc

    started = time.perf_counter()
    try:
        content = llm_server.generate_chat(
            [{"role": message.role, "content": message.content} for message in body.messages],
            temperature=body.temperature,
        )
    except Exception as exc:
        print(f"chapterwise LLM chat failed: {exc}")
        raise HTTPException(status_code=502, detail=f"LLM chat failed: {exc}") from exc

    elapsed = round(time.perf_counter() - started, 2)
    print(f"chapterwise LLM chat complete in {elapsed}s ({len(content)} chars)")
    return {
        "id": "chapterwise-kaggle",
        "object": "chat.completion",
        "model": llm_server.model_id(),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "timing": {"total_seconds": elapsed},
    }


def _install_cloudflared() -> Path:
    dest = Path("/usr/local/bin/cloudflared")
    if dest.exists():
        return dest
    import urllib.request

    url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    print("chapterwise: downloading cloudflared …")
    urllib.request.urlretrieve(url, dest)
    dest.chmod(0o755)
    return dest


def start_tunnel() -> subprocess.Popen:
    cloudflared = _install_cloudflared()
    return subprocess.Popen(
        [str(cloudflared), "tunnel", "--url", f"http://127.0.0.1:{PORT}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def main() -> None:
    print("chapterwise Kaggle extract + LLM server starting …")
    print(
        f"  port={PORT}  workers={extraction_worker_count()}  gpus={_gpu_count()}"
        f"  defer_images={defer_image_b64()}  process_pool={use_process_pool()}"
    )
    try:
        import llm_server

        print(f"  llm_model={llm_server.model_id()}  llm_preload={os.environ.get('CHAPTERWISE_LLM_PRELOAD', '0')}")
        llm_server.preload()
    except Exception as exc:
        print(f"  llm unavailable at startup: {exc}")

    server = threading.Thread(
        target=lambda: uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning"),
        daemon=True,
    )
    server.start()
    time.sleep(2)

    tunnel = start_tunnel()
    public_url = None
    deadline = time.time() + 60
    while time.time() < deadline and public_url is None:
        line = tunnel.stdout.readline() if tunnel.stdout else ""
        if not line:
            time.sleep(0.2)
            continue
        match = re.search(r"(https://[a-z0-9-]+\.trycloudflare\.com)", line)
        if match:
            public_url = match.group(1)

    if public_url:
        print("\n" + "=" * 60)
        print("Add to backend/.env on your laptop:")
        print(f"  KAGGLE_API_BASE_URL={public_url}")
        print("  KAGGLE_API_SECRET=<same secret you set in this notebook>")
        print("  KAGGLE_ENABLED=true")
        print("  EXTRACT_PROVIDER=kaggle")
        print("=" * 60 + "\n")
    else:
        print("chapterwise: cloudflared tunnel URL not found — check notebook output.")

    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
