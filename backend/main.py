"""chapterwise FastAPI backend."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
VENDOR_DIR = BASE_DIR / "vendor"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=BASE_DIR / ".env", override=True)
except ImportError:
    pass

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from pydantic import BaseModel, Field

from azure_clients import probe_all_azure_services
from kaggle_client import (
    kaggle_extract_pdf,
    resolve_provider,
    should_use_kaggle,
)
from pdf_extract import extract_document
from security import MAX_PDF_BYTES, cors_origins, validate_pdf_bytes
from topic_builder import DEFAULT_CHUNK_PAGES, build_topics_from_pages
from topic_detail import build_topic_detail, clear_explanation_cache
from tts import DEFAULT_VOICE, synthesize_speech

ACTIVE_PDF_PATH = BASE_DIR / "active_doc.pdf"
DEFAULT_PORT = 8766

app = FastAPI(title="chapterwise", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_active_extraction: dict[str, Any] | None = None
_active_topics: dict[str, Any] | None = None
_extraction_lock = threading.Lock()


class BuildTopicsRequest(BaseModel):
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    chunk_pages: int | None = Field(default=None, ge=1, le=40)


class TtsRequest(BaseModel):
    text: str = Field(min_length=1)
    voice: str = Field(default=DEFAULT_VOICE, min_length=1, max_length=32)


def _get_extraction() -> dict[str, Any]:
    if _active_extraction is None:
        raise HTTPException(status_code=400, detail="No PDF uploaded yet")
    return _active_extraction


def _get_topics() -> dict[str, Any]:
    if _active_topics is None:
        raise HTTPException(status_code=400, detail="Topics not built yet — call POST /build_topics first")
    return _active_topics

def _page_summary(page: dict[str, Any]) -> dict[str, Any]:
    return {
        "page": page["page"],
        "text_length": page["text_length"],
        "text_preview": page["text"][:240],
        "image_count": len(page["images"]),
        "images": [
            {
                "id": image["id"],
                "w": image["w"],
                "h": image["h"],
                "ext": image["ext"],
                "isBackground": image["isBackground"],
                "overlapWordCount": image["overlapWordCount"],
            }
            for image in page["images"]
        ],
    }


def _serialize_page(page: dict[str, Any], include_image_data: bool) -> dict[str, Any]:
    payload = {
        "page": page["page"],
        "width": page["width"],
        "height": page["height"],
        "text": page["text"],
        "text_length": page["text_length"],
        "images": [],
    }
    for image in page["images"]:
        item = {
            "id": image["id"],
            "xref": image["xref"],
            "w": image["w"],
            "h": image["h"],
            "x": image["x"],
            "y": image["y"],
            "ext": image["ext"],
            "isBackground": image["isBackground"],
            "overlapWordCount": image["overlapWordCount"],
        }
        if include_image_data:
            item["url"] = image["url"]
        payload["images"].append(item)
    return payload


@app.get("/health")
def health() -> dict[str, Any]:
    from kaggle_client import kaggle_status
    from pdf_extract import text_extractor_name, use_rapidocr_text
    from text_provider import text_provider_name

    extract_provider = resolve_provider("EXTRACT_PROVIDER", "local")
    kaggle = kaggle_status()
    return {
        "status": "ok",
        "extract_provider": extract_provider,
        "text_extractor": text_extractor_name(),
        "rapidocr_enabled": use_rapidocr_text(),
        "text_provider": text_provider_name(),
        "kaggle": kaggle,
    }


@app.get("/debug/text")
def debug_text() -> dict[str, Any]:
    """Smoke-test the configured text/chat provider (Kaggle Llama or Azure)."""
    from text_provider import probe_text_provider

    return probe_text_provider()


@app.get("/debug/azure")
def debug_azure() -> dict[str, Any]:
    """Smoke-test text, TTS, and image Azure deployments."""
    return probe_all_azure_services()


@app.post("/tts")
def tts(body: TtsRequest) -> Response:
    try:
        audio_bytes = synthesize_speech(body.text, voice=body.voice)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        print(f"TTS failed: {exc}")
        raise HTTPException(status_code=502, detail=f"TTS failed: {exc}") from exc
    return Response(content=audio_bytes, media_type="audio/mpeg")


@app.post("/upload_pdf")
async def upload_pdf(
    request: Request,
    page_start: int | None = Query(default=None, ge=1),
    page_end: int | None = Query(default=None, ge=1),
) -> dict[str, Any]:
    global _active_extraction, _active_topics

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

    upload_started = time.perf_counter()
    pdf_bytes = await request.body()
    receive_seconds = round(time.perf_counter() - upload_started, 2)
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file bytes")

    try:
        validate_pdf_bytes(pdf_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    with _extraction_lock:
        try:
            ACTIVE_PDF_PATH.write_bytes(pdf_bytes)
            extract_provider = resolve_provider("EXTRACT_PROVIDER", "local")
            extraction = None
            extract_started = time.perf_counter()

            if should_use_kaggle(extract_provider):
                try:
                    range_note = (
                        f" pages {page_start}-{page_end}"
                        if page_start is not None or page_end is not None
                        else ""
                    )
                    print(f"Kaggle upload+extract ({len(pdf_bytes)} bytes{range_note}) …")
                    kaggle_result = kaggle_extract_pdf(
                        pdf_bytes,
                        page_start=page_start,
                        page_end=page_end,
                    )
                    extraction = {
                        "pdf_path": str(ACTIVE_PDF_PATH),
                        "num_pages": kaggle_result["num_pages"],
                        "page_start": kaggle_result.get("page_start", 1),
                        "page_end": kaggle_result.get(
                            "page_end", kaggle_result["num_pages"]
                        ),
                        "pages": kaggle_result["pages"],
                    }
                    print(
                        f"Kaggle extraction complete: pages "
                        f"{extraction['page_start']}-{extraction['page_end']} "
                        f"of {extraction['num_pages']}"
                    )
                except Exception as kaggle_err:
                    if extract_provider == "kaggle":
                        raise HTTPException(
                            status_code=502,
                            detail=f"Kaggle extraction failed: {kaggle_err}",
                        ) from kaggle_err
                    print(f"Kaggle extraction failed ({kaggle_err}); using local extract.")

            if extraction is None:
                extraction = extract_document(
                    ACTIVE_PDF_PATH,
                    page_start=page_start,
                    page_end=page_end,
                )

            _active_extraction = extraction
            _active_topics = None
            clear_explanation_cache()

            extract_seconds = round(time.perf_counter() - extract_started, 2)
            total_seconds = round(time.perf_counter() - upload_started, 2)
            total_chars = sum(page["text_length"] for page in extraction["pages"])
            total_images = sum(len(page["images"]) for page in extraction["pages"])
            pages_extracted = len(extraction["pages"])
            file_mb = round(len(pdf_bytes) / (1024 * 1024), 2)
            print(
                f"Uploaded {ACTIVE_PDF_PATH.name}: "
                f"pages {extraction.get('page_start', 1)}-"
                f"{extraction.get('page_end', extraction['num_pages'])} "
                f"of {extraction['num_pages']}, "
                f"{total_chars} chars, {total_images} images "
                f"({file_mb} MB in {total_seconds}s: "
                f"receive {receive_seconds}s, extract {extract_seconds}s)"
            )
            return {
                "ok": True,
                "num_pages": extraction["num_pages"],
                "page_start": extraction.get("page_start", 1),
                "page_end": extraction.get("page_end", extraction["num_pages"]),
                "pages_extracted": pages_extracted,
                "total_chars": total_chars,
                "total_images": total_images,
                "timing": {
                    "file_bytes": len(pdf_bytes),
                    "file_mb": file_mb,
                    "receive_seconds": receive_seconds,
                    "extract_seconds": extract_seconds,
                    "total_seconds": total_seconds,
                },
            }
        except Exception as exc:
            print(f"PDF upload/extraction failed: {exc}")
            raise HTTPException(status_code=500, detail=f"Failed to process PDF: {exc}") from exc


@app.post("/build_topics")
def build_topics(body: BuildTopicsRequest | None = None) -> dict[str, Any]:
    global _active_topics

    extraction = _get_extraction()
    request = body or BuildTopicsRequest()

    with _extraction_lock:
        try:
            build_started = time.perf_counter()
            result = build_topics_from_pages(
                extraction["pages"],
                page_start=request.page_start,
                page_end=request.page_end,
                chunk_pages=request.chunk_pages or DEFAULT_CHUNK_PAGES,
            )
            build_seconds = round(time.perf_counter() - build_started, 2)
            _active_topics = result
            print(
                f"Built topic tree: {result['topic_count']} topics "
                f"from pages {result['page_start']}-{result['page_end']} "
                f"({result['chunk_count']} chunk(s)) in {build_seconds}s"
            )
            return {
                "ok": True,
                "page_start": result["page_start"],
                "page_end": result["page_end"],
                "chunk_count": result["chunk_count"],
                "topic_count": result["topic_count"],
                "chapters": result["topic_tree"]["chapters"],
                "timing": {
                    "total_seconds": build_seconds,
                },
            }
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            print(f"Topic build failed: {exc}")
            raise HTTPException(status_code=500, detail=f"Failed to build topics: {exc}") from exc


@app.get("/topic/{topic_id}")
def get_topic(topic_id: str) -> dict[str, Any]:
    extraction = _get_extraction()
    topics = _get_topics()
    try:
        return build_topic_detail(topic_id, extraction, topics)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/debug/topics")
def debug_topics() -> dict[str, Any]:
    topics = _get_topics()
    return {
        "page_start": topics["page_start"],
        "page_end": topics["page_end"],
        "chunk_count": topics["chunk_count"],
        "topic_count": topics["topic_count"],
        "topic_tree": topics["topic_tree"],
        "topics_by_id": topics["topics_by_id"],
    }


@app.get("/debug/extraction")
def debug_extraction_summary() -> dict[str, Any]:
    extraction = _get_extraction()
    return {
        "pdf_path": extraction["pdf_path"],
        "num_pages": extraction["num_pages"],
        "pages": [_page_summary(page) for page in extraction["pages"]],
    }


@app.get("/debug/extraction/{page_num}")
def debug_extraction_page(
    page_num: int,
    include_image_data: bool = Query(False, description="Include base64 image data URLs"),
) -> dict[str, Any]:
    extraction = _get_extraction()
    page_start = extraction.get("page_start", 1)
    page_end = extraction.get("page_end", extraction["num_pages"])
    if page_num < page_start or page_num > page_end:
        raise HTTPException(
            status_code=400,
            detail=f"Page {page_num} out of extracted range ({page_start}-{page_end})",
        )
    page = next(item for item in extraction["pages"] if item["page"] == page_num)
    return _serialize_page(page, include_image_data=include_image_data)


if __name__ == "__main__":
    import uvicorn

    print(f"Starting chapterwise backend on http://127.0.0.1:{DEFAULT_PORT}")
    uvicorn.run("main:app", host="127.0.0.1", port=DEFAULT_PORT, reload=False)
