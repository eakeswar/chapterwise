"""Build topic detail payloads from extraction + topic tree."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import fitz

from image_enhance import clear_enhance_cache, enhance_topic_images, generate_decorative_on_demand
from pdf_extract import encode_xref_data_url

_EXPLANATION_CACHE: dict[str, str] = {}
_EXPLANATION_SOURCE_CHAR_LIMIT = 28000

# data URLs for deferred images: "{pdf fingerprint}:{xref}" -> (ext, url)
_IMAGE_URL_CACHE: dict[str, tuple[str, str]] = {}
_IMAGE_URL_CACHE_LOCK = threading.Lock()
_BACKEND_DIR = Path(__file__).resolve().parent
_ACTIVE_PDF_FALLBACK = _BACKEND_DIR / "active_doc.pdf"


def clear_explanation_cache() -> None:
    _EXPLANATION_CACHE.clear()
    clear_enhance_cache()
    with _IMAGE_URL_CACHE_LOCK:
        _IMAGE_URL_CACHE.clear()


def get_cached_explanation(topic_id: str) -> str | None:
    return _EXPLANATION_CACHE.get(topic_id)


def _pages_in_range(
    pages: list[dict[str, Any]], page_start: int, page_end: int
) -> list[dict[str, Any]]:
    return [page for page in pages if page_start <= page["page"] <= page_end]


def _compose_source_text(pages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for page in pages:
        page_num = page["page"]
        text = (page.get("text") or "").strip()
        parts.append(f"--- Page {page_num} ---")
        parts.append(text if text else "[no extractable text on this page]")
    return "\n\n".join(parts).strip()


def _resolve_pdf_path(extraction: dict[str, Any]) -> Path | None:
    raw = extraction.get("pdf_path")
    if raw:
        candidate = Path(raw)
        if candidate.is_file():
            return candidate
    if _ACTIVE_PDF_FALLBACK.is_file():
        return _ACTIVE_PDF_FALLBACK
    return None


def _pdf_fingerprint(pdf_path: Path) -> str:
    try:
        stat = pdf_path.stat()
        return f"{pdf_path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}"
    except OSError:
        return str(pdf_path)


def _image_cache_key(fingerprint: str, xref: int) -> str:
    return f"{fingerprint}:{xref}"


def _hydrate_deferred_image_urls(
    pdf_path: Path,
    images: list[dict[str, Any]],
) -> None:
    fingerprint = _pdf_fingerprint(pdf_path)
    missing_xrefs: list[int] = []
    seen: set[int] = set()
    with _IMAGE_URL_CACHE_LOCK:
        for image in images:
            if image.get("url"):
                continue
            xref = image.get("xref")
            if xref is None:
                continue
            xref_int = int(xref)
            cached = _IMAGE_URL_CACHE.get(_image_cache_key(fingerprint, xref_int))
            if cached:
                image["ext"] = cached[0]
                image["url"] = cached[1]
                continue
            if xref_int not in seen:
                seen.add(xref_int)
                missing_xrefs.append(xref_int)

    if not missing_xrefs:
        return

    encoded: dict[int, tuple[str, str]] = {}
    doc = fitz.open(pdf_path)
    try:
        for xref in missing_xrefs:
            result = encode_xref_data_url(doc, xref)
            if result is None:
                continue
            encoded[xref] = result
    finally:
        doc.close()

    if not encoded:
        return

    with _IMAGE_URL_CACHE_LOCK:
        for xref, payload in encoded.items():
            _IMAGE_URL_CACHE[_image_cache_key(fingerprint, xref)] = payload
        for image in images:
            if image.get("url"):
                continue
            xref = image.get("xref")
            if xref is None:
                continue
            cached = _IMAGE_URL_CACHE.get(_image_cache_key(fingerprint, int(xref)))
            if cached:
                image["ext"] = cached[0]
                image["url"] = cached[1]


def _serialize_topic_images(
    pages: list[dict[str, Any]],
    pdf_path: Path | None,
) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    for page in pages:
        for image in page.get("images") or []:
            if image.get("isBackground"):
                continue
            images.append(
                {
                    "id": image["id"],
                    "page": page["page"],
                    "w": image["w"],
                    "h": image["h"],
                    "ext": image["ext"],
                    "xref": image.get("xref"),
                    "isBackground": image.get("isBackground", False),
                    "overlapWordCount": image.get("overlapWordCount", 0),
                    "url": image.get("url"),
                    "original_url": image.get("url"),
                    "enhance_kind": "original",
                    "enhance_role": "informational",
                    "classify_status": "pending",
                }
            )
    if pdf_path is not None:
        _hydrate_deferred_image_urls(pdf_path, images)
    enhance_topic_images(images)
    return images


def _build_explanation_messages(
    title: str,
    source_text: str,
    page_start: int,
    page_end: int,
) -> list[dict[str, str]]:
    page_label = f"page {page_start}" if page_start == page_end else f"pages {page_start}-{page_end}"
    system_instructions = (
        "You explain textbook topics in clear, plain language for a student. "
        "Use ONLY the provided source text. Do not invent facts, examples, or formulas "
        "not present in the source. If the source is sparse, summarize what is available "
        "and note gaps briefly. Write 2-4 short paragraphs. Use plain text only—no markdown "
        "headings, bullet lists, or code fences."
    )
    user_content = (
        f"Topic: {title}\n"
        f"Source: {page_label}\n\n"
        f"Source text:\n{source_text}"
    )
    return [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": user_content},
    ]


def _trim_source_for_explanation(source_text: str) -> str:
    if len(source_text) <= _EXPLANATION_SOURCE_CHAR_LIMIT:
        return source_text
    keep = _EXPLANATION_SOURCE_CHAR_LIMIT
    return source_text[:keep].rstrip() + "\n\n[source text truncated for model context limit]"


def _has_explainable_text(source_text: str) -> bool:
    for line in source_text.splitlines():
        line = line.strip()
        if not line or line.startswith("--- Page"):
            continue
        if line != "[no extractable text on this page]":
            return True
    return False


def _generate_explanation(
    topic_id: str,
    title: str,
    source_text: str,
    page_start: int,
    page_end: int,
) -> tuple[str | None, str, str | None]:
    """Return (explanation, status, error). status is ready | skipped | failed."""
    cached = _EXPLANATION_CACHE.get(topic_id)
    if cached is not None:
        return cached, "ready", None

    trimmed = source_text.strip()
    if not _has_explainable_text(trimmed):
        return None, "skipped", None

    trimmed = _trim_source_for_explanation(trimmed)
    messages = _build_explanation_messages(title, trimmed, page_start, page_end)

    try:
        from text_provider import chat_completion

        explanation = chat_completion(messages, temperature=0.3).strip()
        if not explanation:
            return None, "failed", "Empty response from text model."
        _EXPLANATION_CACHE[topic_id] = explanation
        return explanation, "ready", None
    except Exception as exc:
        print(f"Topic explanation failed for {topic_id}: {exc}")
        return None, "failed", str(exc)


def build_topic_detail(
    topic_id: str,
    extraction: dict[str, Any],
    topics: dict[str, Any],
) -> dict[str, Any]:
    topics_by_id = topics.get("topics_by_id") or {}
    topic = topics_by_id.get(topic_id)
    if not topic:
        raise ValueError(f"Unknown topic id: {topic_id}")

    page_start = int(topic["page_start"])
    page_end = int(topic["page_end"])
    pages = _pages_in_range(extraction["pages"], page_start, page_end)
    source_text = _compose_source_text(pages)
    images = _serialize_topic_images(pages, _resolve_pdf_path(extraction))

    explanation, explanation_status, explanation_error = _generate_explanation(
        topic_id,
        str(topic["title"]),
        source_text,
        page_start,
        page_end,
    )

    return {
        "ok": True,
        "topic": {
            "id": topic["id"],
            "title": topic["title"],
            "level": topic["level"],
            "page_start": page_start,
            "page_end": page_end,
            "parent_id": topic.get("parent_id"),
        },
        "source_text": source_text,
        "source_char_count": len(source_text),
        "explanation": explanation,
        "explanation_status": explanation_status,
        "explanation_error": explanation_error,
        "images": images,
        "image_count": len(images),
    }


class InformationalImageError(ValueError):
    """Raised when a generate request targets a formula/diagram image."""


def generate_topic_image(
    topic_id: str,
    image_id: str,
    extraction: dict[str, Any],
    topics: dict[str, Any],
) -> dict[str, Any]:
    """On-demand decorative generation. status is ready | failed."""
    topics_by_id = topics.get("topics_by_id") or {}
    topic = topics_by_id.get(topic_id)
    if not topic:
        raise ValueError(f"Unknown topic id: {topic_id}")

    page_start = int(topic["page_start"])
    page_end = int(topic["page_end"])
    pages = _pages_in_range(extraction["pages"], page_start, page_end)
    images = _serialize_topic_images(pages, _resolve_pdf_path(extraction))
    image = next((item for item in images if item.get("id") == image_id), None)
    if not image:
        raise ValueError(f"Unknown image id: {image_id}")

    if (
        image.get("classify_status") != "ready"
        or image.get("enhance_role") != "decorative"
    ):
        raise InformationalImageError(
            "Informational images cannot be regenerated with gpt-image-2."
        )

    original_url = image.get("original_url") or image.get("url")
    if not original_url:
        return {
            "ok": True,
            "status": "failed",
            "error": "Image has no encoded source to generate from.",
            "id": image_id,
            "url": None,
            "original_url": None,
            "ext": image.get("ext"),
            "enhance_kind": "original",
            "enhance_role": "decorative",
        }

    result = generate_decorative_on_demand(
        str(original_url),
        topic_title=str(topic.get("title") or ""),
        explanation=get_cached_explanation(topic_id),
    )
    display_url = result["url"] if result["status"] == "ready" else original_url
    return {
        "ok": True,
        "status": result["status"],
        "error": result.get("error"),
        "id": image_id,
        "url": display_url,
        "original_url": original_url,
        "ext": result.get("ext") or image.get("ext"),
        "enhance_kind": result["kind"],
        "enhance_role": "decorative",
    }
