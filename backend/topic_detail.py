"""Build topic detail payloads from extraction + topic tree."""

from __future__ import annotations

from typing import Any

_EXPLANATION_CACHE: dict[str, str] = {}
_EXPLANATION_SOURCE_CHAR_LIMIT = 28000


def clear_explanation_cache() -> None:
    _EXPLANATION_CACHE.clear()


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


def _serialize_topic_images(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    for page in pages:
        for image in page.get("images") or []:
            images.append(
                {
                    "id": image["id"],
                    "page": page["page"],
                    "w": image["w"],
                    "h": image["h"],
                    "ext": image["ext"],
                    "isBackground": image.get("isBackground", False),
                    "overlapWordCount": image.get("overlapWordCount", 0),
                    "url": image.get("url"),
                }
            )
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
    images = _serialize_topic_images(pages)

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
