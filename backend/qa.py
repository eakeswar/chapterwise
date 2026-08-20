"""Grounded Q&A for a single topic (source text + cached explanation)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from topic_detail import (
    _compose_source_text,
    _pages_in_range,
    _trim_source_for_explanation,
    format_sse_event,
    get_cached_explanation,
)


def _build_ask_messages(
    title: str,
    question: str,
    source_text: str,
    explanation: str | None,
) -> list[dict[str, str]]:
    system_instructions = (
        "Answer ONLY using the provided context from this textbook section. "
        "If the answer is not in the context, say you do not have enough information "
        "from this section. Do not guess or use outside knowledge. "
        "Write plain text only—no markdown headings, bullet lists, or code fences."
    )
    parts = [f"Topic: {title}"]
    if explanation:
        parts.append("Explanation:\n" + explanation)
    parts.append("Source text:\n" + source_text)
    parts.append("Question:\n" + question)
    return [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def iter_ask_sse(
    topic_id: str,
    question: str,
    extraction: dict[str, Any],
    topics: dict[str, Any],
) -> Iterator[str]:
    """Yield SSE events for a grounded answer. Does not cache Q&A."""
    topics_by_id = topics.get("topics_by_id") or {}
    topic = topics_by_id.get(topic_id)
    if not topic:
        raise ValueError(f"Unknown topic id: {topic_id}")

    cleaned = (question or "").strip()
    if not cleaned:
        raise ValueError("Question is empty")

    page_start = int(topic["page_start"])
    page_end = int(topic["page_end"])
    pages = _pages_in_range(extraction["pages"], page_start, page_end)
    source_text = _trim_source_for_explanation(_compose_source_text(pages))
    explanation = get_cached_explanation(topic_id)

    if not source_text.strip() and not explanation:
        yield format_sse_event(
            "error",
            {"status": "failed", "error": "No source text for this topic."},
        )
        return

    messages = _build_ask_messages(str(topic["title"]), cleaned, source_text, explanation)
    assembled: list[str] = []
    try:
        from text_provider import chat_completion_stream

        for piece in chat_completion_stream(messages, temperature=0.2):
            assembled.append(piece)
            yield format_sse_event("delta", {"text": piece})
        answer = "".join(assembled).strip()
        if not answer:
            yield format_sse_event(
                "error",
                {"status": "failed", "error": "Empty response from text model."},
            )
            return
        yield format_sse_event("done", {"status": "ready"})
    except Exception as exc:
        print(f"Ask stream failed for {topic_id}: {exc}")
        status = "incomplete" if "".join(assembled).strip() else "failed"
        yield format_sse_event("error", {"status": status, "error": str(exc)})
