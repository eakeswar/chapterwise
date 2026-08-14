"""Grounded Q&A for a single topic (source text + cached explanation)."""

from __future__ import annotations

from typing import Any

from topic_detail import (
    _compose_source_text,
    _pages_in_range,
    _trim_source_for_explanation,
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


def answer_question(
    topic_id: str,
    question: str,
    extraction: dict[str, Any],
    topics: dict[str, Any],
) -> dict[str, Any]:
    """Return { answer, status, error }. status is ready | failed."""
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
        return {
            "answer": None,
            "status": "failed",
            "error": "No source text for this topic.",
        }

    messages = _build_ask_messages(str(topic["title"]), cleaned, source_text, explanation)
    try:
        from text_provider import chat_completion

        answer = chat_completion(messages, temperature=0.2).strip()
        if not answer:
            return {
                "answer": None,
                "status": "failed",
                "error": "Empty response from text model.",
            }
        return {"answer": answer, "status": "ready", "error": None}
    except Exception as exc:
        print(f"Ask failed for {topic_id}: {exc}")
        return {"answer": None, "status": "failed", "error": str(exc)}
