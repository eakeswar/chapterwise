"""Send extracted page text to Azure OpenAI or Kaggle Llama and build an ordered topic tree."""

from __future__ import annotations

import json
import os
import re
from typing import Any

DEFAULT_CHUNK_PAGES = int(os.environ.get("CHAPTERWISE_TOPIC_CHUNK_PAGES", "18"))


def _repair_topic_json_string(json_str: str) -> str:
    """Apply common fixes for slightly malformed LLM JSON."""
    fixed = json_str.strip()
    fixed = re.sub(r",\s*}", "}", fixed)
    fixed = re.sub(r",\s*]", "]", fixed)
    fixed = re.sub(r'"\s*\]\s*\}', '"\n}', fixed)
    fixed = re.sub(r'"\s*,\s*\]\s*\}', '"\n}', fixed)

    open_braces = fixed.count("{")
    close_braces = fixed.count("}")
    if open_braces > close_braces:
        if fixed.count('"') % 2 != 0:
            fixed += '"'
        fixed += "}" * (open_braces - close_braces)

    open_brackets = fixed.count("[")
    close_brackets = fixed.count("]")
    if open_brackets > close_brackets:
        fixed += "]" * (open_brackets - close_brackets)

    return fixed


def extract_topic_json(text: str) -> dict[str, Any]:
    """Extract and repair a topic-tree JSON object from an LLM response."""
    text = text.strip()

    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace == -1 or last_brace == -1 or last_brace < first_brace:
        raise ValueError("No JSON object found in LLM response.")
    json_str = text[first_brace : last_brace + 1]

    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        fixed_str = _repair_topic_json_string(json_str)
        try:
            parsed = json.loads(fixed_str)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Failed to parse or repair JSON response: {exc}") from exc

    if not isinstance(parsed, dict):
        raise ValueError("Topic tree response must be a JSON object.")
    if "chapters" not in parsed or not isinstance(parsed["chapters"], list):
        raise ValueError('Topic tree response must include a "chapters" array.')
    return parsed


def _call_topic_model(messages: list[dict[str, str]]) -> dict[str, Any]:
    from text_provider import chat_completion

    last_error: ValueError | None = None
    conversation = list(messages)

    for attempt in range(2):
        output = chat_completion(conversation, temperature=0.1 if attempt else 0.2)
        if not output:
            raise ValueError("Empty response from topic extraction model.")
        try:
            return extract_topic_json(output)
        except ValueError as exc:
            last_error = exc
            print(f"Topic JSON parse failed (attempt {attempt + 1}/2): {exc}")
            if attempt == 0:
                conversation = conversation + [
                    {"role": "assistant", "content": output[:12000]},
                    {
                        "role": "user",
                        "content": (
                            "That response was invalid JSON. Return ONLY one valid JSON object "
                            "with the chapters schema. No markdown fences, no trailing commas, "
                            "no comments, no extra text."
                        ),
                    },
                ]

    if last_error is not None:
        raise last_error
    raise ValueError("Empty response from topic extraction model.")


def _format_page_chunk(pages: list[dict[str, Any]], page_start: int, page_end: int) -> str:
    lines = [f"Textbook extract for pages {page_start}-{page_end} (in document order):\n"]
    for page in pages:
        page_num = page["page"]
        if page_num < page_start or page_num > page_end:
            continue
        text = (page.get("text") or "").strip()
        lines.append(f"--- Page {page_num} ---")
        lines.append(text if text else "[no extractable text on this page]")
        lines.append("")
    return "\n".join(lines).strip()


def _build_chunk_messages(page_start: int, page_end: int, chunk_text: str) -> list[dict[str, str]]:
    system_instructions = (
        "You analyze textbook PDF text and return a hierarchical table of contents "
        "for ONLY the page range given.\n"
        "Return ONLY a raw JSON object with exactly this shape:\n"
        "{\n"
        '  "chapters": [\n'
        "    {\n"
        '      "title": "Chapter or major unit title",\n'
        '      "page_start": 1,\n'
        '      "page_end": 15,\n'
        '      "sections": [\n'
        "        {\n"
        '          "title": "Section title",\n'
        '          "page_start": 1,\n'
        '          "page_end": 8,\n'
        '          "subsections": [\n'
        "            {\n"
        '              "title": "Subsection title",\n'
        '              "page_start": 1,\n'
        '              "page_end": 4\n'
        "            }\n"
        "          ]\n"
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Rules:\n"
        f"- Only include topics whose content appears on pages {page_start} through {page_end}.\n"
        "- Preserve the order topics appear in the book. Never alphabetize or reorder.\n"
        "- page_start and page_end must be integers within the chunk range.\n"
        "- Use empty arrays for sections with no subsections.\n"
        "- Prefer a shallow tree: chapters and sections are enough; use subsections only when clearly needed.\n"
        "- Titles should match or closely paraphrase headings in the source text.\n"
        "- Do not invent topics that are not supported by the provided text.\n"
        "- Ignore running headers, footers, and page numbers as topic titles.\n"
        "- Output ONLY the JSON object — no markdown fences, no commentary."
    )
    return [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": chunk_text},
    ]



def _normalize_node(node: dict[str, Any], chunk_start: int, chunk_end: int) -> dict[str, Any]:
    title = str(node.get("title") or "").strip()
    if not title:
        raise ValueError("Topic node is missing a title.")

    page_start = int(node.get("page_start", chunk_start))
    page_end = int(node.get("page_end", page_start))
    page_start = max(chunk_start, min(page_start, chunk_end))
    page_end = max(page_start, min(page_end, chunk_end))

    return {
        "title": title,
        "page_start": page_start,
        "page_end": page_end,
    }


def _normalize_chunk_tree(tree: dict[str, Any], chunk_start: int, chunk_end: int) -> dict[str, Any]:
    chapters: list[dict[str, Any]] = []
    for chapter in tree.get("chapters") or []:
        if not isinstance(chapter, dict):
            continue
        chapter_node = _normalize_node(chapter, chunk_start, chunk_end)
        sections: list[dict[str, Any]] = []
        for section in chapter.get("sections") or []:
            if not isinstance(section, dict):
                continue
            section_node = _normalize_node(section, chunk_start, chunk_end)
            subsections: list[dict[str, Any]] = []
            for subsection in section.get("subsections") or []:
                if not isinstance(subsection, dict):
                    continue
                subsections.append(_normalize_node(subsection, chunk_start, chunk_end))
            section_node["subsections"] = subsections
            sections.append(section_node)
        chapter_node["sections"] = sections
        chapters.append(chapter_node)
    return {"chapters": chapters}


def merge_chunk_trees(trees: list[dict[str, Any]]) -> dict[str, Any]:
    chapters: list[dict[str, Any]] = []
    for tree in trees:
        chapters.extend(tree.get("chapters") or [])
    return {"chapters": chapters}


def assign_topic_ids(tree: dict[str, Any]) -> dict[str, Any]:
    for chapter_index, chapter in enumerate(tree.get("chapters") or [], start=1):
        chapter_id = f"ch{chapter_index}"
        chapter["id"] = chapter_id
        for section_index, section in enumerate(chapter.get("sections") or [], start=1):
            section_id = f"{chapter_id}_s{section_index}"
            section["id"] = section_id
            for subsection_index, subsection in enumerate(section.get("subsections") or [], start=1):
                subsection["id"] = f"{section_id}_ss{subsection_index}"
    return tree


def flatten_topics(tree: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Map topic id -> node metadata for later /topic and /ask endpoints."""
    topics: dict[str, dict[str, Any]] = {}

    def add_node(node: dict[str, Any], level: str, parent_id: str | None) -> None:
        topic_id = node["id"]
        topics[topic_id] = {
            "id": topic_id,
            "title": node["title"],
            "page_start": node["page_start"],
            "page_end": node["page_end"],
            "level": level,
            "parent_id": parent_id,
        }

    for chapter in tree.get("chapters") or []:
        add_node(chapter, "chapter", None)
        for section in chapter.get("sections") or []:
            add_node(section, "section", chapter["id"])
            for subsection in section.get("subsections") or []:
                add_node(subsection, "subsection", section["id"])

    return topics


def build_topics_from_pages(
    pages: list[dict[str, Any]],
    *,
    page_start: int | None = None,
    page_end: int | None = None,
    chunk_pages: int = DEFAULT_CHUNK_PAGES,
) -> dict[str, Any]:
    if not pages:
        raise ValueError("No extracted pages available.")

    total_pages = len(pages)
    start_page = page_start or 1
    end_page = page_end or total_pages
    if start_page < 1 or end_page > total_pages or start_page > end_page:
        raise ValueError(f"Invalid page range {start_page}-{end_page} for {total_pages}-page document.")

    chunk_trees: list[dict[str, Any]] = []
    chunk_pages = max(1, chunk_pages)

    current = start_page
    while current <= end_page:
        chunk_end = min(current + chunk_pages - 1, end_page)
        chunk_text = _format_page_chunk(pages, current, chunk_end)
        print(f"Building topics for pages {current}-{chunk_end} …")
        messages = _build_chunk_messages(current, chunk_end, chunk_text)
        chunk_tree = _call_topic_model(messages)
        chunk_trees.append(_normalize_chunk_tree(chunk_tree, current, chunk_end))
        current = chunk_end + 1

    merged = merge_chunk_trees(chunk_trees)
    merged = assign_topic_ids(merged)
    flat = flatten_topics(merged)
    return {
        "topic_tree": merged,
        "topics_by_id": flat,
        "page_start": start_page,
        "page_end": end_page,
        "chunk_count": len(chunk_trees),
        "topic_count": len(flat),
    }
