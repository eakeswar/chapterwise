"""Send extracted page text to Azure OpenAI or Kaggle Llama and build an ordered topic tree."""

from __future__ import annotations

import json
import os
import re
from typing import Any

DEFAULT_CHUNK_PAGES = int(os.environ.get("CHAPTERWISE_TOPIC_CHUNK_PAGES", "18"))
_RETRY_ASSISTANT_MAX_CHARS = int(os.environ.get("CHAPTERWISE_TOPIC_RETRY_CHARS", "2000"))


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
                    {"role": "assistant", "content": output[:_RETRY_ASSISTANT_MAX_CHARS]},
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


_ANTI_LEAK_SENTENCE = (
    "The example above shows STRUCTURE ONLY using placeholder text in "
    "angle brackets. Do NOT copy any wording from the example. Every "
    "chapter/section/subsection title in your answer must come from the "
    "actual source text below, word-for-word or a close paraphrase of it."
)


def _collect_tree_titles(tree: dict[str, Any]) -> list[str]:
    titles: list[str] = []
    for chapter in tree.get("chapters") or []:
        if not isinstance(chapter, dict):
            continue
        titles.append(str(chapter.get("title") or ""))
        for section in chapter.get("sections") or []:
            if not isinstance(section, dict):
                continue
            titles.append(str(section.get("title") or ""))
            for subsection in section.get("subsections") or []:
                if isinstance(subsection, dict):
                    titles.append(str(subsection.get("title") or ""))
    return [title.strip() for title in titles if title.strip()]


def _title_word_overlap(left: str, right: str) -> float:
    left_words = {word for word in left.lower().split() if word}
    right_words = {word for word in right.lower().split() if word}
    if not left_words or not right_words:
        return 0.0
    shared = left_words & right_words
    smaller = min(len(left_words), len(right_words))
    return len(shared) / smaller


def _warn_sibling_title_duplicates(
    titles: list[str],
    level: str,
    chunk_start: int,
    chunk_end: int,
) -> None:
    cleaned = [title.strip() for title in titles if title.strip()]
    for index, left in enumerate(cleaned):
        for right in cleaned[index + 1:]:
            if left.lower() == right.lower() or _title_word_overlap(left, right) > 0.9:
                print(
                    f"WARNING: topic tree chunk pages {chunk_start}-{chunk_end} has "
                    f"near-duplicate sibling {level} titles: {left!r} vs {right!r}"
                )


def _warn_chunk_tree_sanity(
    tree: dict[str, Any],
    chunk_start: int,
    chunk_end: int,
    *,
    chunk_text: str = "",
    prompt_text: str = "",
) -> None:
    """Log warnings when a chunk tree looks over-split or copied from the prompt."""
    chapters = tree.get("chapters") or []
    chapter_count = len(chapters)
    if chapter_count > 5:
        print(
            f"WARNING: topic tree chunk pages {chunk_start}-{chunk_end} has "
            f"{chapter_count} top-level chapter nodes (expected ≤5 for a single in-chapter chunk)."
        )

    chapter_section_titles: list[str] = []
    for chapter in chapters:
        if isinstance(chapter, dict):
            chapter_section_titles.append(str(chapter.get("title") or ""))
            for section in chapter.get("sections") or []:
                if isinstance(section, dict):
                    chapter_section_titles.append(str(section.get("title") or ""))

    if chapter_section_titles:
        activity_count = sum(
            1 for title in chapter_section_titles if "activity" in title.lower()
        )
        if activity_count > len(chapter_section_titles) / 2:
            print(
                f"WARNING: topic tree chunk pages {chunk_start}-{chunk_end} has "
                f"{activity_count}/{len(chapter_section_titles)} chapter/section titles "
                f"containing 'Activity' (expected mostly real headings, not activity labels)."
            )

    if chunk_text and prompt_text:
        chunk_lower = chunk_text.lower()
        prompt_lower = prompt_text.lower()
        leaked: list[str] = []
        for title in _collect_tree_titles(tree):
            title_lower = title.lower()
            if len(title_lower) < 4:
                continue
            if title_lower in prompt_lower and title_lower not in chunk_lower:
                leaked.append(title)
        if leaked:
            preview = ", ".join(leaked[:5])
            extra = f" (+{len(leaked) - 5} more)" if len(leaked) > 5 else ""
            print(
                f"WARNING: topic tree chunk pages {chunk_start}-{chunk_end} may copy "
                f"prompt/example text (titles in prompt but not source): {preview}{extra}"
            )

    chapter_titles = [
        str(chapter.get("title") or "")
        for chapter in chapters
        if isinstance(chapter, dict)
    ]
    _warn_sibling_title_duplicates(chapter_titles, "chapter", chunk_start, chunk_end)

    for chapter in chapters:
        if not isinstance(chapter, dict):
            continue
        section_titles = [
            str(section.get("title") or "")
            for section in chapter.get("sections") or []
            if isinstance(section, dict)
        ]
        _warn_sibling_title_duplicates(section_titles, "section", chunk_start, chunk_end)
        for section in chapter.get("sections") or []:
            if not isinstance(section, dict):
                continue
            subsection_titles = [
                str(subsection.get("title") or "")
                for subsection in section.get("subsections") or []
                if isinstance(subsection, dict)
            ]
            _warn_sibling_title_duplicates(
                subsection_titles, "subsection", chunk_start, chunk_end
            )


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
        "What counts as each level:\n"
        "1. CHAPTER — only a numbered/titled top-level heading (e.g. chapter number + title "
        "at the start of a major unit, or when a chunk clearly begins a new major topic area). "
        f"Within one chunk that is still inside a single chapter (pages {page_start}-{page_end}), "
        "there should almost always be exactly ONE chapter node, not several parallel chapters.\n"
        "2. SECTION — only a numbered subheading in the pattern like \"1.1 Title Case Heading\" "
        "(a real topic label in the book). NOT an instruction sentence, NOT a bullet step, "
        "NOT an \"Activity N.N\" label.\n"
        "3. ACTIVITY boxes — any line starting with \"Activity\" followed by a number is CONTENT "
        "inside the surrounding section. It may appear at most once as a single subsection leaf "
        "(title like \"Activity 1.1\" only). It must NEVER become a chapter or section. "
        "Never nest sections/subsections under an activity from its instruction steps, bullets, "
        "or follow-up questions. Instruction sentences, bullet steps, and questions inside an "
        "activity are never topic nodes.\n\n"
        "Hard limits for this page range:\n"
        "- Do not create more than 4 section-level nodes.\n"
        "- Do not create more than 3 subsection-level nodes total for this page range.\n"
        "- Exception: only if the text unambiguously contains that many distinct numbered "
        "section headings (e.g. 1.1, 1.2, 1.3 …), not activities or bullets.\n\n"
        "General rules:\n"
        f"- Only include topics whose content appears on pages {page_start} through {page_end}.\n"
        "- Preserve the order topics appear in the book. Never alphabetize or reorder.\n"
        "- page_start and page_end must be integers within the chunk range.\n"
        "- Use empty arrays for sections with no subsections.\n"
        "- Do not invent a subsection from an ordinary sentence just because a "
        "section lacks a numbered subheading or Activity box. If there is no real "
        "numbered subheading and no Activity box under a section, its subsections "
        "array MUST be empty — do not fabricate a subsection title from body text.\n"
        "- Titles should copy the exact numbered heading string from the source "
        "text whenever one is present (e.g. a line matching a pattern like "
        "'1.1 <Title>' or '1.1.1 <TITLE>'). Only paraphrase if no such numbered "
        "heading exists in the source text for that node.\n"
        "- Do not invent topics that are not supported by the provided text.\n"
        "- Ignore running headers, footers, and page numbers as topic titles.\n"
        "- Output ONLY the JSON object — no markdown fences, no commentary.\n\n"
        "Worked example (structure only — placeholder titles, not real book text):\n"
        "Fictional source excerpt:\n"
        "  <example chapter heading line>\n"
        "  <N.N example section heading>\n"
        "  Activity <N.N>\n"
        "  • <example instruction bullet>\n"
        "  • <example instruction bullet>\n"
        "  <example follow-up question>\n"
        "  <N.N another section heading>\n"
        "\n"
        "CORRECT output shape:\n"
        "  chapters: [\n"
        "    { title: \"<chapter title from source text>\", sections: [\n"
        "      { title: \"<N.N section title from source text>\", subsections: [\n"
        "        { title: \"Activity <N.N>\" }\n"
        "      ]},\n"
        "      { title: \"<N.N section title from source text>\", subsections: [] }\n"
        "    ]}\n"
        "  ]\n"
        "WRONG (do NOT do this): separate chapters for \"Activity <N.N>\", sections for "
        "\"<example instruction bullet>\", subsections for \"<example follow-up question>\", "
        "or multiple top-level chapters inside one chunk.\n\n"
        f"{_ANTI_LEAK_SENTENCE}"
    )
    user_content = f"{_ANTI_LEAK_SENTENCE}\n\n{chunk_text}"
    return [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": user_content},
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
        normalized = _normalize_chunk_tree(chunk_tree, current, chunk_end)
        _warn_chunk_tree_sanity(
            normalized,
            current,
            chunk_end,
            chunk_text=chunk_text,
            prompt_text=messages[0]["content"] + "\n" + messages[1]["content"],
        )
        chunk_trees.append(normalized)
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
