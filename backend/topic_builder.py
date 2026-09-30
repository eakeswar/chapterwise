"""Build an ordered topic tree from extracted PDF pages.

Default path (Plan B): heading regex — no LLM.
Optional: CHAPTERWISE_TOPIC_BUILDER=llm for the legacy Kaggle/Azure chunk path.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

DEFAULT_CHUNK_PAGES = int(os.environ.get("CHAPTERWISE_TOPIC_CHUNK_PAGES", "18"))
_RETRY_ASSISTANT_MAX_CHARS = int(os.environ.get("CHAPTERWISE_TOPIC_RETRY_CHARS", "2000"))

# regex (default) | llm
TOPIC_BUILDER = os.environ.get("CHAPTERWISE_TOPIC_BUILDER", "regex").strip().lower()

# Activity with OCR stutter "1.31.31.31.3" → capture 1.3 (requires ≥2 repeats)
_ACTIVITY_STUTTER_RE = re.compile(
    r"Activity(?:Activity|\s|_)*?(?P<num>\d+\.\d+)(?:(?P=num)){2,}",
    re.IGNORECASE,
)
_ACTIVITY_RE = re.compile(
    r"Activity(?:Activity|\s|_)*?(?P<num>\d+\.\d+)\b",
    re.IGNORECASE,
)
# 1.2.3 TITLE … (subsection)
_SUBSECTION_RE = re.compile(
    r"(?<![\d.])(?P<num>\d+\.\d+\.\d+)\s+(?P<title>[A-Z][A-Za-z0-9][A-Za-z0-9 \-]{1,60})"
)
# 1.3 Title … (section) — not 1.3.1
_SECTION_RE = re.compile(
    r"(?<![\d.])(?P<num>\d+\.\d+)(?!\.\d)\s+(?P<title>[A-Z][A-Za-z0-9][A-Za-z0-9 ?'\-]{1,50})"
)
# Chapter 1 / OCR "hapterC 1" — line-start only so mid-sentence
# refs ("see Chapter 7", "of ions in Chapter 4") are not headings.
_CHAPTER_RE = re.compile(
    r"^[ \t]*(?:Chapter|hapterC)\s*(?P<num>\d+)\b",
    re.IGNORECASE | re.MULTILINE,
)

_SECTION_TITLE_BLOCKLIST = re.compile(
    r"^(In|The|This|We|As|Of|And|Or|To|For|With|From|On|At|By|If|When|What|"
    r"How|Why|That|These|Those|There|Here|Also|After|Before|During|While|"
    r"Have|Has|Had|Does|Do|Did|Could|Will|Would|Should|May|Might|"
    r"Observe|Arrange|Give|Collect|Take|Put|Record|Start|Note|Keep|"
    r"Particles|Higher)\b",
    re.IGNORECASE,
)


def _clean_heading_title(raw: str) -> str:
    title = re.sub(r"\s+", " ", raw).strip(" .:-_|")
    title = re.split(
        r"\s{2,}|(?=Activity\b)|(?=\d+\.\d+(?:\.\d+)?\s+[A-Z])",
        title,
        maxsplit=1,
    )[0].strip(" .:-_|")
    # Drop OCR stutter fragments like "ofof"
    title = re.sub(r"\b(\w{2,})\1\b", r"\1", title, flags=re.IGNORECASE)
    words = title.split()
    if len(words) > 8:
        title = " ".join(words[:8])
    return title[:80]


def _looks_like_section_title(title: str) -> bool:
    cleaned = _clean_heading_title(title)
    if len(cleaned) < 3:
        return False
    if _SECTION_TITLE_BLOCKLIST.match(cleaned):
        return False
    words = [word for word in cleaned.split() if any(ch.isalpha() for ch in word)]
    if not words:
        return False
    if len(words) > 10:
        return False
    titled = sum(1 for word in words if word[0].isupper())
    if titled / len(words) >= 0.5:
        return True
    letters = [ch for ch in cleaned if ch.isalpha()]
    upper_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
    return upper_ratio >= 0.45


def _y_for_heading(
    number: str,
    keyword: str | None,
    line_boxes: list[dict[str, Any]] | None,
) -> float | None:
    """Top-down y (PDF points) of the physical line a heading was found on.

    Content-matched (not offset-matched): `line_boxes` comes from a
    separate PyMuPDF/RapidOCR box pass and is not guaranteed to align
    character-for-character with the regex-scanned page text, so this
    searches for the heading's distinctive number token instead. Heuristic,
    not exact — prefers lines where the number appears near the start
    (typical heading position) and, when given, a matching keyword
    (e.g. "Activity", "Chapter").
    """
    if not line_boxes or not number:
        return None
    needle = number.strip()
    if not needle:
        return None

    scored: list[tuple[tuple[int, int, int], dict[str, Any]]] = []
    for line in line_boxes:
        text = line.get("text") or ""
        idx = text.find(needle)
        if idx < 0:
            continue
        has_keyword = 0 if (keyword and keyword.lower() in text.lower()) else 1
        starts_early = 0 if idx <= 3 else 1
        scored.append(((has_keyword, starts_early, idx), line))

    if not scored:
        return None
    scored.sort(key=lambda item: item[0])
    return float(scored[0][1].get("y0", 0.0))


def _scan_page_headings(
    page_num: int, text: str, line_boxes: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    """Find heading-like matches on one page (document order by start index)."""
    if not text or not text.strip():
        return []

    candidates: list[dict[str, Any]] = []

    for match in _CHAPTER_RE.finditer(text):
        num = match.group("num")
        candidates.append(
            {
                "level": "chapter",
                "number": num,
                "title": f"Chapter {num}",
                "page": page_num,
                "start": match.start(),
                "heading_y": _y_for_heading(num, "Chapter", line_boxes),
                "kind": "chapter",
            }
        )

    for match in _ACTIVITY_STUTTER_RE.finditer(text):
        num = match.group("num")
        candidates.append(
            {
                "level": "subsection",
                "number": num,
                "title": f"Activity {num}",
                "page": page_num,
                "start": match.start(),
                "end": match.end(),
                "heading_y": _y_for_heading(num, "Activity", line_boxes),
                "kind": "activity",
            }
        )

    for match in _ACTIVITY_RE.finditer(text):
        # Skip if already covered by stutter match
        if any(
            abs(match.start() - c["start"]) <= 2 and c["kind"] == "activity"
            for c in candidates
        ):
            continue
        num = match.group("num")
        candidates.append(
            {
                "level": "subsection",
                "number": num,
                "title": f"Activity {num}",
                "page": page_num,
                "start": match.start(),
                "end": match.end(),
                "heading_y": _y_for_heading(num, "Activity", line_boxes),
                "kind": "activity",
            }
        )

    for match in _SUBSECTION_RE.finditer(text):
        num = match.group("num")
        title = _clean_heading_title(match.group("title"))
        if not _looks_like_section_title(title):
            continue
        candidates.append(
            {
                "level": "subsection",
                "number": num,
                "title": f"{num} {title}",
                "page": page_num,
                "start": match.start(),
                "heading_y": _y_for_heading(num, None, line_boxes),
                "kind": "subsection",
            }
        )

    for match in _SECTION_RE.finditer(text):
        num = match.group("num")
        title = _clean_heading_title(match.group("title"))
        if not _looks_like_section_title(title):
            continue
        if title.lower().startswith("activity"):
            continue
        candidates.append(
            {
                "level": "section",
                "number": num,
                "title": f"{num} {title}",
                "page": page_num,
                "start": match.start(),
                "heading_y": _y_for_heading(num, None, line_boxes),
                "kind": "section",
            }
        )

    candidates.sort(key=lambda item: (item["start"], {"chapter": 0, "subsection": 1, "section": 2, "activity": 3}.get(item["kind"], 9)))

    filtered: list[dict[str, Any]] = []
    last_start = -10**9
    for item in candidates:
        # Collapse near-duplicate detections at the same spot (OCR double hits).
        if item["start"] - last_start <= 2:
            continue
        filtered.append(item)
        last_start = item["start"]
    return filtered


def extract_headings(pages: list[dict[str, Any]], page_start: int, page_end: int) -> list[dict[str, Any]]:
    headings: list[dict[str, Any]] = []
    for page in pages:
        page_num = int(page["page"])
        if page_num < page_start or page_num > page_end:
            continue
        headings.extend(
            _scan_page_headings(page_num, page.get("text") or "", page.get("line_boxes"))
        )
    return headings


def _assign_page_ranges(
    headings: list[dict[str, Any]],
    page_start: int,
    page_end: int,
) -> list[dict[str, Any]]:
    """
    Approach (a): non-heading body text belongs to the preceding heading via page_start/page_end.
    First heading expands back to page_start; last expands forward to page_end.
    """
    if not headings:
        return []

    enriched: list[dict[str, Any]] = []
    for index, heading in enumerate(headings):
        node = dict(heading)
        start = page_start if index == 0 else int(heading["page"])
        if index + 1 < len(headings):
            next_page = int(headings[index + 1]["page"])
            if next_page > int(heading["page"]):
                end = next_page - 1
            else:
                end = int(heading["page"])
        else:
            end = page_end
        node["page_start"] = start
        node["page_end"] = max(start, min(end, page_end))
        enriched.append(node)
    return enriched


def _ensure_chapter(chapters: list[dict[str, Any]], page_start: int, page_end: int) -> dict[str, Any]:
    if chapters:
        return chapters[-1]
    chapter = {
        "title": f"Pages {page_start}–{page_end}",
        "page_start": page_start,
        "page_end": page_end,
        # No real heading for this synthetic node; treat as "starts at the
        # top of the page" so it still participates in same-page image
        # matching (anything before the first real heading belongs here).
        "heading_y": 0.0,
        "sections": [],
    }
    chapters.append(chapter)
    return chapter


def _ensure_section(
    chapter: dict[str, Any],
    *,
    title: str,
    page_start: int,
    page_end: int,
) -> dict[str, Any]:
    sections = chapter.setdefault("sections", [])
    if sections:
        return sections[-1]
    section = {
        "title": title,
        "page_start": page_start,
        "page_end": page_end,
        # Synthetic "Introduction" section has no real heading; see
        # _ensure_chapter for why this defaults to top-of-page.
        "heading_y": 0.0,
        "subsections": [],
    }
    sections.append(section)
    return section


def build_tree_from_headings(
    headings: list[dict[str, Any]],
    page_start: int,
    page_end: int,
) -> dict[str, Any]:
    """Assemble chapter → section → subsection tree from ordered headings."""
    ranged = _assign_page_ranges(headings, page_start, page_end)
    if not ranged:
        # Approach (b): nothing matched — one fallback topic covering the whole range.
        return {
            "chapters": [
                {
                    "title": f"Pages {page_start}–{page_end}",
                    "page_start": page_start,
                    "page_end": page_end,
                    "sections": [],
                }
            ]
        }

    chapters: list[dict[str, Any]] = []
    current_chapter: dict[str, Any] | None = None
    current_section: dict[str, Any] | None = None

    for heading in ranged:
        kind = heading["kind"]
        page_s = heading["page_start"]
        page_e = heading["page_end"]
        title = heading["title"]

        if kind == "chapter":
            current_chapter = {
                "title": title,
                "page_start": page_s,
                "page_end": page_e,
                "heading_y": heading.get("heading_y"),
                "sections": [],
            }
            chapters.append(current_chapter)
            current_section = None
            continue

        if current_chapter is None:
            current_chapter = _ensure_chapter(chapters, page_start, page_end)
            # Expand synthetic/fallback chapter to include early body pages
            current_chapter["page_start"] = min(int(current_chapter["page_start"]), page_s)
            current_chapter["page_end"] = max(int(current_chapter["page_end"]), page_e)

        if kind == "section":
            current_section = {
                "title": title,
                "page_start": page_s,
                "page_end": page_e,
                "heading_y": heading.get("heading_y"),
                "subsections": [],
            }
            current_chapter["sections"].append(current_section)
            current_chapter["page_end"] = max(int(current_chapter["page_end"]), page_e)
            continue

        # subsection or activity
        if current_section is None:
            current_section = _ensure_section(
                current_chapter,
                title="Introduction",
                page_start=page_s,
                page_end=page_e,
            )

        current_section["subsections"].append(
            {
                "title": title,
                "page_start": page_s,
                "page_end": page_e,
                "heading_y": heading.get("heading_y"),
            }
        )
        current_section["page_end"] = max(int(current_section["page_end"]), page_e)
        current_chapter["page_end"] = max(int(current_chapter["page_end"]), page_e)

    # Stretch first chapter start to range start so preamble body is not dropped
    if chapters:
        chapters[0]["page_start"] = page_start
        chapters[-1]["page_end"] = max(int(chapters[-1]["page_end"]), page_end)

    return {"chapters": chapters}


def build_topics_from_headings(
    pages: list[dict[str, Any]],
    *,
    page_start: int,
    page_end: int,
) -> dict[str, Any]:
    """Pure-regex topic build. Does not call Llama/Azure."""
    headings = extract_headings(pages, page_start, page_end)
    print(
        f"Regex topic build: pages {page_start}-{page_end}, "
        f"{len(headings)} heading match(es)"
    )
    for heading in headings:
        print(
            f"  p{heading['page']} [{heading['kind']}] {heading['title']}"
        )
    tree = build_tree_from_headings(headings, page_start, page_end)
    tree = assign_topic_ids(tree)
    flat = flatten_topics(tree)
    return {
        "topic_tree": tree,
        "topics_by_id": flat,
        "page_start": page_start,
        "page_end": page_end,
        "chunk_count": 1,
        "topic_count": len(flat),
        "builder": "regex",
        "heading_count": len(headings),
    }


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
            "heading_y": node.get("heading_y"),
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

    page_nums = [int(page["page"]) for page in pages]
    min_page = min(page_nums)
    max_page = max(page_nums)
    start_page = page_start or min_page
    end_page = page_end or max_page
    if start_page < min_page or end_page > max_page or start_page > end_page:
        raise ValueError(
            f"Invalid page range {start_page}-{end_page} for extracted pages {min_page}-{max_page}."
        )

    builder = TOPIC_BUILDER
    if builder not in ("regex", "llm"):
        print(f"WARNING: unknown CHAPTERWISE_TOPIC_BUILDER={builder!r}; using regex")
        builder = "regex"

    if builder == "regex":
        return build_topics_from_headings(pages, page_start=start_page, page_end=end_page)

    # Legacy LLM chunk path (CHAPTERWISE_TOPIC_BUILDER=llm)
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
        "builder": "llm",
    }
