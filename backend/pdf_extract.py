"""Per-page text and image extraction from textbook PDFs."""

from __future__ import annotations

import base64
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import fitz

try:
    import pdfplumber as _pdfplumber
    _PDFPLUMBER_IMPORTED = True
except ImportError:
    _pdfplumber = None  # type: ignore[assignment,misc]
    _PDFPLUMBER_IMPORTED = False

MIN_IMAGE_DIM = 50


def defer_image_b64() -> bool:
    return os.environ.get("CHAPTERWISE_DEFER_IMAGE_B64", "0").strip().lower() in ("1", "true", "yes")


def skip_overlap_counts() -> bool:
    return os.environ.get("CHAPTERWISE_SKIP_OVERLAP", "0").strip().lower() in ("1", "true", "yes")


def pymupdf_text_only() -> bool:
    return os.environ.get("CHAPTERWISE_PYMUPDF_TEXT_ONLY", "0").strip().lower() in ("1", "true", "yes")


def use_process_pool() -> bool:
    if sys.platform == "win32":
        return False
    return os.environ.get("CHAPTERWISE_USE_PROCESS_POOL", "1").strip().lower() in ("1", "true", "yes")


def extraction_worker_count() -> int:
    raw = (os.environ.get("CHAPTERWISE_EXTRACT_WORKERS") or "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    return max(1, os.cpu_count() or 1)


def reconstruct_line_text(spans) -> str:
    line_text = ""
    sorted_spans = sorted(spans, key=lambda s: s.get("origin", (0, 0))[0])
    for span in sorted_spans:
        text = span.get("text", "")
        if not text:
            continue
        if not line_text:
            line_text = text
        else:
            needs_space = (
                not line_text.endswith("-")
                and not text.startswith(",")
                and not text.startswith(".")
            )
            line_text += " " + text if needs_space else text
    return line_text.strip()


class PDFMetadataAnalyzer:
    """Detect running headers/footers repeated across pages."""

    def __init__(self, doc: fitz.Document) -> None:
        self.running_headers_footers: set[str] = set()
        self._analyze(doc)

    def _analyze(self, doc: fitz.Document) -> None:
        from collections import Counter

        top_lines: list[str] = []
        bottom_lines: list[str] = []

        for page in doc:
            height = page.rect.height
            page_dict = page.get_text("dict")
            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue
                    line_text = reconstruct_line_text(spans)
                    if not line_text:
                        continue
                    y = spans[0]["origin"][1]
                    if y < height * 0.10:
                        top_lines.append(line_text)
                    elif y > height * 0.90:
                        bottom_lines.append(line_text)

        total_pages = len(doc)
        min_count = max(2, min(3, total_pages))

        for text, count in Counter(top_lines).items():
            if count >= min_count:
                self.running_headers_footers.add(text)

        for text, count in Counter(bottom_lines).items():
            if count >= min_count:
                self.running_headers_footers.add(text)


def _words_overlapping_image_count(words, ix0, iy0, ix1, iy1) -> int:
    count = 0
    for word in words:
        wx0, wy0, wx1, wy1 = word["x0"], word["top"], word["x1"], word["bottom"]
        ox = min(wx1, ix1) - max(wx0, ix0)
        oy = min(wy1, iy1) - max(wy0, iy0)
        if ox > 0 and oy > 0:
            word_w = wx1 - wx0 or 1
            if ox / word_w > 0.45:
                count += 1
    return count


def _attach_overlap_word_counts(
    images: list[dict[str, Any]],
    words: list[dict[str, Any]] | None,
    page_height: float,
) -> None:
    for image in images:
        image["overlapWordCount"] = 0

    if not images or not words:
        return

    for image in images:
        top_down = (
            image["x"],
            page_height - image["y"] - image["h"],
            image["x"] + image["w"],
            page_height - image["y"],
        )
        image["overlapWordCount"] = _words_overlapping_image_count(words, *top_down)


def _is_background_image(img_w: float, img_h: float, page_w: float, page_h: float) -> bool:
    if page_w <= 0 or page_h <= 0:
        return False
    if img_w > page_w * 0.90:
        return True
    if img_w > page_w * 0.85 and img_h > page_h * 0.85:
        return True
    if img_w * img_h > page_w * page_h * 0.45:
        return True
    return False


def _extract_page_images(
    doc: fitz.Document,
    page_obj: fitz.Page,
    page_num: int,
    page_width: float,
    page_height: float,
) -> list[dict[str, Any]]:
    images: list[dict[str, Any]] = []
    seen_xrefs: set[int] = set()

    for img_info in page_obj.get_images(full=True):
        xref = img_info[0]
        if xref in seen_xrefs:
            continue
        seen_xrefs.add(xref)

        try:
            rects = page_obj.get_image_rects(xref)
        except Exception:
            rects = []

        for rect in rects:
            img_w = rect.width
            img_h = rect.height
            if img_w < 5 or img_h < 5:
                continue
            if img_w < MIN_IMAGE_DIM or img_h < MIN_IMAGE_DIM:
                continue

            is_bg = _is_background_image(img_w, img_h, page_width, page_height)

            image_entry: dict[str, Any] = {
                "id": f"p{page_num}_i{len(images)}",
                "xref": xref,
                "x": rect.x0,
                "y": page_height - rect.y1,
                "w": img_w,
                "h": img_h,
                "ext": "jpeg",
                "isBackground": is_bg,
                "overlapWordCount": 0,
            }

            if defer_image_b64():
                images.append(image_entry)
                break

            try:
                img_dict = doc.extract_image(xref)
            except Exception:
                continue

            image_bytes = img_dict.get("image")
            if not image_bytes:
                continue

            ext = img_dict.get("ext", "png")
            image_entry["ext"] = ext
            image_entry["url"] = (
                f"data:image/{ext};base64,{base64.b64encode(image_bytes).decode('utf-8')}"
            )
            images.append(image_entry)
            break

    return images


def _clean_page_text(text: str, headers_footers: frozenset[str]) -> str:
    if not text:
        return ""

    lines = [line.strip() for line in text.splitlines()]
    cleaned: list[str] = []
    for line in lines:
        if not line:
            continue
        if line in headers_footers:
            continue
        if re.match(r"^\d+$", line):
            continue
        if re.match(r"^[ivxIVX]+$", line):
            continue
        if re.match(r"^(page|p\.)\s*\d+$", line, re.IGNORECASE):
            continue
        cleaned.append(line)

    return "\n".join(cleaned).strip()


def _extract_page_text_pymupdf(
    page_obj: fitz.Page,
    headers_footers: frozenset[str],
) -> str:
    page_dict = page_obj.get_text("dict")
    lines: list[str] = []
    height = page_obj.rect.height

    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            line_text = reconstruct_line_text(spans)
            if not line_text:
                continue
            if line_text in headers_footers:
                continue
            origin_y = spans[0]["origin"][1]
            if origin_y < height * 0.10 or origin_y > height * 0.90:
                if re.match(r"^\d+$", line_text) or re.match(r"^[ivxIVX]+$", line_text):
                    continue
                if re.match(r"^(page|p\.)\s*\d+$", line_text, re.IGNORECASE):
                    continue
            lines.append(line_text)

    return "\n".join(lines).strip()


def _extract_pdfplumber_words(pl_page) -> list[dict[str, Any]]:
    try:
        return pl_page.extract_words(
            keep_blank_chars=False,
            x_tolerance=3,
            y_tolerance=3,
        )
    except Exception:
        return pl_page.extract_words(
            keep_blank_chars=False,
            x_tolerance=3,
            y_tolerance=3,
        )


def _extract_page_standalone(
    pdf_path: str,
    page_num: int,
    headers_footers: frozenset[str],
) -> dict[str, Any]:
    """Extract one page using its own document handle (safe for parallel workers)."""
    doc = fitz.open(pdf_path)
    try:
        page_obj = doc[page_num - 1]
        page_width = page_obj.rect.width
        page_height = page_obj.rect.height

        images = _extract_page_images(doc, page_obj, page_num, page_width, page_height)
        text = ""

        if _PDFPLUMBER_IMPORTED and not pymupdf_text_only():
            try:
                with _pdfplumber.open(pdf_path) as pdf:
                    pl_page = pdf.pages[page_num - 1]
                    if not skip_overlap_counts():
                        words = _extract_pdfplumber_words(pl_page)
                        _attach_overlap_word_counts(images, words, page_height)
                    raw_text = pl_page.extract_text() or ""
                    if not raw_text.strip():
                        raw_text = pl_page.extract_text(layout=True) or ""
                    text = _clean_page_text(raw_text, headers_footers)
            except Exception as exc:
                print(f"pdfplumber failed for page {page_num} ({exc}), falling back to PyMuPDF")
                text = _extract_page_text_pymupdf(page_obj, headers_footers)
        else:
            text = _extract_page_text_pymupdf(page_obj, headers_footers)

        return {
            "page": page_num,
            "width": page_width,
            "height": page_height,
            "text": text,
            "text_length": len(text),
            "images": images,
        }
    finally:
        doc.close()


def extract_page(
    doc: fitz.Document,
    analyzer: PDFMetadataAnalyzer | None,
    pdf_path: str,
    page_num: int,
) -> dict[str, Any]:
    headers_footers = frozenset(analyzer.running_headers_footers if analyzer else set())
    return _extract_page_standalone(pdf_path, page_num, headers_footers)


def resolve_page_range(
    num_pages: int,
    page_start: int | None = None,
    page_end: int | None = None,
) -> tuple[int, int]:
    if num_pages < 1:
        raise ValueError("PDF has no pages")

    start = page_start or 1
    end = page_end or num_pages
    if start < 1 or end < 1 or start > num_pages or end > num_pages or start > end:
        raise ValueError(
            f"Invalid page range {start}-{end} for {num_pages}-page document"
        )
    return start, end


def extract_document(
    pdf_path: str | Path,
    *,
    page_start: int | None = None,
    page_end: int | None = None,
) -> dict[str, Any]:
    """Extract ordered per-page text and embedded images from a PDF."""
    path = Path(pdf_path)
    pdf_path = str(path)
    workers = extraction_worker_count()

    doc = fitz.open(pdf_path)
    try:
        pdf_num_pages = len(doc)
        analyzer = PDFMetadataAnalyzer(doc)
        headers_footers = frozenset(analyzer.running_headers_footers)
    finally:
        doc.close()

    start_page, end_page = resolve_page_range(pdf_num_pages, page_start, page_end)
    page_numbers = list(range(start_page, end_page + 1))
    extract_count = len(page_numbers)

    print(
        f"Extracting pages {start_page}-{end_page} of {pdf_num_pages} "
        f"using {workers} worker(s)"
        f" ({'process' if use_process_pool() else 'thread'} pool"
        f"{', deferred images' if defer_image_b64() else ''}) …"
    )

    pages_by_num: dict[int, dict[str, Any]] = {}
    if extract_count == 0:
        pages: list[dict[str, Any]] = []
    elif workers == 1 or extract_count == 1:
        pages = [
            _extract_page_standalone(pdf_path, page_num, headers_footers)
            for page_num in page_numbers
        ]
    else:
        executor_class = ProcessPoolExecutor if use_process_pool() else ThreadPoolExecutor
        try:
            with executor_class(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _extract_page_standalone,
                        pdf_path,
                        page_num,
                        headers_footers,
                    ): page_num
                    for page_num in page_numbers
                }
                done = 0
                for future in as_completed(futures):
                    page_num = futures[future]
                    pages_by_num[page_num] = future.result()
                    done += 1
                    if done % 50 == 0 or done == extract_count:
                        print(f"  … {done}/{extract_count} pages")
        except Exception as exc:
            if not use_process_pool():
                raise
            print(f"Process pool failed ({exc}); falling back to thread pool …")
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _extract_page_standalone,
                        pdf_path,
                        page_num,
                        headers_footers,
                    ): page_num
                    for page_num in page_numbers
                }
                for future in as_completed(futures):
                    pages_by_num[futures[future]] = future.result()

        pages = [pages_by_num[page_num] for page_num in page_numbers]

    total_chars = sum(page["text_length"] for page in pages)
    total_images = sum(len(page["images"]) for page in pages)
    print(
        f"Extraction complete: pages {start_page}-{end_page} of {pdf_num_pages}, "
        f"{total_chars} chars, {total_images} images"
    )

    return {
        "pdf_path": pdf_path,
        "num_pages": pdf_num_pages,
        "page_start": start_page,
        "page_end": end_page,
        "pages": pages,
    }
