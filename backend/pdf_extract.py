"""Per-page text and image extraction from textbook PDFs."""

from __future__ import annotations

import base64
import hashlib
import os
import re
import sys
import threading
import time
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
# Browsers cannot paint JPEG2000/JBIG2 data URLs; convert those to PNG.
_BROWSER_UNSAFE_IMAGE_EXTS = frozenset({"jpx", "jp2", "j2k", "jxr", "jb2"})


def encode_xref_data_url(doc: fitz.Document, xref: int) -> tuple[str, str] | None:
    """Return (ext, data URL) for an image xref, or None if extract fails."""
    try:
        img_dict = doc.extract_image(xref)
    except Exception:
        return None
    image_bytes = img_dict.get("image")
    if not image_bytes:
        return None
    ext = str(img_dict.get("ext") or "png").lower()
    if ext in _BROWSER_UNSAFE_IMAGE_EXTS:
        converted = _xref_to_png_bytes(doc, xref)
        if converted:
            image_bytes = converted
            ext = "png"
    data_url = f"data:image/{ext};base64,{base64.b64encode(image_bytes).decode('utf-8')}"
    return ext, data_url


def _xref_to_png_bytes(doc: fitz.Document, xref: int) -> bytes | None:
    try:
        pix = fitz.Pixmap(doc, xref)
        try:
            return pix.tobytes("png")
        except Exception:
            pix = fitz.Pixmap(fitz.csRGB, pix)
            return pix.tobytes("png")
    except Exception:
        return None


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


def text_extractor_name() -> str:
    return os.environ.get("CHAPTERWISE_TEXT_EXTRACTOR", "pdfplumber").strip().lower()


def use_rapidocr_text() -> bool:
    return text_extractor_name() == "rapidocr"


def rapidocr_force_all_pages() -> bool:
    return os.environ.get("CHAPTERWISE_RAPIDOCR_FORCE", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def rapidocr_dpi() -> int:
    raw = (os.environ.get("CHAPTERWISE_RAPIDOCR_DPI") or "200").strip()
    try:
        return max(72, int(raw))
    except ValueError:
        return 200


def extract_pages_per_worker() -> int:
    raw = (os.environ.get("CHAPTERWISE_EXTRACT_PAGES_PER_WORKER") or "10").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 10


def header_footer_sample_pages() -> int:
    raw = (os.environ.get("CHAPTERWISE_HEADER_FOOTER_SAMPLE_PAGES") or "36").strip()
    try:
        return max(10, min(60, int(raw)))
    except ValueError:
        return 36


def clear_headers_footers_cache() -> None:
    with _HEADERS_FOOTERS_CACHE_LOCK:
        _HEADERS_FOOTERS_CACHE.clear()


def _pdf_metadata_cache_key(path: Path) -> str:
    """Cheap per-PDF fingerprint: path + size + head/tail sample (not full 300MB hash)."""
    stat = path.stat()
    size = stat.st_size
    if size == 0:
        return f"{path.resolve()}|0"

    digest = hashlib.sha256()
    sample_bytes = 65536
    with path.open("rb") as pdf_file:
        digest.update(pdf_file.read(min(sample_bytes, size)))
        if size > sample_bytes:
            pdf_file.seek(max(0, size - sample_bytes))
            digest.update(pdf_file.read())
    return f"{path.resolve()}|{size}|{digest.hexdigest()[:16]}"


def _header_footer_sample_page_indices(total_pages: int) -> list[int]:
    """Evenly spaced 0-based page indices across the full document."""
    target = header_footer_sample_pages()
    if total_pages <= 0:
        return []
    if total_pages <= target:
        return list(range(total_pages))

    count = min(target, total_pages)
    if count == 1:
        return [0]

    step = (total_pages - 1) / (count - 1)
    indices: list[int] = []
    seen: set[int] = set()
    for i in range(count):
        idx = int(round(i * step))
        idx = max(0, min(total_pages - 1, idx))
        if idx not in seen:
            seen.add(idx)
            indices.append(idx)
    return indices


def _detect_headers_footers(
    doc: fitz.Document,
    pdf_num_pages: int,
    cache_key: str,
) -> frozenset[str]:
    with _HEADERS_FOOTERS_CACHE_LOCK:
        cached = _HEADERS_FOOTERS_CACHE.get(cache_key)
    if cached is not None:
        if debug_timing_enabled():
            print(f"header/footer cache hit ({len(cached)} patterns)")
        return cached

    sample_indices = _header_footer_sample_page_indices(pdf_num_pages)
    if debug_timing_enabled():
        print(
            f"header/footer sampling {len(sample_indices)} of {pdf_num_pages} pages "
            f"(0-based indices {sample_indices[0]}…{sample_indices[-1]})"
        )

    analyzer = PDFMetadataAnalyzer(doc, page_indices=sample_indices)
    headers_footers = frozenset(analyzer.running_headers_footers)
    with _HEADERS_FOOTERS_CACHE_LOCK:
        _HEADERS_FOOTERS_CACHE[cache_key] = headers_footers
    return headers_footers


_RAPIDOCR_PDF: Any = None
_RAPIDOCR_LOCK = threading.Lock()

_HEADERS_FOOTERS_CACHE: dict[str, frozenset[str]] = {}
_HEADERS_FOOTERS_CACHE_LOCK = threading.Lock()

_TIMING_LOCK = threading.Lock()
_TIMING_SECONDS: dict[str, float] = {}


def debug_timing_enabled() -> bool:
    return os.environ.get("CHAPTERWISE_DEBUG_TIMING", "0").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _clear_debug_timing() -> None:
    with _TIMING_LOCK:
        _TIMING_SECONDS.clear()


def _add_debug_timing(key: str, seconds: float) -> None:
    if not debug_timing_enabled() or seconds <= 0:
        return
    with _TIMING_LOCK:
        _TIMING_SECONDS[key] = _TIMING_SECONDS.get(key, 0.0) + seconds


def _print_debug_timing_summary(wall_seconds: float | None = None) -> None:
    if not debug_timing_enabled():
        return
    with _TIMING_LOCK:
        if not _TIMING_SECONDS:
            return
        worker_leaves = (
            "rapidocr_init",
            "rapidocr_batch_text",
            "pdfplumber_open",
            "pdfplumber_text",
            "batch_fitz_open",
            "images_extract_encode",
        )
        lines = ["CHAPTERWISE_DEBUG_TIMING breakdown (seconds):"]
        for key, secs in sorted(_TIMING_SECONDS.items(), key=lambda item: -item[1]):
            lines.append(f"  {key}: {secs:.3f}s")
        leaf_sum = sum(_TIMING_SECONDS.get(key, 0.0) for key in worker_leaves)
        pool_wall = _TIMING_SECONDS.get("worker_pool_wall", 0.0)
        pool_other = pool_wall - leaf_sum
        if pool_wall > 0 and pool_other > 1.0:
            lines.append(f"  worker_pool_other: {pool_other:.3f}s")
        if wall_seconds is not None:
            sequential = (
                _TIMING_SECONDS.get("pdf_open_initial", 0.0)
                + _TIMING_SECONDS.get("header_footer_detection", 0.0)
                + pool_wall
            )
            other = wall_seconds - sequential
            lines.append(f"  extract_document_wall: {wall_seconds:.3f}s")
            if other > 1.0:
                lines.append(f"  uncategorized: {other:.3f}s")
        print("\n".join(lines))


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

    def __init__(
        self,
        doc: fitz.Document,
        page_indices: list[int] | None = None,
    ) -> None:
        self.running_headers_footers: set[str] = set()
        self._analyze(doc, page_indices)

    def _analyze(
        self,
        doc: fitz.Document,
        page_indices: list[int] | None = None,
    ) -> None:
        from collections import Counter

        top_lines: list[str] = []
        bottom_lines: list[str] = []

        if page_indices is None:
            indices = list(range(len(doc)))
        else:
            indices = page_indices

        for page_idx in indices:
            page = doc[page_idx]
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

        scanned_pages = len(indices)
        min_count = max(2, min(3, scanned_pages))

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

            encoded = encode_xref_data_url(doc, xref)
            if encoded is None:
                continue
            ext, data_url = encoded
            image_entry["ext"] = ext
            image_entry["url"] = data_url
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


def _get_rapidocr_pdf() -> Any:
    global _RAPIDOCR_PDF
    if _RAPIDOCR_PDF is not None:
        return _RAPIDOCR_PDF
    with _RAPIDOCR_LOCK:
        if _RAPIDOCR_PDF is not None:
            return _RAPIDOCR_PDF
        try:
            from rapidocr_pdf import RapidOCRPDF
        except ImportError as exc:
            raise RuntimeError(
                "CHAPTERWISE_TEXT_EXTRACTOR=rapidocr but rapidocr-pdf is not installed. "
                "Run: python -m pip install --target backend/vendor rapidocr-pdf"
            ) from exc

        ocr_params: dict[str, Any] = {}
        if os.environ.get("CHAPTERWISE_RAPIDOCR_TORCH", "0").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            ocr_params["Global.with_torch"] = True

        init_started = time.perf_counter()
        _RAPIDOCR_PDF = RapidOCRPDF(
            dpi=rapidocr_dpi(),
            ocr_params=ocr_params or None,
        )
        _add_debug_timing("rapidocr_init", time.perf_counter() - init_started)
        return _RAPIDOCR_PDF


def _extract_texts_rapidocr_batch(
    pdf_path: str,
    page_numbers: list[int],
    headers_footers: frozenset[str],
) -> dict[int, str]:
    """Primary RapidOCR text path: one PDF pass per batch (digital pages use direct text)."""
    if not page_numbers:
        return {}

    extractor = _get_rapidocr_pdf()
    page_idx_list = [page_num - 1 for page_num in page_numbers]
    force_ocr = rapidocr_force_all_pages()

    with _RAPIDOCR_LOCK:
        ocr_started = time.perf_counter()
        rows = extractor(
            pdf_path,
            force_ocr=force_ocr,
            page_num_list=page_idx_list,
        )
        _add_debug_timing("rapidocr_batch_text", time.perf_counter() - ocr_started)

    texts: dict[int, str] = {}
    for row in rows or []:
        page_idx = int(row[0])
        raw = str(row[1] or "")
        texts[page_idx + 1] = _clean_page_text(raw, headers_footers)

    for page_num in page_numbers:
        texts.setdefault(page_num, "")

    return texts


def _page_batches(page_numbers: list[int], batch_size: int) -> list[list[int]]:
    batches: list[list[int]] = []
    for start in range(0, len(page_numbers), batch_size):
        batches.append(page_numbers[start : start + batch_size])
    return batches


def _extract_page_batch_standalone(
    pdf_path: str,
    page_numbers: list[int],
    headers_footers: frozenset[str],
) -> list[dict[str, Any]]:
    """Extract a page range with one PyMuPDF open; RapidOCR batches text in one call."""
    if not page_numbers:
        return []

    rapidocr_texts: dict[int, str] = {}
    if use_rapidocr_text():
        rapidocr_texts = _extract_texts_rapidocr_batch(
            pdf_path, page_numbers, headers_footers
        )

    pdfplumber_doc = None
    if not use_rapidocr_text() and _PDFPLUMBER_IMPORTED and not pymupdf_text_only():
        try:
            plumber_started = time.perf_counter()
            pdfplumber_doc = _pdfplumber.open(pdf_path)
            _add_debug_timing("pdfplumber_open", time.perf_counter() - plumber_started)
        except Exception as exc:
            print(f"pdfplumber open failed ({exc}); using PyMuPDF text only for this batch")

    fitz_started = time.perf_counter()
    doc = fitz.open(pdf_path)
    _add_debug_timing("batch_fitz_open", time.perf_counter() - fitz_started)
    pages_out: list[dict[str, Any]] = []
    try:
        for page_num in page_numbers:
            page_obj = doc[page_num - 1]
            page_width = page_obj.rect.width
            page_height = page_obj.rect.height
            images_started = time.perf_counter()
            images = _extract_page_images(doc, page_obj, page_num, page_width, page_height)
            _add_debug_timing("images_extract_encode", time.perf_counter() - images_started)

            if use_rapidocr_text():
                text = rapidocr_texts.get(page_num, "")
            elif pdfplumber_doc is not None:
                try:
                    text_started = time.perf_counter()
                    pl_page = pdfplumber_doc.pages[page_num - 1]
                    if not skip_overlap_counts():
                        words = _extract_pdfplumber_words(pl_page)
                        _attach_overlap_word_counts(images, words, page_height)
                    raw_text = pl_page.extract_text() or ""
                    if not raw_text.strip():
                        raw_text = pl_page.extract_text(layout=True) or ""
                    text = _clean_page_text(raw_text, headers_footers)
                    _add_debug_timing("pdfplumber_text", time.perf_counter() - text_started)
                except Exception as exc:
                    print(f"pdfplumber failed for page {page_num} ({exc}), using PyMuPDF")
                    text = _extract_page_text_pymupdf(page_obj, headers_footers)
            else:
                text = _extract_page_text_pymupdf(page_obj, headers_footers)

            pages_out.append(
                {
                    "page": page_num,
                    "width": page_width,
                    "height": page_height,
                    "text": text,
                    "text_length": len(text),
                    "images": images,
                }
            )
    finally:
        doc.close()
        if pdfplumber_doc is not None:
            pdfplumber_doc.close()

    return pages_out


def _extract_page_standalone(
    pdf_path: str,
    page_num: int,
    headers_footers: frozenset[str],
) -> dict[str, Any]:
    """Extract one page (legacy single-page entry; prefer batch worker)."""
    batch = _extract_page_batch_standalone(pdf_path, [page_num], headers_footers)
    return batch[0]


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
    _clear_debug_timing()
    extract_wall_started = time.perf_counter() if debug_timing_enabled() else 0.0

    open_started = time.perf_counter()
    doc = fitz.open(pdf_path)
    try:
        pdf_num_pages = len(doc)
        _add_debug_timing("pdf_open_initial", time.perf_counter() - open_started)
        cache_key = _pdf_metadata_cache_key(path)
        analyzer_started = time.perf_counter()
        headers_footers = _detect_headers_footers(doc, pdf_num_pages, cache_key)
        _add_debug_timing("header_footer_detection", time.perf_counter() - analyzer_started)
    finally:
        doc.close()

    start_page, end_page = resolve_page_range(pdf_num_pages, page_start, page_end)
    page_numbers = list(range(start_page, end_page + 1))
    extract_count = len(page_numbers)
    batch_size = extract_pages_per_worker()
    text_mode = "rapidocr" if use_rapidocr_text() else "pdfplumber"

    print(
        f"Extracting pages {start_page}-{end_page} of {pdf_num_pages} "
        f"using {workers} worker(s)"
        f" ({'process' if use_process_pool() else 'thread'} pool"
        f", text={text_mode}"
        f", batch={batch_size} pages/worker"
        f"{', deferred images' if defer_image_b64() else ''}) …"
    )

    pages_by_num: dict[int, dict[str, Any]] = {}
    pool_started = time.perf_counter()
    if extract_count == 0:
        pages: list[dict[str, Any]] = []
    elif workers == 1 or extract_count == 1:
        pages = _extract_page_batch_standalone(pdf_path, page_numbers, headers_footers)
    else:
        page_batches = _page_batches(page_numbers, batch_size)
        executor_class = ProcessPoolExecutor if use_process_pool() else ThreadPoolExecutor
        try:
            with executor_class(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _extract_page_batch_standalone,
                        pdf_path,
                        batch,
                        headers_footers,
                    ): batch
                    for batch in page_batches
                }
                done = 0
                for future in as_completed(futures):
                    batch_pages = future.result()
                    for page in batch_pages:
                        pages_by_num[page["page"]] = page
                    done += len(batch_pages)
                    if done % 50 == 0 or done == extract_count:
                        print(f"  … {done}/{extract_count} pages")
        except Exception as exc:
            if not use_process_pool():
                raise
            print(f"Process pool failed ({exc}); falling back to thread pool …")
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _extract_page_batch_standalone,
                        pdf_path,
                        batch,
                        headers_footers,
                    ): batch
                    for batch in page_batches
                }
                for future in as_completed(futures):
                    batch_pages = future.result()
                    for page in batch_pages:
                        pages_by_num[page["page"]] = page

        pages = [pages_by_num[page_num] for page_num in page_numbers]

    _add_debug_timing("worker_pool_wall", time.perf_counter() - pool_started)

    total_chars = sum(page["text_length"] for page in pages)
    total_images = sum(len(page["images"]) for page in pages)
    print(
        f"Extraction complete: pages {start_page}-{end_page} of {pdf_num_pages}, "
        f"{total_chars} chars, {total_images} images"
    )

    wall_seconds = (
        time.perf_counter() - extract_wall_started if debug_timing_enabled() else None
    )
    _print_debug_timing_summary(wall_seconds)

    return {
        "pdf_path": pdf_path,
        "num_pages": pdf_num_pages,
        "page_start": start_page,
        "page_end": end_page,
        "pages": pages,
    }
