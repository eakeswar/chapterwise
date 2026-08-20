"""Classify, quality-check, and enhance topic images (Lanczos vs on-demand gpt-image-2)."""

from __future__ import annotations

import base64
import hashlib
import io
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from azure_clients import get_image_deployment, get_images_client, get_text_client, get_text_deployment

CHART_ASPECT_RATIO = 2.2
LINE_ART_MAX_CHROMA = 25.0
DEFAULT_SMALL_MIN_DIM = 400
DEFAULT_BLUR_VARIANCE = 100.0
GENERATED_IMAGE_SIZE = "1024x1024"
DECORATIVE_EDIT_PROMPT = (
    "Increase resolution and sharpness of this image. Do not change the people, "
    "their clothing, clothing colors, patterns, poses, or any other visual details "
    "— enhance quality only, preserve the exact subject as shown."
)
VISION_MAX_SIDE = 768
OCR_MAX_SIDE = 512
BAKED_TEXT_MIN_LEN = 2
BAKED_TEXT_MIN_SCORE = 0.5
OCR_WORKERS = 2

_ENHANCE_CACHE: dict[str, dict[str, Any]] = {}
_ENHANCE_CACHE_LOCK = threading.Lock()
_OCR_ENGINE: Any | None = None
_OCR_ENGINE_LOCK = threading.Lock()
_OCR_POOL: ThreadPoolExecutor | None = None
_OCR_POOL_LOCK = threading.Lock()
_OCR_INFLIGHT: set[str] = set()


def clear_enhance_cache() -> None:
    with _ENHANCE_CACHE_LOCK:
        _ENHANCE_CACHE.clear()
        _OCR_INFLIGHT.clear()


def small_image_min_dim() -> int:
    raw = (os.environ.get("SMALL_IMAGE_MIN_DIM") or "").strip()
    if not raw:
        return DEFAULT_SMALL_MIN_DIM
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_SMALL_MIN_DIM


def blur_variance_threshold() -> float:
    raw = (os.environ.get("BLUR_VARIANCE_THRESHOLD") or "").strip()
    if not raw:
        return DEFAULT_BLUR_VARIANCE
    try:
        return max(0.0, float(raw))
    except ValueError:
        return DEFAULT_BLUR_VARIANCE


def generate_decorative_image(
    prompt: str,
    *,
    size: str = GENERATED_IMAGE_SIZE,
    source_bytes: bytes | None = None,
) -> str:
    """Return base64 PNG. Prefers images.edit when source bytes are provided."""
    cleaned = (prompt or "").strip()
    if not cleaned:
        raise ValueError("Prompt is required for image generation.")

    client = get_images_client()
    model = get_image_deployment()
    if source_bytes:
        return _edit_decorative_image(client, model, source_bytes, cleaned, size)

    response = client.images.generate(
        model=model,
        prompt=cleaned,
        size=size,
        n=1,
    )
    item = response.data[0]
    if not getattr(item, "b64_json", None):
        raise ValueError("Image model did not return base64 data.")
    return item.b64_json


def _png_file_for_edit(source_bytes: bytes) -> io.BytesIO:
    from PIL import Image

    pil = Image.open(io.BytesIO(source_bytes))
    if pil.mode not in ("RGB", "RGBA"):
        pil = pil.convert("RGBA") if "A" in pil.mode else pil.convert("RGB")
    buffer = io.BytesIO()
    pil.save(buffer, format="PNG")
    buffer.seek(0)
    buffer.name = "source.png"
    return buffer


def _edit_decorative_image(
    client: Any,
    model: str,
    source_bytes: bytes,
    prompt: str,
    size: str,
) -> str:
    image_file = _png_file_for_edit(source_bytes)
    kwargs = {
        "model": model,
        "image": image_file,
        "prompt": prompt,
        "size": size,
        "n": 1,
    }
    try:
        response = client.images.edit(**kwargs, input_fidelity="high")
    except Exception as exc:
        print(f"images.edit input_fidelity=high failed ({exc}); retrying without it.")
        image_file.seek(0)
        response = client.images.edit(**kwargs)
    item = response.data[0]
    if not getattr(item, "b64_json", None):
        raise ValueError("Image edit did not return base64 data.")
    return item.b64_json


def enhance_topic_images(images: list[dict[str, Any]]) -> None:
    """Mutate gallery images: cheap classify inline; baked-text OCR in background."""
    for image in images:
        url = image.get("url")
        image["original_url"] = url
        if not url:
            image["enhance_kind"] = "original"
            image["enhance_role"] = "informational"
            image["classify_status"] = "ready"
            continue
        try:
            result = enhance_image(str(url))
        except Exception as exc:
            print(f"Image enhance failed for {image.get('id')}: {exc}")
            image["enhance_kind"] = "original"
            image["enhance_role"] = "informational"
            image["classify_status"] = "ready"
            continue
        image["url"] = result["url"]
        image["ext"] = result["ext"]
        image["enhance_kind"] = result["kind"]
        image["enhance_role"] = result["role"]
        image["classify_status"] = result.get("classify_status") or "ready"


def enhance_image(data_url: str) -> dict[str, Any]:
    """Route one encoded image. OCR for would-be-decorative images is backgrounded."""
    source_bytes = _bytes_from_data_url(data_url)
    digest = hashlib.sha256(source_bytes).hexdigest()
    with _ENHANCE_CACHE_LOCK:
        cached = _ENHANCE_CACHE.get(digest)
        if cached and cached.get("kind") == "generated":
            return dict(cached)
        if cached and cached.get("classify_status") in ("ready", "pending"):
            return dict(cached)

    from PIL import Image

    pil = Image.open(io.BytesIO(source_bytes))
    pixel_w, pixel_h = pil.size
    cheap_role = classify_role(pixel_w, pixel_h, pil)
    small = min(pixel_w, pixel_h) < small_image_min_dim()
    blurry = _laplacian_variance(pil) < blur_variance_threshold()
    ext = _ext_from_data_url(data_url)

    if cheap_role == "informational":
        return _cache_classified(
            digest,
            _informational_payload(data_url, pil, ext, small, blurry, pixel_w, pixel_h),
        )

    pending = {
        "url": data_url,
        "ext": ext,
        "kind": "pending",
        "role": "informational",
        "classify_status": "pending",
        "small": small,
        "blurry": blurry,
        "pixel_w": pixel_w,
        "pixel_h": pixel_h,
    }
    with _ENHANCE_CACHE_LOCK:
        existing = _ENHANCE_CACHE.get(digest)
        if existing and existing.get("kind") == "generated":
            return dict(existing)
        if existing and existing.get("classify_status") in ("ready", "pending"):
            return dict(existing)
        _ENHANCE_CACHE[digest] = pending
        already_queued = digest in _OCR_INFLIGHT
        if not already_queued:
            _OCR_INFLIGHT.add(digest)
    if not already_queued:
        _ocr_pool().submit(
            _run_baked_text_ocr,
            digest,
            source_bytes,
            data_url,
            ext,
            small,
            blurry,
            pixel_w,
            pixel_h,
        )
    return dict(pending)


def generate_decorative_on_demand(
    data_url: str,
    *,
    topic_title: str = "",
    explanation: str | None = None,
) -> dict[str, Any]:
    """Generate a decorative image. Returns status ready | failed; never raises Azure errors.

    topic_title informs style/tone only. explanation is accepted for callers
    but is never injected into the vision or generation prompts.
    """
    _ = explanation
    source_bytes = _bytes_from_data_url(data_url)
    digest = hashlib.sha256(source_bytes).hexdigest()
    with _ENHANCE_CACHE_LOCK:
        cached = _ENHANCE_CACHE.get(digest)
        if cached and cached.get("kind") == "generated":
            return {
                "status": "ready",
                "error": None,
                "url": cached["url"],
                "ext": cached.get("ext") or "png",
                "kind": "generated",
            }
        classified = cached and cached.get("classify_status") == "ready"
        decorative = classified and cached.get("role") == "decorative"
    if not decorative:
        return {
            "status": "failed",
            "error": "Image is not classified as decorative.",
            "url": data_url,
            "ext": _ext_from_data_url(data_url),
            "kind": "original",
        }

    from PIL import Image

    try:
        b64_png = generate_decorative_image(
            DECORATIVE_EDIT_PROMPT,
            source_bytes=source_bytes,
        )
        print(f"Decorative HD via images.edit ({digest[:12]}…)")
    except Exception as edit_exc:
        print(f"Decorative images.edit failed for {digest[:12]}… ({edit_exc}); falling back to vision + generate.")
        try:
            pil = Image.open(io.BytesIO(source_bytes))
            vision_caption = _caption_source_image(data_url, pil)
            prompt = _build_decorative_generation_prompt(
                vision_caption,
                topic_title=topic_title,
            )
            print(f"Decorative vision caption ({digest[:12]}…): {vision_caption}")
            print(f"Decorative generate prompt ({digest[:12]}…): {prompt}")
            b64_png = generate_decorative_image(prompt)
        except Exception as exc:
            print(f"Decorative image generation failed for {digest[:12]}…: {exc}")
            return {
                "status": "failed",
                "error": str(exc) or exc.__class__.__name__,
                "url": data_url,
                "ext": _ext_from_data_url(data_url),
                "kind": "original",
            }

    generated_url = f"data:image/png;base64,{b64_png}"
    payload = {
        "url": generated_url,
        "ext": "png",
        "kind": "generated",
        "role": "decorative",
        "classify_status": "ready",
    }
    with _ENHANCE_CACHE_LOCK:
        _ENHANCE_CACHE[digest] = payload
    return {
        "status": "ready",
        "error": None,
        "url": generated_url,
        "ext": "png",
        "kind": "generated",
    }


def classify_role(
    pixel_w: int,
    pixel_h: int,
    pil_image: Any | None = None,
) -> str:
    """Cheap classify: charts and gray line-art only. Overlap is ignored (often unset)."""
    short = max(1, min(pixel_w, pixel_h))
    aspect = max(pixel_w, pixel_h) / short
    if aspect >= CHART_ASPECT_RATIO:
        return "informational"
    if pil_image is not None and _is_line_art(pil_image):
        return "informational"
    return "decorative"


def _informational_payload(
    data_url: str,
    pil_image: Any,
    ext: str,
    small: bool,
    blurry: bool,
    pixel_w: int,
    pixel_h: int,
) -> dict[str, Any]:
    if small or blurry:
        return {
            "url": _png_data_url(_lanczos_upscale(pil_image)),
            "ext": "png",
            "kind": "lanczos",
            "role": "informational",
            "classify_status": "ready",
            "small": small,
            "blurry": blurry,
            "pixel_w": pixel_w,
            "pixel_h": pixel_h,
        }
    return {
        "url": data_url,
        "ext": ext,
        "kind": "original",
        "role": "informational",
        "classify_status": "ready",
        "small": small,
        "blurry": blurry,
        "pixel_w": pixel_w,
        "pixel_h": pixel_h,
    }


def _cache_classified(digest: str, payload: dict[str, Any]) -> dict[str, Any]:
    with _ENHANCE_CACHE_LOCK:
        existing = _ENHANCE_CACHE.get(digest)
        if existing and existing.get("kind") == "generated":
            return dict(existing)
        _ENHANCE_CACHE[digest] = payload
    return dict(payload)


def _ocr_pool() -> ThreadPoolExecutor:
    global _OCR_POOL
    with _OCR_POOL_LOCK:
        if _OCR_POOL is None:
            _OCR_POOL = ThreadPoolExecutor(max_workers=OCR_WORKERS, thread_name_prefix="img-ocr")
        return _OCR_POOL


def _get_rapidocr() -> Any:
    global _OCR_ENGINE
    if _OCR_ENGINE is not None:
        return _OCR_ENGINE
    with _OCR_ENGINE_LOCK:
        if _OCR_ENGINE is not None:
            return _OCR_ENGINE
        from rapidocr import RapidOCR

        _OCR_ENGINE = RapidOCR()
        return _OCR_ENGINE


def _run_baked_text_ocr(
    digest: str,
    source_bytes: bytes,
    data_url: str,
    ext: str,
    small: bool,
    blurry: bool,
    pixel_w: int,
    pixel_h: int,
) -> None:
    from PIL import Image

    try:
        pil = Image.open(io.BytesIO(source_bytes))
        has_text = _image_has_baked_text(pil)
        if has_text:
            payload = _informational_payload(data_url, pil, ext, small, blurry, pixel_w, pixel_h)
        else:
            payload = {
                "url": data_url,
                "ext": ext,
                "kind": "original",
                "role": "decorative",
                "classify_status": "ready",
                "small": small,
                "blurry": blurry,
                "pixel_w": pixel_w,
                "pixel_h": pixel_h,
            }
        print(
            f"Baked-text OCR {digest[:12]}…: "
            f"{'informational' if has_text else 'decorative'} kind={payload['kind']}"
        )
        _cache_classified(digest, payload)
    except Exception as exc:
        print(f"Baked-text OCR failed for {digest[:12]}… ({exc}); keeping informational")
        try:
            pil = Image.open(io.BytesIO(source_bytes))
            payload = _informational_payload(data_url, pil, ext, small, blurry, pixel_w, pixel_h)
        except Exception:
            payload = {
                "url": data_url,
                "ext": ext,
                "kind": "original",
                "role": "informational",
                "classify_status": "ready",
                "small": small,
                "blurry": blurry,
                "pixel_w": pixel_w,
                "pixel_h": pixel_h,
            }
        _cache_classified(digest, payload)
    finally:
        with _ENHANCE_CACHE_LOCK:
            _OCR_INFLIGHT.discard(digest)


def _image_has_baked_text(pil_image: Any) -> bool:
    """True when RapidOCR reads meaningful text in the image pixels."""
    import numpy as np

    sample = pil_image.convert("RGB")
    if max(sample.size) > OCR_MAX_SIDE:
        sample = sample.copy()
        sample.thumbnail((OCR_MAX_SIDE, OCR_MAX_SIDE))
    result = _get_rapidocr()(np.asarray(sample))
    texts = list(getattr(result, "txts", None) or [])
    scores = list(getattr(result, "scores", None) or [])
    for index, text in enumerate(texts):
        cleaned = str(text or "").strip()
        if len(cleaned) < BAKED_TEXT_MIN_LEN:
            continue
        score = float(scores[index]) if index < len(scores) else 1.0
        if score >= BAKED_TEXT_MIN_SCORE:
            return True
    return False


def _is_line_art(pil_image: Any) -> bool:
    """Gray / two-tone diagrams, formulas, and QR marks — not color photos."""
    import numpy as np

    rgb = pil_image.convert("RGB")
    sample = rgb
    max_side = 256
    if max(sample.size) > max_side:
        sample = sample.copy()
        sample.thumbnail((max_side, max_side))
    arr = np.asarray(sample, dtype=np.int16)
    chroma = arr.max(axis=2) - arr.min(axis=2)
    return float(chroma.mean()) < LINE_ART_MAX_CHROMA


def _laplacian_variance(pil_image: Any) -> float:
    import cv2
    import numpy as np

    gray = np.asarray(pil_image.convert("L"))
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _lanczos_upscale(pil_image: Any) -> bytes:
    from PIL import Image

    pixel_w, pixel_h = pil_image.size
    short = min(pixel_w, pixel_h)
    target = small_image_min_dim()
    if short <= 0:
        raise ValueError("Invalid image size for Lanczos upscale.")
    if short >= target:
        rgb = pil_image.convert("RGBA") if "A" in pil_image.mode else pil_image.convert("RGB")
        buffer = io.BytesIO()
        rgb.save(buffer, format="PNG")
        return buffer.getvalue()

    scale = target / short
    new_size = (max(1, round(pixel_w * scale)), max(1, round(pixel_h * scale)))
    resample = Image.Resampling.LANCZOS
    rgb = pil_image.convert("RGBA") if "A" in pil_image.mode else pil_image.convert("RGB")
    scaled = rgb.resize(new_size, resample)
    buffer = io.BytesIO()
    scaled.save(buffer, format="PNG")
    return buffer.getvalue()


def _caption_source_image(data_url: str, pil_image: Any) -> str:
    """Describe only what is visible. Never send formulas through this path."""
    vision_url = _downscaled_data_url(pil_image)
    client = get_text_client()
    response = client.chat.completions.create(
        model=get_text_deployment(),
        temperature=0.2,
        max_tokens=160,
        messages=[
            {
                "role": "system",
                "content": (
                    "You describe photographs for an image-redrawing model. "
                    "Describe only what is visually depicted in the attached image, "
                    "in one sentence. Do not reference any textbook chapter or topic. "
                    "Do not invent people, animals, vehicles, tools, or scenery "
                    "that are not clearly visible. If the photo is blurry or pixelated, "
                    "use conservative visual terms (grains, seeds, stalks, leaves) "
                    "instead of guessing a specific manufactured food. "
                    "No printed text, formulas, watermarks, or QR codes."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Describe only what is depicted in this image in one sentence — "
                            "do not reference the textbook chapter or topic at all."
                        ),
                    },
                    {"type": "image_url", "image_url": {"url": vision_url or data_url}},
                ],
            },
        ],
    )
    caption = (response.choices[0].message.content or "").strip()
    if not caption:
        raise ValueError("Empty vision caption for decorative image.")
    return caption


def _style_clause(topic_title: str) -> str:
    """Topic title may set illustration style/tone only — never new subjects."""
    title = (topic_title or "").strip()
    if title:
        return (
            f"Style: educational textbook illustration relevant to a section "
            f"titled {title}, photorealistic or painted, no text, no formulas, "
            "no watermarks."
        )
    return (
        "Style: educational textbook illustration, photorealistic or painted, "
        "no text, no formulas, no watermarks."
    )


def _build_decorative_generation_prompt(
    vision_caption: str,
    *,
    topic_title: str = "",
) -> str:
    """Caption is the subject. Topic title informs style only."""
    caption = (vision_caption or "").strip()
    if not caption:
        raise ValueError("Vision caption is required for image generation.")
    style = _style_clause(topic_title)
    return (
        f"Redraw the following image at higher fidelity, staying faithful to its "
        f"actual subject: {caption} {style} "
        "Do not add people, vehicles, animals, tools, or any subject matter not "
        "described in the image caption above, even if the topic mentions them. "
        "Do not add any people, animals, vehicles, or objects not already present "
        "in the described image."
    )


def _downscaled_data_url(pil_image: Any) -> str:
    sample = pil_image.convert("RGB")
    if max(sample.size) > VISION_MAX_SIDE:
        sample = sample.copy()
        sample.thumbnail((VISION_MAX_SIDE, VISION_MAX_SIDE))
    buffer = io.BytesIO()
    sample.save(buffer, format="JPEG", quality=85)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"


def _bytes_from_data_url(data_url: str) -> bytes:
    if "," not in data_url:
        raise ValueError("Image is not a data URL.")
    return base64.b64decode(data_url.split(",", 1)[1])


def _ext_from_data_url(data_url: str) -> str:
    header = data_url.split(",", 1)[0].lower()
    if "png" in header:
        return "png"
    if "jpeg" in header or "jpg" in header:
        return "jpeg"
    if "webp" in header:
        return "webp"
    return "png"


def _png_data_url(png_bytes: bytes) -> str:
    return f"data:image/png;base64,{base64.b64encode(png_bytes).decode('utf-8')}"
