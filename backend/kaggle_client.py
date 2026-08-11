"""HTTP client for chapterwise PDF extraction on a Kaggle GPU notebook."""
from __future__ import annotations

import os
import time
import uuid

import requests
from requests.exceptions import ConnectionError, SSLError, Timeout

DEFAULT_TIMEOUT = int(os.environ.get("KAGGLE_API_TIMEOUT", "120"))
EXTRACT_TIMEOUT = int(os.environ.get("KAGGLE_EXTRACT_TIMEOUT", "900"))
CHUNK_TIMEOUT = int(os.environ.get("KAGGLE_CHUNK_TIMEOUT", "120"))
UPLOAD_CHUNK_BYTES = int(os.environ.get("KAGGLE_UPLOAD_CHUNK_BYTES", str(4 * 1024 * 1024)))
CHUNK_THRESHOLD_BYTES = int(os.environ.get("KAGGLE_CHUNK_THRESHOLD_BYTES", str(10 * 1024 * 1024)))
UPLOAD_RETRIES = int(os.environ.get("KAGGLE_UPLOAD_RETRIES", "3"))
POLL_INTERVAL = int(os.environ.get("KAGGLE_POLL_INTERVAL", "5"))
POLL_TIMEOUT = int(os.environ.get("KAGGLE_POLL_TIMEOUT", "90"))
RESULT_TIMEOUT = int(os.environ.get("KAGGLE_RESULT_TIMEOUT", "120"))
LLM_TIMEOUT = int(os.environ.get("CHAPTERWISE_LLM_TIMEOUT", "600"))

_RETRYABLE = (SSLError, ConnectionError, Timeout)


def kaggle_base_url() -> str | None:
    url = os.environ.get("KAGGLE_API_BASE_URL", "").strip().rstrip("/")
    return url or None


def kaggle_headers(content_type: str | None = "application/json") -> dict[str, str]:
    headers: dict[str, str] = {}
    if content_type:
        headers["Content-Type"] = content_type
    secret = os.environ.get("KAGGLE_API_SECRET", "").strip()
    if not secret:
        if kaggle_enabled():
            print("WARNING: KAGGLE_ENABLED=true but KAGGLE_API_SECRET is not set")
        return headers
    headers["Authorization"] = f"Bearer {secret}"
    return headers


def kaggle_enabled() -> bool:
    return os.environ.get("KAGGLE_ENABLED", "false").strip().lower() in ("1", "true", "yes")


def kaggle_available() -> bool:
    return kaggle_enabled() and kaggle_base_url() is not None


def resolve_provider(env_name: str, default: str = "local") -> str:
    return os.environ.get(env_name, default).strip().lower()


def should_use_kaggle(provider: str) -> bool:
    mode = provider.strip().lower()
    if mode == "local":
        return False
    if mode in ("kaggle", "auto"):
        return kaggle_available()
    return False


def _is_retryable_tunnel_error(exc: BaseException) -> bool:
    if isinstance(exc, _RETRYABLE):
        return True
    message = str(exc).lower()
    return any(
        token in message
        for token in (
            "ssleoferror",
            "connection aborted",
            "connection reset",
            "max retries exceeded",
            "timed out",
            "unexpected eof",
        )
    )


def _post_with_retry(
    url: str,
    *,
    headers: dict[str, str],
    data: bytes,
    timeout: int,
    retries: int = UPLOAD_RETRIES,
) -> requests.Response:
    last_exc: BaseException | None = None
    for attempt in range(retries):
        try:
            return requests.post(url, headers=headers, data=data, timeout=timeout)
        except _RETRYABLE as exc:
            last_exc = exc
            if attempt + 1 < retries:
                wait_s = 2**attempt
                print(f"Kaggle upload retry {attempt + 2}/{retries} in {wait_s}s ({exc}) …")
                time.sleep(wait_s)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Kaggle upload failed without an exception")


def _post_json_with_retry(
    url: str,
    *,
    headers: dict[str, str],
    json_body: dict,
    timeout: int,
    retries: int = UPLOAD_RETRIES,
    label: str = "request",
) -> requests.Response:
    last_exc: BaseException | None = None
    for attempt in range(retries):
        try:
            return requests.post(url, headers=headers, json=json_body, timeout=timeout)
        except _RETRYABLE as exc:
            last_exc = exc
            if attempt + 1 < retries:
                wait_s = 2**attempt
                print(f"Kaggle {label} retry {attempt + 2}/{retries} in {wait_s}s ({exc}) …")
                time.sleep(wait_s)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Kaggle {label} failed without an exception")


def _get_with_retry(
    url: str,
    *,
    headers: dict[str, str],
    timeout: int,
    retries: int = UPLOAD_RETRIES,
    label: str = "request",
) -> requests.Response:
    last_exc: BaseException | None = None
    for attempt in range(retries):
        try:
            return requests.get(url, headers=headers, timeout=timeout)
        except _RETRYABLE as exc:
            last_exc = exc
            if attempt + 1 < retries:
                wait_s = 2**attempt
                print(f"Kaggle {label} retry {attempt + 2}/{retries} in {wait_s}s ({exc}) …")
                time.sleep(wait_s)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Kaggle {label} failed without an exception")


def _post_json_with_retry(
    url: str,
    *,
    headers: dict[str, str],
    timeout: int,
    json_body: dict | None = None,
    retries: int = UPLOAD_RETRIES,
    label: str = "request",
) -> requests.Response:
    last_exc: BaseException | None = None
    for attempt in range(retries):
        try:
            return requests.post(url, headers=headers, json=json_body or {}, timeout=timeout)
        except _RETRYABLE as exc:
            last_exc = exc
            if attempt + 1 < retries:
                wait_s = 2**attempt
                print(f"Kaggle {label} retry {attempt + 2}/{retries} in {wait_s}s ({exc}) …")
                time.sleep(wait_s)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Kaggle {label} failed without an exception")


def kaggle_status(timeout: int = 5) -> dict:
    base = kaggle_base_url()
    status = {
        "enabled": kaggle_enabled(),
        "base_url": base or None,
        "reachable": False,
        "gpus": None,
    }
    if not kaggle_available():
        return status
    try:
        resp = requests.get(f"{base}/health", headers=kaggle_headers(None), timeout=timeout)
        if resp.ok:
            data = resp.json()
            status["reachable"] = data.get("ok") is True or data.get("status") == "ok"
            status["gpus"] = data.get("gpus")
            status["llm"] = data.get("llm")
    except Exception as exc:
        status["error"] = str(exc)
    return status


def kaggle_health(timeout: int = 10) -> bool:
    base = kaggle_base_url()
    if not base:
        return False
    try:
        resp = requests.get(f"{base}/health", headers=kaggle_headers(None), timeout=timeout)
        if not resp.ok:
            return False
        data = resp.json()
        return data.get("ok") is True or data.get("status") == "ok"
    except Exception:
        return False


def kaggle_upload_pdf(pdf_bytes: bytes, timeout: int = DEFAULT_TIMEOUT) -> dict:
    base = kaggle_base_url()
    if not base:
        raise RuntimeError("KAGGLE_API_BASE_URL is not set")

    headers = kaggle_headers("application/pdf")
    resp = _post_with_retry(
        f"{base}/upload_pdf",
        headers=headers,
        data=pdf_bytes,
        timeout=timeout,
    )
    if not resp.ok:
        detail = resp.text[:500]
        raise RuntimeError(f"Kaggle /upload_pdf failed ({resp.status_code}): {detail}")
    return resp.json()


def kaggle_upload_pdf_chunked(
    pdf_bytes: bytes,
    *,
    chunk_size: int = UPLOAD_CHUNK_BYTES,
    timeout: int = CHUNK_TIMEOUT,
) -> dict:
    """Upload a large PDF in fixed-size chunks to survive cloudflared tunnels."""
    base = kaggle_base_url()
    if not base:
        raise RuntimeError("KAGGLE_API_BASE_URL is not set")

    total_bytes = len(pdf_bytes)
    if total_bytes <= chunk_size:
        return kaggle_upload_pdf(pdf_bytes, timeout=timeout)

    upload_id = uuid.uuid4().hex
    total_chunks = (total_bytes + chunk_size - 1) // chunk_size
    print(
        f"Kaggle chunked upload: {total_bytes} bytes in {total_chunks} chunk(s) "
        f"of up to {chunk_size // (1024 * 1024)} MB …"
    )

    last_result: dict | None = None
    for chunk_index in range(total_chunks):
        start = chunk_index * chunk_size
        end = min(start + chunk_size, total_bytes)
        chunk = pdf_bytes[start:end]

        headers = kaggle_headers("application/octet-stream")
        headers["X-Upload-Id"] = upload_id
        headers["X-Chunk-Index"] = str(chunk_index)
        headers["X-Total-Chunks"] = str(total_chunks)
        headers["X-Total-Bytes"] = str(total_bytes)

        resp = _post_with_retry(
            f"{base}/upload_chunk",
            headers=headers,
            data=chunk,
            timeout=timeout,
        )
        if not resp.ok:
            detail = resp.text[:500]
            raise RuntimeError(
                f"Kaggle /upload_chunk failed on chunk {chunk_index + 1}/{total_chunks} "
                f"({resp.status_code}): {detail}"
            )

        last_result = resp.json()
        if (chunk_index + 1) % 10 == 0 or chunk_index + 1 == total_chunks:
            print(f"  … uploaded chunk {chunk_index + 1}/{total_chunks}")

    if not last_result or not last_result.get("complete"):
        raise RuntimeError("Kaggle chunked upload finished without assembling the PDF")
    return last_result


def kaggle_upload_and_extract(pdf_bytes: bytes, timeout: int = EXTRACT_TIMEOUT) -> dict:
    """Upload PDF and extract in one Kaggle call (best for small files only)."""
    base = kaggle_base_url()
    if not base:
        raise RuntimeError("KAGGLE_API_BASE_URL is not set")

    headers = kaggle_headers("application/pdf")
    resp = _post_with_retry(
        f"{base}/upload_and_extract",
        headers=headers,
        data=pdf_bytes,
        timeout=timeout,
    )
    if not resp.ok:
        detail = resp.text[:500]
        raise RuntimeError(f"Kaggle /upload_and_extract failed ({resp.status_code}): {detail}")
    return resp.json()


def kaggle_extract_document(
    timeout: int = EXTRACT_TIMEOUT,
    *,
    page_start: int | None = None,
    page_end: int | None = None,
) -> dict:
    """Start async extraction on Kaggle and poll until complete."""
    base = kaggle_base_url()
    if not base:
        raise RuntimeError("KAGGLE_API_BASE_URL is not set")

    extract_body: dict[str, int] = {}
    if page_start is not None:
        extract_body["page_start"] = page_start
    if page_end is not None:
        extract_body["page_end"] = page_end

    if extract_body:
        print(f"Kaggle extract pages {page_start or 1}-{page_end or '?'} …")
    else:
        print("Kaggle extract all pages …")

    resp = _post_json_with_retry(
        f"{base}/extract",
        headers=kaggle_headers("application/json"),
        timeout=POLL_TIMEOUT,
        json_body=extract_body,
        label="extract start",
    )
    if not resp.ok:
        detail = resp.text[:500]
        raise RuntimeError(f"Kaggle /extract failed ({resp.status_code}): {detail}")

    start_data = resp.json()
    if start_data.get("status") == "complete":
        return _kaggle_fetch_extract_result(timeout=RESULT_TIMEOUT)

    deadline = time.time() + timeout
    last_status = start_data.get("status", "unknown")
    while time.time() < deadline:
        try:
            status_resp = _get_with_retry(
                f"{base}/extract/status",
                headers=kaggle_headers(None),
                timeout=POLL_TIMEOUT,
                label="extract status",
            )
        except _RETRYABLE as exc:
            print(f"Kaggle extract poll failed ({exc}); will retry …")
            time.sleep(POLL_INTERVAL)
            continue

        if not status_resp.ok:
            detail = status_resp.text[:500]
            raise RuntimeError(
                f"Kaggle /extract/status failed ({status_resp.status_code}): {detail}"
            )

        status_data = status_resp.json()
        last_status = status_data.get("status", "unknown")
        if last_status == "complete":
            num_pages = status_data.get("num_pages")
            total_chars = status_data.get("total_chars")
            print(
                f"Kaggle extract complete: {num_pages} pages, {total_chars} chars — fetching result …"
            )
            return _kaggle_fetch_extract_result(timeout=RESULT_TIMEOUT)
        if last_status == "error":
            raise RuntimeError(f"Kaggle extraction failed: {status_data.get('error')}")

        print(f"Kaggle extract running … (status={last_status})")
        time.sleep(POLL_INTERVAL)

    raise RuntimeError(
        f"Kaggle extraction timed out after {timeout}s (last status={last_status})"
    )


def _kaggle_fetch_extract_result(timeout: int = RESULT_TIMEOUT) -> dict:
    base = kaggle_base_url()
    if not base:
        raise RuntimeError("KAGGLE_API_BASE_URL is not set")

    resp = _get_with_retry(
        f"{base}/extract/result",
        headers=kaggle_headers(None),
        timeout=timeout,
        label="extract result",
    )
    if not resp.ok:
        detail = resp.text[:500]
        raise RuntimeError(f"Kaggle /extract/result failed ({resp.status_code}): {detail}")
    return resp.json()


def kaggle_extract_pdf(
    pdf_bytes: bytes,
    *,
    page_start: int | None = None,
    page_end: int | None = None,
) -> dict:
    """Upload (chunked when needed) and extract on Kaggle."""
    use_chunked = len(pdf_bytes) >= CHUNK_THRESHOLD_BYTES

    if use_chunked:
        kaggle_upload_pdf_chunked(pdf_bytes)
        return kaggle_extract_document(page_start=page_start, page_end=page_end)

    try:
        return kaggle_upload_and_extract(pdf_bytes)
    except Exception as exc:
        if not _is_retryable_tunnel_error(exc):
            raise
        kaggle_upload_pdf_chunked(pdf_bytes)
        return kaggle_extract_document(page_start=page_start, page_end=page_end)


def _kaggle_fetch_chat_result(timeout: int = RESULT_TIMEOUT) -> dict:
    base = kaggle_base_url()
    if not base:
        raise RuntimeError("KAGGLE_API_BASE_URL is not set")

    resp = _get_with_retry(
        f"{base}/chat/result",
        headers=kaggle_headers(None),
        timeout=timeout,
        label="chat result",
    )
    if not resp.ok:
        detail = resp.text[:500]
        raise RuntimeError(f"Kaggle /chat/result failed ({resp.status_code}): {detail}")
    return resp.json()


def _kaggle_chat_completions_async(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.2,
    timeout: int,
) -> str:
    """Start async chat on Kaggle and poll until complete (avoids tunnel 524)."""
    base = kaggle_base_url()
    if not base:
        raise RuntimeError("KAGGLE_API_BASE_URL is not set")

    payload = {
        "messages": messages,
        "temperature": temperature,
    }
    resp = _post_json_with_retry(
        f"{base}/chat",
        headers=kaggle_headers("application/json"),
        json_body=payload,
        timeout=POLL_TIMEOUT,
        label="chat start",
    )
    if not resp.ok:
        detail = resp.text[:500]
        raise RuntimeError(f"Kaggle /chat failed ({resp.status_code}): {detail}")

    start_data = resp.json()
    if start_data.get("status") == "complete":
        return _extract_chat_content(_kaggle_fetch_chat_result(timeout=RESULT_TIMEOUT))

    deadline = time.time() + timeout
    last_status = start_data.get("status", "unknown")
    while time.time() < deadline:
        try:
            status_resp = _get_with_retry(
                f"{base}/chat/status",
                headers=kaggle_headers(None),
                timeout=POLL_TIMEOUT,
                label="chat status",
            )
        except _RETRYABLE as exc:
            print(f"Kaggle chat poll failed ({exc}); will retry …")
            time.sleep(POLL_INTERVAL)
            continue

        if not status_resp.ok:
            detail = status_resp.text[:500]
            raise RuntimeError(
                f"Kaggle /chat/status failed ({status_resp.status_code}): {detail}"
            )

        status_data = status_resp.json()
        last_status = status_data.get("status", "unknown")
        if last_status == "complete":
            total_seconds = status_data.get("total_seconds")
            content_chars = status_data.get("content_chars")
            print(
                f"Kaggle chat complete"
                f"{f' in {total_seconds}s' if total_seconds is not None else ''}"
                f"{f' ({content_chars} chars)' if content_chars is not None else ''}"
                " — fetching result …"
            )
            return _extract_chat_content(_kaggle_fetch_chat_result(timeout=RESULT_TIMEOUT))
        if last_status == "error":
            raise RuntimeError(f"Kaggle chat failed: {status_data.get('error')}")

        print(f"Kaggle chat running … (status={last_status})")
        time.sleep(POLL_INTERVAL)

    raise RuntimeError(
        f"Kaggle chat timed out after {timeout}s (last status={last_status})"
    )


def _extract_chat_content(data: dict) -> str:
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("Kaggle chat response did not include choices.")
    content = (choices[0].get("message") or {}).get("content") or ""
    cleaned = content.strip()
    if not cleaned:
        raise RuntimeError("Kaggle chat returned empty content.")
    return cleaned


def kaggle_chat_completions(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.2,
    timeout: int | None = None,
) -> str:
    """Call Kaggle Llama chat and return assistant text (async poll to avoid tunnel 524)."""
    deadline = timeout or LLM_TIMEOUT
    try:
        return _kaggle_chat_completions_async(
            messages,
            temperature=temperature,
            timeout=deadline,
        )
    except RuntimeError as exc:
        message = str(exc)
        if "Kaggle /chat failed (404)" in message or "Kaggle /chat failed (405)" in message:
            print("Kaggle async /chat not available on server; trying sync /chat/completions …")
        else:
            raise

    base = kaggle_base_url()
    if not base:
        raise RuntimeError("KAGGLE_API_BASE_URL is not set")

    payload = {
        "messages": messages,
        "temperature": temperature,
    }
    resp = _post_json_with_retry(
        f"{base}/chat/completions",
        headers=kaggle_headers("application/json"),
        json_body=payload,
        timeout=deadline,
        label="chat/completions",
    )
    if not resp.ok:
        detail = resp.text[:500]
        raise RuntimeError(f"Kaggle /chat/completions failed ({resp.status_code}): {detail}")

    return _extract_chat_content(resp.json())
