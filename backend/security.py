"""Request limits and PDF validation for chapterwise."""
from __future__ import annotations

import os

MAX_PDF_BYTES = int(os.environ.get("CHAPTERWISE_MAX_PDF_BYTES", str(500 * 1024 * 1024)))

PDF_MAGIC = b"%PDF-"
WEAK_KAGGLE_SECRET = "chapterwise-kaggle-dev"


def require_kaggle_secret(secret: str | None) -> str:
    value = (secret or "").strip()
    if not value:
        raise SystemExit(
            "KAGGLE_API_SECRET must be set to a strong random value before starting the GPU server."
        )
    if value == WEAK_KAGGLE_SECRET:
        raise SystemExit(
            "KAGGLE_API_SECRET cannot be the default 'chapterwise-kaggle-dev'. "
            "Generate a new secret (e.g. openssl rand -hex 32)."
        )
    if len(value) < 16:
        raise SystemExit("KAGGLE_API_SECRET must be at least 16 characters.")
    return value


def cors_origins() -> list[str]:
    raw = os.environ.get(
        "CHAPTERWISE_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def validate_pdf_bytes(data: bytes) -> None:
    if len(data) > MAX_PDF_BYTES:
        max_mb = MAX_PDF_BYTES // (1024 * 1024)
        raise ValueError(f"PDF exceeds maximum size ({max_mb} MB)")
    if not data.startswith(PDF_MAGIC):
        raise ValueError("File is not a valid PDF (missing %PDF- header)")
