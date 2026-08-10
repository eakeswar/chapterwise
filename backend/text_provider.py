"""Route text/chat calls to Kaggle Llama or Azure OpenAI."""

from __future__ import annotations

import os
from typing import Any

from kaggle_client import (
    kaggle_available,
    kaggle_chat_completions,
    kaggle_status,
    resolve_provider,
    should_use_kaggle,
)


def text_provider_name() -> str:
    return resolve_provider("TEXT_PROVIDER", "auto")


def should_use_kaggle_text(provider: str | None = None) -> bool:
    mode = (provider or text_provider_name()).strip().lower()
    if mode == "azure":
        return False
    if mode in ("kaggle", "auto"):
        return kaggle_available()
    return False


def _azure_chat_completion(messages: list[dict[str, str]], *, temperature: float) -> str:
    from azure_clients import get_text_client, get_text_deployment

    client = get_text_client()
    response = client.chat.completions.create(
        model=get_text_deployment(),
        messages=messages,
        temperature=temperature,
    )
    output = (response.choices[0].message.content or "").strip()
    if not output:
        raise ValueError("Empty response from Azure chat model.")
    return output


def chat_completion(messages: list[dict[str, str]], *, temperature: float = 0.2) -> str:
    """Return assistant text from the configured text provider."""
    provider = text_provider_name()

    if should_use_kaggle_text(provider):
        try:
            return kaggle_chat_completions(messages, temperature=temperature)
        except Exception as exc:
            if provider == "kaggle":
                raise
            print(f"Kaggle text failed ({exc}); falling back to Azure.")

    return _azure_chat_completion(messages, temperature=temperature)


def probe_text_provider() -> dict[str, Any]:
    provider = text_provider_name()
    result: dict[str, Any] = {
        "ok": False,
        "provider": provider,
    }

    if should_use_kaggle_text(provider):
        status = kaggle_status(timeout=10)
        llm = status.get("llm") or {}
        result.update(
            {
                "backend": "kaggle",
                "base_url": status.get("base_url"),
                "reachable": status.get("reachable"),
                "model": llm.get("model"),
                "llm_loaded": llm.get("loaded"),
            }
        )
        if not status.get("reachable"):
            result["error"] = status.get("error") or "Kaggle server unreachable"
            return result
        if llm.get("load_error"):
            result["error"] = llm["load_error"]
            return result

        try:
            sample = chat_completion(
                [{"role": "user", "content": "Reply with exactly: text-ok"}],
                temperature=0,
            )
            result["ok"] = True
            result["sample"] = sample.strip()
            return result
        except Exception as exc:
            result["error"] = str(exc)
            return result

    try:
        sample = _azure_chat_completion(
            [{"role": "user", "content": "Reply with exactly: text-ok"}],
            temperature=0,
        )
        result.update({"backend": "azure", "ok": True, "sample": sample.strip()})
    except Exception as exc:
        result["backend"] = "azure"
        result["error"] = str(exc)
    return result
