"""Azure OpenAI client helpers for text, TTS, and image endpoints."""

from __future__ import annotations

import os
from typing import Any, Literal

DEFAULT_SERVICES_BASE_URL = "https://edzhub-resource.services.ai.azure.com/openai/v1"

AzureServiceKind = Literal["text", "tts", "image"]


def _require_env(name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if not value:
        raise ValueError(f"{name} is not set in backend/.env")
    return value


def _optional_env(*names: str, default: str = "") -> str:
    for name in names:
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return default


def get_api_key() -> str:
    return _require_env("AZURE_OPENAI_API_KEY")


def get_chat_api_key() -> str:
    return _require_env("AZURE_CHAT_API_KEY")


def get_text_base_url() -> str:
    return _require_env("AZURE_CHAT_BASE_URL")


def get_tts_base_url() -> str:
    return _optional_env(
        "AZURE_TTS_BASE_URL",
        "AZURE_SERVICES_BASE_URL",
        "AZURE_IMAGES_BASE_URL",
        default=DEFAULT_SERVICES_BASE_URL,
    )


def get_images_base_url() -> str:
    return _optional_env(
        "AZURE_IMAGES_BASE_URL",
        "AZURE_SERVICES_BASE_URL",
        default=DEFAULT_SERVICES_BASE_URL,
    )


def get_text_deployment() -> str:
    return _require_env("AZURE_TEXT_DEPLOYMENT")


def get_tts_deployment() -> str:
    return _require_env("AZURE_TTS_DEPLOYMENT")


def get_image_deployment() -> str:
    return _require_env("AZURE_IMAGE_DEPLOYMENT")


def get_text_client():
    from openai import OpenAI

    return OpenAI(api_key=get_chat_api_key(), base_url=get_text_base_url())


def get_tts_client():
    from openai import OpenAI

    return OpenAI(api_key=get_api_key(), base_url=get_tts_base_url())


def get_images_client():
    from openai import OpenAI

    return OpenAI(api_key=get_api_key(), base_url=get_images_base_url())


def _error_summary(exc: Exception) -> str:
    message = str(exc).strip()
    if "DeploymentNotFound" in message:
        return (
            "DeploymentNotFound — create this deployment in Azure AI Foundry "
            "and set the exact deployment name in backend/.env"
        )
    return message or exc.__class__.__name__


def probe_azure_service(kind: AzureServiceKind) -> dict[str, Any]:
    """Smoke-test one Azure deployment; used by GET /debug/azure."""
    try:
        if kind == "text":
            base_url = get_text_base_url()
            deployment = get_text_deployment()
            client = get_text_client()
            response = client.chat.completions.create(
                model=deployment,
                messages=[{"role": "user", "content": "Reply with exactly: azure-ok"}],
                temperature=0,
                max_tokens=10,
            )
            output = (response.choices[0].message.content or "").strip()
            return {
                "ok": True,
                "kind": kind,
                "base_url": base_url,
                "deployment": deployment,
                "sample": output,
            }

        if kind == "tts":
            base_url = get_tts_base_url()
            deployment = get_tts_deployment()
            client = get_tts_client()
            response = client.audio.speech.create(
                model=deployment,
                voice="alloy",
                input="Azure TTS test.",
                response_format="mp3",
            )
            audio_bytes = response.read()
            return {
                "ok": True,
                "kind": kind,
                "base_url": base_url,
                "deployment": deployment,
                "bytes": len(audio_bytes),
            }

        base_url = get_images_base_url()
        deployment = get_image_deployment()
        client = get_images_client()
        response = client.images.generate(
            model=deployment,
            prompt="A simple blue circle on a white background",
            size="1024x1024",
            n=1,
        )
        item = response.data[0]
        payload_bytes = len(item.b64_json or "") if getattr(item, "b64_json", None) else 0
        return {
            "ok": True,
            "kind": kind,
            "base_url": base_url,
            "deployment": deployment,
            "bytes": payload_bytes,
        }
    except ValueError as exc:
        return {
            "ok": False,
            "kind": kind,
            "error": str(exc),
        }
    except Exception as exc:
        base_url = ""
        deployment = ""
        try:
            if kind == "text":
                base_url = get_text_base_url()
                deployment = get_text_deployment()
            elif kind == "tts":
                base_url = get_tts_base_url()
                deployment = get_tts_deployment()
            else:
                base_url = get_images_base_url()
                deployment = get_image_deployment()
        except ValueError:
            pass
        return {
            "ok": False,
            "kind": kind,
            "base_url": base_url,
            "deployment": deployment,
            "error": _error_summary(exc),
        }


def probe_all_azure_services() -> dict[str, Any]:
    results = {
        "text": probe_azure_service("text"),
        "tts": probe_azure_service("tts"),
        "image": probe_azure_service("image"),
    }
    return {
        "ok": all(item["ok"] for item in results.values()),
        "services": results,
    }
