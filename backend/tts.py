"""Azure OpenAI text-to-speech for topic narration."""

from __future__ import annotations

from azure_clients import get_tts_client, get_tts_deployment

DEFAULT_VOICE = "alloy"
MAX_TTS_CHARS = 4096


def synthesize_speech(text: str, *, voice: str = DEFAULT_VOICE) -> bytes:
    """Return MP3 bytes for the given text."""
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("Text is required for TTS.")
    if len(cleaned) > MAX_TTS_CHARS:
        raise ValueError(f"Text exceeds {MAX_TTS_CHARS} characters.")

    client = get_tts_client()
    response = client.audio.speech.create(
        model=get_tts_deployment(),
        voice=voice,
        input=cleaned,
        response_format="mp3",
    )
    return response.read()
