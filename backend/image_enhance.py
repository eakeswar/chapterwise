"""Image enhancement and decorative image generation."""

from __future__ import annotations

from azure_clients import get_image_deployment, get_images_client


def generate_decorative_image(prompt: str, *, size: str = "1024x1024") -> str:
    """Generate a decorative image and return base64 PNG data."""
    cleaned = (prompt or "").strip()
    if not cleaned:
        raise ValueError("Prompt is required for image generation.")

    client = get_images_client()
    response = client.images.generate(
        model=get_image_deployment(),
        prompt=cleaned,
        size=size,
        n=1,
    )
    item = response.data[0]
    if not getattr(item, "b64_json", None):
        raise ValueError("Image model did not return base64 data.")
    return item.b64_json
