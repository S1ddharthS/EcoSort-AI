"""Small shared test helpers (kept out of conftest for direct importing)."""

from __future__ import annotations

import io


class NotReadyModelService:
    """A faithful stand-in for a missing or unloadable production model."""

    loaded = False
    demo = True
    load_error = "The AI model file is missing for this test."


def upload_payload(data: bytes, filename: str = "photo.jpg", mime: str = "image/jpeg"):
    return {"image": (io.BytesIO(data), filename, mime)}
