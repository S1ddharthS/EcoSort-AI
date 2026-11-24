"""Shared pytest fixtures for EcoSort AI.

The test app uses a temporary SQLite database, a temporary uploads folder and
a deterministic fake model. That keeps tests fast and lets us exercise both
the confident and "not sure" branches without TensorFlow.
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

# Keep the module-level Flask app created by app.py away from the real model
# and the developer database while tests are being collected.
os.environ.setdefault("ECOSORT_MODEL", str(Path(__file__).parent / "__no_model__.keras"))
os.environ.setdefault("ECOSORT_DB", str(Path(__file__).parent / "__pytest_bootstrap.db"))

from app import create_app  # noqa: E402
from database.db import CLASS_DISPLAY, CLASS_NAMES  # noqa: E402


class FakeModelService:
    """Minimal stand-in for ml.predict.ModelService used by API tests."""

    loaded = True
    demo = False
    load_error = None

    def __init__(self):
        self.confidence = 0.93
        self.category = "plastic"
        self.explain_calls = 0
        self.predict_calls = 0

    def _top_predictions(self) -> list[dict]:
        other = [name for name in CLASS_NAMES if name != self.category]
        confidences = {
            self.category: self.confidence,
            other[0]: max(0.01, (1.0 - self.confidence) * 0.65),
            other[1]: max(0.005, (1.0 - self.confidence) * 0.35),
        }
        return [
            {
                "category": name,
                "display_name": CLASS_DISPLAY[name],
                "confidence": round(confidences[name], 4),
            }
            for name in [self.category, other[0], other[1]]
        ]

    def predict(self, source, top_k: int = 3) -> dict:
        self.predict_calls += 1
        top = self._top_predictions()[:top_k]
        best = top[0]
        return {
            "category": best["category"],
            "display_name": best["display_name"],
            "confidence": best["confidence"],
            "top_predictions": top,
            "all_probabilities": {name: 0.01 for name in CLASS_NAMES},
            "image_array": np.zeros((224, 224, 3), dtype=np.float32),
        }

    def explain(self, image_array, category: str) -> str:
        self.explain_calls += 1
        # A real Grad-CAM call returns a data URL; this tiny PNG is enough for
        # asserting the UI payload without doing computer vision in unit tests.
        return (
            "data:image/png;base64,"
            + base64.b64encode(
                b"\x89PNG\r\n\x1a\n"
                b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
                b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
                b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
                b"\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
            ).decode("ascii")
        )


class NotReadyModelService(FakeModelService):
    loaded = False
    demo = True
    load_error = "The AI model file is missing for this test."


@pytest.fixture()
def app(tmp_path):
    app = create_app(
        {
            "TESTING": True,
            "DATABASE": str(tmp_path / "test.db"),
            "UPLOAD_FOLDER": str(tmp_path / "uploads"),
            "MODEL_PATH": str(tmp_path / "missing_model.keras"),
            "CLASS_NAMES_PATH": str(tmp_path / "missing_model" / "class_names.json"),
            "METRICS_PATH": str(tmp_path / "missing_metrics.json"),
            "MODEL_META_PATH": str(tmp_path / "missing_meta.json"),
            "MAX_IMAGE_BYTES": 5000,
            "MAX_IMAGE_PIXELS": 1_000_000,
            "RATE_LIMIT_REQUESTS": 1000,
            "GRADCAM_ENABLED": True,
            "SECRET_KEY": "test-key",
        }
    )
    fake = FakeModelService()
    app.extensions["model_service"] = fake
    app.extensions["fake_model"] = fake
    yield app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def runner(app):
    return app.test_cli_runner()


@pytest.fixture()
def jpeg_bytes():
    """A small, valid, real JPEG with visible shapes (not a stub)."""
    image = Image.new("RGB", (96, 96), (226, 239, 228))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((18, 12, 78, 84), radius=14, fill=(44, 119, 173))
    draw.ellipse((34, 30, 62, 58), fill=(238, 246, 248))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=82)
    return buffer.getvalue()


@pytest.fixture()
def png_bytes():
    image = Image.new("RGBA", (64, 64), (20, 90, 60, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture()
def webp_bytes():
    image = Image.new("RGB", (64, 64), (210, 170, 70))
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP", quality=80)
    return buffer.getvalue()


def upload_payload(data: bytes, filename: str = "photo.jpg", mime: str = "image/jpeg"):
    return {"image": (io.BytesIO(data), filename, mime)}
