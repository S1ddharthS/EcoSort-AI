"""Model loading and inference for EcoSort AI.

The Keras model is loaded exactly once when the Flask app starts. Inference
is serialised with a lock because a single Keras model is not guaranteed to
be thread-safe.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import numpy as np

from database.db import CLASS_DISPLAY, CLASS_NAMES
from ml.preprocess import preprocess_image

logger = logging.getLogger(__name__)

TIP_LIST = [
    "Use bright, even light and avoid harsh shadows.",
    "Photograph one item only, without a cluttered background.",
    "Move closer so the item fills most of the frame.",
    "Avoid blurry photos and heavy filters.",
]


def confidence_label(confidence: float) -> str:
    """Plain-language confidence bands used by the UI."""
    if confidence >= 0.80:
        return "High confidence"
    if confidence >= 0.60:
        return "Medium confidence"
    return "Low confidence"


class ModelService:
    """Holds the trained model and class list for the lifetime of the app."""

    def __init__(self, model_path: str, class_names_path: str):
        self.model_path = Path(model_path)
        self.class_names_path = Path(class_names_path)
        self.model = None
        self.class_names: list[str] = []
        self.load_error: str | None = None
        self.demo = True
        self.meta: dict = {}
        self.grad_model = None
        self.gradcam_error: str | None = None
        self._lock = threading.Lock()

    def load(self) -> None:
        """Load the model and class names, recording a friendly error on failure."""
        if not self.model_path.exists():
            self.load_error = (
                "The AI model file is missing. Add ml/model/ecosort_model.keras "
                "and restart the app. See README.md for the training steps."
            )
            logger.warning("Model file not found: %s", self.model_path)
            return
        if not self.class_names_path.exists():
            self.load_error = (
                "The class names file is missing. Add ml/model/class_names.json "
                "and restart the app."
            )
            logger.warning("Class names file not found: %s", self.class_names_path)
            return

        try:
            self.class_names = json.loads(
                self.class_names_path.read_text(encoding="utf-8")
            )
            if sorted(self.class_names) != sorted(CLASS_NAMES):
                raise ValueError(
                    "class_names.json does not match the six EcoSort classes."
                )

            # Importing TensorFlow is slow, so it happens only when the app boots.
            import tensorflow as tf

            self.model = tf.keras.models.load_model(
                self.model_path, compile=False, safe_mode=True
            )
            output_classes = int(self.model.output_shape[-1])
            if output_classes != len(self.class_names):
                raise ValueError(
                    "Model output size does not match class_names.json."
                )
            self.meta = self._load_meta()
            self.demo = bool(self.meta.get("demo", True))
            if self.demo:
                logger.warning("DEMO model loaded - metrics are not a real evaluation.")
            else:
                logger.info(
                    "Model loaded successfully (%s classes, demo=%s).",
                    len(self.class_names),
                    self.demo,
                )
            self._prepare_gradcam()
            self.load_error = None
        except Exception as exc:  # noqa: BLE001 - startup must never crash the site
            self.model = None
            self.load_error = (
                "The AI model couldn't be loaded. The file may be incomplete or "
                "from an incompatible TensorFlow version. Check the server log."
            )
            logger.exception("Failed to load model: %s", exc)

    def _load_meta(self) -> dict:
        meta_path = self.model_path.parent / "model_meta.json"
        if not meta_path.exists():
            return {"demo": True}
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"demo": True}

    def _prepare_gradcam(self) -> None:
        """Build the explanation graph once, without blocking predictions on it."""
        try:
            from ml.gradcam import build_grad_model

            self.grad_model = build_grad_model(
                self.model, self.meta.get("gradcam_layer")
            )
            logger.info("Grad-CAM graph ready.")
        except Exception as exc:  # noqa: BLE001
            self.grad_model = None
            self.gradcam_error = str(exc)
            logger.warning("Grad-CAM is unavailable: %s", exc)

    def explain(self, image_array, category: str) -> str | None:
        """Return a Grad-CAM overlay for the predicted category, if available."""
        if self.grad_model is None or category not in self.class_names:
            return None
        from ml.gradcam import generate_gradcam

        with self._lock:
            return generate_gradcam(
                self.grad_model, image_array, self.class_names.index(category)
            )

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def predict(self, source, top_k: int = 3) -> dict:
        """Classify an image and return the top-k probabilities."""
        if not self.loaded:
            raise RuntimeError(self.load_error or "The AI model is not available.")

        image = preprocess_image(source)
        batch = np.expand_dims(image, axis=0)
        with self._lock:
            probabilities = self.model.predict(batch, verbose=0)[0]

        order = np.argsort(probabilities)[::-1][:top_k]
        top_predictions = [
            {
                "category": self.class_names[int(index)],
                "display_name": CLASS_DISPLAY.get(
                    self.class_names[int(index)], self.class_names[int(index)].title()
                ),
                "confidence": round(float(probabilities[int(index)]), 4),
            }
            for index in order
        ]
        best = top_predictions[0]
        return {
            "category": best["category"],
            "display_name": best["display_name"],
            "confidence": best["confidence"],
            "top_predictions": top_predictions,
            "all_probabilities": {
                name: float(probabilities[i])
                for i, name in enumerate(self.class_names)
            },
            "image_array": image,
        }
