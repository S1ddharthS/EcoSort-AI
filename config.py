"""Application configuration for EcoSort AI.

All paths are absolute so the app behaves the same whether it is started
from the project root, from an IDE, or inside Docker.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


class Config:
    """Default configuration. Every value can be overridden with an env var."""

    # Flask
    SECRET_KEY = os.environ.get("SECRET_KEY", "ecosort-dev-key-change-in-production")
    MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # request cap; decoded image is capped at 5 MB

    # Storage
    DATABASE = os.environ.get("ECOSORT_DB", str(BASE_DIR / "database" / "ecosort.db"))
    UPLOAD_FOLDER = os.environ.get(
        "ECOSORT_UPLOADS", str(BASE_DIR / "static" / "uploads")
    )

    # Model artefacts
    MODEL_DIR = BASE_DIR / "ml" / "model"
    MODEL_PATH = os.environ.get(
        "ECOSORT_MODEL", str(MODEL_DIR / "ecosort_model.keras")
    )
    CLASS_NAMES_PATH = str(MODEL_DIR / "class_names.json")
    METRICS_PATH = str(MODEL_DIR / "metrics.json")
    MODEL_META_PATH = str(MODEL_DIR / "model_meta.json")

    # Upload validation
    MAX_IMAGE_BYTES = 5 * 1024 * 1024
    MAX_IMAGE_PIXELS = 25_000_000
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
    ALLOWED_MIMES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}

    # Prediction
    UNCERTAIN_THRESHOLD = 0.60
    GRADCAM_ENABLED = True

    # Simple in-memory rate limiting (per client IP)
    RATE_LIMIT_REQUESTS = 20
    RATE_LIMIT_WINDOW = 60  # seconds

    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    TESTING = False


class TestConfig(Config):
    """Used by pytest: isolated database and uploads, no rate limiting."""

    TESTING = True
    RATE_LIMIT_REQUESTS = 1000
