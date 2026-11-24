"""EcoSort AI - Flask application.

Run locally with:  python app.py
Production:        gunicorn -c gunicorn.conf.py app:app

Routes
------
Pages: /, /classify, /history, /learn, /about, and /health
API:   POST /api/predict, GET /api/history, DELETE /api/history/<id>,
       DELETE /api/history, GET /api/stats
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import logging.handlers
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

from flask import (
    Blueprint,
    Flask,
    current_app,
    jsonify,
    render_template,
    request,
    send_from_directory,
    url_for,
)

import database.db as db
from config import Config
from ml.predict import TIP_LIST, ModelService, confidence_label
from ml.preprocess import (
    ImageValidationError,
    inspect_image_bytes,
    make_preview_jpeg,
)

logger = logging.getLogger("ecosort")

bp = Blueprint("main", __name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class ApiError(Exception):
    """An error with a user-facing message and an HTTP status code."""

    def __init__(self, message: str, status: int = 400, hint: str | None = None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.hint = hint


class RateLimiter:
    """Small in-memory sliding-window limiter (single process only)."""

    def __init__(self, limit: int, window_seconds: int):
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = Lock()

    def allow(self, key: str) -> tuple[bool, int]:
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and now - hits[0] > self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                retry_after = max(1, int(self.window - (now - hits[0])))
                return False, retry_after
            hits.append(now)
            return True, 0


def configure_logging(app: Flask) -> None:
    """Log to stdout (for Docker/Render) and to a rotating local file."""
    level = getattr(logging, app.config.get("LOG_LEVEL", "INFO").upper(), logging.INFO)
    app.logger.setLevel(level)
    logger.setLevel(level)
    if app.config.get("TESTING"):
        return

    logs_dir = Path(app.root_path) / "logs"
    logs_dir.mkdir(exist_ok=True)
    if not any(isinstance(h, logging.handlers.RotatingFileHandler) for h in logger.handlers):
        file_handler = logging.handlers.RotatingFileHandler(
            logs_dir / "ecosort.log", maxBytes=1_000_000, backupCount=3
        )
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(file_handler)

    if not any(
        isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.handlers.RotatingFileHandler)
        for h in logger.handlers
    ):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
        logger.addHandler(stream_handler)


def get_model() -> ModelService:
    return current_app.extensions["model_service"]


def _relative_image_path(filename: str) -> str:
    return f"uploads/{filename}"


def _upload_folder() -> Path:
    """Absolute upload directory, which may live outside static/ in production."""
    return Path(current_app.config["UPLOAD_FOLDER"]).resolve()


def _safe_upload_name(image_path: str) -> str | None:
    """Return a generated filename, rejecting absolute paths and traversal."""
    if not image_path:
        return None
    relative = Path(image_path)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    name = relative.name
    return name or None


def _image_url(image_path: str) -> str:
    name = _safe_upload_name(image_path)
    if not name:
        return ""
    return url_for("main.uploaded_file", filename=name)


def _delete_image_file(image_path: str) -> None:
    """Safely remove a preview from the configured upload folder."""
    name = _safe_upload_name(image_path)
    if not name:
        return
    upload_dir = _upload_folder()
    target = (upload_dir / name).resolve()
    try:
        target.relative_to(upload_dir)
    except ValueError:
        return
    if target.is_file():
        try:
            target.unlink()
        except OSError:
            logger.warning("Could not delete upload %s", image_path)


def _read_image_request() -> tuple[bytes, str]:
    """Read and validate an uploaded file or base64 webcam frame.

    Returns (bytes, original_name_or_label). User filenames are never used for
    storage; they are only reported back in logs.
    """
    config = current_app.config

    if "image" in request.files:
        upload = request.files["image"]
        original_name = Path(upload.filename or "").name
        if not upload or not original_name:
            raise ApiError(
                "No image was received. Please choose a photo and try again."
            )
        extension = Path(original_name).suffix.lower().lstrip(".")
        if extension not in config["ALLOWED_EXTENSIONS"]:
            raise ApiError(
                "That file type isn't supported. Please upload a JPG, PNG or WEBP image.",
                hint="If your photo is an HEIC file, export it as JPG first.",
            )
        mime = (upload.mimetype or "").lower()
        if mime not in config["ALLOWED_MIMES"]:
            raise ApiError(
                "That file doesn't look like an image. Please upload a JPG, "
                "PNG or WEBP image."
            )
        data = upload.read(config["MAX_IMAGE_BYTES"] + 1)
        if len(data) > config["MAX_IMAGE_BYTES"]:
            raise ApiError(
                "That image is larger than 5 MB. Please choose a smaller file.",
                status=413,
            )
    elif request.is_json:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            raise ApiError("We couldn't read the request. Please try again.")
        image_data = payload.get("image")
        if not image_data or not isinstance(image_data, str):
            raise ApiError(
                "No camera image was received. Please capture a photo and try again."
            )
        mime = None
        encoded = image_data
        if image_data.startswith("data:"):
            header, separator, encoded = image_data.partition(",")
            if not separator or not header.startswith("data:image/"):
                raise ApiError("That camera image couldn't be read. Please try again.")
            mime = header[5:].split(";")[0].lower()
            if mime not in config["ALLOWED_MIMES"]:
                raise ApiError(
                    "That camera image format isn't supported. Please use JPG or PNG."
                )
        if len(encoded) > (config["MAX_IMAGE_BYTES"] * 4 // 3) + 16:
            raise ApiError(
                "That camera image is larger than 5 MB. Please try again.",
                status=413,
            )
        try:
            data = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            raise ApiError(
                "That camera image couldn't be decoded. Please retake the photo."
            ) from None
        if len(data) > config["MAX_IMAGE_BYTES"]:
            raise ApiError(
                "That camera image is larger than 5 MB. Please try again.",
                status=413,
            )
    else:
        raise ApiError(
            "No image was received. Send a multipart image or a base64 frame."
        )

    try:
        inspect_image_bytes(
            data,
            max_bytes=config["MAX_IMAGE_BYTES"],
            max_pixels=config["MAX_IMAGE_PIXELS"],
        )
    except ImageValidationError as exc:
        raise ApiError(str(exc)) from None
    return data, "webcam" if mime and "image" in mime else "upload"


def _check_model_ready() -> ModelService:
    service = get_model()
    if not service.loaded:
        raise ApiError(
            service.load_error
            or "The AI model isn't available right now. Please try again later.",
            status=503,
            hint="See README.md for how to train and install the model.",
        )
    return service


def _serialise_history_item(row: dict) -> dict:
    confidence = float(row["confidence"])
    image_path = row["image_path"]
    image_name = _safe_upload_name(image_path)
    image_file = _upload_folder() / image_name if image_name else None
    return {
        "id": row["id"],
        "category": row["predicted_class"],
        "display_name": db.CLASS_DISPLAY.get(
            row["predicted_class"], row["predicted_class"].title()
        ),
        "confidence": round(confidence, 4),
        "confidence_percent": round(confidence * 100),
        "confidence_label": confidence_label(confidence),
        "uncertain": confidence < current_app.config["UNCERTAIN_THRESHOLD"],
        "image_url": _image_url(image_path),
        "image_available": bool(image_file and image_file.is_file()),
        "created_at": row["created_at"],
    }


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@bp.get("/")
def home():
    stats = db.get_stats()
    categories = db.list_recommendations()
    return render_template("index.html", stats=stats, categories=categories)


@bp.get("/uploads/<filename>")
def uploaded_file(filename: str):
    """Serve saved preview images from the configured upload folder.

    Generated filenames are hashed UUIDs, and send_from_directory rejects path
    traversal. Keeping this separate from /static means ECOSORT_UPLOADS can
    point at a Docker/Render volume outside the source tree.
    """
    return send_from_directory(_upload_folder(), filename, max_age=3600)


@bp.get("/classify")
def classify_page():
    return render_template("classify.html", tips=TIP_LIST)


@bp.get("/history")
def history_page():
    categories = db.list_recommendations()
    return render_template("history.html", categories=categories)


@bp.get("/learn")
def learn_page():
    categories = db.list_recommendations()
    return render_template("learn.html", categories=categories)


@bp.get("/about")
def about_page():
    config = current_app.config
    metrics = None
    metrics_path = Path(config["METRICS_PATH"])
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("metrics.json exists but could not be parsed")
    meta = {}
    meta_path = Path(config["MODEL_META_PATH"])
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}

    image_paths = {
        "confusion_matrix": Path(config["MODEL_DIR"]).parent.parent
        / "static"
        / "img"
        / "confusion_matrix.png",
        "training_curves": Path(config["MODEL_DIR"]).parent.parent
        / "static"
        / "img"
        / "training_curves.png",
    }
    images = {
        "confusion_matrix": image_paths["confusion_matrix"].is_file(),
        "training_curves": image_paths["training_curves"].is_file(),
    }
    return render_template(
        "about.html", metrics=metrics, meta=meta, images=images
    )


@bp.get("/health")
def health():
    database_status = "ok"
    try:
        db.get_db().execute("SELECT 1").fetchone()
    except Exception:  # noqa: BLE001
        database_status = "error"
    service = get_model()
    status = "ok" if service.loaded and database_status == "ok" else "degraded"
    payload = {
        "status": status,
        "database": database_status,
        "model_loaded": service.loaded,
        "model_error": service.load_error,
        "demo_model": service.demo,
    }
    return jsonify(payload), 200 if status == "ok" else 503


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@bp.post("/api/predict")
def predict():
    limiter: RateLimiter = current_app.extensions["rate_limiter"]
    allowed, retry_after = limiter.allow(request.remote_addr or "unknown")
    if not allowed:
        response = jsonify(
            {
                "ok": False,
                "error": "Too many requests",
                "message": "You're going a little fast. Please wait a moment and try again.",
            }
        )
        response.status_code = 429
        response.headers["Retry-After"] = str(retry_after)
        return response

    service = _check_model_ready()
    try:
        image_bytes, source_kind = _read_image_request()
        result = service.predict(image_bytes)
    except ApiError:
        raise
    except ImageValidationError as exc:
        raise ApiError(str(exc)) from None
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed: %s", exc)
        raise ApiError(
            "Something went wrong while analysing the image.",
            status=500,
            hint="Please try again. If it keeps happening, check the server log.",
        ) from None

    config = current_app.config
    confidence = float(result["confidence"])
    uncertain = confidence < config["UNCERTAIN_THRESHOLD"]
    recommendation = None
    if not uncertain:
        recommendation = db.get_recommendation(result["category"])
        if recommendation is None:
            logger.error("Missing recommendation for %s", result["category"])

    # Store a clean, resized JPEG with a random name. The original file is
    # never written to disk, so user-supplied names and metadata are discarded.
    filename = f"{uuid.uuid4().hex}.jpg"
    image_path = _relative_image_path(filename)
    upload_dir = Path(config["UPLOAD_FOLDER"])
    upload_dir.mkdir(parents=True, exist_ok=True)
    try:
        preview = make_preview_jpeg(image_bytes)
        (upload_dir / filename).write_bytes(preview)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Could not save preview: %s", exc)
        raise ApiError(
            "We couldn't save the image for your history.",
            status=500,
            hint="Please try a different image.",
        ) from None

    try:
        saved = db.add_prediction(
            image_path, result["category"], confidence
        )
    except Exception as exc:  # noqa: BLE001
        _delete_image_file(image_path)
        logger.exception("Database insert failed: %s", exc)
        raise ApiError(
            "The result couldn't be saved to your history.",
            status=500,
            hint="Please try again.",
        ) from None

    gradcam = None
    if config["GRADCAM_ENABLED"] and not uncertain:
        try:
            gradcam = service.explain(result["image_array"], result["category"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Grad-CAM unavailable: %s", exc)

    logger.info(
        "Prediction id=%s class=%s confidence=%.3f uncertain=%s source=%s",
        saved["id"],
        result["category"],
        confidence,
        uncertain,
        source_kind,
    )

    payload = {
        "ok": True,
        "prediction": {
            "id": saved["id"],
            "category": result["category"],
            "display_name": result["display_name"],
            "confidence": round(confidence, 4),
            "confidence_percent": round(confidence * 100),
            "confidence_label": confidence_label(confidence),
            "uncertain": uncertain,
            "image_url": _image_url(image_path),
            "created_at": saved["created_at"],
        },
        "top_predictions": result["top_predictions"],
        "recommendation": recommendation,
        "gradcam": gradcam,
        "tips": TIP_LIST if uncertain else [],
        "message": (
            "The model's best match was below "
            f"{round(config['UNCERTAIN_THRESHOLD'] * 100)}% confidence, so no "
            "recycling advice is shown. Guessing could put the wrong item in "
            "the wrong bin."
            if uncertain
            else None
        ),
    }
    return jsonify(payload)


@bp.get("/api/history")
def api_history():
    category = (request.args.get("category") or "").strip().lower()
    if category and category not in db.CLASS_NAMES:
        raise ApiError(
            "That category isn't recognised. Choose one of the six EcoSort categories."
        )
    rows = db.list_predictions(category or None)
    items = [_serialise_history_item(row) for row in rows]
    return jsonify({"ok": True, "items": items, "count": len(items)})


@bp.delete("/api/history/<int:prediction_id>")
def api_delete_history(prediction_id: int):
    row = db.delete_prediction(prediction_id)
    if row is None:
        raise ApiError(
            "That item has already been deleted or doesn't exist.", status=404
        )
    _delete_image_file(row["image_path"])
    return jsonify({"ok": True, "message": "Item deleted."})


@bp.delete("/api/history")
def api_clear_history():
    image_paths = db.clear_predictions()
    for path in image_paths:
        _delete_image_file(path)
    return jsonify(
        {
            "ok": True,
            "deleted": len(image_paths),
            "message": "History cleared.",
        }
    )


@bp.get("/api/stats")
def api_stats():
    return jsonify({"ok": True, **db.get_stats()})


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)
    if test_config:
        app.config.update(test_config)

    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)
    configure_logging(app)

    db.init_app(app)
    db.init_db(app)

    model_service = ModelService(
        app.config["MODEL_PATH"], app.config["CLASS_NAMES_PATH"]
    )
    model_service.load()
    app.extensions["model_service"] = model_service
    app.extensions["rate_limiter"] = RateLimiter(
        app.config["RATE_LIMIT_REQUESTS"], app.config["RATE_LIMIT_WINDOW"]
    )

    app.register_blueprint(bp)

    @app.context_processor
    def inject_globals():
        service = get_model()
        return {
            "model_ready": service.loaded,
            "model_error": service.load_error,
            "demo_model": bool(service.demo and service.loaded),
            "current_year": datetime.now(timezone.utc).year,
        }

    @app.template_filter("pretty_class")
    def pretty_class(value: str) -> str:
        return db.CLASS_DISPLAY.get(value, str(value).title())

    @app.template_filter("percent")
    def percent(value: float) -> str:
        return f"{float(value) * 100:.1f}%"

    @app.template_filter("format_date")
    def format_date(value: str) -> str:
        """Format an ISO timestamp as a short readable date."""
        if not value:
            return "unknown"
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            return parsed.strftime("%d %b %Y")
        except ValueError:
            return str(value)[:10]

    @app.after_request
    def add_security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault(
            "Permissions-Policy", "camera=(self), microphone=(), geolocation=()"
        )
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; "
            "img-src 'self' data: blob:; "
            "media-src 'self' blob:; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "object-src 'none'; base-uri 'self'; form-action 'self'; "
            "frame-ancestors 'none'",
        )
        if request.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.errorhandler(ApiError)
    def handle_api_error(error: ApiError):
        payload = {"ok": False, "error": error.message, "message": error.message}
        if error.hint:
            payload["hint"] = error.hint
        return jsonify(payload), error.status

    @app.errorhandler(404)
    def not_found(_error):
        if request.path.startswith("/api/"):
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": "Not found",
                        "message": "That API endpoint doesn't exist.",
                    }
                ),
                404,
            )
        return render_template("404.html"), 404

    @app.errorhandler(413)
    def too_large(_error):
        message = "That image is too large. Please use a file under 5 MB."
        if request.path.startswith("/api/"):
            return jsonify({"ok": False, "error": message, "message": message}), 413
        return render_template("404.html", error_title="That file is too big",
                               error_message=message), 413

    @app.errorhandler(500)
    def server_error(error):
        logger.exception("Unhandled server error: %s", error)
        if request.path.startswith("/api/"):
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": "Server error",
                        "message": "Something went wrong. Please try again.",
                    }
                ),
                500,
            )
        return render_template("500.html"), 500

    return app


app = create_app()


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
