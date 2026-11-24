"""API tests for /api/predict, validation, rate limiting and health."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest

from tests.helpers import NotReadyModelService, upload_payload


def post_image(client, data, filename="photo.jpg", mime="image/jpeg"):
    return client.post(
        "/api/predict",
        data=upload_payload(data, filename=filename, mime=mime),
        content_type="multipart/form-data",
        headers={"Accept": "application/json"},
    )


def test_predict_happy_path_saves_history_and_gradcam(app, client, jpeg_bytes):
    response = post_image(client, jpeg_bytes)
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["prediction"]["category"] == "plastic"
    assert payload["prediction"]["display_name"] == "Plastic"
    assert payload["prediction"]["confidence"] == pytest.approx(0.93)
    assert payload["prediction"]["confidence_label"] == "High confidence"
    assert payload["prediction"]["uncertain"] is False
    assert len(payload["top_predictions"]) == 3
    assert payload["recommendation"]["category"] == "plastic"
    assert payload["recommendation"]["bin_colour"]
    assert payload["gradcam"].startswith("data:image/png;base64,")

    fake = app.extensions["fake_model"]
    assert fake.predict_calls == 1
    assert fake.explain_calls == 1

    uploaded = list(Path(app.config["UPLOAD_FOLDER"]).glob("*.jpg"))
    assert len(uploaded) == 1
    assert "photo" not in uploaded[0].name.lower()

    with app.app_context():
        from database import db

        rows = db.list_predictions()
        assert len(rows) == 1
        assert rows[0]["predicted_class"] == "plastic"
        assert rows[0]["image_path"].startswith("uploads/")
        assert "photo" not in rows[0]["image_path"]


def test_low_confidence_returns_no_advice_but_saves_attempt(app, client, jpeg_bytes):
    app.extensions["fake_model"].confidence = 0.48
    response = post_image(client, jpeg_bytes)
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["prediction"]["uncertain"] is True
    assert payload["prediction"]["confidence_label"] == "Low confidence"
    assert payload["recommendation"] is None
    assert payload["gradcam"] is None
    assert payload["tips"]
    assert "below 60% confidence" in payload["message"]
    assert app.extensions["fake_model"].explain_calls == 0

    with app.app_context():
        from database import db

        assert db.get_stats()["total"] == 1


def test_medium_confidence_is_still_confident_enough(app, client, jpeg_bytes):
    app.extensions["fake_model"].confidence = 0.61
    response = post_image(client, jpeg_bytes)
    payload = response.get_json()
    assert payload["prediction"]["uncertain"] is False
    assert payload["prediction"]["confidence_label"] == "Medium confidence"
    assert payload["recommendation"] is not None


def test_predict_accepts_camera_base64_frame(app, client, jpeg_bytes):
    encoded = base64.b64encode(jpeg_bytes).decode("ascii")
    response = client.post(
        "/api/predict",
        json={"image": "data:image/jpeg;base64," + encoded},
        headers={"Accept": "application/json"},
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["prediction"]["category"] == "plastic"


def test_predict_rejects_bad_base64(client):
    response = client.post(
        "/api/predict",
        json={"image": "data:image/jpeg;base64,the=quick=brown=fox"},
        headers={"Accept": "application/json"},
    )
    assert response.status_code == 400
    assert response.get_json()["ok"] is False


def test_predict_rejects_unsupported_extension(client, jpeg_bytes):
    response = post_image(client, jpeg_bytes, filename="photo.gif", mime="image/jpeg")
    assert response.status_code == 400
    assert "file type" in response.get_json()["message"].lower()


def test_predict_rejects_unsupported_mime(client, jpeg_bytes):
    response = post_image(client, jpeg_bytes, filename="photo.jpg", mime="text/plain")
    assert response.status_code == 400
    assert "image" in response.get_json()["message"].lower()


def test_predict_rejects_fake_image_content(client):
    response = post_image(client, b"<svg>not really an image</svg>")
    assert response.status_code == 400
    assert response.get_json()["ok"] is False


def test_predict_rejects_oversized_image(client):
    response = post_image(client, b"x" * 5001)
    assert response.status_code == 413
    assert "larger than" in response.get_json()["message"].lower()


def test_predict_requires_an_image(client):
    response = client.post("/api/predict", data={}, content_type="multipart/form-data")
    assert response.status_code == 400
    assert response.get_json()["ok"] is False


def test_predict_returns_503_when_model_is_not_loaded(app, client, jpeg_bytes):
    app.extensions["model_service"] = NotReadyModelService()
    response = post_image(client, jpeg_bytes)
    assert response.status_code == 503
    payload = response.get_json()
    assert payload["ok"] is False
    assert "model" in payload["message"].lower()


def test_rate_limit_returns_429(app, client, jpeg_bytes):
    limiter = app.extensions["rate_limiter"]
    limiter.limit = 1
    limiter.window = 60
    first = post_image(client, jpeg_bytes)
    assert first.status_code == 200
    second = post_image(client, jpeg_bytes)
    assert second.status_code == 429
    assert "Retry-After" in second.headers
    assert second.get_json()["ok"] is False


def test_health_reports_ok_and_degraded(app, client):
    good = client.get("/health")
    assert good.status_code == 200
    assert good.get_json()["status"] == "ok"
    assert good.get_json()["model_loaded"] is True

    app.extensions["model_service"] = NotReadyModelService()
    degraded = client.get("/health")
    assert degraded.status_code == 503
    assert degraded.get_json()["status"] == "degraded"
    assert degraded.get_json()["model_loaded"] is False


def test_api_unknown_endpoint_is_json_404(client):
    response = client.get("/api/no-such-endpoint")
    assert response.status_code == 404
    assert response.is_json
    assert response.get_json()["ok"] is False


def test_security_headers_are_present(client):
    response = client.get("/")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert "default-src 'self'" in response.headers["Content-Security-Policy"]
    assert "camera=(self)" in response.headers["Permissions-Policy"]
