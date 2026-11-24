"""History, delete, clear and statistics endpoint tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from database import db


def add_with_file(app, filename: str, category: str, confidence: float) -> dict:
    upload_folder = Path(app.config["UPLOAD_FOLDER"])
    upload_folder.mkdir(parents=True, exist_ok=True)
    (upload_folder / filename).write_bytes(b"\xff\xd8\xff\xe0test-jpeg")
    with app.app_context():
        return db.add_prediction(f"uploads/{filename}", category, confidence)


def test_history_lists_newest_first_with_display_fields(app, client):
    add_with_file(app, "one.jpg", "plastic", 0.91)
    add_with_file(app, "two.jpg", "glass", 0.72)
    add_with_file(app, "three.jpg", "paper", 0.81)

    response = client.get("/api/history")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["count"] == 3
    assert [item["category"] for item in payload["items"]] == [
        "paper",
        "glass",
        "plastic",
    ]
    first = payload["items"][0]
    assert first["display_name"] == "Paper"
    assert first["image_available"] is True
    assert first["image_url"].startswith("/uploads/")
    assert first["confidence_percent"] == 81
    assert first["confidence_label"] == "High confidence"
    assert first["uncertain"] is False


def test_history_filters_by_category(app, client):
    add_with_file(app, "one.jpg", "plastic", 0.9)
    add_with_file(app, "two.jpg", "paper", 0.8)
    add_with_file(app, "three.jpg", "plastic", 0.7)

    response = client.get("/api/history?category=plastic")
    payload = response.get_json()
    assert payload["count"] == 2
    assert {item["category"] for item in payload["items"]} == {"plastic"}


def test_history_rejects_unknown_category(client):
    response = client.get("/api/history?category=battery")
    assert response.status_code == 400
    assert response.get_json()["ok"] is False


def test_history_marks_uncertain_items(app, client):
    add_with_file(app, "low.jpg", "organic", 0.42)
    response = client.get("/api/history")
    item = response.get_json()["items"][0]
    assert item["uncertain"] is True
    assert item["confidence_label"] == "Low confidence"
    assert item["confidence_percent"] == 42


def test_delete_removes_row_and_image_file(app, client):
    saved = add_with_file(app, "delete-me.jpg", "metal", 0.84)
    image_path = Path(app.config["UPLOAD_FOLDER"]) / "delete-me.jpg"
    assert image_path.exists()

    response = client.delete(f"/api/history/{saved['id']}")
    assert response.status_code == 200
    assert response.get_json()["ok"] is True
    assert not image_path.exists()
    with app.app_context():
        assert db.list_predictions() == []


def test_delete_missing_item_returns_404(client):
    response = client.delete("/api/history/99999")
    assert response.status_code == 404
    assert response.get_json()["ok"] is False


def test_delete_does_not_follow_unsafe_image_paths(app, client):
    sentinel = Path(app.root_path) / "do-not-delete.txt"
    sentinel.write_text("keep me", encoding="utf-8")
    try:
        with app.app_context():
            saved = db.add_prediction("../do-not-delete.txt", "paper", 0.8)
        response = client.delete(f"/api/history/{saved['id']}")
        assert response.status_code == 200
        assert sentinel.exists()
    finally:
        sentinel.unlink(missing_ok=True)


def test_clear_history_removes_all_rows_and_files(app, client):
    add_with_file(app, "one.jpg", "plastic", 0.9)
    add_with_file(app, "two.jpg", "glass", 0.8)
    response = client.delete("/api/history")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["deleted"] == 2
    assert list(Path(app.config["UPLOAD_FOLDER"]).glob("*.jpg")) == []
    with app.app_context():
        assert db.get_stats()["total"] == 0


def test_stats_endpoint_reflects_database(app, client):
    add_with_file(app, "one.jpg", "paper", 0.9)
    add_with_file(app, "two.jpg", "paper", 0.8)
    add_with_file(app, "three.jpg", "glass", 0.7)
    response = client.get("/api/stats")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ok"] is True
    assert payload["total"] == 3
    assert payload["most_common"]["category"] == "paper"
    assert payload["most_common"]["count"] == 2
    assert payload["per_category"]["metal"] == 0


def test_history_is_local_to_this_app_database(app, client):
    add_with_file(app, "one.jpg", "plastic", 0.9)
    with app.app_context():
        count_before = db.get_stats()["total"]
    assert count_before == 1
    # Delete and confirm both the endpoint and the DB agree.
    client.delete("/api/history")
    assert client.get("/api/history").get_json()["count"] == 0


def test_missing_preview_file_is_reported_honestly(app, client):
    with app.app_context():
        db.add_prediction("uploads/missing-file.jpg", "metal", 0.77)
    response = client.get("/api/history")
    item = response.get_json()["items"][0]
    assert item["image_available"] is False
    assert item["image_url"].endswith("missing-file.jpg")
