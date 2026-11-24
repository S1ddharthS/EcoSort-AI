"""Database-layer tests: schema, seed content and parameterised CRUD."""

from __future__ import annotations

import pytest

import database.db as db
from database.db import CLASS_NAMES


@pytest.fixture()
def db_app(app):
    return app


def test_all_six_recommendations_are_seeded(db_app):
    with db_app.app_context():
        rows = db.list_recommendations()
    assert [row["category"] for row in rows] == list(CLASS_NAMES)
    for row in rows:
        assert row["display_name"]
        assert row["bin_colour"]
        assert isinstance(row["how_to_prepare"], list) and row["how_to_prepare"]
        assert isinstance(row["do_list"], list) and row["do_list"]
        assert isinstance(row["dont_list"], list) and row["dont_list"]
        assert row["fun_fact"]
        assert row["environmental_impact"]


def test_get_recommendation_returns_none_for_unknown_and_injection_safe(db_app):
    with db_app.app_context():
        assert db.get_recommendation("battery") is None
        assert db.get_recommendation("plastic' OR 1=1 --") is None


def test_add_list_delete_prediction(db_app):
    with db_app.app_context():
        saved = db.add_prediction("uploads/first.jpg", "plastic", 0.9123)
        assert saved["id"] > 0
        assert saved["predicted_class"] == "plastic"
        assert saved["confidence"] == pytest.approx(0.9123)

        rows = db.list_predictions()
        assert len(rows) == 1
        assert rows[0]["image_path"] == "uploads/first.jpg"

        deleted = db.delete_prediction(saved["id"])
        assert deleted is not None
        assert deleted["image_path"] == "uploads/first.jpg"
        assert db.list_predictions() == []
        assert db.delete_prediction(saved["id"]) is None


def test_list_predictions_is_newest_first_and_filterable(db_app):
    with db_app.app_context():
        first = db.add_prediction("uploads/one.jpg", "paper", 0.81)
        second = db.add_prediction("uploads/two.jpg", "glass", 0.75)
        third = db.add_prediction("uploads/three.jpg", "paper", 0.67)

        every = db.list_predictions()
        assert [row["id"] for row in every] == [third["id"], second["id"], first["id"]]

        paper = db.list_predictions("paper")
        assert [row["id"] for row in paper] == [third["id"], first["id"]]

        assert db.list_predictions("metal") == []


def test_stats_track_total_most_common_and_last(db_app):
    with db_app.app_context():
        empty = db.get_stats()
        assert empty["total"] == 0
        assert empty["most_common"] is None
        assert empty["last_classified_at"] is None

        db.add_prediction("uploads/a.jpg", "metal", 0.71)
        db.add_prediction("uploads/b.jpg", "metal", 0.88)
        db.add_prediction("uploads/c.jpg", "organic", 0.9)

        stats = db.get_stats()
        assert stats["total"] == 3
        assert stats["most_common"]["category"] == "metal"
        assert stats["most_common"]["count"] == 2
        assert stats["most_common"]["display_name"] == "Metal"
        assert stats["per_category"]["metal"] == 2
        assert stats["per_category"]["paper"] == 0
        assert stats["last_classified_at"]


def test_clear_predictions_returns_image_paths(db_app):
    with db_app.app_context():
        db.add_prediction("uploads/a.jpg", "metal", 0.71)
        db.add_prediction("uploads/b.jpg", "paper", 0.72)
        paths = db.clear_predictions()
        assert sorted(paths) == ["uploads/a.jpg", "uploads/b.jpg"]
        assert db.get_stats()["total"] == 0


def test_invalid_class_is_rejected_by_schema(db_app):
    with db_app.app_context():
        with pytest.raises(Exception):
            db.add_prediction("uploads/no.jpg", "hazardous", 0.7)


def test_confidence_outside_range_is_rejected(db_app):
    with db_app.app_context():
        with pytest.raises(Exception):
            db.add_prediction("uploads/high.jpg", "plastic", 1.2)
