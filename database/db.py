"""Thin SQLite data-access layer for EcoSort AI.

Every query uses parameter binding. The module keeps one connection per
Flask request (stored on ``g``) and closes it automatically.
"""

import json
import sqlite3
from pathlib import Path

from flask import current_app, g

# The canonical class order used everywhere in the project.
CLASS_NAMES = ("plastic", "paper", "cardboard", "glass", "metal", "organic")
CLASS_DISPLAY = {
    "plastic": "Plastic",
    "paper": "Paper",
    "cardboard": "Cardboard",
    "glass": "Glass",
    "metal": "Metal",
    "organic": "Organic waste",
}


def get_db() -> sqlite3.Connection:
    """Return the request-scoped database connection."""
    if "db" not in g:
        connection = sqlite3.connect(
            current_app.config["DATABASE"], timeout=10
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        g.db = connection
    return g.db


def close_db(_error=None) -> None:
    """Close the connection at the end of a request."""
    connection = g.pop("db", None)
    if connection is not None:
        connection.close()


def init_db(app) -> None:
    """Create tables and seed recommendations if they are missing."""
    schema_path = Path(app.root_path) / "database" / "schema.sql"
    with app.app_context():
        connection = get_db()
        connection.executescript(schema_path.read_text(encoding="utf-8"))
        connection.commit()


def init_app(app) -> None:
    """Register the teardown handler (schema setup is done by create_app)."""
    app.teardown_appcontext(close_db)


def _recommendation_from_row(row: sqlite3.Row) -> dict:
    data = dict(row)
    for key in ("how_to_prepare", "do_list", "dont_list"):
        data[key] = json.loads(data[key])
    data["display_name"] = CLASS_DISPLAY.get(data["category"], data["category"].title())
    return data


def get_recommendation(category: str) -> dict | None:
    """Fetch one category's recycling guidance, with JSON fields parsed."""
    row = get_db().execute(
        "SELECT * FROM recommendations WHERE category = ?", (category,)
    ).fetchone()
    return _recommendation_from_row(row) if row else None


def list_recommendations() -> list[dict]:
    """Return all recommendations in the canonical class order."""
    rows = get_db().execute("SELECT * FROM recommendations").fetchall()
    by_category = {row["category"]: _recommendation_from_row(row) for row in rows}
    return [by_category[name] for name in CLASS_NAMES if name in by_category]


def add_prediction(image_path: str, predicted_class: str, confidence: float) -> dict:
    """Insert a prediction and return the saved row."""
    connection = get_db()
    cursor = connection.execute(
        "INSERT INTO predictions (image_path, predicted_class, confidence) "
        "VALUES (?, ?, ?)",
        (image_path, predicted_class, float(confidence)),
    )
    connection.commit()
    row = connection.execute(
        "SELECT id, image_path, predicted_class, confidence, created_at "
        "FROM predictions WHERE id = ?",
        (cursor.lastrowid,),
    ).fetchone()
    return dict(row)


def list_predictions(category: str | None = None) -> list[dict]:
    """Return predictions, newest first, optionally filtered by category."""
    if category:
        rows = get_db().execute(
            "SELECT id, image_path, predicted_class, confidence, created_at "
            "FROM predictions WHERE predicted_class = ? "
            "ORDER BY datetime(created_at) DESC, id DESC",
            (category,),
        ).fetchall()
    else:
        rows = get_db().execute(
            "SELECT id, image_path, predicted_class, confidence, created_at "
            "FROM predictions ORDER BY datetime(created_at) DESC, id DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def delete_prediction(prediction_id: int) -> dict | None:
    """Delete one prediction. Returns the deleted row, or None if absent."""
    connection = get_db()
    row = connection.execute(
        "SELECT id, image_path FROM predictions WHERE id = ?", (prediction_id,)
    ).fetchone()
    if row is None:
        return None
    connection.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
    connection.commit()
    return dict(row)


def clear_predictions() -> list[str]:
    """Delete all predictions and return their image paths for file cleanup."""
    connection = get_db()
    rows = connection.execute("SELECT image_path FROM predictions").fetchall()
    connection.execute("DELETE FROM predictions")
    connection.commit()
    return [row["image_path"] for row in rows]


def get_stats() -> dict:
    """Live statistics shown on the home page and returned by /api/stats."""
    connection = get_db()
    total = connection.execute("SELECT COUNT(*) AS n FROM predictions").fetchone()["n"]

    counts = {name: 0 for name in CLASS_NAMES}
    for row in connection.execute(
        "SELECT predicted_class, COUNT(*) AS n FROM predictions "
        "GROUP BY predicted_class"
    ).fetchall():
        counts[row["predicted_class"]] = row["n"]

    most_common = None
    if total:
        row = connection.execute(
            "SELECT predicted_class FROM predictions "
            "GROUP BY predicted_class ORDER BY COUNT(*) DESC, predicted_class ASC "
            "LIMIT 1"
        ).fetchone()
        if row:
            most_common = {
                "category": row["predicted_class"],
                "display_name": CLASS_DISPLAY.get(
                    row["predicted_class"], row["predicted_class"].title()
                ),
                "count": counts.get(row["predicted_class"], 0),
            }

    last = connection.execute(
        "SELECT created_at FROM predictions ORDER BY id DESC LIMIT 1"
    ).fetchone()

    return {
        "total": total,
        "most_common": most_common,
        "per_category": counts,
        "last_classified_at": last["created_at"] if last else None,
    }
