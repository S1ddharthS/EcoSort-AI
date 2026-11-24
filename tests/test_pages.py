"""Smoke tests for pages, navigation and friendly error handling."""

from __future__ import annotations

import html as html_module
import re


def test_home_page_has_six_categories_and_primary_action(client):
    response = client.get("/")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert "Know what goes" in html
    assert "Classify waste now" in html
    for label in ("Plastic", "Paper", "Cardboard", "Glass", "Metal", "Organic waste"):
        assert f">{label}<" in html or label in html
    assert "https://" not in html.split("class=\"site-footer\"")[0]


def test_all_main_pages_return_200(client):
    for path in ("/", "/classify", "/history", "/learn", "/about"):
        response = client.get(path)
        assert response.status_code == 200, path


def test_classify_page_has_accessible_upload_and_camera_ui(client):
    html = client.get("/classify").get_data(as_text=True)
    assert 'role="tablist"' in html
    assert 'id="dropzone"' in html
    assert 'accept="image/jpeg,image/png,image/webp,.jpg,.jpeg,.png,.webp"' in html
    assert 'id="camera-permission"' in html
    assert 'role="alert"' in html
    assert "JPG, PNG or WEBP" in html


def test_learn_page_lists_all_six_details(client):
    html = client.get("/learn").get_data(as_text=True)
    for category in ("plastic", "paper", "cardboard", "glass", "metal", "organic"):
        assert f'id="category-{category}"' in html
    assert html.count("learn-card") >= 6


def test_about_page_does_not_fabricate_metrics(client):
    html = client.get("/about").get_data(as_text=True)
    assert "Metrics are not available yet" in html or "Real model results" in html
    # Without metrics.json there must be no invented headline figures; the
    # page should show the honest empty state and the exact commands instead.
    if "Metrics are not available yet" in html:
        assert 'class="metric-value"' not in html
        assert "python ml/evaluate.py" in html
        assert "Demo model" in html


def test_history_page_has_loading_and_empty_states(client):
    html = client.get("/history").get_data(as_text=True)
    assert 'id="history-grid"' in html
    assert "skeleton-card" in html
    assert 'id="history-empty"' in html
    assert "No classifications yet" in html


def test_404_page_is_friendly_and_has_navigation(client):
    response = client.get("/not-a-real-page")
    assert response.status_code == 404
    html = response.get_data(as_text=True)
    assert "404" in html
    assert "Back to home" in html
    assert "This page isn't here" in html_module.unescape(html)
    assert "\ufffd" not in html


def test_api_errors_are_json_not_html(client):
    response = client.get("/api/not-here")
    assert response.status_code == 404
    assert response.is_json
    assert response.get_json()["message"]


def test_pages_have_one_h1_and_skip_link(client):
    for path in ("/", "/classify", "/history", "/learn", "/about"):
        html = client.get(path).get_data(as_text=True)
        h1_count = len(re.findall(r"<h1[\s>]", html))
        assert h1_count == 1, f"{path} has {h1_count} h1 elements"
        assert 'class="skip-link"' in html
        assert 'id="main-content"' in html


def test_demo_banner_disappears_for_evaluated_model(app, client):
    app.extensions["fake_model"].demo = False
    html = client.get("/").get_data(as_text=True)
    assert "Demo model" not in html

    app.extensions["fake_model"].demo = True
    html = client.get("/").get_data(as_text=True)
    assert "Demo model" in html


def test_missing_model_banner_is_shown(client, app):
    from tests.helpers import NotReadyModelService

    app.extensions["model_service"] = NotReadyModelService()
    html = client.get("/").get_data(as_text=True)
    assert "The AI model isn't installed" in html
    assert 'data-model-ready="false"' in html
