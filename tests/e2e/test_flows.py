"""Browser end-to-end tests against the real Flask app and trained model.

Run with:
    python -m pytest tests/e2e -q

These tests need the trained model at ml/model/ecosort_model.keras. They start
a separate Flask server with an isolated SQLite database and uploads folder.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from playwright.sync_api import expect, sync_playwright

pytestmark = pytest.mark.e2e

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "ml" / "model" / "ecosort_model.keras"
SAMPLE = ROOT / "data" / "processed" / "test" / "plastic" / "plastic_03aa09f3ba8f.jpg"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture(scope="session")
def e2e_server(tmp_path_factory):
    if not MODEL.exists():
        pytest.skip("Trained model missing. Run python ml/train.py first.")
    if not SAMPLE.exists():
        pytest.skip("A real test image is missing. Run python ml/prepare_data.py first.")

    folder = tmp_path_factory.mktemp("ecosort-e2e")
    port = free_port()
    env = os.environ.copy()
    env.update(
        {
            "PORT": str(port),
            "SECRET_KEY": "e2e-only-secret",
            "ECOSORT_DB": str(folder / "e2e.db"),
            "ECOSORT_UPLOADS": str(folder / "uploads"),
            "ECOSORT_MODEL": str(MODEL),
            "LOG_LEVEL": "WARNING",
            "TF_CPP_MIN_LOG_LEVEL": "2",
            "PYTHONUNBUFFERED": "1",
        }
    )
    log_path = folder / "server.log"
    log_file = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "app.py"],
        cwd=str(ROOT),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    base_url = f"http://127.0.0.1:{port}"
    ready = False
    deadline = time.time() + 120
    while time.time() < deadline:
        if process.poll() is not None:
            break
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=2) as response:
                if response.status == 200:
                    ready = True
                    break
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            time.sleep(0.5)
    if not ready:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        log_file.close()
        pytest.fail(
            "E2E server did not become healthy.\n" + log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
        )

    yield {"url": base_url, "folder": folder, "port": port}

    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
    log_file.close()


@pytest.fixture(scope="session")
def browser():
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture()
def page(browser, e2e_server):
    context = browser.new_context(viewport={"width": 1440, "height": 1000})
    page = context.new_page()
    yield page
    context.close()


def reset_history(base_url: str) -> None:
    request = urllib.request.Request(
        f"{base_url}/api/history", method="DELETE", headers={"Accept": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=10):
        pass


def test_empty_history_state(e2e_server, page):
    reset_history(e2e_server["url"])
    page.goto(e2e_server["url"] + "/history", wait_until="networkidle")
    expect(page.locator("#history-empty")).to_be_visible(timeout=15000)
    expect(page.locator("#history-empty-title")).to_have_text("No classifications yet")
    expect(page.locator("#history-total")).to_have_text("0")
    expect(page.locator("#clear-history")).to_be_disabled()
    expect(page.locator("#history-empty-action")).to_be_visible()


def test_upload_result_history_delete_flow(e2e_server, page):
    reset_history(e2e_server["url"])
    page.goto(e2e_server["url"] + "/classify", wait_until="networkidle")
    page.set_input_files("#file-input", str(SAMPLE))
    expect(page.locator("#selected-preview")).to_be_visible(timeout=10000)
    expect(page.locator("#classify-btn")).to_be_enabled(timeout=10000)

    page.locator("#classify-btn").click()
    expect(page.locator("#analysis-state")).to_be_visible(timeout=5000)
    expect(page.locator("#result-section .result-card")).to_be_visible(timeout=90000)

    # The real model should return a confidence ring or an honest "not sure"
    # screen. Either is a valid prediction result; both save to History.
    assert page.locator("#result-section .result-card h2").first.is_visible()
    page.locator("#result-section .result-actions a[href='/history']").click()
    expect(page).to_have_url(e2e_server["url"] + "/history")

    cards = page.locator(".history-card:not(.skeleton-card)")
    expect(cards.first).to_be_visible(timeout=15000)
    assert cards.count() >= 1
    expect(page.locator("#history-total")).to_have_text("1")

    cards.first.locator(".delete-button").click()
    expect(page.locator("#confirm-dialog")).to_be_visible()
    page.locator("#confirm-accept").click()
    expect(page.locator("#history-empty")).to_be_visible(timeout=15000)
    expect(page.locator("#history-total")).to_have_text("0")


def test_camera_denied_fallback(e2e_server, browser):
    context = browser.new_context(viewport={"width": 390, "height": 844})
    page = context.new_page()
    page.add_init_script(
        """
        Object.defineProperty(navigator, "mediaDevices", {
          configurable: true,
          value: {
            getUserMedia: () => Promise.reject(
              new DOMException("Blocked by test", "NotAllowedError")
            )
          }
        });
        """
    )
    page.goto(e2e_server["url"] + "/classify", wait_until="networkidle")
    page.locator("#tab-camera").click()
    page.locator("#start-camera").click()
    expect(page.locator("#camera-permission")).to_be_visible(timeout=10000)
    expect(page.locator("#camera-error-title")).to_have_text("Camera access was blocked")
    expect(page.locator("#camera-error-text")).to_contain_text("upload a photo instead")

    page.locator("#camera-fallback").click()
    expect(page.locator("#panel-upload")).to_be_visible()
    expect(page.locator("#tab-upload")).to_have_attribute("aria-selected", "true")
    context.close()


def test_learn_search_empty_state(e2e_server, page):
    page.goto(e2e_server["url"] + "/learn", wait_until="networkidle")
    page.locator("#guide-search").fill("quantum banana spaceship")
    expect(page.locator("#learn-empty")).to_be_visible()
    expect(page.locator("#search-count")).to_contain_text("No guides match")
    page.locator("#clear-guide-search").click()
    expect(page.locator("#learn-list")).to_be_visible()
    expect(page.locator(".learn-card")).to_have_count(6)


def test_keyboard_skip_link_and_theme_toggle(e2e_server, page):
    page.goto(e2e_server["url"] + "/", wait_until="networkidle")
    page.keyboard.press("Tab")
    skip = page.locator(".skip-link")
    expect(skip).to_be_focused()
    expect(skip).to_be_in_viewport()
    assert skip.evaluate("el => getComputedStyle(el).transform") != "none"

    page.locator("#theme-toggle").click()
    expect(page.locator("html")).to_have_attribute("data-theme", "dark")
    page.reload(wait_until="networkidle")
    expect(page.locator("html")).to_have_attribute("data-theme", "dark")
    # Return the shared browser profile to light mode for later screenshots.
    page.locator("#theme-toggle").click()
    expect(page.locator("html")).to_have_attribute("data-theme", "light")
