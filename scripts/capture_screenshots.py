"""Capture responsive screenshots of EcoSort AI at 375px, 768px, and 1440px."""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
SCREENSHOTS_DIR = ROOT / "screenshots"
SCREENSHOTS_DIR.mkdir(exist_ok=True)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def main():
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env.update(
        {
            "PORT": str(port),
            "SECRET_KEY": "screenshot-secret-key",
            "ECOSORT_MODEL": str(ROOT / "ml" / "model" / "ecosort_model.keras"),
            "LOG_LEVEL": "WARNING",
            "TF_CPP_MIN_LOG_LEVEL": "2",
            "PYTHONUNBUFFERED": "1",
        }
    )

    print(f"Starting server on port {port}...")
    server = subprocess.Popen([sys.executable, "app.py"], cwd=str(ROOT), env=env)

    # Wait for server health
    deadline = time.time() + 60
    ready = False
    while time.time() < deadline:
        if server.poll() is not None:
            break
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=2) as r:
                if r.status == 200:
                    ready = True
                    break
        except Exception:
            time.sleep(0.5)

    if not ready:
        server.terminate()
        print("Server failed to start.")
        return 1

    print("Server ready. Capturing screenshots...")
    viewports = {
        "375": {"width": 375, "height": 812},
        "768": {"width": 768, "height": 1024},
        "1440": {"width": 1440, "height": 900},
    }

    pages_to_capture = [
        ("home", "/"),
        ("classify", "/classify"),
        ("history", "/history"),
        ("learn", "/learn"),
        ("about", "/about"),
    ]

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            for vp_name, vp_dims in viewports.items():
                print(f"Capturing viewport {vp_name} ({vp_dims['width']}x{vp_dims['height']})...")
                context = browser.new_context(viewport=vp_dims)
                page = context.new_page()

                for page_name, path in pages_to_capture:
                    url = f"{base_url}{path}"
                    page.goto(url, wait_until="networkidle")
                    time.sleep(0.3)
                    out_path = SCREENSHOTS_DIR / f"{page_name}-{vp_name}.png"
                    page.screenshot(path=str(out_path), full_page=False)
                    print(f"  Saved {out_path.name}")

                context.close()
            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()

    print("All screenshots captured successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
