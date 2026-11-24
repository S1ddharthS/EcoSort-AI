"""Script to create a granular, professional git commit history."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

commit_plan = [
    (
        [".gitignore", ".dockerignore", ".env.example"],
        "chore: project initialization with gitignore and editor config",
    ),
    (
        ["requirements.txt", "requirements-dev.txt"],
        "build: define runtime and development dependencies",
    ),
    (
        ["config.py"],
        "config: add application configuration with environment overrides",
    ),
    (
        ["database/schema.sql", "database/__init__.py"],
        "db: design relational schema and seed initial waste categories",
    ),
    (
        ["database/db.py"],
        "db: implement sqlite data access layer with connection management",
    ),
    (
        ["ml/__init__.py", "ml/preprocess.py"],
        "ml: create shared image loading and preprocessing pipeline",
    ),
    (
        ["ml/prepare_data.py"],
        "ml: implement dataset ingestion and stratified split logic",
    ),
    (
        ["ml/train.py"],
        "ml: implement MobileNetV2 transfer learning and custom CNN baseline",
    ),
    (
        ["ml/gradcam.py"],
        "ml: add gradcam interpretability and activation mapping",
    ),
    (
        ["ml/predict.py"],
        "ml: implement model inference service and prediction pipeline",
    ),
    (
        ["ml/evaluate.py"],
        "ml: create held-out model evaluation and metrics generation script",
    ),
    (
        ["data/prepare_log.txt", "data/processed/split_counts.json", "data/train_smoke_log.txt"],
        "ml: record data preparation and training log artifacts",
    ),
    (
        ["static/css/styles.css"],
        "ui: create design system with css variables and responsive layout",
    ),
    (
        ["static/fonts/inter-var.woff2", "static/fonts/sora-var.woff2"],
        "ui: bundle self-hosted web fonts for offline accessibility",
    ),
    (
        ["static/img/logo.svg", "static/img/icons.svg", "static/img/favicon.svg"],
        "ui: add brand logo and scalable vector icon set",
    ),
    (
        ["static/js/theme-init.js"],
        "ui: add client-side theme initialization and dark mode toggle",
    ),
    (
        ["static/js/main.js"],
        "ui: add main navigation interactivity and shared utilities",
    ),
    (
        ["templates/_macros.html"],
        "ui: define reusable jinja2 component macros",
    ),
    (
        ["templates/base.html"],
        "ui: implement base layout template with skip link and header nav",
    ),
    (
        ["templates/index.html", "static/img/example_plastic.jpg"],
        "feat(home): design landing page hero and live activity metrics",
    ),
    (
        ["templates/classify.html"],
        "feat(classify): build accessible photo upload and camera tabs",
    ),
    (
        ["static/js/classify.js"],
        "feat(classify): implement drag-drop upload, webcam stream, and gradcam view",
    ),
    (
        ["templates/history.html"],
        "feat(history): create local prediction history view and category filters",
    ),
    (
        ["static/js/history.js"],
        "feat(history): add history client management, deletion and modal confirmation",
    ),
    (
        ["templates/learn.html"],
        "feat(learn): implement educational waste guides and detail views",
    ),
    (
        ["static/js/learn.js"],
        "feat(learn): add client-side instant search across waste guides",
    ),
    (
        ["templates/about.html"],
        "feat(about): display held-out test evaluation metrics and model specs",
    ),
    (
        ["templates/404.html", "templates/500.html"],
        "feat(errors): create custom accessible 404 and 500 error pages",
    ),
    (
        ["app.py"],
        "feat(api): create flask application factory, rest endpoints, and rate limiting",
    ),
    (
        ["pytest.ini", "tests/__init__.py", "tests/conftest.py", "tests/helpers.py"],
        "test: configure pytest fixtures, test database, and mock model service",
    ),
    (
        ["tests/test_database.py"],
        "test: add unit tests for sqlite database operations and recommendations",
    ),
    (
        ["tests/test_preprocess.py"],
        "test: add unit tests for image preprocessing and validation constraints",
    ),
    (
        ["tests/test_predict_api.py"],
        "test: add comprehensive unit tests for predict api and upload validation",
    ),
    (
        ["tests/test_history_api.py"],
        "test: add unit tests for history api, filtering and bulk delete",
    ),
    (
        ["tests/test_pages.py"],
        "test: add page routing, markup accessibility and error handling tests",
    ),
    (
        ["tests/e2e/test_flows.py"],
        "test: add playwright end-to-end browser user flow test suite",
    ),
    (
        [
            "ml/model/class_names.json",
            "ml/model/metrics.json",
            "ml/model/model_meta.json",
            "ml/model/training_history.json",
        ],
        "ml: record held-out test evaluation metrics and model metadata",
    ),
    (
        ["static/img/confusion_matrix.png", "static/img/training_curves.png"],
        "ml: add confusion matrix and training curves visualization assets",
    ),
    (
        ["ml/model/ecosort_model.keras"],
        "ml: save trained MobileNetV2 production model weights",
    ),
    (
        ["scripts/capture_screenshots.py", "screenshots"],
        "docs: add automated responsive viewport screenshot capture script and assets",
    ),
    (
        ["scripts/diagnose_real_world.py"],
        "tools: add real-world diagnostic script for external dataset evaluation",
    ),
    (
        ["Dockerfile", "gunicorn.conf.py", "Procfile", "DEPLOY.md"],
        "deploy: create production dockerfile, gunicorn config and cloud deployment guide",
    ),
    (
        ["VIVA_NOTES.md"],
        "docs: create comprehensive viva voce defense guide for 22ETP58",
    ),
    (
        ["HANDOFF.md"],
        "docs: add project handoff notes and architecture status",
    ),
    (
        ["README.md", "scripts/make_commits.py"],
        "docs: add comprehensive readme with quickstart, architecture, and api docs",
    ),
]


def run_cmd(cmd):
    p = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if p.returncode != 0:
        print(f"Error running {' '.join(cmd)}: {p.stderr}")
        return False
    return True


def main():
    print("Starting granular commit creation...")
    for idx, (files, message) in enumerate(commit_plan, 1):
        existing_files = [f for f in files if (ROOT / f).exists()]
        if not existing_files:
            print(f"Skipping [{idx}/{len(commit_plan)}]: no files found for {message}")
            continue

        add_cmd = ["git", "add"] + existing_files
        if not run_cmd(add_cmd):
            print(f"Failed to add files for [{idx}]: {existing_files}")
            return 1

        # Check if anything is staged
        status_p = subprocess.run(
            ["git", "diff", "--cached", "--quiet"], cwd=str(ROOT)
        )
        if status_p.returncode == 0:
            print(f"Nothing staged for [{idx}]: {message}")
            continue

        commit_cmd = ["git", "commit", "-m", message]
        if not run_cmd(commit_cmd):
            print(f"Failed to commit [{idx}]: {message}")
            return 1

        print(f"[{idx}/{len(commit_plan)}] Committed: {message}")

    # Check remaining untracked
    run_cmd(["git", "branch", "-M", "main"])
    print("Renamed branch to main.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
