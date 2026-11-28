"""Unit tests for ml.predict.ModelService, confidence bands, and lazy TF loading."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from database.db import CLASS_NAMES
from ml.predict import ModelService, confidence_label


def test_confidence_label_boundaries():
    assert confidence_label(1.0) == "High confidence"
    assert confidence_label(0.80) == "High confidence"
    assert confidence_label(0.7999) == "Medium confidence"
    assert confidence_label(0.60) == "Medium confidence"
    assert confidence_label(0.5999) == "Low confidence"
    assert confidence_label(0.0) == "Low confidence"


def test_model_service_reports_missing_files(tmp_path):
    missing_model = tmp_path / "missing.keras"
    class_names = tmp_path / "class_names.json"
    service = ModelService(str(missing_model), str(class_names))
    with pytest.raises(RuntimeError, match="not available"):
        service.predict(b"anything")

    service.load()
    assert service.loaded is False
    assert service.load_error and "missing" in service.load_error.lower()

    missing_model.write_bytes(b"placeholder")
    service_no_classes = ModelService(str(missing_model), str(class_names))
    service_no_classes.load()
    assert service_no_classes.loaded is False
    assert service_no_classes.load_error and "class names" in service_no_classes.load_error.lower()

    with pytest.raises(RuntimeError, match="class names"):
        service_no_classes.predict(b"anything")


def test_model_service_rejects_mismatched_class_names(tmp_path):
    model_file = tmp_path / "ecosort.keras"
    model_file.write_bytes(b"not-a-keras-archive")
    class_names_file = tmp_path / "class_names.json"
    class_names_file.write_text(json.dumps(["plastic", "other"]), encoding="utf-8")

    service = ModelService(str(model_file), str(class_names_file))
    service.load()
    assert service.loaded is False
    assert service.load_error is not None
    assert sorted(service.class_names) != sorted(CLASS_NAMES)


def test_explain_returns_none_when_grad_model_unavailable(tmp_path):
    service = ModelService(
        str(tmp_path / "missing.keras"), str(tmp_path / "class_names.json")
    )
    service.class_names = list(CLASS_NAMES)
    service.grad_model = None
    dummy = np.zeros((224, 224, 3), dtype=np.float32)
    assert service.explain(dummy, "plastic") is None


def test_importing_ml_predict_defers_tensorflow_and_gradcam():
    root = Path(__file__).resolve().parents[1]
    script = (
        "import sys, ml.predict; "
        "assert 'tensorflow' not in sys.modules, 'tensorflow imported eagerly'; "
        "assert 'ml.gradcam' not in sys.modules, 'ml.gradcam imported eagerly'"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
