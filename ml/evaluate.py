"""Evaluate EcoSort on the held-out test split and create report assets.

Outputs (all based on real measurements, never invented):
    ml/model/metrics.json
    ml/model/model_meta.json  (demo flag cleared)
    static/img/confusion_matrix.png
    static/img/training_curves.png

Usage:
    python ml/evaluate.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from database.db import CLASS_DISPLAY, CLASS_NAMES
from ml.preprocess import load_rgb, resize_rgb

MODEL_DIR = ROOT / "ml" / "model"
PROCESSED_DIR = ROOT / "data" / "processed"
IMAGE_DIR = ROOT / "static" / "img"
MAIN_MODEL = MODEL_DIR / "ecosort_model.keras"
BASELINE_MODEL = MODEL_DIR / "baseline_cnn.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
METRICS_PATH = MODEL_DIR / "metrics.json"
META_PATH = MODEL_DIR / "model_meta.json"

FOREST = "#0F4C3A"
TEAL = "#14B8A6"
AMBER = "#D97706"


def load_test_set() -> tuple[np.ndarray, np.ndarray]:
    images: list[np.ndarray] = []
    labels: list[int] = []
    for index, name in enumerate(CLASS_NAMES):
        files = sorted((PROCESSED_DIR / "test" / name).glob("*"))
        print(f"  test/{name:<10} {len(files):4d} images")
        for path in files:
            try:
                images.append(resize_rgb(load_rgb(path)))
                labels.append(index)
            except Exception as exc:  # noqa: BLE001
                print(f"  Skipping {path}: {exc}")
    if not images:
        raise SystemExit("No test images found. Run ml/prepare_data.py first.")
    return np.stack(images), np.asarray(labels, dtype=np.int32)


def predict_batches(model: tf.keras.Model, images: np.ndarray, batch_size: int = 32):
    probabilities = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size].astype(np.float32) / 255.0
        probabilities.append(model.predict(batch, verbose=0))
    return np.concatenate(probabilities, axis=0)


def metrics_for_predictions(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = np.argmax(probabilities, axis=1)
    report = classification_report(
        labels,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=list(CLASS_NAMES),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(labels, predictions, labels=list(range(len(CLASS_NAMES))))
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(
        matrix.astype(float),
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0,
    )

    per_class = {}
    for name in CLASS_NAMES:
        row = report[name]
        per_class[name] = {
            "precision": round(float(row["precision"]), 4),
            "recall": round(float(row["recall"]), 4),
            "f1": round(float(row["f1-score"]), 4),
            "support": int(row["support"]),
        }

    return {
        "total": int(len(labels)),
        "accuracy": round(float(accuracy_score(labels, predictions)), 4),
        "macro_precision": round(float(report["macro avg"]["precision"]), 4),
        "macro_recall": round(float(report["macro avg"]["recall"]), 4),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 4),
        "weighted_precision": round(float(report["weighted avg"]["precision"]), 4),
        "weighted_recall": round(float(report["weighted avg"]["recall"]), 4),
        "weighted_f1": round(float(report["weighted avg"]["f1-score"]), 4),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_normalized": np.round(normalised, 4).tolist(),
    }


def plot_confusion(matrix: list[list[int]], path: Path) -> None:
    labels = [CLASS_DISPLAY[name] for name in CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(8.2, 6.6))
    sns.heatmap(
        np.asarray(matrix),
        annot=True,
        fmt="d",
        cmap="Greens",
        cbar=True,
        xticklabels=labels,
        yticklabels=labels,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
    )
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_ylabel("True label", fontsize=11)
    ax.set_title("EcoSort confusion matrix - held-out test set", fontsize=13, pad=14)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_training(history_path: Path, path: Path) -> None:
    if not history_path.exists():
        return
    record = json.loads(history_path.read_text(encoding="utf-8"))
    head = record.get("transfer", {}).get("head", {})
    fine = record.get("transfer", {}).get("finetune", {})
    if not head:
        return

    def join(key):
        return list(head.get(key, [])) + list(fine.get(key, []))

    train_acc = join("accuracy")
    val_acc = join("val_accuracy")
    train_loss = join("loss")
    val_loss = join("val_loss")
    if not train_acc:
        return
    epochs = list(range(1, len(train_acc) + 1))
    head_epochs = len(head.get("accuracy", []))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, train_values, val_values, title in (
        (axes[0], train_acc, val_acc, "Accuracy"),
        (axes[1], train_loss, val_loss, "Loss"),
    ):
        ax.plot(epochs, train_values, color=FOREST, linewidth=2, label="Training")
        ax.plot(epochs, val_values, color=TEAL, linewidth=2, label="Validation")
        if fine:
            boundary = head_epochs + 0.5
            ax.axvline(boundary, color=AMBER, linestyle="--", linewidth=1.2)
            ax.text(
                boundary + 0.2,
                ax.get_ylim()[1] * 0.96,
                "fine-tuning",
                color=AMBER,
                fontsize=8,
                va="top",
            )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
    fig.suptitle("EcoSort MobileNetV2 training", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    if not MAIN_MODEL.exists():
        raise SystemExit("ml/model/ecosort_model.keras not found. Run ml/train.py first.")
    if not CLASS_NAMES_PATH.exists():
        raise SystemExit("ml/model/class_names.json not found. Run ml/train.py first.")

    if not (PROCESSED_DIR / "test").exists():
        raise SystemExit("data/processed/test not found. Run ml/prepare_data.py first.")

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading held-out test set ...")
    images, labels = load_test_set()
    print(f"Test images: {len(images)}")

    print("Loading model ...")
    model = tf.keras.models.load_model(MAIN_MODEL, compile=False, safe_mode=True)
    print("Running predictions ...")
    probabilities = predict_batches(model, images)
    test_metrics = metrics_for_predictions(labels, probabilities)
    print(
        f"Accuracy {test_metrics['accuracy']:.4f} | "
        f"macro F1 {test_metrics['macro_f1']:.4f}"
    )

    baseline_metrics = None
    if BASELINE_MODEL.exists():
        print("Evaluating custom CNN baseline ...")
        baseline = tf.keras.models.load_model(
            BASELINE_MODEL, compile=False, safe_mode=True
        )
        baseline_probabilities = predict_batches(baseline, images)
        baseline_metrics = metrics_for_predictions(labels, baseline_probabilities)
        print(
            f"Baseline accuracy {baseline_metrics['accuracy']:.4f} | "
            f"macro F1 {baseline_metrics['macro_f1']:.4f}"
        )

    print("Creating confusion matrix and training curves ...")
    plot_confusion(test_metrics["confusion_matrix"], IMAGE_DIR / "confusion_matrix.png")
    plot_training(MODEL_DIR / "training_history.json", IMAGE_DIR / "training_curves.png")

    summary_path = PROCESSED_DIR / "split_counts.json"
    dataset_summary = {}
    if summary_path.exists():
        dataset_summary = json.loads(summary_path.read_text(encoding="utf-8"))

    training_record = {}
    if (MODEL_DIR / "training_history.json").exists():
        training_record = json.loads(
            (MODEL_DIR / "training_history.json").read_text(encoding="utf-8")
        )
    settings = training_record.get("settings", {})

    metrics = {
        "demo": False,
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {
            "name": "MobileNetV2 transfer learning (head + fine-tuned)",
            "file": MAIN_MODEL.name,
            "input_size": [224, 224],
            "parameters": int(model.count_params()),
            "classes": list(CLASS_NAMES),
        },
        "dataset": {
            "sources": dataset_summary.get("sources", {}),
            "total_images": dataset_summary.get("total_images"),
            "split_totals": dataset_summary.get("split_totals", {}),
            "class_totals": dataset_summary.get("class_totals", {}),
            "test_per_class": {
                name: int(sum(1 for label in labels if CLASS_NAMES[int(label)] == name))
                for name in CLASS_NAMES
            },
        },
        "test": test_metrics,
        "baseline": baseline_metrics,
        "training": {
            "selected_phase": settings.get("selected_phase"),
            "validation_accuracy": settings.get("selected_val_accuracy"),
            "baseline_validation_accuracy": settings.get("baseline_val_accuracy"),
            "head_epochs": settings.get("head_epochs"),
            "finetune_epochs": settings.get("finetune_epochs"),
            "training_seconds": settings.get("training_seconds"),
            "class_weights": settings.get("class_weights"),
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Only now, after a real held-out evaluation succeeded, is the model no
    # longer a demo. The About page reads these values and the banner disappears.
    meta = json.loads(META_PATH.read_text(encoding="utf-8")) if META_PATH.exists() else {}
    meta.update(
        {
            "demo": False,
            "metrics_available": True,
            "test_accuracy": test_metrics["accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "evaluated_at": metrics["evaluated_at"],
            "evaluation_test_size": test_metrics["total"],
            "model_name": "MobileNetV2 transfer learning",
            "model_file": MAIN_MODEL.name,
            "gradcam_layer": meta.get("gradcam_layer", "mobilenet_features"),
        }
    )
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\nPer-class results")
    print(f"{'class':<12}{'precision':>10}{'recall':>8}{'f1':>8}{'support':>9}")
    for name in CLASS_NAMES:
        row = test_metrics["per_class"][name]
        print(
            f"{name:<12}{row['precision']:>10.3f}{row['recall']:>8.3f}"
            f"{row['f1']:>8.3f}{row['support']:>9d}"
        )
    print(f"\nMetrics JSON      : {METRICS_PATH}")
    print(f"Confusion matrix  : {IMAGE_DIR / 'confusion_matrix.png'}")
    print(f"Training curves   : {IMAGE_DIR / 'training_curves.png'}")
    print("The DEMO banner is now disabled because real test metrics exist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
