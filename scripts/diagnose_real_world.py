"""Diagnose EcoSort AI on an external non-TrashNet image archive."""

from __future__ import annotations

import argparse
import random
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from database.db import CLASS_NAMES
from ml.preprocess import preprocess_image

DEFAULT_MODEL = ROOT / "ml" / "model" / "ecosort_model.keras"
DEFAULT_ZIP = ROOT / "data" / "raw" / "garbage12b.zip"

FOLDER_MAPPING = {
    "glass": [
        "garbage_classification/brown-glass",
        "garbage_classification/green-glass",
        "garbage_classification/white-glass",
    ],
    "cardboard": ["garbage_classification/cardboard"],
    "paper": ["garbage_classification/paper"],
    "metal": ["garbage_classification/metal"],
    "plastic": ["garbage_classification/plastic"],
    "organic": ["garbage_classification/biological"],
}


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


def evaluate_archive(
    model_path: Path, zip_path: Path, samples: int, seed: int
) -> tuple[dict, list[tuple[str, str, str, float]]]:
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path, compile=False, safe_mode=True)
    results = defaultdict(
        lambda: {"total": 0, "correct": 0, "predictions": defaultdict(int)}
    )
    glass_details: list[tuple[str, str, str, float]] = []

    with zipfile.ZipFile(zip_path, "r") as archive:
        all_files = archive.namelist()
        for target_class, folders in FOLDER_MAPPING.items():
            class_files = sorted(
                f
                for f in all_files
                if any(f.startswith(fld + "/") for fld in folders)
                and f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            )
            rng = random.Random(seed)
            selected = rng.sample(class_files, min(samples, len(class_files)))

            for filename in selected:
                data = archive.read(filename)
                img_arr = preprocess_image(data)
                pred = model.predict(np.expand_dims(img_arr, 0), verbose=0)[0]
                pred_idx = int(np.argmax(pred))
                pred_class = CLASS_NAMES[pred_idx]
                conf = float(pred[pred_idx])

                results[target_class]["total"] += 1
                if pred_class == target_class:
                    results[target_class]["correct"] += 1
                results[target_class]["predictions"][pred_class] += 1

                if target_class == "glass":
                    subfolder = filename.split("/")[1]
                    glass_details.append((filename, subfolder, pred_class, conf))

    return results, glass_details


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate EcoSort AI on an external ZIP dataset."
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to .keras model")
    parser.add_argument(
        "--zip-path",
        default=str(DEFAULT_ZIP),
        help="Path to external dataset ZIP archive",
    )
    parser.add_argument(
        "--samples", type=int, default=25, help="Max sampled images per class"
    )
    parser.add_argument("--seed", type=int, default=123, help="Sampling seed")
    args = parser.parse_args()

    model_path = resolve_path(args.model)
    zip_path = resolve_path(args.zip_path)
    if not model_path.is_file():
        print(f"Model file not found: {model_path}", file=sys.stderr)
        return 1
    if not zip_path.is_file():
        print(f"Dataset archive not found: {zip_path}", file=sys.stderr)
        return 1

    results, glass_details = evaluate_archive(
        model_path, zip_path, max(1, args.samples), args.seed
    )

    print(f"=== REAL-WORLD EXTERNAL DATASET TEST (up to {args.samples} images per class) ===")
    for category in CLASS_NAMES:
        res = results[category]
        total = res["total"]
        corr = res["correct"]
        acc = corr / total if total > 0 else 0.0
        print(f"Class: {category:<10} | Acc: {acc * 100:5.1f}% ({corr}/{total})")
        preds_str = ", ".join(
            f"{k}: {v}"
            for k, v in sorted(res["predictions"].items(), key=lambda item: -item[1])
        )
        print(f"   Predicted as -> {preds_str}")

    if glass_details:
        print("\n=== GLASS BREAKDOWN BY SUBFOLDER (Brown / Green / White) ===")
        for f, subfolder, pred_class, conf in glass_details:
            print(
                f"  {subfolder:<13} {Path(f).name:<12} -> {pred_class:<10} ({conf * 100:5.1f}%)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
