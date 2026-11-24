"""Diagnose EcoSort AI on external non-TrashNet real photos."""

import random
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf

from database.db import CLASS_NAMES
from ml.preprocess import preprocess_image

model = tf.keras.models.load_model("ml/model/ecosort_model.keras", compile=False)

folder_mapping = {
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

with zipfile.ZipFile("data/raw/garbage12b.zip", "r") as z:
    all_files = z.namelist()
    results = defaultdict(
        lambda: {"total": 0, "correct": 0, "predictions": defaultdict(int)}
    )
    glass_details = []

    for target_class, folders in folder_mapping.items():
        class_files = [
            f
            for f in all_files
            if any(f.startswith(fld + "/") for fld in folders)
            and f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        random.seed(123)
        selected = random.sample(class_files, min(25, len(class_files)))

        for f in selected:
            data = z.read(f)
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
                subfolder = f.split("/")[1]
                glass_details.append((f, subfolder, pred_class, conf))

print("=== REAL-WORLD EXTERNAL DATASET TEST (25 images per class) ===")
for c in CLASS_NAMES:
    res = results[c]
    total = res["total"]
    corr = res["correct"]
    acc = corr / total if total > 0 else 0
    print(f"Class: {c:<10} | Acc: {acc*100:5.1f}% ({corr}/{total})")
    preds_str = ", ".join(
        f"{k}: {v}"
        for k, v in sorted(res["predictions"].items(), key=lambda x: -x[1])
    )
    print(f"   Predicted as -> {preds_str}")

print("\n=== GLASS BREAKDOWN BY SUBFOLDER (Brown / Green / White) ===")
for f, subfolder, pred_class, conf in glass_details:
    filename = Path(f).name
    print(f"  {subfolder:<13} {filename:<12} -> {pred_class:<10} ({conf*100:5.1f}%)")
