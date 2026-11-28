"""Train the EcoSort waste classifier.

Two models are trained for comparison:
1. MobileNetV2 transfer learning with a small custom head, followed by
   fine-tuning of the final block of the base network.
2. A small custom CNN baseline trained from scratch.

Everything is seeded. Class weights compensate for the imbalanced dataset.
The script writes:
    ml/model/ecosort_model.keras       (best validation model)
    ml/model/baseline_cnn.keras        (custom CNN baseline)
    ml/model/class_names.json
    ml/model/training_history.json
    ml/model/model_meta.json

Run evaluate.py afterwards to create real test-set metrics and remove the
"Demo model" banner.

Usage:
    python ml/train.py
    python ml/train.py --head-epochs 6 --finetune-epochs 6 --baseline-epochs 6
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep TensorFlow's startup messages quiet. This must be set before importing TF.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers

from database.db import CLASS_NAMES
from ml.preprocess import load_rgb, resize_rgb

PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_OUTPUT_DIR = ROOT / "ml" / "model"
SEED = 42
IMAGE_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE


def set_seeds(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_split(split: str, limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load one split into uint8 arrays using the shared OpenCV resize."""
    images: list[np.ndarray] = []
    labels: list[int] = []
    for index, name in enumerate(CLASS_NAMES):
        folder = PROCESSED_DIR / split / name
        files = sorted(
            path
            for path in folder.glob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        if limit:
            files = files[: max(1, limit // len(CLASS_NAMES))]
        for path in files:
            try:
                images.append(resize_rgb(load_rgb(path)))
                labels.append(index)
            except Exception as exc:  # noqa: BLE001
                print(f"  Skipping unreadable file {path}: {exc}")
    if not images:
        raise SystemExit(f"No images found for split '{split}'. Run prepare_data.py first.")
    print(f"  {split:<5} loaded {len(images):4d} images")
    return np.stack(images), np.asarray(labels, dtype=np.int32)


def make_augmenter() -> tf.keras.Sequential:
    """Light augmentation suitable for photos of single waste items."""
    return tf.keras.Sequential(
        [
            layers.RandomRotation(0.05, fill_mode="reflect", name="aug_rotate"),
            layers.RandomFlip("horizontal", name="aug_flip"),
            layers.RandomZoom(0.10, fill_mode="reflect", name="aug_zoom"),
            layers.RandomTranslation(0.05, 0.05, fill_mode="reflect", name="aug_shift"),
            layers.RandomBrightness(0.12, value_range=(0, 1), name="aug_brightness"),
        ],
        name="augmentation",
    )


def make_dataset(
    images: np.ndarray,
    labels: np.ndarray,
    training: bool,
    batch_size: int,
    augmenter: tf.keras.Sequential | None = None,
):
    """Build a tf.data pipeline that uses the shared [0,1] normalisation."""
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(images), seed=SEED, reshuffle_each_iteration=True
        )
    dataset = dataset.batch(batch_size)

    def prepare(batch_images, batch_labels):
        batch_images = tf.cast(batch_images, tf.float32) / 255.0
        if training and augmenter is not None:
            batch_images = augmenter(batch_images, training=True)
            batch_images = tf.clip_by_value(batch_images, 0.0, 1.0)
        return batch_images, batch_labels

    return dataset.map(prepare, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)


def build_transfer_model():
    """MobileNetV2 split at its last convolution so Grad-CAM is easy later."""
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3), include_top=False, weights="imagenet"
    )
    last_conv = base.get_layer("Conv_1")
    feature_extractor = tf.keras.Model(
        base.input, last_conv.output, name="mobilenet_features"
    )
    tail = tf.keras.Model(last_conv.output, base.output, name="mobilenet_tail")

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    x = layers.Rescaling(2.0, offset=-1.0, name="rescale")(inputs)
    features = feature_extractor(x, training=False)
    x = tail(features, training=False)
    x = layers.GlobalAveragePooling2D(name="pool")(x)
    x = layers.Dropout(0.25, name="dropout")(x)
    outputs = layers.Dense(
        len(CLASS_NAMES), activation="softmax", name="predictions"
    )(x)
    model = tf.keras.Model(inputs, outputs, name="ecosort_mobilenetv2")
    return model, base, feature_extractor, tail


def build_baseline_model() -> tf.keras.Model:
    """Small custom CNN trained from scratch, used only for comparison."""
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    x = inputs
    for block, filters in enumerate((32, 64, 128, 256), start=1):
        x = layers.Conv2D(
            filters, 3, padding="same", use_bias=False, name=f"conv{block}a"
        )(x)
        x = layers.BatchNormalization(name=f"bn{block}a")(x)
        x = layers.Activation("relu", name=f"relu{block}a")(x)
        x = layers.Conv2D(
            filters, 3, padding="same", use_bias=False, name=f"conv{block}b"
        )(x)
        x = layers.BatchNormalization(name=f"bn{block}b")(x)
        x = layers.Activation("relu", name=f"relu{block}b")(x)
        x = layers.MaxPooling2D(name=f"pool{block}")(x)
        x = layers.Dropout(0.20, name=f"drop{block}")(x)
    # Rename the final feature map for Grad-CAM discovery if it is ever used.
    x = layers.Conv2D(
        256, 3, padding="same", activation="relu", name="gradcam_conv"
    )(x)
    x = layers.GlobalAveragePooling2D(name="global_pool")(x)
    x = layers.Dense(128, activation="relu", name="dense_features")(x)
    x = layers.Dropout(0.35, name="head_dropout")(x)
    outputs = layers.Dense(
        len(CLASS_NAMES), activation="softmax", name="predictions"
    )(x)
    return tf.keras.Model(inputs, outputs, name="ecosort_baseline_cnn")


def make_callbacks(checkpoint_path: Path, patience: int = 3):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def history_to_dict(history) -> dict:
    return {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train EcoSort models.")
    parser.add_argument("--head-epochs", type=int, default=6)
    parser.add_argument("--finetune-epochs", type=int, default=6)
    parser.add_argument("--baseline-epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--limit-train",
        type=int,
        default=None,
        help="Optional cap for a quick smoke test (use a temporary output dir).",
    )
    args = parser.parse_args()

    set_seeds()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_head = output_dir / "_head_best.keras"
    checkpoint_finetune = output_dir / "_finetune_best.keras"
    baseline_path = output_dir / "baseline_cnn.keras"

    print("TensorFlow:", tf.__version__)
    print("Loading data ...")
    train_images, train_labels = load_split("train", args.limit_train)
    val_images, val_labels = load_split("val")
    print("Classes:", CLASS_NAMES)
    print("Train class counts:", dict(Counter(train_labels.tolist())))

    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(CLASS_NAMES)),
        y=train_labels,
    )
    class_weights = {
        index: float(weight) for index, weight in enumerate(class_weights_array)
    }
    print("Class weights:", {CLASS_NAMES[i]: round(w, 3) for i, w in class_weights.items()})

    augmenter = make_augmenter()
    train_dataset = make_dataset(
        train_images, train_labels, True, args.batch_size, augmenter
    )
    val_dataset = make_dataset(
        val_images, val_labels, False, args.batch_size, None
    )

    transfer_history = {}
    phase1_best = 0.0
    phase2_best = 0.0
    started = time.time()

    # ------------------------------------------------------------------
    # Model 1: transfer learning - train the head with a frozen base.
    # ------------------------------------------------------------------
    print("\n=== MobileNetV2 transfer learning: head training ===")
    model, base, feature_extractor, tail = build_transfer_model()
    feature_extractor.trainable = False
    tail.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"Trainable parameters: {model.count_params():,} total, "
          f"{sum(np.prod(w.shape) for w in model.trainable_weights):,} trainable")
    head_history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=args.head_epochs,
        shuffle=False,
        class_weight=class_weights,
        callbacks=make_callbacks(checkpoint_head),
        verbose=1,
    )
    transfer_history["head"] = history_to_dict(head_history)
    phase1_best = max(transfer_history["head"].get("val_accuracy", [0.0]))
    print(f"Best head-phase validation accuracy: {phase1_best:.4f}")

    # ------------------------------------------------------------------
    # Fine-tune: unfreeze the last 30 layers of MobileNetV2 (not BatchNorm).
    # ------------------------------------------------------------------
    if args.finetune_epochs > 0:
        print("\n=== MobileNetV2 transfer learning: fine-tuning ===")
        feature_extractor.trainable = True
        tail.trainable = True
        for layer in base.layers:
            layer.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False
        for layer in base.layers:
            if isinstance(layer, layers.BatchNormalization):
                layer.trainable = False
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        print(
            "Trainable weights after unfreezing:",
            f"{sum(np.prod(w.shape) for w in model.trainable_weights):,}",
        )
        fine_history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=args.finetune_epochs,
            shuffle=False,
            class_weight=class_weights,
            callbacks=make_callbacks(checkpoint_finetune),
            verbose=1,
        )
        transfer_history["finetune"] = history_to_dict(fine_history)
        phase2_best = max(transfer_history["finetune"].get("val_accuracy", [0.0]))
        print(f"Best fine-tune validation accuracy: {phase2_best:.4f}")

    # ------------------------------------------------------------------
    # Choose the best transfer checkpoint based on real validation accuracy.
    # ------------------------------------------------------------------
    if phase2_best >= phase1_best and checkpoint_finetune.exists():
        selected_checkpoint = checkpoint_finetune
        selected_phase = "fine-tune"
        selected_val = phase2_best
    else:
        selected_checkpoint = checkpoint_head
        selected_phase = "head"
        selected_val = phase1_best
    if not selected_checkpoint.exists():
        raise SystemExit("No checkpoint was saved. Check the training output.")

    final_model_path = output_dir / "ecosort_model.keras"
    shutil.copyfile(selected_checkpoint, final_model_path)
    for temporary in (checkpoint_head, checkpoint_finetune):
        if temporary.exists():
            temporary.unlink()
    print(
        f"\nSelected the {selected_phase} checkpoint "
        f"(validation accuracy {selected_val:.4f}) -> {final_model_path.name}"
    )

    # ------------------------------------------------------------------
    # Model 2: custom CNN baseline.
    # ------------------------------------------------------------------
    baseline_history = {}
    baseline_val = None
    if args.baseline_epochs > 0:
        print("\n=== Custom CNN baseline ===")
        baseline = build_baseline_model()
        baseline.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        print(f"Baseline parameters: {baseline.count_params():,}")
        base_history = baseline.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=args.baseline_epochs,
            shuffle=False,
            class_weight=class_weights,
            callbacks=make_callbacks(baseline_path),
            verbose=1,
        )
        baseline_history = history_to_dict(base_history)
        baseline_val = max(baseline_history.get("val_accuracy", [0.0]))
        print(f"Best baseline validation accuracy: {baseline_val:.4f}")

    # ------------------------------------------------------------------
    # Save metadata. evaluate.py flips demo=False only after a real test run.
    # ------------------------------------------------------------------
    (output_dir / "class_names.json").write_text(
        json.dumps(list(CLASS_NAMES), indent=2), encoding="utf-8"
    )
    elapsed = round(time.time() - started, 1)
    training_record = {
        "transfer": transfer_history,
        "baseline": baseline_history,
        "settings": {
            "seed": SEED,
            "batch_size": args.batch_size,
            "head_epochs": args.head_epochs,
            "finetune_epochs": args.finetune_epochs,
            "baseline_epochs": args.baseline_epochs,
            "image_size": list(IMAGE_SIZE),
            "class_weights": class_weights,
            "selected_phase": selected_phase,
            "selected_val_accuracy": round(selected_val, 6),
            "baseline_val_accuracy": (
                round(baseline_val, 6) if baseline_val is not None else None
            ),
            "training_seconds": elapsed,
        },
    }
    (output_dir / "training_history.json").write_text(
        json.dumps(training_record, indent=2), encoding="utf-8"
    )

    meta = {
        "demo": True,
        "metrics_available": False,
        "model_name": "MobileNetV2 transfer learning",
        "model_file": "ecosort_model.keras",
        "baseline_file": "baseline_cnn.keras",
        "input_size": list(IMAGE_SIZE),
        "classes": list(CLASS_NAMES),
        "gradcam_layer": "mobilenet_features",
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selected_phase": selected_phase,
        "validation_accuracy": round(selected_val, 6),
        "note": (
            "DEMO status is shown until evaluate.py has run on the held-out "
            "test set and written metrics.json."
        ),
    }
    (output_dir / "model_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    print("\nTraining complete.")
    print(f"Model saved       : {final_model_path}")
    print(f"Baseline saved    : {baseline_path if baseline_path.exists() else 'not trained'}")
    print(f"Training history  : {output_dir / 'training_history.json'}")
    print(f"Training time     : {elapsed:.1f}s")
    print("Next step         : python ml/evaluate.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
