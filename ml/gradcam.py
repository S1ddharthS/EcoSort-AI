"""Grad-CAM explainability overlay for EcoSort AI.

Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the image
regions that had the strongest influence on the predicted class.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization" (2017).

Keras stores nested models as single layers. To obtain a gradient for an
intermediate feature map, this module rebuilds the model by calling the loaded
layers in order (weights are shared, so the result is identical) and marks the
feature layer. This also works after saving and reloading the .keras file.
"""

from __future__ import annotations

import base64

import cv2
import numpy as np
import tensorflow as tf


def _pick_feature_layer(model: tf.keras.Model, preferred_name: str | None = None):
    """Choose a spatial feature layer for Grad-CAM.

    Preference order:
    1. the layer recorded in model_meta.json;
    2. the named transfer-learning feature extractor;
    3. the last top-level Conv2D layer (custom CNN);
    4. the first nested 4-D layer.
    """
    if preferred_name:
        try:
            layer = model.get_layer(preferred_name)
            if len(layer.output_shape) == 4:
                return layer
        except (ValueError, AttributeError):
            pass

    try:
        return model.get_layer("mobilenet_features")
    except ValueError:
        pass

    conv_layers = [
        layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)
    ]
    if conv_layers:
        return conv_layers[-1]

    for layer in model.layers:
        try:
            if len(layer.output_shape) == 4:
                return layer
        except (AttributeError, ValueError):
            continue
    raise ValueError("No spatial feature layer found for Grad-CAM.")


def build_grad_model(
    model: tf.keras.Model, preferred_name: str | None = None
) -> tf.keras.Model:
    """Rebuild the loaded model and expose the feature layer plus predictions."""
    feature_layer = _pick_feature_layer(model, preferred_name)
    inputs = tf.keras.Input(shape=model.input_shape[1:], name="gradcam_input")
    tensor = inputs
    features = None
    for layer in model.layers[1:]:  # skip the original InputLayer
        tensor = layer(tensor, training=False)
        if layer.name == feature_layer.name:
            features = tensor
    if features is None:
        raise ValueError("Feature layer was not reached while rebuilding the model.")
    return tf.keras.models.Model(inputs, [features, tensor], name="ecosort_gradcam")


def generate_gradcam(
    grad_model: tf.keras.Model,
    image_array: np.ndarray,
    class_index: int,
    output_size: int = 224,
) -> str | None:
    """Return a JPEG data URL with the Grad-CAM heatmap overlaid.

    Any failure returns None so a prediction can still succeed.
    """
    try:
        image_batch = np.expand_dims(image_array, axis=0).astype(np.float32)
        with tf.GradientTape() as tape:
            feature_maps, predictions = grad_model(image_batch, training=False)
            class_score = predictions[:, class_index]
        gradients = tape.gradient(class_score, feature_maps)
        if gradients is None:
            return None

        pooled_gradients = tf.reduce_mean(gradients, axis=(0, 1, 2))
        feature_maps = feature_maps[0]
        heatmap = tf.reduce_sum(feature_maps * pooled_gradients, axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        maximum = tf.reduce_max(heatmap)
        if float(maximum) <= 0:
            return None
        heatmap = (heatmap / maximum).numpy()

        heatmap = cv2.resize(
            heatmap, (output_size, output_size), interpolation=cv2.INTER_LINEAR
        )
        heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0, 1))
        coloured = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        base = cv2.resize(np.uint8(image_array * 255.0), (output_size, output_size))
        base_bgr = cv2.cvtColor(base, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(base_bgr, 0.6, coloured, 0.4, 0)
        ok, buffer = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return None
        encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:  # noqa: BLE001 - explanation is optional, never fail a prediction
        tf.get_logger().warning("Grad-CAM generation failed", exc_info=True)
        return None
