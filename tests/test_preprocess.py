"""Tests for the image preprocessing shared by the app and training."""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from ml.preprocess import (
    ImageValidationError,
    inspect_image_bytes,
    load_rgb,
    make_preview_jpeg,
    normalise_rgb,
    preprocess_image,
    resize_rgb,
)


def test_inspect_accepts_jpeg_png_and_webp(jpeg_bytes, png_bytes, webp_bytes):
    assert inspect_image_bytes(jpeg_bytes) == "JPEG"
    assert inspect_image_bytes(png_bytes) == "PNG"
    assert inspect_image_bytes(webp_bytes) == "WEBP"


def test_inspect_rejects_empty_bytes():
    with pytest.raises(ImageValidationError, match="empty"):
        inspect_image_bytes(b"")


def test_inspect_rejects_non_image_bytes():
    with pytest.raises(ImageValidationError, match="supported image"):
        inspect_image_bytes(b"this is not an image, just plain text")


def test_inspect_rejects_gif():
    buffer = io.BytesIO()
    Image.new("P", (64, 64)).save(buffer, format="GIF")
    with pytest.raises(ImageValidationError, match="type isn't supported"):
        inspect_image_bytes(buffer.getvalue())


def test_inspect_rejects_tiny_images():
    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(buffer, format="JPEG")
    with pytest.raises(ImageValidationError, match="too small"):
        inspect_image_bytes(buffer.getvalue())


def test_inspect_enforces_byte_limit(jpeg_bytes):
    with pytest.raises(ImageValidationError, match="larger than 5 MB"):
        inspect_image_bytes(jpeg_bytes, max_bytes=10)


def test_inspect_enforces_pixel_limit():
    buffer = io.BytesIO()
    Image.new("RGB", (300, 300), "green").save(buffer, format="PNG")
    with pytest.raises(ImageValidationError, match="too many pixels"):
        inspect_image_bytes(buffer.getvalue(), max_pixels=10_000)


def test_load_rgb_converts_bgr_to_rgb(png_bytes):
    # OpenCV stores colour images as BGR. A pure-red PNG must still arrive as
    # red in the RGB array the model receives.
    rgb = load_rgb(png_bytes)
    assert rgb.shape == (64, 64, 3)
    assert int(rgb[0, 0, 0]) == 20
    assert int(rgb[0, 0, 1]) == 90
    assert int(rgb[0, 0, 2]) == 60


def test_preprocess_image_has_expected_shape_and_range(jpeg_bytes):
    image = preprocess_image(jpeg_bytes)
    assert image.shape == (224, 224, 3)
    assert image.dtype == np.float32
    assert float(image.min()) >= 0.0
    assert float(image.max()) <= 1.0


def test_resize_and_normalise_are_deterministic():
    source = np.zeros((40, 80, 3), dtype=np.uint8)
    source[:, 40:] = 255
    resized = resize_rgb(source, (32, 16))
    assert resized.shape == (16, 32, 3)
    normalised = normalise_rgb(resized)
    assert normalised.dtype == np.float32
    assert normalised.min() == 0.0
    assert normalised.max() == 1.0
    # The left half was black and the right half white.
    assert normalised[:, :12].max() == 0.0
    assert normalised[:, 20:].min() == 1.0


def test_make_preview_returns_small_jpeg(jpeg_bytes):
    preview = make_preview_jpeg(jpeg_bytes, max_side=64)
    assert preview.startswith(b"\xff\xd8")
    with Image.open(io.BytesIO(preview)) as image:
        assert image.format == "JPEG"
        assert max(image.size) <= 64


def test_load_rgb_rejects_unreadable_data():
    with pytest.raises(ImageValidationError, match="couldn't read"):
        load_rgb(b"not an image")
