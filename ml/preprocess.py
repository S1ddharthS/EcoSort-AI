"""Image loading and preprocessing shared by training and the Flask app.

Pipeline (identical in both places):
    1. decode the image
    2. convert BGR -> RGB
    3. resize to 224 x 224 with OpenCV
    4. scale pixel values to [0, 1]

Keeping this in one module is what guarantees that the model sees the same
kind of input during training and at prediction time.
"""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

IMAGE_SIZE = (224, 224)
MAX_IMAGE_PIXELS = 25_000_000
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}

# Small files can be read directly. This class gives the API a clean way to
# turn validation failures into user-facing messages.
class ImageValidationError(ValueError):
    """Raised when an uploaded file is not a usable image."""


def _read_source(source: str | Path | bytes) -> bytes:
    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    raise TypeError("source must be a path or bytes")


def inspect_image_bytes(
    data: bytes,
    max_bytes: int = 5 * 1024 * 1024,
    max_pixels: int = MAX_IMAGE_PIXELS,
) -> str:
    """Validate raw image bytes and return the detected image format.

    Pillow reads only the header at this stage, so the pixel check happens
    before the full image is decoded ("decompression bomb" protection).
    """
    if not data:
        raise ImageValidationError(
            "That file is empty. Please choose a JPG, PNG or WEBP image."
        )
    if len(data) > max_bytes:
        raise ImageValidationError(
            "That image is larger than 5 MB. Please choose a smaller file."
        )

    try:
        with Image.open(io.BytesIO(data)) as image:
            fmt = (image.format or "").upper()
            width, height = image.size
    except Image.DecompressionBombError:
        raise ImageValidationError(
            "That image has too many pixels. Please use a smaller image."
        ) from None
    except (UnidentifiedImageError, OSError, ValueError):
        raise ImageValidationError(
            "That file isn't a supported image. Please upload a JPG, PNG or WEBP."
        ) from None

    if fmt not in ALLOWED_FORMATS:
        raise ImageValidationError(
            "That image type isn't supported. Please upload a JPG, PNG or WEBP."
        )
    if width < 32 or height < 32:
        raise ImageValidationError(
            "That image is too small to analyse. Please use an image at least "
            "32 x 32 pixels."
        )
    if width * height > max_pixels:
        raise ImageValidationError(
            "That image has too many pixels. Please use a smaller image."
        )
    return fmt


def load_rgb(source: str | Path | bytes) -> np.ndarray:
    """Decode an image with OpenCV and return an RGB uint8 array.

    OpenCV exposes images as BGR, so they are converted to RGB for the model.
    """
    data = _read_source(source)
    buffer = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ImageValidationError(
            "We couldn't read that image. Please try a different JPG, PNG or "
            "WEBP file."
        )
    if bgr.shape[0] < 32 or bgr.shape[1] < 32:
        raise ImageValidationError("That image is too small to analyse.")
    if bgr.shape[0] * bgr.shape[1] > MAX_IMAGE_PIXELS:
        raise ImageValidationError("That image has too many pixels to analyse.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def resize_rgb(rgb: np.ndarray, size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Resize an RGB uint8 array with OpenCV (shared by training and app)."""
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)


def normalise_rgb(rgb: np.ndarray) -> np.ndarray:
    """Scale uint8 pixels to the [0, 1] range used by the model."""
    return rgb.astype(np.float32) / 255.0


def preprocess_image(source: str | Path | bytes) -> np.ndarray:
    """Return a normalised 224 x 224 x 3 float32 array ready for the model."""
    return normalise_rgb(resize_rgb(load_rgb(source)))


def make_preview_jpeg(source: str | Path | bytes, max_side: int = 1000) -> bytes:
    """Create a reasonably sized JPEG preview for the history page.

    The original upload is never stored as-is. We store a normalised copy with
    a random name so user filenames and EXIF metadata can't leak or be executed.
    """
    rgb = load_rgb(source)
    height, width = rgb.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        rgb = cv2.resize(
            rgb,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise ImageValidationError("We couldn't prepare that image. Please try again.")
    return buffer.tobytes()
