"""Image loading and preprocessing for the EMNIST-trained character model."""

from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError


ImageInput = Image.Image | np.ndarray | bytes | bytearray | Path | str
IMAGE_SIZE = 28
CHARACTER_SIZE = 20


def load_image(source: ImageInput) -> Image.Image:
    """Load supported image inputs and return a three-channel RGB image."""
    if isinstance(source, Image.Image):
        return source.convert("RGB")

    if isinstance(source, np.ndarray):
        if source.size == 0:
            raise ValueError("The provided image array is empty.")
        if source.ndim not in (2, 3):
            raise ValueError("The image array must have two or three dimensions.")
        if source.ndim == 3 and source.shape[2] not in (3, 4):
            raise ValueError("The image array must have 3 (RGB) or 4 (RGBA) channels.")
        if np.issubdtype(source.dtype, np.floating):
            if not np.isfinite(source).all():
                raise ValueError("The image array contains non-finite pixel values.")
            if source.size and source.max() <= 1:
                source = source * 255
        image_array = np.rint(np.clip(source, 0, 255)).astype(np.uint8)
        return Image.fromarray(image_array).convert("RGB")

    if isinstance(source, (bytes, bytearray)):
        try:
            with Image.open(BytesIO(source)) as image:
                return image.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise ValueError(
                "Could not read the image data. Upload a valid PNG or JPG image."
            ) from error

    if isinstance(source, (str, Path)):
        try:
            with Image.open(source) as image:
                return image.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as error:
            raise ValueError(
                f"Could not open image at {source}. Check that it is a valid image file."
            ) from error

    raise TypeError(
        "Image input must be a PIL image, NumPy array, image bytes, or file path."
    )


def _remove_small_components(binary_image: np.ndarray) -> np.ndarray:
    component_count, component_map, stats, _ = cv2.connectedComponentsWithStats(
        binary_image,
        connectivity=8,
    )
    if component_count <= 1:
        return binary_image

    component_areas = stats[1:, cv2.CC_STAT_AREA]
    largest_area = int(component_areas.max())
    minimum_area = max(2, int(largest_area * 0.005))
    keep_ids = [
        component_id
        for component_id in range(1, component_count)
        if stats[component_id, cv2.CC_STAT_AREA] >= minimum_area
    ]
    return np.where(np.isin(component_map, keep_ids), 255, 0).astype(np.uint8)


def _center_image(binary_image: np.ndarray) -> np.ndarray:
    rows, columns = np.nonzero(binary_image)
    if rows.size == 0:
        raise ValueError("No handwritten character was detected in the image.")

    left, right = int(columns.min()), int(columns.max()) + 1
    top, bottom = int(rows.min()), int(rows.max()) + 1
    cropped = binary_image[top:bottom, left:right]
    height, width = cropped.shape
    scale = CHARACTER_SIZE / max(height, width)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(
        cropped,
        (resized_width, resized_height),
        interpolation=interpolation,
    )

    canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    x_offset = (IMAGE_SIZE - resized_width) // 2
    y_offset = (IMAGE_SIZE - resized_height) // 2
    canvas[
        y_offset : y_offset + resized_height,
        x_offset : x_offset + resized_width,
    ] = resized

    moments = cv2.moments(canvas)
    if moments["m00"]:
        center_x = moments["m10"] / moments["m00"]
        center_y = moments["m01"] / moments["m00"]
        shift_x = int(round((IMAGE_SIZE - 1) / 2 - center_x))
        shift_y = int(round((IMAGE_SIZE - 1) / 2 - center_y))
        transform = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        canvas = cv2.warpAffine(
            canvas,
            transform,
            (IMAGE_SIZE, IMAGE_SIZE),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    return canvas


def preprocess_image(source: ImageInput) -> tuple[torch.Tensor, np.ndarray]:
    """Return a normalized 1x28x28 tensor and its uint8 28x28 preview."""
    rgb_image = np.asarray(load_image(source))
    grayscale = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    denoised = (
        cv2.GaussianBlur(grayscale, (3, 3), 0)
        if min(grayscale.shape) >= 3
        else grayscale
    )
    border_pixels = np.concatenate(
        (denoised[0, :], denoised[-1, :], denoised[:, 0], denoised[:, -1])
    )
    if float(np.median(border_pixels)) > 127:
        denoised = cv2.bitwise_not(denoised)

    _, binary_image = cv2.threshold(
        denoised,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )
    foreground_fraction = np.count_nonzero(binary_image) / binary_image.size
    if foreground_fraction == 0 or foreground_fraction > 0.98:
        raise ValueError("No handwritten character was detected in the image.")

    binary_image = _remove_small_components(binary_image)
    processed_image = _center_image(binary_image)
    tensor = torch.from_numpy(processed_image.astype(np.float32) / 255.0)
    tensor = (tensor - 0.5) / 0.5
    return tensor.unsqueeze(0), processed_image
