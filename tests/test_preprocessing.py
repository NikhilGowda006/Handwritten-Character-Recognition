"""Tests for uploaded-image and drawing-canvas preprocessing."""

import unittest

import cv2
import numpy as np
from PIL import Image, ImageDraw

from utils.preprocessing import load_image, preprocess_image


def make_light_background_character() -> Image.Image:
    image = Image.new("RGB", (64, 64), "white")
    draw = ImageDraw.Draw(image)
    draw.line((20, 52, 32, 12, 44, 52), fill="black", width=7)
    draw.line((25, 38, 39, 38), fill="black", width=6)
    return image


class PreprocessingTests(unittest.TestCase):
    def test_light_background_image_is_inverted_and_normalized(self) -> None:
        tensor, preview = preprocess_image(make_light_background_character())

        self.assertEqual(tuple(tensor.shape), (1, 28, 28))
        self.assertEqual(preview.shape, (28, 28))
        self.assertEqual(preview.dtype, np.uint8)
        self.assertGreater(np.count_nonzero(preview), 0)
        self.assertGreaterEqual(float(tensor.min()), -1.0)
        self.assertLessEqual(float(tensor.max()), 1.0)

    def test_rgba_canvas_image_is_accepted(self) -> None:
        canvas = np.zeros((64, 64, 4), dtype=np.uint8)
        canvas[:, :, 3] = 255
        cv2.line(canvas, (20, 52), (32, 12), (255, 255, 255, 255), 7)
        cv2.line(canvas, (32, 12), (44, 52), (255, 255, 255, 255), 7)

        tensor, preview = preprocess_image(canvas)

        self.assertEqual(tuple(tensor.shape), (1, 28, 28))
        self.assertEqual(preview.shape, (28, 28))
        self.assertGreater(np.count_nonzero(preview), 0)

    def test_blank_image_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "No handwritten character"):
            preprocess_image(Image.new("RGB", (32, 32), "white"))

    def test_corrupt_upload_is_rejected_with_actionable_message(self) -> None:
        with self.assertRaisesRegex(ValueError, "Upload a valid PNG or JPG"):
            load_image(b"not an image")


if __name__ == "__main__":
    unittest.main()
