"""Tests for real CNN inference and the top-three prediction contract."""

import unittest
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw
from torchvision.datasets import EMNIST

from training.train import DEFAULT_DATA_DIR
from utils.labels import EMNIST_BALANCED_LABELS
from utils.prediction import PredictionResult, load_model, predict_image

EMNIST_TEST_IMAGES = (
    DEFAULT_DATA_DIR / "EMNIST" / "raw" / "emnist-balanced-test-images-idx3-ubyte"
)
HAS_LOCAL_EMNIST_TEST = EMNIST_TEST_IMAGES.is_file()


def make_character_image() -> Image.Image:
    image = Image.new("RGB", (64, 64), "white")
    draw = ImageDraw.Draw(image)
    draw.line((20, 52, 32, 12, 44, 52), fill="black", width=7)
    draw.line((25, 38, 39, 38), fill="black", width=6)
    return image


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = load_model()

    def assert_top_three_result(self, result: PredictionResult) -> None:
        predictions = result.top_predictions
        self.assertEqual(len(predictions), 3)
        self.assertEqual(len({prediction.class_index for prediction in predictions}), 3)
        self.assertEqual(result.processed_image.shape, (28, 28))
        self.assertEqual(result.processed_image.dtype, np.uint8)

        confidences = [prediction.confidence for prediction in predictions]
        self.assertEqual(confidences, sorted(confidences, reverse=True))
        self.assertTrue(all(0.0 <= confidence <= 1.0 for confidence in confidences))
        self.assertTrue(
            all(
                prediction.character
                == EMNIST_BALANCED_LABELS[prediction.class_index]
                for prediction in predictions
            )
        )

    def test_canvas_array_runs_through_trained_model(self) -> None:
        image = make_character_image()
        canvas = np.asarray(image.convert("RGBA"))

        result = predict_image(canvas, self.model)

        self.assert_top_three_result(result)

    def test_uploaded_png_bytes_run_through_trained_model(self) -> None:
        buffer = BytesIO()
        make_character_image().save(buffer, format="PNG")

        result = predict_image(buffer.getvalue(), self.model)

        self.assert_top_three_result(result)

    @unittest.skipUnless(
        HAS_LOCAL_EMNIST_TEST,
        "EMNIST test data has not been downloaded",
    )
    def test_upright_emnist_test_samples_match_their_labels(self) -> None:
        dataset = EMNIST(
            root=str(DEFAULT_DATA_DIR),
            split="balanced",
            train=False,
            download=False,
            transform=None,
        )

        for character in ("7", "T", "L", "I", "1"):
            class_index = EMNIST_BALANCED_LABELS.index(character)
            sample_index = next(
                index
                for index, label in enumerate(dataset.targets.tolist())
                if label == class_index
            )
            stored_image, actual_label = dataset[sample_index]
            upright_image = stored_image.transpose(Image.Transpose.TRANSPOSE)
            result = predict_image(upright_image, self.model)

            self.assertEqual(EMNIST_BALANCED_LABELS[int(actual_label)], character)
            self.assertEqual(result.top_predictions[0].character, character)
            self.assert_top_three_result(result)


if __name__ == "__main__":
    unittest.main()
