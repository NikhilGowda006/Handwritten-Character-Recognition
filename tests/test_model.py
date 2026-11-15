"""Tests for the CNN architecture, saved weights, and EMNIST data access."""

import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from torchvision.datasets import EMNIST

from model.cnn_model import CharacterCNN, NUM_CLASSES
from training.train import DEFAULT_DATA_DIR, build_transform
from utils.prediction import load_model
from utils.labels import EMNIST_BALANCED_LABELS
from utils.preprocessing import preprocess_image

EMNIST_RAW_DIR = DEFAULT_DATA_DIR / "EMNIST" / "raw"
EMNIST_TRAIN_FILES = (
    EMNIST_RAW_DIR / "emnist-balanced-train-images-idx3-ubyte",
    EMNIST_RAW_DIR / "emnist-balanced-train-labels-idx1-ubyte",
)
HAS_LOCAL_EMNIST = all(path.is_file() for path in EMNIST_TRAIN_FILES)


class ModelTests(unittest.TestCase):
    def test_cnn_returns_47_class_logits(self) -> None:
        model = CharacterCNN().eval()
        with torch.inference_mode():
            logits = model(torch.zeros(2, 1, 28, 28))

        self.assertEqual(tuple(logits.shape), (2, NUM_CLASSES))

    def test_trained_checkpoint_loads_on_available_device(self) -> None:
        model = load_model()
        expected_device = "cuda" if torch.cuda.is_available() else "cpu"

        self.assertFalse(model.training)
        self.assertEqual(next(model.parameters()).device.type, expected_device)

    def test_missing_checkpoint_reports_file_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            missing_path = Path(temporary_directory) / "missing.pth"

            with self.assertRaisesRegex(FileNotFoundError, "Trained model not found"):
                load_model(missing_path)

    @unittest.skipUnless(HAS_LOCAL_EMNIST, "EMNIST data has not been downloaded")
    def test_emnist_balanced_training_split_loads_without_download(self) -> None:
        dataset = EMNIST(
            root=str(DEFAULT_DATA_DIR),
            split="balanced",
            train=True,
            download=False,
            transform=build_transform(),
        )
        image, label = dataset[0]

        self.assertGreater(len(dataset), 100_000)
        self.assertEqual(tuple(image.shape), (1, 28, 28))
        self.assertGreaterEqual(int(label), 0)
        self.assertLess(int(label), NUM_CLASSES)
        self.assertEqual(tuple(dataset.classes), EMNIST_BALANCED_LABELS)

    @unittest.skipUnless(HAS_LOCAL_EMNIST, "EMNIST data has not been downloaded")
    def test_training_transform_matches_prediction_preprocessing(self) -> None:
        transformed_dataset = EMNIST(
            root=str(DEFAULT_DATA_DIR),
            split="balanced",
            train=False,
            download=False,
            transform=build_transform(),
        )
        raw_dataset = EMNIST(
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
                for index, label in enumerate(raw_dataset.targets.tolist())
                if label == class_index
            )
            training_tensor, _ = transformed_dataset[sample_index]
            raw_image, _ = raw_dataset[sample_index]
            upright_image = raw_image.transpose(Image.Transpose.TRANSPOSE)
            prediction_tensor, _ = preprocess_image(upright_image)

            torch.testing.assert_close(
                training_tensor,
                prediction_tensor,
                rtol=0,
                atol=0,
            )
            cached_tensor = (
                (training_tensor + 1.0)
                .mul(127.5)
                .round()
                .to(torch.uint8)
                .float()
                .div(127.5)
                .sub(1.0)
            )
            torch.testing.assert_close(
                cached_tensor,
                prediction_tensor,
                rtol=0,
                atol=2e-6,
            )


if __name__ == "__main__":
    unittest.main()
