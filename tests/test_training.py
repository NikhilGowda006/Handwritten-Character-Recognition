"""Tests for training-only image augmentation."""

import unittest

import torch

from training.train import apply_stroke_width_augmentation


class TrainingAugmentationTests(unittest.TestCase):
    def test_stroke_width_operations_thicken_thin_or_preserve_images(self) -> None:
        images = torch.full((3, 1, 7, 7), -1.0)
        images[:, :, 2:5, 2:5] = 1.0
        operations = torch.tensor([0, 1, 2])

        augmented = apply_stroke_width_augmentation(images, operations)

        self.assertEqual(tuple(augmented.shape), tuple(images.shape))
        torch.testing.assert_close(augmented[0], images[0])
        self.assertEqual(int((augmented[1] > 0).sum()), 25)
        self.assertEqual(int((augmented[2] > 0).sum()), 1)


if __name__ == "__main__":
    unittest.main()
