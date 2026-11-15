"""Convolutional neural network for EMNIST Balanced character classification."""

from torch import nn


NUM_CLASSES = 47


class CharacterCNN(nn.Module):
    """Classify normalized 28x28 grayscale images into 47 EMNIST classes."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=64 * 7 * 7, out_features=128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.25),
            nn.Linear(in_features=128, out_features=NUM_CLASSES),
        )

    def forward(self, images):
        """Return unnormalized class scores (logits) for a batch of images."""
        features = self.features(images)
        return self.classifier(features)
