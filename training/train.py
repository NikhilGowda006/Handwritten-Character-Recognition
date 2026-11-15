"""Train the character CNN on EMNIST Balanced."""

import argparse
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms

from model.cnn_model import CharacterCNN
from utils.preprocessing import preprocess_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = PROJECT_ROOT / "model" / "trained_model.pth"
HISTORY_PATH = PROJECT_ROOT / "results" / "training_history.json"


class EMNISTOrientation:
    """Transpose EMNIST's stored images so characters are upright."""

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.transpose(Image.Transpose.TRANSPOSE)


class EMNISTModelPreprocessing:
    """Apply the same image pipeline used by interactive predictions."""

    def __call__(self, image: Image.Image) -> torch.Tensor:
        tensor, _ = preprocess_image(image)
        return tensor


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            EMNISTOrientation(),
            EMNISTModelPreprocessing(),
        ]
    )


def apply_stroke_width_augmentation(
    images: torch.Tensor,
    operations: torch.Tensor,
) -> torch.Tensor:
    """Apply per-image morphology: 0=unchanged, 1=thicken, 2=thin."""
    dilated = F.max_pool2d(
        F.pad(images, (1, 1, 1, 1), value=-1.0),
        kernel_size=3,
        stride=1,
    )
    eroded = -F.max_pool2d(
        F.pad(-images, (1, 1, 1, 1), value=1.0),
        kernel_size=3,
        stride=1,
    )
    dilation_mask = (operations == 1)[:, None, None, None]
    erosion_mask = (operations == 2)[:, None, None, None]
    return torch.where(
        dilation_mask,
        dilated,
        torch.where(erosion_mask, eroded, images),
    )


def create_data_loaders(
    data_dir: Path,
    batch_size: int,
    validation_fraction: float,
    seed: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    """Download EMNIST training data and split it into train and validation sets."""
    emnist_dataset = datasets.EMNIST(
        root=str(data_dir),
        split="balanced",
        train=True,
        download=True,
        transform=build_transform(),
    )
    print("Preprocessing EMNIST training images once...")
    images = torch.empty(
        (len(emnist_dataset), 1, 28, 28),
        dtype=torch.uint8,
    )
    for index in range(len(emnist_dataset)):
        normalized_image, _ = emnist_dataset[index]
        images[index] = (
            (normalized_image + 1.0)
            .mul(127.5)
            .round()
            .to(torch.uint8)
        )
    training_dataset = TensorDataset(
        images,
        emnist_dataset.targets.to(dtype=torch.long),
    )
    validation_size = int(len(training_dataset) * validation_fraction)
    training_size = len(training_dataset) - validation_size
    train_subset, validation_subset = random_split(
        training_dataset,
        [training_size, validation_size],
        generator=torch.Generator().manual_seed(seed),
    )
    loader_options = {
        "batch_size": batch_size,
        "num_workers": 0,
        "pin_memory": device.type == "cuda",
    }
    return (
        DataLoader(train_subset, shuffle=True, **loader_options),
        DataLoader(validation_subset, shuffle=False, **loader_options),
    )


def run_epoch(
    model: CharacterCNN,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """Run one training or validation epoch and return mean loss and accuracy."""
    model.train(optimizer is not None)
    total_loss = 0.0
    correct = 0
    sample_count = 0

    for images, labels in data_loader:
        images = images.to(device, non_blocking=True)
        if images.dtype == torch.uint8:
            images = images.float().div_(127.5).sub_(1.0)
        labels = labels.to(device, non_blocking=True)

        if optimizer is not None:
            operation_probability = torch.rand(images.size(0), device=device)
            operations = torch.where(
                operation_probability < 0.15,
                1,
                torch.where(operation_probability < 0.30, 2, 0),
            )
            images = apply_stroke_width_augmentation(images, operations)
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(optimizer is not None):
            logits = model(images)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        sample_count += batch_size

    return total_loss / sample_count, correct / sample_count


def save_training_plots(history: dict[str, list[float]]) -> None:
    """Save accuracy and loss plots using the measured training history."""
    epochs = range(1, len(history["train_accuracy"]) + 1)
    results_dir = PROJECT_ROOT / "results"

    figure, axis = plt.subplots()
    axis.plot(epochs, history["train_accuracy"], label="Training accuracy")
    axis.plot(epochs, history["validation_accuracy"], label="Validation accuracy")
    axis.set(
        title="Training and Validation Accuracy",
        xlabel="Epoch",
        ylabel="Accuracy",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(results_dir / "accuracy.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots()
    axis.plot(epochs, history["train_loss"], label="Training loss")
    axis.plot(epochs, history["validation_loss"], label="Validation loss")
    axis.set(
        title="Training and Validation Loss",
        xlabel="Epoch",
        ylabel="Loss",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(results_dir / "loss.png", dpi=160)
    plt.close(figure)


def train(
    data_dir: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    validation_fraction: float,
    seed: int,
    initial_checkpoint: Path | None = None,
) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, validation_loader = create_data_loaders(
        data_dir=data_dir,
        batch_size=batch_size,
        validation_fraction=validation_fraction,
        seed=seed,
        device=device,
    )
    model = CharacterCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }
    best_validation_accuracy = -1.0
    initial_validation_accuracy = None

    if initial_checkpoint is not None:
        if not initial_checkpoint.is_file():
            raise FileNotFoundError(
                f"Initial checkpoint not found at {initial_checkpoint}."
            )
        model.load_state_dict(
            torch.load(initial_checkpoint, map_location=device, weights_only=True)
        )
        _, initial_validation_accuracy = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
        )
        best_validation_accuracy = initial_validation_accuracy
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), MODEL_PATH)
        print(
            "Initial checkpoint validation accuracy: "
            f"{initial_validation_accuracy:.4%}"
        )

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        validation_loss, validation_accuracy = run_epoch(
            model, validation_loader, criterion, device
        )
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"train loss: {train_loss:.4f}, train accuracy: {train_accuracy:.4%} | "
            f"validation loss: {validation_loss:.4f}, "
            f"validation accuracy: {validation_accuracy:.4%}"
        )

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_PATH)

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(
        json.dumps(
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "validation_fraction": validation_fraction,
                "seed": seed,
                "initial_checkpoint": (
                    initial_checkpoint.name
                    if initial_checkpoint is not None
                    else None
                ),
                "initial_validation_accuracy": initial_validation_accuracy,
                "best_validation_accuracy": best_validation_accuracy,
                **history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    save_training_plots(history)
    print(f"Best validation accuracy: {best_validation_accuracy:.4%}")
    print(f"Saved model weights to: {MODEL_PATH}")
    print(f"Saved training history to: {HISTORY_PATH}")
    print(f"Saved accuracy and loss plots to: {HISTORY_PATH.parent}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--initial-checkpoint",
        type=Path,
        help="Optional trained weights to fine-tune with the current preprocessing.",
    )
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be greater than 0")
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    train(
        data_dir=arguments.data_dir,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        validation_fraction=arguments.validation_fraction,
        seed=arguments.seed,
        initial_checkpoint=arguments.initial_checkpoint,
    )
