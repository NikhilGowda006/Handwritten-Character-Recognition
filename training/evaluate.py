"""Evaluate saved character CNN weights on the held-out EMNIST test split."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from model.cnn_model import CharacterCNN, NUM_CLASSES
from training.train import DEFAULT_DATA_DIR, build_transform
from utils.labels import EMNIST_BALANCED_LABELS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "model" / "trained_model.pth"
RESULTS_DIR = PROJECT_ROOT / "results"


def evaluate(data_dir: Path, model_path: Path, results_dir: Path) -> dict:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model weights not found at {model_path}. "
            "Train the model first with `python -m training.train`."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = datasets.EMNIST(
        root=str(data_dir),
        split="balanced",
        train=False,
        download=True,
        transform=build_transform(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = CharacterCNN().to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    all_targets = []
    all_predictions = []
    with torch.no_grad():
        for images, targets in test_loader:
            logits = model(images.to(device, non_blocking=True))
            all_predictions.extend(logits.argmax(dim=1).cpu().tolist())
            all_targets.extend(targets.tolist())

    labels = list(range(NUM_CLASSES))
    matrix = confusion_matrix(all_targets, all_predictions, labels=labels)
    report = classification_report(
        all_targets,
        all_predictions,
        labels=labels,
        target_names=list(EMNIST_BALANCED_LABELS),
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "test_accuracy": accuracy_score(all_targets, all_predictions),
        "test_sample_count": len(all_targets),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "evaluation.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    figure, axis = plt.subplots(figsize=(18, 16))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Purples")
    axis.set(
        title="EMNIST Balanced Test Confusion Matrix",
        xlabel="Predicted label",
        ylabel="True label",
    )
    ticks = range(NUM_CLASSES)
    axis.set_xticks(ticks, labels=EMNIST_BALANCED_LABELS, rotation=90)
    axis.set_yticks(ticks, labels=EMNIST_BALANCED_LABELS)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(results_dir / "confusion_matrix.png", dpi=160)
    plt.close(figure)

    print(f"Test accuracy: {metrics['test_accuracy']:.4%}")
    print(f"Evaluated samples: {metrics['test_sample_count']}")
    print(f"Saved evaluation results to: {results_dir}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    evaluate(
        data_dir=arguments.data_dir,
        model_path=arguments.model_path,
        results_dir=arguments.results_dir,
    )
