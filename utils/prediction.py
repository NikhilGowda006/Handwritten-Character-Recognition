"""Model loading and top-three handwritten character prediction."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from model.cnn_model import CharacterCNN
from utils.labels import EMNIST_BALANCED_LABELS
from utils.preprocessing import ImageInput, preprocess_image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "trained_model.pth"


@dataclass(frozen=True)
class ClassPrediction:
    """A predicted class and its softmax probability."""

    character: str
    confidence: float
    class_index: int


@dataclass(frozen=True)
class PredictionResult:
    """The top predictions and the preprocessed image used for inference."""

    top_predictions: tuple[ClassPrediction, ...]
    processed_image: np.ndarray


def load_model(model_path: Path = DEFAULT_MODEL_PATH) -> CharacterCNN:
    """Load trained weights onto the available device or report a clear error."""
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. "
            "Train it with `python -m training.train` before predicting."
        )
    if len(EMNIST_BALANCED_LABELS) != 47:
        raise RuntimeError("The EMNIST Balanced label mapping must contain 47 classes.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CharacterCNN().to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_image(
    source: ImageInput,
    model: CharacterCNN,
) -> PredictionResult:
    """Preprocess one image and return its three most probable EMNIST classes."""
    input_tensor, processed_image = preprocess_image(source)
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        logits = model(input_tensor.unsqueeze(0).to(device))
        probabilities = torch.softmax(logits, dim=1)
        top_probabilities, top_indices = torch.topk(
            probabilities[0],
            k=3,
        )

    predictions = tuple(
        ClassPrediction(
            character=EMNIST_BALANCED_LABELS[class_index],
            confidence=float(confidence),
            class_index=class_index,
        )
        for confidence, class_index in zip(
            top_probabilities.cpu().tolist(),
            top_indices.cpu().tolist(),
        )
    )
    return PredictionResult(
        top_predictions=predictions,
        processed_image=processed_image,
    )
