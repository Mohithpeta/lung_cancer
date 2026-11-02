"""Evaluation pipeline for the TabICL-X lung-cancer classifier.

Run from the project root:
    python -m src.evaluation.evaluate

Inputs:
    models/tabiclxg_best.pt
    Test DataLoader from create_dataloaders()

Outputs:
    outputs/evaluation/test_metrics.json
    outputs/evaluation/confusion_matrix.csv
    outputs/evaluation/test_predictions.csv
    outputs/evaluation/classification_report.json
"""

import csv
import json
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data_loader.data_loader import create_dataloaders
from src.model.model import TabICLXG


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = PROJECT_ROOT / "models" / "tabiclxg_best.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"

PREDICTIONS_PATH = OUTPUT_DIR / "test_predictions.csv"
METRICS_PATH = OUTPUT_DIR / "test_metrics.json"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.csv"
CLASSIFICATION_REPORT_PATH = OUTPUT_DIR / "classification_report.json"

CLASS_NAMES = ["No", "Yes"]
DEFAULT_THRESHOLD = 0.5


def get_device() -> torch.device:
    """Select an available evaluation device."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> dict:
    """Load a trusted locally generated training checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}\n"
            "Run the training pipeline first."
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except TypeError:
        # Compatibility with PyTorch versions without weights_only.
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    required_keys = {
        "model_state_dict",
        "input_dims",
        "config",
    }

    missing = required_keys - set(checkpoint)
    if missing:
        raise KeyError(
            f"Checkpoint is missing required entries: {sorted(missing)}"
        )

    return checkpoint


def build_model(
    checkpoint: dict,
    device: torch.device,
) -> TabICLXG:
    """Reconstruct the model using its saved architecture configuration."""
    config = checkpoint["config"]

    model = TabICLXG(
        input_dims=checkpoint["input_dims"],
        hidden_dim=config["hidden_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model


def collect_predictions(
    model: TabICLXG,
    test_loader,
    device: torch.device,
) -> tuple[list[int], list[float]]:
    """Collect ground-truth labels and predicted probabilities."""
    y_true = []
    y_prob = []

    with torch.inference_mode():
        for batch in test_loader:
            batch = {
                key: value.to(device)
                if isinstance(value, torch.Tensor)
                else value
                for key, value in batch.items()
            }

            if "target" not in batch:
                raise KeyError("Test batches must contain a 'target' tensor.")

            targets = batch["target"].float().reshape(-1)

            if not torch.all((targets == 0) | (targets == 1)):
                raise ValueError("Test targets must contain only 0 and 1.")

            probabilities = model(batch).reshape(-1)

            if probabilities.shape != targets.shape:
                raise ValueError(
                    "Prediction and target shapes do not match: "
                    f"{tuple(probabilities.shape)} vs "
                    f"{tuple(targets.shape)}"
                )

            if not torch.isfinite(probabilities).all():
                raise FloatingPointError(
                    "Model produced non-finite predictions."
                )

            y_true.extend(targets.cpu().numpy().astype(int).tolist())
            y_prob.extend(probabilities.cpu().numpy().tolist())

    if not y_true:
        raise ValueError("The test loader produced no samples.")

    return y_true, y_prob


def calculate_metrics(
    y_true: list[int],
    y_prob: list[float],
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[dict, dict, list]:
    """Calculate binary classification metrics and confusion matrix."""

    if not 0.0 < threshold < 1.0:
        raise ValueError("Threshold must be strictly between 0 and 1.")

    y_pred = [int(probability >= threshold) for probability in y_prob]

    # Explicit class ordering: 0 = No, 1 = Yes.
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = recall_score(
        y_true,
        y_pred,
        pos_label=1,
        zero_division=0,
    )

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # ROC-AUC and AUPRC require both classes in the test set.
    if len(set(y_true)) == 2:
        roc_auc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
    else:
        roc_auc = None
        auprc = None

    metrics = {
        "test_samples": len(y_true),
        "classification_threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "sensitivity_recall": sensitivity,
        "specificity": specificity,
        "f1_score": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "roc_auc": roc_auc,
        "auprc": auprc,
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "test_class_distribution": {
            "No": int(sum(label == 0 for label in y_true)),
            "Yes": int(sum(label == 1 for label in y_true)),
        },
    }

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    return metrics, report, y_pred


def save_results(
    y_true: list[int],
    y_prob: list[float],
    y_pred: list[int],
    metrics: dict,
    report: dict,
) -> None:
    """Save metrics, predictions, and confusion matrix."""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save summary metrics.
    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    # Save detailed classification report.
    with CLASSIFICATION_REPORT_PATH.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=4)

    # Save confusion matrix with explicit labels.
    cm = metrics["confusion_matrix"]

    with CONFUSION_MATRIX_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["Actual / Predicted", "No", "Yes"])
        writer.writerow(
            ["No", cm["true_negative"], cm["false_positive"]]
        )
        writer.writerow(
            ["Yes", cm["false_negative"], cm["true_positive"]]
        )

    # Save individual predictions.
    with PREDICTIONS_PATH.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "true_label",
                "predicted_probability",
                "predicted_label",
            ]
        )

        for true, probability, prediction in zip(
            y_true, y_prob, y_pred
        ):
            writer.writerow(
                [
                    CLASS_NAMES[true],
                    f"{probability:.8f}",
                    CLASS_NAMES[prediction],
                ]
            )


def print_results(metrics: dict) -> None:
    """Display evaluation results in the terminal."""

    print("\n" + "=" * 55)
    print("TEST SET EVALUATION RESULTS")
    print("=" * 55)

    print(f"Test samples:       {metrics['test_samples']}")
    print(
        f"Threshold:          "
        f"{metrics['classification_threshold']:.2f}"
    )

    print("\nClassification Metrics")
    print("-" * 55)

    metric_names = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Sensitivity", "sensitivity_recall"),
        ("Specificity", "specificity"),
        ("F1-score", "f1_score"),
        ("ROC-AUC", "roc_auc"),
        ("AUPRC", "auprc"),
        ("Balanced Accuracy", "balanced_accuracy"),
        ("MCC", "mcc"),
    ]

    for label, key in metric_names:
        value = metrics[key]
        formatted = "N/A" if value is None else f"{value:.4f}"
        print(f"{label:<22}: {formatted}")

    cm = metrics["confusion_matrix"]

    print("\nConfusion Matrix")
    print("-" * 55)
    print(f"{'':18} Predicted No | Predicted Yes")
    print(
        f"{'Actual No':18} {cm['true_negative']:12} | "
        f"{cm['false_positive']}"
    )
    print(
        f"{'Actual Yes':18} {cm['false_negative']:12} | "
        f"{cm['true_positive']}"
    )

    print("\nTest Class Distribution")
    print("-" * 55)
    print(f"No:  {metrics['test_class_distribution']['No']}")
    print(f"Yes: {metrics['test_class_distribution']['Yes']}")

    print("\nSaved evaluation artifacts:")
    print(f"Metrics:             {METRICS_PATH}")
    print(f"Classification:      {CLASSIFICATION_REPORT_PATH}")
    print(f"Confusion matrix:    {CONFUSION_MATRIX_PATH}")
    print(f"Predictions:         {PREDICTIONS_PATH}")


def evaluate_model(
    threshold: float = DEFAULT_THRESHOLD,
) -> dict:
    """Load the best checkpoint and evaluate it on the untouched test set."""

    device = get_device()

    print("=" * 55)
    print("TabICL-XG Lung Cancer Classification")
    print("Test Set Evaluation")
    print("=" * 55)

    print(f"Device: {device}")

    # Load the best model selected using validation loss.
    checkpoint = load_checkpoint(CHECKPOINT_PATH, device)
    model = build_model(checkpoint, device)

    print(f"Checkpoint epoch: {checkpoint['epoch']}")
    print(f"Best validation loss: {checkpoint['validation_loss']:.4f}")
    print(f"Input dimensions: {checkpoint['input_dims']}")

    # Load the original, untouched test split.
    loaders = create_dataloaders()

    if "test" not in loaders:
        raise KeyError(
            "The data loader must return a 'test' DataLoader."
        )

    test_loader = loaders["test"]

    print(f"Test samples: {len(test_loader.dataset)}")
    print("\nGenerating predictions...")

    y_true, y_prob = collect_predictions(
        model=model,
        test_loader=test_loader,
        device=device,
    )

    metrics, report, y_pred = calculate_metrics(
        y_true=y_true,
        y_prob=y_prob,
        threshold=threshold,
    )

    save_results(
        y_true=y_true,
        y_prob=y_prob,
        y_pred=y_pred,
        metrics=metrics,
        report=report,
    )

    print_results(metrics)

    return metrics


if __name__ == "__main__":
    evaluate_model()