"""Training pipeline for the TabICL-XG lung-cancer classifier.

Run from the project root:
    python -m src.training.trainer

Inputs:
    DataLoaders returned by src.data_loader.data_loader.create_dataloaders()

Outputs:
    models/tabiclxg_best.pt
    outputs/training/training_history.csv
"""
from __future__ import annotations
import csv
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from src.data_loader.data_loader import create_dataloaders
from src.model.model import TabICLXG


PROJECT_ROOT = Path.cwd()
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "tabiclxg_best.pt"
HISTORY_PATH = (
    PROJECT_ROOT / "outputs" / "training" / "training_history.csv"
)


@dataclass
class TrainingConfig:
    """Configuration for model training."""

    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    min_delta: float = 1e-4
    gradient_clip: float = 1.0
    seed: int = 42

    hidden_dim: int = 32
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible initialization and training."""
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Select an available training device."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def infer_input_dims(batch: dict) -> dict[str, int]:
    """Infer encoded feature counts from a real training batch."""
    modalities = ("behavioral", "environmental", "clinical")

    missing = set(modalities) - set(batch)
    if missing:
        raise KeyError(f"Training batch is missing modalities: {sorted(missing)}")

    dimensions = {}

    for modality in modalities:
        tensor = batch[modality]

        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
            raise ValueError(
                f"'{modality}' must be a 2D tensor; "
                "check the data loader output."
            )

        dimensions[modality] = tensor.shape[1]

    return dimensions


def move_batch(
    batch: dict,
    device: torch.device,
) -> dict:
    """Move all batch tensors to the selected device."""
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def run_epoch(
    model: TabICLXG,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip: float = 1.0,
) -> float:
    """Run one training or validation epoch and return mean loss."""

    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0

    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = move_batch(batch, device)

            if "target" not in batch:
                raise KeyError("Each batch must contain a 'target' tensor.")

            targets = batch["target"].float().reshape(-1)

            if targets.numel() == 0:
                continue

            if not torch.all((targets == 0) | (targets == 1)):
                raise ValueError("Targets must contain only binary labels 0 and 1.")

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model.forward_logits(batch)

            if logits.shape != targets.shape:
                raise ValueError(
                    f"Logits shape {tuple(logits.shape)} does not match "
                    f"target shape {tuple(targets.shape)}."
                )

            loss = criterion(logits, targets)

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    "Non-finite loss detected. Check the input data and model."
                )

            if training:
                loss.backward()

                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=gradient_clip,
                )

                optimizer.step()

            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("The data loader produced no samples.")

    return total_loss / total_samples


def save_checkpoint(
    path: Path,
    model: TabICLXG,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    input_dims: dict[str, int],
    epoch: int,
    validation_loss: float,
) -> None:
    """Save the best model and the information needed to restore it."""
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "input_dims": input_dims,
            "validation_loss": validation_loss,
        },
        path,
    )


def save_history(history: list[dict]) -> None:
    """Write per-epoch training results to CSV."""
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    with HISTORY_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "validation_loss",
                "learning_rate",
            ],
        )
        writer.writeheader()
        writer.writerows(history)


def train_model(config: TrainingConfig | None = None) -> TabICLXG:
    """Train the model, stop on validation-loss stagnation, and save the best checkpoint."""
    config = config or TrainingConfig()

    if config.epochs < 1 or config.patience < 1:
        raise ValueError("epochs and patience must both be at least 1.")

    if config.learning_rate <= 0 or config.weight_decay < 0:
        raise ValueError("Learning rate must be positive and weight decay non-negative.")

    if config.min_delta < 0 or config.gradient_clip <= 0:
        raise ValueError("min_delta must be non-negative and gradient_clip positive.")

    set_seed(config.seed)
    device = get_device()

    # The preprocessing pipeline must already have produced the split files.
    loaders = create_dataloaders()

    for split in ("train", "validation"):
        if split not in loaders:
            raise KeyError(
                f"Expected a '{split}' loader. "
                f"Available loaders: {list(loaders)}"
            )

    train_loader = loaders["train"]
    validation_loader = loaders["validation"]

    # Infer modality dimensions from the actual encoded training data.
    first_batch = next(iter(train_loader))
    input_dims = infer_input_dims(first_batch)

    model = TabICLXG(
        input_dims=input_dims,
        hidden_dim=config.hidden_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        dropout=config.dropout,
    ).to(device)

    # Model outputs raw logits here; sigmoid is applied for inference.
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(2, config.patience // 3),
    )

    best_validation_loss = float("inf")
    epochs_without_improvement = 0
    history = []

    print(f"Device: {device}")
    print(f"Input dimensions: {input_dims}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(validation_loader.dataset)}")

    for epoch in range(1, config.epochs + 1):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            gradient_clip=config.gradient_clip,
        )

        validation_loss = run_epoch(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step(validation_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": current_lr,
            }
        )

        print(
            f"Epoch {epoch:03d}/{config.epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Validation loss: {validation_loss:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        if validation_loss < best_validation_loss - config.min_delta:
            best_validation_loss = validation_loss
            epochs_without_improvement = 0

            save_checkpoint(
                path=CHECKPOINT_PATH,
                model=model,
                optimizer=optimizer,
                config=config,
                input_dims=input_dims,
                epoch=epoch,
                validation_loss=validation_loss,
            )

            print(f"  Best checkpoint saved: {CHECKPOINT_PATH}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    save_history(history)

    # Restore the best validation-loss model before returning.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Best validation loss: {best_validation_loss:.4f}")
    print(f"Training history saved: {HISTORY_PATH}")

    return model


if __name__ == "__main__":
    train_model()