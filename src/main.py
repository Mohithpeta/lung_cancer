"""Main entry point for the TabICL-XG lung-cancer pipeline.

Run from the project root:
    python -m src.main
"""

from src.training.trainer import TrainingConfig, train_model


def main() -> None:
    """Run the training pipeline."""
    config = TrainingConfig(
        epochs=100,
        learning_rate=1e-3,
        weight_decay=1e-4,
        patience=10,
        seed=42,
    )

    print("=" * 60)
    print("TabICL-XG Lung Cancer Classification")
    print("=" * 60)
    print("Starting model training...")

    model = train_model(config)

    print("\nTraining completed successfully.")
    print(f"Model architecture: {model.__class__.__name__}")
    print("Best checkpoint: models/tabiclxg_best.pt")
    print("Training history: outputs/training/training_history.csv")


if __name__ == "__main__":
    main()