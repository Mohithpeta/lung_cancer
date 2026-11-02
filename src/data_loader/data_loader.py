"""Data loading for TabICL-XG lung-cancer detection."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
TARGET = "lung_cancer"
BATCH_SIZE = 64
NUM_WORKERS = 0  # Reliable on Windows and in notebooks.

MODALITIES = {
    "behavioral": "behavioral__",
    "environmental": "environmental__",
    "clinical": "clinical__",
}


class LungCancerDataset(Dataset):
    """Return modality-specific inputs and a separate binary target."""

    def __init__(self, csv_path):
        if not csv_path.exists():
            raise FileNotFoundError(f"Processed split not found: {csv_path}")

        df = pd.read_csv(csv_path)

        if TARGET not in df.columns:
            raise ValueError(f"Missing target column: {TARGET}")
        if df.empty:
            raise ValueError(f"Dataset is empty: {csv_path}")
        if df.isna().any().any():
            raise ValueError(f"Missing values found in processed data: {csv_path}")

        self.features = {}
        for modality, prefix in MODALITIES.items():
            columns = [c for c in df.columns if c.startswith(prefix)]
            if not columns:
                raise ValueError(
                    f"No columns found for {modality!r} using prefix {prefix!r}"
                )

            values = df[columns].to_numpy(dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite values found in {modality} features")

            self.features[modality] = torch.from_numpy(values)
            setattr(self, f"{modality}_columns", columns)

        labels = df[TARGET].to_numpy()
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("Target must contain only 0 (No) and 1 (Yes).")

        self.targets = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return {
            "behavioral": self.features["behavioral"][index],
            "environmental": self.features["environmental"][index],
            "clinical": self.features["clinical"][index],
            "target": self.targets[index],
        }


def create_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """Load training, validation, and test splits."""
    datasets = {
        split: LungCancerDataset(DATA_DIR / f"{filename}.csv")
        for split, filename in [
            ("train", "train"),
            ("validation", "validation"),
            ("test", "test"),
        ]
    }

    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
        )
        for split, dataset in datasets.items()
    }

    # Ensure the same modality layout across all splits.
    for split in ("validation", "test"):
        for modality in MODALITIES:
            if (
                datasets[split].features[modality].shape[1]
                != datasets["train"].features[modality].shape[1]
            ):
                raise ValueError(
                    f"{modality} feature count differs in {split} and train."
                )

    return loaders


if __name__ == "__main__":
    loaders = create_dataloaders()

    for split, loader in loaders.items():
        batch = next(iter(loader))
        print(f"\n{split}: {len(loader.dataset)} samples")
        for name, tensor in batch.items():
            print(f"  {name}: {tuple(tensor.shape)}")