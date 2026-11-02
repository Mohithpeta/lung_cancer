"""Standalone preprocessing pipeline for TabICL-XG lung-cancer detection.

Run from the project root:
    python src/preprocessing/preprocessing.py

Input:
    data/lung_cancer_dataset.csv

Outputs:
    data/processed/train.csv
    data/processed/validation.csv
    data/processed/test.csv
    data/processed/preprocessing_metadata.json
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "lung_cancer_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15

BEHAVIORAL_FEATURES = [
    "pack_years",
    "secondhand_smoke_exposure",
]

ENVIRONMENTAL_FEATURES = [
    "radon_exposure",
    "asbestos_exposure",
]

CLINICAL_FEATURES = [
    "age",
    "copd_diagnosis",
    "family_history",
    "gender",
]

TARGET_COLUMN = "lung_cancer"

REQUIRED_COLUMNS = (
    BEHAVIORAL_FEATURES
    + ENVIRONMENTAL_FEATURES
    + CLINICAL_FEATURES
    + [TARGET_COLUMN]
)

NUMERIC_FEATURES = ["pack_years", "age"]
CATEGORICAL_FEATURES = [
    "secondhand_smoke_exposure",
    "radon_exposure",
    "asbestos_exposure",
    "copd_diagnosis",
    "family_history",
    "gender",
]


# ---------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------

def load_data(path: Path) -> pd.DataFrame:
    """Load the raw dataset and validate required columns."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place the CSV at data/lung_cancer_dataset.csv."
        )

    df = pd.read_csv(path)

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Select only the eight input features and the target.
    # This excludes patient identifiers and alcohol consumption.
    df = df[REQUIRED_COLUMNS].copy()

    # Convert numeric features explicitly.
    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Normalize categorical text and treat empty strings as missing.
    for column in CATEGORICAL_FEATURES + [TARGET_COLUMN]:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
        )

    if df[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"{TARGET_COLUMN} contains missing or invalid values. "
            "Resolve these records before supervised training."
        )

    observed_targets = set(df[TARGET_COLUMN].unique())
    if observed_targets != {"Yes", "No"}:
        raise ValueError(
            f"Expected target labels 'Yes' and 'No'; "
            f"found {sorted(observed_targets)}."
        )

    if df.empty:
        raise ValueError("The dataset contains no records.")

    return df


# ---------------------------------------------------------------------
# Patient-level train / validation / test split
# ---------------------------------------------------------------------

def split_data(df: pd.DataFrame):
    """Create stratified, disjoint training, validation, and test sets."""
    train_val, test = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )

    relative_validation_size = VALIDATION_SIZE / (1.0 - TEST_SIZE)

    train, validation = train_test_split(
        train_val,
        test_size=relative_validation_size,
        random_state=RANDOM_STATE,
        stratify=train_val[TARGET_COLUMN],
    )

    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )


# ---------------------------------------------------------------------
# Feature preprocessing
# ---------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """Build transformations fitted exclusively on training data."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    dtype="float32",
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def preprocess_splits(train, validation, test):
    """Fit on training features and transform all three splits."""
    preprocessor = build_preprocessor()

    X_train = preprocessor.fit_transform(train[REQUIRED_COLUMNS[:-1]])
    X_validation = preprocessor.transform(validation[REQUIRED_COLUMNS[:-1]])
    X_test = preprocessor.transform(test[REQUIRED_COLUMNS[:-1]])

    feature_names = preprocessor.get_feature_names_out().tolist()

    def assemble_split(X, df):
        features = pd.DataFrame(X, columns=feature_names)
        features = features.astype("float32")

        # Restore the original three modality groupings.
        behavioral_columns = [
            name for name in feature_names
            if name in BEHAVIORAL_FEATURES
            or name.startswith("secondhand_smoke_exposure_")
        ]
        environmental_columns = [
            name for name in feature_names
            if name.startswith("radon_exposure_")
            or name.startswith("asbestos_exposure_")
        ]
        clinical_columns = [
            name for name in feature_names
            if name == "age"
            or name.startswith("copd_diagnosis_")
            or name.startswith("family_history_")
            or name.startswith("gender_")
        ]

        # Prefixes make each modality identifiable in saved CSV files.
        features = features.rename(
            columns={
                column: f"behavioral__{column}"
                for column in behavioral_columns
            }
            | {
                column: f"environmental__{column}"
                for column in environmental_columns
            }
            | {
                column: f"clinical__{column}"
                for column in clinical_columns
            }
        )

        # Keep Y separate from all model input features.
        target = LabelEncoder().fit(["No", "Yes"]).transform(
            df[TARGET_COLUMN].tolist()
        )

        features["lung_cancer"] = target.astype("int64")
        return features

    return (
        assemble_split(X_train, train),
        assemble_split(X_validation, validation),
        assemble_split(X_test, test),
        preprocessor,
    )


# ---------------------------------------------------------------------
# Output generation
# ---------------------------------------------------------------------

def save_outputs(train, validation, test, preprocessor):
    """Save processed splits and fitted preprocessing metadata."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train.to_csv(OUTPUT_DIR / "train.csv", index=False)
    validation.to_csv(OUTPUT_DIR / "validation.csv", index=False)
    test.to_csv(OUTPUT_DIR / "test.csv", index=False)

    metadata = {
        "input_features": {
            "behavioral": BEHAVIORAL_FEATURES,
            "environmental": ENVIRONMENTAL_FEATURES,
            "clinical": CLINICAL_FEATURES,
        },
        "target": TARGET_COLUMN,
        "target_encoding": {"No": 0, "Yes": 1},
        "excluded_columns": ["patient_id", "alcohol_consumption"],
        "split_sizes": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "validation_size": VALIDATION_SIZE,
        "transformed_features": preprocessor.get_feature_names_out().tolist(),
        "transformed_feature_count": len(
            preprocessor.get_feature_names_out()
        ),
        "note": (
            "Numeric features are standardized and categorical features "
            "are one-hot encoded. Transformations are fitted on training "
            "data only. No imbalance handling is applied."
        ),
    }

    with open(
        OUTPUT_DIR / "preprocessing_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\nPreprocessing completed successfully.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Training records:   {len(train):,}")
    print(f"Validation records: {len(validation):,}")
    print(f"Test records:       {len(test):,}")
    print(f"Transformed features: {metadata['transformed_feature_count']}")


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------

def main():
    df = load_data(INPUT_PATH)

    train, validation, test = split_data(df)

    train_processed, validation_processed, test_processed, preprocessor = (
        preprocess_splits(train, validation, test)
    )

    save_outputs(
        train_processed,
        validation_processed,
        test_processed,
        preprocessor,
    )


if __name__ == "__main__":
    main()