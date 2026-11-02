"""Standalone preprocessing for lung-cancer classification.

Run from the project root:
    python src/preprocessing/preprocessing.py

Input:
    data/lung_cancer_dataset.csv

Output:
    data/cleaned_lung_cancer_dataset.csv

Output columns:
    patient_id + 8 features + lung_cancer (binary target)
"""

from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "data" / "lung_cancer_dataset.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "cleaned_lung_cancer_dataset.csv"

ID_COLUMN = "patient_id"
TARGET = "lung_cancer"

FEATURES = {
    "behavioral": [
        "pack_years",
        "secondhand_smoke_exposure",
    ],
    "environmental": [
        "radon_exposure",
        "asbestos_exposure",
    ],
    "clinical": [
        "age",
        "copd_diagnosis",
        "family_history",
        "gender",
    ],
}

FEATURE_COLUMNS = [
    column
    for columns in FEATURES.values()
    for column in columns
]

OUTPUT_COLUMNS = [ID_COLUMN, *FEATURE_COLUMNS, TARGET]

NUMERIC_RANGES = {
    "pack_years": (0, 300),
    "age": (0, 120),
}

CATEGORIES = {
    "secondhand_smoke_exposure": ("No", "Yes"),
    "radon_exposure": ("Low", "Medium", "High"),
    "asbestos_exposure": ("No", "Yes"),
    "copd_diagnosis": ("No", "Yes"),
    "family_history": ("No", "Yes"),
    "gender": ("Female", "Male"),
}

MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "?"}
TARGET_MAPPING = {"yes": 1, "no": 0, "1": 1, "0": 0}


# ---------------------------------------------------------------------
# Validation and cleaning
# ---------------------------------------------------------------------

def normalize_missing_values(series: pd.Series) -> pd.Series:
    """Strip whitespace and standardize common missing-value tokens."""
    values = series.astype("string").str.strip()
    return values.mask(values.str.casefold().isin(MISSING_TOKENS))


def validate_identifier(data: pd.DataFrame) -> None:
    """Ensure patient identifiers are present and unique."""
    identifiers = normalize_missing_values(data[ID_COLUMN])

    if identifiers.isna().any():
        raise ValueError("patient_id contains missing or empty values.")

    if identifiers.duplicated().any():
        raise ValueError("patient_id contains duplicate identifiers.")


def clean_features(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize feature values and mark invalid observations as missing."""
    data = data.copy()

    # Numeric features: reject non-numeric and out-of-range values.
    for column, (lower, upper) in NUMERIC_RANGES.items():
        values = pd.to_numeric(
            normalize_missing_values(data[column]),
            errors="coerce",
        ).astype("float64")

        data[column] = values.where(values.between(lower, upper))

    # Categorical features: canonicalize case and reject unknown values.
    for column, allowed_values in CATEGORIES.items():
        lookup = {
            value.casefold(): value
            for value in allowed_values
        }

        values = normalize_missing_values(data[column]).str.casefold()
        data[column] = values.map(lookup).astype("string")

    return data


def clean_target(series: pd.Series) -> pd.Series:
    """Encode valid binary labels; reject missing or unknown labels."""
    values = normalize_missing_values(series).str.casefold()
    target = values.map(TARGET_MAPPING)

    if target.isna().any():
        invalid_count = int(target.isna().sum())
        examples = values[target.isna()].drop_duplicates().head(5).tolist()

        raise ValueError(
            f"{invalid_count} row(s) have missing or invalid target labels. "
            f"Examples: {examples}. Correct or remove these rows first."
        )

    return target.astype("int8").rename(TARGET)


# ---------------------------------------------------------------------
# Missing-value imputation
# ---------------------------------------------------------------------

def impute_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Use medians for numeric features and modes for categorical features."""
    data = data.copy()

    missing_before = data[FEATURE_COLUMNS].isna().sum()
    missing_before = missing_before[missing_before > 0]

    print("\nMissing values before imputation:")
    print(
        missing_before.to_string()
        if not missing_before.empty
        else "None"
    )

    for column in NUMERIC_RANGES:
        if data[column].notna().sum() == 0:
            raise ValueError(
                f"Cannot impute '{column}': no valid observations."
            )

        data[column] = data[column].fillna(data[column].median())

    for column in CATEGORIES:
        if data[column].notna().sum() == 0:
            raise ValueError(
                f"Cannot impute '{column}': no valid categories."
            )

        data[column] = data[column].fillna(data[column].mode().iloc[0])

    return data


# ---------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------

def preprocess() -> None:
    """Read, clean, validate, impute, and export the dataset."""
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH, keep_default_na=False)

    # Normalize headers and detect collisions.
    df.columns = df.columns.str.strip().str.lower()

    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(f"Duplicate column names: {duplicates}")

    required_columns = [ID_COLUMN, *FEATURE_COLUMNS, TARGET]
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Keep only the identifier, locked features, and target.
    data = df[required_columns].copy()

    validate_identifier(data)

    # Clean features and validate the target separately.
    data = clean_features(data)
    data[TARGET] = clean_target(data[TARGET])

    # Impute features only; never impute the target or identifier.
    data = impute_missing_values(data)

    # Final integrity checks.
    if data[OUTPUT_COLUMNS].isna().any().any():
        missing = data[OUTPUT_COLUMNS].isna().sum()
        raise ValueError(
            f"Missing values remain: {missing[missing > 0].to_dict()}"
        )

    if not data[TARGET].isin([0, 1]).all():
        raise ValueError("Target contains values other than 0 and 1.")

    if data[ID_COLUMN].duplicated().any():
        raise ValueError("Duplicate patient identifiers detected.")

    data = data[OUTPUT_COLUMNS]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETED")
    print("=" * 60)
    print(f"Rows: {len(data):,}")
    print(f"Identifier: {ID_COLUMN}")
    print(f"Model features: {len(FEATURE_COLUMNS)}")
    print(f"Total columns: {len(data.columns)}")
    print(f"Missing values remaining: {int(data.isna().sum().sum())}")
    print(f"Target distribution: {data[TARGET].value_counts().sort_index().to_dict()}")
    print(f"Positive prevalence: {data[TARGET].mean():.4f}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    preprocess()