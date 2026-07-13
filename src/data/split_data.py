from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


ROOT_DIR = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = ROOT_DIR / "data" / "raw_data" / "raw.csv"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed_data"
PARAMS_PATH = ROOT_DIR / "params.yaml"

TARGET_COLUMN = "silica_concentrate"


def main() -> None:
    # Load configuration
    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    test_size = params["split"]["test_size"]
    random_state = params["split"]["random_state"]

    # Load raw dataset
    dataframe = pd.read_csv(RAW_DATA_PATH)

    print("Original columns:", dataframe.columns.tolist())
    print("Original shape:", dataframe.shape)

    # Verify target
    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found. "
            f"Available columns: {dataframe.columns.tolist()}"
        )

    # Detect columns that cannot be directly used by StandardScaler
    non_numeric_columns = dataframe.drop(
        columns=[TARGET_COLUMN]
    ).select_dtypes(exclude="number").columns.tolist()

    if non_numeric_columns:
        print(
            "Removing non-numeric columns from model features:",
            non_numeric_columns,
        )
        dataframe = dataframe.drop(columns=non_numeric_columns)

    # Separate features and target
    X = dataframe.drop(columns=[TARGET_COLUMN])
    y = dataframe[TARGET_COLUMN]

    # Final validation
    remaining_non_numeric = X.select_dtypes(
        exclude="number"
    ).columns.tolist()

    if remaining_non_numeric:
        raise ValueError(
            f"Non-numeric feature columns remain: {remaining_non_numeric}"
        )

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    # Create output directory
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save outputs
    X_train.to_csv(
        PROCESSED_DATA_DIR / "X_train.csv",
        index=False,
    )
    X_test.to_csv(
        PROCESSED_DATA_DIR / "X_test.csv",
        index=False,
    )
    y_train.to_csv(
        PROCESSED_DATA_DIR / "y_train.csv",
        index=False,
    )
    y_test.to_csv(
        PROCESSED_DATA_DIR / "y_test.csv",
        index=False,
    )

    print("\nSplit completed:")
    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")
    print("Feature columns:", X.columns.tolist())


if __name__ == "__main__":
    main()