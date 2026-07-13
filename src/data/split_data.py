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
    with open(PARAMS_PATH, "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    split_params = params["split"]

    dataframe = pd.read_csv(RAW_DATA_PATH)

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found. "
            f"Available columns: {dataframe.columns.tolist()}"
        )

    X = dataframe.drop(columns=[TARGET_COLUMN])
    y = dataframe[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=split_params["test_size"],
        random_state=split_params["random_state"],
    )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(PROCESSED_DATA_DIR / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED_DATA_DIR / "X_test.csv", index=False)
    y_train.to_csv(PROCESSED_DATA_DIR / "y_train.csv", index=False)
    y_test.to_csv(PROCESSED_DATA_DIR / "y_test.csv", index=False)

    print(f"X_train: {X_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test: {y_test.shape}")


if __name__ == "__main__":
    main()