from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed_data"
MODELS_DIR = ROOT_DIR / "models"

X_TRAIN_PATH = PROCESSED_DATA_DIR / "X_train.csv"
X_TEST_PATH = PROCESSED_DATA_DIR / "X_test.csv"

X_TRAIN_SCALED_PATH = PROCESSED_DATA_DIR / "X_train_scaled.csv"
X_TEST_SCALED_PATH = PROCESSED_DATA_DIR / "X_test_scaled.csv"

SCALER_PATH = MODELS_DIR / "scaler.pkl"


def main() -> None:
    X_train = pd.read_csv(X_TRAIN_PATH)
    X_test = pd.read_csv(X_TEST_PATH)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled_df = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns,
    )

    X_test_scaled_df = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns,
    )

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X_train_scaled_df.to_csv(X_TRAIN_SCALED_PATH, index=False)
    X_test_scaled_df.to_csv(X_TEST_SCALED_PATH, index=False)

    joblib.dump(scaler, SCALER_PATH)

    print(f"X_train_scaled: {X_train_scaled_df.shape}")
    print(f"X_test_scaled: {X_test_scaled_df.shape}")
    print(f"Scaler saved to: {SCALER_PATH}")


if __name__ == "__main__":
    main()