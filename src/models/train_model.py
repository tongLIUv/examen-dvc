from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor


ROOT_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed_data"
MODELS_DIR = ROOT_DIR / "models"
PARAMS_PATH = ROOT_DIR / "params.yaml"

X_TRAIN_PATH = PROCESSED_DATA_DIR / "X_train_scaled.csv"
Y_TRAIN_PATH = PROCESSED_DATA_DIR / "y_train.csv"

BEST_PARAMS_PATH = MODELS_DIR / "best_params.pkl"
MODEL_PATH = MODELS_DIR / "model.pkl"


def main() -> None:
    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze("columns")

    best_params = joblib.load(BEST_PARAMS_PATH)

    model = RandomForestRegressor(
        **best_params,
        random_state=params["model"]["random_state"],
    )

    model.fit(X_train, y_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("Model training completed.")
    print("Parameters used:", best_params)
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()