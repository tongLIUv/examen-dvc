from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed_data"
MODELS_DIR = ROOT_DIR / "models"
METRICS_DIR = ROOT_DIR / "metrics"

X_TEST_PATH = PROCESSED_DATA_DIR / "X_test_scaled.csv"
Y_TEST_PATH = PROCESSED_DATA_DIR / "y_test.csv"

MODEL_PATH = MODELS_DIR / "model.pkl"
PREDICTIONS_PATH = PROCESSED_DATA_DIR / "predictions.csv"
SCORES_PATH = METRICS_DIR / "scores.json"


def main() -> None:
    X_test = pd.read_csv(X_TEST_PATH)
    y_test = pd.read_csv(Y_TEST_PATH).squeeze("columns")

    model = joblib.load(MODEL_PATH)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    predictions = pd.DataFrame(
        {
            "actual": y_test,
            "prediction": y_pred,
        }
    )

    metrics = {
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
    }

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    predictions.to_csv(PREDICTIONS_PATH, index=False)

    with SCORES_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("Evaluation completed.")
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"R²:   {r2:.6f}")
    print(f"Predictions saved to: {PREDICTIONS_PATH}")
    print(f"Metrics saved to: {SCORES_PATH}")


if __name__ == "__main__":
    main()