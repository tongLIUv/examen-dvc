from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


ROOT_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed_data"
MODELS_DIR = ROOT_DIR / "models"
PARAMS_PATH = ROOT_DIR / "params.yaml"

X_TRAIN_PATH = PROCESSED_DATA_DIR / "X_train_scaled.csv"
Y_TRAIN_PATH = PROCESSED_DATA_DIR / "y_train.csv"
BEST_PARAMS_PATH = MODELS_DIR / "best_params.pkl"


def main() -> None:
    with PARAMS_PATH.open("r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    X_train = pd.read_csv(X_TRAIN_PATH)
    y_train = pd.read_csv(Y_TRAIN_PATH).squeeze("columns")

    grid_params = params["grid_search"]
    model_params = params["model"]

    model = RandomForestRegressor(
        random_state=model_params["random_state"]
    )

    parameter_grid = {
        "n_estimators": grid_params["n_estimators"],
        "max_depth": grid_params["max_depth"],
        "min_samples_split": grid_params["min_samples_split"],
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=parameter_grid,
        cv=grid_params["cv"],
        scoring=grid_params["scoring"],
        n_jobs=grid_params["n_jobs"],
        verbose=1,
    )

    grid_search.fit(X_train, y_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(grid_search.best_params_, BEST_PARAMS_PATH)

    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)
    print(f"Saved to: {BEST_PARAMS_PATH}")


if __name__ == "__main__":
    main()