from pathlib import Path

import joblib
import pandas as pd

DATA_DIR = Path("data")
MODEL_PATH = Path("model.joblib")
PREDICTIONS_PATH = DATA_DIR / "predictions.csv"


def main():
    model = joblib.load(MODEL_PATH)
    test = pd.read_csv(DATA_DIR / "test.csv")
    X = test.drop(columns="target")

    result = pd.DataFrame({"y_true": test["target"], "y_pred": model.predict(X)})
    result.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Predictions for {len(result)} rows saved to {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
