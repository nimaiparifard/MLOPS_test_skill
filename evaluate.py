import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PREDICTIONS_PATH = Path("data") / "predictions.csv"
METRICS_PATH = Path("metrics.json")


def main():
    df = pd.read_csv(PREDICTIONS_PATH)
    y_true, y_pred = df["y_true"], df["y_pred"]

    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "r2": r2_score(y_true, y_pred),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    for name, value in metrics.items():
        print(f"{name.upper():5}: {value:.4f}")


if __name__ == "__main__":
    main()
