from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression

DATA_DIR = Path("data")
MODEL_PATH = Path("model.joblib")


def main():
    train = pd.read_csv(DATA_DIR / "train_full.csv")
    X, y = train.drop(columns="target"), train["target"]

    model = LinearRegression()
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Coefficients: {model.coef_}, intercept: {model.intercept_:.3f}")


if __name__ == "__main__":
    main()
