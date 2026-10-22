import pandas as pd

from mlcore.artifacts import load_model
from mlcore.config import DATA_DIR, ID, PREDICTIONS_PATH, TARGET


def main():
    model, threshold = load_model()
    test = pd.read_csv(DATA_DIR / "test.csv")

    proba = model.predict_proba(test.drop(columns=[TARGET, ID]))[:, 1]
    result = pd.DataFrame(
        {"id": test[ID], "y_true": test[TARGET], "y_proba": proba, "y_pred": (proba >= threshold).astype(int)}
    )
    result.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Predictions for {len(result)} rows saved to {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
