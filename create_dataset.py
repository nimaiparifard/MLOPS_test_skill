from pathlib import Path

import pandas as pd
import yaml
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data")


def main():
    params = yaml.safe_load(Path("params.yaml").read_text())["dataset"]

    X, y = make_regression(
        n_samples=params["n_samples"],
        n_features=params["n_features"],
        n_informative=params["n_features"],
        noise=params["noise"],
        random_state=params["seed"],
    )

    df = pd.DataFrame(X, columns=[f"x{i + 1}" for i in range(params["n_features"])])
    df["target"] = y

    train, test = train_test_split(
        df, test_size=params["test_size"], random_state=params["seed"]
    )

    DATA_DIR.mkdir(exist_ok=True)
    train.to_csv(DATA_DIR / "train.csv", index=False)
    test.to_csv(DATA_DIR / "test.csv", index=False)
    print(f"Saved {len(train)} train rows and {len(test)} test rows to {DATA_DIR}/")


if __name__ == "__main__":
    main()
