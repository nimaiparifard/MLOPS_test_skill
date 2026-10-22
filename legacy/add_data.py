from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.datasets import make_regression

DATA_DIR = Path("data")


def main():
    params = yaml.safe_load(Path("params.yaml").read_text())
    base, extra = params["dataset"], params["extra_data"]

    # Same call as create_dataset.py, so we recover the same true coefficients.
    _, _, coef = make_regression(
        n_samples=base["n_samples"],
        n_features=base["n_features"],
        n_informative=base["n_features"],
        noise=base["noise"],
        random_state=base["seed"],
        coef=True,
    )

    # New rows from the same underlying relationship, but fresh samples.
    rng = np.random.default_rng(extra["seed"])
    X = rng.normal(size=(extra["n_samples"], base["n_features"]))
    y = X @ coef + rng.normal(0, base["noise"], extra["n_samples"])

    df = pd.DataFrame(X, columns=[f"x{i + 1}" for i in range(base["n_features"])])
    df["target"] = y

    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(DATA_DIR / "extra.csv", index=False)
    print(f"Saved {len(df)} extra rows to {DATA_DIR / 'extra.csv'}")


if __name__ == "__main__":
    main()
