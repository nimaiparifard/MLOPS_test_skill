from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data")


def main():
    params = yaml.safe_load(Path("params.yaml").read_text())["dataset"]

    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "val.csv")
    extra = pd.read_csv(DATA_DIR / "extra.csv")

    # Split the new rows the same way, so they feed both training and validation.
    # (val_size is a share of the rows that are not test, as in create_dataset.py)
    extra_train, extra_val = train_test_split(
        extra,
        test_size=params["val_size"] / (1 - params["test_size"]),
        random_state=params["seed"],
    )

    train_full = pd.concat([train, extra_train], ignore_index=True)
    val_full = pd.concat([val, extra_val], ignore_index=True)
    train_full.to_csv(DATA_DIR / "train_full.csv", index=False)
    val_full.to_csv(DATA_DIR / "val_full.csv", index=False)
    print(f"train: {len(train)} + {len(extra_train)} = {len(train_full)} rows")
    print(f"val:   {len(val)} + {len(extra_val)} = {len(val_full)} rows")


if __name__ == "__main__":
    main()
