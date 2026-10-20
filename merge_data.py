from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")


def main():
    train = pd.read_csv(DATA_DIR / "train.csv")
    extra = pd.read_csv(DATA_DIR / "extra.csv")

    full = pd.concat([train, extra], ignore_index=True)
    full.to_csv(DATA_DIR / "train_full.csv", index=False)
    print(f"{len(train)} + {len(extra)} = {len(full)} training rows")


if __name__ == "__main__":
    main()
