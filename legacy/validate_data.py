import json
import sys
from pathlib import Path

import pandas as pd
import yaml
from pandera.errors import SchemaErrors

from legacy.data_contract import validate_schema

DATA_DIR = Path("data")
FILES = ["train_full.csv", "val_full.csv", "test.csv"]
REPORT_PATH = Path("validation_report.json")


def main():
    n_features = yaml.safe_load(Path("params.yaml").read_text())["dataset"]["n_features"]

    failed = False
    report = {}
    for name in FILES:
        df = pd.read_csv(DATA_DIR / name)
        try:
            validate_schema(df, n_features)
            report[name] = len(df)
            print(f"OK    {name} ({len(df)} rows)")
        except SchemaErrors as err:
            failed = True
            print(f"FAIL  {name}\n{err.failure_cases}")

    if failed:
        sys.exit(1)

    REPORT_PATH.write_text(json.dumps({"passed": True, "rows": report}, indent=2))


if __name__ == "__main__":
    main()
