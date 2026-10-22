from mlcore.config import DATA_DIR, load_params
from mlcore.data import load_task_frame, split_by_time


def main():
    params = load_params("dataset")
    df = load_task_frame(params["task"])
    parts = split_by_time(df, params["test_size"], params["val_size"])

    DATA_DIR.mkdir(exist_ok=True)
    for name, part in parts.items():
        if part["target"].nunique() < 2:
            raise ValueError(f"{name} split has a single class; use more data or larger splits")
        part.to_csv(DATA_DIR / f"{name}.csv", index=False)
        print(f"{name:<5}{len(part):>8} rows, positive rate {part['target'].mean():.2%}")
    print(f"task: {params['task']}")


if __name__ == "__main__":
    main()
