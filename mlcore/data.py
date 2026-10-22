"""Task definitions, the time-based train / val / test split and loading of split files."""
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATA_DIR, ID, RISKLENS_DIR, TARGET

# which file / columns each task uses; ids and timestamps never become features
TASKS = {
    "credit": {
        "file": "credit_applications.csv",
        "id": "application_id",
        "time": "application_ts",
        "target": "default",
        "drop": ["customer_id"],
    },
    "fraud": {
        "file": "card_transactions.csv",
        "id": "transaction_id",
        "time": "timestamp",
        "target": "is_fraud",
        "drop": ["card_id"],
    },
}


def load_task_frame(task_name):
    """Raw RiskLens table of a task, oldest row first, with the common `id` / `target` columns."""
    task = TASKS[task_name]
    df = pd.read_csv(RISKLENS_DIR / task["file"], parse_dates=[task["time"]])
    df = df.sort_values(task["time"], kind="stable").reset_index(drop=True)
    return df.drop(columns=[task["time"], *task["drop"]]).rename(
        columns={task["id"]: ID, task["target"]: TARGET}
    )


def split_by_time(df, test_size, val_size):
    """Oldest rows -> train, newest -> test, so the model is never tested on the past."""
    n = len(df)
    n_test, n_val = int(n * test_size), int(n * val_size)
    return {
        "train": df.iloc[: n - n_val - n_test],
        "val": df.iloc[n - n_val - n_test : n - n_test],
        "test": df.iloc[n - n_test :],
    }


def load_xy(name):
    """Features and target of a split file; the `id` column is kept out of the features."""
    df = pd.read_csv(DATA_DIR / name)
    return df.drop(columns=[TARGET, ID]), df[TARGET]


def stratified_subsample(X, y, size, seed):
    """A row count (int) or share (float) of the rows, with the same class balance as the full set."""
    X_sub, _, y_sub, _ = train_test_split(X, y, train_size=size, stratify=y, random_state=seed)
    return X_sub, y_sub
