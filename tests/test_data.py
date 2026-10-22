import pandas as pd

from mlcore import data
from mlcore.data import load_task_frame, load_xy, split_by_time, stratified_subsample


def test_split_by_time_sizes_and_order(frame):
    parts = split_by_time(frame, test_size=0.15, val_size=0.15)

    assert len(parts["test"]) == 30
    assert len(parts["val"]) == 30
    assert len(parts["train"]) == 140
    # oldest -> train, newest -> test: ids (our stand-in for time) must increase across the splits
    assert parts["train"]["id"].max() < parts["val"]["id"].min()
    assert parts["val"]["id"].max() < parts["test"]["id"].min()


def test_split_by_time_has_no_overlap_and_loses_no_rows(frame):
    parts = split_by_time(frame, test_size=0.2, val_size=0.1)
    ids = pd.concat([p["id"] for p in parts.values()])

    assert ids.is_unique
    assert len(ids) == len(frame)


def test_load_task_frame_renames_sorts_and_drops(tmp_path, monkeypatch):
    # rows deliberately out of time order; customer_id / timestamp must not reach the model
    pd.DataFrame(
        {
            "application_id": ["a2", "a1"],
            "application_ts": ["2024-01-02", "2024-01-01"],
            "customer_id": ["c2", "c1"],
            "income": [2.0, 1.0],
            "default": [1, 0],
        }
    ).to_csv(tmp_path / "credit_applications.csv", index=False)
    monkeypatch.setattr(data, "RISKLENS_DIR", tmp_path)

    df = load_task_frame("credit")

    assert list(df.columns) == ["id", "income", "target"]
    assert df["id"].tolist() == ["a1", "a2"]  # oldest first


def test_load_xy_keeps_id_and_target_out_of_features(tmp_path, monkeypatch, frame):
    frame.to_csv(tmp_path / "train.csv", index=False)
    monkeypatch.setattr(data, "DATA_DIR", tmp_path)

    X, y = load_xy("train.csv")

    assert "id" not in X.columns and "target" not in X.columns
    assert y.name == "target"
    assert len(X) == len(y) == len(frame)


def test_stratified_subsample_keeps_class_balance(frame):
    X, y = frame.drop(columns="target"), frame["target"]

    X_sub, y_sub = stratified_subsample(X, y, size=100, seed=0)

    assert len(X_sub) == 100
    assert abs(y_sub.mean() - y.mean()) < 0.02
