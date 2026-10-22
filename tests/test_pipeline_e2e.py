"""Runs the real pipeline scripts (python -m pipeline.<stage>) on a tiny generated dataset.

Everything happens in a temp folder because the stages use relative paths (data/, model.joblib, ...).
Training is done in the test with a plain logistic regression, so no MLflow server is needed;
the full tune + train stages are exercised by the `pipeline` job of the GitHub workflow instead.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from mlcore.artifacts import read_json, save_model
from mlcore.data import load_xy
from mlcore.metrics import best_threshold
from mlcore.models import build_model

ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.slow


def run_stage(module, workdir, *args):
    """Run `python -m <module>` inside workdir, with the project importable."""
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONUTF8": "1"}
    return subprocess.run(
        [sys.executable, "-m", module, *args], cwd=workdir, env=env, check=True, capture_output=True, text=True
    )


def test_generate_split_predict_evaluate(tmp_path, monkeypatch):
    shutil.copy(ROOT / "params.yaml", tmp_path / "params.yaml")
    monkeypatch.chdir(tmp_path)  # so mlcore's relative paths point at the temp folder

    # --- data stages (the real scripts) ---
    run_stage(
        "pipeline.generate_risklens_data", tmp_path, "--n-applications", "4000", "--n-cards", "200", "--days", "20"
    )
    run_stage("pipeline.create_dataset", tmp_path)
    splits = {name: pd.read_csv(tmp_path / "data" / f"{name}.csv") for name in ("train", "val", "test")}
    assert all(df["target"].nunique() == 2 for df in splits.values())
    assert len(splits["train"]) > len(splits["val"]) > 0

    # --- "train" stage, simplified: fit a model and save model + threshold like pipeline.train does ---
    X_train, y_train = load_xy("train.csv")
    X_val, y_val = load_xy("val.csv")
    model = build_model("logistic_regression", {"max_iter": 500}, X_train).fit(X_train, y_train)
    save_model(model, best_threshold(y_val, model.predict_proba(X_val)[:, 1]))

    # --- test + evaluate stages (the real scripts) ---
    run_stage("pipeline.test", tmp_path)
    run_stage("pipeline.evaluate", tmp_path)

    predictions = pd.read_csv(tmp_path / "data" / "predictions.csv")
    assert list(predictions.columns) == ["id", "y_true", "y_proba", "y_pred"]
    assert len(predictions) == len(splits["test"])

    metrics = read_json(tmp_path / "metrics.json")
    assert metrics["n_test"] == len(splits["test"])
    assert metrics["roc_auc"] > 0.6  # a sanity floor: the synthetic data has real signal
    for plot in ("roc_curve", "pr_curve", "confusion_matrix"):
        assert (tmp_path / "plots" / f"{plot}.png").exists()
