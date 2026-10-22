"""Paths, constants and access to params.yaml; the single place that knows where files live."""
import os
from pathlib import Path

import yaml

DATA_DIR = Path("data")
RISKLENS_DIR = DATA_DIR / "risklens"  # written by generate_risklens_data.py
PLOTS_DIR = Path("plots")  # test-set curves written by evaluate.py

PARAMS_PATH = Path("params.yaml")
BEST_PARAMS_PATH = Path("best_params.json")  # written by tuning_experiment.py
MODEL_PATH = Path("model.joblib")
THRESHOLD_PATH = Path("threshold.json")  # decision cut-off picked on validation
METRICS_PATH = Path("metrics.json")
PREDICTIONS_PATH = DATA_DIR / "predictions.csv"
REGISTRY_PATH = Path("registry.json")

TARGET = "target"
ID = "id"

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")


def load_params(section=None):
    """The whole params.yaml, or just one section of it (e.g. "dataset")."""
    params = yaml.safe_load(PARAMS_PATH.read_text())
    return params if section is None else params[section]
