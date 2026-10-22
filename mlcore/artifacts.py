"""Reading and writing the files that stages hand to each other (model, threshold, JSON)."""
import json
from pathlib import Path

import joblib

from .config import MODEL_PATH, THRESHOLD_PATH


def write_json(path: Path, data):
    Path(path).write_text(json.dumps(data, indent=2))


def read_json(path: Path):
    return json.loads(Path(path).read_text())


def save_model(model, threshold):
    joblib.dump(model, MODEL_PATH)
    write_json(THRESHOLD_PATH, {"threshold": threshold})


def load_pipeline():
    return joblib.load(MODEL_PATH)


def load_model():
    """The fitted pipeline and its decision threshold."""
    return load_pipeline(), load_threshold()


def load_threshold():
    return read_json(THRESHOLD_PATH)["threshold"]
