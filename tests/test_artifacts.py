import numpy as np
from sklearn.linear_model import LogisticRegression

from mlcore import artifacts


def test_json_roundtrip(tmp_path):
    artifacts.write_json(tmp_path / "x.json", {"a": 1, "b": [1.5, 2]})

    assert artifacts.read_json(tmp_path / "x.json") == {"a": 1, "b": [1.5, 2]}


def test_save_and_load_model_keep_the_threshold_and_the_predictions(tmp_path, monkeypatch):
    monkeypatch.setattr(artifacts, "MODEL_PATH", tmp_path / "model.joblib")
    monkeypatch.setattr(artifacts, "THRESHOLD_PATH", tmp_path / "threshold.json")
    X = np.array([[0.0], [1.0], [2.0], [3.0]])
    model = LogisticRegression().fit(X, [0, 0, 1, 1])

    artifacts.save_model(model, 0.37)
    loaded, threshold = artifacts.load_model()

    assert threshold == 0.37
    assert np.allclose(loaded.predict_proba(X), model.predict_proba(X))
