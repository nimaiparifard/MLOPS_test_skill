import numpy as np
import pytest

from mlcore.metrics import SCORES, best_threshold, compute_metrics, learning_curve
from mlcore.models import build_model


def test_best_threshold_separates_a_perfectly_separable_set():
    y = np.array([0, 0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.3, 0.8, 0.9])

    t = best_threshold(y, proba)

    assert 0.3 < t <= 0.8
    assert compute_metrics(y, proba, t)["macro_f1"] == pytest.approx(1.0)


def test_best_threshold_never_splits_tied_scores():
    # every score is equal: a cut-off cannot send some of them up and some down
    y = np.array([0, 1, 0, 1])
    proba = np.full(4, 0.5)

    t = best_threshold(y, proba)
    pred = (proba >= t).astype(int)

    assert len(set(pred)) == 1


def test_compute_metrics_on_a_known_example():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.6, 0.4, 0.9])  # at 0.5 the predictions are 0, 1, 0, 1

    m = compute_metrics(y, proba, threshold=0.5)

    assert m["accuracy"] == pytest.approx(0.5)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["roc_auc"] == pytest.approx(0.75)


def test_learning_curve_has_a_train_and_val_row_per_fraction(frame):
    X, y = frame.drop(columns=["id", "target"]), frame["target"]
    half = len(X) // 2

    def make_model():
        return build_model("logistic_regression", {"max_iter": 200}, X)

    lc = learning_curve(make_model, X[:half], y[:half], X[half:], y[half:], fractions=[0.5, 1.0], seed=0)

    assert len(lc) == 4  # 2 fractions x (train, val)
    assert set(lc["split"]) == {"train", "val"}
    assert set(SCORES) <= set(lc.columns)
    assert sorted(lc["n_train"].unique()) == [half // 2, half]
