import numpy as np
import pytest

from mlcore.models import MODELS, build_model
from mlcore.search_space import SEARCH_SPACE

# tiny settings so every model fits in well under a second
FAST_PARAMS = {
    "logistic_regression": {"max_iter": 200},
    "random_forest": {"n_estimators": 5, "max_depth": 3},
    "hist_gradient_boosting": {"max_iter": 5},
    "xgboost": {"n_estimators": 5, "max_depth": 2},
    "lightgbm": {"n_estimators": 5, "verbose": -1},
}


def test_every_model_has_a_search_space_and_a_fast_test_config():
    # catches the classic drift: a model is added in one place but not in the others
    assert set(MODELS) == set(SEARCH_SPACE) == set(FAST_PARAMS)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_model_pipeline_handles_nans_and_categoricals(name, frame):
    X, y = frame.drop(columns=["id", "target"]), frame["target"]

    proba = build_model(name, FAST_PARAMS[name], X).fit(X, y).predict_proba(X)[:, 1]

    assert proba.shape == (len(X),)
    assert np.isfinite(proba).all()
    assert ((proba >= 0) & (proba <= 1)).all()


def test_unseen_category_at_predict_time_does_not_crash(frame):
    X, y = frame.drop(columns=["id", "target"]), frame["target"]
    model = build_model("logistic_regression", FAST_PARAMS["logistic_regression"], X).fit(X, y)

    new = X.head(3).copy()
    new["purpose"] = "never-seen-before"

    assert model.predict_proba(new).shape == (3, 2)


@pytest.mark.parametrize("name", sorted(SEARCH_SPACE))
def test_search_space_ranges_are_valid(name):
    for param, spec in SEARCH_SPACE[name].items():
        if spec["kind"] in ("float", "int"):
            assert spec["low"] < spec["high"], f"{name}.{param}"
            if spec.get("log"):
                assert spec["low"] > 0, f"{name}.{param}: a log range must be positive"
        elif spec["kind"] == "cat":
            assert spec["choices"], f"{name}.{param}"
        else:
            assert spec["kind"] == "fixed", f"{name}.{param}"
