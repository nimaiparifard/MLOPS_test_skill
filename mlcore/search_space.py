"""The Optuna search space (which models, which ranges) and helpers that read it."""
import numpy as np

# The search space: which models are tried and the range of every hyperparameter.
# kind: "float" / "int" (low/high, optional log), "cat" (choices) or "fixed" (value).
SEARCH_SPACE = {
    "logistic_regression": {
        "C": {"kind": "float", "low": 1e-4, "high": 1e2, "log": True},
        "class_weight": {"kind": "cat", "choices": [None, "balanced"]},
        "max_iter": {"kind": "fixed", "value": 2000},
    },
    "random_forest": {
        "n_estimators": {"kind": "int", "low": 50, "high": 300},
        "max_depth": {"kind": "int", "low": 3, "high": 20},
        "min_samples_leaf": {"kind": "int", "low": 1, "high": 50, "log": True},
        "max_features": {"kind": "float", "low": 0.3, "high": 1.0},
        "class_weight": {"kind": "cat", "choices": [None, "balanced_subsample"]},
        "n_jobs": {"kind": "fixed", "value": -1},
        "random_state": {"kind": "fixed", "value": 0},
    },
    "hist_gradient_boosting": {
        "learning_rate": {"kind": "float", "low": 0.01, "high": 0.3, "log": True},
        "max_iter": {"kind": "int", "low": 50, "high": 400},
        "max_leaf_nodes": {"kind": "int", "low": 8, "high": 64},
        "l2_regularization": {"kind": "float", "low": 1e-6, "high": 10.0, "log": True},
        "class_weight": {"kind": "cat", "choices": [None, "balanced"]},
        "random_state": {"kind": "fixed", "value": 0},
    },
    "xgboost": {
        "n_estimators": {"kind": "int", "low": 50, "high": 400},
        "learning_rate": {"kind": "float", "low": 0.01, "high": 0.3, "log": True},
        "max_depth": {"kind": "int", "low": 2, "high": 8},
        "subsample": {"kind": "float", "low": 0.5, "high": 1.0},
        "colsample_bytree": {"kind": "float", "low": 0.5, "high": 1.0},
        "reg_lambda": {"kind": "float", "low": 1e-3, "high": 10.0, "log": True},
        "scale_pos_weight": {"kind": "float", "low": 1.0, "high": 20.0, "log": True},
        "tree_method": {"kind": "fixed", "value": "hist"},
        "n_jobs": {"kind": "fixed", "value": -1},
        "random_state": {"kind": "fixed", "value": 0},
    },
    "lightgbm": {
        "n_estimators": {"kind": "int", "low": 50, "high": 400},
        "learning_rate": {"kind": "float", "low": 0.01, "high": 0.3, "log": True},
        "num_leaves": {"kind": "int", "low": 8, "high": 64},
        "min_child_samples": {"kind": "int", "low": 5, "high": 100, "log": True},
        "subsample": {"kind": "float", "low": 0.5, "high": 1.0},
        "colsample_bytree": {"kind": "float", "low": 0.5, "high": 1.0},
        "reg_lambda": {"kind": "float", "low": 1e-3, "high": 10.0, "log": True},
        "scale_pos_weight": {"kind": "float", "low": 1.0, "high": 20.0, "log": True},
        "subsample_freq": {"kind": "fixed", "value": 1},
        "n_jobs": {"kind": "fixed", "value": -1},
        "random_state": {"kind": "fixed", "value": 0},
        "verbose": {"kind": "fixed", "value": -1},
    },
}


def suggest(trial):
    """Choose a model type and its hyperparameters from SEARCH_SPACE."""
    name = trial.suggest_categorical("model", list(SEARCH_SPACE))
    params = {}
    for p, spec in SEARCH_SPACE[name].items():
        key = f"{name}.{p}"  # prefix keeps each model's params apart in the plots
        if spec["kind"] == "float":
            params[p] = trial.suggest_float(key, spec["low"], spec["high"], log=spec.get("log", False))
        elif spec["kind"] == "int":
            params[p] = trial.suggest_int(key, spec["low"], spec["high"], log=spec.get("log", False))
        elif spec["kind"] == "cat":
            params[p] = trial.suggest_categorical(key, spec["choices"])
        else:
            params[p] = spec["value"]
    return name, params


def describe_range(spec):
    if spec["kind"] in ("float", "int"):
        return f"{spec['low']:g} .. {spec['high']:g}" + (" (log scale)" if spec.get("log") else "")
    if spec["kind"] == "cat":
        return ", ".join(map(str, spec["choices"]))
    return f"fixed {spec['value']}"


def sweep_values(spec, n):
    if spec["kind"] == "cat":
        return list(spec["choices"])
    grid = np.geomspace(spec["low"], spec["high"], n) if spec.get("log") else np.linspace(spec["low"], spec["high"], n)
    if spec["kind"] == "int":
        return [int(v) for v in np.unique(np.round(grid))]
    return [float(v) for v in grid]
