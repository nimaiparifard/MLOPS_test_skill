"""Reusable building blocks of the classification pipeline.

config         paths, constants and params.yaml access
data           task definitions, time-based splits, loading features / target
metrics        decision threshold, metrics, learning curve
plots          matplotlib figures
models         preprocessing + classifier pipelines
search_space   Optuna search space and hyperparameter sweeps
artifacts      saving / loading the model, threshold and JSON files
tracking       MLflow setup and logging helpers
"""
