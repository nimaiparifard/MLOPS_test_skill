"""Decision threshold, classification metrics and the learning curve."""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .data import stratified_subsample

SCORES = ("accuracy", "macro_f1", "roc_auc")  # the three metrics we plot against everything


def best_threshold(y, proba):
    """Probability cut-off that maximises macro F1 (exact, over every distinct score).

    0.5 is a poor cut-off for imbalanced data (9% defaults, 0.4% fraud), so it is chosen on
    the validation set instead.
    """
    y = np.asarray(y)
    order = np.argsort(-proba, kind="stable")
    p, yy = np.asarray(proba)[order], y[order]
    tp, fp = np.cumsum(yy), np.cumsum(1 - yy)
    pos, neg = yy.sum(), len(yy) - yy.sum()
    fn, tn = pos - tp, neg - fp
    macro = (2 * tp / np.maximum(2 * tp + fp + fn, 1) + 2 * tn / np.maximum(2 * tn + fn + fp, 1)) / 2
    last_of_tie = np.append(p[1:] != p[:-1], True)  # a cut-off must not split equal scores
    macro = np.where(last_of_tie, macro, -1)
    return float(p[np.argmax(macro)])


def compute_metrics(y, proba, threshold):
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
    }


def learning_curve(make_model, X_train, y_train, X_val, y_val, fractions, seed):
    """Fit on growing, stratified slices of the training set; score train and validation."""
    rows = []
    for frac in fractions:
        if frac < 1:
            Xs, ys = stratified_subsample(X_train, y_train, frac, seed)
        else:
            Xs, ys = X_train, y_train
        model = make_model().fit(Xs, ys)
        p_val = model.predict_proba(X_val)[:, 1]
        threshold = best_threshold(y_val, p_val)
        for split, (X, y, p) in {
            "train": (Xs, ys, model.predict_proba(Xs)[:, 1]),
            "val": (X_val, y_val, p_val),
        }.items():
            m = compute_metrics(y, p, threshold)
            rows.append({"n_train": len(Xs), "split": split, **{k: m[k] for k in SCORES}})
    return pd.DataFrame(rows)
