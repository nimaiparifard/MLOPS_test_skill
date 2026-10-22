"""Matplotlib figures; every function returns a Figure (the caller saves or logs it, then closes it)."""
import matplotlib

matplotlib.use("Agg")  # no display needed; figures are only saved / logged
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve

from .metrics import SCORES

LABELS = {"accuracy": "Accuracy", "macro_f1": "Macro F1", "roc_auc": "ROC AUC", "pr_auc": "PR AUC"}
COLORS = {"accuracy": "#2a7de1", "macro_f1": "#e8710a", "roc_auc": "#1a9850", "pr_auc": "#8e44ad"}


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=11, loc="left")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def learning_curve_figure(lc: pd.DataFrame, title="Learning curve"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True)
    for ax, score in zip(axes, SCORES):
        for split, style in (("train", "--"), ("val", "-")):
            d = lc[lc["split"] == split]
            ax.plot(d["n_train"], d[score], style, marker="o", color=COLORS[score], label=split)
        _style(ax, LABELS[score], "training rows", LABELS[score])
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def roc_figure(curves: dict, title="ROC curve"):
    """curves: {label: (y_true, proba)}"""
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, (y, proba) in curves.items():
        fpr, tpr, _ = roc_curve(y, proba)
        ax.plot(fpr, tpr, label=f"{label} (AUC {roc_auc_score(y, proba):.3f})")
    ax.plot([0, 1], [0, 1], ":", color="grey", label="chance")
    _style(ax, title, "False positive rate", "True positive rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def pr_figure(curves: dict, title="Precision-recall curve"):
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, (y, proba) in curves.items():
        precision, recall, _ = precision_recall_curve(y, proba)
        ax.plot(recall, precision, label=f"{label} (AP {average_precision_score(y, proba):.3f})")
        ax.axhline(np.mean(y), ls=":", color="grey")
    _style(ax, title, "Recall", "Precision")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def confusion_figure(y, proba, threshold, title="Confusion matrix"):
    pred = (np.asarray(proba) >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    share = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n{share[i, j]:.1%}", ha="center", va="center",
                    color="white" if share[i, j] > 0.5 else "black")
    ax.set_xticks([0, 1], ["pred 0", "pred 1"])
    ax.set_yticks([0, 1], ["true 0", "true 1"])
    ax.set_title(f"{title} (threshold {threshold:.3f})", fontsize=11, loc="left")
    fig.tight_layout()
    return fig


def sweep_figure(df: pd.DataFrame, model, param, log_x=False, best_value=None):
    """Accuracy, macro F1 and ROC AUC as one hyperparameter changes (the others stay fixed)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    categorical = df["value"].map(lambda v: isinstance(v, str) or v is None or isinstance(v, bool)).any()
    x = [str(v) for v in df["value"]] if categorical else df["value"].astype(float)
    for score in SCORES:
        ax.plot(x, df[score], marker="o", color=COLORS[score], label=LABELS[score])
    if not categorical:
        if log_x:
            ax.set_xscale("log")
        if best_value is not None:
            ax.axvline(float(best_value), ls="--", color="grey", label="tuned value")
    _style(ax, f"{model}: effect of {param}", param, "score")
    ax.legend()
    fig.tight_layout()
    return fig


def comparison_figure(summary: pd.DataFrame):
    """summary: index = model name, columns = SCORES (best trial of each model)."""
    fig, ax = plt.subplots(figsize=(9, 4))
    width = 0.8 / len(SCORES)
    xs = np.arange(len(summary))
    for k, score in enumerate(SCORES):
        ax.bar(xs + k * width, summary[score], width, color=COLORS[score], label=LABELS[score])
    ax.set_xticks(xs + width, summary.index, rotation=15)
    ax.set_ylim(max(0, summary[list(SCORES)].min().min() - 0.1), 1.0)
    _style(ax, "Best trial of each model (validation)", "", "score")
    ax.legend()
    fig.tight_layout()
    return fig
