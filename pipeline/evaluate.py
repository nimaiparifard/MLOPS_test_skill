import matplotlib.pyplot as plt
import pandas as pd

from mlcore.artifacts import load_threshold, write_json
from mlcore.config import METRICS_PATH, PLOTS_DIR, PREDICTIONS_PATH
from mlcore.metrics import compute_metrics
from mlcore.plots import confusion_figure, pr_figure, roc_figure


def save_test_plots(y_true, proba, threshold):
    PLOTS_DIR.mkdir(exist_ok=True)
    for name, fig in {
        "roc_curve": roc_figure({"test": (y_true, proba)}, "Test ROC curve"),
        "pr_curve": pr_figure({"test": (y_true, proba)}, "Test precision-recall curve"),
        "confusion_matrix": confusion_figure(y_true, proba, threshold, "Test confusion matrix"),
    }.items():
        fig.savefig(PLOTS_DIR / f"{name}.png", dpi=120)
        plt.close(fig)


def main():
    df = pd.read_csv(PREDICTIONS_PATH)
    threshold = load_threshold()
    y_true, proba = df["y_true"], df["y_proba"]

    metrics = compute_metrics(y_true, proba, threshold)
    metrics["threshold"] = threshold
    metrics["n_test"] = len(df)
    write_json(METRICS_PATH, metrics)

    save_test_plots(y_true, proba, threshold)

    for name, value in metrics.items():
        print(f"{name:<10}: {value:.4f}")


if __name__ == "__main__":
    main()
