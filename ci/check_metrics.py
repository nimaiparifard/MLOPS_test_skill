"""Quality gate for CI: exit 1 if metrics.json is broken or the model is worse than the minimum.

Usage: python ci/check_metrics.py [--metrics metrics.json] [--min-roc-auc 0.75] [--min-pr-auc 0.20]

The floors are deliberately loose: CI trains a small, fast model (few Optuna trials, subsampled
rows), so this catches a broken pipeline or a real regression, not small run-to-run noise.
"""
import argparse
import json
import math
import sys
from pathlib import Path

REQUIRED = ["accuracy", "macro_f1", "roc_auc", "pr_auc", "precision", "recall", "threshold", "n_test"]


def check(metrics, min_roc_auc, min_pr_auc):
    """The list of problems found; empty means the model passes."""
    problems = [f"missing metric: {k}" for k in REQUIRED if k not in metrics]
    problems += [f"{k} is not a finite number: {v}" for k, v in metrics.items() if not math.isfinite(v)]
    if problems:
        return problems  # no point comparing numbers that are missing or NaN

    if metrics["n_test"] <= 0:
        problems.append("n_test is 0: the test split is empty")
    if not 0 < metrics["threshold"] < 1:
        problems.append(f"threshold {metrics['threshold']:.4f} is outside (0, 1)")
    if metrics["roc_auc"] < min_roc_auc:
        problems.append(f"roc_auc {metrics['roc_auc']:.4f} is below the minimum {min_roc_auc}")
    if metrics["pr_auc"] < min_pr_auc:
        problems.append(f"pr_auc {metrics['pr_auc']:.4f} is below the minimum {min_pr_auc}")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics", type=Path, default=Path("metrics.json"))
    parser.add_argument("--min-roc-auc", type=float, default=0.75)
    parser.add_argument("--min-pr-auc", type=float, default=0.20)
    args = parser.parse_args()

    metrics = json.loads(args.metrics.read_text())
    for name, value in metrics.items():
        print(f"{name:<10}: {value:.4f}")

    problems = check(metrics, args.min_roc_auc, args.min_pr_auc)
    for problem in problems:
        print(f"FAIL: {problem}")
    if problems:
        sys.exit(1)
    print("metrics gate passed")


if __name__ == "__main__":
    main()
