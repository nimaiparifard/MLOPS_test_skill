"""The quality gate itself needs tests: a gate that never fails (or always fails) is worse than none."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("check_metrics", Path(__file__).resolve().parent.parent / "ci" / "check_metrics.py")
check_metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_metrics)

GOOD = {
    "accuracy": 0.91,
    "macro_f1": 0.72,
    "roc_auc": 0.88,
    "pr_auc": 0.51,
    "precision": 0.49,
    "recall": 0.49,
    "threshold": 0.29,
    "n_test": 7500,
}


def test_good_metrics_pass():
    assert check_metrics.check(GOOD, min_roc_auc=0.75, min_pr_auc=0.2) == []


def test_low_roc_auc_fails():
    problems = check_metrics.check({**GOOD, "roc_auc": 0.6}, min_roc_auc=0.75, min_pr_auc=0.2)

    assert len(problems) == 1 and "roc_auc" in problems[0]


def test_missing_metric_fails():
    broken = {k: v for k, v in GOOD.items() if k != "pr_auc"}

    assert check_metrics.check(broken, min_roc_auc=0.75, min_pr_auc=0.2) == ["missing metric: pr_auc"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_metric_fails(bad):
    problems = check_metrics.check({**GOOD, "recall": bad}, min_roc_auc=0.75, min_pr_auc=0.2)

    assert any("recall" in p for p in problems)


def test_empty_test_split_fails():
    problems = check_metrics.check({**GOOD, "n_test": 0}, min_roc_auc=0.75, min_pr_auc=0.2)

    assert any("n_test" in p for p in problems)
