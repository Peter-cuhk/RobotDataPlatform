from __future__ import annotations

import importlib

import pytest


def test_benchmark_metrics_compute_auc_and_rank_correlation() -> None:
    benchmark = importlib.import_module("scripts.benchmark_quality_scoring")

    assert benchmark.roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert benchmark.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert benchmark.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
