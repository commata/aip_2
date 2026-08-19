from __future__ import annotations

import numpy as np

from automation.analyze_state_separation_v3 import (
    assign_group_folds,
    average_precision,
    grouped_cross_validation,
    roc_auc,
)


def test_group_assignment_never_splits_repeated_state():
    groups = [f"state_{index // 3}" for index in range(30)]
    labels = np.asarray([(index // 3) % 2 for index in range(30)])
    assignment = assign_group_folds(groups, labels, folds=5)
    assert len(assignment) == 10
    assert all(0 <= fold < 5 for fold in assignment.values())


def test_rank_metrics_handle_ties_and_perfect_order():
    labels = np.asarray([0, 0, 1, 1])
    assert roc_auc(labels, np.asarray([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert average_precision(labels, np.asarray([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert roc_auc(labels, np.ones(4)) == 0.5


def test_group_cv_separates_a_real_signal_without_row_leakage():
    groups = [f"state_{index // 2}" for index in range(60)]
    group_signal = np.asarray([(index // 2) % 2 for index in range(60)], dtype=float)
    x = np.column_stack((2.0 * group_signal - 1.0, np.linspace(-1.0, 1.0, 60)))
    labels = group_signal.astype(int)
    result = grouped_cross_validation(x, labels, groups, folds=5)
    assert roc_auc(labels, result["logistic_probability"]) > 0.95
    for fold in result["folds"]:
        assert fold["train_groups"] + fold["test_groups"] == 30
