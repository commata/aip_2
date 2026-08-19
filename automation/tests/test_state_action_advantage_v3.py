from __future__ import annotations

import numpy as np

from automation.train_state_action_advantage_v3 import (
    assign_group_folds,
    encode_factorized_input,
    policy_value,
    prediction_diagnostics,
    select_threshold,
    unique_state_actions,
)


def _row(state: str, candidate: str, damage: float, family: str = "left") -> dict:
    action = "VP_AZ_POS_SMALL" if candidate != "neg" else "VP_AZ_NEG_SMALL"
    sign = 1 if candidate != "neg" else -1
    return {
        "state_hash": state,
        "candidate_id": candidate,
        "family": family,
        "action": action,
        "server_observation": [0.0] * 42,
        "action_parameters": {
            "axis_one_hot": [0.0, 1.0, 0.0],
            "sign": sign,
            "magnitude_norm": 0.5,
            "duration_norm": 1.0,
        },
        "damage_delta": damage,
    }


def test_factorized_input_is_48d_and_contains_action_sign() -> None:
    positive = encode_factorized_input(_row("a", "pos", 0.1))
    negative = encode_factorized_input(_row("a", "neg", -0.1))
    assert positive.shape == (48,)
    assert positive[45] == 1.0
    assert negative[45] == -1.0


def test_duplicate_replicates_are_averaged_without_state_leakage() -> None:
    rows = [_row("a", "pos", 0.1), _row("a", "pos", 0.3), _row("b", "pos", -0.1)]
    unique = unique_state_actions(rows)
    assert len(unique) == 2
    assert unique[0]["damage_delta"] == 0.2
    assignment = assign_group_folds(unique, folds=2)
    assert set(assignment) == {"a", "b"}


def test_policy_value_counts_selected_zero_as_intervention() -> None:
    rows = [_row("a", "pos", 0.0), _row("b", "pos", 0.02)]
    predictions = {
        "mean": np.asarray([0.01, 0.01]),
        "std": np.zeros(2),
        "positive_probability": np.ones(2),
        "regression_probability": np.zeros(2),
    }
    result = policy_value(
        rows,
        predictions,
        {"score": 0.001, "positive": 0.6, "regression": 0.1, "lambda": 1.0},
    )
    assert result["interventions"] == 2
    assert result["coverage"] == 1.0
    assert result["mean"] == 0.01


def test_threshold_selection_reports_failed_gate_honestly() -> None:
    diagnostic = {
        "threshold": {"score": 0.001},
        "policy": {
            "interventions": 10,
            "intervention_precision": 0.9,
            "mean": 0.01,
            "large_regression_ratio": 0.1,
        },
    }
    selected = select_threshold([diagnostic])
    assert not selected["offline_gate_passed"]
    assert selected["selection_status"] == "OFFLINE_POLICY_GATE_FAILED"


def test_threshold_selection_requires_two_risk_consistent_seeds() -> None:
    base = {
        "threshold": {"score": 0.001},
        "policy": {
            "interventions": 10,
            "intervention_precision": 0.9,
            "mean": 0.01,
            "large_regression_ratio": 0.0,
        },
    }
    failed = select_threshold([{**base, "consistent_seed_count": 1}])
    passed = select_threshold([{**base, "consistent_seed_count": 2}])
    assert not failed["offline_gate_passed"]
    assert passed["offline_gate_passed"]


def test_prediction_diagnostics_use_state_grouped_top_action_value() -> None:
    rows = [
        _row("a", "pos", 0.02),
        _row("a", "neg", -0.01),
        _row("b", "pos", 0.0),
        _row("b", "neg", 0.01),
    ]
    damage = np.asarray([row["damage_delta"] for row in rows])
    predictions = {
        "mean": damage.copy(),
        "std": np.zeros(4),
        "positive_probability": np.asarray([0.9, 0.1, 0.1, 0.9]),
        "regression_probability": np.asarray([0.1, 0.9, 0.1, 0.1]),
    }
    result = prediction_diagnostics(rows, damage, predictions)
    assert result["top_action_agreement"] == 1.0
    assert result["ungated_top_action_regret_mean"] == 0.0
    assert result["predicted_actual_advantage_spearman"] == 1.0
