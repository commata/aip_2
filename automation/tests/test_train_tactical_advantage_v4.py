from __future__ import annotations

import numpy as np

from automation.train_tactical_advantage_v4 import (
    assign_scenario_folds,
    encode_input,
    tactical_policy_value,
    transform_rows,
)


def _dataset_row(event: str, scenario: str, mode: str = "PURE_PURSUIT") -> dict:
    hold = 30 if mode != "BT_DEFAULT" else 0
    return {
        "event_id": event,
        "option_id": f"{mode}__d{hold}" if hold else "BT_DEFAULT",
        "mode": mode,
        "hold_frames": hold,
        "damage_advantage": 0.001,
        "observation": np.zeros(93, dtype=np.float32).tolist(),
        "opponent_id": "autopilot",
        "scenario_id": scenario,
        "fight_id": f"fight_{event}",
        "seed": 1,
    }


def test_tactical_training_input_is_98d() -> None:
    row = transform_rows([_dataset_row("e1", "g1")])[0]
    vector = encode_input(row)
    assert vector.shape == (98,)
    assert np.all(np.isfinite(vector))


def test_scenario_group_oof_never_splits_events_from_same_family() -> None:
    rows = transform_rows(
        [_dataset_row(f"e{i}", f"g{i % 5}") for i in range(15)]
    )
    assignment = assign_scenario_folds(rows, folds=5)
    for scenario in {row["scenario_id"] for row in rows}:
        folds = {
            assignment[row["state_hash"]]
            for row in rows
            if row["scenario_id"] == scenario
        }
        assert len(folds) == 1


def test_oof_policy_never_counts_bt_default_as_intervention() -> None:
    rows = [
        {
            "action": "BT_DEFAULT",
            "state_hash": "event-1",
            "candidate_id": "BT_DEFAULT",
            "damage_delta": 0.0,
        },
        {
            "action": "PURE_PURSUIT",
            "state_hash": "event-1",
            "candidate_id": "PURE_PURSUIT__d30",
            "damage_delta": 0.002,
        },
    ]
    predictions = {
        "mean": np.asarray([0.01, 0.002]),
        "std": np.asarray([0.0, 0.0]),
        "positive_probability": np.asarray([0.99, 0.99]),
        "regression_probability": np.asarray([0.0, 0.0]),
    }
    policy = tactical_policy_value(
        rows,
        predictions,
        {"score": 0.0005, "positive": 0.6, "regression": 0.02, "lambda": 1.0},
    )
    assert policy["interventions"] == 1
    assert policy["selected_rows"][0]["candidate_id"] == "PURE_PURSUIT__d30"
