from __future__ import annotations

import numpy as np
import pytest

from automation.build_tactical_dataset_v4 import (
    grouped_split,
    initial_state_from_provider_telemetry,
    state_from_payload,
    validate_group_assignment,
)
from dogfight.ai.temporal_observation import TemporalServerObservationBuilder
from dogfight.sim.state_schema import StateIndex


def test_state_payload_requires_exact_body_velocity() -> None:
    payload = {
        "position_ned_m": [1.0, 2.0, -3000.0],
        "attitude_deg": [3.0, 4.0, 5.0],
        "body_velocity_m_s": [200.0, 2.0, -1.0],
        "speed_kcas": 200.1,
        "altitude_m": 3000.0,
    }
    state = state_from_payload(payload)
    np.testing.assert_array_equal(state[:9], [1, 2, -3000, 3, 4, 5, 200, 2, -1])
    assert state[StateIndex.KCAS] == 200.1
    assert state[StateIndex.ALT] == 3000.0
    with pytest.raises(ValueError, match="exact packet-visible body velocity"):
        state_from_payload({key: value for key, value in payload.items() if key != "body_velocity_m_s"})


def _record(event: str, fight: str, scenario: str, seed: int) -> dict:
    return {
        "event_id": event,
        "fight_id": fight,
        "trajectory_id": fight,
        "scenario_id": scenario,
        "opponent_id": "autopilot",
        "seed": seed,
    }


def test_grouped_split_keeps_all_correlated_rows_together() -> None:
    records = [
        _record("e1", "f1", "g1", 1),
        _record("e1", "f1", "g1", 1),
        _record("e2", "f2", "g2", 2),
    ]
    assignment = grouped_split(records)
    assert assignment["e1"] in {"train", "validation", "test"}
    validate_group_assignment(records, assignment)


def test_grouped_split_keeps_same_scenario_across_fights_together() -> None:
    records = [_record("e1", "f1", "same", 1), _record("e2", "f2", "same", 2)]
    assignment = grouped_split(records)
    assert assignment["e1"] == assignment["e2"]


def test_leakage_validator_rejects_same_fight_in_two_splits() -> None:
    records = [_record("e1", "same", "g1", 1), _record("e2", "same", "g2", 2)]
    with pytest.raises(ValueError, match="group leakage"):
        validate_group_assignment(records, {"e1": "train", "e2": "test"})


def test_temporal_builder_padding_contract_supports_early_events() -> None:
    builder = TemporalServerObservationBuilder()
    current = np.zeros(42, dtype=np.float32)
    first = builder.append_observation(current)
    np.testing.assert_array_equal(first[42:], np.zeros(51, dtype=np.float32))


def test_initial_state_loader_requires_exact_prefix_observable() -> None:
    initial = initial_state_from_provider_telemetry(
        {
            "initial_snapshot": {
                "ownship_server_observable": [0, 0, -5000, 0, 0, 0, 240, 0, 0, 240, 5000],
                "target_server_observable": [1000, 0, -5000, 0, 0, 0, 220, 0, 0, 220, 5000],
            }
        }
    )
    assert initial is not None
    assert initial[0][6] == 240
