from __future__ import annotations

from automation.evaluate_prefix_replay_v4 import (
    compare_trajectory_frames,
    materialize_restart_scenario,
    restart_fidelity_status,
)


def frame(north: float = 0.0, action: float = 0.1) -> dict:
    return {
        "ownship": {
            "position_ned_m": [north, 0.0, -4500.0],
            "altitude_m": 4500.0,
            "attitude_deg": [0.0, 0.0, 0.0],
            "speed_kcas": 220.0,
        },
        "target": {
            "position_ned_m": [900.0, 0.0, -4500.0],
            "altitude_m": 4500.0,
            "attitude_deg": [0.0, 0.0, 0.0],
            "speed_kcas": 210.0,
        },
        "distance_m": 900.0 - north,
        "ata_deg": 0.0,
        "ownship_action": [action, 0.0, 0.0, 1.0],
        "target_action": [0.0, 0.0, 0.0, 0.0],
        "target_damage": 0.0,
        "ownship_damage": 0.0,
        "hybrid": {
            "bt_action": [action, 0.0, 0.0, 1.0],
            "bt_vp": [900.0, 0.0, -4500.0],
        },
    }


def test_exact_trajectory_comparison() -> None:
    left = [frame(float(index)) for index in range(4)]
    result = compare_trajectory_frames(left, left)
    assert result["exact"] is True
    assert result["frames"] == 4
    assert restart_fidelity_status(result) == "RESTART_STATE_PARITY_PASSED"


def test_restart_threshold_rejects_hidden_state_divergence() -> None:
    original = [frame(0.0), frame(1.0)]
    restart = [frame(0.0), frame(4.0, action=0.2)]
    result = compare_trajectory_frames(original, restart)
    assert result["exact"] is False
    assert result["ownship_position_error_m_max"] == 3.0
    assert restart_fidelity_status(result) == "RESTART_STATE_CAUSAL_INVALID"


def test_restart_scenario_contains_only_reconstructed_7d_state() -> None:
    payload = materialize_restart_scenario(
        {"env_config": {"initial_scenario": {"mode": "old"}}}, frame()
    )
    env = payload["env_config"]
    assert len(env["ownship"]) == 7
    assert len(env["target"]) == 7
    assert env["initial_scenario"]["legacy_use_random_scenario"] is False
