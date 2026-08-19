from __future__ import annotations

import numpy as np

from automation.analyze_state_oracle_v3 import (
    analyze_oracle,
    canonical_state_hash,
    encode_action,
    reconstruct_server_v2_observation,
)


def scenario():
    return {
        "env_config": {
            "ownship": [0.0, 0.0, -4700.0, 0.0, 0.0, 0.0, 220.0],
            "target": [700.0, 10.0, -4750.0, 0.0, 0.0, 180.0, 210.0],
        }
    }


def test_server_reconstruction_excludes_both_health_values():
    legacy = np.zeros(45, dtype=np.float32)
    legacy[5] = -1.0
    legacy[13] = 1.0
    first = reconstruct_server_v2_observation(legacy, scenario())
    legacy[5] = 1.0
    legacy[13] = -1.0
    second = reconstruct_server_v2_observation(legacy, scenario())
    assert len(first) == 42
    assert first == second
    assert np.all(np.isfinite(first))
    assert first[39] == np.float32(-17.0 / 18.0)


def test_state_hash_is_canonical_and_sensitive_to_elapsed_state():
    observation = [0.0] * 42
    first, payload = canonical_state_hash(scenario(), observation, 1.0 / 30.0)
    repeated, repeated_payload = canonical_state_hash(scenario(), observation, 1.0 / 30.0)
    changed, _ = canonical_state_hash(scenario(), observation, 2.0 / 30.0)
    assert first == repeated
    assert payload == repeated_payload
    assert first != changed


def test_state_hash_ignores_cosmetic_scenario_name():
    first_scenario = scenario()
    second_scenario = scenario()
    first_scenario["name"] = "display_a"
    second_scenario["name"] = "display_b"
    first, _ = canonical_state_hash(first_scenario, [0.0] * 42, 1.0)
    second, _ = canonical_state_hash(second_scenario, [0.0] * 42, 1.0)
    assert first == second


def test_action_encoding_is_factorized_not_categorical():
    encoded = encode_action("VP_AZ_NEG_SMALL", 0.25, 18)
    assert encoded["axis_one_hot"] == [0.0, 1.0, 0.0]
    assert encoded["sign"] == -1
    assert encoded["magnitude_norm"] == 0.5
    assert encoded["duration_norm"] == 0.5


def test_oracle_can_be_feasible_when_every_static_action_is_below_sixty_percent():
    states = []
    rows = []
    families = ("left", "right")
    for index in range(6):
        state_hash = f"hash_{index}"
        states.append(
            {
                "state_id": f"state_{index}",
                "state_hash": state_hash,
                "family": families[index % 2],
            }
        )
        for candidate, positive_on_even in (("AZ", True), ("EL", False)):
            positive = (index % 2 == 0) == positive_on_even
            rows.append(
                {
                    "state_hash": state_hash,
                    "candidate_id": candidate,
                    "family": families[index % 2],
                    "damage_delta": 0.004 if positive else -0.004,
                }
            )
    # Duplicate the synthetic coverage to satisfy the production minimum-30 static audit.
    states = [
        {**state, "state_id": f"{state['state_id']}_{repeat}", "state_hash": f"{state['state_hash']}_{repeat}"}
        for repeat in range(5)
        for state in states
    ]
    rows = [
        {**row, "state_hash": f"{row['state_hash']}_{repeat}"}
        for repeat in range(5)
        for row in rows
    ]
    analysis = analyze_oracle(states, rows)
    assert analysis["oracle"]["positive_ratio"] == 1.0
    assert analysis["oracle"]["median"] > 0.0
    assert analysis["best_static_min_30_states"]["positive_ratio"] == 0.5
    assert analysis["feasibility"] == "ORACLE_FEASIBLE"
