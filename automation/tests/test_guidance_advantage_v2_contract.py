from __future__ import annotations

import numpy as np

from dogfight.ai.guidance_advantage import (
    GUIDANCE_ADVANTAGE_ACTIONS,
    GUIDANCE_SERVER_FEATURES,
    GUIDANCE_SERVER_OBSERVATION_SIZE,
    build_server_guidance_observation,
    server_observation_contract,
)
from dogfight.ai.guidance_selector import (
    GuidanceSetpoint,
)
from dogfight.sim.state_schema import StateIndex


def state(*, n=0.0, e=0.0, altitude=5000.0, yaw=0.0, speed=230.0, health=1.0):
    value = np.zeros(46, dtype=np.float64)
    value[StateIndex.N] = n
    value[StateIndex.E] = e
    value[StateIndex.D] = -altitude
    value[StateIndex.YAW] = yaw
    value[StateIndex.KCAS] = speed
    value[StateIndex.ALT] = altitude
    value[StateIndex.HEALTH] = health
    value[6] = speed
    return value


def observation(own, target):
    return build_server_guidance_observation(
        own,
        target,
        np.asarray([0.2, -0.1, 0.3, 0.7], dtype=np.float32),
        GuidanceSetpoint(1.0, -2.0, 1000.0, 230.0),
        sim_time_s=12.0,
        previous_action_id=0,
        action_hold_frames=0,
        gate_elapsed_frames=1,
        gate_active=True,
        minimum_action_hold_frames=6,
        maximum_active_frames=36,
    )


def test_server_contract_excludes_health_and_is_finite():
    own = state(health=1.0)
    target = state(n=1000.0, e=100.0, yaw=180.0, health=1.0)
    vector = observation(own, target)
    contract = server_observation_contract()
    assert GUIDANCE_SERVER_OBSERVATION_SIZE == 42
    assert len(GUIDANCE_SERVER_FEATURES) == 42
    assert vector.shape == (42,)
    assert vector.dtype == np.float32
    assert np.all(np.isfinite(vector))
    assert np.all(np.abs(vector) <= 1.0)
    assert contract["health_features"] == []
    assert all("health" not in name for name in GUIDANCE_SERVER_FEATURES)


def test_health_changes_do_not_change_server_observation():
    target = state(n=1000.0, e=100.0, yaw=180.0, health=0.2)
    first = observation(state(health=1.0), target)
    target[StateIndex.HEALTH] = 0.9
    second = observation(state(health=-3.0), target)
    assert np.array_equal(first, second)


def test_server_observation_does_not_require_health_slot():
    own = state()[: StateIndex.HEALTH]
    target = state(n=1000.0, e=100.0, yaw=180.0)[: StateIndex.HEALTH]
    vector = observation(own, target)
    assert vector.shape == (42,)
    assert np.all(np.isfinite(vector))


def test_primary_action_library_removes_semantic_mismatch_actions():
    assert GUIDANCE_ADVANTAGE_ACTIONS == (
        "BT_DEFAULT",
        "VP_AZ_POS_SMALL",
        "VP_AZ_NEG_SMALL",
        "VP_EL_POS_SMALL",
        "VP_EL_NEG_SMALL",
    )
