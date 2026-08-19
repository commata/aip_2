from __future__ import annotations

import numpy as np
import pytest

from dogfight.ai.tactical_modes import (
    TACTICAL_HOLD_FRAMES,
    TACTICAL_MODES_T1,
    TacticalModeConfig,
    apply_tactical_mode,
    champion_vp_to_local_setpoint,
    generate_tactical_vp,
    tactical_action_contract,
)
from dogfight.sim.state_schema import StateIndex


def state(
    position=(0.0, 0.0, -4500.0),
    attitude=(0.0, 0.0, 0.0),
    body_velocity=(220.0, 0.0, 0.0),
) -> np.ndarray:
    value = np.zeros(51, dtype=np.float64)
    value[:3] = position
    value[3:6] = attitude
    value[6:9] = body_velocity
    value[StateIndex.KCAS] = np.linalg.norm(body_velocity)
    value[StateIndex.ALT] = -position[2]
    value[StateIndex.HEALTH] = 1.0
    return value


def test_t1_contract_and_duration_are_frozen() -> None:
    contract = tactical_action_contract()
    assert TACTICAL_MODES_T1 == (
        "BT_DEFAULT",
        "PURE_PURSUIT",
        "LEAD_PURSUIT_T060",
        "LAG_PURSUIT_D250",
    )
    assert TACTICAL_HOLD_FRAMES == (30, 60, 120)
    assert contract["default_action"] == "BT_DEFAULT"
    assert contract["throttle"] == "exact same-frame Pure BT"
    assert contract["champion_vp_coordinates"] == "north/east/altitude_m_z_up"
    assert contract["tactical_vp_coordinates"] == "north/east/down_m_ned"


def test_champion_vp_z_up_contract_converts_to_local_elevation() -> None:
    own = state(position=(0.0, 0.0, -4500.0))
    level = champion_vp_to_local_setpoint(np.array([1000.0, 0.0, 4500.0]), own)
    above = champion_vp_to_local_setpoint(np.array([1000.0, 0.0, 4600.0]), own)
    below = champion_vp_to_local_setpoint(np.array([1000.0, 0.0, 4400.0]), own)
    assert level.local_elevation_deg == pytest.approx(0.0)
    assert above.local_elevation_deg > 0.0
    assert below.local_elevation_deg < 0.0


def test_pure_lead_and_lag_semantics_match_names() -> None:
    own = state()
    target = state(
        position=(1000.0, 100.0, -4600.0),
        attitude=(0.0, 0.0, 90.0),
        body_velocity=(200.0, 0.0, 0.0),
    )
    pure, _ = generate_tactical_vp("PURE_PURSUIT", own, target)
    lead, lead_info = generate_tactical_vp("LEAD_PURSUIT_T060", own, target)
    lag, lag_info = generate_tactical_vp("LAG_PURSUIT_D250", own, target)
    assert np.allclose(pure, target[:3])
    assert lead[1] > pure[1]
    assert lead_info["parameters"]["lead_time_s"] == pytest.approx(0.60)
    assert lag[1] < pure[1]
    assert lag_info["parameters"]["lag_distance_m"] == pytest.approx(250.0)


def test_bt_default_is_exact_and_skips_tactical_controller() -> None:
    own = state()
    target = state(position=(900.0, 100.0, -4500.0))
    bt_action = np.array([0.2, -0.3, 0.1, 0.73], dtype=np.float32)
    final, info = apply_tactical_mode(
        "BT_DEFAULT", bt_action, np.array([800.0, 0.0, -4500.0]), own, target
    )
    assert np.array_equal(final, bt_action)
    assert info["fallback_reason"] == "bt_default"


@pytest.mark.parametrize("mode", TACTICAL_MODES_T1[1:])
def test_tactical_modes_preserve_bt_throttle_and_are_finite(mode: str) -> None:
    own = state()
    target = state(position=(900.0, 180.0, -4550.0), attitude=(0.0, 0.0, 15.0))
    bt_action = np.array([0.1, 0.05, -0.02, 0.61], dtype=np.float32)
    final, info = apply_tactical_mode(
        mode,
        bt_action,
        np.array([1200.0, 0.0, 4500.0]),
        own,
        target,
    )
    assert final.shape == (4,)
    assert np.all(np.isfinite(final))
    assert final[3] == bt_action[3]
    assert info["throttle_bt_only"] is True


def test_nonfinite_input_falls_back_to_exact_bt() -> None:
    own = state()
    target = state()
    target[6] = np.nan
    bt_action = np.array([0.2, -0.3, 0.1, 0.73], dtype=np.float32)
    final, info = apply_tactical_mode(
        "LEAD_PURSUIT_T060",
        bt_action,
        np.array([800.0, 0.0, 4500.0]),
        own,
        target,
    )
    assert np.array_equal(final, bt_action)
    assert info["fallback"] is True


def test_vp_altitude_floor_is_enforced() -> None:
    own = state(position=(0.0, 0.0, -1200.0))
    target = state(position=(500.0, 0.0, -500.0))
    vp, _ = generate_tactical_vp(
        "PURE_PURSUIT",
        own,
        target,
        config=TacticalModeConfig(minimum_vp_altitude_m=1000.0),
    )
    assert vp[2] <= -1000.0
